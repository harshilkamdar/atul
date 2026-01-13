import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..actions import Action
from ..state import GameEngine, GameState
from .features import ACTION_DIM, components_from_action_index, encode_state, legal_action_indices
from .mcts import MCTS
from .model import AzulNet


@dataclass
class SelfPlayConfig:
    num_simulations: int = 200
    cpuct: float = 0.5
    temperature: float = 1.0
    temperature_end: float = 0.1
    temperature_moves: int = 8
    dirichlet_alpha: float | None = 0.3
    dirichlet_epsilon: float = 0.25


@dataclass
class LossWeights:
    win: float = 1.0
    margin: float = 1.0
    score: float = 0.1


@dataclass
class TrainingSample:
    features: torch.Tensor
    policy: torch.Tensor
    z_win: float
    z_margin: float
    final_self_score: float
    final_opp_score: float


class ReplayBuffer:
    """Simple replay buffer for self-play trajectories."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    def add_samples(self, samples: list[TrainingSample]) -> None:
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def iter_batches(self, batch_size: int):
        if not self.buffer:
            return
        idxs = list(range(len(self.buffer)))
        random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            yield [self.buffer[j] for j in idxs[i : i + batch_size]]


def _action_from_index(state: GameState, idx: int) -> Action:
    src, color, dest = components_from_action_index(idx, len(state.supply.factories))
    take_token = src == Action.CENTER and state.first_player_token_in_center
    return Action(
        source_index=src,
        color=color,
        pattern_line=dest if dest < 5 else Action.FLOOR,
        take_first_player_token=take_token,
    )


def _select_action(pi: torch.Tensor, temperature: float) -> int:
    if temperature <= 1e-6:
        return int(torch.argmax(pi).item())
    return int(torch.multinomial(pi, 1).item())


def self_play_game(
    net: AzulNet,
    cfg: SelfPlayConfig,
    *,
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[TrainingSample]:
    """Generate one self-play game using network-guided MCTS."""
    engine = GameEngine(seed=seed)
    state = engine.reset()
    device = device or torch.device("cpu")
    move_history = []
    move_idx = 0

    while not state.is_terminal():
        temp = cfg.temperature if move_idx < cfg.temperature_moves else cfg.temperature_end
        search = MCTS(
            net=net,
            cpuct=cfg.cpuct,
            num_simulations=cfg.num_simulations,
            temperature=temp,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
            device=device,
        )
        pi = torch.tensor(search.run(state.clone()), dtype=torch.float32)
        if pi.sum() <= 0:
            legal = legal_action_indices(state)
            pi = torch.zeros(ACTION_DIM, dtype=torch.float32)
            if legal:
                pi[legal] = 1 / len(legal)
        features = encode_state(state)
        move_history.append((features, pi, state.current_player))

        action_idx = _select_action(pi, temp)
        action = _action_from_index(state, action_idx)
        state = state.apply_action(action)
        move_idx += 1

    final_scores = [p.score for p in state.players]
    samples = []
    for features, pi, player_idx in move_history:
        opp = 1 - player_idx
        margin = float(final_scores[player_idx] - final_scores[opp])
        z_win = 1.0 if margin > 0 else -1.0 if margin < 0 else 0.0
        samples.append(
            TrainingSample(
                features=features,
                policy=pi,
                z_win=z_win,
                z_margin=margin,
                final_self_score=float(final_scores[player_idx]),
                final_opp_score=float(final_scores[opp]),
            )
        )
    return samples


def generate_self_play(
    net: AzulNet,
    cfg: SelfPlayConfig,
    num_games: int,
    *,
    device: torch.device | None = None,
    seed: int | None = None,
) -> list[TrainingSample]:
    rng = random.Random(seed)
    samples = []
    for _ in range(num_games):
        game_seed = rng.randrange(1_000_000_000)
        samples.extend(self_play_game(net, cfg, device=device, seed=game_seed))
    return samples


def _batch_to_tensors(samples: list[TrainingSample], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([s.features for s in samples]).to(device),
        "policy": torch.stack([s.policy for s in samples]).to(device),
        "z_win": torch.tensor([s.z_win for s in samples], dtype=torch.float32, device=device),
        "z_margin": torch.tensor([s.z_margin for s in samples], dtype=torch.float32, device=device),
        "scores": torch.tensor(
            [[s.final_self_score, s.final_opp_score] for s in samples],
            dtype=torch.float32,
            device=device,
        ),
    }


def compute_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], weights: LossWeights) -> tuple[torch.Tensor, dict]:
    log_probs = F.log_softmax(outputs["policy_logits"], dim=1)
    policy_loss = -(targets["policy"] * log_probs).sum(dim=1).mean()
    win_loss = F.mse_loss(outputs["win_value"], targets["z_win"])
    margin_loss = F.mse_loss(outputs["margin"], targets["z_margin"])
    score_loss = F.mse_loss(outputs["scores"], targets["scores"])
    total = policy_loss + weights.win * win_loss + weights.margin * margin_loss + weights.score * score_loss
    metrics = {
        "loss": float(total.item()),
        "policy_loss": float(policy_loss.item()),
        "win_loss": float(win_loss.item()),
        "margin_loss": float(margin_loss.item()),
        "score_loss": float(score_loss.item()),
    }
    return total, metrics


def train_epoch(
    net: AzulNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
    weights: LossWeights | None = None,
) -> list[dict]:
    """Run one epoch over the replay buffer and return per-batch metrics."""
    device = device or torch.device("cpu")
    weights = weights or LossWeights()
    net.train()
    metrics = []
    for batch in buffer.iter_batches(batch_size):
        if not batch:
            continue
        targets = _batch_to_tensors(batch, device)
        outputs = net(targets["features"])
        loss, batch_metrics = compute_loss(outputs, targets, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.append(batch_metrics)
    return metrics
