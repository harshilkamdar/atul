import math
from dataclasses import dataclass, field

import torch

from ..actions import Action
from ..state import GameState
from .features import (
    ACTION_DIM,
    COLOR_ORDER,
    COLOR_TO_IDX,
    components_from_action_index,
    encode_state,
    legal_action_indices,
)
from .model import AzulNet


@dataclass
class Node:
    prior: float
    parent: "Node | None"
    action_index: int | None
    children: dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class MCTS:
    def __init__(
        self,
        net: AzulNet,
        cpuct: float = 0.5,
        num_simulations: int = 200,
        temperature: float = 1.0,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
        device: torch.device | None = None,
        use_state_cache: bool = True,
    ):
        self.net = net
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or torch.device("cpu")
        self.use_state_cache = use_state_cache
        self._pv_cache: dict[tuple, tuple[torch.Tensor, float]] = {}

    def run(self, state: GameState) -> list[float]:
        root = self._expand_root(state)
        for _ in range(self.num_simulations):
            node, sim_state = self._select(root, state)
            value = self._evaluate(node, sim_state)
            self._backpropagate(node, value)
        return self._pi(root)

    def _expand_root(self, state: GameState) -> Node:
        legal = legal_action_indices(state)
        priors, value = self._policy_value(state, legal)
        priors = self._maybe_dirichlet(priors)
        root = Node(prior=1.0, parent=None, action_index=None)
        for idx, p in zip(legal, priors):
            root.children[idx] = Node(prior=p, parent=root, action_index=idx)
        root.value_sum = value
        return root

    def _policy_value(self, state: GameState, legal: list[int]) -> tuple[torch.Tensor, float]:
        if self.use_state_cache:
            key = self._state_key(state)
            cached = self._pv_cache.get(key)
            if cached is not None:
                priors, val = cached
                return priors, val
        with torch.no_grad():
            features = encode_state(state).to(self.device)
            out = self.net(features)
            logits = out["policy_logits"]
            win_value = out["margin"].item()
            mask = torch.full((ACTION_DIM,), float("-inf"), device=self.device)
            mask[legal] = 0.0
            logits = logits + mask
            probs = torch.softmax(logits, dim=0)
            priors = probs[legal]
        if self.use_state_cache:
            self._pv_cache[key] = (priors, win_value)
        return priors.cpu(), win_value

    def _maybe_dirichlet(self, priors: torch.Tensor) -> torch.Tensor:
        if self.dirichlet_alpha is None:
            return priors
        noise = torch.distributions.Dirichlet(torch.full_like(priors, self.dirichlet_alpha)).sample()
        return (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

    def _select(self, root: Node, root_state: GameState) -> tuple[Node, GameState]:
        node = root
        state = root_state.clone()
        while node.children:
            total_visits = sum(child.visits for child in node.children.values())
            best_idx, best_child = None, None
            best_score = float("-inf")
            for idx, child in node.children.items():
                u = self.cpuct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visits)
                score = child.q() + u
                if score > best_score:
                    best_score = score
                    best_idx, best_child = idx, child
            assert best_child is not None
            node = best_child
            src, color, dest = components_from_action_index(best_idx, len(state.supply.factories))
            action = Action(
                source_index=src,
                color=color,
                pattern_line=dest if dest < 5 else Action.FLOOR,
                take_first_player_token=src == Action.CENTER and state.first_player_token_in_center,
            )
            state = state.clone().apply_action(action)
            if node.visits == 0:
                break
        return node, state

    def _evaluate(self, node: Node, state: GameState) -> float:
        if state.is_terminal():
            scores = [p.score for p in state.players]
            if len(scores) >= 2:
                margin = scores[state.current_player] - scores[1 - state.current_player]
            else:
                margin = scores[0]
            return margin
        legal = legal_action_indices(state)
        priors, value = self._policy_value(state, legal)
        for idx, p in zip(legal, priors):
            node.children[idx] = Node(prior=p, parent=node, action_index=idx)
        node.value_sum += value
        return value

    def _backpropagate(self, node: Node, value: float) -> None:
        cur = node
        while cur.parent is not None:
            cur.visits += 1
            cur.value_sum += value
            value = -value
            cur = cur.parent
        cur.visits += 1
        cur.value_sum += value

    def _pi(self, root: Node) -> list[float]:
        counts = torch.zeros(ACTION_DIM, dtype=torch.float32)
        for idx, child in root.children.items():
            counts[idx] = child.visits
        if self.temperature == 0:
            best = torch.argmax(counts).item()
            pi = torch.zeros_like(counts)
            pi[best] = 1.0
        else:
            counts = counts ** (1 / self.temperature)
            pi = counts / counts.sum()
        return pi.tolist()

    def _state_key(self, state: GameState) -> tuple:
        """Hashable key for lightweight transpositions."""
        supply_key = (
            tuple(tuple(t.value for t in f) for f in state.supply.factories),
            tuple(t.value for t in state.supply.center),
            tuple(self._color_counts(state.supply.bag)),
            tuple(self._color_counts(state.supply.discard)),
            state.first_player_token_in_center,
            state.first_player_index,
        )
        players = []
        for p in state.players:
            pattern = tuple(tuple(t.value for t in line) for line in p.pattern_lines)
            wall = tuple(tuple(row) for row in p.wall)
            floor = tuple(t.value for t in p.floor_line)
            players.append((pattern, wall, floor, p.has_first_player_token, p.score))
        return (
            state.current_player,
            state.phase.value,
            state.round_number,
            supply_key,
            tuple(players),
        )

    @staticmethod
    def _color_counts(tiles) -> list[int]:
        counts = [0] * len(COLOR_ORDER)
        for t in tiles:
            counts[COLOR_TO_IDX[t]] += 1
        return counts
