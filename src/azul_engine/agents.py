"""Baseline Azul agents and search helpers."""

import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterable

from .actions import Action
from .agents_evaluation import _leaf_evaluation
from .enums import TileColor
from .player import PATTERN_LINE_SIZES
from .state import FLOOR_PENALTIES, WALL_COLOR_TO_COL, GameState


class Agent:
    def select_action(self, state: GameState) -> Action:
        raise NotImplementedError


COLOR_RANK = {c: i for i, c in enumerate(TileColor)}


def _sorted_actions(actions: Iterable[Action]) -> list[Action]:
    def line_order(a: Action) -> int:
        return 99 if a.pattern_line == Action.FLOOR else a.pattern_line

    return sorted(
        actions,
        key=lambda a: (a.source_index, COLOR_RANK[a.color], line_order(a), a.take_first_player_token),
    )


def _tiles_taken(state: GameState, action: Action) -> int:
    source = state.supply.center if action.source_index == Action.CENTER else state.supply.factories[action.source_index]
    return sum(1 for t in source if t == action.color)


Policy = Callable[[GameState], Action]


def _floor_penalty_estimate(count: int) -> int:
    if count <= 0:
        return 0
    return sum(FLOOR_PENALTIES[:count])


def _adjacency_potential(wall: list[list[bool]], row: int, col: int) -> int:
    horiz = 1
    c = col - 1
    while c >= 0 and wall[row][c]:
        horiz += 1
        c -= 1
    c = col + 1
    while c < 5 and wall[row][c]:
        horiz += 1
        c += 1
    vert = 1
    r = row - 1
    while r >= 0 and wall[r][col]:
        vert += 1
        r -= 1
    r = row + 1
    while r < 5 and wall[r][col]:
        vert += 1
        r += 1
    if horiz == 1 and vert == 1:
        return 1
    return horiz + vert


def _endgame_progress(player_wall: list[list[bool]], row: int, col: int, color: TileColor) -> float:
    bonuses = 0.0
    if all(player_wall[row][c] or c == col for c in range(5)):
        bonuses += 1.5
    if all(player_wall[r][col] or r == row for r in range(5)):
        bonuses += 2.0
    color_hits = 1 + sum(1 for r in range(5) if player_wall[r][WALL_COLOR_TO_COL[r][color]])
    if color_hits >= 4:
        bonuses += 2.5
    return bonuses


def _opp_deny_score(state: GameState, action: Action) -> float:
    tiles_taken = _tiles_taken(state, action)
    if tiles_taken == 0 or len(state.players) <= 1:
        return 0.0
    target_color = action.color
    score = 0.0
    for idx, opp in enumerate(state.players):
        if idx == state.current_player:
            continue
        for line_idx, capacity in enumerate(PATTERN_LINE_SIZES):
            line = opp.pattern_lines[line_idx]
            if len(line) == capacity or not line:
                continue
            if line and line[0] != target_color:
                continue
            if opp.wall[line_idx][WALL_COLOR_TO_COL[line_idx][target_color]]:
                continue
            remaining = capacity - len(line)
            if remaining <= 2:
                score += remaining * 0.3
    return score * tiles_taken


def score_action(state: GameState, action: Action) -> float:
    player = state.players[state.current_player]
    tiles_taken = _tiles_taken(state, action)
    token_penalty = 1 if action.take_first_player_token else 0
    if action.pattern_line == Action.FLOOR:
        spill = tiles_taken + token_penalty
        floor_pen = _floor_penalty_estimate(spill)
        return -5.0 * spill + floor_pen

    capacity = PATTERN_LINE_SIZES[action.pattern_line]
    current_len = len(player.pattern_lines[action.pattern_line])
    spill = max(current_len + tiles_taken - capacity, 0)
    floor_pen = _floor_penalty_estimate(spill + token_penalty)

    score = -3.0 * spill + floor_pen
    score += (5 - capacity) * 0.2
    if current_len > 0:
        score += 0.6

    if current_len + tiles_taken >= capacity:
        col = WALL_COLOR_TO_COL[action.pattern_line][action.color]
        adj = _adjacency_potential(player.wall, action.pattern_line, col)
        endgame = _endgame_progress(player.wall, action.pattern_line, col, action.color)
        score += adj + endgame + 2.0

    score += _opp_deny_score(state, action)

    if action.take_first_player_token:
        score -= 0.5
    return score


def prune_and_order_actions(state: GameState, actions: list[Action], top_k: int | None = None) -> list[Action]:
    if not actions:
        return []
    scored = []
    for a in actions:
        tiles = _tiles_taken(state, a)
        if a.pattern_line == Action.FLOOR:
            spill = tiles
        else:
            capacity = PATTERN_LINE_SIZES[a.pattern_line]
            current_len = len(state.players[state.current_player].pattern_lines[a.pattern_line])
            spill = max(current_len + tiles - capacity, 0)
        scored.append((a, spill, score_action(state, a)))

    min_spill = min(spill for _, spill, _ in scored)
    keep_keys = {
        (a.source_index, a.color, a.pattern_line, a.take_first_player_token)
        for a, spill, _ in scored
        if spill <= min_spill + 2
    }
    if top_k is not None:
        top_global = sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]
        for a, _, _ in top_global:
            keep_keys.add((a.source_index, a.color, a.pattern_line, a.take_first_player_token))
    # Deduplicate using action signature, order by score.
    seen = set()
    scored_actions = sorted(scored, key=lambda x: x[2], reverse=True)
    ordered: list[Action] = []
    for a, _, _ in scored_actions:
        key = (a.source_index, a.color, a.pattern_line, a.take_first_player_token)
        if key in seen:
            continue
        if key not in keep_keys:
            continue
        seen.add(key)
        ordered.append(a)
        if top_k is not None and len(ordered) >= top_k:
            break
    return ordered


def _rollout_value(
    state: GameState,
    perspective: int,
    depth_limit: int | None,
    policy: Policy | None,
    rng: random.Random,
    use_leaf: bool,
) -> float:
    depth = 0
    while not state.is_terminal() and (depth_limit is None or depth < depth_limit):
        if policy is None:
            actions = state.legal_actions()
            if not actions:
                break
            action = rng.choice(actions)
        else:
            action = policy(state)
        state.apply_action(action)
        depth += 1
    if not state.is_terminal() and use_leaf:
        return _leaf_evaluation(state, perspective)
    scores = [p.score for p in state.players]
    opp_best = max(s for i, s in enumerate(scores) if i != perspective) if len(scores) > 1 else 0
    return scores[perspective] - opp_best


def _resolve_policy(name: str | None) -> Policy | None:
    if name is None or name == "random":
        return None
    if name == "greedy":
        return GreedyFillAgent().select_action
    if name == "heuristic":
        def policy(state: GameState) -> Action:
            actions = prune_and_order_actions(state, state.legal_actions(), top_k=8)
            if not actions:
                raise RuntimeError("no legal actions available")
            if len(actions) > 1 and random.random() < 0.1:
                return random.choice(actions[: min(3, len(actions))])
            return actions[0]
        return policy
    raise ValueError(f"unsupported policy name: {name}")


@dataclass
class RandomAgent:
    """Chooses uniformly among legal actions."""

    rng: random.Random = field(default_factory=random.Random)

    def select_action(self, state: GameState) -> Action:
        actions = state.legal_actions()
        if not actions:
            raise RuntimeError("no legal actions available")
        return self.rng.choice(actions)


class FirstLegalAgent:
    """Picks the first action under a stable ordering."""

    def select_action(self, state: GameState) -> Action:
        actions = _sorted_actions(state.legal_actions())
        if not actions:
            raise RuntimeError("no legal actions available")
        return actions[0]


class GreedyFillAgent:
    """
    Prefers completing pattern lines, then maximizing tile intake.

    Tie-breakers fall back to a stable action order.
    """

    def select_action(self, state: GameState) -> Action:
        actions = prune_and_order_actions(state, state.legal_actions())
        if not actions:
            raise RuntimeError("no legal actions available")
        return actions[0]


@dataclass
class RolloutAgent:
    """
    Simple Monte Carlo rollout agent.

    For each legal action, runs a set of random playouts (or until depth_limit)
    and picks the action with the highest average margin for the current player.
    """

    rollout_count: int = 32
    depth_limit: int | None = None
    rng: random.Random = field(default_factory=random.Random)
    rollout_policy: Policy | None = None
    prune_top_k: int | None = 12
    leaf_value: bool = False

    def select_action(self, state: GameState) -> Action:
        actions = prune_and_order_actions(state, state.legal_actions(), top_k=self.prune_top_k)
        if not actions:
            raise RuntimeError("no legal actions available")
        perspective = state.current_player

        def eval_action(action: Action) -> float:
            total = 0.0
            for _ in range(self.rollout_count):
                sim_state = state.clone()
                sim_state.rng = random.Random(self.rng.random())
                sim_state.apply_action(action)
                total += self._playout(sim_state, perspective)
            return total / self.rollout_count

        scored = [(eval_action(a), a) for a in actions]
        best_value = max(score for score, _ in scored)
        best_actions = [a for score, a in scored if score == best_value]
        return _sorted_actions(best_actions)[0]

    def _playout(self, state: GameState, perspective: int) -> float:
        return _rollout_value(
            state,
            perspective,
            self.depth_limit,
            self.rollout_policy,
            self.rng,
            self.leaf_value,
        )


@dataclass
class TimedRolloutAgent:
    """
    Rollout agent with a wall-clock budget.

    Ensures at least one rollout per action, then continues sampling while time
    remains. Useful for keeping per-move latency bounded (e.g., <0.5s).
    """

    time_budget_s: float = 0.25
    depth_limit: int | None = None
    rng: random.Random = field(default_factory=random.Random)
    rollout_policy: Policy | None = None
    prune_top_k: int | None = 12
    leaf_value: bool = True

    def select_action(self, state: GameState) -> Action:
        actions = prune_and_order_actions(state, state.legal_actions(), top_k=self.prune_top_k)
        if not actions:
            raise RuntimeError("no legal actions available")
        perspective = state.current_player

        totals = [0.0] * len(actions)
        counts = [0] * len(actions)

        start = perf_counter()

        def playout(idx: int) -> None:
            sim_state = state.clone()
            sim_state.rng = random.Random(self.rng.random())
            sim_state.apply_action(actions[idx])
            totals[idx] += self._playout(sim_state, perspective)
            counts[idx] += 1

        # One rollout per action to start.
        for i in range(len(actions)):
            playout(i)
        # Continue until time budget is exceeded.
        i = 0
        while perf_counter() - start < self.time_budget_s:
            playout(i)
            i = (i + 1) % len(actions)

        averages = [totals[i] / counts[i] for i in range(len(actions))]
        best_value = max(averages)
        best_actions = [actions[i] for i, val in enumerate(averages) if val == best_value]
        return _sorted_actions(best_actions)[0]

    def _playout(self, state: GameState, perspective: int) -> float:
        return _rollout_value(
            state,
            perspective,
            self.depth_limit,
            self.rollout_policy,
            self.rng,
            self.leaf_value,
        )


def _parallel_rollout_worker(args) -> float:
    state, action, rollout_count, depth_limit, policy_name, seed, perspective = args
    rng = random.Random(seed)
    policy = _resolve_policy(policy_name)
    total = 0.0
    for _ in range(rollout_count):
        sim_state = state.clone()
        sim_state.rng = random.Random(rng.random())
        sim_state.apply_action(action)
        total += _rollout_value(sim_state, perspective, depth_limit, policy, rng, True)
    return total


@dataclass
class ParallelRolloutAgent:
    """
    Parallel rollout agent using a process pool to evaluate actions.
    """

    rollouts_per_action: int = 8
    depth_limit: int | None = 60
    workers: int = 4
    rng: random.Random = field(default_factory=random.Random)
    rollout_policy_name: str | None = "heuristic"

    def select_action(self, state: GameState) -> Action:
        actions = state.legal_actions()
        if not actions:
            raise RuntimeError("no legal actions available")
        perspective = state.current_player
        seeds = [self.rng.random() for _ in actions]
        args = [
            (state.clone(), action, self.rollouts_per_action, self.depth_limit, self.rollout_policy_name, seed, perspective)
            for action, seed in zip(actions, seeds)
        ]
        try:
            with ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn")) as executor:
                totals = list(executor.map(_parallel_rollout_worker, args))
        except (PermissionError, NotImplementedError, OSError):
            # Fallback in restricted environments.
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                totals = list(executor.map(_parallel_rollout_worker, args))
        averages = [totals[i] / self.rollouts_per_action for i in range(len(actions))]
        best_value = max(averages)
        best_actions = [a for a, v in zip(actions, averages) if v == best_value]
        return _sorted_actions(best_actions)[0]


@dataclass
class MCTSAgent:
    """
    Upper Confidence Trees agent with a wall-clock budget.

    Uses random rollouts for value estimation; rewards are score margin from the
    root player's perspective.
    """

    time_budget_s: float = 0.5
    c_param: float = 1.3
    rollout_depth: int | None = 80
    rng: random.Random = field(default_factory=random.Random)
    rollout_policy: Policy | None = field(default_factory=lambda: _resolve_policy("heuristic"))
    max_simulations: int | None = None
    prune_top_k: int | None = 12
    expand_threshold: int = 1
    track_stats: bool = False
    stats_log: list[dict[str, int]] = field(default_factory=list)
    root: "MCTSAgent.Node | None" = field(default=None, init=False, repr=False)

    class Node:
        __slots__ = ("parent", "action", "children", "untried", "visits", "value")

        def __init__(self, parent: "MCTSAgent.Node | None", action: Action | None, untried: list[Action]) -> None:
            self.parent = parent
            self.action = action
            self.children: list["MCTSAgent.Node"] = []
            self.untried = list(untried)
            self.visits = 0
            self.value = 0.0

    def select_action(self, state: GameState) -> Action:
        # Reset root reuse to avoid stale trees across turns/players.
        self.root = None
        actions = prune_and_order_actions(state, state.legal_actions(), top_k=self.prune_top_k)
        if not actions:
            raise RuntimeError("no legal actions available")
        if len(actions) == 1:
            if self.track_stats:
                self.stats_log.append(
                    {"legal_actions": 1, "unique_nodes": 1, "backprop_visits": 0, "simulations": 0}
                )
            return actions[0]
        root = self.Node(parent=None, action=None, untried=actions)
        perspective = state.current_player
        start = perf_counter()
        backprop_visits = 0

        while perf_counter() - start < self.time_budget_s:
            sim_state = state.clone()
            node = root

            # Selection
            while not node.untried and node.children and not sim_state.is_terminal():
                node = self._uct_select(node)
                action = node.action
                if action is None:
                    break
                sim_state.apply_action(action)

            # Expansion
            if node.untried and not sim_state.is_terminal() and node.visits >= self.expand_threshold:
                action = self.rng.choice(node.untried)
                node.untried.remove(action)
                sim_state.apply_action(action)
                child = self.Node(parent=node, action=action, untried=sim_state.legal_actions())
                node.children.append(child)
                node = child

            # Rollout
            reward = self._rollout(sim_state, perspective)

            # Backprop
            while node is not None:
                node.visits += 1
                node.value += reward
                backprop_visits += 1
                node = node.parent

        # Choose best child by average value.
        if not root.children:
            return _sorted_actions(actions)[0]
        best = max(root.children, key=lambda n: n.value / n.visits if n.visits else float("-inf"))
        # Reuse tree by setting best child as new root.
        best.parent = None
        self.root = best

        if self.track_stats:
            self.stats_log.append(
                {
                    "legal_actions": len(actions),
                    "unique_nodes": self._count_nodes(root),
                    "backprop_visits": backprop_visits,
                    "simulations": backprop_visits,
                }
            )
        if best.action is None:
            raise RuntimeError("mcts selected an empty action")
        return best.action

    def _uct_select(self, node: "Node") -> "Node":
        assert node.children
        parent_visits = max(node.visits, 1)
        def uct_score(n: "MCTSAgent.Node") -> float:
            if n.visits == 0:
                return float("inf")
            exploit = n.value / n.visits
            explore = self.c_param * math.sqrt(math.log(parent_visits) / n.visits)
            return exploit + explore

        return max(node.children, key=uct_score)

    def _rollout(self, state: GameState, perspective: int) -> float:
        return _rollout_value(
            state,
            perspective,
            self.rollout_depth,
            self.rollout_policy,
            self.rng,
            True,
        )

    def _count_nodes(self, node: "Node") -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)
