import random
from types import SimpleNamespace

from azul_agents import (
    FirstLegalAgent,
    GreedyFillAgent,
    MCTSAgent,
    ParallelRolloutAgent,
    RandomAgent,
    RolloutAgent,
    TimedRolloutAgent,
)
from azul_engine import Action, GameEngine, TileColor


def test_random_agent_picks_from_available():
    actions = [
        Action(source_index=0, color=TileColor.BLUE, pattern_line=0),
        Action(source_index=0, color=TileColor.BLUE, pattern_line=1),
    ]
    state = SimpleNamespace(legal_actions=lambda: list(actions))
    agent = RandomAgent(rng=random.Random(42))

    chosen = agent.select_action(state)
    assert chosen in actions


def test_first_legal_agent_prefers_pattern_line_over_floor():
    engine = GameEngine(seed=10)
    state = engine.reset()
    state.supply.factories = [[TileColor.BLUE], *[[] for _ in range(1, len(state.supply.factories))]]
    state.supply.center = []

    action = FirstLegalAgent().select_action(state)
    assert action.source_index == 0
    assert action.color == TileColor.BLUE
    assert action.pattern_line == 0  # chooses smallest pattern line, not floor


def test_greedy_fill_completes_pattern_line_when_possible():
    engine = GameEngine(seed=11)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.YELLOW, TileColor.BLUE, TileColor.BLUE]
    state.players[0].pattern_lines[1] = [TileColor.YELLOW]  # needs one more to complete

    action = GreedyFillAgent().select_action(state)
    assert action.source_index == Action.CENTER
    assert action.color == TileColor.YELLOW
    assert action.pattern_line == 1


def test_rollout_agent_prefers_completing_line():
    engine = GameEngine(seed=12)
    state = engine.reset()
    # No randomness from bag/discard during playouts.
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.BLUE, TileColor.BLUE]
    state.supply.bag = []
    state.supply.discard = []
    state.players[0].pattern_lines[0] = []

    agent = RolloutAgent(rollout_count=4, depth_limit=10, rng=random.Random(1))
    action = agent.select_action(state)

    assert action.pattern_line != Action.FLOOR  # should place, not dump to floor


def test_timed_rollout_runs_within_budget():
    engine = GameEngine(seed=13)
    state = engine.reset()
    agent = TimedRolloutAgent(time_budget_s=0.05, depth_limit=20, rng=random.Random(2))

    action = agent.select_action(state)
    assert action in state.legal_actions()


def test_mcts_returns_legal_action():
    engine = GameEngine(seed=14)
    state = engine.reset()
    agent = MCTSAgent(time_budget_s=0.05, rollout_depth=30, rng=random.Random(3))
    action = agent.select_action(state)
    assert action in state.legal_actions()


def test_parallel_rollout_returns_legal_action():
    engine = GameEngine(seed=15)
    state = engine.reset()
    agent = ParallelRolloutAgent(rollouts_per_action=1, depth_limit=10, workers=2, rollout_policy_name="random")
    action = agent.select_action(state)
    assert action in state.legal_actions()
