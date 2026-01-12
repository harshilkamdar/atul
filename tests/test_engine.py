import pytest

from azul_engine import Action, GameEngine, GamePhase, TileColor


def test_reset_sets_up_factories_and_token():
    engine = GameEngine(seed=0)
    state = engine.reset()

    assert state.phase == GamePhase.DRAFTING
    assert state.first_player_token_in_center is True
    assert len(state.supply.factories) == 5
    assert all(len(f) == 4 for f in state.supply.factories)
    bag_size = len(state.supply.bag)
    assert bag_size == (5 * 20) - (len(state.supply.factories) * 4)


def test_legal_actions_respects_wall_constraint():
    engine = GameEngine(seed=1)
    state = engine.reset()
    # Force a simple source: factory 0 has two blues and two reds.
    state.supply.factories = [[TileColor.BLUE, TileColor.BLUE, TileColor.RED, TileColor.RED]] + [
        [] for _ in range(1, len(state.supply.factories))
    ]
    state.supply.center = []
    # Block blue in row 0 on the wall.
    state.players[0].wall[0][0] = True

    actions = state.legal_actions()
    blue_row0 = [
        a
        for a in actions
        if a.color == TileColor.BLUE and a.pattern_line == 0 and a.source_index == 0
    ]
    assert blue_row0 == []


def test_apply_action_from_center_handles_overflow_and_token():
    engine = GameEngine(seed=2)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.BLUE, TileColor.BLUE, TileColor.RED]
    state.first_player_token_in_center = True
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLUE,
        pattern_line=0,
        take_first_player_token=True,
    )

    state.apply_action(action)

    assert state.players[0].pattern_lines[0] == [TileColor.BLUE]
    assert state.players[0].floor_line.count(TileColor.BLUE) == 1  # overflow tile
    assert state.players[0].has_first_player_token is True
    assert state.first_player_token_in_center is False
    assert TileColor.RED in state.supply.center


def test_round_scoring_and_refill_advances_round():
    engine = GameEngine(seed=3)
    state = engine.reset()
    # Empty sources to trigger end of drafting on next advance.
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = []
    # Give player 0 a completed pattern line and a floor penalty tile.
    state.players[0].pattern_lines[0] = [TileColor.BLUE]
    state.players[0].floor_line = [TileColor.RED]
    state.players[0].has_first_player_token = True

    state._advance_turn_after_draft()

    assert state.round_number == 2
    assert state.phase == GamePhase.DRAFTING
    assert state.players[0].wall[0][0] is True
    # Score is clamped at zero after penalties.
    assert state.players[0].score == 0
    assert state.current_player == 0  # first player token holder
    assert state.first_player_token_in_center is True


def test_legal_actions_center_token_flag():
    engine = GameEngine(seed=4)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.BLUE]
    state.first_player_token_in_center = True

    actions = state.legal_actions()
    assert actions
    assert all(a.source_index == Action.CENTER for a in actions)
    assert all(a.take_first_player_token for a in actions)

    state.first_player_token_in_center = False
    actions = state.legal_actions()
    assert actions
    assert all(a.source_index == Action.CENTER for a in actions)
    assert all(not a.take_first_player_token for a in actions)


def test_apply_action_requires_token_in_center():
    engine = GameEngine(seed=5)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.BLUE]
    state.first_player_token_in_center = True
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLUE,
        pattern_line=0,
        take_first_player_token=False,
    )
    with pytest.raises(ValueError, match="must take first player token"):
        state.apply_action(action)


def test_apply_action_rejects_token_when_absent():
    engine = GameEngine(seed=6)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = [TileColor.BLUE]
    state.first_player_token_in_center = False
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLUE,
        pattern_line=0,
        take_first_player_token=True,
    )
    with pytest.raises(ValueError, match="cannot take first player token"):
        state.apply_action(action)


def test_apply_action_outside_drafting_errors():
    engine = GameEngine(seed=7)
    state = engine.reset()
    state.phase = GamePhase.WALL_TILING
    action = Action(source_index=0, color=TileColor.BLUE, pattern_line=0)
    with pytest.raises(RuntimeError, match="cannot apply draft action"):
        state.apply_action(action)


def test_state_clone_deep_copy():
    engine = GameEngine(seed=8)
    state = engine.reset()
    original_factory = list(state.supply.factories[0])
    original_center = list(state.supply.center)
    original_pattern = [list(line) for line in state.players[0].pattern_lines]
    original_score = state.players[0].score
    clone = state.clone()

    clone.players[0].pattern_lines[0].append(TileColor.BLUE)
    clone.players[0].score = 3
    clone.supply.center.append(TileColor.RED)
    clone.supply.factories[0].append(TileColor.YELLOW)

    assert state.players[0].pattern_lines == original_pattern
    assert state.players[0].score == original_score
    assert state.supply.center == original_center
    assert state.supply.factories[0] == original_factory


def test_draw_tiles_refills_from_discard():
    engine = GameEngine(seed=9)
    state = engine.reset()
    state.supply.bag = []
    state.supply.discard = [
        TileColor.BLUE,
        TileColor.RED,
        TileColor.YELLOW,
        TileColor.BLACK,
        TileColor.WHITE,
    ]
    drawn = state._draw_tiles(4)
    assert len(drawn) == 4
    assert state.supply.discard == []
    assert len(state.supply.bag) == 1
    assert set(drawn).issubset({TileColor.BLUE, TileColor.RED, TileColor.YELLOW, TileColor.BLACK, TileColor.WHITE})


def test_refill_factories_clears_center():
    engine = GameEngine(seed=10)
    state = engine.reset()
    needed = len(state.supply.factories) * 4
    state.supply.bag = [TileColor.BLUE] * needed
    state.supply.discard = []
    state.supply.center = [TileColor.RED]

    state._refill_factories()

    assert state.supply.center == []
    assert all(len(f) == 4 for f in state.supply.factories)
