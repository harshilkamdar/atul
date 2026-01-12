from azul_engine import Action, GameEngine, GamePhase, TileColor
from azul_engine.state import WALL_PATTERN


def test_isolated_and_adjacency_scoring():
    engine = GameEngine(seed=0)
    state = engine.reset()
    player = state.players[0]

    # Manually score placements on an empty wall.
    player.wall = [[False] * 5 for _ in range(5)]
    assert state._score_placement(player, 0, 0) == 1  # isolated

    # Horizontal run of length 3.
    player.wall[0][0] = True
    player.wall[0][1] = True
    assert state._score_placement(player, 0, 2) == 3

    # Vertical run of length 3.
    player.wall = [[False] * 5 for _ in range(5)]
    player.wall[0][0] = True
    player.wall[1][0] = True
    assert state._score_placement(player, 2, 0) == 3

    # Both directions.
    player.wall = [[False] * 5 for _ in range(5)]
    player.wall[0][0] = True
    player.wall[1][1] = True
    assert state._score_placement(player, 0, 1) == 4

    # Mixed 3 horizontal + 2 vertical (scores 5).
    player.wall = [[False] * 5 for _ in range(5)]
    player.wall[0][0] = True  # vertical support
    player.wall[1][1] = True
    player.wall[1][2] = True
    assert state._score_placement(player, 1, 0) == 5  # horiz len 3, vert len 2


def test_floor_line_penalty_and_score_floor():
    engine = GameEngine(seed=1)
    state = engine.reset()
    player = state.players[0]
    player.score = 5
    player.floor_line = [TileColor.BLUE] * 4  # penalty -6 -> clamps to 0
    state._score_and_refill()
    assert player.score == 0


def test_first_player_token_floor_counts_penalty():
    engine = GameEngine(seed=2)
    state = engine.reset()
    player = state.players[0]
    player.score = 3
    player.floor_line = [TileColor.RED]
    player.has_first_player_token = True
    state._score_and_refill()
    # Two tiles worth of penalty (-2 cumulative).
    assert player.score == 1
    assert state.first_player_index == 0


def test_first_player_token_only_penalty():
    engine = GameEngine(seed=12)
    state = engine.reset()
    player = state.players[1]
    player.score = 5
    player.floor_line = []
    player.has_first_player_token = True
    state._score_and_refill()
    # Token alone counts as one floor penalty (-1).
    assert player.score == 4
    assert state.first_player_index == 1


def test_pattern_line_completion_moves_only_rightmost_tile():
    engine = GameEngine(seed=3)
    state = engine.reset()
    # Clear sources to end drafting immediately.
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = []
    player = state.players[0]
    player.pattern_lines[2] = [TileColor.RED, TileColor.RED, TileColor.RED]
    state._advance_turn_after_draft()

    # Wall tile placed at correct column for RED in row 3 (index 2).
    col = WALL_PATTERN[2].index(TileColor.RED)
    assert player.wall[2][col] is True
    # Only two tiles discarded (line length - 1).
    assert state.supply.discard.count(TileColor.RED) == 2
    assert player.pattern_lines[2] == []


def test_adjacent_wall_placements_same_round_score():
    engine = GameEngine(seed=11)
    state = engine.reset()
    player = state.players[0]
    player.score = 0
    player.pattern_lines[0] = [TileColor.BLUE]
    player.pattern_lines[1] = [TileColor.WHITE, TileColor.WHITE]
    state.supply.bag = [TileColor.RED] * 50
    state.supply.discard = []

    state._score_and_refill()

    col0 = WALL_PATTERN[0].index(TileColor.BLUE)
    col1 = WALL_PATTERN[1].index(TileColor.WHITE)
    assert col0 == col1
    assert player.wall[0][col0] is True
    assert player.wall[1][col1] is True
    assert player.score == 3


def test_endgame_bonuses_row_column_color():
    engine = GameEngine(seed=4)
    state = engine.reset()
    player = state.players[0]
    # Build a full row and column and full color set for BLUE.
    player.wall = [[True] * 5 for _ in range(5)]
    for r in range(5):
        player.wall[r] = [False] * 5
    # Row 0 complete.
    player.wall[0] = [True] * 5
    # Column 0 complete.
    for r in range(5):
        player.wall[r][0] = True
    # Blue diagonal complete.
    for r in range(5):
        c = WALL_PATTERN[r].index(TileColor.BLUE)
        player.wall[r][c] = True

    player.score = 0
    state._apply_end_game_bonuses()
    # Row bonus +2, column bonus +7, color bonus +10 => 19 total.
    assert player.score == 19


def test_floor_overflow_caps_penalty():
    engine = GameEngine(seed=9)
    state = engine.reset()
    player = state.players[0]
    player.floor_line = [TileColor.RED] * 10  # more than 7 should not add extra penalty
    player.score = 20
    state._score_and_refill()
    # Max floor penalty -14, score floors at zero? starting from 20 -> 6
    assert player.score == 6


def test_game_end_trigger_on_completed_row():
    engine = GameEngine(seed=10)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.center = []
    player = state.players[0]
    # Complete a pattern line that will finish row 0.
    player.pattern_lines[0] = [TileColor.BLUE]
    # Fill wall to ensure row completes.
    for c in range(1, 5):
        player.wall[0][c] = True
    state._advance_turn_after_draft()
    assert state.phase == GamePhase.GAME_END
    assert player.score >= 2  # row bonus should be applied


def test_initial_setup_factories_and_bag():
    engine = GameEngine(num_players=2, seed=5)
    state = engine.reset()
    assert len(state.supply.factories) == 5
    assert all(len(f) == 4 for f in state.supply.factories)
    # 100 total tiles minus those on factories.
    assert len(state.supply.bag) == 100 - (5 * 4)
    assert state.first_player_token_in_center is True
    assert state.phase == GamePhase.DRAFTING


def test_factory_take_moves_rest_to_center():
    engine = GameEngine(seed=6)
    state = engine.reset()
    state.supply.factories = [[TileColor.RED, TileColor.RED, TileColor.BLUE, TileColor.YELLOW]] + [
        [] for _ in range(1, len(state.supply.factories))
    ]
    state.supply.center = []
    action = Action(source_index=0, color=TileColor.RED, pattern_line=0)
    engine.step(action)
    assert state.players[0].pattern_lines[0] == [TileColor.RED]
    assert state.supply.factories[0] == []
    assert sorted(state.supply.center) == [TileColor.BLUE, TileColor.YELLOW]


def test_center_first_player_token_to_floor():
    engine = GameEngine(seed=7)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.factories[0] = [TileColor.RED]  # keep drafting active
    state.supply.center = [TileColor.BLUE, TileColor.BLUE]
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLUE,
        pattern_line=Action.FLOOR,
        take_first_player_token=True,
    )
    engine.step(action)
    assert state.players[0].has_first_player_token is True
    assert len(state.players[0].floor_line) == 2  # two tiles; token tracked separately
    assert state.first_player_token_in_center is False


def test_pattern_line_overflow_sends_excess_to_floor():
    engine = GameEngine(seed=8)
    state = engine.reset()
    state.supply.factories = [[] for _ in state.supply.factories]
    state.supply.factories[0] = [TileColor.RED]  # keep drafting active
    state.supply.center = [TileColor.BLACK] * 3
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLACK,
        pattern_line=0,  # capacity 1, will overflow 2 to floor
        take_first_player_token=True,
    )
    engine.step(action)
    assert state.players[0].pattern_lines[0] == [TileColor.BLACK]
    assert len(state.players[0].floor_line) == 2
