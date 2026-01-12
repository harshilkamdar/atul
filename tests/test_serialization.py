import pytest

from azul_engine import Action, GameEngine, TileColor
from azul_engine.serialization import (
    action_from_dict,
    action_to_dict,
    state_from_dict,
    state_to_dict,
)


def test_action_round_trip():
    action = Action(
        source_index=Action.CENTER,
        color=TileColor.BLUE,
        pattern_line=2,
        take_first_player_token=True,
    )
    payload = action_to_dict(action)
    restored = action_from_dict(payload)
    assert restored == action


def test_state_round_trip_with_rng():
    state = GameEngine(seed=123).reset()
    state.round_log = [{"round": 1, "player": 0, "gained": 2, "floor_penalty": -1, "score_after": 1}]
    state.first_player_index = 1
    state.rng.random()
    snapshot = state_to_dict(
        state,
        include_supply_contents=True,
        include_rng=True,
        include_round_log=True,
        include_first_player_index=True,
    )
    restored = state_from_dict(snapshot)

    assert restored.round_number == state.round_number
    assert restored.phase == state.phase
    assert restored.current_player == state.current_player
    assert restored.first_player_index == state.first_player_index
    assert restored.round_log == state.round_log
    assert restored.rng.getstate() == state.rng.getstate()
    assert [t.value for t in restored.supply.bag] == [t.value for t in state.supply.bag]
    assert [t.value for t in restored.supply.discard] == [t.value for t in state.supply.discard]

    for left, right in zip(restored.players, state.players):
        assert left.score == right.score
        assert [[t.value for t in line] for line in left.pattern_lines] == [
            [t.value for t in line] for line in right.pattern_lines
        ]
        assert left.wall == right.wall
        assert [t.value for t in left.floor_line] == [t.value for t in right.floor_line]
        assert left.has_first_player_token == right.has_first_player_token


def test_state_to_dict_minimal_excludes_hidden():
    state = GameEngine(seed=0).reset()
    snapshot = state_to_dict(state)
    supply = snapshot["supply"]
    assert "bag" not in supply
    assert "discard" not in supply
    assert "rng_state" not in snapshot
    assert "round_log" not in snapshot


def test_state_from_dict_requires_supply_contents():
    state = GameEngine(seed=0).reset()
    snapshot = state_to_dict(state)
    with pytest.raises(ValueError):
        state_from_dict(snapshot)
