import random

from .actions import Action
from .enums import GamePhase, TileColor
from .player import PlayerBoard
from .state import GameState, Supply


def action_to_dict(action: Action) -> dict:
    return {
        "source_index": action.source_index,
        "color": action.color.value,
        "pattern_line": action.pattern_line,
        "take_first_player_token": action.take_first_player_token,
    }


def action_from_dict(data: dict) -> Action:
    return Action(
        source_index=int(data["source_index"]),
        color=TileColor(data["color"]),
        pattern_line=int(data["pattern_line"]),
        take_first_player_token=bool(data["take_first_player_token"]),
    )


def state_to_dict(
    state: GameState,
    *,
    include_supply_contents: bool = False,
    include_rng: bool = False,
    include_round_log: bool = False,
    include_first_player_index: bool = False,
) -> dict:
    def board(p) -> dict:
        return {
            "score": p.score,
            "pattern_lines": [[t.value for t in line] for line in p.pattern_lines],
            "wall": [[bool(x) for x in row] for row in p.wall],
            "floor_line": [t.value for t in p.floor_line],
            "has_first_player_token": p.has_first_player_token,
        }

    supply = {
        "factories": [[t.value for t in f] for f in state.supply.factories],
        "center": [t.value for t in state.supply.center],
        "bag_count": len(state.supply.bag),
        "discard_count": len(state.supply.discard),
        "first_player_token_in_center": state.first_player_token_in_center,
    }
    if include_supply_contents:
        supply["bag"] = [t.value for t in state.supply.bag]
        supply["discard"] = [t.value for t in state.supply.discard]

    data = {
        "round": state.round_number,
        "phase": state.phase.value,
        "current_player": state.current_player,
        "supply": supply,
        "players": [board(p) for p in state.players],
    }
    if include_first_player_index:
        data["first_player_index"] = state.first_player_index
    if include_round_log:
        data["round_log"] = list(state.round_log)
    if include_rng:
        version, internal, gauss = state.rng.getstate()
        data["rng_state"] = {
            "version": version,
            "internal_state": list(internal),
            "gauss_next": gauss,
        }
    return data


def state_from_dict(snapshot: dict, *, require_supply_contents: bool = True) -> GameState:
    def to_colors(values) -> list[TileColor]:
        return [TileColor(v) for v in values]

    supply_data = snapshot["supply"]
    if require_supply_contents and ("bag" not in supply_data or "discard" not in supply_data):
        raise ValueError("snapshot missing supply contents")
    supply = Supply(
        bag=to_colors(supply_data.get("bag", [])),
        discard=to_colors(supply_data.get("discard", [])),
        factories=[to_colors(f) for f in supply_data["factories"]],
        center=to_colors(supply_data["center"]),
    )

    players = []
    for pdata in snapshot["players"]:
        players.append(
            PlayerBoard(
                pattern_lines=[to_colors(line) for line in pdata["pattern_lines"]],
                wall=[[bool(x) for x in row] for row in pdata["wall"]],
                floor_line=to_colors(pdata["floor_line"]),
                has_first_player_token=bool(pdata["has_first_player_token"]),
                score=int(pdata["score"]),
            )
        )

    rng = random.Random()
    rng_state = snapshot.get("rng_state")
    if rng_state:
        rng.setstate(
            (
                rng_state["version"],
                tuple(rng_state["internal_state"]),
                rng_state["gauss_next"],
            )
        )

    return GameState(
        players=players,
        current_player=int(snapshot["current_player"]),
        phase=GamePhase(snapshot["phase"]),
        supply=supply,
        round_number=int(snapshot["round"]),
        first_player_token_in_center=bool(supply_data["first_player_token_in_center"]),
        first_player_index=int(snapshot.get("first_player_index", 0)),
        rng=rng,
        round_log=list(snapshot.get("round_log", [])),
    )
