import random

from azul_engine import GamePhase, GameState, PlayerBoard, Supply, TileColor


def state_from_snapshot(snapshot: dict) -> GameState:
    def to_colors(values) -> list[TileColor]:
        return [TileColor(v) for v in values]

    supply_data = snapshot["supply"]
    supply = Supply(
        bag=to_colors(supply_data["bag"]),
        discard=to_colors(supply_data["discard"]),
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
        first_player_index=int(snapshot["first_player_index"]),
        rng=rng,
        round_log=list(snapshot.get("round_log", [])),
    )
