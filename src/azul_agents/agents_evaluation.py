from azul_engine.enums import TileColor
from azul_engine.state import FLOOR_PENALTIES, WALL_COLOR_TO_COL, WALL_PATTERN, GameState


def _leaf_evaluation(state: GameState, perspective: int) -> float:
    """Heuristic value for non-terminal rollout cutoff."""
    scores = [p.score for p in state.players]
    opp_best = max((s for i, s in enumerate(scores) if i != perspective), default=0)
    margin = scores[perspective] - opp_best

    player = state.players[perspective]
    progress = 0.0
    for idx, line in enumerate(player.pattern_lines):
        progress += len(line) / (idx + 1)

    floor_count = len(player.floor_line) + player.has_first_player_token
    penalty = sum(FLOOR_PENALTIES[:floor_count])

    row_bonus = sum(1 for row in player.wall if sum(row) >= 4)
    col_bonus = sum(1 for c in range(5) if sum(player.wall[r][c] for r in range(5)) >= 4)
    color_potential = sum(
        1
        for color in TileColor
        if sum(player.wall[r][WALL_COLOR_TO_COL[r][color]] for r in range(5)) >= 4
    )

    return margin + 0.5 * progress - 0.3 * penalty + 2.0 * row_bonus + 3.0 * col_bonus + 2.0 * color_potential
