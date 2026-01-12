from .state import FLOOR_PENALTIES, WALL_PATTERN, GameState
from .enums import TileColor


def _leaf_evaluation(state: GameState, perspective: int) -> float:
    """Heuristic value for non-terminal rollout cutoff."""
    scores = [p.score for p in state.players]
    opp_best = max(s for i, s in enumerate(scores) if i != perspective) if len(scores) > 1 else 0
    margin = scores[perspective] - opp_best

    player = state.players[perspective]
    progress = 0.0
    for idx, line in enumerate(player.pattern_lines):
        capacity = idx + 1
        progress += len(line) / capacity

    floor_count = len(player.floor_line) + (1 if player.has_first_player_token else 0)
    penalty = sum(FLOOR_PENALTIES[: min(floor_count, len(FLOOR_PENALTIES))])

    row_bonus = sum(1 for r in player.wall if sum(r) >= 4)
    col_bonus = 0
    for c in range(5):
        filled = sum(player.wall[r][c] for r in range(5))
        if filled >= 4:
            col_bonus += 1
    color_counts = {color: 0 for color in TileColor}
    for r in range(5):
        for c in range(5):
            if player.wall[r][c]:
                color_counts[WALL_PATTERN[r][c]] += 1
    color_potential = sum(1 for v in color_counts.values() if v >= 4)

    return margin + 0.5 * progress - 0.3 * penalty + 2.0 * row_bonus + 3.0 * col_bonus + 2.0 * color_potential
