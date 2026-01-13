from dataclasses import dataclass

from .enums import TileColor
from .state import GameState, WALL_COLOR_TO_COL

BOARD_SIZE = 5


@dataclass
class PlayerStats:
    wall_placements: int = 0
    isolated_placements: int = 0
    adjacent_placements: int = 0
    adjacency_points: int = 0
    floor_penalty: int = 0
    endgame_bonus: int = 0
    final_score: int = 0


@dataclass
class GameStats:
    per_player: list[PlayerStats]
    total_moves: int


def _line_len(wall, r, c, dr, dc):
    length = 1
    rr = r + dr
    cc = c + dc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and wall[rr][cc]:
        length += 1
        rr += dr
        cc += dc
    rr = r - dr
    cc = c - dc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and wall[rr][cc]:
        length += 1
        rr -= dr
        cc -= dc
    return length


def compute_stats(state: GameState) -> GameStats:
    players = state.players
    stats = [PlayerStats(final_score=p.score) for p in players]

    for idx, player in enumerate(players):
        wall = player.wall
        player_stats = stats[idx]
        for r in range(BOARD_SIZE):
            row = wall[r]
            for c in range(BOARD_SIZE):
                if not row[c]:
                    continue
                player_stats.wall_placements += 1
                horiz = _line_len(wall, r, c, 0, 1)
                vert = _line_len(wall, r, c, 1, 0)
                if horiz == 1 and vert == 1:
                    player_stats.isolated_placements += 1
                    player_stats.adjacency_points += 1
                else:
                    player_stats.adjacent_placements += 1
                    player_stats.adjacency_points += horiz + vert - 1

    for idx, player in enumerate(players):
        wall = player.wall
        player_stats = stats[idx]
        player_stats.endgame_bonus += 2 * sum(1 for row in wall if all(row))
        player_stats.endgame_bonus += 7 * sum(
            1
            for c in range(BOARD_SIZE)
            if all(wall[r][c] for r in range(BOARD_SIZE))
        )
        player_stats.endgame_bonus += 10 * sum(
            1
            for color in TileColor
            if all(wall[r][WALL_COLOR_TO_COL[r][color]] for r in range(BOARD_SIZE))
        )

    total_moves = state.round_number * len(players) * BOARD_SIZE

    return GameStats(per_player=stats, total_moves=total_moves)
