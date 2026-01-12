from dataclasses import dataclass

from .state import GameState, WALL_COLOR_TO_COL
from .enums import TileColor


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


def compute_stats(state: GameState) -> GameStats:
    players = state.players
    stats = [PlayerStats(final_score=p.score) for p in players]

    for idx, player in enumerate(players):
        for r in range(5):
            for c in range(5):
                if player.wall[r][c]:
                    stats[idx].wall_placements += 1
                    horiz = 1
                    cc = c - 1
                    while cc >= 0 and player.wall[r][cc]:
                        horiz += 1
                        cc -= 1
                    cc = c + 1
                    while cc < 5 and player.wall[r][cc]:
                        horiz += 1
                        cc += 1
                    vert = 1
                    rr = r - 1
                    while rr >= 0 and player.wall[rr][c]:
                        vert += 1
                        rr -= 1
                    rr = r + 1
                    while rr < 5 and player.wall[rr][c]:
                        vert += 1
                        rr += 1
                    if horiz == 1 and vert == 1:
                        stats[idx].isolated_placements += 1
                        stats[idx].adjacency_points += 1
                    else:
                        stats[idx].adjacent_placements += 1
                        stats[idx].adjacency_points += horiz + vert - 1

    for idx, player in enumerate(players):
        for r in range(5):
            if all(player.wall[r]):
                stats[idx].endgame_bonus += 2
        for c in range(5):
            if all(player.wall[r][c] for r in range(5)):
                stats[idx].endgame_bonus += 7
        for color in TileColor:
            if all(player.wall[r][WALL_COLOR_TO_COL[r][color]] for r in range(5)):
                stats[idx].endgame_bonus += 10

    total_moves = state.round_number * len(players) * 5  # upper bound proxy if we didn't log

    return GameStats(per_player=stats, total_moves=total_moves)
