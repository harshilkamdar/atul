from dataclasses import dataclass, field

from .enums import TileColor

PATTERN_LINE_SIZES = (1, 2, 3, 4, 5)


def default_pattern_lines():
    return [[] for _ in PATTERN_LINE_SIZES]


def default_wall():
    return [[False] * 5 for _ in range(5)]


@dataclass
class PlayerBoard:
    """Represents a player's board, score, and penalties."""

    pattern_lines: list[list[TileColor]] = field(default_factory=default_pattern_lines)
    wall: list[list[bool]] = field(default_factory=default_wall)
    floor_line: list[TileColor] = field(default_factory=list)
    has_first_player_token: bool = False
    score: int = 0

    def clone(self) -> "PlayerBoard":
        return PlayerBoard(
            pattern_lines=[list(line) for line in self.pattern_lines],
            wall=[list(row) for row in self.wall],
            floor_line=list(self.floor_line),
            has_first_player_token=self.has_first_player_token,
            score=self.score,
        )
