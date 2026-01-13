from dataclasses import dataclass

from .enums import TileColor


@dataclass
class Action:
    source_index: int  # factory index, or -1 for center
    color: TileColor
    pattern_line: int  # 0-4 for pattern lines, -1 for floor
    take_first_player_token: bool = False

    FLOOR = -1
    CENTER = -1
