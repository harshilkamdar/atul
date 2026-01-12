from enum import Enum


class TileColor(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"
    WHITE = "white"


class GamePhase(str, Enum):
    DRAFTING = "drafting"
    WALL_TILING = "wall_tiling"
    GAME_END = "game_end"
