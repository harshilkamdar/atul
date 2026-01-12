import random
from dataclasses import dataclass, field

from .actions import Action
from .enums import GamePhase, TileColor
from .player import PATTERN_LINE_SIZES, PlayerBoard

WALL_PATTERN = [
    [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE],
    [TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK],
    [TileColor.BLACK, TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW, TileColor.RED],
    [TileColor.RED, TileColor.BLACK, TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW],
    [TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE, TileColor.BLUE],
]
WALL_COLOR_TO_COL = [{color: col for col, color in enumerate(row)} for row in WALL_PATTERN]

FLOOR_PENALTIES = (-1, -1, -2, -2, -2, -3, -3)
TILES_PER_COLOR = 20


@dataclass
class Supply:
    bag: list[TileColor] = field(default_factory=list)
    discard: list[TileColor] = field(default_factory=list)
    factories: list[list[TileColor]] = field(default_factory=list)
    center: list[TileColor] = field(default_factory=list)

    def clone(self) -> "Supply":
        return Supply(
            bag=list(self.bag),
            discard=list(self.discard),
            factories=[list(f) for f in self.factories],
            center=list(self.center),
        )


@dataclass
class GameState:
    players: list[PlayerBoard]
    current_player: int
    phase: GamePhase
    supply: Supply
    round_number: int = 1
    first_player_token_in_center: bool = True
    first_player_index: int = 0
    rng: random.Random = field(default_factory=random.Random, repr=False)
    round_log: list[dict[str, int | float]] = field(default_factory=list)

    def clone(self) -> "GameState":
        clone_rng = random.Random()
        clone_rng.setstate(self.rng.getstate())
        return GameState(
            players=[p.clone() for p in self.players],
            current_player=self.current_player,
            phase=self.phase,
            supply=self.supply.clone(),
            round_number=self.round_number,
            first_player_token_in_center=self.first_player_token_in_center,
            first_player_index=self.first_player_index,
            rng=clone_rng,
        )

    def legal_actions(self) -> list[Action]:
        if self.phase != GamePhase.DRAFTING:
            return []
        player = self.players[self.current_player]
        actions = []
        append = actions.append
        can_place = self._can_place_in_line
        line_count = len(PATTERN_LINE_SIZES)
        floor = Action.FLOOR
        center_idx = Action.CENTER

        for source_index, tiles in enumerate(self.supply.factories):
            if not tiles:
                continue
            for color in set(tiles):
                for line_idx in range(line_count):
                    if not can_place(player, line_idx, color):
                        continue
                    append(
                        Action(
                            source_index=source_index,
                            color=color,
                            pattern_line=line_idx,
                            take_first_player_token=False,
                        )
                    )
                append(
                    Action(
                        source_index=source_index,
                        color=color,
                        pattern_line=floor,
                        take_first_player_token=False,
                    )
                )

        center_tiles = self.supply.center
        if center_tiles:
            take_token = self.first_player_token_in_center
            for color in set(center_tiles):
                for line_idx in range(line_count):
                    if not can_place(player, line_idx, color):
                        continue
                    append(
                        Action(
                            source_index=center_idx,
                            color=color,
                            pattern_line=line_idx,
                            take_first_player_token=take_token,
                        )
                    )
                append(
                    Action(
                        source_index=center_idx,
                        color=color,
                        pattern_line=floor,
                        take_first_player_token=take_token,
                    )
                )
        return actions

    def apply_action(self, action: Action) -> "GameState":
        if self.phase != GamePhase.DRAFTING:
            raise RuntimeError("cannot apply draft action outside drafting phase")
        player = self.players[self.current_player]
        source_tiles = self._get_source_tiles(action.source_index)
        if action.color not in source_tiles:
            raise ValueError("chosen color not in source")
        take_token = action.source_index == Action.CENTER and self.first_player_token_in_center
        if action.take_first_player_token != take_token:
            if take_token:
                raise ValueError("must take first player token when present in center")
            raise ValueError("cannot take first player token when absent")

        taken = [t for t in source_tiles if t == action.color]
        remaining = [t for t in source_tiles if t != action.color]
        if action.source_index == Action.CENTER:
            self.supply.center = remaining
            if take_token:
                self.first_player_token_in_center = False
                self.first_player_index = self.current_player
        else:
            self.supply.factories[action.source_index] = []
            self.supply.center.extend(remaining)

        self._place_tiles(player, taken, action)

        self._advance_turn_after_draft()
        return self

    def is_terminal(self) -> bool:
        return self.phase == GamePhase.GAME_END

    def _get_source_tiles(self, source_index: int) -> list[TileColor]:
        if source_index == Action.CENTER:
            return list(self.supply.center)
        return list(self.supply.factories[source_index])

    def _can_place_in_line(self, player: PlayerBoard, line_idx: int, color: TileColor) -> bool:
        if player.wall[line_idx][WALL_COLOR_TO_COL[line_idx][color]]:
            return False
        line = player.pattern_lines[line_idx]
        return len(line) < PATTERN_LINE_SIZES[line_idx] and (not line or line[0] == color)

    def _place_tiles(self, player: PlayerBoard, tiles: list[TileColor], action: Action) -> None:
        if action.pattern_line == Action.FLOOR:
            player.floor_line.extend(tiles)
            if action.take_first_player_token:
                player.has_first_player_token = True
            return

        line_idx = action.pattern_line
        if not 0 <= line_idx < len(PATTERN_LINE_SIZES):
            raise ValueError("invalid pattern line index")
        if not self._can_place_in_line(player, line_idx, action.color):
            raise ValueError("cannot place chosen color in that pattern line")
        capacity = PATTERN_LINE_SIZES[line_idx]
        line = player.pattern_lines[line_idx]
        space = capacity - len(line)
        to_line = min(space, len(tiles))
        overflow = len(tiles) - to_line
        line.extend([action.color] * to_line)
        if overflow:
            player.floor_line.extend([action.color] * overflow)
        if action.take_first_player_token:
            player.has_first_player_token = True

    def _advance_turn_after_draft(self) -> None:
        drafting_done = all(not f for f in self.supply.factories) and not self.supply.center
        if drafting_done:
            self.phase = GamePhase.WALL_TILING
            self._score_and_refill()
        else:
            self.current_player = (self.current_player + 1) % len(self.players)

    def _score_and_refill(self) -> None:
        for idx, player in enumerate(self.players):
            gained_this_round = 0
            for line_idx, capacity in enumerate(PATTERN_LINE_SIZES):
                line = player.pattern_lines[line_idx]
                if len(line) != capacity:
                    continue
                color = line[0]
                col = WALL_COLOR_TO_COL[line_idx][color]
                player.wall[line_idx][col] = True
                gained = self._score_placement(player, line_idx, col)
                player.score += gained
                gained_this_round += gained
                if len(line) > 1:
                    self.supply.discard.extend(line[1:])
                player.pattern_lines[line_idx] = []
            floor_count = len(player.floor_line) + (1 if player.has_first_player_token else 0)
            penalty = sum(FLOOR_PENALTIES[: min(floor_count, len(FLOOR_PENALTIES))])
            player.score += penalty
            self.supply.discard.extend(player.floor_line)
            if player.has_first_player_token:
                self.first_player_index = idx
            player.floor_line = []
            player.has_first_player_token = False
            player.score = max(player.score, 0)
            self.round_log.append(
                {
                    "round": self.round_number,
                    "player": idx,
                    "gained": gained_this_round,
                    "floor_penalty": penalty,
                    "score_after": player.score,
                }
            )

        if any(all(row) for p in self.players for row in p.wall):
            self.phase = GamePhase.GAME_END
            self._apply_end_game_bonuses()
            return

        self.round_number += 1
        self.first_player_token_in_center = True
        self.current_player = self.first_player_index
        self._refill_factories()
        self.phase = GamePhase.DRAFTING

    def _score_placement(self, player: PlayerBoard, row: int, col: int) -> int:
        horizontal = 1
        c = col - 1
        while c >= 0 and player.wall[row][c]:
            horizontal += 1
            c -= 1
        c = col + 1
        while c < 5 and player.wall[row][c]:
            horizontal += 1
            c += 1
        vertical = 1
        r = row - 1
        while r >= 0 and player.wall[r][col]:
            vertical += 1
            r -= 1
        r = row + 1
        while r < 5 and player.wall[r][col]:
            vertical += 1
            r += 1
        horiz_score = horizontal if horizontal > 1 else 0
        vert_score = vertical if vertical > 1 else 0
        if horiz_score == 0 and vert_score == 0:
            return 1
        return horiz_score + vert_score

    def _apply_end_game_bonuses(self) -> None:
        for player in self.players:
            for row in player.wall:
                if all(row):
                    player.score += 2
            for col in range(5):
                if all(player.wall[r][col] for r in range(5)):
                    player.score += 7
            for color in TileColor:
                if all(player.wall[r][WALL_COLOR_TO_COL[r][color]] for r in range(5)):
                    player.score += 10

    def _refill_factories(self) -> None:
        for f in range(len(self.supply.factories)):
            self.supply.factories[f] = []
            tiles = self._draw_tiles(4)
            self.supply.factories[f].extend(tiles)
        self.supply.center = []

    def _draw_tiles(self, count: int) -> list[TileColor]:
        drawn = []
        while len(drawn) < count:
            if not self.supply.bag:
                if not self.supply.discard:
                    break
                self.rng.shuffle(self.supply.discard)
                self.supply.bag.extend(self.supply.discard)
                self.supply.discard.clear()
            drawn.append(self.supply.bag.pop())
        return drawn


class GameEngine:
    def __init__(self, num_players: int = 2, *, seed: int | None = None) -> None:
        if num_players not in (2, 3, 4):
            raise ValueError("Azul supports 2-4 players")
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.state: GameState | None = None

    def reset(self) -> GameState:
        factories_by_players = {2: 5, 3: 7, 4: 9}[self.num_players]
        bag = []
        for color in TileColor:
            bag.extend([color] * TILES_PER_COLOR)
        self.rng.shuffle(bag)
        supply = Supply(
            bag=bag,
            discard=[],
            factories=[[] for _ in range(factories_by_players)],
            center=[],
        )
        state = GameState(
            players=[PlayerBoard() for _ in range(self.num_players)],
            current_player=0,
            phase=GamePhase.DRAFTING,
            supply=supply,
            round_number=1,
            first_player_token_in_center=True,
            first_player_index=0,
            rng=self.rng,
        )
        self.state = state
        state._refill_factories()
        return state

    def legal_actions(self) -> list[Action]:
        if self.state is None:
            raise RuntimeError("engine not initialized; call reset() first")
        return self.state.legal_actions()

    def step(self, action: Action) -> GameState:
        if self.state is None:
            raise RuntimeError("engine not initialized; call reset() first")
        return self.state.apply_action(action)
