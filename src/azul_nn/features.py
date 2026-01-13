import torch

from azul_engine.actions import Action
from azul_engine.enums import TileColor
from azul_engine.player import PATTERN_LINE_SIZES
from azul_engine.state import GameState

COLOR_ORDER = [
    TileColor.BLUE,
    TileColor.YELLOW,
    TileColor.RED,
    TileColor.BLACK,
    TileColor.WHITE,
]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_ORDER)}

MAX_FACTORIES = 9
DEST_SLOTS = 6  # pattern lines 0-4 plus floor
ACTION_DIM = (MAX_FACTORIES + 1) * len(COLOR_ORDER) * DEST_SLOTS
FEATURE_DIM = 185


def action_index_from_components(source_index: int, color: TileColor, dest_slot: int, num_factories: int) -> int:
    src = source_index if source_index != Action.CENTER else num_factories
    color_idx = COLOR_TO_IDX[color]
    return (src * len(COLOR_ORDER) + color_idx) * DEST_SLOTS + dest_slot


def components_from_action_index(idx: int, num_factories: int) -> tuple[int, TileColor, int]:
    src_block = idx // DEST_SLOTS
    dest = idx % DEST_SLOTS
    color_idx = src_block % len(COLOR_ORDER)
    src = src_block // len(COLOR_ORDER)
    source_index = src if src < num_factories else Action.CENTER
    return source_index, COLOR_ORDER[color_idx], dest


def legal_action_indices(state: GameState) -> list[int]:
    num_factories = len(state.supply.factories)
    return [
        action_index_from_components(
            act.source_index,
            act.color,
            act.pattern_line if act.pattern_line != Action.FLOOR else 5,
            num_factories,
        )
        for act in state.legal_actions()
    ]


def _pad_factories(state: GameState) -> torch.Tensor:
    factories = torch.zeros((MAX_FACTORIES + 1, len(COLOR_ORDER)), dtype=torch.float32)
    for i, factory in enumerate(state.supply.factories):
        for t in factory:
            factories[i, COLOR_TO_IDX[t]] += 1.0
    for t in state.supply.center:
        factories[len(state.supply.factories), COLOR_TO_IDX[t]] += 1.0
    return factories


def _pattern_features(player) -> tuple[torch.Tensor, torch.Tensor]:
    fills = torch.zeros((5, 1), dtype=torch.float32)
    colors = torch.zeros((5, len(COLOR_ORDER)), dtype=torch.float32)
    for i, line in enumerate(player.pattern_lines):
        capacity = PATTERN_LINE_SIZES[i]
        fills[i, 0] = len(line) / capacity
        if line:
            colors[i, COLOR_TO_IDX[line[0]]] = 1.0
    return fills, colors


def _wall_features(player) -> torch.Tensor:
    wall = torch.tensor(player.wall, dtype=torch.float32)
    return wall.view(5, 5)


def _tile_counts(tiles) -> torch.Tensor:
    counts = torch.zeros(len(COLOR_ORDER), dtype=torch.float32)
    for t in tiles:
        counts[COLOR_TO_IDX[t]] += 1.0
    return counts


def encode_state(state: GameState) -> torch.Tensor:
    """Encode game state into a flat feature vector (current-player perspective)."""
    num_players = len(state.players)
    if num_players != 2:
        raise ValueError("encoder currently supports 2-player Azul")
    self_idx = state.current_player
    opp_idx = 1 - self_idx
    player = state.players[self_idx]
    opponent = state.players[opp_idx]

    factories = _pad_factories(state).flatten()

    self_fills, self_colors = _pattern_features(player)
    opp_fills, opp_colors = _pattern_features(opponent)

    self_wall = _wall_features(player).flatten()
    opp_wall = _wall_features(opponent).flatten()

    def floor_feats(p) -> torch.Tensor:
        hist = _tile_counts(p.floor_line)
        return torch.cat([torch.tensor([len(p.floor_line)], dtype=torch.float32), hist])

    self_floor = floor_feats(player)
    opp_floor = floor_feats(opponent)

    bag_counts = _tile_counts(state.supply.bag)
    discard_counts = _tile_counts(state.supply.discard)

    scores = torch.tensor(
        [player.score / 100.0, opponent.score / 100.0, state.round_number / 10.0], dtype=torch.float32
    )

    feats = torch.cat(
        [
            factories,
            self_fills.flatten(),
            self_colors.flatten(),
            opp_fills.flatten(),
            opp_colors.flatten(),
            self_wall,
            opp_wall,
            self_floor,
            opp_floor,
            bag_counts,
            discard_counts,
            scores,
        ]
    )
    return feats
