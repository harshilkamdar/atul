import matplotlib.pyplot as plt
from matplotlib import patches

from .enums import TileColor
from .state import GameState, WALL_PATTERN

COLOR_MAP = {
    TileColor.BLUE: "#4A90E2",
    TileColor.YELLOW: "#F5D547",
    TileColor.RED: "#D64045",
    TileColor.BLACK: "#2C2C2C",
    TileColor.WHITE: "#E5E5E5",
}


def plot_player_board(ax: plt.Axes, state: GameState, player_idx: int) -> None:
    player = state.players[player_idx]
    ax.set_title(f"Player {player_idx} (score {player.score})")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.invert_yaxis()
    ax.axis("off")

    for r, line in enumerate(player.pattern_lines):
        capacity = r + 1
        for c in range(capacity):
            x = 4 - c
            y = r
            color = COLOR_MAP.get(line[0], "#CCCCCC") if c < len(line) else "#FFFFFF"
            rect = patches.Rectangle((x, y), 0.9, 0.9, linewidth=1, edgecolor="gray", facecolor=color)
            ax.add_patch(rect)

    for r, row in enumerate(player.wall):
        for c, filled in enumerate(row):
            x = c + 0.1
            y = r
            base_color = COLOR_MAP[WALL_PATTERN[r][c]]
            face = base_color if filled else "#FFFFFF"
            rect = patches.Rectangle((x, y), 0.9, 0.9, linewidth=1, edgecolor="gray", facecolor=face)
            ax.add_patch(rect)

    for i, tile in enumerate(player.floor_line):
        rect = patches.Rectangle((i, 5), 0.9, 0.9, linewidth=1, edgecolor="gray", facecolor=COLOR_MAP[tile])
        ax.add_patch(rect)
    if player.has_first_player_token:
        ax.text(5.2, 5.5, "FP", fontsize=10, color="black")


def plot_state(state: GameState) -> plt.Figure:
    fig, axes = plt.subplots(1, len(state.players), figsize=(4 * len(state.players), 4))
    if len(state.players) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        plot_player_board(ax, state, idx)
    fig.suptitle(f"Round {state.round_number} | Phase: {state.phase.value}")
    fig.tight_layout()
    return fig


def plot_score_history(score_history: list[list[int]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    players = len(score_history[0])
    for p in range(players):
        ax.plot([s[p] for s in score_history], label=f"Player {p}")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
