"""Shared rendering helpers for Connect-4 workshop widgets."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

COLS = 7
ROWS = 6
COL_STRIDE = 9

C_BOARD_BG = "#2c5aa0"
C_YELLOW = "#f1c40f"
C_RED = "#e74c3c"
C_EMPTY_CELL = "#ffffff"
C_GRID_LINE = "#1a1a2e"


def player_tokens(all_tokens: int, active_tokens: int, moves_left: int):
    """Return (yellow_bb, red_bb) bitboards regardless of whose turn it is."""
    yellow_active = (moves_left % 2) == 0
    opponent = all_tokens ^ active_tokens
    if yellow_active:
        return active_tokens, opponent
    return opponent, active_tokens


def draw_board(
    ax: plt.Axes,
    all_tokens: int,
    active_tokens: int,
    moves_left: int,
    *,
    cell_radius: float = 0.4,
    show_col_labels: bool = True,
    highlight_col: int | None = None,
    highlight_color: str = "#2ecc71",
) -> None:
    """Draw a Connect-4 board state on a matplotlib Axes."""
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(C_BOARD_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    yellow_bb, red_bb = player_tokens(all_tokens, active_tokens, moves_left)

    for col in range(COLS):
        for row in range(ROWS):
            bit = col * COL_STRIDE + row
            if (yellow_bb >> bit) & 1:
                fc = C_YELLOW
            elif (red_bb >> bit) & 1:
                fc = C_RED
            else:
                fc = C_EMPTY_CELL

            ec = C_GRID_LINE
            lw = 0.5
            if highlight_col is not None and col == highlight_col:
                ec = highlight_color
                lw = 2.5
            ax.add_patch(mpatches.Circle((col, row), cell_radius, fc=fc, ec=ec, lw=lw))

    if show_col_labels:
        for c in range(COLS):
            ax.text(c, -0.45, str(c), ha="center", va="top", fontsize=7, color="white")


def snap(board, idx: int = 0) -> tuple[int, int, int]:
    """Extract (all_tokens, active_tokens, moves_left) for board[idx]."""
    return (
        int(board.all_tokens[idx]),
        int(board.active_tokens[idx]),
        int(board.moves_left[idx]),
    )


def random_legal_col(board, idx: int = 0) -> int | None:
    """Pick a uniformly random legal column for board[idx], or None if none."""
    import random

    legal_int = int(board.generate_legal_moves()[idx])
    cols = [c for c in range(COLS) if (legal_int >> (c * COL_STRIDE)) & 0x3F]
    return random.choice(cols) if cols else None


def legal_columns(board, idx: int = 0) -> list[int]:
    """Return list of legal column indices for board[idx]."""
    legal_int = int(board.generate_legal_moves()[idx])
    return [c for c in range(COLS) if (legal_int >> (c * COL_STRIDE)) & 0x3F]
