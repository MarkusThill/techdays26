"""Interactive Bitboard Visualization Widget for BitBully Connect-4.

This module provides a Jupyter notebook widget that illustrates the internal
bitboard representation used by the BitBully Connect-4 engine. It visualizes:

- The 64-bit layout with 7 columns × 9 bits (6 playable + 3 sentinel)
- The two core bitboards: allTokens and activePTokens
- Derived boards: each player's tokens via XOR
- Legal move generation via carry-propagation trick
- Win detection via shift-AND operations
- Winning positions / threat maps

Usage::

    from gui_bitboard import BitboardVisualizer

    vis = BitboardVisualizer()
    vis.show()
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ipywidgets import (
    HTML,
    AppLayout,
    Button,
    Dropdown,
    HBox,
    Label,
    Layout,
    Output,
    Text,
    ToggleButton,
    ToggleButtons,
    VBox,
)

from bitbully import Board

# ──────────────────────────────────────────────────────────────────────
# Constants matching the C++ engine
# ──────────────────────────────────────────────────────────────────────
N_COLS = 7
N_ROWS = 6
COL_BIT_OFFSET = 9  # each column occupies 9 bits in the uint64

# Bit-index grid (row 5 = top, row 0 = bottom), same layout as Board.h
BIT_INDEX_GRID = np.zeros((N_ROWS + 3, N_COLS), dtype=int)  # +3 sentinel rows
for c in range(N_COLS):
    for r in range(N_ROWS + 3):
        BIT_INDEX_GRID[r, c] = c * COL_BIT_OFFSET + r

# Sentinel mask: bits that are not part of the playable 6×7 area
SENTINEL_BITS = set()
for c in range(N_COLS):
    for r in range(N_ROWS, N_ROWS + 3):
        SENTINEL_BITS.add(c * COL_BIT_OFFSET + r)
# Also bit 63 (unused)
SENTINEL_BITS.add(63)

BB_BOTTOM_ROW = 0
for c in range(N_COLS):
    BB_BOTTOM_ROW |= 1 << (c * COL_BIT_OFFSET)

BB_ALL_LEGAL = 0
for c in range(N_COLS):
    for r in range(N_ROWS):
        BB_ALL_LEGAL |= 1 << (c * COL_BIT_OFFSET + r)

# Direction shifts for win detection (matching Board.cpp hasWin)
WIN_DIRECTIONS = {
    "Vertical (↕)": 1,
    "Horizontal (↔)": COL_BIT_OFFSET,
    "Diagonal (⟋)": COL_BIT_OFFSET - 1,
    "Diagonal (⟍)": COL_BIT_OFFSET + 1,
}

# Colors
C_EMPTY = "#2c3e50"
C_YELLOW = "#f1c40f"
C_RED = "#e74c3c"
C_SENTINEL = "#1a1a2e"
C_BIT_ON = "#3498db"
C_BIT_OFF = "#ecf0f1"
C_LEGAL = "#2ecc71"
C_WIN = "#e74c3c"
C_THREAT = "#e67e22"
C_HIGHLIGHT = "#9b59b6"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _bit_set(bb: int, bit: int) -> bool:
    return bool(bb & (1 << bit))


def _bb_to_grid(bb: int, include_sentinels: bool = True) -> np.ndarray:
    """Convert a 64-bit bitboard integer to a 2D array matching the board layout.

    Returns shape (n_rows, n_cols) where n_rows = 9 if include_sentinels else 6.
    Row index 0 = bottom of column.
    """
    rows = N_ROWS + 3 if include_sentinels else N_ROWS
    grid = np.zeros((rows, N_COLS), dtype=int)
    for c in range(N_COLS):
        for r in range(rows):
            bit = c * COL_BIT_OFFSET + r
            grid[r, c] = 1 if (bb & (1 << bit)) else 0
    return grid


def _format_hex(val: int) -> str:
    return f"0x{val:016X}"


def _format_bin_grouped(val: int) -> str:
    """Format as 64-bit binary grouped by column (9 bits each) + 1 extra bit."""
    s = f"{val:064b}"
    # bit 63 is leftmost; group: [63], [62..54], [53..45], ... [8..0]
    groups = [s[0]]  # bit 63 (unused)
    for i in range(1, 64, 9):
        groups.append(s[i : i + 9])
    return " ".join(groups)


# ──────────────────────────────────────────────────────────────────────
# Main Visualizer Class
# ──────────────────────────────────────────────────────────────────────
class BitboardVisualizer:
    """Interactive bitboard visualization widget for Jupyter notebooks."""

    def __init__(self, init_moves: str = "") -> None:
        self._board = Board(init_moves) if init_moves else Board()
        self._move_history: str = init_moves

        # ── UI state ──
        self._selected_overlay = "None"

        # ── Build widgets ──
        self._build_widgets()
        self._refresh_all()

    # ──────────────────────────────────────────────────────────────
    # Widget construction
    # ──────────────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        # Main board output (matplotlib)
        self._out_board = Output(layout=Layout(width="420px", height="410px"))

        # Bitboard grid outputs
        self._out_bb_all = Output(layout=Layout(width="420px", height="410px"))
        self._out_bb_active = Output(layout=Layout(width="420px", height="410px"))
        self._out_overlay = Output(layout=Layout(width="420px", height="410px"))

        # Info panel
        self._html_info = HTML(layout=Layout(width="100%", padding="8px"))

        # Binary representations
        self._html_binary = HTML(layout=Layout(width="100%", padding="8px"))

        # Move input
        self._txt_moves = Text(
            value=self._move_history,
            placeholder="e.g. 3344250",
            description="Moves:",
            layout=Layout(width="300px"),
            continuous_update=False,
        )
        self._txt_moves.observe(self._on_moves_change, names="value")

        # Column buttons for interactive play
        col_buttons = []
        for c in range(N_COLS):
            btn = Button(
                description=f"⏬{c}",
                layout=Layout(width="50px", height="32px"),
                button_style="primary",
            )
            btn.on_click(lambda _, col=c: self._on_col_click(col))
            col_buttons.append(btn)
        self._hbox_cols = HBox(col_buttons)

        # Control buttons
        self._btn_undo = Button(
            description="↩ Undo", button_style="warning", layout=Layout(width="80px")
        )
        self._btn_undo.on_click(self._on_undo)
        self._btn_reset = Button(
            description="🔄 Reset", button_style="danger", layout=Layout(width="80px")
        )
        self._btn_reset.on_click(self._on_reset)

        # Overlay selector
        self._dd_overlay = Dropdown(
            options=[
                "None",
                "Legal Moves",
                "Winning Positions (current player)",
                "Win Check: Vertical (↕)",
                "Win Check: Horizontal (↔)",
                "Win Check: Diagonal (⟋)",
                "Win Check: Diagonal (⟍)",
                "Bit Index Map",
            ],
            value="None",
            description="Overlay:",
            layout=Layout(width="380px"),
        )
        self._dd_overlay.observe(self._on_overlay_change, names="value")

        # ── Explanation panel ──
        self._html_explain = HTML(layout=Layout(width="100%", padding="8px"))

    def show(self) -> None:
        """Display the widget in the notebook."""
        top_controls = HBox(
            [self._txt_moves, self._btn_undo, self._btn_reset],
            layout=Layout(align_items="center", gap="8px"),
        )

        board_label = HTML("<b>Board State</b>")
        all_label = HTML("<b>allTokens</b> (all pieces)")
        active_label = HTML("<b>activePTokens</b> (current player)")
        overlay_label = HTML("<b>Overlay / Derived</b>")

        grids = HBox(
            [
                VBox([board_label, self._out_board]),
                VBox([all_label, self._out_bb_all]),
                VBox([active_label, self._out_bb_active]),
                VBox([overlay_label, self._out_overlay]),
            ],
            layout=Layout(gap="4px"),
        )

        overlay_row = HBox(
            [self._dd_overlay],
            layout=Layout(align_items="center", gap="8px"),
        )

        layout = VBox(
            [
                HTML("<h2>🔬 BitBully — Bitboard Internals Visualizer</h2>"),
                top_controls,
                self._hbox_cols,
                grids,
                overlay_row,
                self._html_info,
                self._html_binary,
                self._html_explain,
            ],
            layout=Layout(padding="10px"),
        )
        display(layout)

    # ──────────────────────────────────────────────────────────────
    # Event handlers
    # ──────────────────────────────────────────────────────────────
    def _on_col_click(self, col: int) -> None:
        if self._board.is_game_over():
            return
        if self._board.native.isLegalMove(col):
            self._board.play(col)
            self._move_history += str(col)
            self._txt_moves.value = self._move_history
            self._refresh_all()

    def _on_undo(self, _=None) -> None:
        if self._move_history:
            self._move_history = self._move_history[:-1]
            self._board = Board(self._move_history) if self._move_history else Board()
            self._txt_moves.value = self._move_history
            self._refresh_all()

    def _on_reset(self, _=None) -> None:
        self._move_history = ""
        self._board = Board()
        self._txt_moves.value = ""
        self._refresh_all()

    def _on_moves_change(self, change) -> None:
        new_val = change["new"].strip()
        try:
            b = Board(new_val) if new_val else Board()
            self._board = b
            self._move_history = new_val
            self._refresh_all()
        except Exception:
            pass  # invalid sequence, ignore

    def _on_overlay_change(self, change) -> None:
        self._selected_overlay = change["new"]
        self._refresh_all()

    # ──────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────
    def _refresh_all(self) -> None:
        raw = self._board.native.rawState()
        all_tokens = raw[0]
        active_tokens = raw[1]
        moves_left = raw[2]

        self._draw_board()
        self._draw_bitboard_grid(self._out_bb_all, all_tokens, "allTokens", C_BIT_ON)
        self._draw_bitboard_grid(
            self._out_bb_active, active_tokens, "activePTokens", C_HIGHLIGHT
        )
        self._draw_overlay(all_tokens, active_tokens)
        self._update_info(all_tokens, active_tokens, moves_left)
        self._update_binary(all_tokens, active_tokens)
        self._update_explanation()

    def _draw_board(self) -> None:
        """Draw the Connect-4 board state."""
        self._out_board.clear_output(wait=True)
        with self._out_board:
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.8))
            fig.patch.set_facecolor("#34495e")
            ax.set_facecolor("#2c3e50")

            arr = self._board.to_array()  # 7×6 column-major
            for c in range(N_COLS):
                for r in range(N_ROWS):
                    val = arr[c][r]
                    if val == 1:
                        color = C_YELLOW
                    elif val == 2:
                        color = C_RED
                    else:
                        color = "#3d566e"
                    circle = plt.Circle(
                        (c + 0.5, r + 0.5),
                        0.38,
                        color=color,
                        ec="white",
                        linewidth=0.5,
                    )
                    ax.add_patch(circle)

            ax.set_xlim(0, N_COLS)
            ax.set_ylim(0, N_ROWS)
            ax.set_xticks(np.arange(N_COLS) + 0.5)
            ax.set_xticklabels(
                [str(i) for i in range(N_COLS)], color="white", fontsize=9
            )
            ax.set_yticks(np.arange(N_ROWS) + 0.5)
            # Row labels: bottom row (R0) at bottom, top row (R5) at top
            ax.set_yticklabels(
                [str(i) for i in range(N_ROWS)], color="white", fontsize=9
            )
            ax.tick_params(length=0)
            ax.set_aspect("equal")
            fig.tight_layout(pad=0.5)
            display(fig)
            plt.close(fig)

    def _draw_bitboard_grid(
        self,
        out: Output,
        bb: int,
        title: str,
        on_color: str,
        highlight_mask: int = 0,
        highlight_color: str = C_WIN,
        show_sentinels: bool = True,
    ) -> None:
        """Draw a bitboard as a grid of cells, 9 rows × 7 cols (including sentinel rows)."""
        out.clear_output(wait=True)
        with out:
            total_rows = N_ROWS + 3 if show_sentinels else N_ROWS
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.8))
            fig.patch.set_facecolor("#f8f9fa")
            ax.set_facecolor("#ffffff")

            for c in range(N_COLS):
                for r in range(total_rows):
                    bit = c * COL_BIT_OFFSET + r
                    is_sentinel = bit in SENTINEL_BITS
                    is_on = _bit_set(bb, bit)
                    is_highlight = _bit_set(highlight_mask, bit)

                    # r=0 is bottom row, place at y=0; sentinel rows at top
                    dy = r

                    if is_highlight:
                        fc = highlight_color
                    elif is_sentinel:
                        fc = C_SENTINEL
                    elif is_on:
                        fc = on_color
                    else:
                        fc = C_BIT_OFF

                    rect = mpatches.FancyBboxPatch(
                        (c + 0.05, dy + 0.05),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0.05",
                        facecolor=fc,
                        edgecolor="#bdc3c7",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

                    # Label: bit value
                    txt_color = (
                        "white" if (is_on or is_sentinel or is_highlight) else "#2c3e50"
                    )
                    ax.text(
                        c + 0.5,
                        dy + 0.5,
                        "1" if is_on else ("×" if is_sentinel else "0"),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=txt_color,
                        fontfamily="monospace",
                    )
                    # Subscript: bit index
                    ax.text(
                        c + 0.85,
                        dy + 0.15,
                        str(bit),
                        ha="right",
                        va="bottom",
                        fontsize=5,
                        color=txt_color,
                        alpha=0.7,
                        fontfamily="monospace",
                    )

            ax.set_xlim(0, N_COLS)
            ax.set_ylim(0, total_rows)
            ax.set_xticks(np.arange(N_COLS) + 0.5)
            ax.set_xticklabels([f"C{i}" for i in range(N_COLS)], fontsize=8)
            ax.set_yticks(np.arange(total_rows) + 0.5)
            row_labels = (
                ([f"R{r}" for r in range(N_ROWS)] + ["S0", "S1", "S2"])
                if show_sentinels
                else [f"R{r}" for r in range(N_ROWS)]
            )
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.tick_params(length=0)
            ax.set_aspect("equal")

            # Draw separator line between playable and sentinel rows
            if show_sentinels:
                sep_y = N_ROWS + 0.02  # sentinel rows start at index 6
                ax.axhline(
                    y=sep_y, color="#e74c3c", linewidth=2, linestyle="--", alpha=0.7
                )
                ax.text(
                    N_COLS / 2,
                    sep_y + 0.15,
                    "▲ sentinel bits (illegal)",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#e74c3c",
                    fontstyle="italic",
                )

            fig.tight_layout(pad=0.5)
            display(fig)
            plt.close(fig)

    def _draw_overlay(self, all_tokens: int, active_tokens: int) -> None:
        """Draw the selected overlay visualization."""
        overlay = self._selected_overlay

        if overlay == "None":
            # Show opponent's tokens (XOR trick)
            opponent = active_tokens ^ all_tokens
            self._draw_bitboard_grid(
                self._out_overlay,
                opponent,
                "Opponent Tokens",
                "#e67e22",
                show_sentinels=True,
            )
            return

        if overlay == "Legal Moves":
            legal = (all_tokens + BB_BOTTOM_ROW) & BB_ALL_LEGAL
            self._draw_bitboard_grid(
                self._out_overlay,
                0,
                "Legal Moves",
                C_BIT_ON,
                highlight_mask=legal,
                highlight_color=C_LEGAL,
                show_sentinels=True,
            )
            return

        if overlay == "Winning Positions (current player)":
            # Use the engine to compute
            # winningPositions for current player
            active = active_tokens
            wp = self._compute_winning_positions(active)
            # Filter to legal
            legal = (all_tokens + BB_BOTTOM_ROW) & BB_ALL_LEGAL
            threats = wp & legal & ~all_tokens
            self._draw_bitboard_grid(
                self._out_overlay,
                wp & BB_ALL_LEGAL,
                "Winning Positions",
                C_BIT_ON,
                highlight_mask=threats,
                highlight_color=C_THREAT,
                show_sentinels=True,
            )
            return

        if overlay == "Bit Index Map":
            self._draw_bit_index_map()
            return

        # Win check directions
        for dir_name, shift in WIN_DIRECTIONS.items():
            if overlay == f"Win Check: {dir_name}":
                opponent = active_tokens ^ all_tokens
                self._draw_win_check(opponent, shift, dir_name)
                return

    def _draw_bit_index_map(self) -> None:
        """Draw a grid showing the bit index assigned to each cell."""
        self._out_overlay.clear_output(wait=True)
        with self._out_overlay:
            total_rows = N_ROWS + 3
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.8))
            fig.patch.set_facecolor("#f8f9fa")
            ax.set_facecolor("#ffffff")

            for c in range(N_COLS):
                for r in range(total_rows):
                    bit = c * COL_BIT_OFFSET + r
                    is_sentinel = bit in SENTINEL_BITS
                    dy = r

                    # Color by column
                    col_colors = [
                        "#3498db",
                        "#2ecc71",
                        "#e74c3c",
                        "#f39c12",
                        "#9b59b6",
                        "#1abc9c",
                        "#e67e22",
                    ]
                    fc = C_SENTINEL if is_sentinel else col_colors[c]

                    rect = mpatches.FancyBboxPatch(
                        (c + 0.05, dy + 0.05),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0.05",
                        facecolor=fc,
                        edgecolor="white",
                        linewidth=0.5,
                        alpha=0.85 if not is_sentinel else 0.4,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        c + 0.5,
                        dy + 0.5,
                        str(bit),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                        fontfamily="monospace",
                    )

            ax.set_xlim(0, N_COLS)
            ax.set_ylim(0, total_rows)
            ax.set_xticks(np.arange(N_COLS) + 0.5)
            ax.set_xticklabels([f"C{i}" for i in range(N_COLS)], fontsize=8)
            ax.set_yticks(np.arange(total_rows) + 0.5)
            row_labels = [f"R{r}" for r in range(N_ROWS)] + ["S0", "S1", "S2"]
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.tick_params(length=0)
            ax.set_aspect("equal")

            sep_y = N_ROWS + 0.02
            ax.axhline(y=sep_y, color="#e74c3c", linewidth=2, linestyle="--", alpha=0.7)

            fig.tight_layout(pad=0.5)
            display(fig)
            plt.close(fig)

    def _draw_win_check(self, player_bb: int, shift: int, direction: str) -> None:
        """Illustrate the shift-AND win-detection steps for a given direction."""
        self._out_overlay.clear_output(wait=True)
        with self._out_overlay:
            # Step 1: y & (y << 2*shift)
            step1 = player_bb & (player_bb << (2 * shift))
            # Step 2: step1 & (step1 << shift) => non-zero means win
            step2 = step1 & (step1 << shift)
            has_win = (step2 & BB_ALL_LEGAL) != 0

            # Show step2 as highlight on player_bb
            self._out_overlay.clear_output(wait=True)
            with self._out_overlay:
                total_rows = N_ROWS + 3
                fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.8))
                fig.patch.set_facecolor("#f8f9fa")
                ax.set_facecolor("#ffffff")

                for c in range(N_COLS):
                    for r in range(total_rows):
                        bit = c * COL_BIT_OFFSET + r
                        is_sentinel = bit in SENTINEL_BITS
                        dy = r

                        is_player = _bit_set(player_bb, bit)
                        is_step1 = _bit_set(step1, bit) and not is_sentinel
                        is_step2 = _bit_set(step2, bit) and not is_sentinel

                        if is_step2:
                            fc = C_WIN
                        elif is_step1:
                            fc = C_THREAT
                        elif is_player and not is_sentinel:
                            fc = C_BIT_ON
                        elif is_sentinel:
                            fc = C_SENTINEL
                        else:
                            fc = C_BIT_OFF

                        rect = mpatches.FancyBboxPatch(
                            (c + 0.05, dy + 0.05),
                            0.9,
                            0.9,
                            boxstyle="round,pad=0.05",
                            facecolor=fc,
                            edgecolor="#bdc3c7",
                            linewidth=0.5,
                        )
                        ax.add_patch(rect)

                        label = "1" if is_player else ("×" if is_sentinel else "0")
                        txt_c = (
                            "white"
                            if (is_player or is_sentinel or is_step1 or is_step2)
                            else "#2c3e50"
                        )
                        ax.text(
                            c + 0.5,
                            dy + 0.5,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=txt_c,
                            fontfamily="monospace",
                        )

                ax.set_xlim(0, N_COLS)
                ax.set_ylim(0, total_rows)
                ax.set_xticks(np.arange(N_COLS) + 0.5)
                ax.set_xticklabels([f"C{i}" for i in range(N_COLS)], fontsize=8)
                ax.set_yticks(np.arange(total_rows) + 0.5)
                row_labels = [f"R{r}" for r in range(N_ROWS)] + ["S0", "S1", "S2"]
                ax.set_yticklabels(row_labels, fontsize=7)
                ax.tick_params(length=0)
                ax.set_aspect("equal")

                # Legend
                legend_elements = [
                    mpatches.Patch(facecolor=C_BIT_ON, label="Player bits"),
                    mpatches.Patch(facecolor=C_THREAT, label="y & (y≪2d) [pairs]"),
                    mpatches.Patch(
                        facecolor=C_WIN, label="pairs & (pairs≪d) [4-in-row!]"
                    ),
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper center",
                    fontsize=6,
                    ncol=1,
                    framealpha=0.9,
                    bbox_to_anchor=(0.5, -0.02),
                )

                win_text = f"{'WIN DETECTED' if has_win else '— no win'} ({direction}, d={shift})"
                ax.set_title(
                    win_text,
                    fontsize=8,
                    fontweight="bold",
                    color=C_WIN if has_win else "#2c3e50",
                )

                fig.tight_layout(pad=0.5)
                display(fig)
                plt.close(fig)

    def _compute_winning_positions(self, player_bb: int) -> int:
        """Pure-Python reimplementation of Board::winningPositions for visualization."""
        wins = (player_bb << 1) & (player_bb << 2) & (player_bb << 3)  # vertical

        for b in [COL_BIT_OFFSET - 1, COL_BIT_OFFSET, COL_BIT_OFFSET + 1]:
            tmp = (player_bb << b) & (player_bb << (2 * b))
            wins |= tmp & (player_bb << (3 * b))
            wins |= tmp & (player_bb >> b)

            tmp = (player_bb >> b) & (player_bb >> (2 * b))
            wins |= tmp & (player_bb << b)
            wins |= tmp & (player_bb >> (3 * b))

        return wins & BB_ALL_LEGAL

    # ──────────────────────────────────────────────────────────────
    # Info panels
    # ──────────────────────────────────────────────────────────────
    def _update_info(
        self, all_tokens: int, active_tokens: int, moves_left: int
    ) -> None:
        opponent_tokens = active_tokens ^ all_tokens
        n_tokens = 42 - moves_left
        current_player = 1 if n_tokens % 2 == 0 else 2
        player_name = "Yellow (X)" if current_player == 1 else "Red (O)"
        legal_mask = (all_tokens + BB_BOTTOM_ROW) & BB_ALL_LEGAL
        legal_cols = [
            c
            for c in range(N_COLS)
            if legal_mask
            & (1 << (c * COL_BIT_OFFSET + self._board.native.getColumnHeight(c)))
        ]

        has_win = self._board.has_win()
        can_win = self._board.native.canWin()
        game_over = self._board.is_game_over()

        status = ""
        if game_over:
            w = self._board.winner()
            if w == 1:
                status = "🏆 <b>Yellow (X) wins!</b>"
            elif w == 2:
                status = "🏆 <b>Red (O) wins!</b>"
            else:
                status = "🤝 <b>Draw!</b>"
        else:
            status = f"Current player: <b>{player_name}</b>"

        uid = self._board.uid()

        self._html_info.value = f"""
        <div style="font-family: monospace; font-size: 13px; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6;">
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td style="padding: 2px 12px;"><b>Status:</b></td><td>{status}</td>
                    <td style="padding: 2px 12px;"><b>Tokens placed:</b></td><td>{n_tokens} / 42</td>
                    <td style="padding: 2px 12px;"><b>Moves left:</b></td><td>{moves_left}</td></tr>
                <tr><td style="padding: 2px 12px;"><b>Has win (prev player):</b></td><td>{"✅ Yes" if has_win else "❌ No"}</td>
                    <td style="padding: 2px 12px;"><b>Can win next:</b></td><td>{"✅ Yes" if can_win else "❌ No"}</td>
                    <td style="padding: 2px 12px;"><b>Legal columns:</b></td><td>{self._board.legal_moves()}</td></tr>
                <tr><td style="padding: 2px 12px;"><b>UID:</b></td><td>{uid}</td>
                    <td style="padding: 2px 12px;"><b>Hash:</b></td><td>{self._board.native.hash()}</td>
                    <td style="padding: 2px 12px;"><b>Move sequence:</b></td><td>{self._move_history or "(empty)"}</td></tr>
            </table>
        </div>
        """

    def _update_binary(self, all_tokens: int, active_tokens: int) -> None:
        opponent = active_tokens ^ all_tokens
        legal = (all_tokens + BB_BOTTOM_ROW) & BB_ALL_LEGAL

        def _row(label, val, color="#2c3e50"):
            return f"""
            <tr>
                <td style="padding: 3px 8px; font-weight: bold; color: {color};">{label}</td>
                <td style="padding: 3px 8px;">{_format_hex(val)}</td>
                <td style="padding: 3px 8px; font-size: 11px; letter-spacing: 1px;">{_format_bin_grouped(val)}</td>
            </tr>"""

        rows = "".join([
            _row("allTokens", all_tokens, C_BIT_ON),
            _row("activePTokens", active_tokens, C_HIGHLIGHT),
            _row("opponentTokens (active⊕all)", opponent, "#e67e22"),
            _row("BB_BOTTOM_ROW", BB_BOTTOM_ROW, "#95a5a6"),
            _row("allTokens + BB_BOTTOM_ROW", all_tokens + BB_BOTTOM_ROW, "#95a5a6"),
            _row("legalMovesMask", legal, C_LEGAL),
            _row("uid (active + all)", active_tokens + all_tokens, "#2c3e50"),
        ])

        self._html_binary.value = f"""
        <div style="font-family: monospace; font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; overflow-x: auto;">
            <table style="border-collapse: collapse;">
                <tr style="background: #e9ecef;">
                    <th style="padding: 3px 8px; text-align: left;">Name</th>
                    <th style="padding: 3px 8px; text-align: left;">Hex</th>
                    <th style="padding: 3px 8px; text-align: left;">Binary (bit63 | C6 | C5 | C4 | C3 | C2 | C1 | C0)</th>
                </tr>
                {rows}
            </table>
        </div>
        """

    def _update_explanation(self) -> None:
        overlay = self._selected_overlay
        explanations = {
            "None": """
                <b>Opponent Tokens</b> — derived via <code>activePTokens ⊕ allTokens</code> (XOR).<br>
                The engine stores only <i>two</i> bitboards. The active player's pieces are in <code>activePTokens</code>.
                The opponent's pieces are obtained by XOR with <code>allTokens</code>.
                After each move, <code>activePTokens ⊕= allTokens</code> swaps perspective — no branching needed.
            """,
            "Legal Moves": """
                <b>Legal Moves</b> — computed as <code>(allTokens + BB_BOTTOM_ROW) & BB_ALL_LEGAL</code>.<br>
                The key trick: adding <code>BB_BOTTOM_ROW</code> (one bit per column at row 0) to <code>allTokens</code>
                causes carry propagation <i>within each column</i>. The carry stops at the first empty cell — exactly the
                legal move position. Masking with <code>BB_ALL_LEGAL</code> removes overflow into sentinel bits.
                This is why each column uses 9 bits (6 + 3 sentinel): the 3 extra bits absorb carry from full columns.
            """,
            "Winning Positions (current player)": """
                <b>Winning Positions</b> — cells where placing a token completes a four-in-a-row.<br>
                Computed by checking all 4 directions (vertical, horizontal, 2 diagonals) using bit shifts.
                Orange highlighted cells are <i>threats</i>: winning positions that are also legal moves.
            """,
            "Bit Index Map": """
                <b>Bit Index Map</b> — shows which bit in the <code>uint64_t</code> corresponds to each cell.<br>
                Column <i>c</i> occupies bits <code>[c×9 .. c×9+5]</code> (rows 0–5) plus 3 sentinel bits
                <code>[c×9+6 .. c×9+8]</code>. Total: 63 bits used, bit 63 is unused.
                This non-contiguous layout enables the carry-propagation trick for legal move generation.
            """,
        }

        for dir_name in WIN_DIRECTIONS:
            shift = WIN_DIRECTIONS[dir_name]
            explanations[f"Win Check: {dir_name}"] = f"""
                <b>Win Detection — {dir_name}</b> (shift distance d={shift})<br>
                <b>Step 1:</b> <code>pairs = y & (y ≪ 2d)</code> — finds bits that have a matching bit
                two positions away in this direction (potential middle of 4-in-a-row).<br>
                <b>Step 2:</b> <code>result = pairs & (pairs ≪ d)</code> — checks if two such pairs overlap,
                confirming four consecutive bits. Non-zero ⇒ win found.<br>
                <span style="color: {C_BIT_ON};">■</span> player bits &nbsp;
                <span style="color: {C_THREAT};">■</span> pairs (step 1) &nbsp;
                <span style="color: {C_WIN};">■</span> four-in-a-row confirmed (step 2)
            """

        text = explanations.get(overlay, "")
        self._html_explain.value = f"""
        <div style="font-family: sans-serif; font-size: 13px; background: #fff3cd; padding: 12px;
                    border-radius: 8px; border: 1px solid #ffc107; line-height: 1.6;">
            💡 {text.strip()}
        </div>
        """
