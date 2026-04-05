"""Batched simulation visualizer — watch many games run in parallel.

Usage::

    from techdays26.gui_batch import BatchSimulationVisualizer

    vis = BatchSimulationVisualizer(n_boards=8)
    vis.show()
"""

from __future__ import annotations

import torch
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output, Layout, HTML

from techdays26.gui_common import draw_board, snap, random_legal_col, COLS
from techdays26.torch_board import BoardBatch


class BatchSimulationVisualizer:
    """Click Step to advance all games by one random move."""

    def __init__(self, n_boards: int = 8) -> None:
        self._B = n_boards
        self._board = BoardBatch.empty(n_boards, device="cpu")
        self._step = 0
        self._total_wins = 0
        self._total_draws = 0

        self._btn_step = Button(description="▶ Step", button_style="primary")
        self._btn_step.on_click(self._on_step)
        self._btn_reset = Button(description="↺ Reset All", button_style="warning")
        self._btn_reset.on_click(self._on_reset)
        self._html = HTML()
        self._out = Output()

        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _on_step(self, _) -> None:
        # Reset finished games from previous step
        done = self._board.done()
        self._total_wins += int(self._board.has_win().sum())
        self._total_draws += int((done & ~self._board.has_win()).sum())
        self._board.reset(done)

        # Play one random legal move per board
        cols = torch.zeros(self._B, dtype=torch.long)
        for i in range(self._B):
            c = random_legal_col(self._board, i)
            if c is not None:
                cols[i] = c
        self._board.play_columns(cols)
        self._step += 1
        self._redraw()

    def _on_reset(self, _) -> None:
        self._board = BoardBatch.empty(self._B, device="cpu")
        self._step = 0
        self._total_wins = 0
        self._total_draws = 0
        self._redraw()

    def _redraw(self) -> None:
        done = self._board.done()
        wins = self._board.has_win()

        self._html.value = (
            f"<b>Step {self._step}</b> &nbsp;|&nbsp; "
            f"{self._B} boards &nbsp;|&nbsp; "
            f"<span style='color:#2ecc71'>{int(wins.sum())} wins now</span> &nbsp;|&nbsp; "
            f"Total: {self._total_wins} wins, {self._total_draws} draws"
        )

        ncols = min(self._B, 4)
        nrows = (self._B + ncols - 1) // ncols

        self._out.clear_output(wait=True)
        with self._out:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 2.8, nrows * 2.4),
                squeeze=False,
            )
            fig.patch.set_facecolor("#f8f9fa")

            for i in range(self._B):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                at, act, ml = snap(self._board, i)
                draw_board(ax, at, act, ml, cell_radius=0.38, show_col_labels=False)

                is_done = bool(done[i])
                is_win = bool(wins[i])
                if is_done:
                    border = "#2ecc71" if is_win else "#95a5a6"
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_color(border)
                        sp.set_linewidth(3)
                    label = "WIN" if is_win else "DRAW"
                    ax.set_title(
                        f"Game {i} — {label}",
                        fontsize=8,
                        color=border,
                        fontweight="bold",
                    )
                else:
                    moves = 42 - ml
                    ax.set_title(f"Game {i}  ({moves} moves)", fontsize=8)

            for i in range(self._B, nrows * ncols):
                r, c = divmod(i, ncols)
                axes[r][c].set_visible(False)

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>Batched Simulation</b> — click <i>Step</i> to advance "
                    "all games by one random move. Finished games auto-reset."
                ),
                HBox([self._btn_step, self._btn_reset]),
                self._html,
                self._out,
            ])
        )
