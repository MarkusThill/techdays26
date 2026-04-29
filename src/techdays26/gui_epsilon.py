"""ε-Greedy explorer — watch the agent decide: exploit or explore?

Usage::

    from techdays26.gui_epsilon import EpsilonGreedyVisualizer

    vis = EpsilonGreedyVisualizer(model_path="../td_weights_clean.tdw.zip")
    vis.show()
"""

from __future__ import annotations

import random as pyrandom

import matplotlib.pyplot as plt
import torch
from IPython.display import display
from ipywidgets import HTML, Button, FloatSlider, HBox, Layout, Output, VBox

from techdays26.gui_common import (
    COLS,
    draw_board,
    legal_columns,
    snap,
)
from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


class EpsilonGreedyVisualizer:
    """Interactively step through agent moves with an ε slider."""

    def __init__(self, model_path: str = "") -> None:
        if not model_path:
            raise ValueError("model_path is required for EpsilonGreedyVisualizer")

        self._model = NTupleNetwork.load(model_path, device="cpu")
        self._model.eval()

        self._board = BoardBatch.empty(1, device="cpu")
        self._last_action: dict | None = None  # info about last move

        self._sl_eps = FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="ε:",
            continuous_update=False,
            style={"description_width": "20px"},
            layout=Layout(width="350px"),
        )
        self._btn_move = Button(description="▶ Agent Move", button_style="primary")
        self._btn_move.on_click(self._on_move)
        self._btn_reset = Button(description="↺ Reset", button_style="warning")
        self._btn_reset.on_click(self._on_reset)
        self._html = HTML()
        self._out = Output()

        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _evaluate_moves(self) -> list[tuple[int, float]]:
        """Return [(col, V(afterstate))] for each legal column."""
        results = []
        legal = legal_columns(self._board, 0)
        for col in legal:
            tmp = BoardBatch(
                all_tokens=self._board.all_tokens.clone(),
                active_tokens=self._board.active_tokens.clone(),
                moves_left=self._board.moves_left.clone(),
            )
            tmp.play_columns(torch.tensor([col]))

            r = float(tmp.reward()[0])
            if not (r != r):  # not NaN → terminal
                v = r
            else:
                with torch.no_grad():
                    v = float(self._model(tmp)[0])
            results.append((col, v))
        return results

    def _on_move(self, _) -> None:
        if bool(self._board.done()[0]):
            return

        eps = self._sl_eps.value
        evals = self._evaluate_moves()
        if not evals:
            return

        # Determine if Yellow (maximize) or Red (minimize)
        ml = int(self._board.moves_left[0])
        yellow = (ml % 2) == 0

        if yellow:
            best_col = max(evals, key=lambda x: x[1])[0]
        else:
            best_col = min(evals, key=lambda x: x[1])[0]

        explored = pyrandom.random() < eps
        if explored:
            chosen = pyrandom.choice(evals)[0]
        else:
            chosen = best_col

        self._last_action = {
            "evals": evals,
            "chosen": chosen,
            "explored": explored,
            "yellow": yellow,
        }

        self._board.play_columns(torch.tensor([chosen]))
        self._redraw()

    def _on_reset(self, _) -> None:
        self._board = BoardBatch.empty(1, device="cpu")
        self._last_action = None
        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _redraw(self) -> None:
        at, act, ml = snap(self._board, 0)
        done = bool(self._board.done()[0])

        # Status text
        if done:
            r = float(self._board.reward()[0])
            outcome = {1.0: "Yellow wins!", -1.0: "Red wins!", 0.0: "Draw!"}
            self._html.value = f"<b>{outcome.get(r, 'Game over')}</b>"
        elif self._last_action:
            la = self._last_action
            mode = (
                "<span style='color:#e67e22; font-weight:bold'>EXPLORE (random)</span>"
                if la["explored"]
                else "<span style='color:#2ecc71; font-weight:bold'>EXPLOIT (best)</span>"
            )
            player = "Yellow" if la["yellow"] else "Red"
            self._html.value = f"{player} played col {la['chosen']} — {mode} &nbsp; (ε = {self._sl_eps.value:.2f})"
        else:
            self._html.value = "Yellow to move — click <i>Agent Move</i>"

        # Evaluate current position (for bar chart)
        if not done:
            evals = self._evaluate_moves()
        else:
            evals = []

        self._out.clear_output(wait=True)
        with self._out:
            fig, axes = plt.subplots(
                1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.0, 1.4]}
            )
            fig.patch.set_facecolor("#f8f9fa")

            # ── Board ──
            highlight = self._last_action["chosen"] if self._last_action else None
            h_color = (
                "#e67e22"
                if self._last_action and self._last_action["explored"]
                else "#2ecc71"
            )
            draw_board(
                axes[0],
                at,
                act,
                ml,
                highlight_col=highlight,
                highlight_color=h_color,
            )
            moves = 42 - ml
            player = "Yellow" if (ml % 2) == 0 else "Red"
            if done:
                axes[0].set_title(f"Game over ({moves} moves)", fontsize=9)
            else:
                axes[0].set_title(f"Move {moves + 1} — {player} to move", fontsize=9)

            # ── Value bar chart ──
            ax_v = axes[1]
            if evals:
                cols_list = [c for c, _ in evals]
                vals = [v for _, v in evals]

                yellow_to_move = (ml % 2) == 0
                if yellow_to_move:
                    best_v = max(vals)
                else:
                    best_v = min(vals)

                bar_colors = []
                for c, v in evals:
                    if v == best_v:
                        bar_colors.append("#2ecc71")
                    else:
                        bar_colors.append("#3498db")

                ax_v.bar(cols_list, vals, color=bar_colors, width=0.6)
                ax_v.axhline(0, color="#7f8c8d", lw=0.5)

                for c, v in evals:
                    ax_v.text(
                        c,
                        v + 0.03 * (1 if v >= 0 else -1),
                        f"{v:+.3f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7,
                    )

                ax_v.set_xticks(range(COLS))
                ax_v.set_xticklabels([str(c) for c in range(COLS)])

            ax_v.set_ylim(-1.15, 1.15)
            ax_v.set_xlabel("Column")
            ax_v.set_ylabel("V(afterstate)")
            ax_v.set_title("Afterstate values (green = best)", fontsize=9)

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>ε-Greedy Explorer</b> — the agent evaluates each legal "
                    "move's afterstate. With probability ε it explores (random); "
                    "otherwise it exploits (best value)."
                ),
                HBox([self._sl_eps, self._btn_move, self._btn_reset]),
                self._html,
                self._out,
            ])
        )
