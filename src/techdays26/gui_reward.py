"""Reward sparsity visualizer — see how little signal the agent gets.

Usage::

    from techdays26.gui_reward import RewardTimelineVisualizer

    vis = RewardTimelineVisualizer()
    vis.show()

    # With value function overlay:
    vis = RewardTimelineVisualizer(model_path="../td_weights_clean.tdw.zip")
    vis.show()
"""

from __future__ import annotations

import math

import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output, Layout, HTML

from techdays26.gui_common import draw_board, COLS, COL_STRIDE
from techdays26.torch_board import BoardBatch


class RewardTimelineVisualizer:
    """Step through a random game and watch the reward timeline."""

    def __init__(self, model_path: str = "") -> None:
        self._model = None
        if model_path:
            from techdays26.ntuple_network import NTupleNetwork

            self._model = NTupleNetwork.load(model_path, device="cpu")
            self._model.eval()

        self._pre_play_game()

        self._step = 0
        self._btn_next = Button(description="▶ Next", button_style="primary")
        self._btn_next.on_click(self._on_next)
        self._btn_all = Button(description="▶▶ Play All", button_style="info")
        self._btn_all.on_click(self._on_all)
        self._btn_reset = Button(description="↺ New Game", button_style="warning")
        self._btn_reset.on_click(self._on_reset)
        self._out = Output()

        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _pre_play_game(self) -> None:
        """Play a full random game and record the trajectory."""
        import random

        board = BoardBatch.empty(1, device="cpu")
        self._traj: list[dict] = []

        # Record initial state
        self._traj.append({
            "at": int(board.all_tokens[0]),
            "act": int(board.active_tokens[0]),
            "ml": int(board.moves_left[0]),
            "reward": float("nan"),
            "v": 0.0,
            "col": -1,
        })

        while not bool(board.done()[0]):
            legal_int = int(board.generate_legal_moves()[0])
            cols = [c for c in range(COLS) if (legal_int >> (c * COL_STRIDE)) & 0x3F]
            col = random.choice(cols)
            board.play_columns(torch.tensor([col]))

            r = float(board.reward()[0])
            v = 0.0
            if self._model is not None:
                with torch.no_grad():
                    v = float(self._model(board)[0])

            self._traj.append({
                "at": int(board.all_tokens[0]),
                "act": int(board.active_tokens[0]),
                "ml": int(board.moves_left[0]),
                "reward": r,
                "v": v,
                "col": col,
            })

    # ──────────────────────────────────────────────────────────────
    def _on_next(self, _) -> None:
        if self._step < len(self._traj) - 1:
            self._step += 1
            self._redraw()

    def _on_all(self, _) -> None:
        self._step = len(self._traj) - 1
        self._redraw()

    def _on_reset(self, _) -> None:
        self._pre_play_game()
        self._step = 0
        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _redraw(self) -> None:
        t = self._step
        s = self._traj[t]
        n_moves = len(self._traj) - 1  # total moves in game

        self._out.clear_output(wait=True)
        with self._out:
            n_plots = 3 if self._model else 2
            fig, axes = plt.subplots(
                1,
                n_plots,
                figsize=(4 * n_plots, 3.6),
                gridspec_kw={"width_ratios": [1.2] + [1.8] * (n_plots - 1)},
            )
            fig.patch.set_facecolor("#f8f9fa")

            # ── Board ──
            ax_board = axes[0]
            draw_board(
                ax_board,
                s["at"],
                s["act"],
                s["ml"],
                cell_radius=0.38,
            )
            player = "Yellow" if (s["ml"] % 2) == 0 else "Red"
            if t == 0:
                ax_board.set_title("Empty board", fontsize=9)
            elif t == n_moves and not math.isnan(s["reward"]):
                outcome = {1.0: "Yellow wins!", -1.0: "Red wins!", 0.0: "Draw!"}
                ax_board.set_title(
                    f"Move {t}: {outcome.get(s['reward'], '?')}",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                ax_board.set_title(f"Move {t}/{n_moves} ({player} to move)", fontsize=9)

            # ── Reward timeline ──
            ax_rew = axes[1]
            steps_so_far = range(1, t + 1)
            rewards = []
            bar_colors = []
            for i in steps_so_far:
                r = self._traj[i]["reward"]
                if math.isnan(r):
                    rewards.append(0.0)
                    bar_colors.append("#bdc3c7")
                elif r > 0:
                    rewards.append(r)
                    bar_colors.append("#2ecc71")
                elif r < 0:
                    rewards.append(r)
                    bar_colors.append("#e74c3c")
                else:
                    rewards.append(0.0)
                    bar_colors.append("#f39c12")

            if rewards:
                ax_rew.bar(list(steps_so_far), rewards, color=bar_colors, width=0.8)
            ax_rew.set_xlim(0.5, max(n_moves, 1) + 0.5)
            ax_rew.set_ylim(-1.3, 1.3)
            ax_rew.axhline(0, color="#7f8c8d", lw=0.5)
            ax_rew.set_xlabel("Move")
            ax_rew.set_ylabel("Reward")
            ax_rew.set_title("Reward signal (sparse!)", fontsize=9)

            # Annotation for sparsity
            if t == n_moves and n_moves > 5:
                ax_rew.annotate(
                    f"{n_moves - 1} moves\nwith r = 0",
                    xy=(n_moves / 2, 0),
                    xytext=(n_moves / 2, 0.7),
                    fontsize=8,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
                    color="#7f8c8d",
                )

            # ── V(afterstate) overlay ──
            if self._model and n_plots == 3:
                ax_v = axes[2]
                v_vals = [self._traj[i]["v"] for i in range(1, t + 1)]
                if v_vals:
                    x = list(range(1, t + 1))
                    # Color by who just moved
                    colors = []
                    for i in range(1, t + 1):
                        ml = self._traj[i]["ml"]
                        yellow_just = (ml % 2) == 1
                        colors.append("#f1c40f" if yellow_just else "#e74c3c")
                    ax_v.bar(x, v_vals, color=colors, width=0.8, alpha=0.8)
                ax_v.set_xlim(0.5, max(n_moves, 1) + 0.5)
                ax_v.set_ylim(-1.3, 1.3)
                ax_v.axhline(0, color="#7f8c8d", lw=0.5)
                ax_v.set_xlabel("Move")
                ax_v.set_ylabel("V(afterstate)")
                ax_v.set_title("Learned value function (smooth!)", fontsize=9)

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        label = (
            "<b>Reward Timeline</b> — step through a random game. "
            "Notice: the reward is <b>0 for every move except the last</b>. "
        )
        if self._model:
            label += (
                "The learned value function V(s') provides a <b>smooth signal</b> "
                "at every step — this is what makes learning possible."
            )
        display(
            VBox([
                HTML(label),
                HBox([self._btn_next, self._btn_all, self._btn_reset]),
                self._out,
            ])
        )
