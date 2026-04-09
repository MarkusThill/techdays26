"""Credit-assignment animation — watch a reward flow backward through a game.

Usage::

    from techdays26.gui_credit_assignment import CreditAssignmentVisualizer

    vis = CreditAssignmentVisualizer(model_path="../td_weights_clean.tdw.zip")
    vis.show()

The widget plays a full self-play game, then lets the user propagate the
terminal reward backward one TD(0) step at a time. Pedagogically: this is
what makes TD learning work at all — a single non-zero reward at the end
of the game becomes a learning signal for *every* move that led up to it.
"""

from __future__ import annotations

import math
import random as pyrandom

import torch
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import Button, HBox, VBox, Output, HTML, IntSlider, Layout

from techdays26.gui_common import draw_board, COLS, COL_STRIDE
from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


class CreditAssignmentVisualizer:
    """Show how a terminal reward propagates backward through a trajectory.

    Each "tape" cell in the V-trace starts at zero — i.e., the agent initially
    has no idea what any position is worth. Clicking "▶ Propagate 1 step"
    performs one backward TD(0) update: ``V[t] ← V[t+1]`` (or the reward at
    the terminal step). Watching the reward seep backward one step per click
    makes the credit-assignment problem — and why ``TD(λ)`` is such a big win
    over ``TD(0)`` — viscerally obvious.
    """

    def __init__(self, model_path: str = "") -> None:
        self._model: NTupleNetwork | None = None
        if model_path:
            self._model = NTupleNetwork.load(model_path, device="cpu")
            self._model.eval()

        self._pre_play_game()
        self._v_learned: list[float] = [0.0] * len(self._traj)
        self._front: int = len(self._traj) - 1  # next index to update (backward)
        # Seed: terminal step already knows its reward
        last_r = self._traj[-1]["reward"]
        if last_r == last_r:  # not NaN
            self._v_learned[-1] = last_r
        self._front = len(self._traj) - 2  # next index to update

        self._btn_step = Button(
            description="▶ Propagate 1 step",
            button_style="primary",
            layout=Layout(width="170px"),
        )
        self._btn_step.on_click(self._on_step)
        self._btn_all = Button(
            description="▶▶ Propagate all",
            button_style="info",
            layout=Layout(width="160px"),
        )
        self._btn_all.on_click(self._on_all)
        self._btn_reset = Button(
            description="↺ Reset",
            button_style="warning",
            layout=Layout(width="100px"),
        )
        self._btn_reset.on_click(self._on_reset)
        self._btn_new = Button(
            description="🎲 New game",
            button_style="",
            layout=Layout(width="120px"),
        )
        self._btn_new.on_click(self._on_new_game)

        self._out = Output()
        self._html = HTML()
        self._redraw()

    # ──────────────────────────────────────────────────────────────
    def _pre_play_game(self) -> None:
        """Play one self-play game and record the trajectory.

        If a model is loaded we play greedily; otherwise random moves.
        We keep trying until the game has a non-zero terminal reward, so the
        backward propagation is visually meaningful.
        """
        for _attempt in range(20):
            board = BoardBatch.empty(1, device="cpu")
            traj: list[dict] = []
            traj.append({
                "at": int(board.all_tokens[0]),
                "act": int(board.active_tokens[0]),
                "ml": int(board.moves_left[0]),
                "reward": float("nan"),
                "col": -1,
            })
            while not bool(board.done()[0]):
                legal_int = int(board.generate_legal_moves()[0])
                legal_cols = [
                    c for c in range(COLS) if (legal_int >> (c * COL_STRIDE)) & 0x3F
                ]
                if not legal_cols:
                    break
                # Mix: 60% greedy, 40% random so games vary between runs
                if self._model is not None and pyrandom.random() < 0.6:
                    yellow = (int(board.moves_left[0]) % 2) == 0
                    best_col = legal_cols[0]
                    best_v = float("-inf") if yellow else float("inf")
                    for c in legal_cols:
                        tmp = BoardBatch(
                            all_tokens=board.all_tokens.clone(),
                            active_tokens=board.active_tokens.clone(),
                            moves_left=board.moves_left.clone(),
                        )
                        tmp.play_columns(torch.tensor([c]))
                        with torch.no_grad():
                            v = float(self._model(tmp)[0])
                        if (yellow and v > best_v) or (not yellow and v < best_v):
                            best_v = v
                            best_col = c
                    col = best_col
                else:
                    col = pyrandom.choice(legal_cols)
                board.play_columns(torch.tensor([col]))
                r = float(board.reward()[0])
                traj.append({
                    "at": int(board.all_tokens[0]),
                    "act": int(board.active_tokens[0]),
                    "ml": int(board.moves_left[0]),
                    "reward": r,
                    "col": col,
                })
            last_r = traj[-1]["reward"]
            if last_r == last_r and last_r != 0.0:
                self._traj = traj
                return
        # Fall back to whatever game we last generated
        self._traj = traj

    # ──────────────────────────────────────────────────────────────
    def _on_step(self, _=None) -> None:
        if self._front < 0:
            return
        # TD(0) backward update: V[t] ← V[t+1]
        # (No intermediate rewards in Connect-4, so this is just copy-left.)
        self._v_learned[self._front] = self._v_learned[self._front + 1]
        self._front -= 1
        self._redraw()

    def _on_all(self, _=None) -> None:
        while self._front >= 0:
            self._v_learned[self._front] = self._v_learned[self._front + 1]
            self._front -= 1
        self._redraw()

    def _on_reset(self, _=None) -> None:
        self._v_learned = [0.0] * len(self._traj)
        last_r = self._traj[-1]["reward"]
        if last_r == last_r:
            self._v_learned[-1] = last_r
        self._front = len(self._traj) - 2
        self._redraw()

    def _on_new_game(self, _=None) -> None:
        self._pre_play_game()
        self._on_reset()

    # ──────────────────────────────────────────────────────────────
    def _redraw(self) -> None:
        n = len(self._traj) - 1  # number of half-moves
        last_r = self._traj[-1]["reward"]
        outcome_map = {1.0: "Yellow wins", -1.0: "Red wins", 0.0: "Draw"}
        outcome = outcome_map.get(last_r, "?")
        updated_steps = (n - 1) - self._front  # how many backward updates so far
        total_steps = n

        status = (
            f"<b>Game:</b> {n} half-moves &nbsp;&mdash;&nbsp; <b>Terminal reward:</b> "
            f"<span style='color:{'#2ecc71' if last_r > 0 else '#e74c3c' if last_r < 0 else '#7f8c8d'};"
            f" font-weight:bold'>r = {last_r:+.0f} ({outcome})</span><br>"
            f"<b>Backward updates applied:</b> {updated_steps} / {total_steps}"
        )
        if self._front < 0:
            status += " &nbsp;✅ <b>reward propagated all the way back to the opening move</b>"
        self._html.value = status

        self._out.clear_output(wait=True)
        with self._out:
            fig = plt.figure(figsize=(12, 4.2))
            fig.patch.set_facecolor("#f8f9fa")
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.3], wspace=0.22)

            # ── Terminal board (left) ─────────────────────────────
            ax_board = fig.add_subplot(gs[0, 0])
            last = self._traj[-1]
            draw_board(
                ax_board,
                last["at"],
                last["act"],
                last["ml"],
                cell_radius=0.38,
            )
            ax_board.set_title(
                f"Final position — {outcome}",
                fontsize=9,
                fontweight="bold",
            )

            # ── V-trace (right) ───────────────────────────────────
            ax_v = fig.add_subplot(gs[0, 1])
            steps = list(range(n + 1))
            bar_colors = []
            for i, v in enumerate(self._v_learned):
                if i > self._front:  # already updated
                    if v > 0:
                        bar_colors.append("#2ecc71")
                    elif v < 0:
                        bar_colors.append("#e74c3c")
                    else:
                        bar_colors.append("#f39c12")
                else:
                    bar_colors.append("#dfe4ea")  # not yet reached

            ax_v.bar(
                steps,
                self._v_learned,
                color=bar_colors,
                width=0.8,
                edgecolor="#7f8c8d",
                linewidth=0.5,
            )

            # Highlight the next step to be updated with an arrow
            if self._front >= 0:
                target_x = self._front
                target_v = self._v_learned[self._front + 1]
                ax_v.annotate(
                    "",
                    xy=(target_x, target_v * 0.9),
                    xytext=(target_x + 1, target_v * 0.9),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#2c3e50",
                        lw=2.0,
                    ),
                )
                ax_v.text(
                    target_x + 0.5,
                    target_v * 0.9 + 0.08 * (1 if target_v >= 0 else -1),
                    "V[t] ← V[t+1]",
                    ha="center",
                    va="bottom" if target_v >= 0 else "top",
                    fontsize=8,
                    color="#2c3e50",
                    fontweight="bold",
                )

            # Terminal reward marker
            if last_r == last_r:
                ax_v.plot(
                    n,
                    last_r,
                    marker="*",
                    markersize=16,
                    color="#f1c40f" if last_r > 0 else "#c0392b",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=5,
                )

            ax_v.axhline(0, color="#7f8c8d", lw=0.5)
            ax_v.set_xlim(-0.8, n + 0.8)
            ax_v.set_ylim(-1.25, 1.25)
            ax_v.set_xlabel("Half-move index t")
            ax_v.set_ylabel("learned V(s_t)")
            ax_v.set_title(
                "Reward propagates backward: one TD(0) update = one step",
                fontsize=9,
            )

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>Credit Assignment</b> — a full Connect-4 game has been "
                    "played below. The learned V(s_t) starts at 0 at every step "
                    "(agent knows nothing), except the final step which holds the "
                    "<b>true reward</b>. Click <i>Propagate 1 step</i> to apply a "
                    "single backward TD(0) update — the reward moves one step "
                    "closer to the opening. This is why TD(0) is slow on long "
                    "games, and why TD(λ) with λ&gt;0 is such a big win."
                ),
                HBox([
                    self._btn_step,
                    self._btn_all,
                    self._btn_reset,
                    self._btn_new,
                ]),
                self._html,
                self._out,
            ])
        )
