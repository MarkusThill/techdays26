"""TD step-through visualizer — see V values evolve and λ-returns propagate.

Usage::

    from techdays26.gui_td_step import TDStepVisualizer

    vis = TDStepVisualizer(model_path="../td_weights_clean.tdw.zip")
    vis.show()
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from ipywidgets import (
    HTML,
    Button,
    FloatSlider,
    HBox,
    IntSlider,
    Layout,
    Output,
    VBox,
)

from techdays26.gui_common import COL_STRIDE, COLS, draw_board
from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


class TDStepVisualizer:
    """Step through a pre-played game showing V(afterstate) and λ-return targets."""

    def __init__(
        self,
        model_path: str = "",
        moves: list[int] | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required for TDStepVisualizer")

        self._model = NTupleNetwork.load(model_path, device="cpu")
        self._model.eval()

        self._moves = moves
        self._pre_play(moves)

        self._sl_step = IntSlider(
            value=0,
            min=0,
            max=len(self._traj) - 1,
            step=1,
            description="Step:",
            continuous_update=True,
            style={"description_width": "40px"},
            layout=Layout(width="500px"),
        )
        self._sl_lam = FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description="λ:",
            continuous_update=True,
            style={"description_width": "20px"},
            layout=Layout(width="350px"),
        )
        self._btn_new = Button(description="↺ New Game", button_style="warning")
        self._btn_new.on_click(self._on_new)
        self._out = Output()

        self._sl_step.observe(self._redraw, names="value")
        self._sl_lam.observe(self._redraw, names="value")
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _pre_play(self, moves: list[int] | None = None) -> None:
        """Play a game and record trajectory."""
        board = BoardBatch.empty(1, device="cpu")

        self._traj: list[dict] = []

        # Step 0: initial empty board
        with torch.no_grad():
            v0 = float(self._model(board)[0])
        self._traj.append({
            "at": int(board.all_tokens[0]),
            "act": int(board.active_tokens[0]),
            "ml": int(board.moves_left[0]),
            "v": v0,
            "reward": float("nan"),
            "col": -1,
        })

        step = 0
        while not bool(board.done()[0]):
            if moves and step < len(moves):
                col = moves[step]
            else:
                # Use the model to pick moves (greedy, no exploration)
                legal_int = int(board.generate_legal_moves()[0])
                legal_cols = [
                    c for c in range(COLS) if (legal_int >> (c * COL_STRIDE)) & 0x3F
                ]
                if not legal_cols:
                    break
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

            board.play_columns(torch.tensor([col]))
            step += 1

            r = float(board.reward()[0])
            with torch.no_grad():
                v = float(self._model(board)[0])

            self._traj.append({
                "at": int(board.all_tokens[0]),
                "act": int(board.active_tokens[0]),
                "ml": int(board.moves_left[0]),
                "v": v,
                "reward": r,
                "col": col,
            })

    def _on_new(self, _) -> None:
        self._pre_play(None)  # new random-ish game
        self._sl_step.max = len(self._traj) - 1
        self._sl_step.value = 0
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _compute_lambda_return(self, t: int, lam: float) -> list[float]:
        """Compute λ-return weights for step t looking forward.

        Returns list of weights for steps t+1, t+2, ..., up to game end.
        """
        n = len(self._traj) - 1  # last step index
        remaining = n - t
        if remaining <= 0:
            return []

        k = remaining
        if k == 1:
            return [1.0]

        weights = [(1 - lam) * lam**i for i in range(k - 1)] + [lam ** (k - 1)]
        return weights

    # ──────────────────────────────────────────────────────────────
    def _redraw(self, _) -> None:
        t = self._sl_step.value
        lam = self._sl_lam.value
        s = self._traj[t]
        n_total = len(self._traj) - 1

        self._out.clear_output(wait=True)
        with self._out:
            fig = plt.figure(figsize=(13, 7))
            fig.patch.set_facecolor("#f8f9fa")

            # Layout: board (top-left), V timeline (top-right),
            #          λ-return weights (bottom)
            gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2], height_ratios=[1.2, 1])
            ax_board = fig.add_subplot(gs[0, 0])
            ax_v = fig.add_subplot(gs[0, 1])
            ax_lam = fig.add_subplot(gs[1, :])

            # ── Board ──
            draw_board(ax_board, s["at"], s["act"], s["ml"], cell_radius=0.38)
            moves_played = 42 - s["ml"]
            if t == 0:
                ax_board.set_title("Empty board", fontsize=9)
            elif t == n_total and not math.isnan(s["reward"]):
                outcome = {1.0: "Yellow wins!", -1.0: "Red wins!", 0.0: "Draw!"}
                ax_board.set_title(
                    f"Move {t}: {outcome.get(s['reward'], '?')}",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                player = "Yellow" if (s["ml"] % 2) == 0 else "Red"
                ax_board.set_title(
                    f"After move {t}/{n_total} ({player} to move)", fontsize=9
                )

            # ── V timeline ──
            steps = list(range(n_total + 1))
            v_vals = [self._traj[i]["v"] for i in steps]

            # Color bars by who just moved (step 0 = initial, no move yet)
            bar_colors = []
            for i in steps:
                ml = self._traj[i]["ml"]
                if i == 0:
                    bar_colors.append("#bdc3c7")
                elif (ml % 2) == 1:
                    # moves_left odd → Yellow just moved
                    bar_colors.append("#f1c40f")
                else:
                    bar_colors.append("#e74c3c")

            ax_v.bar(steps, v_vals, color=bar_colors, width=0.7, alpha=0.6)

            # Highlight current step
            ax_v.bar(
                [t],
                [v_vals[t]],
                color=bar_colors[t],
                width=0.7,
                edgecolor="black",
                linewidth=2,
            )

            # Shade the λ-return window
            weights = self._compute_lambda_return(t, lam)
            if weights:
                future_steps = list(range(t + 1, t + 1 + len(weights)))
                max_w = max(weights) if weights else 1
                for fs, w in zip(future_steps, weights):
                    if fs <= n_total:
                        alpha = 0.15 + 0.6 * (w / max_w)
                        ax_v.axvspan(fs - 0.4, fs + 0.4, alpha=alpha, color="#2ecc71")

            # Terminal reward marker
            last_r = self._traj[n_total]["reward"]
            if not math.isnan(last_r):
                ax_v.plot(
                    n_total,
                    last_r,
                    marker="*",
                    markersize=12,
                    color="#e74c3c" if last_r < 0 else "#2ecc71",
                    zorder=5,
                )

            ax_v.axhline(0, color="#7f8c8d", lw=0.5)
            ax_v.set_xlim(-0.5, n_total + 0.5)
            ax_v.set_ylim(-1.2, 1.2)
            ax_v.set_xlabel("Step")
            ax_v.set_ylabel("V(afterstate)")
            ax_v.set_title(
                f"Value function over time  (step {t} selected, green = λ-return window)",
                fontsize=9,
            )
            ax_v.legend(
                handles=[
                    mpatches.Patch(fc="#f1c40f", alpha=0.6, label="Yellow moved"),
                    mpatches.Patch(fc="#e74c3c", alpha=0.6, label="Red moved"),
                ],
                fontsize=7,
                loc="upper left",
            )

            # ── λ-return weights (bottom) ──
            if weights:
                x_w = list(range(1, len(weights) + 1))
                w_colors = ["#3498db"] * (len(weights) - 1) + ["#e74c3c"]
                ax_lam.bar(x_w, weights, color=w_colors, width=0.6)

                for i, w in enumerate(weights):
                    if w > 0.005:
                        ax_lam.text(
                            x_w[i],
                            w + max(weights) * 0.03,
                            f"{w:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

                # Compute λ-return target
                future_v = [self._traj[t + 1 + i]["v"] for i in range(len(weights))]
                # Use reward for terminal step
                last_idx = t + len(weights)
                if last_idx <= n_total:
                    r_last = self._traj[last_idx]["reward"]
                    if not math.isnan(r_last):
                        future_v[-1] = r_last

                lam_target = sum(w * v for w, v in zip(weights, future_v))

                ax_lam.set_title(
                    f"λ-return weights from step {t}  "
                    f"(λ={lam:.2f})  →  target = {lam_target:+.4f}  "
                    f"vs  V(s_{t}) = {v_vals[t]:+.4f}",
                    fontsize=9,
                )
            else:
                ax_lam.set_title(
                    f"Step {t}: terminal — no λ-return (reward = {s['reward']:.1f})",
                    fontsize=9,
                )
                ax_lam.text(
                    0.5,
                    0.5,
                    "Terminal state — target is the reward directly",
                    transform=ax_lam.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#7f8c8d",
                )

            ax_lam.set_xlabel("n-step return")
            ax_lam.set_ylabel("Weight")

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>TD Step-Through</b> — drag the step slider to move "
                    "through the game. The <b>green shading</b> in the V-timeline "
                    "shows which future values contribute to the λ-return at the "
                    "selected step. Drag λ to see how the weight distribution changes."
                ),
                HBox([self._sl_step, self._btn_new]),
                self._sl_lam,
                self._out,
            ])
        )
