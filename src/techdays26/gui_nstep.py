"""N-step return visualizer — see how bootstrap horizon changes the target.

Usage::

    from techdays26.gui_nstep import NStepReturnVisualizer

    vis = NStepReturnVisualizer(
        model_path="../exp_20260228_13-46/repeat_0/step_500_model_weights.pt"
    )
    vis.show()
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from ipywidgets import HTML, Button, HBox, IntSlider, Layout, Output, VBox

from techdays26.gui_common import COL_STRIDE, COLS, draw_board
from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


class NStepReturnVisualizer:
    """Visualize how different n-step returns use different bootstrap targets."""

    def __init__(self, model_path: str = "", moves: list[int] | None = None) -> None:
        if not model_path:
            raise ValueError("model_path is required for NStepReturnVisualizer")

        self._model = NTupleNetwork.load(model_path, device="cpu")
        self._model.eval()

        self._moves = moves
        self._pre_play(moves)

        self._sl_step = IntSlider(
            value=0,
            min=0,
            max=max(len(self._traj) - 2, 0),
            step=1,
            description="Step t:",
            continuous_update=True,
            style={"description_width": "45px"},
            layout=Layout(width="450px"),
        )
        self._sl_n = IntSlider(
            value=1,
            min=1,
            max=max(len(self._traj) - 1, 1),
            step=1,
            description="n:",
            continuous_update=True,
            style={"description_width": "20px"},
            layout=Layout(width="350px"),
        )
        self._btn_new = Button(description="↺ New Game", button_style="warning")
        self._btn_new.on_click(self._on_new)
        self._out = Output()

        self._sl_step.observe(self._on_step_change, names="value")
        self._sl_n.observe(self._redraw, names="value")
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _pre_play(self, moves: list[int] | None = None) -> None:
        """Play a game and record trajectory of afterstate values."""
        board = BoardBatch.empty(1, device="cpu")
        self._traj: list[dict] = []

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
        self._pre_play(None)
        self._sl_step.max = max(len(self._traj) - 2, 0)
        self._sl_step.value = 0
        self._sl_n.max = max(len(self._traj) - 1, 1)
        self._sl_n.value = 1
        self._redraw(None)

    def _on_step_change(self, _) -> None:
        t = self._sl_step.value
        max_n = len(self._traj) - 1 - t
        self._sl_n.max = max(max_n, 1)
        if self._sl_n.value > max_n:
            self._sl_n.value = max(max_n, 1)
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _nstep_return(self, t: int, n: int) -> float:
        """Compute the n-step return G_t^(n).

        For Connect-4 with sparse rewards (r=0 except terminal):
          G_t^(n) = V(s_{t+n})    if t+n < T  (bootstrap)
          G_t^(n) = r_T           if t+n >= T  (actual outcome)
        """
        T = len(self._traj) - 1  # last index
        target_idx = min(t + n, T)
        last_r = self._traj[T]["reward"]

        if target_idx >= T and not math.isnan(last_r):
            return last_r
        return self._traj[target_idx]["v"]

    # ──────────────────────────────────────────────────────────────
    def _redraw(self, _) -> None:
        t = self._sl_step.value
        n = self._sl_n.value
        s = self._traj[t]
        T = len(self._traj) - 1

        self._out.clear_output(wait=True)
        with self._out:
            fig = plt.figure(figsize=(13, 6.5))
            fig.patch.set_facecolor("#f8f9fa")

            gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2], height_ratios=[1.2, 1])
            ax_board = fig.add_subplot(gs[0, 0])
            ax_v = fig.add_subplot(gs[0, 1])
            ax_comp = fig.add_subplot(gs[1, :])

            # ── Board ──
            draw_board(ax_board, s["at"], s["act"], s["ml"], cell_radius=0.38)
            if t == 0:
                ax_board.set_title("Empty board (step 0)", fontsize=9)
            else:
                player = "Yellow" if (s["ml"] % 2) == 0 else "Red"
                ax_board.set_title(f"After move {t}/{T} ({player} to move)", fontsize=9)

            # ── V timeline with n-step highlight ──
            steps = list(range(T + 1))
            v_vals = [self._traj[i]["v"] for i in steps]

            bar_colors = []
            for i in steps:
                ml = self._traj[i]["ml"]
                if i == 0:
                    bar_colors.append("#bdc3c7")
                elif (ml % 2) == 1:
                    bar_colors.append("#f1c40f")
                else:
                    bar_colors.append("#e74c3c")

            ax_v.bar(steps, v_vals, color=bar_colors, width=0.7, alpha=0.4)

            # Highlight current step t
            ax_v.bar(
                [t],
                [v_vals[t]],
                color="#3498db",
                width=0.7,
                edgecolor="black",
                linewidth=2,
                label=f"V(s$_t$) = {v_vals[t]:+.4f}",
            )

            # Highlight bootstrap target at t+n
            target_idx = min(t + n, T)
            g_n = self._nstep_return(t, n)
            last_r = self._traj[T]["reward"]
            is_terminal = target_idx >= T and not math.isnan(last_r)

            if is_terminal:
                ax_v.plot(
                    T,
                    last_r,
                    marker="*",
                    markersize=14,
                    color="#e74c3c",
                    zorder=5,
                    label=f"G$_t^{{({n})}}$ = r$_T$ = {last_r:+.1f}",
                )
            else:
                ax_v.bar(
                    [target_idx],
                    [v_vals[target_idx]],
                    color="#2ecc71",
                    width=0.7,
                    edgecolor="black",
                    linewidth=2,
                    label=f"G$_t^{{({n})}}$ = V(s$_{{{t + n}}}$) = {g_n:+.4f}",
                )

            # Draw arrow from t to bootstrap target
            ax_v.annotate(
                "",
                xy=(target_idx, g_n),
                xytext=(t, v_vals[t]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#2c3e50",
                    lw=1.8,
                    connectionstyle="arc3,rad=-0.15",
                ),
            )

            # Shade the n-step window
            for i in range(t, min(t + n, T) + 1):
                ax_v.axvspan(i - 0.4, i + 0.4, alpha=0.08, color="#2ecc71")

            # Terminal reward marker
            if not math.isnan(last_r):
                ax_v.plot(
                    T,
                    last_r,
                    marker="*",
                    markersize=10,
                    color="#e74c3c" if last_r < 0 else "#2ecc71",
                    zorder=4,
                )

            ax_v.axhline(0, color="#7f8c8d", lw=0.5)
            ax_v.set_xlim(-0.5, T + 0.5)
            ax_v.set_ylim(-1.3, 1.3)
            ax_v.set_xlabel("Step")
            ax_v.set_ylabel("V(afterstate)")
            ax_v.set_title(
                f"V timeline — {n}-step return from step {t}  "
                f"(bootstrap at step {target_idx}{'=T, terminal' if is_terminal else ''})",
                fontsize=9,
            )
            ax_v.legend(fontsize=7, loc="upper left")

            # ── Comparison: all n-step returns ──
            max_n = T - t
            if max_n >= 1:
                ns = list(range(1, max_n + 1))
                g_vals = [self._nstep_return(t, ni) for ni in ns]

                comp_colors = []
                for ni in ns:
                    ti = min(t + ni, T)
                    if ti >= T and not math.isnan(last_r):
                        comp_colors.append("#e74c3c")  # uses terminal reward
                    else:
                        comp_colors.append("#3498db")  # bootstraps from V

                ax_comp.bar(ns, g_vals, color=comp_colors, width=0.6, alpha=0.6)

                # Highlight the selected n
                ax_comp.bar(
                    [n],
                    [g_vals[n - 1]],
                    color=comp_colors[n - 1],
                    width=0.6,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.9,
                )

                # Value labels on bars
                for ni, gv in zip(ns, g_vals):
                    if abs(gv) > 0.001:
                        ax_comp.text(
                            ni,
                            gv + 0.03 * (1 if gv >= 0 else -1),
                            f"{gv:+.3f}",
                            ha="center",
                            va="bottom" if gv >= 0 else "top",
                            fontsize=6,
                        )

                # Reference line: actual outcome
                if not math.isnan(last_r):
                    ax_comp.axhline(
                        last_r,
                        color="#e74c3c",
                        lw=1,
                        ls="--",
                        alpha=0.7,
                        label=f"actual outcome r$_T$ = {last_r:+.1f}",
                    )

                # Reference line: current V(s_t)
                ax_comp.axhline(
                    v_vals[t],
                    color="#3498db",
                    lw=1,
                    ls=":",
                    alpha=0.7,
                    label=f"V(s$_t$) = {v_vals[t]:+.4f}",
                )

                ax_comp.legend(
                    handles=[
                        mpatches.Patch(
                            fc="#3498db",
                            alpha=0.6,
                            label="bootstraps from V(s$_{t+n}$)",
                        ),
                        mpatches.Patch(
                            fc="#e74c3c",
                            alpha=0.6,
                            label="uses terminal reward r$_T$ (Monte Carlo)",
                        ),
                        plt.Line2D(
                            [0],
                            [0],
                            color="#e74c3c",
                            ls="--",
                            alpha=0.7,
                            label=f"actual outcome = {last_r:+.1f}"
                            if not math.isnan(last_r)
                            else "game not over",
                        ),
                        plt.Line2D(
                            [0],
                            [0],
                            color="#3498db",
                            ls=":",
                            alpha=0.7,
                            label=f"V(s$_t$) = {v_vals[t]:+.4f}",
                        ),
                    ],
                    fontsize=7,
                    loc="upper right",
                )

                # Bias-variance annotation
                ax_comp.text(
                    1,
                    -1.15,
                    "← low variance, high bias",
                    fontsize=7,
                    color="#7f8c8d",
                    ha="left",
                )
                ax_comp.text(
                    max_n,
                    -1.15,
                    "low bias, high variance →",
                    fontsize=7,
                    color="#7f8c8d",
                    ha="right",
                )

                ax_comp.set_xticks(ns)
                ax_comp.set_xticklabels(
                    [f"n={ni}" for ni in ns],
                    fontsize=7,
                    rotation=45 if max_n > 15 else 0,
                )
            else:
                ax_comp.text(
                    0.5,
                    0.5,
                    "Terminal state — no forward returns",
                    transform=ax_comp.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#7f8c8d",
                )

            ax_comp.axhline(0, color="#7f8c8d", lw=0.5)
            ax_comp.set_ylim(-1.3, 1.3)
            ax_comp.set_xlabel("n (number of steps)")
            ax_comp.set_ylabel("G$_t^{(n)}$")
            ax_comp.set_title(
                f"All n-step returns from step {t}  (n = 1 … {max_n})",
                fontsize=9,
            )

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>N-Step Returns</b> — drag <i>n</i> to change the bootstrap "
                    "horizon. With <b>n = 1</b> we bootstrap from the next value "
                    "(biased but stable). As <b>n → T−t</b> we approach the Monte "
                    "Carlo return (unbiased but noisy). The bottom chart compares "
                    "<i>all</i> n-step returns from the selected step."
                ),
                HBox([self._sl_step, self._btn_new]),
                self._sl_n,
                self._out,
            ])
        )
