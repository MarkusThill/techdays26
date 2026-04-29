"""Continuous λ-return decay visualization (Sutton & Barto Fig 12.2 style).

Usage::

    from techdays26.gui_lambda_decay import LambdaDecayVisualizer

    vis = LambdaDecayVisualizer()
    vis.show()
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import HTML, FloatSlider, IntSlider, Layout, Output, VBox


class LambdaDecayVisualizer:
    """Continuous decay view of the λ-return weights (Sutton & Barto style)."""

    def __init__(self, *, lam: float = 0.5, T: int = 12, t: int = 1) -> None:
        self._sl_lam = FloatSlider(
            value=lam,
            min=0.01,
            max=0.99,
            step=0.01,
            description="λ:",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=Layout(width="420px"),
        )
        self._sl_T = IntSlider(
            value=T,
            min=4,
            max=30,
            step=1,
            description="T:",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=Layout(width="420px"),
        )
        self._sl_t = IntSlider(
            value=t,
            min=1,
            max=T - 1,
            step=1,
            description="t:",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=Layout(width="420px"),
        )
        self._out = Output(layout=Layout(width="auto"))

        self._sl_lam.observe(self._redraw, names="value")
        self._sl_T.observe(self._on_T_change, names="value")
        self._sl_t.observe(self._redraw, names="value")
        self._redraw(None)

    def _on_T_change(self, _) -> None:
        self._sl_t.max = max(self._sl_T.value - 1, 1)
        if self._sl_t.value > self._sl_t.max:
            self._sl_t.value = self._sl_t.max
        self._redraw(None)

    def _redraw(self, _) -> None:
        lam = self._sl_lam.value
        T = self._sl_T.value
        t = self._sl_t.value

        # n-step returns from n=1 to n=(T-t)
        # Weight on G_t^(n): (1-λ) λ^(n-1) for n < T-t
        # Weight on G_t (actual return at T): λ^(T-t-1)  (remaining mass)
        horizon = T - t  # number of steps from t to T

        self._out.clear_output(wait=True)
        with self._out:
            fig, (ax_stack, ax) = plt.subplots(
                1,
                2,
                figsize=(11.5, 4.5),
                gridspec_kw={"width_ratios": [1, max(horizon, 5)], "wspace": 0.25},
            )
            fig.patch.set_facecolor("#f8f9fa")

            # ── Draw the continuous decay envelope ──
            x_env = np.linspace(0, horizon + 1, 500)
            y_env = (1 - lam) * lam**x_env
            ax.plot(
                x_env + t,
                y_env,
                color="#2c3e50",
                lw=2,
                zorder=3,
            )

            # ── Draw bars for each n-step return ──
            bar_width = 0.8
            for n in range(1, horizon):
                w = (1 - lam) * lam ** (n - 1)
                x_pos = t + n - 1
                ax.bar(
                    x_pos,
                    w,
                    width=bar_width,
                    color="#3498db",
                    alpha=0.5,
                    edgecolor="#2c3e50",
                    linewidth=0.8,
                    zorder=2,
                )

            # ── Terminal bar: remaining mass λ^(T-t-1) ──
            terminal_w = lam ** (horizon - 1)
            # Draw as a wider/highlighted bar at T
            ax.bar(
                T - 1,
                terminal_w,
                width=bar_width,
                color="#e74c3c",
                alpha=0.6,
                edgecolor="#2c3e50",
                linewidth=0.8,
                zorder=2,
                hatch="///",
            )

            # ── Annotation: first bar weight ──
            first_w = 1 - lam
            ax.annotate(
                f"$1-\\lambda = {first_w:.2f}$",
                xy=(t, first_w),
                xytext=(t - 0.5, first_w + 0.08),
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
                color="#2c3e50",
                fontweight="bold",
            )

            # ── Annotation: decay ──
            mid_n = min(3, horizon - 1)
            if mid_n >= 2:
                mid_w = (1 - lam) * lam ** (mid_n - 1)
                ax.annotate(
                    f"decay by $\\lambda={lam:.2f}$",
                    xy=(t + mid_n - 0.5, mid_w + 0.02),
                    xytext=(t + mid_n + 1, mid_w + 0.12),
                    fontsize=9,
                    ha="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#7f8c8d",
                        lw=1,
                        connectionstyle="arc3,rad=-0.2",
                    ),
                    color="#7f8c8d",
                )

            # ── Annotation: terminal weight ──
            ax.annotate(
                f"weight on actual return\n"
                f"$\\lambda^{{T-t-1}} = {lam:.2f}^{{{horizon - 1}}} = {terminal_w:.4f}$\n"
                f"(remaining mass)",
                xy=(T - 1, terminal_w),
                xytext=(T - 1 + 0.3, terminal_w + 0.10),
                fontsize=9,
                ha="left",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                color="#c0392b",
                fontweight="bold",
            )

            # (the stacked bar on the left now shows Σ = 1)

            # ── Annotation: example bar ──
            if horizon > 3:
                ex_n = 3
                ex_w = (1 - lam) * lam ** (ex_n - 1)
                ax.annotate(
                    f"weight on {ex_n}-step return\n"
                    f"$(1-\\lambda)\\lambda^{{{ex_n - 1}}} = {ex_w:.4f}$",
                    xy=(t + ex_n - 1, ex_w),
                    xytext=(t + ex_n + 1.5, ex_w + 0.05),
                    fontsize=8,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
                    color="#7f8c8d",
                )

            # ── Axis labels and formatting ──
            ax.set_xlabel("Time", fontsize=11)
            ax.set_ylabel("Weighting", fontsize=11)

            # X ticks: show t, some intermediate, T
            xticks = [t]
            for n in range(1, horizon):
                xticks.append(t + n - 1 + 0.0)
            xticks.append(T - 1)
            xtick_labels = []
            for x in xticks:
                xi = int(round(x))
                if xi == t:
                    xtick_labels.append(f"$t={t}$")
                elif xi == T - 1:
                    xtick_labels.append(f"$T={T}$")
                else:
                    xtick_labels.append("")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, fontsize=9)

            ax.set_xlim(t - 1.5, T + 1.5)
            ax.set_ylim(0, max(first_w, terminal_w) + 0.32)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # ── Legend ──
            ax.legend(
                handles=[
                    mpatches.Patch(
                        fc="#3498db",
                        alpha=0.5,
                        ec="#2c3e50",
                        label="$(1{-}\\lambda)\\lambda^{n-1}$ — n-step bootstrap",
                    ),
                    mpatches.Patch(
                        fc="#e74c3c",
                        alpha=0.6,
                        ec="#2c3e50",
                        hatch="///",
                        label="$\\lambda^{T{-}t{-}1}$ — actual return (remaining mass)",
                    ),
                ],
                fontsize=8,
                loc="upper right",
            )

            ax.set_title(
                f"λ-return weight decay  (λ = {lam:.2f}, t = {t}, T = {T})",
                fontsize=12,
                fontweight="bold",
            )

            # ── Stacked bar on the left: weights sum to 1 ──
            bar_weights = [(1 - lam) * lam ** (n - 1) for n in range(1, horizon)]
            bar_colors = ["#3498db"] * len(bar_weights)
            bar_weights.append(terminal_w)
            bar_colors.append("#e74c3c")

            bottom = 0.0
            for bw, bc in zip(bar_weights, bar_colors):
                ax_stack.bar(
                    0,
                    bw,
                    bottom=bottom,
                    color=bc,
                    alpha=0.6,
                    edgecolor="white",
                    width=0.5,
                )
                if bw > 0.04:
                    ax_stack.text(
                        0,
                        bottom + bw / 2,
                        f"{bw:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                        fontweight="bold",
                    )
                bottom += bw
            ax_stack.set_ylim(0, 1.08)
            ax_stack.set_xlim(-0.6, 0.6)
            ax_stack.set_xticks([])
            ax_stack.set_ylabel("Cumulative weight")
            ax_stack.set_title("$\\Sigma = 1$", fontsize=10, fontweight="bold")
            ax_stack.axhline(1.0, color="#2c3e50", ls="--", lw=0.8, zorder=3)
            ax_stack.spines["top"].set_visible(False)
            ax_stack.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.show()

    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>λ-Return Weight Decay</b> — the weight on each n-step "
                    "return decays geometrically by λ. The <b>last bar</b> (actual "
                    "return at T) receives all remaining mass λ<sup>T−t−1</sup>, "
                    "which can be <i>larger</i> than the preceding bars when λ is "
                    "high and T−t is small."
                ),
                self._sl_lam,
                self._sl_T,
                self._sl_t,
                self._out,
            ])
        )
