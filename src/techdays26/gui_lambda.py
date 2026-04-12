"""Interactive λ-return weight distribution visualizer.

Usage::

    from techdays26.gui_lambda import LambdaReturnVisualizer

    vis = LambdaReturnVisualizer()
    vis.show()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import HTML, FloatSlider, IntSlider, Layout, Output, VBox
from matplotlib.patches import Patch


class LambdaReturnVisualizer:
    """Drag λ and k to see how TD(λ) weights its n-step returns."""

    def __init__(self, *, lam: float = 0.5, k: int = 10) -> None:
        self._sl_lam = FloatSlider(
            value=lam,
            min=0.0,
            max=1.0,
            step=0.01,
            description="λ:",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=Layout(width="420px"),
        )
        self._sl_k = IntSlider(
            value=k,
            min=2,
            max=30,
            step=1,
            description="k:",
            continuous_update=True,
            style={"description_width": "30px"},
            layout=Layout(width="420px"),
        )
        self._out = Output(layout=Layout(width="auto"))

        self._sl_lam.observe(self._redraw, names="value")
        self._sl_k.observe(self._redraw, names="value")
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _redraw(self, _) -> None:
        lam = self._sl_lam.value
        k = self._sl_k.value

        # Truncated λ-return weights:
        #   n = 1 .. k-1 :  (1-λ) λ^(n-1)
        #   n = k         :  λ^(k-1)         (remaining mass → bootstrap)
        if k == 1:
            weights = np.array([1.0])
        else:
            weights = np.array(
                [(1 - lam) * lam**n for n in range(k - 1)] + [lam ** (k - 1)]
            )

        self._out.clear_output(wait=True)
        with self._out:
            fig_w = min(0.7 * k + 3, 14)
            fig, (ax_stack, ax) = plt.subplots(
                1,
                2,
                figsize=(fig_w + 1.2, 3.2),
                gridspec_kw={"width_ratios": [1, max(k, 4)], "wspace": 0.3},
            )
            fig.patch.set_facecolor("#f8f9fa")

            x = np.arange(1, k + 1)
            colors = ["#3498db"] * (k - 1) + ["#e74c3c"]
            ax.bar(x, weights, color=colors, edgecolor="white", width=0.7)

            # Value labels on bars
            for i, w in enumerate(weights):
                if w > 0.005:
                    ax.text(
                        x[i],
                        w + max(weights) * 0.03,
                        f"{w:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xlabel("n-step return  $G_t^{(n)}$")
            ax.set_ylabel("Weight")
            ax.set_title(f"λ-return weights  (λ = {lam:.2f}, k = {k})")
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in x], fontsize=7)
            ax.set_ylim(0, min(max(weights) * 1.3, 1.08))

            ax.legend(
                handles=[
                    Patch(fc="#3498db", label="$(1{-}\\lambda)\\,\\lambda^{n-1}$"),
                    Patch(
                        fc="#e74c3c",
                        label="$\\lambda^{k-1}$  (truncated bootstrap)",
                    ),
                ],
                fontsize=8,
                loc="upper right",
            )

            # ── Stacked bar on the left: weights sum to 1 ──
            bottom = 0.0
            for i, w in enumerate(weights):
                ax_stack.bar(
                    0,
                    w,
                    bottom=bottom,
                    color=colors[i],
                    edgecolor="white",
                    width=0.5,
                )
                if w > 0.04:
                    ax_stack.text(
                        0,
                        bottom + w / 2,
                        f"{w:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                        fontweight="bold",
                    )
                bottom += w
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

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>TD(λ) Return Weights</b> — drag the sliders to see "
                    "how λ and the truncation horizon <i>k</i> affect the "
                    "weight on each n-step return."
                ),
                self._sl_lam,
                self._sl_k,
                self._out,
            ])
        )
