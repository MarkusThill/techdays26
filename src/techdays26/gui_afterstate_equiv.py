"""Afterstate equivalence demo — different (s,a) pairs, same afterstate.

Usage::

    from techdays26.gui_afterstate_equiv import AfterstateEquivVisualizer

    vis = AfterstateEquivVisualizer()
    vis.show()

    # With value function:
    vis = AfterstateEquivVisualizer(model_path="../td_weights_clean.tdw.zip")
    vis.show()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from IPython.display import display
from ipywidgets import HTML, Dropdown, Layout, Output, VBox

from techdays26.gui_common import draw_board
from techdays26.torch_board import BoardBatch

# ── Pre-defined scenarios ─────────────────────────────────────────────────────
# Each scenario has two paths that produce the same afterstate.
# The "state" is the board BEFORE the last move; the "action" is the last move.
SCENARIOS = [
    {
        "name": "Simple swap (3 moves)",
        "desc": (
            "Yellow plays columns 3 and 2 in different order. "
            "Same tokens on the board, same afterstate!"
        ),
        "path_a": [3, 4, 2],  # Y c3, R c4, Y c2
        "path_b": [2, 4, 3],  # Y c2, R c4, Y c3
    },
    {
        "name": "Different last move (5 moves)",
        "desc": (
            "State A has Yellow at c3,c4 — Yellow plays c5. "
            "State B has Yellow at c5,c4 — Yellow plays c3. "
            "Different states, different actions, same afterstate!"
        ),
        "path_a": [3, 0, 4, 6, 5],  # Y c3, R c0, Y c4, R c6, Y c5
        "path_b": [5, 0, 4, 6, 3],  # Y c5, R c0, Y c4, R c6, Y c3
    },
    {
        "name": "Longer game, swapped moves (7 moves)",
        "desc": (
            "State A has Yellow at c0,c1,c2 — Yellow plays c4. "
            "State B has Yellow at c4,c1,c2 — Yellow plays c0. "
            "Different states, different actions, same afterstate!"
        ),
        "path_a": [0, 3, 1, 3, 2, 3, 4],  # last move c4
        "path_b": [4, 3, 1, 3, 2, 3, 0],  # last move c0
    },
]


class AfterstateEquivVisualizer:
    """Show that different (state, action) pairs produce the same afterstate."""

    def __init__(self, model_path: str = "") -> None:
        self._model = None
        if model_path:
            from techdays26.ntuple_network import NTupleNetwork

            self._model = NTupleNetwork.load(model_path, device="cpu")
            self._model.eval()

        self._dd = Dropdown(
            options=[(s["name"], i) for i, s in enumerate(SCENARIOS)],
            value=0,
            description="Scenario:",
            style={"description_width": "70px"},
            layout=Layout(width="320px"),
        )
        self._dd.observe(self._redraw, names="value")
        self._out = Output()
        self._redraw(None)

    # ──────────────────────────────────────────────────────────────
    def _build_boards(self, path: list[int]):
        """Play moves and return (state_before_last, action, afterstate)."""
        board = BoardBatch.empty(1, device="cpu")
        for col in path[:-1]:
            board.play_columns(torch.tensor([col]))

        state = (
            int(board.all_tokens[0]),
            int(board.active_tokens[0]),
            int(board.moves_left[0]),
        )
        action = path[-1]

        board.play_columns(torch.tensor([action]))
        afterstate = (
            int(board.all_tokens[0]),
            int(board.active_tokens[0]),
            int(board.moves_left[0]),
        )
        return state, action, afterstate

    def _eval_v(self, at: int, act: int, ml: int) -> float | None:
        """Evaluate V(afterstate) if model is loaded."""
        if self._model is None:
            return None
        b = BoardBatch(
            all_tokens=torch.tensor([at]),
            active_tokens=torch.tensor([act]),
            moves_left=torch.tensor([ml], dtype=torch.int16),
        )
        with torch.no_grad():
            return float(self._model(b)[0])

    # ──────────────────────────────────────────────────────────────
    def _redraw(self, _) -> None:
        sc = SCENARIOS[self._dd.value]
        state_a, act_a, after_a = self._build_boards(sc["path_a"])
        state_b, act_b, after_b = self._build_boards(sc["path_b"])

        # Sanity: afterstates must match
        assert after_a == after_b, "Bug: afterstates differ!"

        v = self._eval_v(*after_a)

        self._out.clear_output(wait=True)
        with self._out:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
            fig.patch.set_facecolor("#f8f9fa")

            # State A
            draw_board(
                axes[0],
                *state_a,
                cell_radius=0.38,
                highlight_col=act_a,
                highlight_color="#2ecc71",
            )
            axes[0].set_title(
                f"State A → plays col {act_a}",
                fontsize=9,
                fontweight="bold",
            )
            # Annotate path
            path_str = ", ".join(f"c{c}" for c in sc["path_a"])
            axes[0].text(
                3,
                -1.0,
                f"Path: {path_str}",
                ha="center",
                fontsize=7,
                color="#7f8c8d",
            )

            # State B
            draw_board(
                axes[1],
                *state_b,
                cell_radius=0.38,
                highlight_col=act_b,
                highlight_color="#e67e22",
            )
            axes[1].set_title(
                f"State B → plays col {act_b}",
                fontsize=9,
                fontweight="bold",
            )
            path_str = ", ".join(f"c{c}" for c in sc["path_b"])
            axes[1].text(
                3,
                -1.0,
                f"Path: {path_str}",
                ha="center",
                fontsize=7,
                color="#7f8c8d",
            )

            # Afterstate (same for both!)
            draw_board(axes[2], *after_a, cell_radius=0.38)
            title = "Afterstate (same!)"
            if v is not None:
                title += f"\nV = {v:+.4f}"
            axes[2].set_title(title, fontsize=9, fontweight="bold", color="#2ecc71")

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────
    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>Afterstate Equivalence</b> — different (state, action) "
                    "pairs can produce the <b>same afterstate</b>. "
                    "A Q(s, a) function would learn separate values for each; "
                    "V(s') shares one value for all."
                ),
                self._dd,
                self._out,
            ])
        )
