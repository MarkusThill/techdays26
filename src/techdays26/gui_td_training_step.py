"""TD training-step visualizer — walk through one half-move TD update.

In self-play TD learning, **each training step is ONE half-move**: the player
to move (Yellow OR Red) picks an action, plays it, and the value function is
updated. The same model plays both colors. After Yellow's update comes Red's
update, then Yellow again, and so on.

This widget walks through that single half-move update step by step.

Usage::

    from techdays26.gui_td_training_step import TDTrainingStepVisualizer

    vis = TDTrainingStepVisualizer(
        model_path="../exp_20260228_13-46/repeat_0/step_500_model_weights.pt"
    )
    vis.show()
"""

from __future__ import annotations

import math
import random as pyrandom

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from IPython.display import display
from ipywidgets import (
    Button,
    FloatSlider,
    HBox,
    VBox,
    Output,
    Layout,
    HTML,
)

from techdays26.gui_common import (
    draw_board,
    snap,
    legal_columns,
    COLS,
)
from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch

_PHASE_TITLES = [
    "Phase 0 — Current State s",
    "Phase 1 — Evaluate Afterstates",
    "Phase 2 — Select Action (\u03b5-greedy)",
    "Phase 3 — Play the Move",
    "Phase 4 — Compute TD(0) Target",
    "Phase 5 — Weight Update",
    "Phase 6 — Next Half-Move",
]

_N_PHASES = len(_PHASE_TITLES)


class TDTrainingStepVisualizer:
    """Walk through one half-move TD training step on a single board.

    Each call to "Continue" advances by ONE half-move (the player flips).
    Within a half-move, the user steps through 7 phases that show how the
    value function is updated for that specific board.
    """

    def __init__(
        self,
        model_path: str = "",
        initial_moves: list[int] | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required")

        self._model = NTupleNetwork.load(model_path, device="cpu")
        self._model.eval()

        # ── State ──────────────────────────────────────────────────────
        self._initial_moves = initial_moves
        self._board = BoardBatch.empty(1, device="cpu")
        self._phase: int = 0

        # Per-half-move computed data (cleared on each new half-move)
        self._s_before_snap: tuple = (0, 0, 42)  # state before the move
        self._v_pred: float = 0.0  # V(s_before)  — the prediction we update
        self._move_evals: list[tuple[int, float, tuple]] = []  # (col, V, snap)
        self._chosen_col: int = -1
        self._explored: bool = False
        self._s_after_snap: tuple = (0, 0, 42)  # state after the move
        self._v_after: float = 0.0  # V(s_after)  — value of next state
        self._td_target: float = 0.0  # reward if terminal else V(s_after)
        self._td_error: float = 0.0
        self._terminal: bool = False
        self._terminal_reward: float = float("nan")

        self._history: list[dict] = []
        self._phase_computed: set[int] = set()

        self._setup_position(initial_moves)

        # ── Widgets ────────────────────────────────────────────────────
        self._btn_prev = Button(
            description="\u25c0 Prev",
            layout=Layout(width="80px"),
        )
        self._btn_next = Button(
            description="Next \u25b6",
            button_style="primary",
            layout=Layout(width="80px"),
        )
        self._btn_continue = Button(
            description="\u25b6\u25b6 Next Half-Move",
            button_style="info",
            layout=Layout(width="160px"),
        )
        self._btn_reset = Button(
            description="\u21ba New Game",
            button_style="warning",
            layout=Layout(width="110px"),
        )
        self._sl_eps = FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="\u03b5:",
            continuous_update=False,
            style={"description_width": "20px"},
            layout=Layout(width="280px"),
        )
        self._html_phase = HTML()
        self._out = Output()

        self._btn_prev.on_click(self._on_prev)
        self._btn_next.on_click(self._on_next)
        self._btn_continue.on_click(self._on_continue)
        self._btn_reset.on_click(self._on_reset)

        self._redraw()

    # ══════════════════════════════════════════════════════════════════
    # Setup & data computation
    # ══════════════════════════════════════════════════════════════════

    def _setup_position(self, moves: list[int] | None = None) -> None:
        self._board = BoardBatch.empty(1, device="cpu")
        if moves:
            for col in moves:
                self._board.play_columns(torch.tensor([col]))
                if bool(self._board.done()[0]):
                    break
        self._phase = 0
        self._phase_computed = set()
        self._terminal = False
        self._terminal_reward = float("nan")

    def _eval_model(self, board: BoardBatch) -> float:
        with torch.no_grad():
            return float(self._model(board)[0])

    def _eval_snap(self, snap_tuple: tuple) -> float:
        b = BoardBatch(
            all_tokens=torch.tensor([snap_tuple[0]]),
            active_tokens=torch.tensor([snap_tuple[1]]),
            moves_left=torch.tensor([snap_tuple[2]]),
        )
        return self._eval_model(b)

    def _player_to_move(self, moves_left: int) -> str:
        return "Yellow" if (moves_left % 2) == 0 else "Red"

    def _is_yellow_to_move(self, moves_left: int) -> bool:
        return (moves_left % 2) == 0

    def _compute_phase(self, phase: int) -> None:
        """Lazily compute data needed for the given phase."""
        if phase in self._phase_computed:
            return

        if phase == 0:
            self._s_before_snap = snap(self._board, 0)
            if bool(self._board.done()[0]):
                self._v_pred = float(self._board.reward()[0])
            else:
                self._v_pred = self._eval_model(self._board)

        elif phase == 1:
            self._compute_phase(0)
            self._move_evals = []
            for col in legal_columns(self._board, 0):
                tmp = BoardBatch(
                    all_tokens=self._board.all_tokens.clone(),
                    active_tokens=self._board.active_tokens.clone(),
                    moves_left=self._board.moves_left.clone(),
                )
                tmp.play_columns(torch.tensor([col]))
                r = float(tmp.reward()[0])
                if not math.isnan(r):
                    v = r  # terminal — use reward
                else:
                    v = self._eval_model(tmp)
                self._move_evals.append((col, v, snap(tmp, 0)))

        elif phase == 2:
            self._compute_phase(1)
            if not self._move_evals:
                return
            ml = self._s_before_snap[2]
            yellow = self._is_yellow_to_move(ml)

            # Yellow MAXIMIZES (positive V = good for Yellow)
            # Red MINIMIZES (negative V = good for Red)
            if yellow:
                best_col = max(self._move_evals, key=lambda x: x[1])[0]
            else:
                best_col = min(self._move_evals, key=lambda x: x[1])[0]

            eps = self._sl_eps.value
            self._explored = pyrandom.random() < eps and len(self._move_evals) > 1
            if self._explored:
                self._chosen_col = pyrandom.choice([c for c, _, _ in self._move_evals])
            else:
                self._chosen_col = best_col

        elif phase == 3:
            self._compute_phase(2)
            # Find chosen afterstate from cached evaluations
            for col, v, s in self._move_evals:
                if col == self._chosen_col:
                    self._s_after_snap = s
                    self._v_after = v
                    break

        elif phase == 4:
            self._compute_phase(3)
            # Determine if the move was terminal
            after_board = BoardBatch(
                all_tokens=torch.tensor([self._s_after_snap[0]]),
                active_tokens=torch.tensor([self._s_after_snap[1]]),
                moves_left=torch.tensor([self._s_after_snap[2]]),
            )
            self._terminal = bool(after_board.done()[0])
            if self._terminal:
                self._terminal_reward = float(after_board.reward()[0])
                # Reward is from the perspective of who just moved.
                # For consistency we want Yellow's global perspective:
                #   if Yellow just moved and won → +1 (good for Yellow)
                #   if Red just moved and won → -1 (good for Red)
                # board.reward() returns +1 if last mover won, -1 if not (or 0 draw)
                # The "last mover" is determined by post-move parity.
                # The torch_board reward() already returns Yellow-positive (+1=Yellow win)
                # So we just use it directly.
                self._td_target = self._terminal_reward
            else:
                # Bootstrap: target = V(s_after), evaluated from the same model.
                # The model output is in a consistent (Yellow-positive) frame, so
                # V(s_before) and V(s_after) are directly comparable.
                self._td_target = self._v_after

            self._td_error = self._td_target - self._v_pred

        elif phase == 5:
            self._compute_phase(4)

        elif phase == 6:
            self._compute_phase(5)
            # Record this half-move in history (only once)
            ml_before = self._s_before_snap[2]
            self._history.append({
                "player": self._player_to_move(ml_before),
                "col": self._chosen_col,
                "v_pred": self._v_pred,
                "target": self._td_target,
                "error": self._td_error,
                "explored": self._explored,
                "terminal": self._terminal,
            })

        self._phase_computed.add(phase)

    # ══════════════════════════════════════════════════════════════════
    # Widget callbacks
    # ══════════════════════════════════════════════════════════════════

    def _on_next(self, _) -> None:
        if self._phase < _N_PHASES - 1:
            self._phase += 1
            self._redraw()

    def _on_prev(self, _) -> None:
        # All phases 0-6 only inspect cached snapshots and don't mutate
        # the board, so we can freely move backward within a half-move.
        if self._phase > 0:
            self._phase -= 1
            self._redraw()

    def _on_continue(self, _) -> None:
        """Commit the current half-move and advance to the next half-move."""
        # Make sure phase 6 has run so the history is recorded
        self._compute_phase(6)

        # Apply the chosen move to the actual board
        if self._chosen_col >= 0 and not bool(self._board.done()[0]):
            self._board.play_columns(torch.tensor([self._chosen_col]))

        # Reset for the next half-move
        self._phase = 0
        self._phase_computed = set()
        self._terminal = False
        self._terminal_reward = float("nan")
        self._chosen_col = -1
        self._redraw()

    def _on_reset(self, _) -> None:
        self._history = []
        self._setup_position(self._initial_moves)
        self._redraw()

    # ══════════════════════════════════════════════════════════════════
    # Narrative
    # ══════════════════════════════════════════════════════════════════

    def _narrative(self, phase: int) -> str:
        ml_before = self._s_before_snap[2]
        player = self._player_to_move(ml_before)
        player_color = "#f1c40f" if player == "Yellow" else "#e74c3c"
        player_html = (
            f"<span style='color:{player_color}; font-weight:bold'>{player}</span>"
        )

        if phase == 0:
            if bool(self._board.done()[0]):
                return "The game is over. Click <b>New Game</b> to start fresh."
            return (
                f"It is <b>{player_html}\u2019s turn</b>. The model evaluates the "
                f"current state and predicts <b>V(s) = {self._v_pred:+.3f}</b>. "
                f"This is the value we will <i>update</i> in this training step. "
                f"Positive = good for Yellow, negative = good for Red."
            )

        elif phase == 1:
            n = len(self._move_evals)
            if not self._move_evals:
                return "No legal moves available."
            yellow = self._is_yellow_to_move(ml_before)
            if yellow:
                best = max(self._move_evals, key=lambda x: x[1])
                rule = "Yellow wants to <b>maximize</b> V"
            else:
                best = min(self._move_evals, key=lambda x: x[1])
                rule = "Red wants to <b>minimize</b> V"
            return (
                f"For each of the {n} legal moves, {player_html} imagines the "
                f"resulting board (the <i>afterstate</i>) and evaluates it. "
                f"{rule}, so the best move is column <b>{best[0]}</b> "
                f"with V = <b>{best[1]:+.3f}</b>."
            )

        elif phase == 2:
            eps = self._sl_eps.value
            if self._explored:
                return (
                    f"With <b>\u03b5 = {eps:.2f}</b>, the coin flip said "
                    f"<span style='color:#e67e22; font-weight:bold'>EXPLORE</span>. "
                    f"{player_html} picks a <b>random</b> column "
                    f"<b>{self._chosen_col}</b> instead of the greedy one. "
                    f"Exploration helps discover strategies the agent might miss."
                )
            return (
                f"With <b>\u03b5 = {eps:.2f}</b>, {player_html} "
                f"<span style='color:#2ecc71; font-weight:bold'>EXPLOITS</span> "
                f"and picks the best column: <b>{self._chosen_col}</b>."
            )

        elif phase == 3:
            ml_after = self._s_after_snap[2]
            next_player = self._player_to_move(ml_after)
            next_color = "#f1c40f" if next_player == "Yellow" else "#e74c3c"
            return (
                f"{player_html} plays column <b>{self._chosen_col}</b>. "
                f"The board transitions to a new state s\u2032. "
                f"It is now <span style='color:{next_color}; font-weight:bold'>"
                f"{next_player}</span>\u2019s turn. (The next half-move will be "
                f"<i>their</i> training step.)"
            )

        elif phase == 4:
            if self._terminal:
                outcome_map = {1.0: "Yellow wins!", -1.0: "Red wins!", 0.0: "Draw!"}
                outcome = outcome_map.get(self._terminal_reward, "?")
                return (
                    f"The move ended the game (<b>{outcome}</b>). The TD target "
                    f"is the <b>true reward</b> = {self._td_target:+.3f}. "
                    f"No bootstrapping needed \u2014 we know the truth!"
                )
            return (
                f"The <b>TD(0) target</b> is the model\u2019s value of the new state: "
                f"<b>target = V(s\u2032) = {self._td_target:+.3f}</b>. "
                f"This is <i>bootstrapping</i> \u2014 the network teaches itself "
                f"using its own future estimate. "
                f"<b>TD error</b> = target \u2212 V(s) = "
                f"<b>{self._td_error:+.3f}</b>. "
                f"<span style='color:#7f8c8d'>(\u00a75.3\u20135.4 will replace this "
                f"with a smarter <b>\u03bb-return</b> target that blends several "
                f"future steps.)</span>"
            )

        elif phase == 5:
            loss = self._td_error**2
            direction = "up" if self._td_error > 0 else "down"
            return (
                f"<b>Loss</b> = (V(s) \u2212 target)\u00b2 "
                f"= ({self._v_pred:+.3f} \u2212 {self._td_target:+.3f})\u00b2 "
                f"= <b>{loss:.4f}</b>. "
                f"The gradient pushes V(s) <b>{direction}</b> toward the target. "
                f"This is how {player_html} learns from a single half-move "
                f"\u2014 no need to wait for the game to end!"
            )

        elif phase == 6:
            n = len(self._history)
            ml_after = self._s_after_snap[2]
            next_player = self._player_to_move(ml_after)
            next_color = "#f1c40f" if next_player == "Yellow" else "#e74c3c"
            if self._terminal:
                return (
                    f"<b>{n} half-move update(s)</b> completed. The game ended. "
                    f"Click <b>New Game</b> to start a fresh self-play episode."
                )
            return (
                f"<b>{n} half-move update(s)</b> completed. Click "
                f"<b>Next Half-Move</b> to apply this move and run the next "
                f"training step \u2014 it will be "
                f"<span style='color:{next_color}; font-weight:bold'>"
                f"{next_player}</span>\u2019s turn."
            )

        return ""

    # ══════════════════════════════════════════════════════════════════
    # Drawing
    # ══════════════════════════════════════════════════════════════════

    def _redraw(self, _=None) -> None:
        phase = self._phase
        self._compute_phase(phase)

        # Update button states
        self._btn_prev.disabled = phase == 0
        max_phase = _N_PHASES - 1
        game_done = bool(self._board.done()[0])
        self._btn_next.disabled = phase >= max_phase or (game_done and phase == 0)
        self._btn_continue.disabled = (
            phase < max_phase or self._chosen_col < 0 or game_done
        )

        if game_done and phase == 0:
            title = "Game Over"
        else:
            title = _PHASE_TITLES[phase]
        self._html_phase.value = (
            f"<h3 style='margin:4px 0'>{title}</h3>"
            f"<p style='margin:2px 0; color:#555'>{self._narrative(phase)}</p>"
        )

        self._out.clear_output(wait=True)
        with self._out:
            draw_fn = [
                self._draw_phase_0,
                self._draw_phase_1,
                self._draw_phase_2,
                self._draw_phase_3,
                self._draw_phase_4,
                self._draw_phase_5,
                self._draw_phase_6,
            ][phase]
            draw_fn()

    # ── Phase 0: Current state ────────────────────────────────────────

    def _draw_phase_0(self) -> None:
        fig, (ax_board, ax_info) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.0, 1.5]}
        )
        fig.patch.set_facecolor("#f8f9fa")

        at, act, ml = self._s_before_snap
        draw_board(ax_board, at, act, ml, cell_radius=0.38)
        moves_played = 42 - ml
        player = self._player_to_move(ml)
        if bool(self._board.done()[0]):
            ax_board.set_title(f"Game Over ({moves_played} moves)", fontsize=9)
        else:
            ax_board.set_title(
                f"State s \u2014 {player} to move ({moves_played} moves played)",
                fontsize=9,
            )

        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")

        v = self._v_pred
        color = "#2ecc71" if v >= 0 else "#e74c3c"
        ax_info.text(
            0.5,
            0.65,
            "V(s)",
            fontsize=14,
            ha="center",
            va="center",
            color="#7f8c8d",
            transform=ax_info.transAxes,
        )
        ax_info.text(
            0.5,
            0.5,
            f"{v:+.3f}",
            fontsize=32,
            fontweight="bold",
            ha="center",
            va="center",
            color=color,
            transform=ax_info.transAxes,
        )
        ax_info.text(
            0.5,
            0.32,
            "the prediction we will\nupdate in this step",
            fontsize=9,
            ha="center",
            va="center",
            color="#7f8c8d",
            style="italic",
            transform=ax_info.transAxes,
        )

        # Yellow/Red orientation bar
        ax_bar = fig.add_axes([0.58, 0.10, 0.35, 0.06])
        gradient = np.linspace(-1, 1, 256).reshape(1, -1)
        ax_bar.imshow(
            gradient,
            aspect="auto",
            cmap="RdYlGn",
            extent=[-1, 1, 0, 1],
        )
        ax_bar.axvline(v, color="black", lw=2)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_bar.set_xlabel("Red wins \u2190 \u2192 Yellow wins", fontsize=7)
        ax_bar.tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()

    # ── Phase 1: Evaluate afterstates ─────────────────────────────────

    def _draw_phase_1(self) -> None:
        n_moves = len(self._move_evals)
        if n_moves == 0:
            return

        fig = plt.figure(figsize=(13, 6))
        fig.patch.set_facecolor("#f8f9fa")
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 2.0],
            height_ratios=[1.0, 1.0],
            hspace=0.4,
            wspace=0.3,
        )

        # Current board (left, full height)
        ax_board = fig.add_subplot(gs[:, 0])
        at, act, ml = self._s_before_snap
        draw_board(ax_board, at, act, ml, cell_radius=0.38)
        player = self._player_to_move(ml)
        ax_board.set_title(f"State s ({player} to move)", fontsize=9)

        yellow = self._is_yellow_to_move(ml)
        cols_list = [c for c, _, _ in self._move_evals]
        vals = [v for _, v, _ in self._move_evals]
        best_v = max(vals) if yellow else min(vals)

        # Bar chart of V(afterstate)
        ax_bar = fig.add_subplot(gs[0, 1])
        bar_colors = [
            "#2ecc71" if v == best_v else "#3498db" for _, v, _ in self._move_evals
        ]
        ax_bar.bar(cols_list, vals, color=bar_colors, width=0.6)
        ax_bar.axhline(0, color="#7f8c8d", lw=0.5)
        for c, v in zip(cols_list, vals):
            ax_bar.text(
                c,
                v + 0.03 * (1 if v >= 0 else -1),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )
        ax_bar.set_xticks(range(COLS))
        ax_bar.set_ylim(-1.15, 1.15)
        ax_bar.set_xlabel("Column")
        ax_bar.set_ylabel("V(afterstate)")
        rule = "max" if yellow else "min"
        ax_bar.set_title(
            f"V(afterstate) per legal move (green = best for {player}, {rule})",
            fontsize=9,
        )

        # Mini afterstate boards
        ax_mini = fig.add_subplot(gs[1, 1])
        ax_mini.axis("off")
        ax_mini.set_title("Imagined afterstates", fontsize=9)
        mini_w = 0.85 / max(n_moves, 1)
        for i, (col, v, (a, ac, m)) in enumerate(self._move_evals):
            x = 0.08 + i * mini_w
            inset = ax_mini.inset_axes(
                [x, 0.05, mini_w * 0.85, 0.85],
                transform=ax_mini.transAxes,
            )
            draw_board(inset, a, ac, m, cell_radius=0.35, show_col_labels=False)
            is_best = v == best_v
            ec = "#2ecc71" if is_best else "#cccccc"
            lw = 3 if is_best else 1
            for spine in inset.spines.values():
                spine.set_edgecolor(ec)
                spine.set_linewidth(lw)
                spine.set_visible(True)
            inset.set_title(f"col {col}\n{v:+.3f}", fontsize=7, pad=2)

        plt.show()

    # ── Phase 2: ε-greedy selection ───────────────────────────────────

    def _draw_phase_2(self) -> None:
        fig, (ax_board, ax_bar) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.0, 1.5]}
        )
        fig.patch.set_facecolor("#f8f9fa")

        at, act, ml = self._s_before_snap
        h_color = "#e67e22" if self._explored else "#2ecc71"
        draw_board(
            ax_board,
            at,
            act,
            ml,
            cell_radius=0.38,
            highlight_col=self._chosen_col,
            highlight_color=h_color,
        )
        mode = "EXPLORE" if self._explored else "EXPLOIT"
        ax_board.set_title(f"Chosen: col {self._chosen_col} ({mode})", fontsize=9)

        cols_list = [c for c, _, _ in self._move_evals]
        vals = [v for _, v, _ in self._move_evals]
        bar_colors = []
        for col, v, _ in self._move_evals:
            if col == self._chosen_col:
                bar_colors.append(h_color)
            else:
                bar_colors.append("#3498db")

        ax_bar.bar(cols_list, vals, color=bar_colors, width=0.6)
        ax_bar.axhline(0, color="#7f8c8d", lw=0.5)
        for c, v in zip(cols_list, vals):
            label = f"{v:+.3f}"
            if c == self._chosen_col:
                label += " \u2190"
            ax_bar.text(
                c,
                v + 0.03 * (1 if v >= 0 else -1),
                label,
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
                fontweight="bold" if c == self._chosen_col else "normal",
            )
        ax_bar.set_xticks(range(COLS))
        ax_bar.set_ylim(-1.15, 1.15)
        ax_bar.set_xlabel("Column")
        ax_bar.set_ylabel("V(afterstate)")

        eps = self._sl_eps.value
        ax_bar.set_title(
            f"\u03b5-greedy (\u03b5={eps:.2f}): "
            f"{'EXPLORE (random)' if self._explored else 'EXPLOIT (best)'}",
            fontsize=9,
        )

        legend_handles = [
            mpatches.Patch(fc="#2ecc71", label="Exploit (best)"),
            mpatches.Patch(fc="#e67e22", label="Explore (random)"),
            mpatches.Patch(fc="#3498db", label="Other moves"),
        ]
        ax_bar.legend(handles=legend_handles, fontsize=7, loc="upper right")

        plt.tight_layout()
        plt.show()

    # ── Phase 3: Play the move ────────────────────────────────────────

    def _draw_phase_3(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor("#f8f9fa")

        # Before
        at, act, ml = self._s_before_snap
        draw_board(ax1, at, act, ml, cell_radius=0.38)
        player = self._player_to_move(ml)
        ax1.set_title(f"State s\n({player} to move)", fontsize=9)

        # After
        at2, act2, ml2 = self._s_after_snap
        h_color = "#e67e22" if self._explored else "#2ecc71"
        draw_board(
            ax2,
            at2,
            act2,
            ml2,
            cell_radius=0.38,
            highlight_col=self._chosen_col,
            highlight_color=h_color,
        )
        ml_after = ml2
        next_player = self._player_to_move(ml_after)
        after_board = BoardBatch(
            all_tokens=torch.tensor([at2]),
            active_tokens=torch.tensor([act2]),
            moves_left=torch.tensor([ml2]),
        )
        if bool(after_board.done()[0]):
            ax2.set_title(
                f"State s\u2032 (col {self._chosen_col})\nGAME OVER",
                fontsize=9,
            )
        else:
            ax2.set_title(
                f"State s\u2032 (col {self._chosen_col})\n({next_player} to move)",
                fontsize=9,
            )

        # Arrow between boards
        fig.text(
            0.5,
            0.5,
            f"\u27a1\nplay col {self._chosen_col}",
            fontsize=14,
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        plt.tight_layout()
        plt.show()

    # ── Phase 4: Compute TD target ────────────────────────────────────

    def _draw_phase_4(self) -> None:
        fig = plt.figure(figsize=(12, 5))
        fig.patch.set_facecolor("#f8f9fa")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], wspace=0.3, hspace=0.45)

        # State s
        ax_pred = fig.add_subplot(gs[0, 0])
        at, act, ml = self._s_before_snap
        draw_board(ax_pred, at, act, ml, cell_radius=0.35)
        ax_pred.set_title(
            f"State s\nV(s) = {self._v_pred:+.3f}  (prediction)",
            fontsize=9,
        )

        # State s'
        ax_target = fig.add_subplot(gs[0, 1])
        at2, act2, ml2 = self._s_after_snap
        draw_board(ax_target, at2, act2, ml2, cell_radius=0.35)
        if self._terminal:
            ax_target.set_title(
                f"State s\u2032 (terminal)\nreward = {self._td_target:+.3f}",
                fontsize=9,
            )
        else:
            ax_target.set_title(
                f"State s\u2032\nV(s\u2032) = {self._td_target:+.3f}  (target)",
                fontsize=9,
            )

        fig.text(
            0.5,
            0.72,
            "\u27a1",
            fontsize=28,
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # TD-error number line — zoom to the interesting range so that
        # V(s) and the target don't overplot each other when they're close.
        lo = min(self._v_pred, self._td_target)
        hi = max(self._v_pred, self._td_target)
        span = hi - lo
        # Ensure a minimum visual span so labels never stack on top of each
        # other even when V(s) ≈ target.
        min_span = 0.4
        if span < min_span:
            pad = (min_span - span) / 2 + 0.1
        else:
            pad = span * 0.35
        x_lo = lo - pad
        x_hi = hi + pad
        # Keep within the theoretical V range [-1, 1] when possible
        x_lo = max(x_lo, -1.15)
        x_hi = min(x_hi, 1.15)

        ax_err = fig.add_subplot(gs[1, :])
        ax_err.set_xlim(x_lo, x_hi)
        ax_err.set_ylim(-0.5, 1.6)
        ax_err.axis("off")
        ax_err.axhline(0.5, xmin=0.05, xmax=0.95, color="#bdc3c7", lw=2)

        # Offset labels horizontally when points are close, to prevent overlap.
        # If V(s) < target → V(s) label left-aligned, target label right-aligned.
        # If V(s) > target → reverse. If equal, stack vertically.
        if self._v_pred < self._td_target:
            pred_ha, target_ha = "right", "left"
            pred_dx, target_dx = -0.01 * (x_hi - x_lo), 0.01 * (x_hi - x_lo)
        elif self._v_pred > self._td_target:
            pred_ha, target_ha = "left", "right"
            pred_dx, target_dx = 0.01 * (x_hi - x_lo), -0.01 * (x_hi - x_lo)
        else:
            pred_ha, target_ha = "center", "center"
            pred_dx, target_dx = 0.0, 0.0

        ax_err.plot(self._v_pred, 0.5, "o", markersize=14, color="#3498db", zorder=5)
        ax_err.text(
            self._v_pred + pred_dx,
            0.2,
            f"V(s)\n{self._v_pred:+.3f}",
            ha=pred_ha,
            va="top",
            fontsize=9,
            color="#3498db",
            fontweight="bold",
        )

        ax_err.plot(
            self._td_target,
            0.5,
            "s",
            markersize=14,
            color="#e74c3c",
            zorder=5,
        )
        target_label = "reward" if self._terminal else "V(s\u2032)"
        ax_err.text(
            self._td_target + target_dx,
            0.2,
            f"{target_label}\n{self._td_target:+.3f}",
            ha=target_ha,
            va="top",
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
        )

        if abs(self._td_error) > 0.001:
            ax_err.annotate(
                "",
                xy=(self._td_target, 0.85),
                xytext=(self._v_pred, 0.85),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#2c3e50",
                    lw=2.5,
                    connectionstyle="arc3,rad=0.15",
                ),
            )
            mid = (self._v_pred + self._td_target) / 2
            ax_err.text(
                mid,
                1.25,
                f"TD error = target \u2212 V(s) = {self._td_error:+.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#2c3e50",
            )

        # Dynamic ticks: pick a nice step so we get roughly 5-7 labels
        # across the visible range.
        def _nice_step(span: float) -> float:
            rough = span / 6.0
            magnitude = 10 ** math.floor(math.log10(rough)) if rough > 0 else 0.1
            for m in (1, 2, 2.5, 5, 10):
                if m * magnitude >= rough:
                    return m * magnitude
            return 10 * magnitude

        step = _nice_step(x_hi - x_lo)
        first = math.ceil(x_lo / step) * step
        tick = first
        while tick <= x_hi + 1e-9:
            ax_err.plot(tick, 0.5, "|", markersize=8, color="#bdc3c7")
            ax_err.text(
                tick,
                0.42,
                f"{tick:+.2f}",
                ha="center",
                va="top",
                fontsize=7,
                color="#95a5a6",
            )
            tick += step

        plt.show()

    # ── Phase 5: Weight update ────────────────────────────────────────

    def _draw_phase_5(self) -> None:
        fig, (ax_para, ax_info) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]}
        )
        fig.patch.set_facecolor("#f8f9fa")

        target = self._td_target
        v_pred = self._v_pred
        loss = self._td_error**2

        v_range = np.linspace(min(target, v_pred) - 0.4, max(target, v_pred) + 0.4, 200)
        l_range = (v_range - target) ** 2

        ax_para.plot(
            v_range,
            l_range,
            color="#3498db",
            lw=2,
            label="Loss = (V(s) \u2212 target)\u00b2",
        )
        ax_para.plot(
            v_pred,
            loss,
            "o",
            markersize=12,
            color="#e74c3c",
            zorder=5,
            label=f"Current: V(s)={v_pred:+.3f}",
        )
        ax_para.plot(
            target,
            0,
            "*",
            markersize=16,
            color="#2ecc71",
            zorder=5,
            label=f"Target: {target:+.3f}",
        )

        if abs(self._td_error) > 0.001:
            arrow_len = min(abs(self._td_error) * 0.5, 0.2)
            direction = 1 if self._td_error > 0 else -1
            grad = 2 * (v_pred - target)
            ax_para.annotate(
                "",
                xy=(
                    v_pred + direction * arrow_len,
                    loss - abs(grad) * arrow_len * 0.5,
                ),
                xytext=(v_pred, loss),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5),
            )
            ax_para.text(
                v_pred + direction * 0.05,
                loss + 0.02,
                f"push V(s) {'up' if direction > 0 else 'down'}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="#e74c3c",
            )

        ax_para.axhline(0, color="#bdc3c7", lw=0.5)
        ax_para.set_xlabel("V(s)", fontsize=10)
        ax_para.set_ylabel("Loss", fontsize=10)
        ax_para.set_title("MSE Loss Landscape", fontsize=10)
        ax_para.legend(fontsize=8, loc="upper right")

        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        info_lines = [
            ("V(s)", f"{v_pred:+.4f}", "#3498db"),
            ("target", f"{target:+.4f}", "#2ecc71"),
            ("error", f"{self._td_error:+.4f}", "#2c3e50"),
            ("loss", f"{loss:.6f}", "#e74c3c"),
        ]
        y = 0.85
        for label, val, color in info_lines:
            ax_info.text(
                0.05,
                y,
                f"{label}:",
                fontsize=11,
                ha="left",
                va="center",
                color="#7f8c8d",
                transform=ax_info.transAxes,
            )
            ax_info.text(
                0.95,
                y,
                val,
                fontsize=13,
                ha="right",
                va="center",
                color=color,
                fontweight="bold",
                family="monospace",
                transform=ax_info.transAxes,
            )
            y -= 0.15

        ax_info.text(
            0.5,
            0.15,
            "After this update, V(s)\nmoves slightly toward target.\n"
            "The next half-move will\nupdate the OTHER player\u2019s V.",
            fontsize=9,
            ha="center",
            va="center",
            color="#7f8c8d",
            style="italic",
            transform=ax_info.transAxes,
        )

        plt.tight_layout()
        plt.show()

    # ── Phase 6: Summary / next half-move ─────────────────────────────

    def _draw_phase_6(self) -> None:
        fig, (ax_board, ax_table) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.0, 1.6]}
        )
        fig.patch.set_facecolor("#f8f9fa")

        # Show the AFTERSTATE — this is the new "current" board for the next half-move
        at, act, ml = self._s_after_snap
        draw_board(ax_board, at, act, ml, cell_radius=0.38)
        if self._terminal:
            outcome_map = {1.0: "Yellow wins!", -1.0: "Red wins!", 0.0: "Draw!"}
            outcome = outcome_map.get(self._terminal_reward, "?")
            ax_board.set_title(f"Game Over: {outcome}", fontsize=9)
        else:
            next_player = self._player_to_move(ml)
            ax_board.set_title(
                f"State s\u2032 ({next_player} to move next)",
                fontsize=9,
            )

        # History table
        ax_table.axis("off")
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(0, 1)
        ax_table.set_title("Half-Move Update History", fontsize=10, fontweight="bold")

        if self._history:
            col_labels = ["#", "Player", "Col", "V(s)", "Target", "Error"]
            # Show only the most recent 10 to keep the table compact
            recent = self._history[-10:]
            offset = len(self._history) - len(recent)
            table_data = []
            for i, h in enumerate(recent):
                table_data.append([
                    str(offset + i + 1),
                    h["player"][0],  # "Y" / "R"
                    str(h["col"]),
                    f"{h['v_pred']:+.3f}",
                    f"{h['target']:+.3f}",
                    f"{h['error']:+.3f}",
                ])

            table = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)

            for j in range(len(col_labels)):
                table[0, j].set_facecolor("#3498db")
                table[0, j].set_text_props(color="white", fontweight="bold")

            for i, h in enumerate(recent):
                # Color player cell by color
                if h["player"] == "Yellow":
                    table[i + 1, 1].set_facecolor("#fef9e7")
                else:
                    table[i + 1, 1].set_facecolor("#fadbd8")
                # Color error cell by magnitude
                err = h["error"]
                if abs(err) < 0.05:
                    table[i + 1, 5].set_facecolor("#d5f5e3")
                elif abs(err) > 0.2:
                    table[i + 1, 5].set_facecolor("#fadbd8")
        else:
            ax_table.text(
                0.5,
                0.5,
                "No updates recorded yet.",
                ha="center",
                va="center",
                fontsize=11,
                color="#95a5a6",
                transform=ax_table.transAxes,
            )

        plt.tight_layout()
        plt.show()

    # ══════════════════════════════════════════════════════════════════
    # Public
    # ══════════════════════════════════════════════════════════════════

    def show(self) -> None:
        display(
            VBox([
                HTML(
                    "<b>TD Training Step</b> \u2014 each <i>training step</i> is "
                    "<b>one half-move</b>. The same model plays both colors, and "
                    "the value function is updated after every single move "
                    "(Yellow, then Red, then Yellow, ...). Click <b>Next</b> to "
                    "step through the 7 phases of one update; click "
                    "<b>Next Half-Move</b> to apply the move and continue with "
                    "the other player\u2019s training step."
                ),
                HBox([
                    self._btn_prev,
                    self._btn_next,
                    self._btn_continue,
                    self._btn_reset,
                    self._sl_eps,
                ]),
                self._html_phase,
                self._out,
            ])
        )
