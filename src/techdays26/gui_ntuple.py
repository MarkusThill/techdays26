"""Interactive N-Tuple Value Function Visualizer for Connect-4 workshop.

Usage::

    from techdays26.gui_ntuple import NTupleVisualizer

    vis = NTupleVisualizer()
    vis.show()

    # Optionally pre-load a model:
    vis = NTupleVisualizer(model_path="td_weights_clean.tdw.zip")
    vis.show()
"""

from __future__ import annotations

import importlib.resources
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ipywidgets import (
    HTML,
    Button,
    Dropdown,
    HBox,
    Label,
    Layout,
    Output,
    Text,
    ToggleButton,
    VBox,
)

from bitbully import Board

# ── Constants ─────────────────────────────────────────────────────────────────
N_COLS = 7
N_ROWS = 6
STRIDE_STD = 6  # standard index: col * 6 + row_from_bottom
P = 4  # states per cell: 0=empty, 1=yellow, 2=red, 3=reachable

# Colours
C_TUPLE_HL = "#9b59b6"  # selected tuple overlay on board
STATE_COLORS = {0: "#ecf0f1", 1: "#f1c40f", 2: "#e74c3c", 3: "#2ecc71"}
STATE_LABELS = {0: "empty", 1: "yellow", 2: "red", 3: "reachable"}

# ── Demo patterns (4-tuples, standard indices col*6+row, row 0 = bottom) ─────
_DEMO_PATTERNS: dict[str, list[int]] = {
    "Horiz-4 · row 0": [0, 6, 12, 18],
    "Horiz-4 · row 2": [2, 8, 14, 20],
    "Vert-4 · col 3": [18, 19, 20, 21],
    "Vert-4 · col 1": [6, 7, 8, 9],
    "Diag ↗ · center": [7, 14, 21, 28],
    "Diag ↘ · center": [21, 16, 11, 6],
    "L-shape": [0, 1, 6, 12],
    "T-shape": [6, 7, 8, 13],
}


class _Pattern(NamedTuple):
    name: str
    idxs: list[int]  # original cell indices (standard)
    idxs_m: list[int]  # mirror cell indices


def _mirror_std(idxs: list[int]) -> list[int]:
    return [(6 - idx // STRIDE_STD) * STRIDE_STD + idx % STRIDE_STD for idx in idxs]


def _lut_index(states: np.ndarray) -> int:
    powers = P ** np.arange(len(states), dtype=np.int64)
    return int(states.astype(np.int64) @ powers)


def _board_to_flat(board: Board) -> np.ndarray:
    """Convert a bitbully Board to a 42-element P=4 feature array."""
    arr = board.to_array()  # arr[col][row], row 0 = bottom; 0=empty,1=yellow,2=red
    flat = np.zeros(42, dtype=np.int8)
    for col in range(N_COLS):
        reachable_row: int | None = None
        for row in range(N_ROWS):
            if arr[col][row] == 0:
                reachable_row = row
                break
        for row in range(N_ROWS):
            idx = col * STRIDE_STD + row
            cell = arr[col][row]
            if cell == 1:
                flat[idx] = 1
            elif cell == 2:
                flat[idx] = 2
            else:
                flat[idx] = 3 if row == reachable_row else 0
    return flat


# ── Main class ─────────────────────────────────────────────────────────────────
class NTupleVisualizer:
    """Interactive N-tuple value function visualizer for Jupyter notebooks."""

    def __init__(self, model_path: str = "", top_n: int = 20) -> None:
        self._board = Board()
        self._move_history = ""
        self._top_n = top_n
        self._weights = None  # TwoPlayerWeights | None
        self._patterns: list[_Pattern] = [
            _Pattern(name, idxs, _mirror_std(idxs))
            for name, idxs in _DEMO_PATTERNS.items()
        ]
        self._sel_idx = 0
        self._show_mirror = False

        # PNG tiles
        _assets = Path(str(importlib.resources.files("bitbully").joinpath("assets")))
        self._png = {
            0: plt.imread(_assets / "empty.png"),
            1: plt.imread(_assets / "yellow.png"),
            2: plt.imread(_assets / "red.png"),
        }

        self._build_widgets()

        if model_path:
            self._txt_model_path.value = model_path
            self._load_model(model_path)

        self._refresh_all()

    # ── Widget construction ────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self._out_board = Output(layout=Layout(width="auto", overflow="hidden"))
        self._out_detail = Output(layout=Layout(width="auto", overflow="hidden"))
        self._out_lut = Output(layout=Layout(width="auto", overflow="hidden"))
        self._out_value = Output(layout=Layout(width="100%", overflow="hidden"))
        self._html_lut_table = HTML(layout=Layout(padding="4px"))

        # Pattern dropdown
        self._dd_pattern = Dropdown(
            options=[p.name for p in self._patterns],
            value=self._patterns[0].name,
            layout=Layout(width="260px"),
        )
        self._dd_pattern.observe(self._on_pattern_change, names="value")

        # Mirror toggle
        self._btn_mirror = ToggleButton(
            value=False,
            description="⇔ Mirror",
            button_style="info",
            layout=Layout(width="100px"),
        )
        self._btn_mirror.observe(lambda c: self._set_mirror(c["new"]), names="value")

        # Move input
        self._txt_moves = Text(
            value="",
            placeholder="e.g. 3344",
            description="Moves:",
            layout=Layout(width="240px"),
            continuous_update=False,
        )
        self._txt_moves.observe(self._on_moves_change, names="value")

        # Column buttons sized to match board figure
        _fig_tmp = plt.figure(figsize=(N_COLS * 0.5, N_ROWS * 0.5))
        _btn_w = int((_fig_tmp.get_size_inches() * _fig_tmp.dpi)[0] / N_COLS) - 4
        plt.close(_fig_tmp)
        col_buttons = []
        for c in range(N_COLS):
            btn = Button(
                description=f"⏬{c}",
                layout=Layout(width=f"{_btn_w}px", height="32px"),
                button_style="primary",
            )
            btn.on_click(lambda _, col=c: self._on_col_click(col))
            col_buttons.append(btn)
        self._hbox_cols = HBox(
            [VBox(layout=Layout(padding="0px 0px 0px 6px")), *col_buttons],
            layout=Layout(
                display="flex",
                flex_flow="row wrap",
                justify_content="center",
                align_items="center",
            ),
        )

        self._btn_undo = Button(
            description="↩ Undo", button_style="warning", layout=Layout(width="80px")
        )
        self._btn_undo.on_click(self._on_undo)
        self._btn_reset = Button(
            description="🔄 Reset", button_style="danger", layout=Layout(width="80px")
        )
        self._btn_reset.on_click(self._on_reset)

        # Model loader
        self._txt_model_path = Text(
            value="",
            placeholder="path/to/model.pt or .tdw.zip",
            layout=Layout(width="260px"),
        )
        self._btn_load = Button(
            description="📂 Load",
            button_style="",
            layout=Layout(width="80px"),
        )
        self._btn_load.on_click(lambda _: self._load_model(self._txt_model_path.value))
        self._lbl_model = HTML(
            "<span style='color:#7f8c8d;font-family:monospace'>"
            " no model — showing demo 4-tuples with random weights</span>"
        )

    def show(self) -> None:
        """Render the widget in the notebook."""
        controls = HBox(
            [
                self._txt_moves,
                self._btn_undo,
                self._btn_reset,
                HTML("&nbsp;|&nbsp;"),
                Label("Pattern:"),
                self._dd_pattern,
                self._btn_mirror,
            ],
            layout=Layout(align_items="center", gap="6px", flex_wrap="wrap"),
        )
        model_row = HBox(
            [self._txt_model_path, self._btn_load, self._lbl_model],
            layout=Layout(align_items="center", gap="6px"),
        )

        board_panel = VBox([
            HTML("<b>Board</b> <span style='color:#9b59b6'>■</span> = tuple cells"),
            self._hbox_cols,
            self._out_board,
        ])
        detail_panel = VBox(
            [
                HTML("<b>Pattern cells &amp; LUT index</b>"),
                self._out_detail,
                self._html_lut_table,
            ],
            layout=Layout(min_width="240px"),
        )
        lut_panel = VBox(
            [
                HTML("<b>LUT</b>"),
                self._out_lut,
            ],
            layout=Layout(min_width="300px"),
        )

        main_row = HBox(
            [board_panel, detail_panel, lut_panel],
            layout=Layout(gap="16px", align_items="flex-start"),
        )
        value_panel = VBox([
            HTML(
                f"<b>Value decomposition</b>"
                f" <span style='font-weight:normal;font-size:11px'>"
                f"top {self._top_n} by |contribution|"
                f" — selected pattern outlined</span>"
            ),
            self._out_value,
        ])

        display(
            VBox(
                [
                    HTML("<h2>🔢 N-Tuple Network — Value Function Visualizer</h2>"),
                    controls,
                    model_row,
                    main_row,
                    value_panel,
                ],
                layout=Layout(padding="10px"),
            )
        )

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_col_click(self, col: int) -> None:
        if not self._board.is_game_over() and self._board.native.isLegalMove(col):
            self._board.play(col)
            self._move_history += str(col)
            self._txt_moves.value = self._move_history
            self._refresh_all()

    def _on_undo(self, _=None) -> None:
        if self._move_history:
            self._move_history = self._move_history[:-1]
            self._board = Board(self._move_history) if self._move_history else Board()
            self._txt_moves.value = self._move_history
            self._refresh_all()

    def _on_reset(self, _=None) -> None:
        self._move_history = ""
        self._board = Board()
        self._txt_moves.value = ""
        self._refresh_all()

    def _on_moves_change(self, change) -> None:
        val = change["new"].strip()
        try:
            self._board = Board(val) if val else Board()
            self._move_history = val
            self._refresh_all()
        except Exception:
            pass

    def _on_pattern_change(self, change) -> None:
        name = change["new"]
        self._sel_idx = next(i for i, p in enumerate(self._patterns) if p.name == name)
        self._refresh_all()

    def _set_mirror(self, val: bool) -> None:
        self._show_mirror = val
        self._refresh_all()

    # ── Model loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _bitidx_to_std(bit_idx: int, col_height: int = 6, stride: int = 9) -> int:
        """Convert a single bitboard index (col*9+row) to standard index (col*6+row)."""
        col = bit_idx // stride
        row = bit_idx % stride
        return col * col_height + row

    def _load_pt_model(self, path: str) -> None:
        """Load a .pt NTupleNetwork and convert to TwoPlayerWeights."""
        from techdays26.legacy_ntuple_agent import (
            TupleLUT,
            PlayerWeights,
            TwoPlayerWeights,
        )
        from techdays26.ntuple_network import NTupleNetwork

        model = NTupleNetwork.load(path, device="cpu")
        model.eval()

        M, N, K = model.M, model.N, model.K
        bit_patterns = model.n_tuple_tensor.cpu().tolist()  # [M, N]
        W = model.W.detach().cpu().numpy()  # [2, M, K]

        def _make_luts(player: int) -> tuple[TupleLUT, ...]:
            luts = []
            for m, bit_idxs in enumerate(bit_patterns):
                std_idxs = np.array(
                    [self._bitidx_to_std(b) for b in bit_idxs], dtype=np.int32
                )
                std_idxs_m = np.array(_mirror_std(std_idxs.tolist()), dtype=np.int32)
                luts.append(
                    TupleLUT(
                        n=N,
                        m=m,
                        idxs=std_idxs,
                        idxs_m=std_idxs_m,
                        lut=W[player, m, :].astype(np.float64),
                    )
                )
            return tuple(luts)

        weights = TwoPlayerWeights(
            p0=PlayerWeights(t=M, p=P, luts=_make_luts(0)),
            p1=PlayerWeights(t=M, p=P, luts=_make_luts(1)),
        )
        self._weights = weights
        self._patterns = [
            _Pattern(
                f"#{i:03d}  (n={lut.n})",
                list(map(int, lut.idxs)),
                list(map(int, lut.idxs_m)),
            )
            for i, lut in enumerate(weights.p0.luts)
        ]
        self._sel_idx = 0
        self._dd_pattern.options = [p.name for p in self._patterns]
        self._dd_pattern.value = self._patterns[0].name
        self._lbl_model.value = (
            f"<span style='color:#27ae60;font-family:monospace'>"
            f" ✓ {Path(path).name}  ({M} patterns, n={N}, "
            f"LUT size={K:,})</span>"
        )
        self._refresh_all()

    def _load_model(self, path: str) -> None:
        try:
            if path.endswith(".pt"):
                self._load_pt_model(path)
                return

            from techdays26.legacy_ntuple_agent import import_two_player_weights_zip

            weights = import_two_player_weights_zip(path, int_dtype=np.int32)
            self._weights = weights
            t, n = weights.p0.t, weights.p0.luts[0].n
            self._patterns = [
                _Pattern(
                    f"#{i:03d}  (n={lut.n})",
                    list(map(int, lut.idxs)),
                    list(map(int, lut.idxs_m)),
                )
                for i, lut in enumerate(weights.p0.luts)
            ]
            self._sel_idx = 0
            self._dd_pattern.options = [p.name for p in self._patterns]
            self._dd_pattern.value = self._patterns[0].name
            self._lbl_model.value = (
                f"<span style='color:#27ae60;font-family:monospace'>"
                f" ✓ {Path(path).name}  ({t} patterns, n={n}, "
                f"LUT size={P**n:,})</span>"
            )
            self._refresh_all()
        except Exception as exc:
            self._lbl_model.value = (
                f"<span style='color:#e74c3c;font-family:monospace'> ✗ {exc}</span>"
            )

    # ── Core helpers ───────────────────────────────────────────────────────────

    def _flat(self) -> np.ndarray:
        return _board_to_flat(self._board)

    def _player(self) -> int:
        """0 = yellow to move, 1 = red to move."""
        return len(self._move_history) % 2

    def _active_idxs(self) -> list[int]:
        pat = self._patterns[self._sel_idx]
        return pat.idxs_m if self._show_mirror else pat.idxs

    def _lut_for_sel(self) -> np.ndarray | None:
        """Return the LUT array for the selected pattern and current player, or None."""
        if self._weights is None:
            return None
        pw = self._weights.for_player(self._player())
        if self._sel_idx >= len(pw.luts):
            return None
        return pw.luts[self._sel_idx].lut

    def _demo_lut(self, n: int, seed: int) -> np.ndarray:
        """Random-init LUT for demo mode."""
        return np.random.default_rng(seed).normal(0, 0.15, P**n)

    # ── Master refresh ─────────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        flat = self._flat()
        act_idxs = self._active_idxs()
        states = flat[act_idxs].astype(np.int8)
        lut_idx = _lut_index(states)

        self._draw_board(act_idxs)
        self._draw_detail(flat, act_idxs, states, lut_idx)
        self._draw_lut(states, lut_idx)
        self._draw_value_chart(flat)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_board(self, highlight_idxs: list[int]) -> None:
        """6×7 board with PNG tiles; selected tuple cells tinted purple."""
        self._out_board.clear_output(wait=True)
        h_set = set(highlight_idxs)
        arr = self._board.to_array()

        with self._out_board:
            fig, axs = plt.subplots(
                N_ROWS, N_COLS, figsize=(N_COLS * 0.5, N_ROWS * 0.5)
            )
            for r_disp in range(N_ROWS):
                row = N_ROWS - 1 - r_disp  # display top → board top (row 5)
                for col in range(N_COLS):
                    ax = axs[r_disp][col]
                    ax.imshow(self._png[arr[col][row]])
                    std_idx = col * STRIDE_STD + row
                    if std_idx in h_set:
                        rect = mpatches.FancyBboxPatch(
                            (0.05, 0.05),
                            0.9,
                            0.9,
                            boxstyle="round,pad=0.05",
                            facecolor=C_TUPLE_HL,
                            alpha=0.55,
                            edgecolor="white",
                            linewidth=1.5,
                            transform=ax.transAxes,
                        )
                        ax.add_patch(rect)
                        ax.text(
                            0.5,
                            0.5,
                            str(std_idx),
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="white",
                            fontweight="bold",
                            transform=ax.transAxes,
                        )
                    ax.axis("off")
            plt.subplots_adjust(
                wspace=0.05, hspace=0.05, left=0, right=1, top=1, bottom=0
            )
            display(fig)
            plt.close(fig)

    def _draw_detail(
        self,
        flat: np.ndarray,
        act_idxs: list[int],
        states: np.ndarray,
        lut_idx: int,
    ) -> None:
        """Isolated 6×7 grid showing only the N tuple cells, coloured by state."""
        self._out_detail.clear_output(wait=True)
        h_set = set(act_idxs)

        with self._out_detail:
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            fig.patch.set_facecolor("#f8f9fa")
            ax.set_facecolor("#e8e8e8")

            for col in range(N_COLS):
                for row in range(N_ROWS):
                    std_idx = col * STRIDE_STD + row
                    in_tuple = std_idx in h_set
                    state = int(flat[std_idx])

                    fc = STATE_COLORS[state] if in_tuple else "#c8c8c8"
                    ec = C_TUPLE_HL if in_tuple else "#aaa"
                    lw = 2.0 if in_tuple else 0.3

                    rect = mpatches.FancyBboxPatch(
                        (col + 0.05, row + 0.05),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0.04",
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=lw,
                    )
                    ax.add_patch(rect)
                    if in_tuple:
                        ax.text(
                            col + 0.5,
                            row + 0.5,
                            str(std_idx),
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="#2c3e50",
                        )

            ax.set_xlim(0, N_COLS)
            ax.set_ylim(0, N_ROWS)
            ax.set_xticks(np.arange(N_COLS) + 0.5)
            ax.set_xticklabels([f"C{c}" for c in range(N_COLS)], fontsize=7)
            ax.set_yticks(np.arange(N_ROWS) + 0.5)
            ax.set_yticklabels([f"R{r}" for r in range(N_ROWS)], fontsize=7)
            ax.tick_params(length=0)
            ax.set_aspect("equal")

            legend_patches = [
                mpatches.Patch(
                    facecolor=STATE_COLORS[s],
                    label=STATE_LABELS[s],
                    edgecolor="#555",
                    linewidth=0.5,
                )
                for s in range(P)
            ]
            ax.legend(
                handles=legend_patches,
                loc="upper right",
                fontsize=6,
                framealpha=0.85,
                ncol=2,
            )
            fig.tight_layout(pad=0.4)
            display(fig)
            plt.close(fig)

        # ── LUT index breakdown table ──────────────────────────────────────────
        rows_html = ""
        running = 0
        for i, (cell_idx, s) in enumerate(zip(act_idxs, states)):
            col, row = cell_idx // STRIDE_STD, cell_idx % STRIDE_STD
            contrib = int(s) * (P**i)
            running += contrib
            fc = STATE_COLORS[int(s)]
            rows_html += (
                f"<tr>"
                f"<td style='padding:2px 5px;text-align:center'>{i}</td>"
                f"<td style='padding:2px 5px;text-align:center;font-weight:bold'>{cell_idx}</td>"
                f"<td style='padding:2px 5px;text-align:center'>C{col}/R{row}</td>"
                f"<td style='padding:2px 5px;text-align:center;"
                f"background:{fc};border-radius:3px'>{STATE_LABELS[int(s)]}</td>"
                f"<td style='padding:2px 5px;text-align:right'>{int(s)}</td>"
                f"<td style='padding:2px 5px'>×{P}<sup>{i}</sup></td>"
                f"<td style='padding:2px 5px;text-align:right'>{contrib}</td>"
                f"</tr>"
            )

        # Look up the actual weight if model is loaded
        weight_html = ""
        lut = self._lut_for_sel()
        if lut is not None:
            w = float(lut[lut_idx])
            weight_html = (
                f"<tr style='background:#d5f5e3'>"
                f"<td colspan='6' style='padding:3px 5px;text-align:right'>"
                f"W[{lut_idx}] =</td>"
                f"<td style='padding:3px 5px;font-weight:bold'>{w:+.5f}</td>"
                f"</tr>"
            )

        self._html_lut_table.value = (
            "<div style='font-family:monospace;font-size:11px;"
            "background:#f8f9fa;padding:6px;border-radius:6px;"
            "border:1px solid #dee2e6'>"
            "<table style='border-collapse:collapse;width:100%'>"
            "<tr style='background:#e9ecef;font-size:10px'>"
            "<th style='padding:2px 5px'>#</th>"
            "<th style='padding:2px 5px'>cell</th>"
            "<th style='padding:2px 5px'>pos</th>"
            "<th style='padding:2px 5px'>state</th>"
            "<th style='padding:2px 5px'>val</th>"
            "<th style='padding:2px 5px'></th>"
            "<th style='padding:2px 5px'>contrib</th>"
            "</tr>"
            f"{rows_html}"
            f"<tr style='background:#fff3cd;font-weight:bold'>"
            f"<td colspan='6' style='padding:3px 5px;text-align:right'>"
            f"LUT index =</td>"
            f"<td style='padding:3px 5px'>{lut_idx}</td>"
            f"</tr>"
            f"{weight_html}"
            "</table></div>"
        )

    def _draw_lut(self, states: np.ndarray, current_idx: int) -> None:
        """LUT heatmap (≤5 cells) or neighbourhood bar chart (>5 cells)."""
        self._out_lut.clear_output(wait=True)
        n = len(states)
        lut_size = P**n

        lut = self._lut_for_sel()
        if lut is None:
            lut = self._demo_lut(n, seed=self._sel_idx)

        w_cur = float(lut[current_idx])
        player_label = "Yellow" if self._player() == 0 else "Red"

        with self._out_lut:
            if n <= 5:
                # ── Full heatmap ───────────────────────────────────────────────
                side = int(round(lut_size**0.5))
                grid = lut[: side * side].reshape(side, side)
                vabs = max(abs(lut.min()), abs(lut.max()), 1e-9)

                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                fig.patch.set_facecolor("#f8f9fa")
                im = ax.imshow(
                    grid,
                    cmap="RdYlGn",
                    vmin=-vabs,
                    vmax=vabs,
                    aspect="auto",
                    interpolation="nearest",
                )
                plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

                # Highlight current index
                r_h, c_h = divmod(current_idx, side)
                ax.add_patch(
                    mpatches.Rectangle(
                        (c_h - 0.5, r_h - 0.5),
                        1,
                        1,
                        linewidth=2.5,
                        edgecolor="black",
                        facecolor="none",
                    )
                )
                ax.set_title(
                    f"{lut_size} entries  [{player_label}]\n"
                    f"idx={current_idx}   W={w_cur:+.5f}",
                    fontsize=8,
                    pad=4,
                )
                ax.axis("off")
                fig.tight_layout(pad=0.4)
                display(fig)
                plt.close(fig)

            else:
                # ── Neighbourhood bar ──────────────────────────────────────────
                half = 64
                lo = max(0, current_idx - half)
                hi = min(lut_size, current_idx + half + 1)
                xs = np.arange(lo, hi)
                ys = lut[lo:hi]
                colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in ys]

                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                fig.patch.set_facecolor("#f8f9fa")
                ax.bar(xs, ys, color=colors, width=1.0)
                ax.axvline(
                    current_idx,
                    color="black",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"idx={current_idx}",
                )
                ax.axhline(0, color="#555", linewidth=0.5)
                ax.set_title(
                    f"{lut_size:,} entries  [{player_label}]\n"
                    f"showing ±{half} around idx={current_idx}   W={w_cur:+.5f}",
                    fontsize=8,
                    pad=4,
                )
                ax.set_xlabel("LUT index", fontsize=7)
                ax.tick_params(labelsize=7)
                fig.tight_layout(pad=0.4)
                display(fig)
                plt.close(fig)

    def _draw_value_chart(self, flat: np.ndarray) -> None:
        """Horizontal bar chart: top-N pattern contributions, selected outlined."""
        self._out_value.clear_output(wait=True)
        player = self._player()

        # Gather (name, contribution) for every pattern
        contribs: list[tuple[str, float]] = []
        if self._weights is not None:
            pw = self._weights.for_player(player)
            for i, (pat, tup) in enumerate(zip(self._patterns, pw.luts)):
                s1 = _lut_index(flat[pat.idxs])
                s2 = _lut_index(flat[pat.idxs_m])
                contribs.append((pat.name, float(tup.lut[s1]) + float(tup.lut[s2])))
        else:
            for i, pat in enumerate(self._patterns):
                n = len(pat.idxs)
                lut = self._demo_lut(n, seed=i)
                s1 = _lut_index(flat[pat.idxs])
                s2 = _lut_index(flat[pat.idxs_m])
                contribs.append((pat.name, float(lut[s1]) + float(lut[s2])))

        if not contribs:
            return

        # Top-N by |contribution|; always include selected pattern
        order = sorted(range(len(contribs)), key=lambda i: -abs(contribs[i][1]))
        top = order[: self._top_n]
        if self._sel_idx not in top:
            top.append(self._sel_idx)

        names = [contribs[i][0] for i in top]
        vals = [contribs[i][1] for i in top]
        colors = ["#3498db" if v >= 0 else "#e74c3c" for v in vals]
        ec = ["black" if i == self._sel_idx else "none" for i in top]
        lw = [1.8 if i == self._sel_idx else 0.0 for i in top]

        raw_total = sum(c for _, c in contribs)
        value = math.tanh(raw_total)
        player_label = "Yellow" if player == 0 else "Red"

        with self._out_value:
            h = max(3.0, len(names) * 0.30 + 1.2)
            fig, ax = plt.subplots(figsize=(9, h))
            fig.patch.set_facecolor("#f8f9fa")
            ax.set_facecolor("#ffffff")

            y = np.arange(len(names))
            bars = ax.barh(y, vals, color=colors, edgecolor=ec, linewidth=lw)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=7)
            ax.axvline(0, color="#555", linewidth=0.8)
            ax.set_xlabel("LUT contribution  (original + mirror)", fontsize=8)
            ax.set_title(
                f"Raw sum = {raw_total:.4f}   →   "
                f"tanh({raw_total:.4f}) = {value:.4f}   "
                f"[{player_label} to move]",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)

            for bar, v in zip(bars, vals):
                xoff = 0.003 if v >= 0 else -0.003
                ha = "left" if v >= 0 else "right"
                ax.text(
                    v + xoff,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:+.3f}",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    color="#2c3e50",
                )

            fig.tight_layout(pad=0.5)
            display(fig)
            plt.close(fig)
