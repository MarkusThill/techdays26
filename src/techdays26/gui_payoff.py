"""Training-payoff widget — checkpoint progress plot.

Usage::

    from techdays26.gui_payoff import plot_checkpoint_progress

    plot_checkpoint_progress("../exp_20260228_13-46/repeat_0")

To rematch the trained N-Tuple agent via ``GuiC4``, use the existing
``TDConnect4AgentTorch`` from :mod:`techdays26.td_agent` — no wrapper needed:

    from bitbully.gui_c4 import GuiC4
    from techdays26.td_agent import TDConnect4AgentTorch

    agent = TDConnect4AgentTorch(
        model_path="../exp_20260228_13-46/repeat_0/step_500_model_weights.pt",
    )
    GuiC4(agents={"N-Tuple TD(λ) (trained)": agent}, autoplay=True).get_widget()
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from techdays26.ntuple_network import NTupleNetwork
from techdays26.ntuples import NTUPLE_BITIDX_LIST_200
from techdays26.torch_board import BoardBatch

# ─── Checkpoint progress plot ────────────────────────────────────────


def _read_arena(path: Path) -> dict[str, Counter]:
    """Tally W/D/L per matchup from one arena_result.json (ntuple POV)."""
    with Path(path).open() as f:
        d = json.load(f)
    tally: dict[str, Counter] = {}
    for g in d["result"]["games"]:
        cfg = g["game_cfg"]["players"]
        y, r = cfg["yellow_id"], cfg["red_id"]
        winner = g.get("winner")  # 0=draw, 1=yellow, 2=red
        ntuple_side = 1 if y == "ntuple" else (2 if r == "ntuple" else None)
        if ntuple_side is None:
            continue
        opp = r if ntuple_side == 1 else y
        side_key = "yellow" if ntuple_side == 1 else "red"
        key = f"{opp}:{side_key}"
        t = tally.setdefault(key, Counter())
        if winner == 0 or winner is None:
            t["D"] += 1
        elif winner == ntuple_side:
            t["W"] += 1
        else:
            t["L"] += 1
    return tally


def plot_checkpoint_progress(run_dir: str) -> None:
    """Draw win rate + V(empty) across training checkpoints."""
    run = Path(run_dir)
    ckpts = sorted(
        int(p.stem.split("_")[1]) for p in run.glob("step_*_model_weights.pt")
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    # ─ V(empty) progression ─
    empty = BoardBatch.empty(1, device="cpu")
    v_empty: list[float] = []
    untrained = NTupleNetwork(NTUPLE_BITIDX_LIST_200)
    untrained.eval()
    with torch.no_grad():
        v0 = float(untrained(empty)[0])
    v_empty.append(v0)
    x_v = [0] + ckpts
    for step in ckpts:
        net = NTupleNetwork.load(
            str(run / f"step_{step}_model_weights.pt"), device="cpu"
        )
        net.eval()
        with torch.no_grad():
            v_empty.append(float(net(empty)[0]))

    # ─ Arena win rates ─
    arena = {}
    for step in ckpts:
        p = run / f"step_{step}_arena_result.json"
        if p.exists():
            arena[step] = _read_arena(p)

    # Aggregate per opponent, combining yellow+red sides:
    opponents: set[str] = set()
    for t in arena.values():
        opponents.update(key.split(":")[0] for key in t)

    def winrate(step: int, opp: str) -> float | None:
        t = arena.get(step, {})
        w = t.get(f"{opp}:yellow", Counter())
        r = t.get(f"{opp}:red", Counter())
        total = sum(w.values()) + sum(r.values())
        if total == 0:
            return None
        wins = w["W"] + r["W"]
        draws = w["D"] + r["D"]
        return (wins + 0.5 * draws) / total

    # ─ Plot ─
    fig, (ax_v, ax_wr) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor("#f8f9fa")

    ax_v.plot(x_v, v_empty, "o-", color="#2c3e50", lw=2)
    ax_v.axhline(0, color="#bdc3c7", lw=0.5)
    ax_v.set_xlabel("Training step")
    ax_v.set_ylabel("V(empty board)")
    ax_v.set_title(
        "Value of the opening position\n(Yellow-positive: >0 = Yellow edge)",
        fontsize=9,
    )
    ax_v.set_ylim(-0.5, 0.5)
    ax_v.grid(alpha=0.3)

    colors = {"random": "#95a5a6", "bitbully-2ply": "#e74c3c"}
    markers = {"random": "s", "bitbully-2ply": "o"}
    for opp in sorted(opponents):
        ys = [winrate(s, opp) for s in ckpts]
        mask = [y is not None for y in ys]
        xs_ = [s for s, m in zip(ckpts, mask) if m]
        ys_ = [y for y, m in zip(ys, mask) if m]
        if not ys_:
            continue
        ax_wr.plot(
            xs_,
            ys_,
            marker=markers.get(opp, "o"),
            label=f"vs {opp}",
            color=colors.get(opp, "#3498db"),
            lw=2,
            markersize=7,
        )
    ax_wr.axhline(
        0.5, color="#bdc3c7", lw=0.8, linestyle="--", label="50% (break-even)"
    )
    ax_wr.set_xlabel("Training step")
    ax_wr.set_ylabel("Win rate (win + 0.5·draw)")
    ax_wr.set_title(
        "Trained N-Tuple vs. baselines\n(averaged over Yellow + Red sides)", fontsize=9
    )
    ax_wr.set_ylim(-0.05, 1.05)
    ax_wr.legend(fontsize=8, loc="lower right")
    ax_wr.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Textual summary
    if 500 in arena:
        final = arena[500]
        print("\nFinal (step 500) arena results from the ntuple's perspective:")
        for opp in sorted(opponents):
            w = final.get(f"{opp}:yellow", Counter())
            r = final.get(f"{opp}:red", Counter())
            total = sum(w.values()) + sum(r.values())
            wins = w["W"] + r["W"]
            draws = w["D"] + r["D"]
            losses = w["L"] + r["L"]
            rate = (wins + 0.5 * draws) / total if total else 0
            print(
                f"  vs {opp:<16} {wins:>3}W / {draws:>3}D / {losses:>3}L "
                f"({total} games)  →  {rate:.1%}"
            )


# ─── NTuple-as-Connect4Agent wrapper for rematch ─────────────────────

try:
    _BITBULLY_OK = True
except Exception:
    _BITBULLY_OK = False


def _bbboard_to_batch(board) -> BoardBatch:
    """Convert a bitbully `Board` to our `BoardBatch` with batch size 1."""
    arr = np.array(board.to_array())  # (7, 6), 0=empty, 1=yellow, 2=red
    yellow_bb = 0
    red_bb = 0
    for col in range(7):
        for row in range(6):
            v = int(arr[col, row])
            bit = col * 9 + row
            if v == 1:
                yellow_bb |= 1 << bit
            elif v == 2:
                red_bb |= 1 << bit
    all_tokens = yellow_bb | red_bb
    tokens_placed = bin(all_tokens).count("1")
    moves_left = 42 - tokens_placed
    yellow_to_move = (moves_left % 2) == 0
    active_tokens = yellow_bb if yellow_to_move else red_bb
    return BoardBatch(
        all_tokens=torch.tensor([all_tokens], dtype=torch.int64),
        active_tokens=torch.tensor([active_tokens], dtype=torch.int64),
        moves_left=torch.tensor([moves_left], dtype=torch.int16),
    )


class NTupleC4Agent:
    """Wrap an `NTupleNetwork` as a `Connect4Agent` usable by `GuiC4`.

    Scoring convention: `GuiC4` expects integer scores where *larger is better*
    for the side to move. We multiply V(afterstate) by 10,000 and flip the sign
    for Red (since V is Yellow-positive and Red wants to minimise it).
    """

    def __init__(self, model_path: str, *, scale: int = 10000) -> None:
        if not _BITBULLY_OK:
            raise ImportError("bitbully is required to use NTupleC4Agent with GuiC4")
        self._net = NTupleNetwork.load(model_path, device="cpu")
        self._net.eval()
        self._scale = scale

    def _legal_cols(self, board) -> list[int]:
        return list(board.legal_moves())

    def _score_col(self, board, col: int) -> int:
        """Play `col` on a copy of `board` and return a GuiC4-compatible score."""
        tmp_board = board.play_on_copy(col)
        # If this move wins, return max score
        if tmp_board.has_win():
            # The player who just moved won — that's the side to move on `board`
            return self._scale * 2  # very large positive
        if tmp_board.is_full():
            return 0  # draw
        batch = _bbboard_to_batch(tmp_board)
        with torch.no_grad():
            v = float(self._net(batch)[0])  # Yellow-positive
        # Flip for Red, since Red wants MIN V but score must be "larger is better"
        # for the side to move on `board`.
        yellow_to_move = (int(board.moves_left()) % 2) == 0
        signed = v if yellow_to_move else -v
        return int(round(signed * self._scale))

    # ─ Connect4Agent protocol ─
    def score_all_moves(self, board) -> dict[int, int]:
        return {c: self._score_col(board, c) for c in self._legal_cols(board)}

    def best_move(self, board) -> int:
        scores = self.score_all_moves(board)
        if not scores:
            raise ValueError("No legal moves")
        # Tie-break: prefer central columns
        best = max(
            scores.items(),
            key=lambda kv: (kv[1], -abs(3 - kv[0])),
        )
        return best[0]
