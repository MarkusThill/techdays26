from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from techdays26.ntuple_network import NTupleNetwork
    from techdays26.torch_board import BoardBatch


class TrainingLogger:
    """Periodic console logging and arena evaluation for the TD training loop.

    Args:
        repeat_dir:   Output directory for this repeat's files.
        n_evaluate:   Run arena evaluation every this many steps.
        n_truncate:   TD lookahead horizon (stored for reference).
        n_repeats:    Total number of repeats (used to prefix log lines).
        repeat_idx:   Index of the current repeat (0-based).
        evaluate_fn:  Callable that accepts either a weights path (str) or a
                      deep-copied CPU NTupleNetwork and returns an ArenaResult.
                      Signature: ``evaluate_fn(weights_path_or_net)``.
        save_weights: If True (default), save model weights to disk at each
                      evaluation step and pass the file path to evaluate_fn.
                      If False, pass a deep-copied CPU snapshot of the network
                      directly — no disk I/O.
        save_detailed_arena_results: If True (default), write the full per-step
                      ``step_<N>_arena_result.json`` files containing complete
                      per-game match details. The aggregated scores in
                      ``0_arena_metrics.json`` (used by ``2_plot_metrics.ipynb``)
                      are always written regardless of this flag.
    """

    def __init__(
        self,
        repeat_dir: Path,
        n_evaluate: int,
        n_truncate: int,
        n_repeats: int,
        repeat_idx: int,
        evaluate_fn: Callable[[str | NTupleNetwork], object],
        *,
        save_weights: bool = True,
        save_snapshot_steps: list[int] | None = None,
        save_detailed_arena_results: bool = True,
    ) -> None:
        self._dir = repeat_dir
        self._n_eval = n_evaluate
        self._log_every = n_evaluate // 10
        self._n_repeats = n_repeats
        self._ri = repeat_idx
        self._eval_fn = evaluate_fn
        self._save_weights = save_weights
        self._snapshot_steps = set(save_snapshot_steps or [])
        self._save_detailed_arena_results = save_detailed_arena_results

        self._metrics_path = repeat_dir / "0_metrics.json"
        self._arena_path = repeat_dir / "0_arena_metrics.json"
        self._log_path = repeat_dir / "0_log.txt"
        self._all_metrics: list[dict] = []
        self._all_arena: list[dict] = []
        self._t0 = time.time()
        self._last_eval_t = time.time()

        # weight-delta state (captured just before opt.step)
        self._W_snap: torch.Tensor | None = None
        self._W_mask: torch.Tensor | None = None
        self._W_norm: float = 0.0

    def snapshot_weights(self, net: torch.nn.Module, step: int) -> None:
        """Capture a weight snapshot just before ``opt.step()``.

        Call this every step; the snapshot is only taken at logging intervals
        so the overhead is negligible outside those steps.
        """
        if step % self._log_every == 0:
            with torch.no_grad():
                mask = net.W != 0.0
                snap = net.W[mask].data.clone()
                self._W_snap, self._W_mask, self._W_norm = (
                    snap,
                    mask,
                    snap.norm().item(),
                )

    def __call__(
        self,
        step: int,
        net: torch.nn.Module,
        opt: torch.optim.Optimizer,
        loss: torch.Tensor,
        done: torch.Tensor,
        randomize: torch.Tensor,
        update_mask: torch.Tensor,
        V_pred: torch.Tensor,
        board: BoardBatch,
    ) -> None:
        pfx = f"[R{self._ri}] " if self._n_repeats > 1 else ""

        # ── Console + JSON metrics every log_every steps ──────────────────
        if step % self._log_every == 0:
            now = time.time()
            elapsed = now - self._t0
            lr = opt.param_groups[0]["lr"]

            rel_update = dW_norm = 0.0
            if self._W_snap is not None:
                with torch.no_grad():
                    dW = net.W[self._W_mask].data - self._W_snap
                    dW_norm = dW.norm().item()
                    rel_update = (
                        dW_norm / self._W_norm if self._W_norm > 0 else float("inf")
                    )

            grad = net.W.grad
            grad_nz = grad[grad != 0.0] if grad is not None else torch.zeros(0)
            mv_left = board.moves_left

            print(
                f"{pfx}step {step:6d} | {self._fmt(elapsed)} | lr={lr:.3e} | "
                f"loss={loss.item():.5f} | ||ΔW||/||W||={rel_update:.3e} | "
                f"V={V_pred.mean().item():.3f}±{V_pred.std().item():.3f} | "
                f"done={done.float().mean():.2f} | "
                f"moves_left={mv_left.float().mean().item():.1f}"
            )

            m = {
                "step": step,
                "training_elapsed_s": elapsed,
                "training_elapsed": self._fmt(elapsed),
                "wall_time_s": now - self._last_eval_t,
                "lr": lr,
                "loss": loss.item(),
                "rel_weight_update": rel_update,
                "delta_W_norm": dW_norm,
                "W_norm": self._W_norm,
                "V_old_min": V_pred.min().item(),
                "V_old_max": V_pred.max().item(),
                "V_old_mean": V_pred.mean().item(),
                "V_old_std": V_pred.std().item(),
                "V_old_abs_mean": V_pred.abs().mean().item(),
                "grad_nnz": int(grad_nz.shape[0]),
                "grad_mean": grad_nz.mean().item() if grad_nz.numel() > 0 else 0.0,
                "grad_std": grad_nz.std().item() if grad_nz.numel() > 1 else 0.0,
                "update_frac": float(update_mask.float().mean()),
                "done_frac": float(done.float().mean()),
                "randomize_frac": float(randomize.float().mean()),
                "n_wins": int(board.has_win().sum().item()),
                "moves_left_mean": float(mv_left.float().mean()),
                "moves_left_std": float(mv_left.float().std()),
            }
            self._all_metrics.append(m)
            self._safe_write(self._metrics_path, self._all_metrics)

            with self._log_path.open("a") as f:
                sep = "=" * 80
                f.write(
                    f"{sep}\nStep {step:6d} | {m['training_elapsed']} | "
                    f"lr={m['lr']:.3e} | loss={m['loss']:.5f}\n"
                    f"  V: {m['V_old_mean']:.3f}±{m['V_old_std']:.3f}  "
                    f"||ΔW||/||W||={m['rel_weight_update']:.3e}\n"
                    f"  grad_nnz={m['grad_nnz']}  done={m['done_frac']:.2f}  "
                    f"rand={m['randomize_frac']:.2f}\n{sep}\n\n"
                )

        # ── Save model snapshot at user-specified steps ──────────────────
        if step in self._snapshot_steps:
            snap = copy.deepcopy(net).cpu()
            snap.eval()
            snap_path = self._dir / f"step_{step}_snapshot.pt"
            snap.save(str(snap_path))
            print(f"{pfx}snapshot saved → {snap_path}")

        # ── Arena evaluation every n_evaluate steps ───────────────────────
        if step % self._n_eval == 0:
            print(f"{pfx}evaluate at step {step}...")
            # Deep-copy onto CPU before any disk or arena work so the training
            # tensors on the original device are never touched by the arena.
            net_snapshot = copy.deepcopy(net).cpu()
            net_snapshot.eval()
            if self._save_weights:
                weights_path = str(self._dir / f"step_{step}_model_weights.pt")
                net_snapshot.save(weights_path)
                eval_arg: str | NTupleNetwork = weights_path
            else:
                eval_arg = net_snapshot
            result = self._eval_fn(eval_arg)

            if self._save_detailed_arena_results:
                result.save_json(str(self._dir / f"step_{step}_arena_result.json"))
            self._all_arena.append({
                "step": step,
                "training_elapsed_s": time.time() - self._t0,
                "aggregates": self._result_rows(result),
            })
            self._safe_write(self._arena_path, self._all_arena)

            from techdays26.bitbully_arena import format_aggregate_table

            with self._log_path.open("a") as f:
                f.write(
                    "=" * 90 + "\n" + format_aggregate_table(result) + "=" * 90 + "\n\n"
                )

            self._last_eval_t = time.time()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(s: float) -> str:
        h, r = divmod(int(s), 3600)
        m, sec = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    @staticmethod
    def _result_rows(result) -> list[dict]:
        rows = []
        for r in result.aggregates:
            score, games = int(r.yellow_wins) - int(r.red_wins), int(r.games)
            rows.append({
                "agent_yellow": r.agent_yellow,
                "agent_red": r.agent_red,
                "epsilon_yellow": float(r.epsilon_yellow),
                "epsilon_red": float(r.epsilon_red),
                "games": games,
                "yellow_wins": int(r.yellow_wins),
                "red_wins": int(r.red_wins),
                "draws": int(r.draws),
                "score": score,
                "avg": (score / games) if games else 0.0,
            })
        return rows

    @staticmethod
    def _safe_write(path: Path, data: list) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(path)
