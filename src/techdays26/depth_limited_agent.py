"""Depth-limited wrapper for BitBully agents.

Allows creating multiple arena agents with different search depths while
sharing a single BitBully backend (and its opening book / transposition table).
"""

from __future__ import annotations

from bitbully import Board
from bitbully.agent_interface import Connect4Agent


class DepthLimitedAgent:
    """Wrapper that delegates to a BitBully backend but limits search depth.

    Args:
        backend: A ``BitBully`` instance used for search.
        depth: Depth parameter forwarded to ``backend.negamax(…, depth=depth)``.
            ``0`` means full-depth (strongest); higher values reduce search depth.
    """

    def __init__(self, backend: Connect4Agent, depth: int) -> None:
        self._backend = backend
        self._depth = depth

    def score_all_moves(self, board: Board) -> dict[int, int]:
        scores: dict[int, int] = {}
        for col in board.legal_moves():
            child = board.copy()
            child.play(col)
            # negamax returns the score for the side to move in `child`,
            # which is the opponent — negate to get the score for *us*.
            scores[col] = -self._backend.negamax(child, depth=self._depth)
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def best_move(self, board: Board) -> int:
        scores = self.score_all_moves(board)
        if not scores:
            raise ValueError("No legal moves available.")
        best_score = max(scores.values())
        best_cols = [c for c, s in scores.items() if s == best_score]
        # Center-preference tie-break (matches BitBully default).
        return min(best_cols, key=lambda c: (abs(c - 3), c))
