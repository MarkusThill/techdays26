from __future__ import annotations

import torch

from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


class TDConnect4AgentTorch:
    """TD n-tuple agent compatible with `Connect4Agent` Protocol.

    Implements:
      - score_all_moves(board) -> dict[int,int]
      - best_move(board) -> int
      - score_move(board, column, first_guess=0) -> int
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: NTupleNetwork | None = None,
        tie_break: str = "center",
    ) -> None:
        if model is not None and model_path is not None:
            raise ValueError("Provide model_path or model, not both.")
        if model is not None:
            net2 = model.cpu()
        elif model_path is not None:
            net2 = NTupleNetwork.load(model_path, device="cpu")
        else:
            raise ValueError("Provide either model_path or model.")
        net2.eval()
        self._tie_break = tie_break
        self._eval = net2

    # ---- Protocol method 1 ----
    def score_all_moves(self, board) -> dict[int, int]:
        """Return {col: score} for all legal moves. Illegal/full columns are excluded."""
        player_to_move = board.current_player() - 1  # BitBully: {1,2} -> {0,1}
        if player_to_move not in (0, 1):
            raise ValueError(f"Unexpected current_player(): {board.current_player()}")

        scores: dict[int, int] = {}

        for col in range(7):
            if not board.is_legal_move(col):
                continue

            score = self.score_move(board=board, column=col)
            scores[col] = score

        return scores

    # ---- Protocol method 2 ----
    def best_move(self, board) -> int:
        """Return best legal move using BitBully-like tie breaking."""
        scores = self.score_all_moves(board)
        if not scores:
            raise ValueError("No legal moves available.")

        best = max(scores.values())
        candidates = [c for c, v in scores.items() if v == best]

        if len(candidates) == 1:
            return candidates[0]

        if self._tie_break == "center":
            center = 3
            return min(candidates, key=lambda c: (abs(c - center), c))
        if self._tie_break == "left":
            return min(candidates)
        if self._tie_break == "right":
            return max(candidates)

        raise ValueError("tie_break must be one of: 'center', 'left', 'right'")

    # ---- Optional Protocol method ----
    def score_move(self, board, column: int, first_guess: int = 0) -> int:
        """Evaluate a single legal move. `first_guess` is accepted for Protocol compatibility."""
        _ = first_guess  # this TD agent ignores it
        if not board.is_legal_move(column):
            raise ValueError(f"Illegal move: column {column}")

        player_to_move = board.current_player()
        after = board.play_on_copy(column)

        if after.has_win():
            return 100

        all_tokens, active_tokens, moves_left = after._board.rawState()
        torch_board = BoardBatch(
            all_tokens=torch.tensor([all_tokens]),
            active_tokens=torch.tensor([active_tokens]),
            moves_left=torch.tensor([moves_left]),
        )

        # opponent can win
        if after.can_win_next():
            return -100

        score = float(self._eval.forward(torch_board)[0].item())

        if player_to_move == 2:
            score = -score

        return int(score * 100.0)
