from __future__ import annotations

import torch
import torch.nn as nn


class NTupleNetwork(nn.Module):
    def __init__(self, n_tuple_list: list[list[int]]):
        super().__init__()

        self.M = len(n_tuple_list)
        self.N = len(n_tuple_list[0])
        self.K = 4**self.N

        # [M, N] bit indices
        self.register_buffer(
            "n_tuple_tensor", torch.tensor(n_tuple_list, dtype=torch.int64)
        )

        # Two players × M LUTs × K entries
        # 0 = Yellow, 1 = Red
        self.W = nn.Parameter(torch.zeros(2, self.M, self.K))

    def forward(self, board: "BoardBatch") -> torch.Tensor:
        """Returns [B] tensor in [-1, 1]."""
        # [B, M] table indices
        T = board.table_positions(self.n_tuple_tensor)
        T_mir = board.mirror().table_positions(self.n_tuple_tensor)
        B, M = T.shape
        dev = T.device

        # Active player per board: 0=Yellow, 1=Red
        player_idx = ((board.moves_left.to(torch.int64) & 1) != 0).to(torch.int64)

        # Pattern indices [M]
        m_idx = torch.arange(M, device=dev)

        # Gather: W[player_idx[b], m, T[b,m]]
        w = self.W[
            player_idx.unsqueeze(1),  # [B,1]
            m_idx.unsqueeze(0),  # [1,M]
            T,  # [B,M]
        ]  # -> [B,M]
        w_mir = self.W[player_idx.unsqueeze(1), m_idx.unsqueeze(0), T_mir]

        # Sum over patterns (and symmetry): [B]
        score = (w + w_mir).sum(dim=1)
        return torch.tanh(score)

    @torch.no_grad()
    def save(self, path: str) -> None:
        """Save model weights + architecture-relevant metadata."""
        payload = {
            "state_dict": self.state_dict(),
            "n_tuple_list": self.n_tuple_tensor.cpu().tolist(),
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
        strict: bool = True,
    ) -> NTupleNetwork:
        """Load model fully onto the specified device (CPU or GPU)."""
        # 1) Always load checkpoint onto CPU first (safe & portable)
        payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("Invalid checkpoint format.")

        n_tuple_list = payload.get("n_tuple_list")
        if n_tuple_list is None:
            raise ValueError("Checkpoint missing 'n_tuple_list'.")

        # 2) Construct model (still on CPU)
        model = cls(n_tuple_list=n_tuple_list)

        # 3) Load weights (still CPU tensors)
        model.load_state_dict(payload["state_dict"], strict=strict)

        # 4) Move EVERYTHING (parameters + buffers) in one go
        model = model.to(device)

        return model
