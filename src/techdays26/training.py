from __future__ import annotations

import torch

from techdays26.ntuple_network import NTupleNetwork
from techdays26.torch_board import BoardBatch


def best_afterstate_values(
    board: BoardBatch,
    net: NTupleNetwork,
    *,
    moves_mask: torch.Tensor | None = None,
    randomize: torch.Tensor | None = None,  # [B] bool (epsilon-greedy)
    use_non_losing: bool = True,
    no_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the best afterstate for each board in the batch.

    Iterates over all legal (or non-losing) moves, evaluates each afterstate
    with *net*, and returns the greedy-best move per board.  When *randomize*
    is supplied, boards flagged True receive a uniformly-random legal move
    instead (ε-greedy exploration via reservoir sampling — no extra pass).

    Args:
        board:       Current batch of board states [B].
        net:         Value network; called as ``net(afterstate) -> [B] float32``.
        moves_mask:  Pre-computed move bitboard [B] int64, or None to generate
                     non-losing / legal moves automatically.
        randomize:   [B] bool — True means "pick a random move for this board".
        use_non_losing: Use ``generate_non_losing_moves`` (default) instead of
                     ``generate_legal_moves``.  Ignored when *moves_mask* is given.
        no_grad:     Wrap the forward pass in ``torch.no_grad()`` (default True).

    Returns:
        chosen_mv:  [B] int64 — one-hot landing-square bitboard.
        chosen_val: [B] float32 — network value (or terminal reward) of the
                    chosen afterstate.
    """
    dev = board.all_tokens.device
    B = board.all_tokens.shape[0]

    if moves_mask is None:
        moves_mask = (
            board.generate_non_losing_moves()
            if use_non_losing
            else board.generate_legal_moves()
        )
    moves_mask = moves_mask.to(device=dev, dtype=torch.int64)

    yellow_to_move = (board.moves_left.to(torch.int64) & 1) == 0  # [B] bool

    neg_inf = torch.tensor(float("-inf"), device=dev)
    pos_inf = torch.tensor(float("inf"), device=dev)
    best_val = (
        torch
        .where(yellow_to_move, neg_inf, pos_inf)
        .to(torch.float32)
        .expand(B)
        .clone()
    )
    best_mv = torch.zeros((B,), device=dev, dtype=torch.int64)

    # Reservoir sampling for uniform-random exploration (no extra move iteration)
    rand_mv = torch.zeros((B,), device=dev, dtype=torch.int64)
    rand_val = torch.full((B,), float("nan"), device=dev, dtype=torch.float32)
    seen = torch.zeros((B,), device=dev, dtype=torch.int32)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for mv in board.iter_move_masks(moves_mask):
            active = mv != 0
            if not active.any():
                break

            tmp = type(board)(
                all_tokens=board.all_tokens.clone(),
                active_tokens=board.active_tokens.clone(),
                moves_left=board.moves_left.clone(),
            )
            legal = tmp.play_masks(mv)
            active = active & legal

            r = tmp.reward().to(torch.float32)  # +1/-1/0 or NaN
            v = net(tmp).to(torch.float32)
            v = v + 1e-4 * torch.randn_like(v)  # tiebreak noise
            val = torch.where(torch.isnan(r), v, r)  # use reward for terminal states

            # Greedy best
            better = (
                torch.where(yellow_to_move, val > best_val, val < best_val) & active
            )
            best_val = torch.where(better, val, best_val)
            best_mv = torch.where(better, mv, best_mv)

            # Reservoir sampling (uniform random)
            seen = seen + active.to(seen.dtype)
            replace = active & (
                torch.rand((B,), device=dev) < (1.0 / seen.to(torch.float32))
            )
            rand_mv = torch.where(replace, mv, rand_mv)
            rand_val = torch.where(replace, val, rand_val)

    if randomize is None:
        return best_mv, best_val

    randomize = randomize.to(device=dev, dtype=torch.bool)
    return (
        torch.where(randomize, rand_mv, best_mv),
        torch.where(randomize, rand_val, best_val),
    )


@torch.no_grad()
def bootstrap_target_from_afterstate(
    after: BoardBatch,
    target_net: NTupleNetwork,
) -> torch.Tensor:
    """Return the bootstrap target for an afterstate.

    Uses the terminal reward for finished games; falls back to the target
    network's value estimate for ongoing games.

    Args:
        after:      Afterstate batch [B] (state *after* the action was applied).
        target_net: Frozen / Polyak-averaged target network.

    Returns:
        [B] float32 in [-1, 1].
    """
    r = after.reward().to(torch.float32)  # +1/-1/0 or NaN if not terminal
    v = target_net(after).to(torch.float32)
    return torch.where(torch.isnan(r), v, r)
