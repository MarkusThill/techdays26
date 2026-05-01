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
    batch_afterstates: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the best afterstate for each board in the batch.

    Iterates over all legal (or non-losing) moves, evaluates each afterstate
    with *net*, and returns the greedy-best move per board.  When *randomize*
    is supplied, boards flagged True receive a uniformly-random legal move
    instead.

    Two strategies are available:

    - **Loop** (default on CPU): one ``net()`` forward per move, up to 7 calls.
      Best on CPU because each forward fits in cache.
    - **Batched** (default on CUDA): builds all up-to-7 afterstates, stacks
      them into a single ``B*K`` flat batch, and calls ``net()`` once. Saves
      kernel-launch overhead on GPU.

    Args:
        board:       Current batch of board states [B].
        net:         Value network; called as ``net(afterstate) -> [B] float32``.
        moves_mask:  Pre-computed move bitboard [B] int64, or None to generate
                     non-losing / legal moves automatically.
        randomize:   [B] bool — True means "pick a random move for this board".
        use_non_losing: Use ``generate_non_losing_moves`` (default) instead of
                     ``generate_legal_moves``. Ignored when *moves_mask* is given.
        no_grad:     Wrap the forward pass in ``torch.no_grad()`` (default True).
        batch_afterstates: If True, use the batched single-forward path.
                     If False, use the per-move loop path. If None (default),
                     auto-select: True on CUDA, False otherwise.

    Returns:
        chosen_mv:  [B] int64 — one-hot landing-square bitboard.
        chosen_val: [B] float32 — network value (or terminal reward) of the
                    chosen afterstate.
    """
    dev = board.all_tokens.device

    if moves_mask is None:
        moves_mask = (
            board.generate_non_losing_moves()
            if use_non_losing
            else board.generate_legal_moves()
        )
    moves_mask = moves_mask.to(device=dev, dtype=torch.int64)

    if batch_afterstates is None:
        batch_afterstates = dev.type == "cuda"

    if batch_afterstates:
        return _bav_batched(
            board, net, moves_mask=moves_mask, randomize=randomize, no_grad=no_grad
        )
    return _bav_loop(
        board, net, moves_mask=moves_mask, randomize=randomize, no_grad=no_grad
    )


def _bav_loop(
    board: BoardBatch,
    net: NTupleNetwork,
    *,
    moves_mask: torch.Tensor,
    randomize: torch.Tensor | None,
    no_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-move loop: K forwards (K ≤ 7). CPU-friendly."""
    dev = board.all_tokens.device
    B = board.all_tokens.shape[0]

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

            r = tmp.reward().to(torch.float32)
            v = net(tmp).to(torch.float32)
            v = v + 1e-4 * torch.randn_like(v)
            val = torch.where(torch.isnan(r), v, r)

            better = (
                torch.where(yellow_to_move, val > best_val, val < best_val) & active
            )
            best_val = torch.where(better, val, best_val)
            best_mv = torch.where(better, mv, best_mv)

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


def _bav_batched(
    board: BoardBatch,
    net: NTupleNetwork,
    *,
    moves_mask: torch.Tensor,
    randomize: torch.Tensor | None,
    no_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched: stack up-to-K=7 afterstates into one B*K forward. GPU-friendly."""
    dev = board.all_tokens.device
    B = board.all_tokens.shape[0]

    yellow_to_move = (board.moves_left.to(torch.int64) & 1) == 0  # [B] bool
    neg_inf = torch.tensor(float("-inf"), device=dev, dtype=torch.float32)
    pos_inf = torch.tensor(float("inf"), device=dev, dtype=torch.float32)

    all_toks_list: list[torch.Tensor] = []
    act_toks_list: list[torch.Tensor] = []
    ml_list: list[torch.Tensor] = []
    mv_list: list[torch.Tensor] = []
    active_list: list[torch.Tensor] = []

    for mv in board.iter_move_masks(moves_mask):
        if not (mv != 0).any():
            break

        tmp = type(board)(
            all_tokens=board.all_tokens.clone(),
            active_tokens=board.active_tokens.clone(),
            moves_left=board.moves_left.clone(),
        )
        legal = tmp.play_masks(mv)
        active = (mv != 0) & legal

        all_toks_list.append(tmp.all_tokens)
        act_toks_list.append(tmp.active_tokens)
        ml_list.append(tmp.moves_left)
        mv_list.append(mv)
        active_list.append(active)

    K = len(mv_list)

    if K == 0:
        best_mv = torch.zeros((B,), device=dev, dtype=torch.int64)
        best_val = torch.where(yellow_to_move, neg_inf, pos_inf).to(torch.float32)
        if randomize is None:
            return best_mv, best_val
        rand_val = torch.full((B,), float("nan"), device=dev, dtype=torch.float32)
        return best_mv, rand_val

    all_toks = torch.stack(all_toks_list, dim=1)  # [B, K]
    act_toks = torch.stack(act_toks_list, dim=1)
    moves_left = torch.stack(ml_list, dim=1)
    mvs = torch.stack(mv_list, dim=1)  # [B, K]
    active = torch.stack(active_list, dim=1)  # [B, K] bool

    flat_board = type(board)(
        all_tokens=all_toks.reshape(-1),
        active_tokens=act_toks.reshape(-1),
        moves_left=moves_left.reshape(-1),
    )

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        r_flat = flat_board.reward().to(torch.float32)  # [B*K]
        v_flat = net(flat_board).to(torch.float32)  # [B*K] — single forward
        v_flat = v_flat + 1e-4 * torch.randn_like(v_flat)
        val_flat = torch.where(torch.isnan(r_flat), v_flat, r_flat)
    val = val_flat.view(B, K)  # [B, K]

    # Greedy: yellow maximizes, red minimizes. Inactive slots get the worst
    # value so argmax never picks them.
    fill = torch.where(yellow_to_move, neg_inf, pos_inf).unsqueeze(1)  # [B, 1]
    val_eff = torch.where(active, val, fill)
    sign = torch.where(
        yellow_to_move,
        torch.tensor(1.0, device=dev, dtype=torch.float32),
        torch.tensor(-1.0, device=dev, dtype=torch.float32),
    ).unsqueeze(1)
    best_idx = (val_eff * sign).argmax(dim=1, keepdim=True)
    best_mv = mvs.gather(1, best_idx).squeeze(1)
    best_val = val_eff.gather(1, best_idx).squeeze(1)

    if randomize is None:
        return best_mv, best_val

    rand_score = torch.where(
        active,
        torch.rand((B, K), device=dev),
        torch.tensor(-1.0, device=dev),
    )
    rand_idx = rand_score.argmax(dim=1, keepdim=True)
    rand_mv = mvs.gather(1, rand_idx).squeeze(1)
    rand_val_picked = val.gather(1, rand_idx).squeeze(1)
    # Match original API: rand_val is NaN where no slot is active.
    any_active = active.any(dim=1)
    rand_val = torch.where(
        any_active,
        rand_val_picked,
        torch.full_like(rand_val_picked, float("nan")),
    )

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
