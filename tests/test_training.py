"""Tests for NTupleNetwork and best_afterstate_values."""

from __future__ import annotations

import os
import tempfile

import bitbully.bitbully_core as bbc
import pytest
import torch

from techdays26.ntuple_network import NTupleNetwork
from techdays26.ntuples import NTUPLE_BITIDX_LIST_150
from techdays26.torch_board import BoardBatch, move_mask_to_column
from techdays26.training import best_afterstate_values

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_net(*, random_weights: bool = False) -> NTupleNetwork:
    """Fresh NTupleNetwork, optionally with small random weights."""
    net = NTupleNetwork(n_tuple_list=NTUPLE_BITIDX_LIST_150)
    if random_weights:
        with torch.no_grad():
            net.W.normal_(0, 0.05)
    net.eval()
    return net


def _build_midgame_boards(B: int, *, n_rounds: int = 15, seed: int = 42) -> BoardBatch:
    """Batch of diverse mid-game positions built by playing random moves."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    board = BoardBatch.empty(B, DEVICE)
    for _ in range(n_rounds):
        actions = torch.randint(0, 7, (B,), device=DEVICE, generator=g)
        board.play_columns(actions)
        board.reset(board.done())
    return board


def _build_synced_boards(
    B: int, *, n_rounds: int = 15, seed: int = 42
) -> tuple[BoardBatch, list[bbc.BoardCore]]:
    """Mid-game boards with a matching list of BoardCore references."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    board = BoardBatch.empty(B, DEVICE)
    cores = [bbc.BoardCore() for _ in range(B)]
    for _ in range(n_rounds):
        actions = torch.randint(0, 7, (B,), device=DEVICE, generator=g)
        board.play_columns(actions)
        for i in range(B):
            cores[i].play(int(actions[i].item()))
        done = board.done()
        board.reset(done)
        for i in range(B):
            if cores[i].hasWin() or cores[i].movesLeft() <= 0:
                cores[i].setRawState(0, 0, 42)
    return board, cores


# ---------------------------------------------------------------------------
# NTupleNetwork tests
# ---------------------------------------------------------------------------


def test_ntuple_network_output_in_tanh_range():
    """forward() must return values in [-1, 1] (tanh activation)."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(256)

    with torch.no_grad():
        vals = net(board)

    assert vals.shape == (256,)
    assert vals.min() >= -1.0 and vals.max() <= 1.0, (
        f"Values outside [-1, 1]: min={vals.min():.6f}, max={vals.max():.6f}"
    )


def test_ntuple_network_zero_weights_give_zero():
    """With W=0 the network must output exactly 0 for every board."""
    net = _mk_net(random_weights=False)
    board = _build_midgame_boards(128)

    with torch.no_grad():
        vals = net(board)

    assert (vals == 0.0).all(), (
        f"Expected all zeros, got min={vals.min()}, max={vals.max()}"
    )


def test_ntuple_network_mirror_symmetry():
    """v(board) == v(mirror(board)) because the network averages both."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(256)
    mir = board.mirror()

    with torch.no_grad():
        v_orig = net(board)
        v_mir = net(mir)

    assert torch.allclose(v_orig, v_mir, atol=1e-6), (
        f"Mirror asymmetry: max diff = {(v_orig - v_mir).abs().max():.8f}"
    )


def test_ntuple_network_save_load_roundtrip():
    """Save + load must produce bit-identical forward-pass results."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(64)

    with torch.no_grad():
        v_before = net(board).clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        net.save(path)
        net2 = NTupleNetwork.load(path, device="cpu")
        net2.eval()
        with torch.no_grad():
            v_after = net2(board)
        assert torch.equal(v_before, v_after), "Forward pass differs after save/load"
    finally:
        os.remove(path)


def test_ntuple_network_gradient_flow():
    """A single forward+backward step must produce non-zero gradients on W."""
    net = NTupleNetwork(n_tuple_list=NTUPLE_BITIDX_LIST_150)
    with torch.no_grad():
        net.W.normal_(0, 0.01)
    net.train()

    board = _build_midgame_boards(64)
    loss = net(board).sum()
    loss.backward()

    assert net.W.grad is not None, "W.grad is None"
    assert (net.W.grad != 0).any(), "W.grad is all zeros (no gradient flow)"


# ---------------------------------------------------------------------------
# best_afterstate_values tests
# ---------------------------------------------------------------------------


def test_bav_returns_legal_onehot_moves():
    """Every non-zero move must be a legal one-hot landing square."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(512)

    best_mv, _ = best_afterstate_values(board, net, use_non_losing=False)

    legal = board.generate_legal_moves()
    active = best_mv != 0

    assert ((best_mv[active] & legal[active]) == best_mv[active]).all(), (
        "Some returned moves are not legal landing squares"
    )
    one_hot_or_zero = (best_mv == 0) | ((best_mv & (best_mv - 1)) == 0)
    assert one_hot_or_zero.all(), "Some returned moves have multiple bits set"


def test_bav_moves_subset_of_non_losing():
    """With use_non_losing=True, moves must be within the non-losing set."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(512)
    nl_moves = board.generate_non_losing_moves()

    best_mv, _ = best_afterstate_values(board, net, use_non_losing=True)

    active = best_mv != 0
    assert ((best_mv[active] & nl_moves[active]) == best_mv[active]).all(), (
        "Some returned moves are outside the non-losing move set"
    )


def test_bav_random_mode_still_legal():
    """With randomize=True for all boards, moves must still be legal and one-hot."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(512)
    randomize = torch.ones(512, dtype=torch.bool)

    rand_mv, _ = best_afterstate_values(
        board, net, randomize=randomize, use_non_losing=False
    )

    legal = board.generate_legal_moves()
    active = rand_mv != 0
    assert ((rand_mv[active] & legal[active]) == rand_mv[active]).all(), (
        "Random moves are not all legal"
    )
    one_hot_or_zero = (rand_mv == 0) | ((rand_mv & (rand_mv - 1)) == 0)
    assert one_hot_or_zero.all(), "Random moves are not one-hot"


def test_bav_terminal_rewards():
    """For terminal afterstates, the returned value must equal the game reward."""
    net = _mk_net(random_weights=True)
    board = _build_midgame_boards(2048, n_rounds=20)

    best_mv, best_val = best_afterstate_values(board, net, use_non_losing=False)

    tmp = BoardBatch(
        all_tokens=board.all_tokens.clone(),
        active_tokens=board.active_tokens.clone(),
        moves_left=board.moves_left.clone(),
    )
    tmp.play_masks(best_mv)
    done = tmp.done()
    reward = tmp.reward()

    n_terminal = done.sum().item()
    assert n_terminal > 0, "No terminal afterstates found; increase n_rounds or B"
    assert torch.equal(best_val[done], reward[done]), (
        f"Terminal values don't match rewards.\n"
        f"  vals    = {best_val[done][:10].tolist()}\n"
        f"  rewards = {reward[done][:10].tolist()}"
    )

    has_win = tmp.has_win()
    yellow_won = done & has_win & ((tmp.moves_left.to(torch.int64) & 1) == 1)
    red_won = done & has_win & ((tmp.moves_left.to(torch.int64) & 1) == 0)
    draw = done & ~has_win & (tmp.moves_left == 0)

    if yellow_won.any():
        assert (best_val[yellow_won] == 1.0).all()
    if red_won.any():
        assert (best_val[red_won] == -1.0).all()
    if draw.any():
        assert (best_val[draw] == 0.0).all()


def test_bav_consistency_with_core():
    """Moves from best_afterstate_values must match BoardCore state after play."""
    net = _mk_net(random_weights=True)
    B = 200
    board, cores = _build_synced_boards(B, n_rounds=15)

    best_mv, V_new = best_afterstate_values(
        board, net, use_non_losing=False, no_grad=True
    )

    for b_idx in range(B):
        mv = int(best_mv[b_idx].item())
        if mv == 0:
            continue

        col = move_mask_to_column(mv)
        assert cores[b_idx].isLegalMove(col), (
            f"Board {b_idx}: col {col} not legal in core"
        )

        ok = cores[b_idx].play(col)
        assert ok, f"Board {b_idx}: core.play({col}) returned False"
        bb_done = cores[b_idx].hasWin() or cores[b_idx].movesLeft() <= 0

        tmp = BoardBatch(
            all_tokens=board.all_tokens[b_idx : b_idx + 1].clone(),
            active_tokens=board.active_tokens[b_idx : b_idx + 1].clone(),
            moves_left=board.moves_left[b_idx : b_idx + 1].clone(),
        )
        tmp.play_masks(torch.tensor([mv], dtype=torch.int64))

        a_core, b_core, m_core = cores[b_idx].rawState()
        assert int(a_core) == int(tmp.all_tokens[0].item()), (
            f"Board {b_idx}: all_tokens mismatch"
        )
        assert int(b_core) == int(tmp.active_tokens[0].item()), (
            f"Board {b_idx}: active_tokens mismatch"
        )
        assert int(m_core) == int(tmp.moves_left[0].item()), (
            f"Board {b_idx}: moves_left mismatch"
        )
        assert bb_done == bool(tmp.done()[0].item()), f"Board {b_idx}: done mismatch"

        if bb_done:
            v = float(V_new[b_idx].item())
            ml = cores[b_idx].movesLeft()
            if cores[b_idx].hasWin() and ml % 2 == 1:
                assert v == 1.0, f"Board {b_idx}: yellow win should be +1, got {v}"
            elif cores[b_idx].hasWin() and ml % 2 == 0:
                assert v == -1.0, f"Board {b_idx}: red win should be -1, got {v}"
            elif ml <= 0:
                assert v == 0.0, f"Board {b_idx}: draw should be 0, got {v}"
            cores[b_idx].setRawState(0, 0, 42)


def test_bav_greedy_is_optimal():
    """The greedy move's noise-free value must be within tiebreak tolerance of the optimum."""
    net = _mk_net(random_weights=True)
    B = 128
    board = _build_midgame_boards(B, n_rounds=12)

    best_mv, _ = best_afterstate_values(board, net, use_non_losing=False)

    yellow_to_move = (board.moves_left.to(torch.int64) & 1) == 0
    legal_mask = board.generate_legal_moves()
    neg_inf = torch.full((B,), float("-inf"))
    pos_inf = torch.full((B,), float("inf"))
    true_best = torch.where(yellow_to_move, neg_inf, pos_inf).to(torch.float32)

    for mv in board.iter_move_masks(legal_mask):
        active = mv != 0
        if not active.any():
            break
        tmp = BoardBatch(
            all_tokens=board.all_tokens.clone(),
            active_tokens=board.active_tokens.clone(),
            moves_left=board.moves_left.clone(),
        )
        tmp.play_masks(mv)
        r = tmp.reward().to(torch.float32)
        with torch.no_grad():
            v = net(tmp).to(torch.float32)
        val = torch.where(torch.isnan(r), v, r)
        better = torch.where(yellow_to_move, val > true_best, val < true_best) & active
        true_best = torch.where(better, val, true_best)

    # Recompute chosen move's noise-free value
    tmp_chosen = BoardBatch(
        all_tokens=board.all_tokens.clone(),
        active_tokens=board.active_tokens.clone(),
        moves_left=board.moves_left.clone(),
    )
    tmp_chosen.play_masks(best_mv)
    r_chosen = tmp_chosen.reward().to(torch.float32)
    with torch.no_grad():
        v_chosen = net(tmp_chosen).to(torch.float32)
    chosen_noisefree = torch.where(torch.isnan(r_chosen), v_chosen, r_chosen)

    has_move = best_mv != 0
    tol = 5e-4  # noise is ~1e-4 * randn; 5-sigma margin
    assert (~has_move | ~yellow_to_move | (chosen_noisefree >= true_best - tol)).all()
    assert (~has_move | yellow_to_move | (chosen_noisefree <= true_best + tol)).all()


def test_training_loop_invariants():
    """Simulate training steps: all moves must be legal, done boards must have rewards."""
    net = _mk_net(random_weights=True)
    B = 1024
    board = BoardBatch.empty(B, DEVICE)
    epsilon = 0.1

    for step in range(50):
        randomize = torch.rand((B,), device=DEVICE) < epsilon

        with torch.no_grad():
            best_mv, _ = best_afterstate_values(
                board, net, randomize=randomize, use_non_losing=False, no_grad=True
            )

        legal = board.play_masks(best_mv)
        assert legal.all(), f"Step {step}: {(~legal).sum()} illegal moves"

        done = board.done()
        assert not (done & torch.isnan(board.reward())).any(), (
            f"Step {step}: done boards have NaN reward"
        )
        board.reset(done)
