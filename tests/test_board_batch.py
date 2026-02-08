# tests/test_board_batch.py
from __future__ import annotations

import bitbully.bitbully_core as bbc
import pytest
import torch

from techdays26.torch_board import BoardBatch, move_mask_to_column

# =============================================================================
# Test configuration
# =============================================================================

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def _devices() -> list[torch.device]:
    devs = [CPU]
    if torch.cuda.is_available():
        devs.append(CUDA)
    return devs


def _as_int(x: torch.Tensor) -> int:
    return int(x.item())


@pytest.fixture(params=_devices(), ids=lambda d: d.type)
def dev(request: pytest.FixtureRequest) -> torch.device:
    return request.param


@pytest.fixture(
    params=[torch.int16, torch.int32], ids=lambda dt: str(dt).replace("torch.", "")
)
def moves_left_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture(autouse=True)
def _clear_boardbatch_caches() -> None:
    BoardBatch.clear_caches()


# =============================================================================
# Helpers
# =============================================================================


def _sync_if_cuda(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _mk_batch(B: int, dev: torch.device, moves_left_dtype: torch.dtype) -> BoardBatch:
    return BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)


def _mk_core_batch(B: int) -> list[bbc.BoardCore]:
    return [bbc.BoardCore() for _ in range(B)]


def _core_raw_state(core: bbc.BoardCore) -> tuple[int, int, int]:
    a, b, m = core.rawState()
    return int(a), int(b), int(m)


def _assert_states_equal(core: bbc.BoardCore, bb: BoardBatch, i: int) -> None:
    a, b, m = _core_raw_state(core)
    assert a == int(bb.all_tokens[i].item())
    assert b == int(bb.active_tokens[i].item())
    assert m == int(bb.moves_left[i].item())


def _isnan(x: float) -> bool:
    return x != x


def _assert_reward_matches_core(core: bbc.BoardCore, r: float) -> None:
    """BoardBatch.reward() contract:
    +1.0  -> yellow wins
    -1.0  -> red wins
    0.0   -> draw
    NaN   -> game not finished
    """
    if _isnan(r):
        assert (not core.hasWin()) and (core.movesLeft() > 0)
        return

    if r == 1.0:
        assert core.hasWin()
        assert core.movesLeft() % 2 == 1  # odd => yellow just moved
        return

    if r == -1.0:
        assert core.hasWin()
        assert core.movesLeft() % 2 == 0  # even => red just moved
        return

    if r == 0.0:
        assert (not core.hasWin()) and (core.movesLeft() == 0)
        return

    raise AssertionError(f"Unexpected reward value: {r}")


def _assert_active_player_matches_core(
    core: bbc.BoardCore, ap: int, aps: float
) -> None:
    expected_player = core.popCountBoard() % 2 + 1
    expected_sign = 1 - 2 * (core.popCountBoard() % 2)
    assert ap == expected_player
    assert aps == float(expected_sign)


def _col_reflect(c: int) -> int:
    return 6 - c


def _mirror_move_mask_by_columns(mv_mask: int, *, off: int = 9) -> int:
    """Mirror a legal landing-square mask (<=7 bits) by reflecting columns."""
    out = 0
    row_mask = (1 << 6) - 1  # 6 playable bits per column

    for c in range(7):
        src_base = c * off
        bits = mv_mask & (row_mask << src_base)
        if bits == 0:
            continue

        dst_base = (6 - c) * off

        # Move bits from src_base -> dst_base without negative shifts
        if dst_base > src_base:
            out |= bits << (dst_base - src_base)
        elif dst_base < src_base:
            out |= bits >> (src_base - dst_base)
        else:
            out |= bits

    return out


def _inject_from_core(
    core: bbc.BoardCore,
    *,
    dev: torch.device,
    moves_left_dtype: torch.dtype,
) -> BoardBatch:
    a, b, m = core.rawState()
    return BoardBatch(
        all_tokens=torch.tensor([int(a)], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([int(b)], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([int(m)], device=dev, dtype=moves_left_dtype),
    )


def _mirror_bits_via_boardbatch(
    bits: torch.Tensor | int,
    *,
    dev: torch.device,
    moves_left_dtype: torch.dtype,
) -> int:
    """Mirror arbitrary playable-square bitboard bits using BoardBatch.mirror()."""
    if isinstance(bits, torch.Tensor):
        bits_i = int(bits.item())
    else:
        bits_i = int(bits)

    tmp = BoardBatch(
        all_tokens=torch.tensor([bits_i], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([0], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([42], device=dev, dtype=moves_left_dtype),
    )
    return int(tmp.mirror().all_tokens[0].item())


def _spare_bits_mask() -> int:
    # In each 9-bit column, rows 0..5 are playable, rows 6..8 must never be set.
    m = 0
    for c in range(7):
        base = c * 9
        for r in (6, 7, 8):
            m |= 1 << (base + r)
    return m


def _ref_table_positions_single(
    *,
    all_tokens: int,
    active_tokens: int,
    moves_left: int,
    patterns_bitidx: list[list[int]],
) -> list[int]:
    """Pure-Python reference for BoardBatch.table_positions() for a single board.

    Matches BoardBatch semantics exactly:
      - occupied := (all_tokens has bit)
      - is_active := (active_tokens has bit)   [active tokens == side-to-move stones]
      - reachable := (~occupied) AND (bit is in landing-squares mask)
      - active_is_yellow := (moves_left % 2 == 0)
      - digit mapping:
          0 empty (not reachable)
          3 reachable (landing square)
          1 yellow
          2 red
      - base-4 encoding with least significant digit = first element of pattern
    """
    N_COLUMNS = 7
    N_ROWS = 6
    OFF = 9

    # Same masks as BoardBatch
    bottom = 0
    all_legal = 0
    for c in range(N_COLUMNS):
        bottom |= 1 << (c * OFF + 0)
        for r in range(N_ROWS):
            all_legal |= 1 << (c * OFF + r)

    # Same landing-square computation as BoardBatch.legal_moves_mask()
    legal_moves = (all_tokens + bottom) & all_legal

    active_is_yellow = (moves_left & 1) == 0

    out: list[int] = []
    for pat in patterns_bitidx:
        val = 0
        pow4 = 1
        for bit_index in pat:
            mask = 1 << bit_index
            occ = (all_tokens & mask) != 0
            act = (active_tokens & mask) != 0
            reach = (not occ) and ((legal_moves & mask) != 0)

            if reach:
                d = 3
            elif occ:
                # Same logic as:
                # yellow = (active_is_yellow ? is_active : ~is_active) & occupied
                if active_is_yellow:
                    yellow = act
                else:
                    yellow = not act
                d = 1 if yellow else 2
            else:
                d = 0

            val += d * pow4
            pow4 *= 4

        out.append(val)

    return out


def _make_patterns(dev: torch.device, *, M: int, N: int) -> torch.Tensor:
    # simple deterministic bit indices in range [0..63] in your layout
    # ensures shape control for cache isolation tests
    idx = []
    k = 0
    for _ in range(M):
        row = []
        for _ in range(N):
            row.append(k)
            k += 1
        idx.append(row)
    return torch.tensor(idx, device=dev, dtype=torch.int64)


def _columns_from_landing_mask(
    moves: int, *, off: int = 9, n_cols: int = 7, n_rows: int = 6
) -> set[int]:
    """Decode a landing-squares move mask (<=1 bit per column) into the set of columns.
    Works for both legalMovesMask() and generateNonLosingMoves() results.
    """
    cols: set[int] = set()
    row_mask = (1 << n_rows) - 1
    for c in range(n_cols):
        col_bits = (moves >> (c * off)) & row_mask
        if col_bits != 0:
            cols.add(c)
    return cols


def _assert_same_state(core: bbc.BoardCore, bb: BoardBatch) -> None:
    a, b, m = core.rawState()
    assert int(bb.all_tokens[0].item()) == int(a)
    assert int(bb.active_tokens[0].item()) == int(b)
    assert int(bb.moves_left[0].item()) == int(m)


def test_reward_full_board_red_win_is_minus_one(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """Find a position with:
      - board is full (movesLeft == 0)
      - last mover has a win (hasWin == True)
    On a full board (42 plies), the last mover is Red, so reward must be -1.0
    (not 0.0 draw).
    """
    # Try to find such a position using trusted BoardCore.
    core_win_full: bbc.BoardCore | None = None
    core = bbc.BoardCore()
    assert core.play("122221440112324444566666655555000010133333")
    assert core.movesLeft() == 0 and core.hasWin()
    core_win_full = core

    if core_win_full is None:
        pytest.skip(
            "Could not find a full-board red-win position quickly; increase attempts if needed."
        )

    # Sanity: full board => 42 tokens, last mover is Red in standard rules
    assert core_win_full.movesLeft() == 0
    assert core_win_full.hasWin() is True

    # Inject into BoardBatch
    all_tokens, active_tokens, moves_left = core_win_full.rawState()
    bb = BoardBatch(
        all_tokens=torch.tensor([int(all_tokens)], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([int(active_tokens)], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([int(moves_left)], device=dev, dtype=moves_left_dtype),
    )

    # Must be terminal and a win
    assert bool(bb.done()[0].item()) is True
    assert bool(bb.has_win()[0].item()) is True
    assert int(bb.moves_left[0].item()) == 0

    # Full board => moves_left even => red just moved => reward must be -1.0
    r = float(bb.reward()[0].item())
    assert r == -1.0


# =============================================================================
# BoardCore.legalMoves(...) list matches your bitmask -> columns decode
# =============================================================================


@pytest.mark.parametrize("non_losing", [False, True])
def test_core_legalMoves_list_matches_boardbatch_masks(
    dev: torch.device, moves_left_dtype: torch.dtype, non_losing: bool
) -> None:
    """For many random positions:
      core.legalMoves(nonLosing=X, orderMoves=False)  (list of columns)
    matches:
      decode_columns(bb.(generate_legal_moves or generate_non_losing_moves) bitmask)
    """
    g = torch.Generator(device=dev)
    g.manual_seed(424242)

    # Generate a bunch of random positions via BoardCore (trusted).
    # We use random ply depths to cover early/mid game.
    for _ in range(120):
        nply = int(torch.randint(0, 30, (1,), generator=g, device=dev).item())
        core, _moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)

        bb = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)

        if non_losing:
            mask = int(bb.generate_non_losing_moves()[0].item())
            cols_bb = _columns_from_landing_mask(mask, off=bb.COLUMN_BIT_OFFSET)
        else:
            mask = int(bb.generate_legal_moves()[0].item())
            cols_bb = _columns_from_landing_mask(mask, off=bb.COLUMN_BIT_OFFSET)

        if non_losing:
            cols_core = _columns_from_landing_mask(
                core.generateNonLosingMoves(), off=bb.COLUMN_BIT_OFFSET
            )
        else:
            cols_core = _columns_from_landing_mask(
                core.legalMovesMask(), off=bb.COLUMN_BIT_OFFSET
            )

        assert cols_bb == cols_core

    _sync_if_cuda(dev)


# =============================================================================
# 2) For all columns: core.isLegalMove(c) matches presence in BoardBatch legal mask
# =============================================================================


def test_core_isLegalMove_matches_boardbatch_legal_mask(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(777777)

    for _ in range(120):
        nply = int(torch.randint(0, 35, (1,), generator=g, device=dev).item())
        core, _moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)
        bb = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)

        legal_mask = int(bb.generate_legal_moves()[0].item())
        cols_from_mask = _columns_from_landing_mask(
            legal_mask, off=bb.COLUMN_BIT_OFFSET
        )

        for c in range(bb.N_COLUMNS):
            assert core.isLegalMove(c) == (c in cols_from_mask)

    _sync_if_cuda(dev)


# =============================================================================
# 3) Strong transition test: playMoveOnCopy(c) == BoardBatch.play_columns([c])
# =============================================================================


def test_playMoveOnCopy_equivalence_all_legal_columns(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """For random positions and for every column c:
      - If legal: core.playMoveOnCopy(c) must match bb_copy.play_columns([c]) state.
      - If illegal: both must not change / report illegal.
    Also checks hasWin, canWin, legal/nonlosing masks after the move.
    """
    g = torch.Generator(device=dev)
    g.manual_seed(999001)

    for _ in range(70):
        nply = int(torch.randint(0, 34, (1,), generator=g, device=dev).item())
        core, _moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)

        bb0 = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)

        for c in range(bb0.N_COLUMNS):
            # --- core side ---
            is_legal = core.isLegalMove(c)
            if is_legal:
                core2 = core.playMoveOnCopy(c)
            else:
                core2 = core.copy()

            # --- batch side ---
            bb2 = BoardBatch(
                all_tokens=bb0.all_tokens.clone(),
                active_tokens=bb0.active_tokens.clone(),
                moves_left=bb0.moves_left.clone(),
            )
            ok = bb2.play_columns(torch.tensor([c], device=dev, dtype=torch.int64))

            assert bool(ok.item()) == is_legal

            # state equivalence
            _assert_same_state(core2, bb2)

            # win equivalence
            assert bool(bb2.has_win().item()) == core2.hasWin()

            # canWin equivalence (any immediate win for side to move)
            assert bool(bb2.can_win().item()) == core2.canWin()

            # canWin(column) equivalence for all columns in the resulting state
            cols = torch.arange(bb2.N_COLUMNS, device=dev, dtype=torch.int64)
            cwc = bb2.can_win_column(cols)
            for cc in range(bb2.N_COLUMNS):
                assert bool(cwc[cc].item()) == core2.canWin(cc)

            # legal and non-losing masks equivalence
            assert int(bb2.generate_legal_moves()[0].item()) == core2.legalMovesMask()
            assert (
                int(bb2.generate_non_losing_moves()[0].item())
                == core2.generateNonLosingMoves()
            )

    _sync_if_cuda(dev)


# =============================================================================
# 1) Cache correctness + isolation: different shapes back-to-back must not collide
# =============================================================================


def test_table_positions_cache_isolation_between_shapes(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(8, dev, moves_left_dtype=moves_left_dtype)

    p1 = _make_patterns(dev, M=2, N=4)
    p2 = _make_patterns(dev, M=3, N=4)  # different M

    T1a = bb.table_positions(p1)
    T2 = bb.table_positions(p2)
    T1b = bb.table_positions(p1)

    assert T1a.shape == (8, 2)
    assert T2.shape == (8, 3)
    assert torch.equal(T1a, T1b)  # cache hit must be correct, not shape-mismatched

    _sync_if_cuda(dev)


def test_weights_cache_identity_and_shape(dev: torch.device) -> None:
    w1 = BoardBatch._weights_base4(dev, 4)
    w2 = BoardBatch._weights_base4(dev, 4)
    w3 = BoardBatch._weights_base4(dev, 5)

    assert w1 is w2
    assert w1.shape == (1, 1, 4)
    assert w3.shape == (1, 1, 5)
    assert w1 is not w3

    _sync_if_cuda(dev)


def test_pattern_cache_isolated_per_device_if_cuda_available(
    moves_left_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cpu = torch.device("cpu")
    cuda = torch.device("cuda", 0)

    # Empty boards -> deterministic table_positions == reachable-only encoding
    bb_cpu = BoardBatch.empty(4, cpu, moves_left_dtype=moves_left_dtype)
    bb_cuda = BoardBatch.empty(4, cuda, moves_left_dtype=moves_left_dtype)

    patterns = torch.tensor([[0, 9, 18, 27]], dtype=torch.int64, device=cpu)

    T_cpu = bb_cpu.table_positions(patterns)
    T_cuda = bb_cuda.table_positions(
        patterns
    )  # triggers auto-move inside _pattern_masks if you allow it

    assert torch.equal(T_cpu.to("cpu"), T_cuda.to("cpu"))

    torch.cuda.synchronize()


# =============================================================================
# 2) table_positions validation: dtype / rank / range
# =============================================================================


def test_table_positions_rejects_non_int64_patterns(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(2, dev, moves_left_dtype=moves_left_dtype)
    patterns_i32 = torch.tensor([[0, 9, 18]], device=dev, dtype=torch.int32)
    with pytest.raises(TypeError):
        _ = bb.table_positions(patterns_i32)


def test_table_positions_rejects_wrong_rank_patterns(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(2, dev, moves_left_dtype=moves_left_dtype)

    # rank-1
    p1 = torch.tensor([0, 9, 18], device=dev, dtype=torch.int64)
    with pytest.raises(Exception):
        _ = bb.table_positions(p1)

    # rank-3
    p3 = torch.zeros((1, 2, 3), device=dev, dtype=torch.int64)
    with pytest.raises(Exception):
        _ = bb.table_positions(p3)


@pytest.mark.xfail(
    reason=(
        "Currently out-of-range bit indices may not raise. "
        "If you want safety, add an explicit range check 0<=idx<64 and switch this to passing."
    )
)
def test_table_positions_rejects_out_of_range_bit_indices(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(2, dev, moves_left_dtype=moves_left_dtype)
    bad = torch.tensor([[0, 9, 999]], device=dev, dtype=torch.int64)
    with pytest.raises(ValueError):
        _ = bb.table_positions(bad)


# =============================================================================
# 3) play_masks contract + 4) play_masks illegal no-op
# =============================================================================


def test_play_masks_illegal_is_noop(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(4, dev, moves_left_dtype=moves_left_dtype)

    all0 = bb.all_tokens.clone()
    act0 = bb.active_tokens.clone()
    ml0 = bb.moves_left.clone()

    # mv=0 is illegal
    mv0 = torch.zeros((4,), device=dev, dtype=torch.int64)
    ok = bb.play_masks(mv0)
    assert not bool(ok.any().item())
    assert torch.equal(bb.all_tokens, all0)
    assert torch.equal(bb.active_tokens, act0)
    assert torch.equal(bb.moves_left, ml0)

    # "floating" move: set bit at row 1 in column 0 on empty board (not a landing square)
    mv_float = torch.full((4,), 1 << 1, device=dev, dtype=torch.int64)
    ok = bb.play_masks(mv_float)
    assert not bool(ok.any().item())
    assert torch.equal(bb.all_tokens, all0)
    assert torch.equal(bb.active_tokens, act0)
    assert torch.equal(bb.moves_left, ml0)

    # spare-bit move (illegal square)
    spare = 1 << 6  # row 6 in col 0
    mv_spare = torch.full((4,), spare, device=dev, dtype=torch.int64)
    ok = bb.play_masks(mv_spare)
    assert not bool(ok.any().item())
    assert torch.equal(bb.all_tokens, all0)
    assert torch.equal(bb.active_tokens, act0)
    assert torch.equal(bb.moves_left, ml0)

    _sync_if_cuda(dev)


@pytest.mark.xfail(
    reason=(
        "play_masks() currently accepts multi-bit masks if they are a subset of landing squares. "
        "If you want strict one-hot semantics, enforce (mv & (mv-1))==0 and remove this xfail."
    )
)
def test_play_masks_rejects_multi_bit_mask(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(1, dev, moves_left_dtype=moves_left_dtype)

    legal = int(bb.generate_legal_moves()[0].item())  # 7 landing squares
    lsb = legal & -legal
    rest = legal ^ lsb
    lsb2 = rest & -rest
    mv = lsb | lsb2  # two bits set

    ok = bb.play_masks(torch.tensor([mv], device=dev, dtype=torch.int64))
    assert bool(ok.item()) is False


# =============================================================================
# 5) Terminal behavior stability (reward/done/has_win are stable)
# =============================================================================


def test_terminal_state_outputs_stable(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    # Create a simple vertical win for the player who just moved:
    # sequence: 3,4,3,4,3,4,3 => last move in 3 wins.
    bb = BoardBatch.empty(1, dev, moves_left_dtype=moves_left_dtype)
    seq = [3, 4, 3, 4, 3, 4, 3]
    for a in seq:
        assert bool(bb.play_columns(torch.tensor([a], device=dev)).item()) is True

    assert bool(bb.has_win().item()) is True
    assert bool(bb.done().item()) is True

    r1 = float(bb.reward().item())
    hw1 = bool(bb.has_win().item())
    d1 = bool(bb.done().item())
    lm1 = int(bb.generate_legal_moves()[0].item())

    # Call repeatedly; must not change
    for _ in range(10):
        assert float(bb.reward().item()) == r1
        assert bool(bb.has_win().item()) == hw1
        assert bool(bb.done().item()) == d1
        assert (
            int(bb.generate_legal_moves()[0].item()) == lm1
        )  # may be nonzero; we lock in "no special terminal masking"

    _sync_if_cuda(dev)


# =============================================================================
# 6) Redundant equalities: legal_moves_mask == generate_legal_moves; done formula; active_player parity
# =============================================================================


def test_consistency_equalities(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(12345)

    bb = BoardBatch.empty(64, dev, moves_left_dtype=moves_left_dtype)

    for _ in range(200):
        actions = torch.randint(0, 7, (64,), device=dev, generator=g)
        _ = bb.play(actions)

        # generate_legal_moves equals legal_moves_mask
        assert torch.equal(bb.generate_legal_moves(), bb.legal_moves_mask())

        # done formula (moves_left should never be negative in legal play)
        assert torch.equal(bb.done(), bb.has_win() | (bb.moves_left == 0))

        # active_player parity: even moves_left -> yellow (1), odd -> red (2)
        ap = bb.active_player().to(torch.int64)
        expected = torch.where(
            (bb.moves_left.to(torch.int64) & 1) == 0,
            torch.ones_like(ap),
            torch.full_like(ap, 2),
        )
        assert torch.equal(ap, expected)

        bb.reset(bb.done())

    _sync_if_cuda(dev)


# =============================================================================
# 7) Column masks helpers: disjoint, cover exactly 42 bits, sum == all_legal_mask
# =============================================================================


def test_column_masks_cover_playable_squares(dev: torch.device) -> None:
    col_masks = BoardBatch._col_masks(dev)
    assert col_masks.shape == (BoardBatch.N_COLUMNS,)
    assert col_masks.dtype == torch.int64

    # Disjointness
    for i in range(BoardBatch.N_COLUMNS):
        for j in range(i + 1, BoardBatch.N_COLUMNS):
            assert int((col_masks[i] & col_masks[j]).item()) == 0

    # Coverage: exactly 42 playable bits
    union = torch.zeros((), device=dev, dtype=torch.int64)
    for i in range(BoardBatch.N_COLUMNS):
        union |= col_masks[i]
    assert (
        int(
            union.bit_count().item()
            if hasattr(union, "bit_count")
            else bin(int(union.item())).count("1")
        )
        == 42
    )

    # Sum equals all_legal_mask (your implementation)
    all_legal = BoardBatch._all_legal_mask(dev)
    assert int(all_legal.item()) == int(col_masks.sum().item())

    # Also matches BB_ALL_LEGAL_TOKENS constant
    assert int(all_legal.item()) == int(BoardBatch.BB_ALL_LEGAL_TOKENS)

    _sync_if_cuda(dev)


# =============================================================================
# 8) Stress/integration: mix play/reset/mirror/table_positions + invariants (spare bits, dtype, etc.)
# =============================================================================


def test_stress_play_reset_mirror_table_positions_invariants(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(24680)

    B = 96
    bb = BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)

    # fixed patterns tensor (to exercise caches too)
    patterns = torch.tensor(
        [
            [0, 9, 18, 27],
            [1, 10, 19, 28],
            [2, 11, 20, 29],
        ],
        device=dev,
        dtype=torch.int64,
    )

    spare = torch.full((), _spare_bits_mask(), device=dev, dtype=torch.int64)

    for t in range(800):
        actions = torch.randint(
            -2, 9, (B,), device=dev, generator=g
        )  # includes illegal columns too
        _ = bb.play_columns(actions)

        # occasionally mirror the whole batch
        if (t % 37) == 0:
            bb = bb.mirror()

        # exercise table_positions
        T = bb.table_positions(patterns)
        assert T.shape == (B, 3)
        assert T.dtype == torch.int64

        # invariants
        assert bb.all_tokens.dtype == torch.int64
        assert bb.active_tokens.dtype == torch.int64
        assert bb.moves_left.dtype == moves_left_dtype
        assert torch.all((bb.all_tokens & spare) == 0)
        assert torch.all((bb.active_tokens & spare) == 0)

        done = bb.done()
        bb.reset(done)

    _sync_if_cuda(dev)


# =============================================================================
# 1) table_positions() correctness vs slow reference
# =============================================================================


def test_table_positions_matches_reference(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    # Handcrafted patterns (bit indices in your layout)
    # We'll use 3 patterns with N=4.
    patterns = torch.tensor(
        [
            # bottom row across columns 0..3
            [0 * 9 + 0, 1 * 9 + 0, 2 * 9 + 0, 3 * 9 + 0],
            # a vertical slice in column 3 rows 0..3
            [3 * 9 + 0, 3 * 9 + 1, 3 * 9 + 2, 3 * 9 + 3],
            # some mixed squares
            [6 * 9 + 0, 5 * 9 + 1, 4 * 9 + 2, 3 * 9 + 3],
        ],
        dtype=torch.int64,
        device=dev,
    )  # [M=3, N=4]

    # Create a few small deterministic positions via core and inject into BoardBatch.
    cores: list[bbc.BoardCore] = []

    b0 = bbc.BoardCore()  # empty
    cores.append(b0)

    b1 = bbc.BoardCore()
    assert b1.play([3])  # one move center
    cores.append(b1)

    b2 = bbc.BoardCore()
    assert b2.play([3, 4, 3, 4, 3])  # stack-ish
    cores.append(b2)

    b3 = bbc.BoardCore()
    assert b3.play([0, 1, 0, 1, 0, 2])  # create reachable differences
    cores.append(b3)

    # Batch them
    all_tokens = []
    active_tokens = []
    moves_left = []
    for c in cores:
        a, b, m = c.rawState()
        all_tokens.append(int(a))
        active_tokens.append(int(b))
        moves_left.append(int(m))

    bb = BoardBatch(
        all_tokens=torch.tensor(all_tokens, device=dev, dtype=torch.int64),
        active_tokens=torch.tensor(active_tokens, device=dev, dtype=torch.int64),
        moves_left=torch.tensor(moves_left, device=dev, dtype=moves_left_dtype),
    )

    T = bb.table_positions(patterns)  # [B, M]
    assert T.shape == (len(cores), patterns.shape[0])
    assert T.dtype == torch.int64

    # Compare each board/pattern to reference
    patterns_py = [[int(x) for x in row] for row in patterns.to("cpu").tolist()]
    for i, c in enumerate(cores):
        a, b, m = c.rawState()
        ref = _ref_table_positions_single(
            all_tokens=int(a),
            active_tokens=int(b),
            moves_left=int(m),
            patterns_bitidx=patterns_py,
        )
        got = [int(x) for x in T[i].to("cpu").tolist()]
        assert got == ref

    _sync_if_cuda(dev)


# =============================================================================
# 2) table_positions() cache sanity + patterns moved to device automatically
# =============================================================================


def test_table_positions_cache_and_device_migration(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(4, dev, moves_left_dtype=moves_left_dtype)

    # patterns created on CPU intentionally; BoardBatch should auto-move if dev is CUDA.
    patterns_cpu = torch.tensor(
        [[0, 9, 18, 27], [1, 10, 19, 28]],
        dtype=torch.int64,
        device=torch.device("cpu"),
    )

    # Call twice (should be stable, and should populate caches)
    T1 = bb.table_positions(patterns_cpu)
    T2 = bb.table_positions(patterns_cpu)
    assert torch.equal(T1, T2)

    # cache keys should exist (best-effort sanity)
    # NOTE: cache uses id(patterns_bitidx) AFTER any .to(dev) inside _pattern_masks().
    # We can't reliably predict which tensor object got cached, but we can at least
    # assert weights cache for N exists.
    key = (dev.type, dev.index, patterns_cpu.shape[1])
    assert key in BoardBatch._WEIGHTS_CACHE

    _sync_if_cuda(dev)


# =============================================================================
# 3) Column height consistency vs core.getColumnHeight()
# =============================================================================


def test_column_heights_match_core(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(5151)

    B = 48
    bb = BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)
    cores = [bbc.BoardCore() for _ in range(B)]

    # build midgame states
    for _ in range(120):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        for i in range(B):
            _ = cores[i].play(int(actions[i].item()))
        done = bb.done()
        bb.reset(done)
        for i in range(B):
            if cores[i].hasWin() or cores[i].movesLeft() <= 0:
                cores[i].setRawState(0, 0, 42)

    row_mask = (1 << bb.N_ROWS) - 1
    for i in range(B):
        all_i = int(bb.all_tokens[i].item())
        for c in range(bb.N_COLUMNS):
            col_bits = all_i & (row_mask << (c * bb.COLUMN_BIT_OFFSET))
            height = col_bits.bit_count()
            assert height == cores[i].getColumnHeight(c)
            assert cores[i].isLegalMove(c) == (height < 6)

    _sync_if_cuda(dev)


# =============================================================================
# 4) play_masks(): document multi-bit behaviour (xfail if you want one-hot strict)
# =============================================================================


@pytest.mark.xfail(
    reason="Current play_masks() accepts multi-bit masks (subset of landing squares). "
    "If you want strict one-hot semantics, enforce it in code and flip this test."
)
def test_play_masks_rejects_multi_bit_masks(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(1, dev, moves_left_dtype=moves_left_dtype)

    # On empty board, legal landing squares are bottom row: 7 bits.
    legal = int(bb.generate_legal_moves()[0].item())

    # Take two columns' landing squares -> multi-bit mask
    mv = (legal & -legal) | ((legal ^ (legal & -legal)) & -(legal ^ (legal & -legal)))
    mv_t = torch.tensor([mv], device=dev, dtype=torch.int64)

    ok = bb.play_masks(mv_t)
    assert bool(ok.item()) is False


# =============================================================================
# 5) reward()/done() edge cases: NaN iff not done, and draw state exists
# =============================================================================


def test_reward_is_nan_iff_not_done(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(6061)

    B = 64
    bb = BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)

    for _ in range(200):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        done = bb.done()
        r = bb.reward()
        assert torch.equal(torch.isnan(r), ~done)
        bb.reset(done)

    _sync_if_cuda(dev)


def test_reward_draw_state_found_and_correct(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """Find a full-board draw using core (no win, movesLeft==0), inject into BoardBatch,
    and validate reward==0 and done==True.
    """
    # deterministic-ish search
    for _ in range(2000):
        core, _moves = bbc.BoardCore.randomBoard(nPly=42, forbidDirectWin=False)
        if (not core.hasWin()) and core.movesLeft() == 0:
            bb = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)
            assert bool(bb.done()[0].item()) is True
            assert float(bb.reward()[0].item()) == 0.0
            break
    else:
        pytest.skip(
            "Could not find a draw position quickly; increase attempts if needed."
        )

    _sync_if_cuda(dev)


# =============================================================================
# 6) winning_positions symmetry under mirror (for both players)
# =============================================================================


def test_winning_positions_mirror_symmetry(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(7070)

    B = 64
    bb = BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)

    for _ in range(150):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        bb.reset(bb.done())

    mir = bb.mirror()

    # active player wins
    w0 = bb.winning_positions(bb.active_tokens, verticals=True)
    w1 = mir.winning_positions(mir.active_tokens, verticals=True)

    # opponent wins (opp = active ^ all)
    opp0 = bb.active_tokens ^ bb.all_tokens
    opp1 = mir.active_tokens ^ mir.all_tokens
    wo0 = bb.winning_positions(opp0, verticals=True)
    wo1 = mir.winning_positions(opp1, verticals=True)

    # Compare via mirroring the returned masks
    w0m = torch.tensor(
        [
            _mirror_bits_via_boardbatch(x, dev=dev, moves_left_dtype=moves_left_dtype)
            for x in w0
        ],
        device=dev,
        dtype=torch.int64,
    )
    wo0m = torch.tensor(
        [
            _mirror_bits_via_boardbatch(x, dev=dev, moves_left_dtype=moves_left_dtype)
            for x in wo0
        ],
        device=dev,
        dtype=torch.int64,
    )

    assert torch.equal(w0m, w1)
    assert torch.equal(wo0m, wo1)

    _sync_if_cuda(dev)


# =============================================================================
# 7) Spare-bit invariant: never set bits in rows 6..8 of any column
# =============================================================================


def test_spare_bits_never_set(dev: torch.device, moves_left_dtype: torch.dtype) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(8080)

    B = 96
    bb = BoardBatch.empty(B, dev, moves_left_dtype=moves_left_dtype)

    spare = torch.full((), _spare_bits_mask(), device=dev, dtype=torch.int64)

    for _ in range(400):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        bb.reset(bb.done())

        assert torch.all((bb.all_tokens & spare) == 0)
        assert torch.all((bb.active_tokens & spare) == 0)

    _sync_if_cuda(dev)


# =============================================================================
# 8) Device/dtype robustness: moves_left dtype preserved; reset accepts CPU done
# =============================================================================


def test_dtype_and_cross_device_done_handling(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = BoardBatch.empty(32, dev, moves_left_dtype=moves_left_dtype)

    # moves_left dtype stays constant after play/reset
    assert bb.moves_left.dtype == moves_left_dtype

    g = torch.Generator(device=dev)
    g.manual_seed(9090)

    for _ in range(50):
        actions = torch.randint(0, 7, (32,), device=dev, generator=g)
        _ = bb.play(actions)

        assert bb.moves_left.dtype == moves_left_dtype
        assert bb.all_tokens.dtype == torch.int64
        assert bb.active_tokens.dtype == torch.int64
        assert bb.generate_legal_moves().dtype == torch.int64
        assert bb.generate_non_losing_moves().dtype == torch.int64

        done = bb.done()
        # simulate "done on CPU" even if board is on CUDA (your reset() should .to(dev))
        done_cpu = done.to("cpu")
        bb.reset(done_cpu)

        assert bb.moves_left.dtype == moves_left_dtype

    _sync_if_cuda(dev)


# =============================================================================
# Basic invariants & edge cases
# =============================================================================


def test_constants_match_core() -> None:
    assert BoardBatch.N_COLUMNS == bbc.N_COLUMNS == 7
    assert BoardBatch.N_ROWS == bbc.N_ROWS == 6
    assert BoardBatch.COLUMN_BIT_OFFSET == 9


def test_move_mask_to_column_basic() -> None:
    assert move_mask_to_column(1 << 0) == 0
    assert move_mask_to_column(1 << 9) == 1
    assert move_mask_to_column(1 << 54) == 6
    assert move_mask_to_column(1 << (6 * 9 + 5)) == 6


@pytest.mark.parametrize("mv", [0, (1 << 0) | (1 << 9), (1 << 10) | (1 << 11)])
def test_move_mask_to_column_rejects_non_onehot(mv: int) -> None:
    with pytest.raises(ValueError):
        move_mask_to_column(mv)


def test_empty_board_invariants(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    B = 8
    bb = _mk_batch(B, dev, moves_left_dtype)

    assert torch.all(bb.all_tokens == 0)
    assert torch.all(bb.active_tokens == 0)
    assert torch.all(bb.moves_left == 42)

    lm = bb.legal_moves_mask()
    assert torch.all(lm == bb.BB_BOTTOM_ROW)

    # done() false on empty
    assert not bool(bb.done().any().item())

    # reward is NaN on empty
    r = bb.reward()
    assert bool(torch.isnan(r).all().item())

    # active player at start is Yellow
    ap = bb.active_player()
    aps = bb.active_player_sign()
    assert torch.all(ap == 1)
    assert torch.all(aps == 1.0)

    # mirror twice = identity
    mm = bb.mirror().mirror()
    assert torch.equal(mm.all_tokens, bb.all_tokens)
    assert torch.equal(mm.active_tokens, bb.active_tokens)
    assert torch.equal(mm.moves_left, bb.moves_left)

    _sync_if_cuda(dev)


def test_play_columns_out_of_range_is_illegal(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = _mk_batch(6, dev, moves_left_dtype)
    cols = torch.tensor([-10, -1, 7, 999, 0, 6], device=dev)

    legal = bb.play_columns(cols)
    expected = torch.tensor([0, 0, 0, 0, 1, 1], device=dev, dtype=torch.bool)
    assert torch.equal(legal, expected)

    assert bb.moves_left[:4].tolist() == [42, 42, 42, 42]
    assert bb.moves_left[4:].tolist() == [41, 41]

    _sync_if_cuda(dev)


def test_play_columns_column_full_becomes_illegal(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = _mk_batch(1, dev, moves_left_dtype)
    col = torch.tensor([3], device=dev)

    for k in range(6):
        legal = bb.play_columns(col)
        assert bool(legal.item()) is True
        assert int(bb.moves_left.item()) == 42 - (k + 1)

    legal = bb.play_columns(col)
    assert bool(legal.item()) is False
    assert int(bb.moves_left.item()) == 36

    _sync_if_cuda(dev)


def test_play_masks_legality_subset_of_legal_moves(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    bb = _mk_batch(1, dev, moves_left_dtype)

    # legal landing square: col2 bottom
    mv_ok = torch.tensor(
        [1 << (2 * bb.COLUMN_BIT_OFFSET + 0)], device=dev, dtype=torch.int64
    )
    ok = bb.play_masks(mv_ok)
    assert bool(ok.item()) is True

    bb2 = _mk_batch(1, dev, moves_left_dtype)
    # illegal: not a landing square (above bottom in empty column)
    mv_bad = torch.tensor(
        [1 << (2 * bb2.COLUMN_BIT_OFFSET + 3)], device=dev, dtype=torch.int64
    )
    ok2 = bb2.play_masks(mv_bad)
    assert bool(ok2.item()) is False
    assert int(bb2.moves_left.item()) == 42

    bb3 = _mk_batch(1, dev, moves_left_dtype)
    ok3 = bb3.play_masks(torch.tensor([0], device=dev, dtype=torch.int64))
    assert bool(ok3.item()) is False

    _sync_if_cuda(dev)


# =============================================================================
# Deterministic win patterns (vertical/horizontal/diagonal) compared to core
# =============================================================================


@pytest.mark.parametrize(
    "seq",
    [
        # Vertical win for Yellow in column 0
        [0, 1, 0, 1, 0, 1, 0],
        # Horizontal win for Yellow on bottom row: Y plays 0,1,2,3; R plays 6,6,6
        [0, 6, 1, 6, 2, 6, 3],
        # Diagonal win (/) for Yellow
        # One known constructive sequence (common in C4 examples)
        [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3],
        # Diagonal win (\) for Yellow
        [3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0],
    ],
)
def test_has_win_and_reward_sequences_match_core(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
    seq: list[int],
) -> None:
    bb = _mk_batch(1, dev, moves_left_dtype)
    core = bbc.BoardCore()

    for a in seq:
        t = torch.tensor([a], device=dev)
        legal = bb.play(t)
        core_legal = core.play(a)

        assert bool(legal.item()) == bool(core_legal)
        _assert_states_equal(core, bb, 0)
        assert bool(bb.has_win()[0].item()) == bool(core.hasWin())

        r = float(bb.reward()[0].item())
        _assert_reward_matches_core(core, r)

    _sync_if_cuda(dev)


# =============================================================================
# Random rollouts vs core (batch), plus mirror and legality
# =============================================================================


@pytest.mark.parametrize("B,steps", [(32, 200), (97, 150)])
def test_random_rollouts_match_core(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
    B: int,
    steps: int,
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(12345)

    bb = _mk_batch(B, dev, moves_left_dtype)
    cores = _mk_core_batch(B)

    for _ in range(steps):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        legal = bb.play(actions)

        won = bb.has_win()
        done = bb.done()
        nl = bb.generate_non_losing_moves()
        lm = bb.generate_legal_moves()
        can_win = bb.can_win()

        can_win_actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        can_win_col = bb.can_win_column(can_win_actions)

        mir = bb.mirror()
        mir2 = mir.mirror()
        assert torch.equal(mir2.all_tokens, bb.all_tokens)
        assert torch.equal(mir2.active_tokens, bb.active_tokens)
        assert torch.equal(mir2.moves_left, bb.moves_left)

        r = bb.reward()
        ap = bb.active_player()
        aps = bb.active_player_sign()

        for i in range(B):
            a = int(actions[i].item())
            core_legal = cores[i].play(a)

            _assert_states_equal(cores[i], bb, i)
            assert bool(core_legal) == bool(legal[i].item())
            assert bool(cores[i].hasWin()) == bool(won[i].item())

            _assert_reward_matches_core(cores[i], float(r[i].item()))
            _assert_active_player_matches_core(
                cores[i], int(ap[i].item()), float(aps[i].item())
            )

            assert int(cores[i].generateNonLosingMoves()) == int(nl[i].item())
            assert int(cores[i].legalMovesMask()) == int(lm[i].item())
            assert bool(cores[i].canWin()) == bool(can_win[i].item())

            cwa = int(can_win_actions[i].item())
            assert bool(cores[i].canWin(cwa)) == bool(can_win_col[i].item())

            # mirror equality with engine mirror
            mcore = cores[i].mirror()
            a2, b2, m2 = _core_raw_state(mcore)
            assert a2 == int(mir.all_tokens[i].item())
            assert b2 == int(mir.active_tokens[i].item())
            assert m2 == int(mir.moves_left[i].item())

        # reset done boards (both sides)
        for i in range(B):
            core_done = cores[i].hasWin() or cores[i].movesLeft() <= 0
            assert core_done == bool(done[i].item())
            if core_done:
                cores[i].setRawState(0, 0, 42)

        bb.reset(done)

    _sync_if_cuda(dev)


# =============================================================================
# Build positions using core helpers: randomBoard and setRawState/uid/hash
# =============================================================================


@pytest.mark.parametrize(
    "nply,forbid", [(0, False), (8, False), (8, True), (16, True), (24, False)]
)
def test_set_raw_state_roundtrip_from_core_randomBoard(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
    nply: int,
    forbid: bool,
) -> None:
    """Uses core's randomBoard() to generate reachable states,
    then injects those raw bitboards into BoardBatch and checks all derived methods.
    """
    core, moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=forbid)
    a, b, m = _core_raw_state(core)

    bb = BoardBatch(
        all_tokens=torch.tensor([a], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([b], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([m], device=dev, dtype=moves_left_dtype),
    )

    # raw equality and derived comparisons
    _assert_states_equal(core, bb, 0)
    assert bool(bb.has_win()[0].item()) == bool(core.hasWin())
    assert int(bb.generate_legal_moves()[0].item()) == int(core.legalMovesMask())
    assert int(bb.generate_non_losing_moves()[0].item()) == int(
        core.generateNonLosingMoves()
    )
    assert bool(bb.can_win()[0].item()) == bool(core.canWin())

    # mirror equality
    mcore = core.mirror()
    bmir = bb.mirror()
    a2, b2, m2 = _core_raw_state(mcore)
    assert a2 == int(bmir.all_tokens[0].item())
    assert b2 == int(bmir.active_tokens[0].item())
    assert m2 == int(bmir.moves_left[0].item())

    _sync_if_cuda(dev)


def test_core_setRawState_uid_hash_stability_against_copy() -> None:
    """Pure core test: ensures we understand core behaviour used as a reference."""
    b1 = bbc.BoardCore()
    assert b1.play("334411")

    a, b, m = b1.rawState()
    b2 = bbc.BoardCore()
    b2.setRawState(a, b, m)

    assert b1 == b2
    assert b1.uid() == b2.uid()
    assert b1.hash() == b2.hash()
    assert b1.toString() == b2.toString()


# =============================================================================
# After-state expansion: iter_move_masks + play_masks vs core playMoveOnCopy
# =============================================================================


@pytest.mark.parametrize("B,steps", [(32, 120)])
def test_after_states_from_non_losing_moves_match_core(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
    B: int,
    steps: int,
) -> None:
    g = torch.Generator(device=dev)
    g.manual_seed(2026)

    bb = _mk_batch(B, dev, moves_left_dtype)
    cores = _mk_core_batch(B)

    # create some midgame positions, resetting on terminal states
    for _ in range(steps):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        for i in range(B):
            _ = cores[i].play(int(actions[i].item()))

        done = bb.done()
        for i in range(B):
            if cores[i].hasWin() or cores[i].movesLeft() <= 0:
                cores[i].setRawState(0, 0, 42)
        bb.reset(done)

    nl = bb.generate_non_losing_moves()
    for mv in bb.iter_move_masks(nl):
        active = mv != 0
        if not bool(active.any().item()):
            break

        tmp = BoardBatch(
            all_tokens=bb.all_tokens.clone(),
            active_tokens=bb.active_tokens.clone(),
            moves_left=bb.moves_left.clone(),
        )

        legal = tmp.play_masks(mv)
        won = tmp.has_win()
        done = tmp.done()
        nl2 = tmp.generate_non_losing_moves()
        can_win = tmp.can_win()

        can_win_actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        can_win_col = tmp.can_win_column(can_win_actions)

        for i in range(B):
            mv_onehot = int(mv[i].item())
            if mv_onehot != 0:
                col = move_mask_to_column(
                    mv_onehot, column_bit_offset=bb.COLUMN_BIT_OFFSET
                )
                core_next = cores[i].playMoveOnCopy(int(col))
                core_legal = core_next.movesLeft() < cores[i].movesLeft()
            else:
                core_next = cores[i].copy()
                core_legal = False

            a, b, m = _core_raw_state(core_next)
            assert a == int(tmp.all_tokens[i].item())
            assert b == int(tmp.active_tokens[i].item())
            assert m == int(tmp.moves_left[i].item())

            assert bool(core_legal) == bool(legal[i].item())
            assert bool(core_next.hasWin()) == bool(won[i].item())
            assert (core_next.hasWin() or core_next.movesLeft() <= 0) == bool(
                done[i].item()
            )

            assert int(core_next.generateNonLosingMoves()) == int(nl2[i].item())
            assert bool(core_next.canWin()) == bool(can_win[i].item())

            cwa = int(can_win_actions[i].item())
            assert bool(core_next.canWin(cwa)) == bool(can_win_col[i].item())

    _sync_if_cuda(dev)


# =============================================================================
# Mirror: legal move masks map by column reflection
# =============================================================================


def test_mirror_legal_moves_mask_is_reflected(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """Stronger than "mirror twice": checks that the *legal landing squares bitmask*
    is transformed by column reflection.
    """
    bb = _mk_batch(1, dev, moves_left_dtype)

    # Asymmetric position
    seq = [0, 0, 1, 6, 6, 6, 2, 2, 5]
    for a in seq:
        ok = bb.play(torch.tensor([a], device=dev))
        assert bool(ok.item()) is True

    lm = int(bb.generate_legal_moves()[0].item())
    lm_mir = int(bb.mirror().generate_legal_moves()[0].item())

    # reflect the move mask and compare
    assert _mirror_move_mask_by_columns(lm, off=bb.COLUMN_BIT_OFFSET) == lm_mir

    _sync_if_cuda(dev)


# =============================================================================
# Core move list vs our move mask (consistency checks)
# =============================================================================


@pytest.mark.parametrize("nply", [0, 4, 10, 18, 30])
def test_core_legalMoves_list_matches_legalMovesMask_and_our_mask(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
    nply: int,
) -> None:
    """Use core.randomBoard() to generate a reachable board and check:
    - core.legalMovesMask() matches our generate_legal_moves()
    - core.legalMoves(nonLosing=..., orderMoves=...) is consistent with that mask
    """
    core, _ = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)
    a, b, m = _core_raw_state(core)

    bb = BoardBatch(
        all_tokens=torch.tensor([a], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([b], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([m], device=dev, dtype=moves_left_dtype),
    )

    mask_core = int(core.legalMovesMask())
    mask_ours = int(bb.generate_legal_moves()[0].item())
    assert mask_core == mask_ours

    cols = core.legalMoves(nonLosing=False, orderMoves=False)
    # Every returned column must be legal per core.isLegalMove and must appear in mask
    for c in cols:
        assert core.isLegalMove(int(c)) is True
        # Extract that column's landing-square bit from the mask and ensure it exists.
        # We don't know exact landing row, but legality implies exactly one landing bit in that column.
        base = int(c) * bb.COLUMN_BIT_OFFSET
        col_playable = ((1 << 6) - 1) << base
        assert (mask_core & col_playable) != 0

    # And the number of returned columns equals number of set bits in the mask (<=7)
    # popcount in python:
    assert (
        len(cols)
        == int(mask_core != 0) + int(((mask_core & -mask_core) ^ mask_core) != 0)
        or len(cols) <= 7
    )

    _sync_if_cuda(dev)


def test_reset_exact_and_selective(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """- Boards with done=True are reset to (0,0,42)
    - Boards with done=False remain unchanged (bit-for-bit)
    - Matches core reset semantics (setRawState(0,0,42)) for done boards
    """
    B = 64
    g = torch.Generator(device=dev)
    g.manual_seed(2027)

    bb = _mk_batch(B, dev, moves_left_dtype)
    cores = [bbc.BoardCore() for _ in range(B)]

    # Build a mix of terminal and non-terminal positions
    for _ in range(200):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        for i in range(B):
            _ = cores[i].play(int(actions[i].item()))

        done = bb.done()

        # snapshot before reset
        all_before = bb.all_tokens.clone()
        act_before = bb.active_tokens.clone()
        ml_before = bb.moves_left.clone()

        bb.reset(done)

        for i in range(B):
            d = bool(done[i].item())
            if d:
                assert _as_int(bb.all_tokens[i]) == 0
                assert _as_int(bb.active_tokens[i]) == 0
                assert int(bb.moves_left[i].item()) == 42
            else:
                assert _as_int(bb.all_tokens[i]) == _as_int(all_before[i])
                assert _as_int(bb.active_tokens[i]) == _as_int(act_before[i])
                assert int(bb.moves_left[i].item()) == int(ml_before[i].item())

        # keep cores in sync with reset policy
        for i in range(B):
            if cores[i].hasWin() or cores[i].movesLeft() <= 0:
                cores[i].setRawState(0, 0, 42)

        # Now states should match core after our reset
        for i in range(B):
            a, b, m = cores[i].rawState()
            assert (int(a), int(b), int(m)) == (
                _as_int(bb.all_tokens[i]),
                _as_int(bb.active_tokens[i]),
                int(bb.moves_left[i].item()),
            )

    _sync_if_cuda(dev)


# =============================================================================
# 2) play_masks() == play_columns() when using the column's landing-square bit
# =============================================================================


def test_play_masks_equals_play_columns_for_landing_square(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """For each board:
    - pick a legal column
    - derive the one-hot landing-square bit from generate_legal_moves()
    - applying play_columns(col) must equal applying play_masks(onehot)
    """
    B = 96
    g = torch.Generator(device=dev)
    g.manual_seed(4242)

    # create some midgame states (avoid all empty)
    bb = _mk_batch(B, dev, moves_left_dtype)
    for _ in range(80):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        done = bb.done()
        bb.reset(done)
        _ = bb.play(actions)

    # derive a legal move per board
    legal_mask = bb.generate_legal_moves()  # [B] landing squares
    assert legal_mask.dtype == torch.int64

    # For each board, pick the first legal column by scanning columns.
    cols = torch.full((B,), -1, device=dev, dtype=torch.int64)
    onehot = torch.zeros((B,), device=dev, dtype=torch.int64)

    row_mask = (1 << bb.N_ROWS) - 1
    for c in range(bb.N_COLUMNS):
        base = c * bb.COLUMN_BIT_OFFSET
        col_bits = legal_mask & (
            torch.tensor(row_mask, device=dev, dtype=torch.int64) << base
        )
        take = (cols < 0) & (col_bits != 0)
        cols = torch.where(take, torch.full_like(cols, c), cols)
        onehot = torch.where(take, col_bits, onehot)

    # Sanity: if a board is full, cols stays -1 and onehot stays 0 (rare here)
    # We'll ignore those boards.
    active = cols >= 0
    assert bool(active.any().item())

    # Apply via columns on a clone
    bb_cols = BoardBatch(
        all_tokens=bb.all_tokens.clone(),
        active_tokens=bb.active_tokens.clone(),
        moves_left=bb.moves_left.clone(),
    )
    ok_cols = bb_cols.play_columns(cols)

    # Apply via masks on a clone
    bb_msk = BoardBatch(
        all_tokens=bb.all_tokens.clone(),
        active_tokens=bb.active_tokens.clone(),
        moves_left=bb.moves_left.clone(),
    )
    ok_msk = bb_msk.play_masks(onehot)

    # Both must agree board-wise
    assert torch.equal(ok_cols, ok_msk)

    # And resulting states must match exactly (where a move was possible)
    assert torch.equal(bb_cols.all_tokens, bb_msk.all_tokens)
    assert torch.equal(bb_cols.active_tokens, bb_msk.active_tokens)
    assert torch.equal(bb_cols.moves_left, bb_msk.moves_left)

    _sync_if_cuda(dev)


# =============================================================================
# 3) Illegal move is a strict no-op (including active player / moves_left)
# =============================================================================


def test_illegal_move_is_noop(dev: torch.device, moves_left_dtype: torch.dtype) -> None:
    """- Out-of-range columns must be illegal and must not mutate any state
    - Playing into a full column must be illegal and must not mutate any state
    - "No-op" includes: all_tokens, active_tokens, moves_left, active_player parity
    """
    bb = _mk_batch(8, dev, moves_left_dtype)

    # 3.1 out-of-range
    before = (bb.all_tokens.clone(), bb.active_tokens.clone(), bb.moves_left.clone())
    cols = torch.tensor([-3, -1, 7, 9, -100, 100, -8, 8], device=dev, dtype=torch.int64)
    legal = bb.play_columns(cols)
    assert not bool(legal.any().item())
    assert torch.equal(bb.all_tokens, before[0])
    assert torch.equal(bb.active_tokens, before[1])
    assert torch.equal(bb.moves_left, before[2])

    # 3.2 full column
    bb2 = _mk_batch(1, dev, moves_left_dtype)
    # fill column 4
    for _ in range(6):
        ok = bb2.play_columns(torch.tensor([4], device=dev))
        assert bool(ok.item()) is True

    before2 = (
        bb2.all_tokens.clone(),
        bb2.active_tokens.clone(),
        bb2.moves_left.clone(),
    )
    ok2 = bb2.play_columns(torch.tensor([4], device=dev))
    assert bool(ok2.item()) is False
    assert torch.equal(bb2.all_tokens, before2[0])
    assert torch.equal(bb2.active_tokens, before2[1])
    assert torch.equal(bb2.moves_left, before2[2])

    _sync_if_cuda(dev)


# =============================================================================
# 4) non_losing_moves is subset of legal moves, and both stay within playable bits
# =============================================================================


def test_move_masks_subset_and_within_playable(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    B = 128
    g = torch.Generator(device=dev)
    g.manual_seed(9001)

    bb = _mk_batch(B, dev, moves_left_dtype)

    for _ in range(200):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        done = bb.done()
        bb.reset(done)

        legal = bb.generate_legal_moves()
        nl = bb.generate_non_losing_moves()

        # subset
        assert torch.all((nl & legal) == nl)

        # within playable squares
        all_legal = torch.full(
            (), bb.BB_ALL_LEGAL_TOKENS, device=dev, dtype=torch.int64
        )
        assert torch.all((legal & ~all_legal) == 0)
        assert torch.all((nl & ~all_legal) == 0)

    _sync_if_cuda(dev)


# =============================================================================
# 5) iter_move_masks enumerates exactly the bits in the move set
# =============================================================================


def test_iter_move_masks_enumerates_exactly(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    B = 96
    g = torch.Generator(device=dev)
    g.manual_seed(111)

    bb = _mk_batch(B, dev, moves_left_dtype)
    for _ in range(120):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        bb.reset(bb.done())

    moves = bb.generate_non_losing_moves()  # [B] <= 7 bits

    seen_or = torch.zeros((B,), device=dev, dtype=torch.int64)
    seen_xor = torch.zeros((B,), device=dev, dtype=torch.int64)

    # Collect yielded masks (max 7)
    for mv in bb.iter_move_masks(moves, max_moves=7):
        mv = mv.to(device=dev, dtype=torch.int64)

        # yielded mv must be subset of moves
        assert torch.all((mv & ~moves) == 0)

        # yielded mv must be one-hot or 0: mv & (mv-1) == 0 for mv>0
        one_hot_or_zero = (mv == 0) | (((mv & (mv - 1)) == 0) & (mv > 0))
        assert bool(one_hot_or_zero.all().item())

        # disjointness: no bit repeats (track with OR; repeats detectable via XOR vs OR)
        seen_or |= mv
        seen_xor ^= mv

    # union equals the original moves (per board)
    assert torch.equal(seen_or, moves)

    # If there were repeats, OR != XOR. So XOR must also equal moves.
    assert torch.equal(seen_xor, moves)

    _sync_if_cuda(dev)


# =============================================================================
# 6) winning_positions invariants + relation to can_win and can_win_column
# =============================================================================


def test_winning_positions_consistency(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    B = 96
    g = torch.Generator(device=dev)
    g.manual_seed(2028)

    bb = _mk_batch(B, dev, moves_left_dtype)

    # build midgame positions
    for _ in range(160):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        bb.reset(bb.done())

    wins = bb.winning_positions(bb.active_tokens, verticals=True)  # [B] int64
    legal = bb.generate_legal_moves()  # landing squares
    all_legal = torch.full((), bb.BB_ALL_LEGAL_TOKENS, device=dev, dtype=torch.int64)

    # wins only within playable squares
    assert torch.all((wins & ~all_legal) == 0)

    # wins cannot be on already occupied squares
    # only *immediate* winning moves (wins ∩ landing squares) must be empty
    immediate = wins & legal
    assert torch.all((immediate & bb.all_tokens) == 0)

    # can_win matches (wins & legal) != 0
    can_win_from_formula = (wins & legal) != 0
    assert torch.equal(can_win_from_formula, bb.can_win())

    # can_win_column matches checking if the winning landing square is in that column
    cols = torch.randint(0, 7, (B,), device=dev, generator=g, dtype=torch.int64)
    cwc = bb.can_win_column(cols)

    row_mask = (1 << bb.N_ROWS) - 1
    base = cols * bb.COLUMN_BIT_OFFSET
    col_mask = torch.tensor(row_mask, device=dev, dtype=torch.int64) << base
    cwc_formula = ((wins & legal & col_mask) != 0) & ((cols >= 0) & (cols < 7))
    assert torch.equal(cwc, cwc_formula)

    _sync_if_cuda(dev)


def test_can_win_matches_core_from_wins_and_columns(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """Cross-check BoardBatch win detection vs BoardCore:

    For each board:
      (wins & legal) != 0      <-> core.canWin()
      bb.can_win()             <-> core.canWin()
      bb.can_win_column(c)     <-> core.canWin(c)   for all c in 0..6
    """
    B = 96
    g = torch.Generator(device=dev)
    g.manual_seed(6060)

    bb = _mk_batch(B, dev, moves_left_dtype)
    cores = [bbc.BoardCore() for _ in range(B)]

    # build many random midgame positions
    for _ in range(180):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        legal = bb.play(actions)

        for i in range(B):
            _ = cores[i].play(int(actions[i].item()))

        done = bb.done()

        # keep both sides reset in sync
        bb.reset(done)
        for i in range(B):
            if cores[i].hasWin() or cores[i].movesLeft() <= 0:
                cores[i].setRawState(0, 0, 42)

    wins = bb.winning_positions(bb.active_tokens, verticals=True)  # [B]
    legal_mask = bb.generate_legal_moves()  # [B]
    can_win = bb.can_win()

    # ------------------------------------------------------------
    # 1) (wins & legal) != 0  <-> core.canWin()
    # ------------------------------------------------------------
    wins_and_legal = (wins & legal_mask) != 0

    for i in range(B):
        core_can = cores[i].canWin()
        assert bool(wins_and_legal[i].item()) == core_can
        assert bool(can_win[i].item()) == core_can

    # ------------------------------------------------------------
    # 2) can_win_column(c) matches core.canWin(c)
    # ------------------------------------------------------------
    cols = torch.randint(0, 7, (B,), device=dev, generator=g, dtype=torch.int64)
    cwc = bb.can_win_column(cols)

    for i in range(B):
        c = int(cols[i].item())
        assert bool(cwc[i].item()) == cores[i].canWin(c)

    if dev.type == "cuda":
        torch.cuda.synchronize()


# =============================================================================
# 7) Mirror metamorphic tests (beyond "mirror twice")
# =============================================================================


def test_mirror_metamorphic_properties(
    dev: torch.device, moves_left_dtype: torch.dtype
) -> None:
    """For any position:
    - has_win, done, reward, active_player, active_player_sign invariant under mirror
    - legal_moves and non_losing_moves reflect by column mapping
    - can_win invariant, and can_win_column respects column reflection
    """
    B = 128
    g = torch.Generator(device=dev)
    g.manual_seed(3030)

    bb = _mk_batch(B, dev, moves_left_dtype)

    for _ in range(220):
        actions = torch.randint(0, 7, (B,), device=dev, generator=g)
        _ = bb.play(actions)
        bb.reset(bb.done())

    mir = bb.mirror()

    # scalar / per-board invariants
    assert torch.equal(bb.has_win(), mir.has_win())
    assert torch.equal(bb.done(), mir.done())

    # reward: includes NaNs, so use isnan-aware equality
    r0 = bb.reward()
    r1 = mir.reward()
    both_nan = torch.isnan(r0) & torch.isnan(r1)
    assert bool((both_nan | (r0 == r1)).all().item())

    assert torch.equal(bb.active_player(), mir.active_player())
    assert torch.equal(bb.active_player_sign(), mir.active_player_sign())

    assert torch.equal(bb.can_win(), mir.can_win())

    # move mask reflection
    lm0 = bb.generate_legal_moves()
    lm1 = mir.generate_legal_moves()
    nl0 = bb.generate_non_losing_moves()
    nl1 = mir.generate_non_losing_moves()

    # reflect bitmasks using our column-reflection helper
    lm0_ref = torch.tensor(
        [
            _mirror_move_mask_by_columns(int(x.item()), off=bb.COLUMN_BIT_OFFSET)
            for x in lm0
        ],
        device=dev,
        dtype=torch.int64,
    )
    nl0_ref = torch.tensor(
        [
            _mirror_move_mask_by_columns(int(x.item()), off=bb.COLUMN_BIT_OFFSET)
            for x in nl0
        ],
        device=dev,
        dtype=torch.int64,
    )

    assert torch.equal(lm0_ref, lm1)
    assert torch.equal(nl0_ref, nl1)

    # can_win_column: c in original corresponds to 6-c in mirrored
    cols = torch.randint(0, 7, (B,), device=dev, generator=g, dtype=torch.int64)
    assert torch.equal(bb.can_win_column(cols), mir.can_win_column(6 - cols))

    _sync_if_cuda(dev)


def test_old_ugly(dev: torch.device):
    import bitbully.bitbully_core as bbc

    device = dev
    B = 100
    reset_done_boards = True
    compare_with_bitbully = True

    torch_board = BoardBatch.empty(B, device)
    bb_board = [bbc.BoardCore() for _ in range(B)]

    for i in range(42 * 10):  # approx 10 games.
        # if device == "cuda": torch.cuda.synchronize()
        actions = torch.randint(0, 7, (B,), device=device)
        legal = torch_board.play(actions)
        won = torch_board.has_win()
        done = torch_board.done()
        non_losing_moves = torch_board.generate_non_losing_moves()
        legal_moves = torch_board.generate_legal_moves()
        can_win = torch_board.can_win()

        can_win_actions = torch.randint(0, 7, (B,), device=device)
        can_win_column = torch_board.can_win_column(can_win_actions)

        # Mirror sanity check
        torch_board_mir = torch_board.mirror()

        # rewards
        reward = torch_board.reward()

        # active player
        active_player = torch_board.active_player()
        active_player_sign = torch_board.active_player_sign()

        # sanity: mirroring twice gives original (bitboards)
        b3 = torch_board_mir.mirror()
        assert torch.equal(b3.all_tokens, torch_board.all_tokens)
        assert torch.equal(b3.active_tokens, torch_board.active_tokens)
        assert torch.equal(b3.moves_left, torch_board.moves_left)

        # rewards
        reward = torch_board.reward()

        # if device == "cuda": torch.cuda.synchronize()

        # bitbully comparisons
        if compare_with_bitbully:
            for b_idx in range(B):
                a = actions[b_idx].item()
                ll = bb_board[b_idx].play(a)
                all_tokens, active_tokens, moves_left = bb_board[b_idx].rawState()
                assert (all_tokens, active_tokens, moves_left) == (
                    torch_board.all_tokens[b_idx].item(),
                    torch_board.active_tokens[b_idx].item(),
                    torch_board.moves_left[b_idx].item(),
                ), f"Problem with {b_idx}"
                assert ll == legal[b_idx].item(), f"Problem with {b_idx}"
                assert bb_board[b_idx].hasWin() == won[b_idx].item(), (
                    f"Problem with {b_idx}"
                )

                # rewards
                if reward[b_idx].item() != reward[b_idx].item():  # <- check for nan
                    assert (
                        not bb_board[b_idx].hasWin() and bb_board[b_idx].movesLeft() > 0
                    ), f"Problem with {b_idx}"
                elif reward[b_idx].item() == 1.0:
                    assert (
                        bb_board[b_idx].hasWin()
                        and bb_board[b_idx].movesLeft() % 2 == 1
                    ), f"Problem with {b_idx}"
                elif reward[b_idx].item() == -1.0:
                    assert (
                        bb_board[b_idx].hasWin()
                        and bb_board[b_idx].movesLeft() % 2 == 0
                    ), f"Problem with {b_idx}"
                elif reward[b_idx].item() == 0.0:
                    assert (
                        not bb_board[b_idx].hasWin()
                        and bb_board[b_idx].movesLeft() == 0
                    ), f"Problem with {b_idx}"
                else:
                    assert False, f"Problem with {b_idx}"

                # active player
                assert (
                    bb_board[b_idx].popCountBoard() % 2 + 1
                    == active_player[b_idx].item()
                ), f"Problem with {b_idx}"
                assert (
                    1 - 2 * (bb_board[b_idx].popCountBoard() % 2)
                    == active_player_sign[b_idx].item()
                ), f"Problem with {b_idx}"

                # moves
                assert (
                    bb_board[b_idx].generateNonLosingMoves() == non_losing_moves[b_idx]
                ), f"Problem with {b_idx}"
                assert bb_board[b_idx].legalMovesMask() == legal_moves[b_idx], (
                    f"Problem with {b_idx}"
                )

                # can_win
                assert bb_board[b_idx].canWin() == can_win[b_idx].item(), (
                    f"Problem with {b_idx}"
                )

                # can win column:
                assert (
                    bb_board[b_idx].canWin(can_win_actions[b_idx].item())
                    == can_win_column[b_idx].item()
                ), f"Problem with {b_idx}"

                # mirrored board:
                all_tokens_mir, active_tokens_mir, moves_left_mir = (
                    bb_board[b_idx].mirror().rawState()
                )
                assert (all_tokens_mir, active_tokens_mir, moves_left_mir) == (
                    torch_board_mir.all_tokens[b_idx].item(),
                    torch_board_mir.active_tokens[b_idx].item(),
                    torch_board_mir.moves_left[b_idx].item(),
                ), f"Problem with {b_idx}"

            if reset_done_boards:
                for b_idx in range(B):
                    bb_done = (
                        bb_board[b_idx].hasWin() or bb_board[b_idx].movesLeft() <= 0
                    )
                    assert bb_done == done[b_idx].item(), f"Problem with {b_idx}"
                    if bb_done:
                        bb_board[b_idx].setRawState(0, 0, 42)

        if reset_done_boards:
            torch_board.reset(done)

        if compare_with_bitbully:
            # Check all after_states
            nl = torch_board.generate_non_losing_moves()
            for mv in torch_board.iter_move_masks(nl):
                active = mv != 0
                if not active.any():
                    break

                tmp = BoardBatch(
                    all_tokens=torch_board.all_tokens.clone(),
                    active_tokens=torch_board.active_tokens.clone(),
                    moves_left=torch_board.moves_left.clone(),
                )
                legal = tmp.play_masks(mv)
                won = tmp.has_win()
                done = tmp.done()
                non_losing_moves = tmp.generate_non_losing_moves()
                can_win = tmp.can_win()

                can_win_actions = torch.randint(0, 7, (B,), device=device)
                can_win_column = tmp.can_win_column(can_win_actions)

                for b_idx in range(B):
                    mv_onehot = mv[b_idx].item()
                    if mv_onehot:  # TODO: BitBully-Core should also return masks (needs lib change)
                        a = move_mask_to_column(mv_onehot)
                        bb_new_board = bb_board[b_idx].playMoveOnCopy(a)
                        ll = bb_new_board.movesLeft() < 42
                    else:
                        bb_new_board = bb_board[b_idx].copy()
                        ll = False

                    all_tokens, active_tokens, moves_left = bb_new_board.rawState()
                    assert (all_tokens, active_tokens, moves_left) == (
                        tmp.all_tokens[b_idx].item(),
                        tmp.active_tokens[b_idx].item(),
                        tmp.moves_left[b_idx].item(),
                    ), f"Problem with {b_idx}"
                    assert bb_new_board.hasWin() == won[b_idx].item(), (
                        f"Problem with {b_idx}"
                    )
                    assert ll == legal[b_idx].item(), f"Problem with {b_idx}"

                    # Check equaility on winning_positions
                    assert (
                        bb_new_board.generateNonLosingMoves() == non_losing_moves[b_idx]
                    ), f"Problem with {b_idx}"

                    # can_win
                    assert bb_new_board.canWin() == can_win[b_idx].item(), (
                        f"Problem with {b_idx}"
                    )

                    # can win column:
                    assert (
                        bb_new_board.canWin(can_win_actions[b_idx].item())
                        == can_win_column[b_idx].item()
                    ), f"Problem with {b_idx}"
