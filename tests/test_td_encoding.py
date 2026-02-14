import bitbully.bitbully_core as bbc
import numpy as np
import pytest
import torch

from techdays26.legacy_ntuple_agent import (
    _lut_index_from_states,
    board_cols_to_flat_features,
)
from techdays26.ntuples import NTUPLE_BITIDX_LIST
from techdays26.torch_board import BoardBatch

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


def _inject_from_core(
    core: bbc.BoardCore, *, dev: torch.device, moves_left_dtype: torch.dtype
) -> "BoardBatch":
    a, b, m = core.rawState()
    return BoardBatch(
        all_tokens=torch.tensor([int(a)], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([int(b)], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([int(m)], device=dev, dtype=moves_left_dtype),
    )


def _std_to_our_bitidx(tup_std: list[int]) -> list[int]:
    """Convert "standard" 42-cell indexing (col*6 + row) into our bitboard bit index (col*9 + row).
    Given x = col*6 + row:
      our_bit = x + 3*(x//6)  == (col*6+row) + 3*col == col*9 + row
    """
    return [x + 3 * (x // 6) for x in tup_std]


def _bitidx_to_flatidx(bitidx: int, *, off: int = 9) -> int:
    col = bitidx // off
    row = bitidx % off
    if not (0 <= col < 7) or not (0 <= row < 6):
        raise ValueError(f"bitidx not in playable area: {bitidx}")
    return col * 6 + row


def _make_ntuple_patterns_tensor(dev: torch.device) -> torch.Tensor:
    return torch.tensor(NTUPLE_BITIDX_LIST, dtype=torch.int64, device=dev)  # [T,8]


def _table_positions_via_flat(
    core: bbc.BoardCore, patterns_bitidx: np.ndarray
) -> np.ndarray:
    """Reference LUT indices using the TD-style flat features + _lut_index_from_states.
    patterns_bitidx is [M,N] in our bitboard bit-index space (col*9+row).
    """
    flat = board_cols_to_flat_features(core.toArray()).astype(
        np.int64, copy=False
    )  # [42]
    M, N = patterns_bitidx.shape
    out = np.empty((M,), dtype=np.int64)

    for i in range(M):
        flat_idxs = np.array(
            [_bitidx_to_flatidx(int(bi), off=9) for bi in patterns_bitidx[i]],
            dtype=np.int64,
        )
        states = flat[flat_idxs]  # [N] in {0,1,2,3}
        out[i] = _lut_index_from_states(states, p=4)
    return out


def test_table_positions_matches_td_lut_indices_for_standard_ntuple_set(
    dev: torch.device,
    moves_left_dtype: torch.dtype,
) -> None:
    """Compare LUT indices for the provided "standard" n-tuple set (converted to our bitboard layout):
    - BoardBatch.table_positions(patterns)
    - TD flat-features + _lut_index_from_states reference
    """
    patterns = _make_ntuple_patterns_tensor(dev=dev)  # [T,8]
    patterns_np = patterns.to("cpu").numpy().astype(np.int64, copy=False)

    # Try many trusted random positions from BoardCore.
    rng = np.random.default_rng(12345)
    for _ in range(911):
        nply = int(rng.integers(0, 34))
        core, _moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)

        bb = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)

        got = (
            bb
            .table_positions(patterns)[0]
            .to("cpu")
            .numpy()
            .astype(np.int64, copy=False)
        )  # [T]
        ref = _table_positions_via_flat(core, patterns_np)  # [T]

        assert np.array_equal(got, ref), f"Mismatch at nply={nply}"


def _bitidx_patterns_to_flatidx(
    patterns_bitidx: np.ndarray, *, off: int = 9
) -> np.ndarray:
    """Convert [M,N] bit indices (col*9+row) to [M,N] flat indices (col*6+row)."""
    if patterns_bitidx.ndim != 2:
        raise ValueError("patterns_bitidx must be 2D [M,N].")
    col = patterns_bitidx // off
    row = patterns_bitidx % off
    if not np.all((col >= 0) & (col < 7)):
        raise ValueError("col out of range in patterns.")
    if not np.all((row >= 0) & (row < 6)):
        raise ValueError("row out of range in patterns (must be 0..5 playable rows).")
    return col * 6 + row


def _inject_from_core(
    core: bbc.BoardCore, *, dev: torch.device, moves_left_dtype: torch.dtype
) -> "BoardBatch":
    # assumes BoardBatch is in scope in this test module, same as your other tests
    a, b, m = core.rawState()
    return BoardBatch(
        all_tokens=torch.tensor([int(a)], device=dev, dtype=torch.int64),
        active_tokens=torch.tensor([int(b)], device=dev, dtype=torch.int64),
        moves_left=torch.tensor([int(m)], device=dev, dtype=moves_left_dtype),
    )


def _mk_random_patterns(*, M: int, N: int, seed: int = 123) -> np.ndarray:
    """Generate [M,N] patterns of playable bit indices in the bitboard layout.
    Each entry is in {col*9+row} with col in 0..6 and row in 0..5.
    """
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 7, size=(M, N), dtype=np.int64)
    rows = rng.integers(0, 6, size=(M, N), dtype=np.int64)
    return cols * 9 + rows


@pytest.mark.parametrize("M,N", [(3, 4), (16, 8), (64, 8)])
def test_table_positions_matches_lut_index_from_states(
    dev: torch.device, moves_left_dtype: torch.dtype, M: int, N: int
) -> None:
    """Ensure BoardBatch.table_positions() produces the same base-4 LUT indices as
    _lut_index_from_states() computed from the TD flat features.

    Important: patterns are specified in bit-index space (col*9+row), but the TD
    flat state is indexed by (col*6+row). We convert patterns accordingly.
    """
    # 1) Create patterns in bit-index space (BoardBatch expects this)
    patterns_np = _mk_random_patterns(M=M, N=N, seed=20260207)
    patterns = torch.tensor(patterns_np, device=dev, dtype=torch.int64)  # [M,N]

    # 2) Convert those patterns to flat-index space (for TD encoding)
    patterns_flat_np = _bitidx_patterns_to_flatidx(patterns_np, off=9)  # [M,N]

    # 3) Compare on many random *trusted* core positions
    #    Use a range of ply depths to cover early/mid-game.
    rng = np.random.default_rng(999)
    for _ in range(911):
        nply = int(rng.integers(0, 32))
        core, _moves = bbc.BoardCore.randomBoard(nPly=nply, forbidDirectWin=False)

        bb = _inject_from_core(core, dev=dev, moves_left_dtype=moves_left_dtype)

        # BoardBatch result: [1,M]
        T = (
            bb
            .table_positions(patterns)[0]
            .to("cpu")
            .numpy()
            .astype(np.int64, copy=False)
        )

        # TD reference: compute flat features from BoardCore's toArray()
        # board_cols_to_flat_features expects 7x6 column-major grid (same as core.toArray()).
        board_cols = core.toArray()
        flat = board_cols_to_flat_features(board_cols).astype(
            np.int64, copy=False
        )  # [42]

        # For each pattern, compute LUT index using your numpy helper
        ref = np.empty((M,), dtype=np.int64)
        for i in range(M):
            states = flat[patterns_flat_np[i]]  # [N], values in {0,1,2,3}
            ref[i] = _lut_index_from_states(states, p=4)

        # Must match exactly
        assert np.array_equal(T, ref), (
            f"Mismatch for nply={nply}\n"
            f"T   ={T[: min(10, M)]}\n"
            f"ref ={ref[: min(10, M)]}"
        )
