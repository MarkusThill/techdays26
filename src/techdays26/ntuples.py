"""Standard n-tuple definitions for Connect-4 TD agents.

Indexing convention (standard layout, 42 cells):

# 5 11 17 23 29 35 41
# 4 10 16 22 28 34 40
# 3  9 15 21 27 33 39
# 2  8 14 20 26 32 38
# 1  7 13 19 25 31 37
# 0  6 12 18 24 30 36
"""

# Our Layout:
# [ *,  *,  *,  *,  *,  *,  *]
# [ *,  *,  *,  *,  *,  *,  *]
# [ *,  *,  *,  *,  *,  *,  *]
# [ 5, 14, 23, 32, 41, 50, 59],
# [ 4, 13, 22, 31, 40, 49, 58],
# [ 3, 12, 21, 30, 39, 48, 57],
# [ 2, 11, 20, 29, 38, 47, 56],
# [ 1, 10, 19, 28, 37, 46, 55],
# [ 0,  9, 18, 27, 36, 45, 54]

NTUPLE_STD_LIST: list[list[int]] = [
    [0, 6, 7, 12, 13, 14, 19, 21],
    [13, 18, 19, 20, 21, 26, 27, 33],
    [1, 3, 4, 6, 7, 8, 9, 10],
    [7, 8, 9, 12, 15, 19, 25, 30],
    [4, 5, 9, 10, 11, 15, 16, 17],
    [1, 2, 3, 8, 9, 10, 16, 17],
    [3, 8, 9, 10, 11, 14, 15, 16],
    [0, 1, 2, 6, 8, 12, 13, 18],
    [25, 26, 27, 32, 33, 37, 38, 39],
    [3, 4, 8, 9, 11, 14, 15, 21],
    [2, 3, 4, 8, 9, 14, 15, 20],
    [18, 19, 24, 30, 31, 32, 36, 37],
    [3, 4, 8, 9, 10, 14, 15, 16],
    [5, 10, 11, 16, 17, 21, 22, 27],
    [4, 10, 15, 20, 21, 22, 27, 28],
    [18, 24, 25, 30, 31, 32, 37, 38],
    [11, 17, 21, 23, 27, 28, 33, 39],
    [21, 25, 26, 27, 32, 34, 35, 41],
    [22, 25, 26, 27, 30, 32, 33, 37],
    [4, 10, 11, 16, 20, 21, 22, 23],
    [0, 6, 7, 8, 12, 13, 14, 15],
    [17, 23, 28, 29, 32, 33, 34, 35],
    [0, 6, 7, 12, 18, 25, 32, 38],
    [2, 3, 4, 5, 8, 9, 10, 11],
    [27, 32, 33, 34, 37, 38, 39, 40],
    [4, 10, 16, 21, 26, 32, 33, 38],
    [0, 6, 7, 12, 13, 20, 27, 28],
    [0, 6, 12, 19, 25, 31, 32, 33],
    [1, 2, 6, 7, 13, 14, 15, 20],
    [1, 2, 5, 8, 11, 15, 16, 17],
    [13, 14, 16, 18, 21, 22, 23, 24],
    [2, 3, 9, 10, 11, 16, 17, 22],
    [15, 16, 17, 20, 22, 23, 25, 31],
    [15, 16, 17, 21, 22, 23, 28, 29],
    [24, 26, 30, 31, 32, 33, 36, 37],
    [12, 13, 18, 19, 20, 26, 27, 33],
    [1, 2, 3, 8, 9, 13, 14, 21],
    [13, 14, 18, 20, 24, 25, 31, 37],
    [14, 15, 16, 21, 26, 31, 38, 39],
    [1, 2, 6, 7, 12, 13, 14, 20],
    [4, 5, 10, 11, 17, 22, 23, 29],
    [2, 4, 5, 7, 9, 10, 14, 19],
    [5, 9, 10, 11, 15, 16, 21, 27],
    [1, 2, 3, 7, 8, 13, 14, 20],
    [1, 2, 8, 9, 14, 15, 21, 26],
    [22, 23, 29, 33, 34, 35, 38, 41],
    [13, 18, 19, 24, 25, 26, 31, 32],
    [27, 28, 29, 31, 32, 33, 37, 38],
    [10, 14, 15, 16, 17, 20, 21, 23],
    [4, 5, 9, 10, 15, 20, 21, 22],
    [13, 20, 25, 26, 27, 32, 34, 41],
    [30, 31, 33, 34, 36, 37, 38, 39],
    [11, 16, 23, 28, 34, 35, 40, 41],
    [3, 4, 10, 11, 14, 15, 16, 17],
    [15, 20, 21, 22, 26, 32, 33, 39],
    [18, 19, 25, 26, 31, 32, 34, 39],
    [4, 9, 11, 15, 16, 22, 23, 29],
    [26, 27, 31, 32, 33, 37, 38, 39],
    [20, 27, 28, 33, 34, 35, 40, 41],
    [1, 2, 7, 14, 20, 27, 28, 29],
    [8, 9, 10, 15, 16, 17, 22, 23],
    [9, 14, 15, 20, 21, 22, 27, 32],
    [1, 2, 3, 6, 7, 8, 9, 13],
    [10, 14, 15, 16, 20, 23, 25, 26],
    [0, 1, 2, 6, 7, 8, 13, 14],
    [1, 6, 7, 12, 13, 20, 26, 27],
    [8, 14, 20, 25, 26, 31, 33, 38],
    [20, 21, 26, 27, 28, 33, 35, 40],
    [2, 3, 4, 8, 9, 11, 16, 21],
    [1, 2, 3, 4, 5, 6, 11, 12],
]


def std_to_bitidx(
    tuples: list[list[int]], *, col_height: int = 6, stride: int = 9
) -> list[list[int]]:
    """Convert standard 42-index tuples (col*6+row) to bitboard indices (col*9+row)."""
    out: list[list[int]] = []
    for tup in tuples:
        out.append([x + (stride - col_height) * (x // col_height) for x in tup])
    return out


NTUPLE_BITIDX_LIST = std_to_bitidx(NTUPLE_STD_LIST)

# 150 n-tuples: the 70 from NTUPLE_STD_LIST + 80 additional random-walk tuples
# (generated with generate_random_ntuples(100, 8, seed=42), deduplicated via merge_ntuples)
NTUPLE_STD_LIST_150: list[list[int]] = NTUPLE_STD_LIST + [
    [13, 19, 25, 26, 30, 31, 33, 40],
    [1, 6, 7, 8, 12, 13, 14, 19],
    [1, 6, 7, 8, 12, 13, 15, 16],
    [14, 21, 22, 25, 26, 27, 29, 32],
    [14, 19, 21, 25, 26, 27, 31, 32],
    [3, 5, 8, 10, 15, 16, 22, 23],
    [2, 6, 7, 8, 12, 13, 14, 19],
    [19, 26, 28, 33, 34, 35, 40, 41],
    [1, 2, 6, 8, 12, 14, 15, 19],
    [16, 20, 21, 23, 28, 29, 35, 40],
    [26, 32, 33, 34, 35, 38, 40, 41],
    [2, 3, 4, 5, 8, 11, 15, 16],
    [26, 31, 32, 36, 37, 38, 39, 40],
    [10, 15, 16, 21, 22, 23, 28, 29],
    [11, 16, 17, 22, 23, 27, 28, 29],
    [3, 4, 5, 8, 9, 14, 16, 21],
    [4, 10, 11, 17, 22, 26, 27, 31],
    [0, 6, 7, 12, 13, 14, 15, 19],
    [14, 19, 21, 22, 24, 25, 27, 31],
    [10, 11, 16, 17, 21, 28, 34, 39],
    [16, 17, 21, 23, 26, 28, 32, 38],
    [8, 9, 14, 15, 16, 20, 21, 27],
    [3, 4, 5, 10, 11, 16, 17, 23],
    [8, 13, 14, 15, 19, 20, 26, 31],
    [12, 13, 18, 24, 30, 31, 37, 38],
    [1, 2, 7, 8, 12, 13, 14, 15],
    [0, 1, 7, 12, 18, 24, 31, 37],
    [3, 4, 5, 10, 11, 15, 16, 21],
    [18, 24, 30, 31, 32, 36, 37, 38],
    [6, 12, 18, 25, 26, 27, 34, 40],
    [1, 2, 3, 8, 13, 20, 21, 27],
    [4, 5, 8, 9, 10, 11, 14, 15],
    [3, 8, 9, 10, 15, 16, 17, 22],
    [20, 21, 26, 27, 32, 33, 40, 41],
    [27, 28, 31, 33, 34, 38, 40, 41],
    [0, 6, 12, 18, 19, 24, 31, 37],
    [10, 11, 17, 23, 28, 34, 35, 40],
    [18, 20, 24, 25, 27, 31, 32, 34],
    [3, 4, 5, 10, 11, 16, 22, 23],
    [14, 15, 16, 19, 20, 21, 22, 25],
    [5, 10, 16, 21, 26, 30, 31, 36],
    [13, 18, 19, 20, 25, 26, 31, 36],
    [0, 1, 2, 6, 7, 9, 14, 20],
    [3, 4, 8, 9, 10, 11, 14, 16],
    [22, 27, 28, 29, 33, 34, 35, 40],
    [3, 4, 9, 10, 13, 14, 16, 19],
    [20, 21, 26, 32, 33, 38, 39, 40],
    [0, 1, 2, 6, 7, 8, 9, 16],
    [3, 8, 9, 10, 15, 22, 28, 29],
    [12, 13, 14, 18, 19, 21, 28, 35],
    [10, 15, 20, 23, 27, 28, 29, 33],
    [1, 2, 5, 7, 8, 9, 10, 16],
    [17, 22, 23, 28, 29, 34, 35, 40],
    [1, 2, 8, 13, 20, 26, 33, 39],
    [13, 14, 20, 24, 25, 30, 31, 32],
    [1, 2, 8, 9, 13, 14, 19, 20],
    [21, 22, 26, 28, 33, 34, 40, 41],
    [5, 8, 9, 10, 11, 15, 17, 22],
    [14, 19, 20, 25, 26, 30, 31, 36],
    [1, 2, 7, 8, 14, 21, 27, 33],
    [1, 2, 6, 7, 8, 12, 13, 14],
    [6, 7, 8, 13, 14, 21, 25, 26],
    [7, 8, 12, 13, 14, 18, 19, 25],
    [29, 32, 34, 35, 38, 39, 40, 41],
    [5, 10, 15, 21, 24, 27, 31, 32],
    [4, 5, 10, 11, 15, 16, 17, 22],
    [30, 31, 32, 33, 34, 37, 39, 40],
    [24, 26, 30, 31, 32, 33, 37, 38],
    [6, 8, 12, 13, 15, 21, 26, 27],
    [3, 4, 5, 10, 11, 16, 21, 27],
    [28, 33, 34, 35, 38, 39, 40, 41],
    [10, 17, 23, 28, 34, 35, 40, 41],
    [32, 33, 34, 35, 37, 38, 39, 40],
    [10, 11, 15, 16, 17, 22, 23, 28],
    [7, 8, 9, 14, 16, 20, 21, 26],
    [14, 15, 16, 19, 21, 22, 28, 29],
    [1, 3, 4, 8, 9, 10, 11, 16],
    [6, 12, 13, 14, 19, 24, 25, 26],
    [24, 25, 26, 31, 32, 33, 38, 39],
    [24, 30, 31, 32, 33, 37, 39, 40],
]

NTUPLE_BITIDX_LIST_150 = std_to_bitidx(NTUPLE_STD_LIST_150)


def _bitidx_valid_cells(
    *, n_cols: int = 7, n_rows: int = 6, stride: int = 9
) -> list[int]:
    """Return sorted list of all valid bitboard cell indices."""
    return [col * stride + row for col in range(n_cols) for row in range(n_rows)]


def _bitidx_neighbors(
    cell: int, *, n_cols: int = 7, n_rows: int = 6, stride: int = 9
) -> list[int]:
    """Return 8-connected neighbors of *cell* in bitboard layout."""
    col, row = divmod(cell, stride)
    neighbors = []
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            nc, nr = col + dc, row + dr
            if 0 <= nc < n_cols and 0 <= nr < n_rows:
                neighbors.append(nc * stride + nr)
    return neighbors


def generate_random_ntuples(
    m: int,
    n: int,
    *,
    seed: int | None = None,
    max_retries: int = 1000,
) -> list[list[int]]:
    """Generate *m* n-tuples of length *n* via random walks on the bitboard grid.

    Each walk starts at a uniformly random cell, then repeatedly steps to a
    random **unvisited** 8-connected neighbor.  If the walk gets stuck before
    reaching length *n*, it is discarded and restarted (up to *max_retries*
    per tuple).

    Args:
        m: Number of tuples to generate.
        n: Desired length of each tuple (number of cells).
        seed: Optional RNG seed for reproducibility.
        max_retries: Maximum restart attempts per tuple before raising.

    Returns:
        List of *m* tuples, each a sorted list of *n* bitboard cell indices.

    Raises:
        RuntimeError: If a tuple cannot be completed within *max_retries*.
    """
    import random

    rng = random.Random(seed)
    cells = _bitidx_valid_cells()

    # Pre-compute neighbor lookup
    adj: dict[int, list[int]] = {c: _bitidx_neighbors(c) for c in cells}

    tuples: list[list[int]] = []
    for i in range(m):
        for attempt in range(max_retries):
            visited: set[int] = set()
            start = rng.choice(cells)
            path = [start]
            visited.add(start)
            for _ in range(n - 1):
                candidates = [nb for nb in adj[path[-1]] if nb not in visited]
                if not candidates:
                    break  # stuck — restart
                nxt = rng.choice(candidates)
                path.append(nxt)
                visited.add(nxt)
            if len(path) == n:
                tuples.append(sorted(path))
                break
        else:
            raise RuntimeError(
                f"Failed to generate tuple {i} of length {n} "
                f"after {max_retries} retries"
            )

    return tuples


def format_ntuple(
    ntuple: list[int],
    *,
    n_cols: int = 7,
    n_rows: int = 6,
    stride: int = 9,
    cell_char: str = "X",
    empty_char: str = "_",
) -> str:
    """Return a human-readable board string showing the cells of one n-tuple.

    Example output (for an n-tuple containing cells 9, 27, 36, 45)::

        _  _  _  _  _  _  _
        _  _  _  _  _  _  _
        _  _  _  _  _  _  _
        _  _  _  _  _  _  _
        _  _  _  _  _  _  _
        _  X  _  X  X  X  _
    """
    cells = set(ntuple)
    lines: list[str] = []
    for row in range(n_rows - 1, -1, -1):  # top row first
        cols = []
        for col in range(n_cols):
            idx = col * stride + row
            cols.append(cell_char if idx in cells else empty_char)
        lines.append("  ".join(cols))
    return "\n".join(lines)


def print_ntuple(ntuple: list[int], **kwargs: object) -> None:
    """Print a single n-tuple on the board grid. See :func:`format_ntuple`."""
    print(format_ntuple(ntuple, **kwargs))  # type: ignore[arg-type]


def ntuple_summary(tuples: list[list[int]]) -> dict[str, object]:
    """Return a metadata dict describing an n-tuple set.

    Keys:
        count       - number of tuples (M)
        length      - tuple length (N) if uniform, else list of lengths
        uniform     - True if all tuples have the same length
        lut_size    - 4^N (only present when uniform)
        hash        - deterministic hex digest of the tuple set (SHA-256)
    """
    import hashlib
    import json

    lengths = [len(t) for t in tuples]
    uniform = len(set(lengths)) == 1

    # Deterministic hash: canonical JSON of sorted tuples
    canonical = json.dumps(sorted(tuple(t) for t in tuples), separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()

    info: dict[str, object] = {
        "count": len(tuples),
        "length": lengths[0] if uniform else lengths,
        "uniform": uniform,
        "hash": digest,
    }
    if uniform:
        info["lut_size"] = 4 ** lengths[0]

    return info


def merge_ntuples(*tuple_sets: list[list[int]]) -> list[list[int]]:
    """Merge multiple n-tuple lists, removing duplicates.

    Duplicates are detected by comparing sorted cell lists.  The first
    occurrence (in iteration order across all inputs) is kept.

    Args:
        *tuple_sets: One or more lists of n-tuples to merge.

    Returns:
        A single deduplicated list of n-tuples preserving first-seen order.
    """
    seen: set[tuple[int, ...]] = set()
    merged: list[list[int]] = []
    for ts in tuple_sets:
        for tup in ts:
            key = tuple(sorted(tup))
            if key not in seen:
                seen.add(key)
                merged.append(tup)
    return merged
