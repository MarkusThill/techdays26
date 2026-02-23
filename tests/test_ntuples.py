"""Tests for techdays26.ntuples module."""

from __future__ import annotations

import pytest

from techdays26.ntuples import (
    NTUPLE_BITIDX_LIST,
    NTUPLE_STD_LIST,
    _bitidx_neighbors,
    _bitidx_valid_cells,
    format_ntuple,
    generate_random_ntuples,
    merge_ntuples,
    ntuple_summary,
    std_to_bitidx,
)

# ---------------------------------------------------------------------------
# Valid cells
# ---------------------------------------------------------------------------
VALID_CELLS = set(_bitidx_valid_cells())


class TestBitidxValidCells:
    def test_count(self):
        assert len(VALID_CELLS) == 42  # 7 cols * 6 rows

    def test_known_corners(self):
        # bottom-left=0, top-left=5, bottom-right=54, top-right=59
        assert {0, 5, 54, 59}.issubset(VALID_CELLS)

    def test_gap_cells_excluded(self):
        # rows 6,7,8 of each column are padding (not valid board cells)
        for col in range(7):
            for row in range(6, 9):
                assert col * 9 + row not in VALID_CELLS


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------
class TestBitidxNeighbors:
    def test_corner_has_3_neighbors(self):
        # bottom-left corner (cell 0) -> neighbors: 1, 9, 10
        nb = _bitidx_neighbors(0)
        assert sorted(nb) == [1, 9, 10]

    def test_top_right_corner(self):
        # cell 59 (col=6, row=5)
        nb = _bitidx_neighbors(59)
        assert sorted(nb) == [49, 50, 58]

    def test_center_has_8_neighbors(self):
        # cell 28 (col=3, row=1) has all 8 neighbors on the board
        nb = _bitidx_neighbors(28)
        assert len(nb) == 8
        assert all(c in VALID_CELLS for c in nb)

    def test_edge_cell(self):
        # cell 3 (col=0, row=3) -> left edge, should have 5 neighbors
        # same col: row 2 (cell 2), row 4 (cell 4)
        # col 1: row 2 (cell 11), row 3 (cell 12), row 4 (cell 13)
        nb = _bitidx_neighbors(3)
        assert len(nb) == 5
        assert sorted(nb) == [2, 4, 11, 12, 13]

    def test_all_neighbors_are_valid(self):
        for cell in VALID_CELLS:
            for nb in _bitidx_neighbors(cell):
                assert nb in VALID_CELLS, f"Cell {cell} has invalid neighbor {nb}"


# ---------------------------------------------------------------------------
# std_to_bitidx
# ---------------------------------------------------------------------------
class TestStdToBitidx:
    def test_known_conversion(self):
        # std cell 0 (col=0, row=0) -> bitidx 0  (0 + 3*(0//6) = 0)
        # std cell 6 (col=1, row=0) -> bitidx 9  (6 + 3*(6//6) = 9)
        # std cell 7 (col=1, row=1) -> bitidx 10 (7 + 3*(7//6) = 10)
        result = std_to_bitidx([[0, 6, 7]])
        assert result == [[0, 9, 10]]

    def test_all_std_cells_map_to_valid(self):
        bitidx = std_to_bitidx(NTUPLE_STD_LIST)
        for tup in bitidx:
            for cell in tup:
                assert cell in VALID_CELLS, f"Cell {cell} not valid"

    def test_preserves_tuple_count(self):
        assert len(NTUPLE_BITIDX_LIST) == len(NTUPLE_STD_LIST)

    def test_preserves_tuple_length(self):
        for std, bit in zip(NTUPLE_STD_LIST, NTUPLE_BITIDX_LIST):
            assert len(std) == len(bit)


# ---------------------------------------------------------------------------
# NTUPLE_STD_LIST / NTUPLE_BITIDX_LIST
# ---------------------------------------------------------------------------
class TestStandardTuples:
    def test_count(self):
        assert len(NTUPLE_STD_LIST) == 70

    def test_all_length_8(self):
        for i, tup in enumerate(NTUPLE_STD_LIST):
            assert len(tup) == 8, f"Tuple {i} has length {len(tup)}"

    def test_no_duplicates_within_tuple(self):
        for i, tup in enumerate(NTUPLE_STD_LIST):
            assert len(set(tup)) == len(tup), f"Tuple {i} has duplicate cells"

    def test_std_cells_in_range(self):
        for tup in NTUPLE_STD_LIST:
            for cell in tup:
                assert 0 <= cell <= 41, f"Cell {cell} out of range [0, 41]"

    def test_bitidx_cells_valid(self):
        for tup in NTUPLE_BITIDX_LIST:
            for cell in tup:
                assert cell in VALID_CELLS, f"Cell {cell} not a valid bitidx"


# ---------------------------------------------------------------------------
# generate_random_ntuples
# ---------------------------------------------------------------------------
class TestGenerateRandomNtuples:
    def test_correct_count_and_length(self):
        tuples = generate_random_ntuples(20, 6, seed=0)
        assert len(tuples) == 20
        assert all(len(t) == 6 for t in tuples)

    def test_all_cells_valid(self):
        tuples = generate_random_ntuples(50, 8, seed=1)
        for tup in tuples:
            for cell in tup:
                assert cell in VALID_CELLS

    def test_no_duplicate_cells_within_tuple(self):
        tuples = generate_random_ntuples(50, 8, seed=2)
        for i, tup in enumerate(tuples):
            assert len(set(tup)) == len(tup), f"Tuple {i} has duplicates"

    def test_tuples_are_sorted(self):
        tuples = generate_random_ntuples(30, 8, seed=3)
        for tup in tuples:
            assert tup == sorted(tup)

    def test_connectivity(self):
        """Each tuple should form a connected subgraph (random walk property)."""
        tuples = generate_random_ntuples(30, 8, seed=4)
        for i, tup in enumerate(tuples):
            cells = set(tup)
            # BFS from first cell, only visiting cells in the tuple
            visited = {tup[0]}
            queue = [tup[0]]
            while queue:
                c = queue.pop()
                for nb in _bitidx_neighbors(c):
                    if nb in cells and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            assert visited == cells, f"Tuple {i} is not connected: {tup}"

    def test_seed_reproducibility(self):
        a = generate_random_ntuples(10, 8, seed=42)
        b = generate_random_ntuples(10, 8, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_random_ntuples(10, 8, seed=100)
        b = generate_random_ntuples(10, 8, seed=200)
        assert a != b

    def test_length_1(self):
        tuples = generate_random_ntuples(5, 1, seed=0)
        assert all(len(t) == 1 for t in tuples)

    def test_large_batch(self):
        tuples = generate_random_ntuples(200, 8, seed=99)
        assert len(tuples) == 200

    def test_impossible_length_raises(self):
        # 43 cells on a 42-cell board is impossible
        with pytest.raises(RuntimeError, match="Failed to generate"):
            generate_random_ntuples(1, 43, seed=0, max_retries=10)


# ---------------------------------------------------------------------------
# ntuple_summary
# ---------------------------------------------------------------------------
class TestNtupleSummary:
    def test_standard_tuples(self):
        info = ntuple_summary(NTUPLE_BITIDX_LIST)
        assert info["count"] == 70
        assert info["length"] == 8
        assert info["uniform"] is True
        assert info["lut_size"] == 4**8
        assert isinstance(info["hash"], str)
        assert len(info["hash"]) == 64  # SHA-256 hex

    def test_non_uniform_lengths(self):
        tuples = [[0, 1, 2], [0, 1]]
        info = ntuple_summary(tuples)
        assert info["uniform"] is False
        assert info["length"] == [3, 2]
        assert "lut_size" not in info

    def test_hash_is_deterministic(self):
        a = ntuple_summary(NTUPLE_BITIDX_LIST)["hash"]
        b = ntuple_summary(NTUPLE_BITIDX_LIST)["hash"]
        assert a == b

    def test_hash_order_independent(self):
        """Reordering tuples should not change the hash (we sort canonically)."""
        tuples_a = [[0, 1, 2], [3, 4, 5]]
        tuples_b = [[3, 4, 5], [0, 1, 2]]
        assert ntuple_summary(tuples_a)["hash"] == ntuple_summary(tuples_b)["hash"]

    def test_different_tuples_different_hash(self):
        a = ntuple_summary([[0, 1, 2]])
        b = ntuple_summary([[0, 1, 3]])
        assert a["hash"] != b["hash"]


# ---------------------------------------------------------------------------
# format_ntuple / print_ntuple
# ---------------------------------------------------------------------------
class TestFormatNtuple:
    def test_empty_board(self):
        """No cells marked -> all underscores."""
        result = format_ntuple([])
        lines = result.strip().split("\n")
        assert len(lines) == 6
        for line in lines:
            assert set(line.replace(" ", "")) == {"_"}

    def test_single_cell_bottom_left(self):
        """Cell 0 = col 0, row 0 -> bottom-left."""
        result = format_ntuple([0])
        lines = result.strip().split("\n")
        # Last line (row 0) should have X in first position
        assert lines[-1].startswith("X")
        # All other lines should be all underscores
        for line in lines[:-1]:
            assert "X" not in line

    def test_single_cell_top_right(self):
        """Cell 59 = col 6, row 5 -> top-right."""
        result = format_ntuple([59])
        lines = result.strip().split("\n")
        # First line (row 5) should have X in last position
        assert lines[0].endswith("X")
        for line in lines[1:]:
            assert "X" not in line

    def test_known_tuple_cell_count(self):
        """The number of X's should equal the tuple length."""
        tup = NTUPLE_BITIDX_LIST[0]
        result = format_ntuple(tup)
        assert result.count("X") == len(tup)

    def test_dimensions(self):
        result = format_ntuple([0])
        lines = result.strip().split("\n")
        assert len(lines) == 6  # 6 rows
        for line in lines:
            # 7 characters separated by 2 spaces each: "X  _  _  _  _  _  _"
            parts = line.split("  ")
            assert len(parts) == 7

    def test_custom_chars(self):
        result = format_ntuple([0], cell_char="O", empty_char=".")
        lines = result.strip().split("\n")
        assert "O" in lines[-1]
        assert "X" not in result
        assert "_" not in result

    def test_full_bottom_row(self):
        """All 7 cells in bottom row (row=0): 0, 9, 18, 27, 36, 45, 54."""
        bottom = [col * 9 for col in range(7)]
        result = format_ntuple(bottom)
        lines = result.strip().split("\n")
        # Last line should be all X
        assert lines[-1] == "X  X  X  X  X  X  X"
        # Other lines should be all _
        for line in lines[:-1]:
            assert "X" not in line


# ---------------------------------------------------------------------------
# merge_ntuples
# ---------------------------------------------------------------------------
class TestMergeNtuples:
    def test_no_overlap(self):
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[6, 7, 8], [9, 10, 11]]
        assert merge_ntuples(a, b) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    def test_exact_duplicates_removed(self):
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[3, 4, 5], [6, 7, 8]]
        result = merge_ntuples(a, b)
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_unsorted_duplicate_detected(self):
        """Even if a duplicate is unsorted, it should be detected."""
        a = [[0, 1, 2]]
        b = [[2, 0, 1]]
        result = merge_ntuples(a, b)
        assert len(result) == 1

    def test_preserves_first_occurrence(self):
        a = [[0, 1, 2]]
        b = [[2, 0, 1]]
        result = merge_ntuples(a, b)
        assert result[0] == [0, 1, 2]  # keeps a's version

    def test_empty_inputs(self):
        assert merge_ntuples([], []) == []
        assert merge_ntuples([[0, 1]], []) == [[0, 1]]
        assert merge_ntuples([], [[0, 1]]) == [[0, 1]]

    def test_single_input(self):
        a = [[0, 1], [2, 3], [0, 1]]
        result = merge_ntuples(a)
        assert result == [[0, 1], [2, 3]]

    def test_three_inputs(self):
        a = [[0, 1]]
        b = [[2, 3]]
        c = [[0, 1], [4, 5]]
        result = merge_ntuples(a, b, c)
        assert result == [[0, 1], [2, 3], [4, 5]]

    def test_with_real_tuples(self):
        merged = merge_ntuples(NTUPLE_BITIDX_LIST, NTUPLE_BITIDX_LIST)
        assert len(merged) == len(NTUPLE_BITIDX_LIST)
