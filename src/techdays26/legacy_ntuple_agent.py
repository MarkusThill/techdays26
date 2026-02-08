from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# If you want a static type check against the Protocol:


# =============================================================================
# Models
# =============================================================================


@dataclass(frozen=True, slots=True)
class TupleLUT:
    """One n-tuple LUT (keeps both original + mirrored index sets)."""

    n: int
    m: int
    idxs: np.ndarray  # (n,), original indices (mirror1)
    idxs_m: np.ndarray  # (n,), mirrored indices (mirror2)
    lut: np.ndarray  # (P**n,), float64


@dataclass(frozen=True, slots=True)
class PlayerWeights:
    """Weights file for one 'player to move' (p0 or p1)."""

    t: int
    p: int
    luts: tuple[TupleLUT, ...]


@dataclass(frozen=True, slots=True)
class TwoPlayerWeights:
    """Weights for both players to move: p0 (yellow), p1 (red)."""

    p0: PlayerWeights
    p1: PlayerWeights

    def for_player(self, player: int) -> PlayerWeights:
        if player == 0:
            return self.p0
        if player == 1:
            return self.p1
        raise ValueError("player must be 0 (yellow-to-move) or 1 (red-to-move)")


# =============================================================================
# Parsing helpers (text -> TwoPlayerWeights)
# =============================================================================


@dataclass(slots=True)
class Block:
    text: str | list[str]
    children: list[Block]


_ALLOWED_CHARS = re.compile(r"^[\s0-9+\-\.eE{}]*$")


def _normalize_text(text: str) -> str:
    text = text.replace("\r", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_text(text: str) -> None:
    if text.count("{") != text.count("}"):
        raise ValueError("Brace count mismatch: unbalanced '{' and '}'.")
    if not _ALLOWED_CHARS.match(text):
        for i, ch in enumerate(text):
            if not re.match(r"[\s0-9+\-\.eE{}]", ch):
                raise ValueError(
                    f"Invalid character {ch!r} (ord={ord(ch)}) at position {i}"
                )
        raise ValueError("Invalid character(s) found.")


def _parse_blocks(text: str) -> Block:
    root = Block(text="", children=[])
    stack: list[Block] = [root]
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        s = "".join(buf)
        buf.clear()
        cur = stack[-1]
        assert isinstance(cur.text, str)
        cur.text += s

    for pos, ch in enumerate(text):
        if ch == "{":
            flush()
            child = Block(text="", children=[])
            stack[-1].children.append(child)
            stack.append(child)
        elif ch == "}":
            flush()
            stack.pop()
            if not stack:
                raise ValueError(f"Unmatched '}}' at character index {pos}")
        else:
            buf.append(ch)

    if len(stack) != 1:
        raise ValueError("Unmatched '{' (file ended while blocks were still open).")

    flush()
    return root


def _split_tokens_inplace(block: Block) -> None:
    if isinstance(block.text, str):
        s = block.text.strip()
        block.text = s.split() if s else []
    for child in block.children:
        _split_tokens_inplace(child)


def _block_to_tuple_lut(block: Block, *, p: int, int_dtype=np.int32) -> TupleLUT:
    if not isinstance(block.text, list) or len(block.text) != 2:
        raise ValueError(f"Expected tuple header tokens [N, M], got: {block.text!r}")
    if len(block.children) != 3:
        raise ValueError(
            f"Expected 3 children (set, mirror, weights), got {len(block.children)}"
        )

    n = int(block.text[0])
    m = int(block.text[1])

    b_set, b_mirror, b_w = block.children
    if (
        not isinstance(b_set.text, list)
        or not isinstance(b_mirror.text, list)
        or not isinstance(b_w.text, list)
    ):
        raise ValueError(
            "Child blocks must be token lists. Did you call _split_tokens_inplace()?"
        )

    if len(b_set.text) != n:
        raise ValueError(
            f"Sample set length mismatch: expected {n}, got {len(b_set.text)}"
        )
    if len(b_mirror.text) != n:
        raise ValueError(
            f"Mirrored set length mismatch: expected {n}, got {len(b_mirror.text)}"
        )

    expected_w = int(p**n)
    if len(b_w.text) != expected_w:
        raise ValueError(
            f"LUT length mismatch: expected {expected_w} (= {p}^{n}), got {len(b_w.text)}"
        )

    return TupleLUT(
        n=n,
        m=m,
        idxs=np.asarray([int(x) for x in b_set.text], dtype=int_dtype),
        idxs_m=np.asarray([int(x) for x in b_mirror.text], dtype=int_dtype),
        lut=np.asarray([float(x) for x in b_w.text], dtype=np.float64),
    )


class TDWeightsLoader:
    """Load TD-agent weights from `p0.txt` and `p1.txt` (directory or zip)."""

    def __init__(
        self, *, int_dtype=np.int32, validate: bool = True, strict_t: bool = True
    ) -> None:
        self._int_dtype = int_dtype
        self._validate = validate
        self._strict_t = strict_t

    def _load_from_text(self, raw: str) -> PlayerWeights:
        text = _normalize_text(raw)
        if self._validate:
            _validate_text(text)

        root = _parse_blocks(text)
        _split_tokens_inplace(root)

        if not root.children:
            raise ValueError(
                "No top-level block found. Expected file to start with '{'."
            )
        file_block = root.children[0]

        if not isinstance(file_block.text, list) or len(file_block.text) < 2:
            raise ValueError(
                f"Expected file header tokens [T, P], got: {file_block.text!r}"
            )

        t = int(file_block.text[0])
        p = int(file_block.text[1])

        luts = tuple(
            _block_to_tuple_lut(b, p=p, int_dtype=self._int_dtype)
            for b in file_block.children
        )

        if self._validate and self._strict_t and t != len(luts):
            raise ValueError(f"T mismatch: header T={t}, parsed tuples={len(luts)}")

        return PlayerWeights(t=t, p=p, luts=luts)

    def load_file(self, path: str | Path) -> PlayerWeights:
        raw = Path(path).read_text(encoding="utf-8", errors="strict")
        return self._load_from_text(raw)

    def load_file_from_zip(self, zip_path: str | Path, member: str) -> PlayerWeights:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                data = zf.read(member)
            except KeyError as e:
                raise FileNotFoundError(
                    f"{member!r} not found in zip {str(zip_path)!r}"
                ) from e
        raw = data.decode("utf-8", errors="strict")
        return self._load_from_text(raw)

    def load_two_player(
        self, directory: str | Path, *, p0: str = "p0.txt", p1: str = "p1.txt"
    ) -> TwoPlayerWeights:
        d = Path(directory)
        w0 = self.load_file(d / p0)
        w1 = self.load_file(d / p1)
        if self._validate and (w0.p != w1.p):
            raise ValueError(f"P mismatch between p0/p1: {w0.p} vs {w1.p}")
        return TwoPlayerWeights(p0=w0, p1=w1)

    def load_two_player_from_zip(
        self,
        zip_path: str | Path,
        *,
        p0: str = "p0.txt",
        p1: str = "p1.txt",
    ) -> TwoPlayerWeights:
        w0 = self.load_file_from_zip(zip_path, p0)
        w1 = self.load_file_from_zip(zip_path, p1)
        if self._validate and (w0.p != w1.p):
            raise ValueError(f"P mismatch between p0/p1: {w0.p} vs {w1.p}")
        return TwoPlayerWeights(p0=w0, p1=w1)


# =============================================================================
# Board encoding (7 columns x 6 rows, bottom->top) -> flat 42 state array
# =============================================================================


def board_cols_to_flat_features(board_cols: list[list[int]]) -> np.ndarray:
    """Convert board_cols[col][row] into 42-length state array with P=4 encoding."""
    if len(board_cols) != 7 or any(len(col) != 6 for col in board_cols):
        raise ValueError(
            "Expected board_cols shape (7, 6): 7 columns each with 6 cells."
        )

    flat = np.zeros(42, dtype=np.int8)

    for col in range(7):
        column = board_cols[col]

        reachable_row: int | None = None
        for row in range(6):
            if column[row] == 0:
                reachable_row = row
                break

        for row in range(6):
            idx = col * 6 + row
            cell = column[row]
            if cell == 1:
                flat[idx] = 1
            elif cell == 2:
                flat[idx] = 2
            else:
                flat[idx] = (
                    3 if (reachable_row is not None and row == reachable_row) else 0
                )

    return flat


# =============================================================================
# TD evaluator
# =============================================================================


def _lut_index_from_states(states: np.ndarray, p: int) -> int:
    powers = p ** np.arange(states.size, dtype=np.int64)
    return int(states.astype(np.int64, copy=False) @ powers)


class TDEvaluator:
    """Compute TD n-tuple value from TwoPlayerWeights."""

    def __init__(self, weights: TwoPlayerWeights) -> None:
        self._weights = weights

    def value(self, *, board_cols: list[list[int]], player_to_move: int) -> float:
        pw = self._weights.for_player(player_to_move)
        flat = board_cols_to_flat_features(board_cols).astype(np.int64, copy=False)

        total = 0.0
        for tup in pw.luts:
            idx1 = _lut_index_from_states(flat[tup.idxs], pw.p)
            idx2 = _lut_index_from_states(flat[tup.idxs_m], pw.p)
            total += float(tup.lut[idx1]) + float(tup.lut[idx2])
        return total

    @staticmethod
    def to_score(raw_value: float) -> int:
        return int(round(100.0 * math.tanh(raw_value)))


# =============================================================================
# Agent that satisfies the Connect4Agent Protocol
# =============================================================================


class TDConnect4Agent:
    """TD n-tuple agent compatible with `Connect4Agent` Protocol.

    Implements:
      - score_all_moves(board) -> dict[int,int]
      - best_move(board) -> int
      - score_move(board, column, first_guess=0) -> int
    """

    def __init__(self, evaluator: TDEvaluator, *, tie_break: str = "center") -> None:
        self._eval = evaluator
        self._tie_break = tie_break

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

            after = board.play_on_copy(col)
            next_player = after.current_player() - 1
            if next_player not in (0, 1):
                raise ValueError(
                    f"Unexpected current_player() after move: {after.current_player()}"
                )

            raw = self._eval.value(
                board_cols=after.to_array(), player_to_move=next_player
            )
            score = self._eval.to_score(raw)

            # normalize to "bigger is better for side-to-move"
            if player_to_move == 1:
                score = -score

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

        player_to_move = board.current_player() - 1
        after = board.play_on_copy(column)
        next_player = after.current_player() - 1

        raw = self._eval.value(board_cols=after.to_array(), player_to_move=next_player)
        score = self._eval.to_score(raw)

        if player_to_move == 1:
            score = -score

        return score


# Assumes your dataclasses exist in scope:
# TupleLUT, PlayerWeights, TwoPlayerWeights


_FORMAT_NAME = "bitbully.td_weights.zip"
_FORMAT_VERSION = 1


def _write_npy_to_zip(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    """Write a numpy array as a .npy file into the zip."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    zf.writestr(name, buf.getvalue())


def _read_npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read a numpy array from a .npy file inside the zip."""
    raw = zf.read(name)
    with io.BytesIO(raw) as buf:
        arr = np.load(buf, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name!r} did not decode to an ndarray.")
    return arr


def export_two_player_weights_zip(path: str | Path, both: TwoPlayerWeights) -> None:
    """Export `both` (TwoPlayerWeights) to a single zip file containing:
      - meta.json
      - arrays as .npy files

    This avoids pickle and is stable across platforms / Python versions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "players": {},
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        def write_player(tag: str, pw: PlayerWeights) -> None:
            player_meta: dict = {
                "t": int(pw.t),
                "p": int(pw.p),
                "luts": [],
            }

            for i, tup in enumerate(pw.luts):
                base = f"{tag}/lut_{i:04d}"

                # Store arrays as .npy
                idxs_name = f"{base}/idxs.npy"
                idxs_m_name = f"{base}/idxs_m.npy"
                lut_name = f"{base}/lut.npy"

                _write_npy_to_zip(zf, idxs_name, tup.idxs)
                _write_npy_to_zip(zf, idxs_m_name, tup.idxs_m)
                _write_npy_to_zip(zf, lut_name, tup.lut)

                # Store the small scalar metadata in JSON
                player_meta["luts"].append({
                    "n": int(tup.n),
                    "m": int(tup.m),
                    "idxs": idxs_name,
                    "idxs_m": idxs_m_name,
                    "lut": lut_name,
                })

            meta["players"][tag] = player_meta

        write_player("p0", both.p0)
        write_player("p1", both.p1)

        # Write meta.json last
        zf.writestr(
            "meta.json", json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")
        )


def import_two_player_weights_zip(
    path: str | Path,
    *,
    int_dtype=np.int32,
    validate: bool = True,
    strict_t: bool = True,
) -> TwoPlayerWeights:
    """Import TwoPlayerWeights from the zip written by export_two_player_weights_zip().

    Args:
        int_dtype: dtype for idx arrays (e.g. np.int32).
        validate: check basic invariants (p match, lengths).
        strict_t: enforce pw.t equals number of LUTs.
    """
    path = Path(path)

    with zipfile.ZipFile(path, "r") as zf:
        meta_raw = zf.read("meta.json").decode("utf-8")
        meta = json.loads(meta_raw)

        if meta.get("format") != _FORMAT_NAME:
            raise ValueError(f"Unexpected format: {meta.get('format')!r}")
        if int(meta.get("version", -1)) != _FORMAT_VERSION:
            raise ValueError(f"Unexpected version: {meta.get('version')!r}")

        players = meta.get("players")
        if not isinstance(players, dict) or "p0" not in players or "p1" not in players:
            raise ValueError("meta.json missing players p0/p1.")

        def read_player(tag: str) -> PlayerWeights:
            pm = players[tag]
            t = int(pm["t"])
            p = int(pm["p"])
            lut_entries = pm["luts"]
            if not isinstance(lut_entries, list):
                raise ValueError(f"players[{tag!r}].luts must be a list.")

            luts: list[TupleLUT] = []
            for entry in lut_entries:
                n = int(entry["n"])
                m = int(entry["m"])

                idxs = _read_npy_from_zip(zf, entry["idxs"]).astype(
                    int_dtype, copy=False
                )
                idxs_m = _read_npy_from_zip(zf, entry["idxs_m"]).astype(
                    int_dtype, copy=False
                )
                lut = _read_npy_from_zip(zf, entry["lut"]).astype(
                    np.float64, copy=False
                )

                if validate:
                    if idxs.ndim != 1 or idxs.size != n:
                        raise ValueError(
                            f"{tag}: idxs shape mismatch (expected ({n},), got {idxs.shape})."
                        )
                    if idxs_m.ndim != 1 or idxs_m.size != n:
                        raise ValueError(
                            f"{tag}: idxs_m shape mismatch (expected ({n},), got {idxs_m.shape})."
                        )
                    expected = int(p**n)
                    if lut.ndim != 1 or lut.size != expected:
                        raise ValueError(
                            f"{tag}: lut length mismatch (expected {expected} (= {p}^{n}), got {lut.size})."
                        )

                luts.append(TupleLUT(n=n, m=m, idxs=idxs, idxs_m=idxs_m, lut=lut))

            if validate and strict_t and t != len(luts):
                raise ValueError(f"{tag}: T mismatch: header t={t}, tuples={len(luts)}")

            return PlayerWeights(t=t, p=p, luts=tuple(luts))

        p0 = read_player("p0")
        p1 = read_player("p1")

        if validate and p0.p != p1.p:
            raise ValueError(f"P mismatch between p0/p1: {p0.p} vs {p1.p}")

        return TwoPlayerWeights(p0=p0, p1=p1)
