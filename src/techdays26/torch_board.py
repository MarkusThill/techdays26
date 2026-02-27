from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


def _i64(x: int) -> int:
    return x


@dataclass(slots=True)
class BoardBatch:
    all_tokens: torch.Tensor  # [B] int64
    active_tokens: torch.Tensor  # [B] int64
    moves_left: torch.Tensor  # [B] int16/int32

    N_COLUMNS: ClassVar[int] = 7
    N_ROWS: ClassVar[int] = 6
    COLUMN_BIT_OFFSET: ClassVar[int] = 9

    BB_BOTTOM_ROW: ClassVar[int] = _i64(
        (1 << 54) | (1 << 45) | (1 << 36) | (1 << 27) | (1 << 18) | (1 << 9) | (1 << 0)
    )
    BB_ALL_LEGAL_TOKENS: ClassVar[int] = _i64(
        sum(1 << (c * 9 + r) for c in range(7) for r in range(6))
    )

    # -------- caching (per-process) --------
    _WEIGHTS_CACHE: ClassVar[dict[tuple[str, int | None, int], torch.Tensor]] = {}
    _PATTERN_MASKS_CACHE: ClassVar[dict[tuple[str, int | None, int], torch.Tensor]] = {}
    _COL_MASKS_CACHE: ClassVar[dict[tuple[str, int | None], torch.Tensor]] = {}

    @staticmethod
    def _dev_key(dev: torch.device) -> tuple[str, int | None]:
        # e.g. ("cpu", None) or ("cuda", 0)
        return (dev.type, dev.index)

    @classmethod
    def _col_masks(cls, dev: torch.device) -> torch.Tensor:
        """[7] int64 masks for each column's 6 playable bits."""
        dkey = cls._dev_key(dev)
        if dkey in cls._COL_MASKS_CACHE:
            return cls._COL_MASKS_CACHE[dkey]

        row_mask = (1 << cls.N_ROWS) - 1  # 0b111111
        cols = torch.arange(cls.N_COLUMNS, device=dev, dtype=torch.int64)
        base = cols * cls.COLUMN_BIT_OFFSET
        col_masks = torch.tensor(row_mask, device=dev, dtype=torch.int64) << base  # [7]
        cls._COL_MASKS_CACHE[dkey] = col_masks
        return col_masks

    @classmethod
    def _all_legal_mask(cls, dev: torch.device) -> torch.Tensor:
        """Scalar int64 mask with all legal playable squares set."""
        col_masks = cls._col_masks(dev)
        return col_masks.sum().to(dtype=torch.int64)

    @classmethod
    def _weights_base4(cls, dev: torch.device, n: int) -> torch.Tensor:
        """[1,1,n] int64 weights = 4**i cached per device and n."""
        dkey = cls._dev_key(dev)
        key = (dkey[0], dkey[1], n)
        w = cls._WEIGHTS_CACHE.get(key)
        if w is None:
            w = (4 ** torch.arange(n, device=dev, dtype=torch.int64)).view(1, 1, n)
            cls._WEIGHTS_CACHE[key] = w
        return w

    @classmethod
    def _pattern_masks(
        cls, patterns_bitidx: torch.Tensor, dev: torch.device
    ) -> torch.Tensor:
        """[1,M,N] int64 one-hot bit masks for patterns, cached per device + pattern object id."""
        if patterns_bitidx.dtype != torch.int64:
            raise TypeError("patterns_bitidx must be int64 (bit indices 0..63).")
        if patterns_bitidx.device != dev:
            # you can decide: enforce caller moves it, or auto-move.
            patterns_bitidx = patterns_bitidx.to(device=dev)

        dkey = cls._dev_key(dev)
        key = (dkey[0], dkey[1], id(patterns_bitidx))

        masks = cls._PATTERN_MASKS_CACHE.get(key)
        if masks is None:
            pat = patterns_bitidx.view(1, *patterns_bitidx.shape)  # [1,M,N]
            one = torch.ones((), device=dev, dtype=torch.int64)
            masks = one << pat  # [1,M,N] int64
            cls._PATTERN_MASKS_CACHE[key] = masks
        return masks

    def legal_moves_mask(self) -> torch.Tensor:
        """[B] int64 landing squares (reachable in next move)."""
        dev = self.all_tokens.device
        bottom = torch.full((), self.BB_BOTTOM_ROW, device=dev, dtype=torch.int64)
        all_legal = self._all_legal_mask(dev)
        return (self.all_tokens + bottom) & all_legal

    def _legal_moves_mask(self) -> torch.Tensor:
        return self.legal_moves_mask()
        # dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)
        # bottom = torch.full((), self.BB_BOTTOM_ROW, device=dev, dtype=torch.int64)
        # legal_sq = torch.full((), self.BB_ALL_LEGAL_TOKENS, device=dev, dtype=torch.int64)
        # return (self.all_tokens + bottom) & legal_sq  # [B] landing squares in each column

    def table_positions(self, patterns_bitidx: torch.Tensor) -> torch.Tensor:
        """Compute T for each board and each pattern (n-tuple).

        patterns_bitidx:
            [M,N] int64 tensor of bit indices (0..63) in your bitboard layout.

        Returns:
            T: [B,M] int64
        """
        dev = self.all_tokens.device
        B = self.all_tokens.shape[0]

        # cached: [1,M,N] one-hot bit masks
        masks = self._pattern_masks(patterns_bitidx, dev)  # [1,M,N]
        M, N = patterns_bitidx.shape

        all_tok = self.all_tokens.view(B, 1, 1)
        act_tok = self.active_tokens.view(B, 1, 1)

        occupied = (all_tok & masks) != 0  # [B,M,N] bool
        is_active = (act_tok & masks) != 0  # [B,M,N] bool

        # reachable = landing squares only
        reachable_mask = self.legal_moves_mask().view(B, 1, 1)  # [B,1,1]
        reachable = (~occupied) & ((reachable_mask & masks) != 0)  # [B,M,N] bool

        # Determine which color is "active player" from moves_left parity (matches your C++)
        active_is_yellow = (self.moves_left.to(torch.int64) & 1) == 0  # [B] bool

        # If active is yellow: active tokens are yellow, else active tokens are red.
        yellow = (
            torch.where(active_is_yellow.view(B, 1, 1), is_active, ~is_active)
            & occupied
        )
        red = occupied & ~yellow

        # s in {0,1,2,3}
        s = torch.zeros((B, M, N), device=dev, dtype=torch.int64)
        s = torch.where(reachable, torch.full_like(s, 3), s)
        s = torch.where(yellow, torch.full_like(s, 1), s)
        s = torch.where(red, torch.full_like(s, 2), s)

        # cached weights: [1,1,N]
        w = self._weights_base4(dev, N)
        return (s * w).sum(dim=2)  # [B,M]

    @staticmethod
    def _device_check(*tensors: torch.Tensor) -> torch.device:
        dev = tensors[0].device
        for t in tensors[1:]:
            if t.device != dev:
                raise ValueError("All tensors must be on the same device.")
        return dev

    @classmethod
    def empty(
        cls,
        batch_size: int,
        device: torch.device | str,
        *,
        moves_left_dtype: torch.dtype = torch.int16,
    ) -> BoardBatch:
        dev = torch.device(device)
        all_tokens = torch.zeros(batch_size, device=dev, dtype=torch.int64)
        active_tokens = torch.zeros(batch_size, device=dev, dtype=torch.int64)
        moves_left = torch.full(
            (batch_size,),
            cls.N_COLUMNS * cls.N_ROWS,
            device=dev,
            dtype=moves_left_dtype,
        )
        return cls(
            all_tokens=all_tokens, active_tokens=active_tokens, moves_left=moves_left
        )

    @staticmethod
    def _pow2(shift: torch.Tensor) -> torch.Tensor:
        # shift int64 in [0, 62] -> safe in int64
        return torch.ones_like(shift, dtype=torch.int64) << shift

    def _apply_move(self, mv: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """Apply move bitboards (mv) where legal, in-place. Returns legal mask."""
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, mv, legal
        )
        mv = mv.to(device=dev, dtype=torch.int64)
        legal = legal.to(device=dev, dtype=torch.bool)

        mask = -legal.to(torch.int64)  # -1 where legal else 0
        mv = mv & mask  # illegal -> 0

        # strict C++ semantics: only switch player if legal
        self.active_tokens ^= self.all_tokens & mask
        self.all_tokens ^= mv
        self.moves_left -= legal.to(self.moves_left.dtype)
        return legal

    # ---------------------------------------------------------------------
    # Play by columns (0..6), returns [B] bool legal
    # ---------------------------------------------------------------------
    def play_columns(self, columns: torch.Tensor) -> torch.Tensor:
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, columns
        )
        cols = columns.to(device=dev, dtype=torch.int64)

        in_range = (cols >= 0) & (cols < self.N_COLUMNS)
        cols_c = cols.clamp(0, self.N_COLUMNS - 1)

        # top cell in that column must be empty
        top_shift = cols_c * self.COLUMN_BIT_OFFSET + (self.N_ROWS - 1)
        top = self._pow2(top_shift)
        not_full = (self.all_tokens & top) == 0
        legal = in_range & not_full

        # column mask: (1<<(base+N_ROWS)) - (1<<base)
        base = cols_c * self.COLUMN_BIT_OFFSET
        lo = self._pow2(base)
        hi = self._pow2(base + self.N_ROWS)
        col_mask = hi - lo

        bottom = torch.full((), self.BB_BOTTOM_ROW, device=dev, dtype=torch.int64)

        # landing square
        mv = (self.all_tokens + bottom) & col_mask
        return self._apply_move(mv, legal)

    # ---------------------------------------------------------------------
    # Play by move masks (one-hot landing square per board), returns [B] bool legal
    # ---------------------------------------------------------------------
    def play_masks(self, mv: torch.Tensor) -> torch.Tensor:
        """mv: [B] int64, expected one bit set at the landing square."""
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, mv
        )
        mv = mv.to(device=dev, dtype=torch.int64)

        # Legal if:
        # 1) mv is non-zero
        # 2) mv is subset of legal landing squares (exactly those bits from legalMovesMask)
        legal_moves = self._legal_moves_mask()  # [B]
        legal = (mv != 0) & ((mv & legal_moves) == mv)  # mv ⊆ legal_moves

        # Optionally enforce "single bit" (one-hot). This costs popcount; skip for speed.
        # If you want a cheap-ish check: mv & (mv-1) == 0 (works for signed too if mv>0).
        # one_hot = (mv > 0) & ((mv & (mv - 1)) == 0)
        # legal &= one_hot

        return self._apply_move(mv, legal)

    # Backwards-compatible name
    def play(self, columns: torch.Tensor) -> torch.Tensor:
        return self.play_columns(columns)

    def has_win(self) -> torch.Tensor:
        y = self.active_tokens ^ self.all_tokens  # player who just moved

        x = y & (y << 2)
        win = (x & (x << 1)) != 0

        off = self.COLUMN_BIT_OFFSET

        x = y & (y << (2 * off))
        win |= (x & (x << off)) != 0

        d1 = off - 1
        x = y & (y << (2 * d1))
        win |= (x & (x << d1)) != 0

        d2 = off + 1
        x = y & (y << (2 * d2))
        win |= (x & (x << d2)) != 0

        return win

    def winning_positions(self, x: torch.Tensor, verticals: bool) -> torch.Tensor:
        """[B] int64 bitboard of winning squares for player bitboard x."""
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, x
        )
        x = x.to(device=dev, dtype=torch.int64)

        all_legal = torch.full(
            (), self.BB_ALL_LEGAL_TOKENS, device=dev, dtype=torch.int64
        )

        wins = torch.zeros_like(x)

        if verticals:
            wins |= (x << 1) & (x << 2) & (x << 3)

        off = self.COLUMN_BIT_OFFSET
        for b in (off - 1, off, off + 1):
            # left-ish directions
            tmp = (x << b) & (x << (2 * b))
            wins |= tmp & (x << (3 * b))
            wins |= tmp & (x >> b)

            # right-ish directions
            tmp = (x >> b) & (x >> (2 * b))
            wins |= tmp & (x << b)
            wins |= tmp & (x >> (3 * b))

        return wins & all_legal

    def can_win(self) -> torch.Tensor:
        """[B] bool: whether side to move has any immediate winning move."""
        dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)
        wins = self.winning_positions(self.active_tokens, verticals=True)  # [B] int64
        moves = self._legal_moves_mask()  # [B] int64
        return (wins & moves) != 0

    def generate_legal_moves(self) -> torch.Tensor:
        """[B] int64: all legal landing squares (one bit per legal column)."""
        return self._legal_moves_mask()

    def generate_non_losing_moves(self) -> torch.Tensor:
        """[B] int64: non-losing move bitboard (may be 0 for forced-loss positions)."""
        dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)
        moves = self._legal_moves_mask()  # landing squares for all columns

        # Opponent stones = active ^ all  (same as C++)
        opp = self.active_tokens ^ self.all_tokens

        threats = self.winning_positions(opp, verticals=True)

        direct_threats = threats & moves  # moves that block immediate opponent win
        has_direct = direct_threats != 0

        # "more than one direct threat?"  => (x & (x-1)) != 0  (same as C++)
        multi = (direct_threats & (direct_threats - 1)) != 0

        # If there is a direct threat:
        #   - if multiple: moves = 0
        #   - else:       moves = direct_threats
        moves = torch.where(
            has_direct,
            torch.where(multi, torch.zeros_like(moves), direct_threats),
            moves,
        )

        # No token under an opponent's threat.
        return moves & ~(threats >> 1)

    def can_win_column(self, columns: torch.Tensor) -> torch.Tensor:
        """[B] bool: whether playing `columns[b]` wins immediately for each board."""
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, columns
        )
        cols = columns.to(device=dev, dtype=torch.int64)

        in_range = (cols >= 0) & (cols < self.N_COLUMNS)
        cols_c = cols.clamp(0, self.N_COLUMNS - 1)

        # isLegalMove(column): check top cell in that column is empty
        top_shift = cols_c * self.COLUMN_BIT_OFFSET + (self.N_ROWS - 1)
        top = self._pow2(top_shift)  # [B] int64
        not_full = (self.all_tokens & top) == 0
        legal = in_range & not_full

        # getColumnMask(column): (1<<(base+N_ROWS)) - (1<<base)
        base = cols_c * self.COLUMN_BIT_OFFSET
        lo = self._pow2(base)
        hi = self._pow2(base + self.N_ROWS)
        col_mask = hi - lo

        wins = self.winning_positions(self.active_tokens, verticals=True)  # [B] int64
        moves = self._legal_moves_mask()  # [B] int64

        return legal & ((wins & moves & col_mask) != 0)

    def reset(self, done: torch.Tensor) -> None:
        """Reset boards where `done` is True back to the empty position (in-place).

        Args:
            done: [B] bool tensor. True means "reset this board".

        Notes:
            - No CPU sync.
            - No advanced indexing.
            - Keeps tensors allocated; only writes values.
        """
        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, done
        )
        d = done.to(device=dev, dtype=torch.bool)

        # mask: -1 where done else 0  (int64)
        m = -d.to(torch.int64)

        # Clear bitboards for done boards; keep others unchanged.
        # keep_mask is 0 for done, -1 for keep (all bits set).
        keep_mask = ~m
        self.all_tokens &= keep_mask
        self.active_tokens &= keep_mask

        # Reset moves_left for done boards to 42 (full empty board).
        full = torch.full(
            (), self.N_COLUMNS * self.N_ROWS, device=dev, dtype=self.moves_left.dtype
        )
        self.moves_left = torch.where(d, full, self.moves_left)

    def iter_move_masks(self, moves: torch.Tensor | None = None, *, max_moves: int = 7):
        """Yield one-hot move masks from a move-set bitboard.

        Args:
            moves:
                [B] int64 tensor where each entry is a bitboard with <= 7 bits set.
                If None, uses `self.generate_non_losing_moves()`.
            max_moves:
                Upper bound on number of yielded moves (Connect-4: 7).

        Yields:
            mv:
                [B] int64 tensor where each entry is either 0 (no move for that board
                in this iteration) or a one-hot bitboard with a single bit set.

        Example:
            Iterate non-losing moves and apply them on copies:
            ```python
            nl = board.generate_non_losing_moves()
            for mv in board.iter_move_masks(nl):
                active = mv != 0
                if not active.any():
                    break

                tmp = BoardBatch(
                    all_tokens=board.all_tokens.clone(),
                    active_tokens=board.active_tokens.clone(),
                    moves_left=board.moves_left.clone(),
                )
                tmp.play_masks(mv)
                win_after = tmp.has_win()
            ```
        """
        if moves is None:
            moves = self.generate_non_losing_moves()

        dev = self._device_check(
            self.all_tokens, self.active_tokens, self.moves_left, moves
        )
        m = moves.to(device=dev, dtype=torch.int64)

        for _ in range(max_moves):
            mv = m & -m  # extract lsb (one-hot)
            yield mv
            m = m ^ mv  # clear bit

    def mirror(self) -> BoardBatch:
        """Return a horizontally mirrored copy of the batch (0<->6, 1<->5, 2<->4)."""

        def mirror_bits(x: torch.Tensor) -> torch.Tensor:
            # Column masks for the 6 playable bits in each column.
            # mask(c) = ((1<<N_ROWS)-1) << (c*COLUMN_BIT_OFFSET)
            row_mask = (1 << self.N_ROWS) - 1  # 0b111111
            m0 = row_mask << (0 * self.COLUMN_BIT_OFFSET)
            m1 = row_mask << (1 * self.COLUMN_BIT_OFFSET)
            m2 = row_mask << (2 * self.COLUMN_BIT_OFFSET)
            m3 = row_mask << (3 * self.COLUMN_BIT_OFFSET)
            m4 = row_mask << (4 * self.COLUMN_BIT_OFFSET)
            m5 = row_mask << (5 * self.COLUMN_BIT_OFFSET)
            m6 = row_mask << (6 * self.COLUMN_BIT_OFFSET)

            s54 = 6 * self.COLUMN_BIT_OFFSET  # 54
            s36 = 4 * self.COLUMN_BIT_OFFSET  # 36
            s18 = 2 * self.COLUMN_BIT_OFFSET  # 18

            # Same logic as the C++ mirrorBitBoard():
            y = torch.zeros_like(x)
            y |= (x & m6) >> s54
            y |= (x & m0) << s54

            y |= (x & m5) >> s36
            y |= (x & m1) << s36

            y |= (x & m4) >> s18
            y |= (x & m2) << s18

            y |= x & m3  # center column unchanged
            return y

        return BoardBatch(
            all_tokens=mirror_bits(self.all_tokens),
            active_tokens=mirror_bits(self.active_tokens),
            moves_left=self.moves_left.clone(),  # preserve dtype; keep separate tensor
        )

    def reward(self) -> torch.Tensor:
        """Returns:
        [B] float tensor with:
            +1.0  -> yellow wins
            -1.0  -> red wins
            0.0  -> draw
            NaN  -> game not finished
        """
        dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)

        B = self.moves_left.shape[0]
        r = torch.full((B,), float("nan"), device=dev)

        win = self.has_win()  # last mover won
        draw = (self.moves_left == 0) & ~win

        # moves_left AFTER move:
        # odd -> yellow just moved (yellow starts and places the first token)
        # even  -> red just moved
        yellow_just_moved = (self.moves_left.to(torch.int64) & 1) == 1

        # Assign rewards
        r = torch.where(win & yellow_just_moved, torch.tensor(1.0, device=dev), r)
        r = torch.where(win & ~yellow_just_moved, torch.tensor(-1.0, device=dev), r)
        r = torch.where(draw, torch.tensor(0.0, device=dev), r)

        return r

    def active_player(self) -> torch.Tensor:
        """Returns:
        [B] int8 tensor:
            1 -> Yellow (starting player)
            2 -> Red (second player)
        """
        dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)

        # Even moves_left -> Yellow to move
        yellow_to_move = (self.moves_left.to(torch.int64) & 1) == 0

        return torch.where(
            yellow_to_move,
            torch.ones_like(self.moves_left, dtype=torch.int8, device=dev),
            torch.full_like(self.moves_left, 2, dtype=torch.int8, device=dev),
        )

    def active_player_sign(self) -> torch.Tensor:
        """Returns:
        [B] float32 tensor:
            +1.0 -> Yellow to move
            -1.0 -> Red to move
        """
        dev = self._device_check(self.all_tokens, self.active_tokens, self.moves_left)

        # Even moves_left -> Yellow to move
        yellow_to_move = (self.moves_left.to(torch.int64) & 1) == 0

        return torch.where(
            yellow_to_move,
            torch.ones_like(self.moves_left, dtype=torch.float32, device=dev),
            -torch.ones_like(self.moves_left, dtype=torch.float32, device=dev),
        )

    def done(self) -> torch.Tensor:
        return self.has_win() | (self.moves_left <= 0)

    @classmethod
    def clear_caches(cls) -> None:
        cls._WEIGHTS_CACHE.clear()
        cls._PATTERN_MASKS_CACHE.clear()
        cls._COL_MASKS_CACHE.clear()


def move_mask_to_column(mv: int, *, column_bit_offset: int = 9) -> int:
    """Return the column index (0..6) for a one-hot move mask.

    Args:
        mv: int64 bitboard with exactly one bit set (landing square).
        column_bit_offset: Bit stride between columns (default: 9).

    Returns:
        int: Column index (0..6).

    Raises:
        ValueError: If mv == 0 or mv has more than one bit set.
    """
    if mv == 0 or (mv & (mv - 1)) != 0:
        raise ValueError(f"mv must be one-hot, got {mv:#x}")

    bit_index = mv.bit_length() - 1
    return bit_index // column_bit_offset
