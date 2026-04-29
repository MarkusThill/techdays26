"""BitBullyArena: a lightweight Connect-4 arena for pitting agents against each other.

Design goals (v1):
- Deterministic runs via a global seed and per-game derived seeds (schedule order independent).
- Best-effort timeouts (no subprocess): measure after `best_move()` returns.
- ε-randomization per agent: each agent has its own ε-schedule; when colors swap,
  each agent keeps its ε for that game.
- Illegal move / exception / timeout => immediate loss (logged + recorded).
- Agents receive only a copy of the board (`board.copy()`).
- Optional score logging (`score_all_moves`) for diagnostics.

Assumptions (based on your Board semantics):
- Yellow == Player 1 == integer 1, starts the game.
- Red    == Player 2 == integer 2, goes second.

You can drop this module into your project and adapt naming/paths as needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from bitbully import Board
from bitbully.agent_interface import Connect4Agent

# -----------------------------
# Public configuration + types
# -----------------------------


class Color(int, Enum):
    """Player color / identity mapping (matches `Board.current_player()`)."""

    YELLOW = 1  # Player 1, starts
    RED = 2  # Player 2


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    agent: Connect4Agent
    colors: tuple[Color, ...] = (Color.YELLOW, Color.RED)

    # Per-agent epsilon schedule (sweep). This is NOT tied to color.
    # When colors swap, each agent keeps its epsilon for that game.
    epsilons: tuple[float, ...] = (0.0,)

    def can_play(self, color: Color) -> bool:
        return color in self.colors


@dataclass(frozen=True, slots=True)
class Matchup:
    """Explicit matchup: which agent plays yellow, which plays red.

    Use this to specify exact constellations instead of the default all-vs-all
    round-robin.  Example::

        Matchup(yellow_id="bitbully", red_id="random")
    """

    yellow_id: str
    red_id: str


@dataclass(frozen=True, slots=True)
class TimeControl:
    """Time controls for agents.

    Best-effort timeouts (no subprocess): we measure elapsed time and forfeit
    *after* `best_move()` returns if a limit is exceeded.
    """

    per_move_timeout_s: float | None = None
    per_game_budget_s: float | None = None


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Configuration for running an arena tournament.

    When *matchups* is ``None`` (default), the arena generates a full
    round-robin across all agents (both color assignments).
    When *matchups* is provided, **only** those explicit constellations are
    scheduled (each still crossed with the epsilon sweep from the respective
    ``AgentSpec``).
    """

    agents: tuple[AgentSpec, ...]
    n_games: int
    matchups: tuple[Matchup, ...] | None = None
    time_control: TimeControl = TimeControl()
    seed: int = 0

    log_scores: bool = False
    use_tqdm: bool = False
    logger: logging.Logger | None = None


class TerminationReason(str, Enum):
    CONNECT4 = "connect4"
    DRAW = "draw"
    ILLEGAL_MOVE = "illegal_move"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    INCOMPATIBLE_CONSTRAINTS = "incompatible_constraints"


@dataclass(frozen=True, slots=True)
class GamePlayers:
    """Assignment for a single game."""

    yellow_id: str
    red_id: str


@dataclass(frozen=True, slots=True)
class GameConfig:
    players: GamePlayers
    # Epsilon associated with each agent_id for this game instance.
    # Example: {"bitbully": 0.0, "td": 0.10}
    epsilon_by_agent: dict[str, float]
    seed: int


@dataclass(frozen=True, slots=True)
class MoveMeta:
    ply: int
    player: Color
    agent_id: str
    epsilon: float
    move: int
    was_epsilon_random: bool
    elapsed_s: float
    remaining_budget_s: float | None
    # Optional: store score dictionary for that move decision.
    # Only present if ArenaConfig.log_scores=True and call succeeded.
    scores: dict[int, int] | None = None


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Complete record of a single played game."""

    game_cfg: GameConfig
    moves: tuple[int, ...]
    move_meta: tuple[MoveMeta, ...]
    winner: Color | None
    termination: TerminationReason
    # Extra diagnostic info (exception text, illegal move, etc.)
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SkippedPairing:
    """Record a pairing that was not scheduled due to incompatible constraints."""

    agent_a: str
    agent_b: str
    reason: TerminationReason = TerminationReason.INCOMPATIBLE_CONSTRAINTS
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class AggregateRow:
    """Aggregate W/L/D counts for a matchup under a specific epsilon and color assignment.

    eps values are the epsilons of the *agents assigned to those colors* in that game.
    """

    agent_yellow: str
    agent_red: str
    epsilon_yellow: float
    epsilon_red: float
    games: int
    yellow_wins: int
    red_wins: int
    draws: int
    timeouts: int
    illegal_moves: int
    exceptions: int
    total_time_s: float


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses/enums/tuples into JSON-serializable structures."""
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: tuple[GameRecord, ...]
    aggregates: tuple[AggregateRow, ...]
    skipped: tuple[SkippedPairing, ...]

    _SERDE_VERSION: int = 1  # class-level constant-ish (still stored due to dataclass)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representing this result."""
        return {
            "version": self._SERDE_VERSION,
            "result": _to_jsonable(self),
        }

    def to_json_str(self, *, indent: int = 2) -> str:
        """Serialize this result to a JSON string."""
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Save this result as JSON to `path`."""
        path = Path(path)
        path.write_text(self.to_json_str(indent=indent), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ArenaResult:
        """Load an ArenaResult from a JSON file created by `save_json`."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        version = payload.get("version")
        if version != 1:
            raise ValueError(f"Unsupported arena result version: {version}")

        d = payload["result"]

        def mk_game_players(x: dict[str, Any]) -> GamePlayers:
            return GamePlayers(yellow_id=str(x["yellow_id"]), red_id=str(x["red_id"]))

        def mk_game_config(x: dict[str, Any]) -> GameConfig:
            return GameConfig(
                players=mk_game_players(x["players"]),
                epsilon_by_agent={
                    str(k): float(v) for k, v in x["epsilon_by_agent"].items()
                },
                seed=int(x["seed"]),
            )

        def mk_move_meta(x: dict[str, Any]) -> MoveMeta:
            return MoveMeta(
                ply=int(x["ply"]),
                player=Color(int(x["player"])),
                agent_id=str(x["agent_id"]),
                epsilon=float(x["epsilon"]),
                move=int(x["move"]),
                was_epsilon_random=bool(x["was_epsilon_random"]),
                elapsed_s=float(x["elapsed_s"]),
                remaining_budget_s=(
                    None
                    if x["remaining_budget_s"] is None
                    else float(x["remaining_budget_s"])
                ),
                scores=(
                    None
                    if x.get("scores") is None
                    else {int(k): int(v) for k, v in x["scores"].items()}
                ),
            )

        def mk_game_record(x: dict[str, Any]) -> GameRecord:
            return GameRecord(
                game_cfg=mk_game_config(x["game_cfg"]),
                moves=tuple(int(m) for m in x["moves"]),
                move_meta=tuple(mk_move_meta(mm) for mm in x["move_meta"]),
                winner=(None if x["winner"] is None else Color(int(x["winner"]))),
                termination=TerminationReason(str(x["termination"])),
                detail=(None if x.get("detail") is None else str(x["detail"])),
            )

        def mk_skipped_pairing(x: dict[str, Any]) -> SkippedPairing:
            return SkippedPairing(
                agent_a=str(x["agent_a"]),
                agent_b=str(x["agent_b"]),
                reason=TerminationReason(
                    str(
                        x.get(
                            "reason", TerminationReason.INCOMPATIBLE_CONSTRAINTS.value
                        )
                    )
                ),
                detail=(None if x.get("detail") is None else str(x["detail"])),
            )

        def mk_aggregate_row(x: dict[str, Any]) -> AggregateRow:
            return AggregateRow(
                agent_yellow=str(x["agent_yellow"]),
                agent_red=str(x["agent_red"]),
                epsilon_yellow=float(x["epsilon_yellow"]),
                epsilon_red=float(x["epsilon_red"]),
                games=int(x["games"]),
                yellow_wins=int(x["yellow_wins"]),
                red_wins=int(x["red_wins"]),
                draws=int(x["draws"]),
                timeouts=int(x["timeouts"]),
                illegal_moves=int(x["illegal_moves"]),
                exceptions=int(x["exceptions"]),
                total_time_s=float(x.get("total_time_s", 0.0)),
            )

        return cls(
            games=tuple(mk_game_record(gr) for gr in d["games"]),
            aggregates=tuple(mk_aggregate_row(ar) for ar in d["aggregates"]),
            skipped=tuple(mk_skipped_pairing(sp) for sp in d["skipped"]),
        )


# -----------------------------
# Usage
# -----------------------------
# result = arena.run(cfg)
# result.save_json("arena_result.json")
# result2 = ArenaResult.load_json("arena_result.json")


# -----------------------------
# Example baseline agent
# -----------------------------


class RandomAgent:
    """Agent that plays a uniformly random legal move."""

    def score_all_moves(self, board: Board) -> dict[int, int]:
        return dict.fromkeys(board.legal_moves(), 0)

    def best_move(self, board: Board) -> int:
        return random.choice(board.legal_moves())


# -----------------------------
# Arena implementation
# -----------------------------


class BitBullyArena:
    def __init__(self) -> None:
        pass

    def run(self, cfg: ArenaConfig) -> ArenaResult:
        logger = cfg.logger or logging.getLogger(__name__)

        # Validate config basics
        if cfg.n_games <= 0:
            raise ValueError(f"n_games must be > 0, got {cfg.n_games}")

        for spec in cfg.agents:
            if len(spec.epsilons) == 0:
                raise ValueError(f"Agent {spec.agent_id}: epsilons must not be empty.")
            for eps in spec.epsilons:
                if not (0.0 <= eps <= 1.0):
                    raise ValueError(
                        f"Agent {spec.agent_id}: epsilon must be in [0,1], got {eps}"
                    )

        # Build lookup from agent_id -> AgentSpec
        agent_specs = list(cfg.agents)
        spec_by_id: dict[str, AgentSpec] = {s.agent_id: s for s in agent_specs}

        skipped: list[SkippedPairing] = []
        planned_games: list[GameConfig] = []

        if cfg.matchups is not None:
            # --- Explicit matchups mode ---
            for mu in cfg.matchups:
                if mu.yellow_id not in spec_by_id:
                    raise ValueError(
                        f"Matchup references unknown agent_id: {mu.yellow_id!r}"
                    )
                if mu.red_id not in spec_by_id:
                    raise ValueError(
                        f"Matchup references unknown agent_id: {mu.red_id!r}"
                    )
                y_spec = spec_by_id[mu.yellow_id]
                r_spec = spec_by_id[mu.red_id]

                for eps_y in y_spec.epsilons:
                    for eps_r in r_spec.epsilons:
                        eps_map = {mu.yellow_id: eps_y, mu.red_id: eps_r}
                        for k in range(cfg.n_games):
                            planned_games.append(
                                GameConfig(
                                    players=GamePlayers(
                                        yellow_id=mu.yellow_id,
                                        red_id=mu.red_id,
                                    ),
                                    epsilon_by_agent=eps_map,
                                    seed=_derive_game_seed(
                                        cfg.seed,
                                        mu.yellow_id,
                                        mu.red_id,
                                        Color.YELLOW,
                                        Color.RED,
                                        eps_y,
                                        eps_r,
                                        k,
                                    ),
                                )
                            )
        else:
            # --- Round-robin mode (default) ---
            # For each unordered pairing (i<j), run both color swaps and
            # all (eps_a, eps_b) combinations.
            for i in range(len(agent_specs)):
                for j in range(i + 1, len(agent_specs)):
                    a = agent_specs[i]
                    b = agent_specs[j]

                    for eps_a in a.epsilons:
                        for eps_b in b.epsilons:
                            eps_map = {a.agent_id: eps_a, b.agent_id: eps_b}

                            # assignment 1: A yellow, B red
                            if a.can_play(Color.YELLOW) and b.can_play(Color.RED):
                                for k in range(cfg.n_games):
                                    planned_games.append(
                                        GameConfig(
                                            players=GamePlayers(
                                                yellow_id=a.agent_id, red_id=b.agent_id
                                            ),
                                            epsilon_by_agent=eps_map,
                                            seed=_derive_game_seed(
                                                cfg.seed,
                                                a.agent_id,
                                                b.agent_id,
                                                Color.YELLOW,
                                                Color.RED,
                                                eps_a,
                                                eps_b,
                                                k,
                                            ),
                                        )
                                    )
                            else:
                                skipped.append(
                                    SkippedPairing(
                                        agent_a=a.agent_id,
                                        agent_b=b.agent_id,
                                        detail=(
                                            f"Cannot assign {a.agent_id}=YELLOW and {b.agent_id}=RED under constraints."
                                        ),
                                    )
                                )

                            # assignment 2: B yellow, A red
                            if b.can_play(Color.YELLOW) and a.can_play(Color.RED):
                                for k in range(cfg.n_games):
                                    planned_games.append(
                                        GameConfig(
                                            players=GamePlayers(
                                                yellow_id=b.agent_id, red_id=a.agent_id
                                            ),
                                            epsilon_by_agent=eps_map,
                                            seed=_derive_game_seed(
                                                cfg.seed,
                                                a.agent_id,
                                                b.agent_id,
                                                Color.RED,
                                                Color.YELLOW,
                                                eps_a,
                                                eps_b,
                                                k,
                                            ),
                                        )
                                    )
                            else:
                                skipped.append(
                                    SkippedPairing(
                                        agent_a=a.agent_id,
                                        agent_b=b.agent_id,
                                        detail=(
                                            f"Cannot assign {b.agent_id}=YELLOW and {a.agent_id}=RED under constraints."
                                        ),
                                    )
                                )

        # Map agent_id -> agent instance
        agent_by_id: dict[str, Connect4Agent] = {
            aid: s.agent for aid, s in spec_by_id.items()
        }

        # Optional progress bar
        iterator = planned_games
        if cfg.use_tqdm:
            try:
                from tqdm import tqdm  # type: ignore[import-not-found]

                iterator = tqdm(planned_games, desc="BitBullyArena", unit="game")
            except Exception:
                iterator = planned_games

        games_out: list[GameRecord] = []
        for game_cfg in iterator:
            gr = self._play_one_game(
                game_cfg=game_cfg,
                agent_by_id=agent_by_id,
                time_control=cfg.time_control,
                seed=game_cfg.seed,
                log_scores=cfg.log_scores,
                logger=logger,
            )
            games_out.append(gr)

        aggregates = _aggregate_games(games_out)

        return ArenaResult(
            games=tuple(games_out),
            aggregates=tuple(aggregates),
            skipped=tuple(skipped),
        )

    def _play_one_game(
        self,
        *,
        game_cfg: GameConfig,
        agent_by_id: dict[str, Connect4Agent],
        time_control: TimeControl,
        seed: int,
        log_scores: bool,
        logger: logging.Logger,
    ) -> GameRecord:
        rng = random.Random(seed)

        yellow = agent_by_id[game_cfg.players.yellow_id]
        red = agent_by_id[game_cfg.players.red_id]

        board = Board()  # always start empty (no opening noise)
        moves: list[int] = []
        meta: list[MoveMeta] = []

        # Remaining per-game budgets (seconds)
        budget_y: float | None = time_control.per_game_budget_s
        budget_r: float | None = time_control.per_game_budget_s

        while not board.is_game_over():
            player_int = board.current_player()
            player = Color(player_int)

            agent_id = (
                game_cfg.players.yellow_id
                if player == Color.YELLOW
                else game_cfg.players.red_id
            )
            agent = yellow if player == Color.YELLOW else red
            eps = game_cfg.epsilon_by_agent[agent_id]

            # Choose move (epsilon-random or agent)
            board_for_agent = board.copy()  # pass a copy to agent
            ply = board.count_tokens()

            start = time.perf_counter()
            was_eps = False
            scores: dict[int, int] | None = None

            try:
                if rng.random() < eps:
                    was_eps = True
                    move = self._random_non_losing_move(board, rng)
                    elapsed = time.perf_counter() - start
                else:
                    move = agent.best_move(board_for_agent)
                    elapsed = time.perf_counter() - start

                    if log_scores:
                        # Score logging is best-effort and should not affect legality enforcement.
                        try:
                            scores = agent.score_all_moves(board_for_agent)
                        except Exception as e:
                            logger.warning(
                                "score_all_moves() failed for agent=%s at ply=%d: %r",
                                agent_id,
                                ply,
                                e,
                            )
                            scores = None

            except Exception as e:
                # Agent exception => loss for that agent
                elapsed = time.perf_counter() - start
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Agent exception: agent=%s player=%s ply=%d err=%r",
                    agent_id,
                    player.name,
                    ply,
                    e,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.EXCEPTION,
                    detail=f"agent={agent_id} err={e!r}",
                )

            # Time accounting (best-effort)
            remaining_budget: float | None
            if player == Color.YELLOW:
                if budget_y is not None:
                    budget_y -= elapsed
                remaining_budget = budget_y
            else:
                if budget_r is not None:
                    budget_r -= elapsed
                remaining_budget = budget_r

            # Record meta for the decision (even if it ends game immediately after)
            meta.append(
                MoveMeta(
                    ply=ply,
                    player=player,
                    agent_id=agent_id,
                    epsilon=float(eps),
                    move=int(move),
                    was_epsilon_random=was_eps,
                    elapsed_s=float(elapsed),
                    remaining_budget_s=(
                        None if remaining_budget is None else float(remaining_budget)
                    ),
                    scores=scores,
                )
            )

            # Timeout checks (best-effort: we can only decide now)
            if (
                time_control.per_move_timeout_s is not None
                and elapsed > time_control.per_move_timeout_s
            ):
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Per-move timeout: agent=%s player=%s ply=%d elapsed=%.6f limit=%.6f",
                    agent_id,
                    player.name,
                    ply,
                    elapsed,
                    time_control.per_move_timeout_s,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.TIMEOUT,
                    detail=(
                        f"per-move timeout: agent={agent_id} elapsed={elapsed:.6f}s "
                        f"limit={time_control.per_move_timeout_s:.6f}s"
                    ),
                )

            if remaining_budget is not None and remaining_budget < 0.0:
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Per-game budget exceeded: agent=%s player=%s ply=%d remaining=%.6f",
                    agent_id,
                    player.name,
                    ply,
                    remaining_budget,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.TIMEOUT,
                    detail=f"per-game budget exceeded: agent={agent_id} remaining={remaining_budget:.6f}s",
                )

            # Illegal move => immediate loss
            if not isinstance(move, int) or not board.is_legal_move(int(move)):
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Illegal move: agent=%s player=%s ply=%d move=%r",
                    agent_id,
                    player.name,
                    ply,
                    move,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.ILLEGAL_MOVE,
                    detail=f"illegal move: agent={agent_id} move={move!r}",
                )

            # Apply move
            ok = board.play(int(move))
            if not ok:
                # Extremely defensive: `is_legal_move` said OK but play failed.
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Move application failed: agent=%s player=%s ply=%d move=%r",
                    agent_id,
                    player.name,
                    ply,
                    move,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.ILLEGAL_MOVE,
                    detail=f"play() failed despite is_legal_move: agent={agent_id} move={move!r}",
                )

            moves.append(int(move))

        # Normal termination
        winner_int = board.winner()
        if winner_int is None:
            return GameRecord(
                game_cfg=game_cfg,
                moves=tuple(moves),
                move_meta=tuple(meta),
                winner=None,
                termination=TerminationReason.DRAW,
                detail=None,
            )

        winner = Color(winner_int)
        return GameRecord(
            game_cfg=game_cfg,
            moves=tuple(moves),
            move_meta=tuple(meta),
            winner=winner,
            termination=TerminationReason.CONNECT4,
            detail=None,
        )

    @staticmethod
    def _random_non_losing_move(board: Board, rng: random.Random) -> int:
        candidates = board.legal_moves(non_losing=True)
        if candidates:
            return rng.choice(candidates)
        # Forced-loss state: pick any legal move
        all_legal = board.legal_moves()
        # Defensive: if empty, game should be over, but avoid crash
        return rng.choice(all_legal) if all_legal else 0


# -----------------------------
# Helpers
# -----------------------------


def _derive_game_seed(
    global_seed: int,
    agent_a: str,
    agent_b: str,
    # include assignment info so the two color-swapped schedules differ deterministically
    a_as: Color,
    b_as: Color,
    # epsilons are agent-owned (eps_a for agent_a, eps_b for agent_b)
    eps_a: float,
    eps_b: float,
    game_idx: int,
) -> int:
    """Derive a deterministic per-game seed from the global seed and matchup tuple.

    This makes randomness stable even if scheduling order changes.
    """
    payload = (
        f"{global_seed}|{agent_a}|{agent_b}|{int(a_as)}|{int(b_as)}|"
        f"{eps_a:.10f}|{eps_b:.10f}|{game_idx}"
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    # Convert 8 bytes -> int in [0, 2^64)
    return int.from_bytes(digest, byteorder="little", signed=False)


def _aggregate_games(games: list[GameRecord]) -> list[AggregateRow]:
    """Aggregate per (agent_yellow, agent_red, eps_y, eps_r)."""
    # Key: (yellow_id, red_id, eps_y, eps_r)
    agg: dict[tuple[str, str, float, float], dict[str, float]] = {}

    for g in games:
        y_id = g.game_cfg.players.yellow_id
        r_id = g.game_cfg.players.red_id
        eps_y = g.game_cfg.epsilon_by_agent[y_id]
        eps_r = g.game_cfg.epsilon_by_agent[r_id]

        key = (y_id, r_id, eps_y, eps_r)

        if key not in agg:
            agg[key] = dict(
                games=0,
                yellow_wins=0,
                red_wins=0,
                draws=0,
                timeouts=0,
                illegal_moves=0,
                exceptions=0,
                total_time_s=0.0,
            )

        a = agg[key]
        a["games"] += 1
        a["total_time_s"] += sum(mm.elapsed_s for mm in g.move_meta)

        if g.termination == TerminationReason.DRAW:
            a["draws"] += 1
        elif g.winner == Color.YELLOW:
            a["yellow_wins"] += 1
        elif g.winner == Color.RED:
            a["red_wins"] += 1

        if g.termination == TerminationReason.TIMEOUT:
            a["timeouts"] += 1
        elif g.termination == TerminationReason.ILLEGAL_MOVE:
            a["illegal_moves"] += 1
        elif g.termination == TerminationReason.EXCEPTION:
            a["exceptions"] += 1

    out: list[AggregateRow] = []
    for (y_id, r_id, eps_y, eps_r), d in agg.items():
        out.append(
            AggregateRow(
                agent_yellow=y_id,
                agent_red=r_id,
                epsilon_yellow=eps_y,
                epsilon_red=eps_r,
                games=int(d["games"]),
                yellow_wins=int(d["yellow_wins"]),
                red_wins=int(d["red_wins"]),
                draws=int(d["draws"]),
                timeouts=int(d["timeouts"]),
                illegal_moves=int(d["illegal_moves"]),
                exceptions=int(d["exceptions"]),
                total_time_s=d["total_time_s"],
            )
        )

    # Stable ordering for convenience
    out.sort(
        key=lambda row: (
            row.agent_yellow,
            row.agent_red,
            row.epsilon_yellow,
            row.epsilon_red,
        )
    )
    return out


def format_aggregate_table(result: Any) -> str:
    """Builds a nicely formatted table for `result.aggregates` and adds a final score:
      score = (+1 * yellow_wins) + (-1 * red_wins) + (0 * draws) = yellow_wins - red_wins

    Returns:
        A single string (ready to print or write to a file).
    """
    rows: list[dict[str, Any]] = []
    for r in result.aggregates:
        score = int(r.yellow_wins) - int(r.red_wins)

        rows.append({
            "yellow": r.agent_yellow,
            "red": r.agent_red,
            "eps_y": float(r.epsilon_yellow),
            "eps_r": float(r.epsilon_red),
            "games": int(r.games),
            "Y_w": int(r.yellow_wins),
            "R_w": int(r.red_wins),
            "D": int(r.draws),
            "score": score,
            "avg": (score / int(r.games))
            if int(r.games)
            else 0.0,  # normalized [-1, 1]
            "timeouts": int(getattr(r, "timeouts", 0)),
            "illegal": int(getattr(r, "illegal_moves", 0)),
            "exc": int(getattr(r, "exceptions", 0)),
            "time": float(getattr(r, "total_time_s", 0.0)),
        })

    # Column order + formatting
    cols = [
        ("yellow", "Y", "{}"),
        ("red", "R", "{}"),
        ("eps_y", "εY", "{:.2f}"),
        ("eps_r", "εR", "{:.2f}"),
        ("games", "G", "{:d}"),
        ("Y_w", "Ywin", "{:d}"),
        ("R_w", "Rwin", "{:d}"),
        ("D", "Draw", "{:d}"),
        ("score", "Score", "{:d}"),
        ("avg", "Avg", "{:+.3f}"),
        ("timeouts", "TO", "{:d}"),
        ("illegal", "Ill", "{:d}"),
        ("exc", "Exc", "{:d}"),
        ("time", "Time", "{:.1f}s"),
    ]

    # Pre-render cells to compute widths
    rendered: list[list[str]] = []
    for row in rows:
        rendered.append([fmt.format(row[key]) for key, _, fmt in cols])

    widths = []
    for j, (_, header, _) in enumerate(cols):
        w = len(header)
        for i in range(len(rendered)):
            w = max(w, len(rendered[i][j]))
        widths.append(w)

    def pack_line(parts: list[str]) -> str:
        return " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    header = pack_line([h for _, h, _ in cols])
    sep = "-+-".join("-" * w for w in widths)

    lines = [header, sep]
    for cells in rendered:
        lines.append(pack_line(cells))

    return "\n".join(lines) + "\n"


def get_table_legend():
    legend = (
        "Table legend:\n"
        "Y      = Yellow agent ID\n"
        "R      = Red agent ID\n"
        "εY     = Epsilon value used for the Yellow agent\n"
        "εR     = Epsilon value used for the Red agent\n"
        "G      = Number of games played for this pairing/configuration\n"
        "Ywin   = Number of games won by Yellow\n"
        "Rwin   = Number of games won by Red\n"
        "Draw   = Number of draws\n"
        "Score  = Final score = Ywin - Rwin  (Yellow win = +1, Red win = -1, Draw = 0)\n"
        "Avg    = Normalized score per game = Score / G  (range [-1, +1])\n"
        "TO     = Games lost due to timeout\n"
        "Ill    = Games lost due to illegal moves\n"
        "Exc    = Games lost due to exceptions\n"
        "Time   = Total wall-clock time (seconds) spent on all moves in this group\n"
    )
    return legend


# print(legend)

# Usage:
# table_str = format_aggregate_table(result)
# print(table_str)
# or write it to a file:
# with open("arena_summary.txt", "a", encoding="utf-8") as f:
#     f.write(table_str + "\n")
