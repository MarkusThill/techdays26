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

from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
import random
import time

from bitbully import Board
from bitbully.agent_interface import Connect4Agent


# -----------------------------
# Public configuration + types
# -----------------------------

class Color(int, Enum):
    """Player color / identity mapping (matches `Board.current_player()`)."""
    YELLOW = 1  # Player 1, starts
    RED = 2     # Player 2


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
class TimeControl:
    """Time controls for agents.

    Best-effort timeouts (no subprocess): we measure elapsed time and forfeit
    *after* `best_move()` returns if a limit is exceeded.
    """
    per_move_timeout_s: float | None = None
    per_game_budget_s: float | None = None


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Configuration for running an arena tournament."""
    agents: tuple[AgentSpec, ...]
    n_games: int
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


@dataclass(frozen=True, slots=True)
class ArenaResult:
    games: tuple[GameRecord, ...]
    aggregates: tuple[AggregateRow, ...]
    skipped: tuple[SkippedPairing, ...]


# -----------------------------
# Example baseline agent
# -----------------------------

class RandomAgent:
    """Agent that plays a uniformly random legal move."""

    def score_all_moves(self, board: Board) -> dict[int, int]:
        return {c: 0 for c in board.legal_moves()}

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
                    raise ValueError(f"Agent {spec.agent_id}: epsilon must be in [0,1], got {eps}")

        # Plan schedule:
        # for each unordered pairing (i<j), run both color swaps and all (eps_a, eps_b) combinations.
        agent_specs = list(cfg.agents)
        skipped: list[SkippedPairing] = []
        planned_games: list[GameConfig] = []

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
                                        players=GamePlayers(yellow_id=a.agent_id, red_id=b.agent_id),
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
                                        players=GamePlayers(yellow_id=b.agent_id, red_id=a.agent_id),
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
        agent_by_id: dict[str, Connect4Agent] = {s.agent_id: s.agent for s in agent_specs}

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

            agent_id = game_cfg.players.yellow_id if player == Color.YELLOW else game_cfg.players.red_id
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
                                agent_id, ply, e,
                            )
                            scores = None

            except Exception as e:
                # Agent exception => loss for that agent
                elapsed = time.perf_counter() - start
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Agent exception: agent=%s player=%s ply=%d err=%r",
                    agent_id, player.name, ply, e,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.EXCEPTION,
                    detail=f"agent={agent_id} err={repr(e)}",
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
                    remaining_budget_s=(None if remaining_budget is None else float(remaining_budget)),
                    scores=scores,
                )
            )

            # Timeout checks (best-effort: we can only decide now)
            if time_control.per_move_timeout_s is not None and elapsed > time_control.per_move_timeout_s:
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Per-move timeout: agent=%s player=%s ply=%d elapsed=%.6f limit=%.6f",
                    agent_id, player.name, ply, elapsed, time_control.per_move_timeout_s,
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
                    agent_id, player.name, ply, remaining_budget,
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
                    agent_id, player.name, ply, move,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.ILLEGAL_MOVE,
                    detail=f"illegal move: agent={agent_id} move={repr(move)}",
                )

            # Apply move
            ok = board.play(int(move))
            if not ok:
                # Extremely defensive: `is_legal_move` said OK but play failed.
                winner = Color.RED if player == Color.YELLOW else Color.YELLOW
                logger.warning(
                    "Move application failed: agent=%s player=%s ply=%d move=%r",
                    agent_id, player.name, ply, move,
                )
                return GameRecord(
                    game_cfg=game_cfg,
                    moves=tuple(moves),
                    move_meta=tuple(meta),
                    winner=winner,
                    termination=TerminationReason.ILLEGAL_MOVE,
                    detail=f"play() failed despite is_legal_move: agent={agent_id} move={repr(move)}",
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
    agg: dict[tuple[str, str, float, float], dict[str, int]] = {}

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
            )

        a = agg[key]
        a["games"] += 1

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
                games=d["games"],
                yellow_wins=d["yellow_wins"],
                red_wins=d["red_wins"],
                draws=d["draws"],
                timeouts=d["timeouts"],
                illegal_moves=d["illegal_moves"],
                exceptions=d["exceptions"],
            )
        )

    # Stable ordering for convenience
    out.sort(key=lambda row: (row.agent_yellow, row.agent_red, row.epsilon_yellow, row.epsilon_red))
    return out