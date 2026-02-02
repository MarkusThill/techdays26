from __future__ import annotations

from typing import Protocol
import bitbully
from bitbully.agent_interface import Connect4Agent

def play_match(
    agent_yellow: Connect4Agent,
    agent_red: Connect4Agent,
    *,
    start: bitbully.Board | None = None,
    max_plies: int = 42,
    verbose: int = 0,
) -> int:
    """
    Play a full game between two agents on your BitBully `Board`.

    Returns:
        0 -> draw
        1 -> yellow win
        2 -> red win
    """
    board = start.copy() if start is not None else bitbully.Board()

    # Safety: if someone passes a terminal board
    if board.is_game_over():
        w = board.winner()
        return 0 if w is None else int(w)

    plies = 0
    while not board.is_game_over():
        assert plies < max_plies
        # Should never happen in Connect-4, but keeps us robust

        player = board.current_player()  # 1=yellow, 2=red
        agent = agent_yellow if player == 1 else agent_red

        move = agent.best_move(board)

        if not board.is_legal_move(move):
            raise ValueError(
                f"Agent selected illegal move {move} for player {player}. "
                f"Legal moves: {board.legal_moves()}"
            )

        ok = board.play(move)
        if not ok:
            # `play()` returns False on illegal moves; we already checked, so this would indicate a bug.
            raise RuntimeError(f"Board.play({move}) returned False although move was legal.")

        plies += 1

        if verbose >= 1:
            print(board)

    w = board.winner()
    return 0 if w is None else int(w)