# Connect-4 Example
---
- Try Re-training from scratch
- Evaluate performance during training (against random and partially random agents)
- Allow fix match combinations for GameArena (or skip certain combinations)
- Refactorings: Notebooks -> codebase
- Unit Tests for refactored classes necessary.
- Docstrings + ruff + mypy/pyrefly/ty linting
- Investigate efficient way to implement eligibility traces (see Sutton & Barto)
- final learning rate -> do not converge to 0
- look at the the non-zero updates
- implement own update rule

# BitBully TODOs:
- Reset function for Board
- Tree-search with random rollout
- expose winningMoves to pybind
- get onehot move masks from move generation
- play fast move using onehot move mask
- play() is not documented apparently in .pyi file
- dumb down BitBully using limited tree depth, etc.