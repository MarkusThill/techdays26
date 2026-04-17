# Illustration Example
---
- Let RL agent play against ChatGPT. Compare #weights against ChatGPT.
- compare to https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl

# Connect-4 Example
---
- Combine TD agent with shallow tree search
- Unit Tests for refactored classes necessary.
- Docstrings + ruff + mypy/pyrefly/ty linting
- Investigate efficient way to implement eligibility traces (see Sutton & Barto)
- probably own step-size-adaptation needed
- implement own update rule

# BitBully TODOs:
- Reset function for Board
- expose winningMoves to pybind
- get onehot move masks from move generation
- play fast move using onehot move mask
- play() is not documented apparently in .pyi file