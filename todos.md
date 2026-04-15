# General
- Write General README file
- Write Readme files for the labs lab1 & lab2

# Frozen Lake Example
- Align Lab 1 with Lab 2

# Illustration Example
---
- Add a visualization of the tanh() and its deriv
- Let RL agent play against ChatGPT. Compare #weights against ChatGPT.
- compare to https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl

# Connect-4 Example
---
- keep track of the currently best model (weighted scores?).
- N-Tuple systems. Sparsity: Compute exactly how many realizations are possible per LUT for a given n-tuple
- Combine TD agent with shallow tree search
- Refactorings: Notebooks -> codebase
- Unit Tests for refactored classes necessary.
- Docstrings + ruff + mypy/pyrefly/ty linting
- Investigate efficient way to implement eligibility traces (see Sutton & Barto)
- implement own update rule

# BitBully TODOs:
- Reset function for Board
- expose winningMoves to pybind
- get onehot move masks from move generation
- play fast move using onehot move mask
- play() is not documented apparently in .pyi file