# Frozen Lake Example
- On-Policy vs. Off-Policy:
  On-Policy would avoid getting too close to lakes:
    - Maybe some example where off-policy would be faster if there is no random chance of falling in the lake
      but on-policy is better (keeping further from the holes) although it might require some Umweg!

# Illustration Example
---
- Add a visualization of the tanh() and its deriv
- Formula for number of states for n-tuples. Integrate into widget!
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