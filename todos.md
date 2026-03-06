# Frozen Lake Example
- On-Policy vs. Off-Policy:
  On-Policy would avoid getting too close to lakes:
    - Maybe some example where off-policy would be faster if there is no random chance of falling in the lake
      but on-policy is better (keeping further from the holes) although it might require some Umweg!

# Connect-4 Example
---
- look for "afterstates" in suttons book!
- compare to https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl
- allow to evaluate a model without saving it first to hard-drive.
- keep track of the currently best model (weighted scores?).
- Interactive Bitboard Illustration
  - Then: N-Tuple systems. Sparsity: Compute how many realizations are possible per LUT for a given n-tuple
- Combine TD agent with shallow tree search
- visualize n-tuples
- Refactorings: Notebooks -> codebase
- Try CNNs?
- Unit Tests for refactored classes necessary.
- Docstrings + ruff + mypy/pyrefly/ty linting
- Investigate efficient way to implement eligibility traces (see Sutton & Barto)
- look at the the non-zero updates
- implement own update rule

# BitBully TODOs:
- Reset function for Board
- expose winningMoves to pybind
- get onehot move masks from move generation
- play fast move using onehot move mask
- play() is not documented apparently in .pyi file

# C4 Facts
- 4,531,985,219,092 states
- weakly solved by Allis (1988)
- first strong solution took 40,000 h (4.5 years)
- Checking one position per millisecond (1000 pos / s) would still take ~143 years.
- Even at a million positions per second, brute force takes ~52 days.
- There are over 500 Connect-4 positions for every human on Earth.

