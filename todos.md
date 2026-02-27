# Connect-4 Example
---
- AUC for curves
- Connect-4 Basics:
  - Number of positions
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
- Checking one position per millisecond (1000 pos / s) would still take ~143 years.
- Even at a million positions per second, brute force takes ~52 days.
- There are over 500 Connect-4 positions for every human on Earth.