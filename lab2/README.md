# Lab 2 — 🎮 Connect-4 with an N-Tuple TD(λ) Agent

Train a Temporal-Difference agent to play Connect-4 by self-play, using an
N-Tuple Network as the value function and a bitboard environment for
thousands of parallel games per GPU/CPU tensor.

## Notebooks (workshop flow)

1. [`0_problem_illustration.ipynb`](0_problem_illustration.ipynb) — concepts: bitboards, afterstates, N-Tuple Network, TD(λ) update.
2. [`1_train_ntuple_net.ipynb`](1_train_ntuple_net.ipynb) — parallel self-play training loop with target network, Adam, LR schedule, and BitBully arena evaluation.
3. [`2_plot_metrics.ipynb`](2_plot_metrics.ipynb) — compare training metrics and arena results across runs.

## Utility notebook

- [`3_legacy_td_ntuple_c4_agent.ipynb`](3_legacy_td_ntuple_c4_agent.ipynb) — load and export **legacy** TD-agent weight files. Not part of the main workshop flow.

## What you'll learn

- The bitboard representation and how to run batched Connect-4 games.
- Afterstate value functions and why they simplify learning.
- Truncated TD(λ) as a bias-variance trade-off between TD(0) and Monte Carlo.
- Target networks, Polyak averaging, and ε-greedy exploration.
- How to evaluate an RL agent against a strong α–β baseline.

> ⬅️ Prerequisites: [`../lab1/`](../lab1/) for the core RL concepts used here.
