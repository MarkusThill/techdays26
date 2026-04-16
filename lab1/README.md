# Lab 1 — ❄️ Frozen Lake: A Gentle Introduction to Reinforcement Learning

Grid-world playground for the core RL concepts: states, actions, policies,
value functions, Monte Carlo rollouts, TD(0), and Q-learning.

## Notebooks

1. [`0_frozen_lake_problem.ipynb`](0_frozen_lake_problem.ipynb) — the full lab walkthrough.

## What you'll learn

- The Gymnasium interaction loop (`reset` / `step` / `done`).
- Representing a policy as $\pi(a \mid s)$ and sampling actions from it.
- Estimating $V^\pi(s)$ via first-visit Monte Carlo.
- Deriving the TD(0) update from the Bellman equation.
- Comparing deterministic and slippery (stochastic) environments.
- Off-policy control with Q-learning and an ε-greedy behaviour policy.

> ➡️ Next: [`../lab2/`](../lab2/) — Connect-4 with an N-Tuple TD(λ) agent.
