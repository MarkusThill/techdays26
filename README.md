# 🎮 techdays26 — Reinforcement Learning Workshop with Frozen Lake & Connect-4

Learn the basics of **Reinforcement Learning (RL)** and train your own agents through two hands-on labs. No prior AI experience is required — just a laptop and basic Python.

This repository contains both:

- the **workshop materials** for the labs,
- and a reusable **`techdays26` Python package** with environments, agents, training utilities, visualizations, and supporting code.

---

## 🗺️ Workshop Journey

The workshop consists of **two labs** that build on each other. Work through them in order.

| Lab | Topic | Notebooks |
|---|---|---|
| **[Lab 1 — ❄️ Frozen Lake](lab1/)** | Core RL concepts in a tiny grid-world | [`0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) |
| **[Lab 2 — 🔴🟡 Connect-4 with TD(λ)](lab2/)** | Scale up to a neural N-Tuple agent trained by self-play | [`0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) → [`1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) → [`2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) |

---

## What you'll learn

- The RL interaction loop — states, actions, rewards, policies.
- Value functions $V^\pi(s)$ and $Q^\pi(s, a)$ — what they are and when to use each.
- **Monte Carlo rollouts** and **TD(0)**, derived from the Bellman equation.
- **Q-learning** with ε-greedy exploration on a stochastic (slippery) environment.
- **Truncated TD(λ)** as a bias–variance trade-off between TD(0) and Monte Carlo.
- **N-Tuple Networks** as fast, differentiable value functions.
- Parallel self-play, target networks, Polyak averaging, and evaluation against an α–β baseline.

---

## Prerequisites

- Basic Python knowledge (variables, loops, functions, NumPy arrays).
- A laptop with Python 3.10+ installed, **or** a Google account to run the notebooks on [Google Colab](https://colab.research.google.com/) (each notebook has an **Open in Colab** badge).

Some **lab2** experiments benefit from a GPU and may optionally use the external **BitBully** engine for evaluation. See the lab2 notebook for details.

---

## 🚀 Quick Start

### Option A — Google Colab (no setup)

Click the **Open in Colab** badge at the top of any notebook. The setup cell in each notebook clones the repo and installs the package automatically.

### Option B — Local install

```bash
git clone https://github.com/MarkusThill/techdays26.git
cd techdays26
pip install -e .[dev,lab1]
jupyter lab
```

Then open [`lab1/0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) to start.

---

## Installation

This project is packaged as `techdays26` with optional extras for development and the labs.

From the repo root:

```bash
# standard install
pip install -e .

# with dev tools (formatters, pytest, etc.)
pip install -e .[dev]

# with lab1 dependencies (gymnasium, pygame, etc.)
pip install -e .[lab1]
```

---

## Lab 1 – ❄️ Frozen Lake (Intro to RL)

**Notebook:** [`lab1/0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb)  
**Environment:** `src/techdays26/frozen_lake/frozen_lake_enhanced.py`

Lab 1 uses the classic **FrozenLake** environment (via Gymnasium) to introduce RL concepts in an intuitive, visual way.

### What you do in this lab

- **Play Frozen Lake manually** with arrow keys to experience:
  - states, actions, rewards,
  - episodes and termination,
  - deterministic vs. slippery (stochastic) transitions.
- Define simple **policies** `π(a|s)` such as “favor right and down”.
- Implement **Monte Carlo rollouts** to estimate state values $V^\pi(s)$.
- Implement **TD(0)** and discuss:
  - bootstrapping vs. pure Monte Carlo,
  - the bias–variance trade-off.
- See why **V(s)** is not enough to choose good actions in a **slippery** world and motivate **Q(s, a)**.

### Enhanced teaching environment

The enhanced `FrozenLakeEnv` adds:

- Pygame rendering with:
  - **Q-value overlays** per cell,
  - **V(s)** overlays,
  - episode and info text,
  - keyboard shortcuts for FPS and rendering controls.
- Visual **slip arrows**:
  - red arrow = actual move taken due to slip,
  - blue arrow = intended move without slip.

See [`lab1/README.md`](lab1/README.md) for lab-specific instructions.

---

## Lab 2 – 🔴🟡 Connect-4 with TD(λ) and N-Tuple Networks

**Main notebooks:**

- [`lab2/0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) – problem setup & encoding
- [`lab2/1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) – train a TD(λ) N-tuple agent
- [`lab2/2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) – plot training and arena metrics
- [`lab2/3_legacy_td_ntuple_c4_agent.ipynb`](lab2/3_legacy_td_ntuple_c4_agent.ipynb) – legacy agent comparison
- [`lab2/README.md`](lab2/README.md) – lab overview and instructions

### 2.1 Problem Illustration & Bitboard Encoding

[`lab2/0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) explains:

- how the **Connect-4 board** is encoded as bitboards,
- why the representation uses **9 bits per column** (6 playable + 3 sentinel bits),
- the meaning of `all_tokens` and `active_tokens`,
- XOR invariants for player swapping,
- and how horizontal, vertical, and diagonal structures emerge from bit strides.

It also motivates what constitutes an RL **state** in an MDP: all information the agent needs to act and be evaluated, reduced here to a compact bitboard representation.

### 2.2 Training TD(λ) with an N-Tuple Network

[`lab2/1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) implements the full training pipeline:

- **N-Tuple Network (`NTupleNetwork`)**
  - a sum of LUTs defined by `NTUPLE_BITIDX_LIST_200`,
  - evaluates **afterstate values** $V(s')$,
  - exploits horizontal mirror symmetry.

- **Parallel self-play with `BoardBatch`**
  - runs `B` Connect-4 games in parallel on CPU/GPU,
  - supports legal move masks, move application, terminal checks, rewards, and resets.

- **TD(λ) with truncated λ-returns**
  - interpolates between TD(0) and Monte Carlo,
  - uses a ring buffer and λ-weighted return construction,
  - bootstraps from a target network.

- **Action selection via afterstate search**
  - enumerates legal moves,
  - evaluates afterstates,
  - uses ε-greedy exploration,
  - can optionally avoid losing moves.

- **Target network and learning-rate scheduling**
  - Adam with exponential LR decay,
  - Polyak averaging / EMA updates,
  - optional `torch.compile` for speed.

- **Evaluation vs. BitBully**
  - tournament-style evaluation against BitBully opponents and a random baseline,
  - metrics saved as JSON and tables in experiment folders.

### 2.3 Analysis & Legacy Agent

- [`lab2/2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb)
  - reads `0_metrics.json` and `0_arena_metrics.json`,
  - plots training loss, value statistics, and arena performance over time.

- [`lab2/3_legacy_td_ntuple_c4_agent.ipynb`](lab2/3_legacy_td_ntuple_c4_agent.ipynb)
  - compares the current Torch-based TD agent with a legacy TD N-tuple implementation.

More details are described in [`lab2/README.md`](lab2/README.md).

---

## Core Library Components

The `src/techdays26` package contains the reusable components backing the labs.

### Connect-4 engine & value network

- **`src/techdays26/torch_board.py`**
  - `BoardBatch`: vectorized Connect-4 environment for batched play.
  - Supports legal move masks, `play_masks`, terminal checks, rewards, and in-place resets.

- **`src/techdays26/ntuple_network.py`**
  - `NTupleNetwork`: PyTorch `nn.Module` implementing an N-tuple afterstate value function.
  - Supports mirror symmetry and checkpoint save/load.

- **`src/techdays26/ntuples.py`**
  - `NTUPLE_BITIDX_LIST_200`: main 200-tuple set.
  - Helper functions such as `ntuple_summary` and `format_ntuple`.

- **`src/techdays26/training.py`**
  - `best_afterstate_values(board, net, randomize, use_non_losing=False)`
  - Evaluates legal afterstates, applies greedy or random move selection, and optionally filters losing moves.

### Agents & arena

- **`src/techdays26/td_agent.py`**
  - `TDConnect4AgentTorch`: wraps `NTupleNetwork` as an evaluation-ready Connect-4 agent.

- **`src/techdays26/bitbully_arena.py`**
  - Arena and tournament infrastructure for evaluation against BitBully.
  - Includes config classes and result formatting utilities.

### Logging & plotting

- **`src/techdays26/logger.py`**
  - `TrainingLogger` for console logging, JSON metrics, arena results, and optional weight snapshots.

- **`src/techdays26/plots.py`**
  - Plot helpers such as `plot_adam_effective_lr`.

### Frozen Lake (Lab 1)

- **`src/techdays26/frozen_lake/frozen_lake_enhanced.py`**
  - `FrozenLakeEnv`: Gymnasium environment with overlays, rendering, and slip visualizations.

- **`src/techdays26/frozen_lake/frozen_lake_utils.py`**
  - Keyboard and visualization helpers.

### Miscellaneous

- **`src/techdays26/utils.py`**
  - Utility helpers for logging commit SHAs, requirements, and version info.

- **`src/techdays26/legacy.py`, `src/techdays26/legacy_ntuple_agent.py`**
  - Legacy TD N-tuple implementations kept for reference and comparison.

---

## Teaching GUIs

The `gui_*` modules under `src/techdays26/` are designed for **live demos** and interactive teaching.

### Bitboards & batched environments

- `gui_bitboard.py` – visualize bitboard layout and operations
- `gui_batch.py` – inspect multiple boards in a batch

### TD learning & returns

- `gui_td_step.py`, `gui_td_training_step.py` – step-by-step TD updates
- `gui_nstep.py` – n-step returns
- `gui_lambda.py`, `gui_lambda_decay.py` – λ-returns and decay

### Exploration & credit assignment

- `gui_epsilon.py` – ε-greedy exploration visualization
- `gui_credit_assignment.py` – illustrate delayed credit assignment

### Rewards & payoff

- `gui_reward.py`, `gui_payoff.py` – reward design and payoff structure

### N-tuple visualization

- `gui_ntuple.py` – show tuple patterns directly on the Connect-4 board

These GUIs are not required for the labs, but they are useful for workshop delivery and live explanation.

---

## Tests

Tests live under `tests/`:

- `tests/test_board_batch.py` – validates `BoardBatch`
- `tests/test_ntuples.py` – checks N-tuple definitions
- `tests/test_td_encoding.py` – tests TD encoding helpers
- `tests/test_training.py` – tests training utilities such as `best_afterstate_values`

Run them with:

```bash
pytest
```

from the repo root after installing with `.[dev]`.

---

## 🛠️ Development Setup

### Install

```bash
pip install -e .[dev]
```

### Pre-commit Hooks

This project uses [commitizen](https://commitizen-tools.github.io/commitizen/) to enforce [Conventional Commits](https://www.conventionalcommits.org/). A pre-commit hook validates every commit message automatically.

Install the hooks after cloning:

```bash
pre-commit install --hook-type commit-msg
pre-commit install
```

The pre-commit hooks will:

- **ruff-format** — auto-format Python files on commit
- **nbstripout** — strip notebook outputs before committing
- **commitizen** — validate commit messages follow Conventional Commits

---

## Commit Message Format

All commit messages must follow the **Conventional Commits** format:

```text
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Common types:

| Type | Description |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `chore` | Maintenance tasks (deps, CI, configs, etc.) |
| `docs` | Documentation changes |
| `refactor` | Code restructuring without changing behavior |
| `test` | Adding or updating tests |
| `perf` | Performance improvements |
| `ci` | CI/CD changes |

Examples:

```bash
git commit -m "feat: add depth-limited agent wrapper"
git commit -m "fix: correct win rate calculation in arena"
git commit -m "chore: update dependencies"
```

---

## Version Bumping

Commitizen manages the project version. To bump the version based on commit history:

```bash
# Dry run
cz bump --dry-run

# Actually bump
cz bump

# Push commit and tag
git push && git push --tags
```

The version is tracked in:

- `pyproject.toml` (`version = "..."`)
- `src/techdays26/__init__.py` (`__version__ = "..."`)

---

## Development Notes

- Code quality is enforced via **pre-commit** (`.pre-commit-config.yaml`).
- `lab2/1_train_ntuple_net.ipynb` may use `torch.compile`; on some systems this can trigger TorchDynamo recompilation warnings. You can disable it by setting `use_torch_compile = False` in the notebook.
- Lab2 experiments write metadata and results into timestamped folders such as `exp_YYYYMMDD_HH-MM/` for reproducibility.

---

## For workshop participants

- Start with **Lab 1** to build intuition for states, actions, rewards, policies, and value functions.
- Then move to **Lab 2** to see how the same ideas scale to a more complex game with efficient representations and TD(λ) training.

Enjoy exploring reinforcement learning 🚀
