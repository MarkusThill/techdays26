# 🎮 techdays26 — Reinforcement Learning Workshop with Frozen Lake & Connect-4

Learn the basics of **Reinforcement Learning (RL)** and train your own agents through two hands-on labs. No prior AI experience is required — just a laptop and basic Python.

This repository contains both:

- the **workshop materials** for the labs, and
- a reusable **`techdays26` Python package** with environments, agents, training utilities, visualizations, and supporting code.

---

## 🗺️ Workshop Journey
<video src="https://github.com/user-attachments/assets/24683e9f-f7ba-4653-ba21-3566bd409dc4" width="352" height="720"></video>

The workshop consists of **two labs** that build on each other. Work through them in order.

| Lab | Topic | Runs on | Notebooks |
|---|---|---|---|
| **[Lab 1 — ❄️ Frozen Lake](lab1/)** | Core RL concepts in a tiny grid-world | **Local only** (pygame GUI) | [`0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) |
| **[Lab 2 — 🔴🟡 Connect-4 with TD(λ)](lab2/)** | Scale up to a neural N-Tuple agent trained by self-play | Local **or** Google Colab | [`0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) → [`1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) → [`2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) |

---

## ⚠️ Important — Lab 1 must be run locally

**Lab 1 cannot run in Google Colab.** It uses [Pygame](https://www.pygame.org/) to open an interactive window in which you control the elf with the **arrow keys** and watch live value-function overlays. Colab has no desktop display surface, so the environment will fail to open.

- **Lab 1** → clone the repo and run it on your own laptop (macOS, Windows, or Linux).
- **Lab 2** → works locally and also in Colab (each Lab 2 notebook has an *Open in Colab* badge). A GPU is helpful but not required.

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

- **Python 3.11 or 3.12** (see `pyproject.toml` — `requires-python = ">=3.11"`).
- Basic Python knowledge (variables, loops, functions, NumPy arrays).
- For **Lab 1**: a local machine with a working display (macOS / Windows / Linux desktop). On Linux you may need the SDL2 system libraries that `pygame` depends on.
- For **Lab 2**: a local machine **or** a Google account for [Google Colab](https://colab.research.google.com/). A GPU accelerates training (the default training loop runs ~20,000 games in parallel), but a modern CPU is sufficient for the shorter experiments and for evaluating pre-trained checkpoints.
- Lab 2 evaluation and the "play the trained agent" widgets depend on the external [**BitBully**](https://pypi.org/project/bitbully/) α–β Connect-4 engine. It is pulled in automatically by the `lab2` extra (see below).

---

## 🚀 Quick Start

### Option A — Google Colab (Lab 2 only, no setup required)

Click the **Open in Colab** badge at the top of any Lab 2 notebook. The setup cell at the top of each notebook clones this repo and installs the package automatically. Note that Lab 2 widgets use `ipympl` / `ipywidgets`, so the first cell also runs `output.enable_custom_widget_manager()` to turn them on.

### Option B — Local install (required for Lab 1, works for Lab 2 too)

```bash
git clone https://github.com/MarkusThill/techdays26.git
cd techdays26

# Option B1 — install everything (both labs + dev tools)
pip install -e .[dev,lab1,lab2]

# Option B2 — install only what you need
pip install -e .[lab1]        # Lab 1 only
pip install -e .[lab2]        # Lab 2 only
pip install -e .[dev,lab1]    # Lab 1 + developer tooling

# Launch Jupyter (either works)
jupyter lab
# or
jupyter notebook
```

Then open [`lab1/0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) to start.

> 💡 It is recommended to install `techdays26` into a fresh virtual environment (`python -m venv venv && source venv/bin/activate`) or a conda env to keep dependencies isolated.

---

## Installation details

This project is packaged as `techdays26` with optional extras declared in `pyproject.toml`:

| Extra | Adds | When to install |
|---|---|---|
| *(none)* | `numpy` only | — |
| `lab1` | `gymnasium[toy_text]`, `matplotlib`, `pygame` | You want to run Lab 1 |
| `lab2` | `bitbully[gui]`, `torch`, `matplotlib`, `pandas`, `ipywidgets`, `ipympl` | You want to run Lab 2 |
| `dev` | `commitizen`, `pre-commit`, `ruff`, `pytest`, `nbstripout`, … | You want to contribute or bump versions |

---

## 📁 Project Structure

```
techdays26/
├── lab1/                       # Frozen Lake lab (LOCAL ONLY)
│   ├── 0_frozen_lake_problem.ipynb
│   ├── img/                    # Static images for the notebook
│   └── README.md
├── lab2/                       # Connect-4 TD(λ) lab
│   ├── 0_problem_illustration.ipynb
│   ├── 1_train_ntuple_net.ipynb
│   ├── 2_plot_metrics.ipynb
│   ├── 3_legacy_td_ntuple_c4_agent.ipynb   # utility notebook
│   ├── img/
│   └── README.md
├── src/techdays26/             # Reusable Python package
│   ├── frozen_lake/            # Enhanced FrozenLake env + helpers
│   ├── gui_*.py                # Interactive Jupyter widgets for Lab 2
│   ├── ntuple_network.py       # PyTorch N-Tuple Network
│   ├── ntuples.py              # 70 / 150 / 200-tuple sets
│   ├── torch_board.py          # Batched Connect-4 environment
│   ├── training.py             # Afterstate search helpers
│   ├── td_agent.py             # Torch TD agent (Connect4Agent interface)
│   ├── bitbully_arena.py       # Tournament arena
│   ├── logger.py               # Training logger / JSON metrics
│   ├── legacy*.py              # Legacy TD agent (reference only)
│   └── plots.py
├── tests/
├── exp_YYYYMMDD_HH-MM/         # Training runs — produced by Lab 2
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

---

## Lab 1 – ❄️ Frozen Lake (Intro to RL)

**Notebook:** [`lab1/0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb)
**Environment:** `src/techdays26/frozen_lake/frozen_lake_enhanced.py`

> ⚠️ **Must run locally.** The lab opens a Pygame window that you interact with using the keyboard. It cannot run in Colab.

Lab 1 uses the classic **FrozenLake** environment (via Gymnasium) to introduce RL concepts in an intuitive, visual way.

### What you do in this lab

- **Play Frozen Lake manually** with the arrow keys to experience:
  - states, actions, rewards,
  - episodes and termination,
  - deterministic vs. slippery (stochastic) transitions.
- Define simple **policies** $\pi(a \mid s)$ such as "favor right and down".
- Implement **Monte Carlo rollouts** to estimate state values $V^\pi(s)$.
- Implement **TD(0)** and discuss bootstrapping vs. pure Monte Carlo and the bias–variance trade-off.
- See why $V(s)$ is not enough to choose good actions in a **slippery** world and motivate $Q(s, a)$.
- Learn a good policy from scratch with **Q-learning** + ε-greedy exploration.

### Enhanced teaching environment

The enhanced `FrozenLakeEnv` adds:

- Pygame rendering with **Q-value** and **V(s)** overlays per cell, episode and info text.
- Visual **slip arrows**: red arrow = actual move after a slip; blue arrow = intended move without slip.

### ⌨️ Keyboard shortcuts (Pygame window must be focused)

| Key | Action |
|---|---|
| `←` `↓` `→` `↑` | Move the elf (manual play) |
| `ESC` | Quit the current run |
| `P` | Pause / resume |
| `+` / `-` | Increase / decrease rendering FPS |
| `0` | Unlimited FPS (fastest) |
| `1` | Reset FPS to default (4) |
| `9` | Toggle rendering on/off (speeds up training) |

See [`lab1/README.md`](lab1/README.md) for additional lab-specific notes.

---

## Lab 2 – 🔴🟡 Connect-4 with TD(λ) and N-Tuple Networks

**Main notebooks:**

- [`lab2/0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) – problem setup & encoding
- [`lab2/1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) – train a TD(λ) N-tuple agent
- [`lab2/2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) – plot training and arena metrics
- [`lab2/3_legacy_td_ntuple_c4_agent.ipynb`](lab2/3_legacy_td_ntuple_c4_agent.ipynb) – legacy agent comparison (utility)
- [`lab2/README.md`](lab2/README.md) – lab overview and instructions

> 💡 The interactive visualizations in notebook `0_problem_illustration.ipynb` use `ipympl` (activated per notebook via `%matplotlib ipympl`). Make sure you installed the `lab2` extra.

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

- **N-Tuple Network (`NTupleNetwork`)** — a sum of LUTs defined by one of the tuple sets in `ntuples.py`:
  - `NTUPLE_BITIDX_LIST` — 70 tuples,
  - `NTUPLE_BITIDX_LIST_150` — 150 tuples,
  - `NTUPLE_BITIDX_LIST_200` — **200 tuples (default)**,
  - evaluates **afterstate values** $V(s')$ and exploits horizontal mirror symmetry.
- **Parallel self-play with `BoardBatch`** — runs `B` Connect-4 games in parallel on CPU/GPU.
- **Truncated TD(λ)** — interpolates between TD(0) and Monte Carlo, bootstraps from a target network.
- **Action selection via afterstate search** — enumerates legal moves, evaluates afterstates, uses ε-greedy exploration, can optionally avoid losing moves.
- **Target network** with Polyak averaging, Adam + exponential LR decay, and optional `torch.compile` for speed.
- **Evaluation vs. BitBully** — tournament-style evaluation against BitBully opponents at multiple search depths and a random baseline; metrics saved as JSON and tables in timestamped experiment folders.

### 2.3 Analysis & Legacy Agent

- [`lab2/2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) — reads `0_metrics.json` and `0_arena_metrics.json`, plots training loss, value statistics, and arena performance over time, with optional bootstrap CIs across repeated runs.
- [`lab2/3_legacy_td_ntuple_c4_agent.ipynb`](lab2/3_legacy_td_ntuple_c4_agent.ipynb) — compares the current Torch-based TD agent with a legacy TD N-tuple implementation; useful if you have old `.txt.zip` weight files to convert.

More details are described in [`lab2/README.md`](lab2/README.md).

### Hardware recommendations for Lab 2 training

| Setup | Batch size `B` | Approximate throughput |
|---|---|---|
| **CPU laptop** | `B ≈ 1k – 5k` | Enough to reproduce the main results in minutes–hours |
| **Single consumer GPU** (e.g. RTX / T4) | `B ≈ 20k` (default) | Reaches strong play against BitBully-level opponents |
| **Google Colab (T4 GPU)** | `B ≈ 10k – 20k` | Works well for the default hyperparameters |

### Pre-trained checkpoints

Several Lab 2 notebooks (e.g. the interactive widgets in `0_problem_illustration.ipynb`) reference a **pre-trained checkpoint** at a path like:

```
exp_20260228_13-46/repeat_0/step_500_model_weights.pt
```

These files are produced by running [`1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) — every training run writes a timestamped `exp_YYYYMMDD_HH-MM/` folder with model snapshots, metrics, and arena results. Either:

1. **Run the training notebook first** (10–30 minutes on a GPU for a short run) and then point the visualization widgets at *your* new checkpoint path, **or**
2. **Use the provided pre-trained checkpoint** at the path referenced in the notebooks.

Both workflows are supported.

---

## Core Library Components

The `src/techdays26` package contains the reusable components backing the labs.

### Connect-4 engine & value network

- **`torch_board.py`** — `BoardBatch`: vectorized Connect-4 environment for batched play (legal-move masks, `play_masks`, terminal checks, rewards, in-place resets).
- **`ntuple_network.py`** — `NTupleNetwork`: PyTorch `nn.Module` implementing an N-tuple afterstate value function with mirror symmetry and checkpoint save/load.
- **`ntuples.py`** — tuple sets (`NTUPLE_BITIDX_LIST`, `…_150`, `…_200`) plus helpers such as `ntuple_summary` and `format_ntuple`.
- **`training.py`** — `best_afterstate_values(board, net, randomize, use_non_losing=False)` evaluates legal afterstates, applies greedy or random move selection, and optionally filters losing moves.

### Agents & arena

- **`td_agent.py`** — `TDConnect4AgentTorch`: wraps `NTupleNetwork` as an evaluation-ready Connect-4 agent compatible with `bitbully.agent_interface.Connect4Agent`.
- **`bitbully_arena.py`** — `BitBullyArena` + config classes for tournaments and result formatting.

### Logging & plotting

- **`logger.py`** — `TrainingLogger` for console logging, JSON metrics, arena results, and optional weight snapshots.
- **`plots.py`** — plot helpers such as `plot_adam_effective_lr`.

### Frozen Lake (Lab 1)

- **`frozen_lake/frozen_lake_enhanced.py`** — `FrozenLakeEnv`: Gymnasium environment with overlays, rendering, and slip visualizations.
- **`frozen_lake/frozen_lake_utils.py`** — keyboard and visualization helpers.

### Miscellaneous

- **`utils.py`** — helpers for logging commit SHAs, requirements, and version info (including `extract_params_from_log` for older runs).
- **`legacy.py`, `legacy_ntuple_agent.py`** — legacy TD N-tuple implementations kept for reference and comparison.

---

## Teaching GUIs

The `gui_*` modules under `src/techdays26/` are designed for **live demos** and interactive teaching. All run inside a Jupyter notebook with `%matplotlib ipympl`.

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

### N-tuple & afterstate visualization

- `gui_ntuple.py` – show tuple patterns directly on the Connect-4 board
- `gui_afterstate_equiv.py` – visualize afterstate equivalence

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
pytest                              # run all tests
pytest tests/test_board_batch.py    # run a specific file
pytest -k "mirror"                  # run tests matching a keyword
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

## 🩺 Troubleshooting

**Lab 1 — "pygame" installation fails or the window does not open.**
Make sure you have SDL2 installed on your system. On Debian/Ubuntu: `sudo apt install python3-pygame libsdl2-dev`. On macOS, `pip install pygame` usually works out of the box. The notebook must be run locally — it will not work in Colab, JupyterHub in a browser tab without a display, or a headless SSH session.

**Lab 2 — "No module named `bitbully`".**
Install the `lab2` extra: `pip install -e .[lab2]`. BitBully is published on PyPI as `bitbully[gui]`.

**Lab 2 — interactive widgets show raw text / don't update.**
Activate the widget backend explicitly at the top of the notebook: `%matplotlib ipympl`. In Colab, add `from google.colab import output; output.enable_custom_widget_manager()` (the notebooks already do this).

**Lab 2 — `torch._dynamo` recompile-limit warning or slow startup.**
The training notebook uses `torch.compile` by default. If you hit the recompilation limit on your setup, set `use_torch_compile = False` in the hyperparameters cell.

**Lab 2 — visualization cells can't find a checkpoint file.**
Either run `1_train_ntuple_net.ipynb` to produce your own `exp_YYYYMMDD_HH-MM/repeat_0/step_*_model_weights.pt`, or update the hard-coded path in the widget cell to a checkpoint you have. See the *Pre-trained checkpoints* section above.

**`pip install -e .` says Python version too old.**
`techdays26` requires Python **3.11 or later** (see `pyproject.toml`). Upgrade your interpreter or create a new virtual environment with a newer Python.

---

## Development Notes

- Code quality is enforced via **pre-commit** (`.pre-commit-config.yaml`).
- `lab2/1_train_ntuple_net.ipynb` may use `torch.compile`; on some systems this can trigger TorchDynamo recompilation warnings. You can disable it by setting `use_torch_compile = False` in the notebook.
- Lab 2 experiments write metadata and results into timestamped folders such as `exp_YYYYMMDD_HH-MM/` for reproducibility.

---

## For workshop participants

- Start with **Lab 1** (locally!) to build intuition for states, actions, rewards, policies, and value functions.
- Then move to **Lab 2** (locally or in Colab) to see how the same ideas scale to a more complex game with efficient representations and TD(λ) training.

Enjoy exploring reinforcement learning 🚀

---

## License

MIT — see [LICENSE](LICENSE).
