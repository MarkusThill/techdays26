# 🎮 techdays26 — A Beginner-Friendly Workshop on Reinforcement Learning

Learn the basics of **Reinforcement Learning (RL)** and train your own RL agents through two hands-on labs. No prior AI experience required — just a laptop and basic Python.

> 🚧 *Heavily under construction.*

---

## 🗺️ Workshop Journey

The workshop consists of **two labs** that build on each other. Work through them in order.

| Lab | Topic | Notebooks |
|---|---|---|
| **[Lab 1 — ❄️ Frozen Lake](lab1/)** | Core RL concepts in a tiny grid-world | [`0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) |
| **[Lab 2 — 🔴🟡 Connect-4 with TD(λ)](lab2/)** | Scale up to a neural N-Tuple agent trained by self-play | [`0_problem_illustration.ipynb`](lab2/0_problem_illustration.ipynb) → [`1_train_ntuple_net.ipynb`](lab2/1_train_ntuple_net.ipynb) → [`2_plot_metrics.ipynb`](lab2/2_plot_metrics.ipynb) |

### What you'll learn

- The RL interaction loop — states, actions, rewards, policies.
- Value functions $V^\pi(s)$ and $Q^\pi(s, a)$ — what they are and when to use each.
- **Monte Carlo rollouts** and **TD(0)**, derived from the Bellman equation.
- **Q-learning** with ε-greedy exploration on a stochastic (slippery) environment.
- **Truncated TD(λ)** as a bias–variance trade-off between TD(0) and Monte Carlo.
- **N-Tuple Networks** as fast, differentiable value functions.
- Parallel self-play, target networks, Polyak averaging, and evaluation against an α–β baseline.

### Prerequisites

- Basic Python (variables, loops, functions, NumPy arrays).
- A laptop with Python 3.10+ installed, **or** a Google account to run the notebooks on [Google Colab](https://colab.research.google.com/) (each notebook has an "Open in Colab" badge).

---

## 🚀 Quick Start

### Option A — Google Colab (no setup)

Click the **Open in Colab** badge at the top of any notebook. The Colab-setup cell in each notebook clones the repo and installs the package automatically.

### Option B — Local install

```bash
git clone https://github.com/MarkusThill/techdays26.git
cd techdays26
pip install -e .[dev,lab1]
jupyter lab
```

Then open [`lab1/0_frozen_lake_problem.ipynb`](lab1/0_frozen_lake_problem.ipynb) to start.

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

### Commit Message Format

All commit messages must follow the **Conventional Commits** format:

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Common types:

| Type       | Description                                      |
|------------|--------------------------------------------------|
| `feat`     | A new feature                                    |
| `fix`      | A bug fix                                        |
| `chore`    | Maintenance tasks (deps, CI, configs, etc.)      |
| `docs`     | Documentation changes                            |
| `refactor` | Code restructuring without changing behavior     |
| `test`     | Adding or updating tests                         |
| `perf`     | Performance improvements                         |
| `ci`       | CI/CD changes                                    |

Examples:

```bash
git commit -m "feat: add depth-limited agent wrapper"
git commit -m "fix: correct win rate calculation in arena"
git commit -m "chore: update dependencies"
```

### Version Bumping

Commitizen manages the project version. To bump the version based on commit history:

```bash
# Dry run (see what would happen):
cz bump --dry-run

# Actually bump (updates pyproject.toml, __init__.py, creates tag, updates CHANGELOG.md):
cz bump

# Push the commit and tag to remote:
git push && git push --tags
```

The version is tracked in:
- `pyproject.toml` (`version = "..."`)
- `src/techdays26/__init__.py` (`__version__ = "..."`)
