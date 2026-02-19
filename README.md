**Nothing to see here yet**

(Heavily under Construction 🚧👷‍♀️👷‍♂️)

# techdays26
Beginner-friendly workshop on Reinforcement Learning (RL). Learn the basics and train RL agents through practical examples. No AI experience needed—just a laptop and optional Python skills!

## Development Setup

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
```

The version is tracked in:
- `pyproject.toml` (`version = "..."`)
- `src/techdays26/__init__.py` (`__version__ = "..."`)

