import subprocess
import sys
from pathlib import Path


def get_commit_hash(repo_path: str | Path) -> str:
    repo_path = Path(repo_path)
    return subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def get_requirements_string() -> str:
    return subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )


# Usage:
# reqs = get_requirements_string()
# commit = get_commit_hash("/content/techdays25")
# print(commit)
# print( reqs)
