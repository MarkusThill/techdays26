import json
import re
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


def convert_arena_results(folder: str | Path) -> Path:
    """Convert individual step_X_arena_result.json files to a single 0_arena_metrics.json.

    Reads all ``step_*_arena_result.json`` files in *folder*, extracts the
    aggregate rows (same fields as :func:`bitbully_arena.format_aggregate_table`),
    and writes a compact ``0_arena_metrics.json`` alongside them.

    Args:
        folder: Path to the experiment directory containing the arena result files.

    Returns:
        Path to the written ``0_arena_metrics.json``.

    Raises:
        FileNotFoundError: If no ``step_*_arena_result.json`` files are found.
    """
    folder = Path(folder)
    files = sorted(folder.glob("step_*_arena_result.json"))
    if not files:
        raise FileNotFoundError(f"No step_*_arena_result.json files found in {folder}")

    all_arena_metrics: list[dict] = []
    for p in files:
        m = re.search(r"step_(\d+)_arena_result", p.name)
        if m is None:
            continue
        step = int(m.group(1))
        with p.open() as f:
            data = json.load(f)

        rows = []
        for r in data["result"]["aggregates"]:
            score = int(r["yellow_wins"]) - int(r["red_wins"])
            games = int(r["games"])
            rows.append({
                "agent_yellow": r["agent_yellow"],
                "agent_red": r["agent_red"],
                "epsilon_yellow": float(r["epsilon_yellow"]),
                "epsilon_red": float(r["epsilon_red"]),
                "games": games,
                "yellow_wins": int(r["yellow_wins"]),
                "red_wins": int(r["red_wins"]),
                "draws": int(r["draws"]),
                "score": score,
                "avg": (score / games) if games else 0.0,
                "timeouts": int(r.get("timeouts", 0)),
                "illegal_moves": int(r.get("illegal_moves", 0)),
                "exceptions": int(r.get("exceptions", 0)),
                "total_time_s": float(r.get("total_time_s", 0.0)),
            })

        all_arena_metrics.append({
            "step": step,
            "aggregates": rows,
        })

    out_path = folder / "0_arena_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_arena_metrics, f, indent=2)

    return out_path
