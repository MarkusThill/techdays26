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


def extract_params_from_log(folder: str | Path) -> Path:
    """Extract training parameters from ``0_log.txt`` and write ``0_params.json``.

    Parses the structured ``--- Training Settings ---``, ``--- Environment ---``,
    and ``--- N-Tuple Info ---`` blocks from the log file.  Fields that cannot be
    found are set to ``None`` in the output JSON.

    Args:
        folder: Path to the experiment directory containing ``0_log.txt``.

    Returns:
        Path to the written ``0_params.json``.

    Raises:
        FileNotFoundError: If ``0_log.txt`` does not exist in *folder*.
    """
    folder = Path(folder)
    log_path = folder / "0_log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"No 0_log.txt found in {folder}")

    text = log_path.read_text(encoding="utf-8")

    def _find(pattern: str, convert=str):
        """Search *text* for *pattern* (MULTILINE) and return the first group, converted."""
        m = re.search(pattern, text, re.MULTILINE)
        if m is None:
            return None
        raw = m.group(1).strip()
        try:
            return convert(raw)
        except (ValueError, TypeError):
            return raw

    def _find_na(pattern: str, convert=str):
        """Like _find but returns None for 'N/A' values."""
        val = _find(pattern, str)
        if val is None or val.strip() == "N/A":
            return None
        try:
            return convert(val)
        except (ValueError, TypeError):
            return val

    # Environment block
    start_time = _find(r"^start_time:\s*(.+)")
    # "device:" appears in both Environment and Training Settings blocks;
    # extract the one inside Training Settings
    device_m = re.search(r"--- Training Settings ---\n(?:.*\n)*?device:\s*(.+)", text)
    device = device_m.group(1).strip() if device_m else _find(r"^device:\s*(.+)")

    python_version = _find(r"^python:\s*(.+)", str)
    pytorch_version = _find(r"^pytorch:\s*(.+)", str)
    commit_sha = _find(r"Git commit SHA:\s*(.+)")
    techdays26_version = _find(r"techdays26 version:\s*(.+)")
    pre_trained_model = _find(r"Base model:\s*(.+)")
    if pre_trained_model == "None":
        pre_trained_model = None

    # Training Settings block
    n_steps = _find(r"n_steps:\s*(\d+)", int)
    n_evaluate = _find(r"n_evaluate:\s*(\d+)", int)
    batch_size = _find(r"batch_size \(B\):\s*(\d+)", int)
    lr_initial = _find(r"lr_initial:\s*([0-9eE.+-]+)", float)
    lr_final = _find(r"lr_final:\s*([0-9eE.+-]+)", float)
    gamma = _find(r"gamma \(lr decay\):\s*([0-9eE.+-]+)", float)
    epsilon = _find(r"epsilon \(exploration\):\s*([0-9eE.+-]+)", float)
    use_target_net = _find(r"use_target_net:\s*(True|False)")
    if use_target_net is not None:
        use_target_net = use_target_net == "True"
    use_online_net_for_action = _find_na(r"use_online_net_for_action:\s*(.+)")
    if use_online_net_for_action is not None:
        use_online_net_for_action = use_online_net_for_action.strip() == "True"
    tau = _find_na(r"tau \(Polyak\):\s*(.+)", float)

    # Optimizer: "Adam (betas=(0.9, 0.999), eps=1e-08, weight_decay=0)"
    optimizer = _find(r"optimizer:\s*(\w+)")
    betas_m = re.search(r"betas=\(([0-9.]+),\s*([0-9.]+)\)", text)
    optimizer_betas = (
        [float(betas_m.group(1)), float(betas_m.group(2))] if betas_m else None
    )
    optimizer_eps = _find(r"optimizer:.*?eps=([0-9eE.+-]+)", float)
    optimizer_weight_decay = _find(r"weight_decay=([0-9eE.+-]+)", float)

    loss = _find(r"^loss:\s*(.+)", str)
    use_non_losing = _find(r"use_non_losing:\s*(True|False)")
    if use_non_losing is not None:
        use_non_losing = use_non_losing == "True"
    activation = _find(r"activation:\s*(.+)")

    # N-Tuple Info block
    n_tuples = _find(r"n_tuples \(M\):\s*(\d+)", int)
    tuple_length = _find(r"tuple_length \(N\):\s*(\d+)", int)
    lut_size = _find(r"LUT_size \(K=4\^N\):\s*(\d+)", int)
    total_params = _find(r"total_params:\s*(\d+)", int)
    ntuple_hash = _find(r"ntuple_hash:\s*(\S+)")
    mirror_symmetry = _find(r"mirror_symmetry:\s*(True|False)")
    if mirror_symmetry is not None:
        mirror_symmetry = mirror_symmetry == "True"

    params = {
        "start_time": start_time,
        "device": device,
        "pre_trained_model": pre_trained_model,
        "n_steps": n_steps,
        "n_evaluate": n_evaluate,
        "batch_size": batch_size,
        "lr_initial": lr_initial,
        "lr_final": lr_final,
        "gamma": gamma,
        "epsilon": epsilon,
        "use_target_net": use_target_net,
        "use_online_net_for_action": use_online_net_for_action,
        "tau": tau,
        "optimizer": optimizer,
        "optimizer_betas": optimizer_betas,
        "optimizer_eps": optimizer_eps,
        "optimizer_weight_decay": optimizer_weight_decay,
        "loss": loss,
        "use_non_losing": use_non_losing,
        "activation": activation,
        "mirror_symmetry": mirror_symmetry,
        "n_tuples": n_tuples,
        "tuple_length": tuple_length,
        "lut_size": lut_size,
        "total_params": total_params,
        "ntuple_hash": ntuple_hash,
        "commit_sha": commit_sha,
        "techdays26_version": techdays26_version,
        "python_version": python_version,
        "pytorch_version": pytorch_version,
    }

    out_path = folder / "0_params.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    return out_path
