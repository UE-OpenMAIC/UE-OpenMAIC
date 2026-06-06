from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from io import StringIO
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
OURCLAP_ROOT = THIS_DIR.parent
REPO_ROOT = OURCLAP_ROOT.parent
DEFAULT_RUNNER = THIS_DIR / "our_clap_pid_meta_runner.py"

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def parse_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Expected boolean value, got: {value!r}")


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(",", " ").replace(";", " ").strip()
    return [part for part in text.split() if part]


def parse_config(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    settings: dict[str, str] = {}
    branch_lines: list[str] = []
    in_branches = False

    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.lower() in {"[clap_branches]", "[branches]"}:
            in_branches = True
            continue

        if in_branches:
            branch_lines.append(raw)
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            raise ValueError(f"Config line {line_no}: unsupported section {stripped!r}")

        if "=" not in stripped:
            raise ValueError(f"Config line {line_no} should be key=value or [clap_branches]: {raw}")

        key, value = stripped.split("=", 1)
        settings[key.strip().lower().replace("-", "_")] = value.strip()

    if branch_lines:
        reader = csv.DictReader(StringIO("\n".join(branch_lines)))
        fieldnames = set(reader.fieldnames or [])
        missing = {"branch_name"} - fieldnames
        if missing:
            raise ValueError(f"CLaP branch table is missing required column(s): {sorted(missing)}")
        for row_no, row in enumerate(reader, start=2):
            if str(row.get("branch_name", "")).strip().lower() == "branch_name":
                raise ValueError(f"Repeated branch-table header detected at row {row_no}.")

    return settings, branch_lines


def expand_path_value(
    value: str | None,
    *,
    default: Path,
    config_dir: Path,
    script_dir: Path,
    repo_root: Path,
) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{THIS_DIR}", str(script_dir))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
    text = text.replace("{OURCLAP_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{OUR_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{REPO_ROOT}", str(repo_root))
    text = text.replace("{CLAP_ROOT}", str(repo_root / "classification-label-profile-main"))
    path = Path(text)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def optional_arg(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([flag, text])


def build_command(config_path: Path, dataset_default: str) -> tuple[list[str], Path, Path, Path]:
    config_path = config_path.resolve()
    script_dir = config_path.parent
    settings, branch_lines = parse_config(config_path)

    repo_root = expand_path_value(
        settings.get("repo_root"),
        default=REPO_ROOT,
        config_dir=script_dir,
        script_dir=script_dir,
        repo_root=REPO_ROOT,
    )
    runner = expand_path_value(
        settings.get("runner"),
        default=DEFAULT_RUNNER,
        config_dir=script_dir,
        script_dir=script_dir,
        repo_root=repo_root,
    )
    out_dir = expand_path_value(
        settings.get("out_dir"),
        default=script_dir / f"results_{dataset_default}_clap_pid_meta",
        config_dir=script_dir,
        script_dir=script_dir,
        repo_root=repo_root,
    )
    clap_repo = expand_path_value(
        settings.get("clap_repo"),
        default=repo_root / "classification-label-profile-main",
        config_dir=script_dir,
        script_dir=script_dir,
        repo_root=repo_root,
    )
    public_data_root = expand_path_value(
        settings.get("public_data_root"),
        default=repo_root / "Time2State" / "Baselines" / "public_ts_datasets",
        config_dir=script_dir,
        script_dir=script_dir,
        repo_root=repo_root,
    )

    datasets = split_values(settings.get("datasets")) or [dataset_default]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(runner),
        "--repo-root",
        str(repo_root),
        "--out-dir",
        str(out_dir),
        "--datasets",
        *datasets,
        "--clap-repo",
        str(clap_repo),
        "--public-data-root",
        str(public_data_root),
    ]

    if branch_lines:
        generated = out_dir / f"_generated_{dataset_default}_clap_branches.txt"
        generated.write_text("\n".join(branch_lines) + "\n", encoding="utf-8")
        cmd.extend(["--clap-branch-config-txt", str(generated)])

    for key, flag in [
        ("loader_runner", "--loader-runner"),
        ("max_series_per_dataset", "--max-series-per-dataset"),
        ("max_synthetic", "--max-synthetic"),
        ("only_case_ids", "--only-case-ids"),
        ("priority_case_ids", "--priority-case-ids"),
        ("case_indexes", "--case-indexes"),
        ("rounds", "--rounds"),
        ("limit_rows", "--limit-rows"),
        ("seed", "--seed"),
        ("n_jobs", "--n-jobs"),
        ("fallback_uniform_segments", "--fallback-uniform-segments"),
        ("adf_sample_max", "--adf-sample-max"),
        ("select_top_k_branches", "--select-top-k-branches"),
        ("branch_select_metric", "--branch-select-metric"),
        ("meta_vote_weight_mode", "--meta-vote-weight-mode"),
        ("meta_min_len", "--meta-min-len"),
        ("state_cluster_smooth", "--state-cluster-smooth"),
        ("peer_health_weight", "--peer-health-weight"),
        ("peer_consensus_weight", "--peer-consensus-weight"),
        ("pid_kp", "--pid-kp"),
        ("pid_ki", "--pid-ki"),
        ("pid_kd", "--pid-kd"),
        ("pid_softmax_tau", "--pid-softmax-tau"),
        ("pamap2_feature_mode", "--pamap2-feature-mode"),
        ("public_max_rows", "--public-max-rows"),
    ]:
        optional_arg(cmd, flag, settings.get(key))

    if parse_bool(settings.get("skip_completed", "0")):
        cmd.append("--skip-completed")
    if parse_bool(settings.get("save_predictions", "1")):
        cmd.append("--save-predictions")
    if parse_bool(settings.get("pamap2_remove_zero", "0")):
        cmd.append("--pamap2-remove-zero")

    return cmd, runner, out_dir, clap_repo


def main(dataset_default: str, default_config_name: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / dataset_default / default_config_name,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd, runner, out_dir, clap_repo = build_command(args.config, dataset_default)
    print("Config :", args.config.resolve())
    print("Runner :", runner)
    print("CLaP   :", clap_repo)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(out_dir.parent))
    return int(completed.returncode)
