from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
OURCLAP_ROOT = THIS_DIR.parent
REPO_ROOT = OURCLAP_ROOT.parent
OUR_ROOT = REPO_ROOT / "our"
DEFAULT_RUNNER = THIS_DIR / "our_clap_runner.py"

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def parse_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Expected boolean value, got: {value!r}")


def parse_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    settings: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            raise ValueError(f"Config line {line_no}: unsupported section {stripped!r}")
        if "=" not in stripped:
            raise ValueError(f"Config line {line_no} should be key=value: {raw}")
        key, value = stripped.split("=", 1)
        settings[key.strip().lower().replace("-", "_")] = value.strip()
    return settings


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(",", " ").replace(";", " ").strip()
    return [part for part in text.split() if part]


def expand_path_value(value: str | None, *, default: Path, config_dir: Path, script_dir: Path, repo_root: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{THIS_DIR}", str(script_dir))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
    text = text.replace("{OURCLAP_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{OUR_ROOT}", str(repo_root / "our"))
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


def build_command(config_path: Path, dataset_default: str) -> tuple[list[str], Path, Path, Path, Path]:
    config_path = config_path.resolve()
    script_dir = config_path.parent
    settings = parse_config(config_path)

    repo_root = expand_path_value(settings.get("repo_root"), default=REPO_ROOT, config_dir=script_dir, script_dir=script_dir, repo_root=REPO_ROOT)
    runner = expand_path_value(settings.get("runner"), default=DEFAULT_RUNNER, config_dir=script_dir, script_dir=script_dir, repo_root=repo_root)
    clap_repo = expand_path_value(settings.get("clap_repo"), default=repo_root / "classification-label-profile-main", config_dir=script_dir, script_dir=script_dir, repo_root=repo_root)
    loader_runner = expand_path_value(settings.get("loader_runner"), default=repo_root / "our" / "_shared" / "our_multit2s_runner.py", config_dir=script_dir, script_dir=script_dir, repo_root=repo_root)
    out_dir = expand_path_value(settings.get("out_dir"), default=script_dir / f"results_{dataset_default}_clap_baseline_default", config_dir=script_dir, script_dir=script_dir, repo_root=repo_root)
    public_data_root = expand_path_value(settings.get("public_data_root"), default=repo_root / "Time2State" / "Baselines" / "public_ts_datasets", config_dir=script_dir, script_dir=script_dir, repo_root=repo_root)

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
        "--loader-runner",
        str(loader_runner),
        "--public-data-root",
        str(public_data_root),
    ]

    for key, flag in [
        ("max_series_per_dataset", "--max-series-per-dataset"),
        ("max_synthetic", "--max-synthetic"),
        ("priority_case_ids", "--priority-case-ids"),
        ("only_case_ids", "--only-case-ids"),
        ("case_ids", "--case-ids"),
        ("case_indexes", "--case-indexes"),
        ("rounds", "--rounds"),
        ("limit_rows", "--limit-rows"),
        ("seed", "--seed"),
        ("n_jobs", "--n-jobs"),
        ("init_cps_source", "--init-cps-source"),
        ("fallback_uniform_segments", "--fallback-uniform-segments"),
        ("adf_sample_max", "--adf-sample-max"),
        ("normalize_input", "--normalize-input"),
        ("clap_window_size", "--clap-window-size"),
        ("clap_classifier", "--clap-classifier"),
        ("clap_merge_score", "--clap-merge-score"),
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

    return cmd, runner, out_dir, clap_repo, loader_runner


def main(dataset_default: str, default_config_name: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=OURCLAP_ROOT / dataset_default / default_config_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd, runner, out_dir, clap_repo, loader_runner = build_command(args.config, dataset_default)
    print("Config :", args.config.resolve())
    print("Runner :", runner)
    print("CLaP   :", clap_repo)
    print("Loader :", loader_runner)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(out_dir.parent))
    return int(completed.returncode)
