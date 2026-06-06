from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
OURCLAP_ROOT = THIS_DIR.parent
BASELINE_ROOT = OURCLAP_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parent
DEFAULT_CONFIG = THIS_DIR / "pamap2_zero_all_full_clasp_suss_config.txt"
DEFAULT_RUNNER = OURCLAP_ROOT / "_shared" / "our_clap_pamap2_zero_smoke_runner.py"
DEFAULT_CLAP_REPO = REPO_ROOT / "classification-label-profile-main"

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
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
        if "=" not in stripped:
            raise ValueError(f"Config line {line_no} should be key=value: {raw}")
        key, value = stripped.split("=", 1)
        settings[key.strip().lower().replace("-", "_")] = value.strip()
    return settings


def expand_path_value(value: str | None, *, default: Path, config_dir: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{REPO_ROOT}", str(REPO_ROOT))
    text = text.replace("{BASELINE_ROOT}", str(BASELINE_ROOT))
    text = text.replace("{OURCLAP_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{SHARED_DIR}", str(OURCLAP_ROOT / "_shared"))
    text = text.replace("{SHARD_DIR}", str(OURCLAP_ROOT / "_shard"))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
    text = text.replace("{CLAP_REPO}", str(DEFAULT_CLAP_REPO))
    p = Path(text)
    if not p.is_absolute():
        p = config_dir / p
    return p.resolve()


def optional_arg(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        cmd.extend([flag, text])


def bool_arg(cmd: list[str], flag: str, value: str | None, default: bool = False) -> None:
    if parse_bool(value, default=default):
        cmd.append(flag)


def build_command(config_path: Path) -> tuple[list[str], Path, Path]:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    s = parse_config(config_path)

    repo_root = expand_path_value(s.get("repo_root"), default=REPO_ROOT, config_dir=config_dir)
    runner = expand_path_value(
        s.get("pamap2_runner") or s.get("runner_pamap2") or s.get("runner"),
        default=DEFAULT_RUNNER,
        config_dir=config_dir,
    )
    clap_repo = expand_path_value(s.get("clap_repo"), default=DEFAULT_CLAP_REPO, config_dir=config_dir)
    out_dir = expand_path_value(
        s.get("out_dir"),
        default=config_dir / "results_pamap2_zero_clap_all_full_clasp_suss",
        config_dir=config_dir,
    )
    public_data_root = expand_path_value(
        s.get("public_data_root"),
        default=repo_root / "Time2State" / "Baselines" / "public_ts_datasets",
        config_dir=config_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(runner),
        "--repo-root", str(repo_root),
        "--out-dir", str(out_dir),
        "--datasets", s.get("dataset", "pamap2_zero"),
        "--clap-repo", str(clap_repo),
        "--public-data-root", str(public_data_root),
    ]

    for key, flag in [
        ("max_series_per_dataset", "--max-series-per-dataset"),
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
        ("pamap2_subjects", "--pamap2-subjects"),
    ]:
        optional_arg(cmd, flag, s.get(key))

    bool_arg(cmd, "--save-predictions", s.get("save_predictions"), default=True)
    bool_arg(cmd, "--skip-completed", s.get("skip_completed"), default=False)

    if parse_bool(s.get("pamap2_keep_zero"), default=False):
        cmd.append("--pamap2-keep-zero")
    else:
        cmd.append("--pamap2-remove-zero")

    if parse_bool(s.get("pamap2_no_loader_normalize"), default=False):
        cmd.append("--pamap2-no-loader-normalize")
    else:
        cmd.append("--pamap2-loader-normalize")

    return cmd, runner, out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Launch PAMAP2_zero ALL-subject full-length clasp+suss run without touching the formal config.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cmd, runner, out_dir = build_command(args.config)
    print("Config :", args.config.resolve())
    print("Runner :", runner)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in cmd))
    if args.dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
