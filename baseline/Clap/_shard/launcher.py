from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path





THIS_DIR = Path(__file__).resolve().parent
CLAP_ROOT = THIS_DIR.parent
BASELINE_ROOT = CLAP_ROOT.parent
REPO_ROOT = BASELINE_ROOT.parent

DEFAULT_RUNNER = REPO_ROOT / "ourClap" / "_shared" / "our_clap_runner.py"
DEFAULT_CLAP_REPO = REPO_ROOT / "classification-label-profile-main"

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
        if "=" not in stripped:
            raise ValueError(f"Config line {line_no} should be key=value: {raw}")
        key, value = stripped.split("=", 1)
        settings[key.strip().lower().replace("-", "_")] = value.strip()
    return settings


def expand_path_value(value: str | None, *, default: Path, config_dir: Path, repo_root: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{REPO_ROOT}", str(repo_root))
    text = text.replace("{CLAP_ROOT}", str(CLAP_ROOT))
    text = text.replace("{SHARD_DIR}", str(THIS_DIR))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
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
    config_dir = config_path.parent
    settings = parse_config(config_path)

    repo_root = expand_path_value(settings.get("repo_root"), default=REPO_ROOT, config_dir=config_dir, repo_root=REPO_ROOT)
    runner = expand_path_value(settings.get("runner"), default=DEFAULT_RUNNER, config_dir=config_dir, repo_root=repo_root)
    clap_repo = expand_path_value(settings.get("clap_repo"), default=DEFAULT_CLAP_REPO, config_dir=config_dir, repo_root=repo_root)
    out_dir = expand_path_value(
        settings.get("out_dir"),
        default=config_dir / f"results_{dataset_default}_clap_baseline_default",
        config_dir=config_dir,
        repo_root=repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = settings.get("dataset", dataset_default)
    cmd = [
        sys.executable,
        "-u",
        str(runner),
        "--repo-root", str(repo_root),
        "--out-dir", str(out_dir),
        "--datasets", dataset,
        "--clap-repo", str(clap_repo),
    ]

    for key, flag in [
        ("max_series_per_dataset", "--max-series-per-dataset"),
        ("priority_case_ids", "--priority-case-ids"),
        ("seed", "--seed"),
        ("n_jobs", "--n-jobs"),
        ("init_cps_source", "--init-cps-source"),
        ("fallback_uniform_segments", "--fallback-uniform-segments"),
        ("adf_sample_max", "--adf-sample-max"),
        ("normalize_input", "--normalize-input"),
        ("clap_window_size", "--clap-window-size"),
        ("clap_classifier", "--clap-classifier"),
        ("clap_merge_score", "--clap-merge-score"),
    ]:
        optional_arg(cmd, flag, settings.get(key))

    if parse_bool(settings.get("save_predictions", "1")):
        cmd.append("--save-predictions")

    extra_args = settings.get("extra_args", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd, runner, clap_repo, out_dir


def main(dataset_default: str, default_config_name: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CLAP_ROOT / dataset_default / default_config_name,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd, runner, clap_repo, out_dir = build_command(args.config, dataset_default)
    print("Config   :", args.config.resolve())
    print("Runner   :", runner)
    print("CLaP repo:", clap_repo)
    print("Output   :", out_dir)
    print("Command:")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in cmd))

    if args.dry_run:
        return 0

    if not runner.exists():
        raise FileNotFoundError(f"Cannot find CLaP baseline runner: {runner}")
    if not (clap_repo / "src" / "clap.py").exists():
        raise FileNotFoundError(f"Cannot find original CLaP source: {clap_repo / 'src' / 'clap.py'}")

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main("mocap", "mocap_config.txt"))
