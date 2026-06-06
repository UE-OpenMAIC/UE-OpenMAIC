from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AUTOPLAIT_ROOT = THIS_DIR.parent
REPO_ROOT = AUTOPLAIT_ROOT.parent.parent
DEFAULT_RUNNER = THIS_DIR / "autoplait_runner.py"

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


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(",", " ").replace(";", " ").strip()
    return [part for part in text.split() if part]


def expand_path_value(value: str | None, *, default: Path, config_dir: Path, repo_root: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{AUTOPLAIT_ROOT}", str(AUTOPLAIT_ROOT))
    text = text.replace("{SHARD_DIR}", str(THIS_DIR))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
    text = text.replace("{REPO_ROOT}", str(repo_root))
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


def build_command(config_path: Path, dataset_default: str) -> tuple[list[str], Path, Path]:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    settings = parse_config(config_path)

    repo_root = expand_path_value(settings.get("repo_root"), default=REPO_ROOT, config_dir=config_dir, repo_root=REPO_ROOT)
    runner = expand_path_value(settings.get("runner"), default=DEFAULT_RUNNER, config_dir=config_dir, repo_root=repo_root)
    out_dir = expand_path_value(settings.get("out_dir"), default=config_dir / f"results_autoplait_{dataset_default}", config_dir=config_dir, repo_root=repo_root)
    data_root = expand_path_value(settings.get("data_root"), default=repo_root / "Time2State" / "data", config_dir=config_dir, repo_root=repo_root)
    autoplait_bin = expand_path_value(
        settings.get("autoplait_bin"),
        default=repo_root / "E2Usd-main" / "Baselines" / "AutoPlait" / "experiments" / "src" / "autoplait",
        config_dir=config_dir,
        repo_root=repo_root,
    )

    datasets = split_values(settings.get("datasets")) or [dataset_default]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(runner),
        "--repo-root", str(repo_root),
        "--out-dir", str(out_dir),
        "--datasets", *datasets,
        "--data-root", str(data_root),
        "--autoplait-bin", str(autoplait_bin),
    ]

    for key, flag in [
        ("max_cases", "--max-cases"),
        ("cp_margin_ratio", "--cp-margin-ratio"),
        ("segment_index_base", "--segment-index-base"),
        ("pamap2_downsample", "--pamap2-downsample"),
    ]:
        optional_arg(cmd, flag, settings.get(key))

    if parse_bool(settings.get("skip_completed", "0")):
        cmd.append("--skip-completed")
    if parse_bool(settings.get("dry_run", "0")):
        cmd.append("--dry-run")

    return cmd, runner, out_dir


def main(dataset_default: str, default_config_name: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / dataset_default / default_config_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd, runner, out_dir = build_command(args.config, dataset_default)
    print("Config :", args.config.resolve())
    print("Runner :", runner)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(AUTOPLAIT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main("mocap", "mocap_autoplait_config.txt"))
