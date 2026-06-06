from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
T2S_ROOT = THIS_DIR.parent
REPO_ROOT = T2S_ROOT.parent.parent
DEFAULT_RUNNER = THIS_DIR / "t2s_runner.py"


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
    text = text.replace("{T2S_ROOT}", str(T2S_ROOT))
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
    out_dir = expand_path_value(
        settings.get("out_dir"),
        default=config_dir / f"results_{dataset_default}_t2s_single_train_all",
        config_dir=config_dir,
        repo_root=repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(runner),
        "--repo-root", str(repo_root),
        "--out-dir", str(out_dir),
        "--dataset", settings.get("dataset", dataset_default),
    ]

    for key, flag in [
        ("feature_mode", "--feature-mode"),
        ("train_subject", "--train-subject"),
        ("subjects", "--subjects"),
        ("win_size", "--win-size"),
        ("step", "--step"),
        ("m", "--m"),
        ("n", "--n"),
        ("out_channels", "--out-channels"),
        ("nb_steps", "--nb-steps"),
        ("kernel_size", "--kernel-size"),
        ("win_type", "--win-type"),
        ("seed", "--seed"),
        ("device", "--device"),
        ("gpu", "--gpu"),
    ]:
        optional_arg(cmd, flag, settings.get(key))

    if parse_bool(settings.get("remove_zero", "0")):
        cmd.append("--remove-zero")

    return cmd, runner, out_dir


def main(dataset_default: str, default_config_name: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=T2S_ROOT / dataset_default / default_config_name,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd, runner, out_dir = build_command(args.config, dataset_default)
    print("Config :", args.config.resolve())
    print("Runner :", runner)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)
