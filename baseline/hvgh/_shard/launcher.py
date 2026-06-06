
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
HVGH_ROOT = THIS_DIR.parent
REPO_ROOT = HVGH_ROOT.parent.parent
TRUE = {"1", "true", "yes", "y", "on"}


def parse_config(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    settings = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise ValueError(f"Config line {line_no} should be key=value: {raw}")
        k, v = s.split("=", 1)
        settings[k.strip().lower().replace("-", "_")] = v.strip()
    return settings


def expand_path(v: str | None, *, default: Path, config_dir: Path, repo_root: Path) -> Path:
    if not v:
        return default.resolve()
    text = v.replace("{REPO_ROOT}", str(repo_root)).replace("{CONFIG_DIR}", str(config_dir)).replace("{SHARD_DIR}", str(THIS_DIR)).replace("{HVGH_ROOT}", str(HVGH_ROOT))
    p = Path(text)
    if not p.is_absolute():
        p = config_dir / p
    return p.resolve()


def is_true(v: object) -> bool:
    return str(v or "").strip().lower() in TRUE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun-completed", action="store_true", help="Forward to hvgh_runner.py and rerun already successful cases.")
    args = ap.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent
    s = parse_config(config_path)

    repo_root = expand_path(s.get("repo_root"), default=REPO_ROOT, config_dir=config_dir, repo_root=REPO_ROOT)
    runner = expand_path(s.get("runner"), default=THIS_DIR / "hvgh_runner.py", config_dir=config_dir, repo_root=repo_root)
    out_dir = expand_path(s.get("out_dir"), default=config_dir / "results_hvgh", config_dir=config_dir, repo_root=repo_root)
    datasets = s.get("datasets", config_dir.name).replace(",", " ").split()

    cmd = [sys.executable, "-u", str(runner), "--repo-root", str(repo_root), "--out-dir", str(out_dir), "--datasets", *datasets]

    for key, flag in [
        ("max_cases", "--max-cases"),
        ("epoch", "--epoch"),
        ("iteration", "--iteration"),
        ("win_size", "--win-size"),
        ("gamma", "--gamma"),
        ("eta", "--eta"),
        ("initial_class", "--initial-class"),
        ("max_retries", "--max-retries"),
    ]:
        if s.get(key, "").strip():
            cmd.extend([flag, s[key]])

    if is_true(s.get("skip_completed")):
        cmd.append("--skip-completed")
    if is_true(s.get("rerun_completed")) or args.rerun_completed:
        cmd.append("--rerun-completed")
    if is_true(s.get("eval_existing_only")):
        cmd.append("--eval-existing-only")

    print("Config :", config_path)
    print("Runner :", runner)
    print("Output :", out_dir)
    print("Command:")
    print(" ".join(f'\"{x}\"' if " " in x else x for x in map(str, cmd)))

    if args.dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
