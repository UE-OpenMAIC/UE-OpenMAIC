                       
r"""
PID ablation runner for UCR-SEG.

This file is safe for Windows paths because this module docstring is raw.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_BASE_CONFIG = Path(r"D:\code\teacherT2S\our\ucrseg\ucrseg_config.txt")
DEFAULT_ENTRY = Path(r"D:\code\teacherT2S\our\ucrseg\run_our_ucrseg.py")
DEFAULT_PYTHON = Path(r"D:\software\anaconda\envs\multit2s_cuda\python.exe")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    updates: dict[str, str]


CORE_VARIANTS = [
    AblationVariant("full_pid", "Original full method: PID branch selection + PID meta voting.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.45",
        "pid_ki": "0.35",
        "pid_kd": "0.20",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("no_pid_peer", "Remove PID completely: PEER selection + PEER reliability voting.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PEER",
        "meta_vote_weight_mode": "branch_reliability",
        "pid_kp": "0.45",
        "pid_ki": "0.35",
        "pid_kd": "0.20",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_select_uniform", "Keep PID branch selection, remove PID voting weights.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "uniform",
        "pid_kp": "0.45",
        "pid_ki": "0.35",
        "pid_kd": "0.20",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("peer_select_pid_weight", "Remove PID from branch selection, keep PID meta voting weights.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PEER",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.45",
        "pid_ki": "0.35",
        "pid_kd": "0.20",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("all_branches_uniform", "No top-k selection and no reliability weights; all candidate branches vote uniformly.", {
        "select_top_k_branches": "0",
        "branch_select_metric": "PEER",
        "meta_vote_weight_mode": "uniform",
        "pid_kp": "0.45",
        "pid_ki": "0.35",
        "pid_kd": "0.20",
        "pid_softmax_tau": "0.15",
    }),
]


COMPONENT_VARIANTS = [
    AblationVariant("pid_p_only", "PID component ablation: P term only.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "1.0",
        "pid_ki": "0.0",
        "pid_kd": "0.0",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_i_only", "PID component ablation: I term only.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.0",
        "pid_ki": "1.0",
        "pid_kd": "0.0",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_d_only", "PID component ablation: D term only.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.0",
        "pid_ki": "0.0",
        "pid_kd": "1.0",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_no_p", "PID component ablation: remove P, keep I:D ratio normalized.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.0",
        "pid_ki": "0.6363636364",
        "pid_kd": "0.3636363636",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_no_i", "PID component ablation: remove I, keep P:D ratio normalized.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.6923076923",
        "pid_ki": "0.0",
        "pid_kd": "0.3076923077",
        "pid_softmax_tau": "0.15",
    }),
    AblationVariant("pid_no_d", "PID component ablation: remove D, keep P:I ratio normalized.", {
        "select_top_k_branches": "16",
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": "0.5625",
        "pid_ki": "0.4375",
        "pid_kd": "0.0",
        "pid_softmax_tau": "0.15",
    }),
]


def choose_variants(plan: str) -> list[AblationVariant]:
    plan = plan.strip().lower()
    if plan == "core":
        return CORE_VARIANTS
    if plan == "components":
        return COMPONENT_VARIANTS
    if plan == "all":
        return CORE_VARIANTS + COMPONENT_VARIANTS

    names = {v.name: v for v in CORE_VARIANTS + COMPONENT_VARIANTS}
    wanted = [x.strip() for x in plan.replace(";", ",").split(",") if x.strip()]
    if not wanted:
        raise ValueError("Empty plan.")
    missing = [x for x in wanted if x not in names]
    if missing:
        raise ValueError(f"Unknown variant(s): {missing}. Available: {sorted(names)}")
    return [names[x] for x in wanted]


def split_config_sections(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "[branches]":
            return "\n".join(lines[:idx]).rstrip() + "\n", "\n".join(lines[idx:]).rstrip() + "\n"
    raise ValueError("Base config does not contain a standalone [branches] line.")


def update_preamble(preamble: str, updates: dict[str, str]) -> str:
    used: set[str] = set()
    out: list[str] = []

    for line in preamble.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue

        key, _value = stripped.split("=", 1)
        normalized = key.strip().lower().replace("-", "_")

        if normalized in updates:
            out.append(f"{key.strip()}={updates[normalized]}")
            used.add(normalized)
        else:
            out.append(line)

    missing = [k for k in updates if k not in used]
    if missing:
        out.append("")
        out.append("# Added by PID ablation script")
        for k in missing:
            out.append(f"{k}={updates[k]}")

    return "\n".join(out).rstrip() + "\n\n"


def make_variant_config(
    *,
    base_text: str,
    variant: AblationVariant,
    out_dir: Path,
    max_series: int | None,
    case_ids: str,
    skip_completed: bool,
) -> str:
    preamble, branches = split_config_sections(base_text)

    updates = dict(variant.updates)
    updates["out_dir"] = str(out_dir)
    updates["skip_completed"] = "1" if skip_completed else "0"

    if max_series is not None:
        updates["max_series_per_dataset"] = str(int(max_series))

    if case_ids.strip():
        updates["case_ids"] = case_ids.strip()

    header = (
        f"# PID ablation variant: {variant.name}\n"
        f"# {variant.description}\n"
        "# Generated automatically. Dataset preprocessing and the [branches] table are inherited from the base config.\n\n"
    )
    return header + update_preamble(preamble, updates) + branches


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["T2S_USE_LOCAL_DEPS"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_command(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("Running command:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    print("Log:", log_path)
    print("============================================================\n")

    env = child_env()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd) + "\n\n")
        log.write(f"T2S_USE_LOCAL_DEPS={env.get('T2S_USE_LOCAL_DEPS')}\n")
        log.write(f"PYTHONNOUSERSITE={env.get('PYTHONNOUSERSITE')}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return int(proc.wait())


def find_case_results(out_dir: Path) -> Path | None:
    files = sorted(out_dir.glob("all_case_results*.csv"))
    if files:
        return files[0]

    files = sorted(out_dir.glob("case_results*.csv"))
    if files:
        return files[0]

    return None


def summarize_result(variant: AblationVariant, result_dir: Path, config_path: Path, log_path: Path) -> dict:
    result_csv = find_case_results(result_dir)

    row = {
        "variant": variant.name,
        "description": variant.description,
        "status": "missing_result_csv",
        "cases": 0,
        "ARI_mean": math.nan,
        "NMI_mean": math.nan,
        "ARI_std": math.nan,
        "NMI_std": math.nan,
        "ARI_min": math.nan,
        "NMI_min": math.nan,
        "ARI_max": math.nan,
        "NMI_max": math.nan,
        "config_path": str(config_path),
        "result_dir": str(result_dir),
        "result_csv": "",
        "log_path": str(log_path),
    }

    if result_csv is None or not result_csv.exists():
        return row

    df = pd.read_csv(result_csv, encoding="utf-8-sig")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip().eq("")]

    if "ARI" not in df.columns or "NMI" not in df.columns:
        row["status"] = "missing_ARI_NMI_columns"
        row["result_csv"] = str(result_csv)
        return row

    ari_all = pd.to_numeric(df["ARI"], errors="coerce")
    nmi_all = pd.to_numeric(df["NMI"], errors="coerce")
    valid = df[ari_all.notna() & nmi_all.notna()]
    ari = pd.to_numeric(valid["ARI"], errors="coerce")
    nmi = pd.to_numeric(valid["NMI"], errors="coerce")

    row.update({
        "status": "ok",
        "cases": int(len(valid)),
        "ARI_mean": float(ari.mean()) if len(ari) else math.nan,
        "NMI_mean": float(nmi.mean()) if len(nmi) else math.nan,
        "ARI_std": float(ari.std(ddof=1)) if len(ari) > 1 else 0.0,
        "NMI_std": float(nmi.std(ddof=1)) if len(nmi) > 1 else 0.0,
        "ARI_min": float(ari.min()) if len(ari) else math.nan,
        "NMI_min": float(nmi.min()) if len(nmi) else math.nan,
        "ARI_max": float(ari.max()) if len(ari) else math.nan,
        "NMI_max": float(nmi.max()) if len(nmi) else math.nan,
        "result_csv": str(result_csv),
    })
    return row


def write_summary(rows: list[dict], out_root: Path) -> None:
    if not rows:
        return

    summary_csv = out_root / "pid_ablation_summary.csv"
    summary_xlsx = out_root / "pid_ablation_summary.xlsx"

    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        pd.DataFrame(rows).to_excel(summary_xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Could not write Excel summary: {exc!r}")

    print("\n============================================================")
    print("PID ABLATION SUMMARY")
    print("CSV :", summary_csv)
    print("XLSX:", summary_xlsx)
    print("============================================================")

    view = pd.DataFrame(rows)
    cols = [c for c in ["variant", "status", "cases", "ARI_mean", "NMI_mean", "description"] if c in view.columns]
    print(view[cols].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PID ablation configs for UCR-SEG.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--plan", default="core", help="core, components, all, or comma-separated variant names.")
    parser.add_argument("--max-series", type=int, default=None, help="Override max_series_per_dataset for quick testing.")
    parser.add_argument("--case-ids", default="", help="Optional comma-separated UCR-SEG case_ids if supported by the runner.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    entry = args.entry.resolve()
    py_exe = args.python.resolve()

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not entry.exists():
        raise FileNotFoundError(f"Entry script not found: {entry}")
    if not py_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {py_exe}")

    out_root = args.out_root.resolve() if args.out_root else (base_config.parent / "_pid_ablation").resolve()
    config_dir = out_root / "configs"
    log_dir = out_root / "logs"
    result_root = out_root / "results"

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base_text = base_config.read_text(encoding="utf-8-sig")
    variants = choose_variants(args.plan)

    print("============================================================")
    print("PID ablation runner for UCR-SEG")
    print("Base config:", base_config)
    print("Entry      :", entry)
    print("Python     :", py_exe)
    print("Out root   :", out_root)
    print("Plan       :", args.plan)
    print("Variants   :", ", ".join(v.name for v in variants))
    print("T2S_USE_LOCAL_DEPS will be forced to 0 for child runs.")
    if args.max_series is not None:
        print("max_series :", args.max_series)
    if args.case_ids.strip():
        print("case_ids   :", args.case_ids)
    print("============================================================")

    summary_rows: list[dict] = []
    any_failure = False

    for variant in variants:
        result_dir = result_root / variant.name
        config_path = config_dir / f"ucrseg_pid_ablation_{variant.name}.txt"
        log_path = log_dir / f"{variant.name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}.log"

        config_text = make_variant_config(
            base_text=base_text,
            variant=variant,
            out_dir=result_dir,
            max_series=args.max_series,
            case_ids=args.case_ids,
            skip_completed=args.skip_completed,
        )
        config_path.write_text(config_text, encoding="utf-8-sig")
        print(f"\n[CONFIG] {variant.name}: {config_path}")

        if args.dry_run:
            continue

        if not args.summarize_only:
            cmd = [str(py_exe), "-u", str(entry), "--config", str(config_path)]
            code = run_command(cmd, log_path, cwd=entry.parent)
            if code != 0:
                any_failure = True
                print(f"[ERROR] Variant {variant.name} failed with exit code {code}. Continue to next variant.")
        else:
            log_path = log_dir / f"{variant.name}_summarize_only.log"

        summary_rows.append(summarize_result(variant, result_dir, config_path, log_path))

    if args.dry_run:
        print("\n[DONE] Configs generated only. No runs were executed.")
        return 0

    write_summary(summary_rows, out_root)
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
