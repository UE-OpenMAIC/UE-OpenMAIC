                       
r"""
PID ablation runner for PAMAP2_zero.

Default target:
    D:\code\teacherT2S\our\PAMAP2_zero\pamap2_zero_config.txt
    D:\code\teacherT2S\our\PAMAP2_zero\run_our_pamap2_zero.py

Purpose:
    Generate and run the same 5 core PID ablations as MoCap:
      1. full_pid
      2. no_pid_peer
      3. pid_select_uniform
      4. peer_select_pid_weight
      5. all_branches_uniform

It preserves:
    - PAMAP2_zero preprocessing from the base config
    - the original 64-branch table
    - full_sensor + remove activity_id=0 protocol
    - same seed / model / meta settings unless overridden by the ablation updates

Usage:
    cd /d D:\code\teacherT2S\our\PAMAP2_zero
    RUN_PID_ABLATION_PAMAP2_ZERO.cmd core quick
    RUN_PID_ABLATION_PAMAP2_ZERO.cmd core
    RUN_PID_ABLATION_PAMAP2_ZERO.cmd all

Outputs:
    D:\code\teacherT2S\our\PAMAP2_zero\_pid_ablation\
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_BASE_CONFIG = Path(r"D:\code\teacherT2S\our\PAMAP2_zero\pamap2_zero_config.txt")
DEFAULT_ENTRY = Path(r"D:\code\teacherT2S\our\PAMAP2_zero\run_our_pamap2_zero.py")
DEFAULT_PYTHON = Path(r"D:\software\anaconda\envs\multit2s_cuda\python.exe")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    updates: dict[str, str]


CORE_VARIANTS = [
    AblationVariant(
        name="full_pid",
        description="Original full method: PID branch selection + PID meta voting.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.45",
            "pid_ki": "0.35",
            "pid_kd": "0.20",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="no_pid_peer",
        description="Remove PID completely: PEER selection + PEER reliability voting.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PEER",
            "meta_vote_weight_mode": "branch_reliability",
            "pid_kp": "0.45",
            "pid_ki": "0.35",
            "pid_kd": "0.20",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_select_uniform",
        description="Keep PID branch selection, remove PID voting weights.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "uniform",
            "pid_kp": "0.45",
            "pid_ki": "0.35",
            "pid_kd": "0.20",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="peer_select_pid_weight",
        description="Remove PID from branch selection, keep PID meta voting weights.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PEER",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.45",
            "pid_ki": "0.35",
            "pid_kd": "0.20",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="all_branches_uniform",
        description="No top-k selection and no reliability weights; all candidate branches vote uniformly.",
        updates={
            "select_top_k_branches": "0",
            "branch_select_metric": "PEER",
            "meta_vote_weight_mode": "uniform",
            "pid_kp": "0.45",
            "pid_ki": "0.35",
            "pid_kd": "0.20",
            "pid_softmax_tau": "0.15",
        },
    ),
]


COMPONENT_VARIANTS = [
    AblationVariant(
        name="pid_p_only",
        description="PID component ablation: P term only.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "1.0",
            "pid_ki": "0.0",
            "pid_kd": "0.0",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_i_only",
        description="PID component ablation: I term only.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.0",
            "pid_ki": "1.0",
            "pid_kd": "0.0",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_d_only",
        description="PID component ablation: D term only.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.0",
            "pid_ki": "0.0",
            "pid_kd": "1.0",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_no_p",
        description="PID component ablation: remove P, keep I:D ratio normalized.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.0",
            "pid_ki": "0.6363636364",
            "pid_kd": "0.3636363636",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_no_i",
        description="PID component ablation: remove I, keep P:D ratio normalized.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.6923076923",
            "pid_ki": "0.0",
            "pid_kd": "0.3076923077",
            "pid_softmax_tau": "0.15",
        },
    ),
    AblationVariant(
        name="pid_no_d",
        description="PID component ablation: remove D, keep P:I ratio normalized.",
        updates={
            "select_top_k_branches": "16",
            "branch_select_metric": "PID",
            "meta_vote_weight_mode": "pid_weight",
            "pid_kp": "0.5625",
            "pid_ki": "0.4375",
            "pid_kd": "0.0",
            "pid_softmax_tau": "0.15",
        },
    ),
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
        raise ValueError(f"Empty plan: {plan!r}")
    missing = [x for x in wanted if x not in names]
    if missing:
        raise ValueError(f"Unknown variant(s): {missing}. Available: {sorted(names)}")
    return [names[x] for x in wanted]


def read_config(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def split_config_sections(text: str) -> tuple[str, str]:
    """Split config into preamble and branch table.

    The marker must be a standalone line exactly equal to [branches].
    This avoids matching comments that merely mention [branches].
    """
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "[branches]":
            preamble = "\n".join(lines[:idx]).rstrip() + "\n"
            branches = "\n".join(lines[idx:]).rstrip() + "\n"
            return preamble, branches
    raise ValueError("Base config does not contain a standalone [branches] line.")


def update_preamble(preamble: str, updates: dict[str, str]) -> str:
    lines = preamble.splitlines()
    used = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key, _value = stripped.split("=", 1)
        normalized = key.strip().lower().replace("-", "_")
        if normalized in updates:
            new_lines.append(f"{key.strip()}={updates[normalized]}")
            used.add(normalized)
        else:
            new_lines.append(line)

    missing = [k for k in updates if k not in used]
    if missing:
        new_lines.append("")
        new_lines.append("# Added by PID ablation script")
        for k in missing:
            new_lines.append(f"{k}={updates[k]}")
    return "\n".join(new_lines).rstrip() + "\n\n"


def make_variant_config(
    *,
    base_text: str,
    variant: AblationVariant,
    out_dir: Path,
    max_series: int | None,
    only_case_ids: str,
    skip_completed: bool,
) -> str:
    preamble, branches = split_config_sections(base_text)
    updates = dict(variant.updates)

                                                                        
                                                       
    updates["out_dir"] = str(out_dir)
    updates["skip_completed"] = "1" if skip_completed else "0"

    if max_series is not None:
        updates["max_series_per_dataset"] = str(int(max_series))
    if only_case_ids.strip():
        updates["only_case_ids"] = only_case_ids.strip()
        updates["priority_case_ids"] = ""

    new_preamble = update_preamble(preamble, updates)
    header = (
        f"# PID ablation variant: {variant.name}\n"
        f"# {variant.description}\n"
        f"# Generated automatically. Dataset preprocessing and the [branches] table are inherited from the base config.\n\n"
    )
    return header + new_preamble + branches


def build_child_env() -> dict[str, str]:
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

    env = build_child_env()
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
    candidates = sorted(out_dir.glob("all_case_results*.csv"))
    if candidates:
        return candidates[0]
    candidates = sorted(out_dir.glob("case_results*.csv"))
    return candidates[0] if candidates else None


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


def write_summary(summary_rows: list[dict], out_root: Path) -> None:
    summary_csv = out_root / "pid_ablation_summary.csv"
    summary_xlsx = out_root / "pid_ablation_summary.xlsx"

    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    try:
        df = pd.DataFrame(summary_rows)
        df.to_excel(summary_xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Could not write Excel summary: {exc!r}")

    print("\n============================================================")
    print("PID ABLATION SUMMARY")
    print("CSV :", summary_csv)
    print("XLSX:", summary_xlsx)
    print("============================================================")
    view = pd.DataFrame(summary_rows)
    cols = [c for c in ["variant", "status", "cases", "ARI_mean", "NMI_mean", "description"] if c in view.columns]
    print(view[cols].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PID ablation configs for PAMAP2_zero.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--plan", default="core", help="core, components, all, or comma-separated variant names.")
    parser.add_argument("--max-series", type=int, default=None, help="Override max_series_per_dataset for quick testing.")
    parser.add_argument("--only-case-ids", default="", help="Run only selected case IDs, e.g. 101.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but do not run.")
    parser.add_argument("--summarize-only", action="store_true", help="Do not run; only summarize existing result folders.")
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

    base_text = read_config(base_config)
    variants = choose_variants(args.plan)

    print("============================================================")
    print("PID ablation runner for PAMAP2_zero")
    print("Base config:", base_config)
    print("Entry      :", entry)
    print("Python     :", py_exe)
    print("Out root   :", out_root)
    print("Plan       :", args.plan)
    print("Variants   :", ", ".join(v.name for v in variants))
    print("T2S_USE_LOCAL_DEPS will be forced to 0 for child runs.")
    if args.max_series is not None:
        print("max_series :", args.max_series)
    if args.only_case_ids.strip():
        print("only cases :", args.only_case_ids)
    print("============================================================")

    summary_rows = []
    any_failure = False

    for variant in variants:
        result_dir = result_root / variant.name
        config_path = config_dir / f"pamap2_zero_pid_ablation_{variant.name}.txt"
        log_path = log_dir / f"{variant.name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}.log"

        config_text = make_variant_config(
            base_text=base_text,
            variant=variant,
            out_dir=result_dir,
            max_series=args.max_series,
            only_case_ids=args.only_case_ids,
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
