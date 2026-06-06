                       
r"""
MoCap sensitivity runner for ER-MSSF / P-I-S reliability fusion.

Target location:
    D:\code\teacherT2S\our\mocapM\run_mocap_sensitivity.py

This script is adapted from your existing MoCap runner style:
  - base config: mocap_config.txt
  - entry script: run_our_mocap.py
  - branch grid is preserved after the standalone [branches] marker
  - only selected preamble keys are overridden per variant

Experiments:
  1) topk: Top-K fusion-branch sensitivity, default K={2,4,8}
  2) pis : P/I/S weight sensitivity, where code-level keys still use
           pid_kp / pid_ki / pid_kd for compatibility with your runner.

Outputs are written under the MoCap method directory by default:
  - _topk_sensitivity/
  - _pis_sensitivity/
Each folder contains generated configs, logs, per-variant result dirs,
CSV/XLSX summaries, and a small LaTeX table fragment.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_METHOD_DIR = Path(r"D:\code\teacherT2S\our\mocapM")
DEFAULT_BASE_CONFIG = DEFAULT_METHOD_DIR / "mocap_config.txt"
DEFAULT_FALLBACK_BASE_CONFIG = Path(r"D:\code\teacherT2S\our\mocap\mocap_config.txt")
DEFAULT_ENTRY = DEFAULT_METHOD_DIR / "run_our_mocap.py"
DEFAULT_PYTHON = Path(r"D:\software\anaconda\envs\multit2s_cuda\python.exe")


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    updates: dict[str, str]
    meta: dict[str, object]


def _fmt_float(x: float) -> str:
    return f"{x:.10g}"


def pis_updates(p: float, i: float, s: float, topk: int = 16, tau: float = 0.15) -> dict[str, str]:
    """Return config updates for P/I/S reliability weights.

    The downstream code currently expects pid_kp / pid_ki / pid_kd and
    branch_select_metric=PID. We keep these names for compatibility, but the
    experimental meaning is P/I/S reliability weighting.
    """
    return {
        "select_top_k_branches": str(int(topk)),
        "branch_select_metric": "PID",
        "meta_vote_weight_mode": "pid_weight",
        "pid_kp": _fmt_float(float(p)),
        "pid_ki": _fmt_float(float(i)),
        "pid_kd": _fmt_float(float(s)),
        "pid_softmax_tau": _fmt_float(float(tau)),
    }


def make_topk_variants(topk_list: list[int], p: float, i: float, s: float, tau: float) -> list[Variant]:
    variants: list[Variant] = []
    for k in topk_list:
        if k < 0:
            raise ValueError(f"Top-K must be >= 0, got {k}")
        if k == 0:
            name = "topk_all"
            desc = "Use all candidate branches without Top-K truncation; P/I/S reliability voting is kept."
        else:
            name = f"topk_{k:02d}"
            desc = f"Top-{k} reliability-selected fusion branches with fixed P/I/S weights."
        variants.append(
            Variant(
                name=name,
                description=desc,
                updates=pis_updates(p, i, s, topk=k, tau=tau),
                meta={"experiment": "topk", "top_k": k, "P": p, "I": i, "S": s, "tau": tau},
            )
        )
    return variants


PIS_CORE: list[Variant] = [
    Variant(
        "pis_default_045_035_020",
        "Default P/I/S setting used by the main method.",
        pis_updates(0.45, 0.35, 0.20),
        {"experiment": "pis", "P": 0.45, "I": 0.35, "S": 0.20, "group": "core"},
    ),
    Variant(
        "pis_balanced_033_033_033",
        "Balanced P/I/S weights.",
        pis_updates(1 / 3, 1 / 3, 1 / 3),
        {"experiment": "pis", "P": 1 / 3, "I": 1 / 3, "S": 1 / 3, "group": "core"},
    ),
    Variant(
        "pis_p_high_060_025_015",
        "Higher weight on P: state-distribution health.",
        pis_updates(0.60, 0.25, 0.15),
        {"experiment": "pis", "P": 0.60, "I": 0.25, "S": 0.15, "group": "core"},
    ),
    Variant(
        "pis_i_high_030_055_015",
        "Higher weight on I: inter-branch consistency.",
        pis_updates(0.30, 0.55, 0.15),
        {"experiment": "pis", "P": 0.30, "I": 0.55, "S": 0.15, "group": "core"},
    ),
    Variant(
        "pis_s_high_030_025_045",
        "Higher weight on S: prediction stability.",
        pis_updates(0.30, 0.25, 0.45),
        {"experiment": "pis", "P": 0.30, "I": 0.25, "S": 0.45, "group": "core"},
    ),
]

PIS_COMPONENTS: list[Variant] = [
    Variant(
        "pis_p_only",
        "P term only.",
        pis_updates(1.0, 0.0, 0.0),
        {"experiment": "pis", "P": 1.0, "I": 0.0, "S": 0.0, "group": "components"},
    ),
    Variant(
        "pis_i_only",
        "I term only.",
        pis_updates(0.0, 1.0, 0.0),
        {"experiment": "pis", "P": 0.0, "I": 1.0, "S": 0.0, "group": "components"},
    ),
    Variant(
        "pis_s_only",
        "S term only.",
        pis_updates(0.0, 0.0, 1.0),
        {"experiment": "pis", "P": 0.0, "I": 0.0, "S": 1.0, "group": "components"},
    ),
    Variant(
        "pis_no_p_000_064_036",
        "Remove P; keep the I:S ratio of the default setting.",
        pis_updates(0.0, 0.6363636364, 0.3636363636),
        {"experiment": "pis", "P": 0.0, "I": 0.6363636364, "S": 0.3636363636, "group": "components"},
    ),
    Variant(
        "pis_no_i_069_000_031",
        "Remove I; keep the P:S ratio of the default setting.",
        pis_updates(0.6923076923, 0.0, 0.3076923077),
        {"experiment": "pis", "P": 0.6923076923, "I": 0.0, "S": 0.3076923077, "group": "components"},
    ),
    Variant(
        "pis_no_s_056_044_000",
        "Remove S; keep the P:I ratio of the default setting.",
        pis_updates(0.5625, 0.4375, 0.0),
        {"experiment": "pis", "P": 0.5625, "I": 0.4375, "S": 0.0, "group": "components"},
    ),
]


def choose_pis_variants(plan: str) -> list[Variant]:
    plan = plan.strip().lower()
    if plan == "core":
        return PIS_CORE
    if plan == "components":
        return PIS_COMPONENTS
    if plan == "all":
        return PIS_CORE + PIS_COMPONENTS

    names = {v.name: v for v in PIS_CORE + PIS_COMPONENTS}
    wanted = [x.strip() for x in plan.replace(";", ",").split(",") if x.strip()]
    if not wanted:
        raise ValueError(f"Empty PIS plan: {plan!r}")
    missing = [x for x in wanted if x not in names]
    if missing:
        raise ValueError(f"Unknown PIS variant(s): {missing}. Available: {sorted(names)}")
    return [names[x] for x in wanted]


def parse_topk_list(text: str) -> list[int]:
    values = [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    if not values:
        raise ValueError("Empty --topk-list")
    topks = []
    for x in values:
        if x.lower() in {"all", "full"}:
            topks.append(0)
        else:
            topks.append(int(x))
    seen = set()
    unique = []
    for k in topks:
        if k not in seen:
            unique.append(k)
            seen.add(k)
    return unique


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
    new_lines: list[str] = []
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
        new_lines.append("# Added by MoCap sensitivity script")
        for k in missing:
            new_lines.append(f"{k}={updates[k]}")
    return "\n".join(new_lines).rstrip() + "\n\n"


def make_variant_config(
    base_text: str,
    variant: Variant,
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
    meta_lines = "\n".join(f"# meta.{k}={v}" for k, v in variant.meta.items())
    header = (
        f"# MoCap sensitivity variant: {variant.name}\n"
        f"# {variant.description}\n"
        f"{meta_lines}\n"
        f"# Generated automatically. Branch table after [branches] is preserved.\n\n"
    )
    return header + new_preamble + branches


def run_command(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n============================================================")
    print("Running command:")
    print(" ".join(f'\"{x}\"' if " " in str(x) else str(x) for x in cmd))
    print("Log:", log_path)
    print("============================================================\n")

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(f'\"{x}\"' if " " in str(x) else str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
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


def summarize_result(variant: Variant, result_dir: Path, config_path: Path, log_path: Path) -> dict:
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
    row.update(variant.meta)

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

    row.update(
        {
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
        }
    )
    return row


def _num(x: object, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "--"
    if math.isnan(v):
        return "--"
    return f"{v:.{digits}f}"


def write_latex_fragment(df: pd.DataFrame, experiment: str, path: Path) -> None:
    lines: list[str] = []
    latex_newline = " " + "\\\\"
    lines.append("% Auto-generated LaTeX rows. Copy into your table body if needed.")
    if experiment == "topk":
        lines.append("% Top-$K_b$ & ARI & NMI " + "\\\\")
        sort_cols = ["top_k"] if "top_k" in df.columns else ["variant"]
        view = df.sort_values(sort_cols)
        for _, r in view.iterrows():
            k = r.get("top_k", r.get("variant", ""))
            k_label = "All" if str(k) == "0" else str(k)
            lines.append(f"{k_label} & {_num(r.get('ARI_mean'))} & {_num(r.get('NMI_mean'))}{latex_newline}")
    else:
        lines.append("% Variant & $\\eta_P$ & $\\eta_I$ & $\\eta_S$ & ARI & NMI " + "\\\\")
        view = df.copy()
        if "group" in view.columns:
            view["_group_order"] = view["group"].map({"core": 0, "components": 1}).fillna(2)
            view = view.sort_values(["_group_order", "variant"])
        for _, r in view.iterrows():
            name = str(r.get("variant", ""))
            lines.append(
                f"{name} & {_num(r.get('P'), 2)} & {_num(r.get('I'), 2)} & {_num(r.get('S'), 2)} & "
                f"{_num(r.get('ARI_mean'))} & {_num(r.get('NMI_mean'))}{latex_newline}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_summary(summary_rows: list[dict], out_root: Path, experiment: str) -> None:
    if not summary_rows:
        return

    prefix = f"{experiment}_sensitivity"
    summary_csv = out_root / f"{prefix}_summary.csv"
    summary_xlsx = out_root / f"{prefix}_summary.xlsx"
    latex_path = out_root / f"{prefix}_latex_rows.tex"

    fieldnames = list(summary_rows[0].keys())
    for row in summary_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    df = pd.DataFrame(summary_rows)
    try:
        df.to_excel(summary_xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Could not write Excel summary: {exc!r}")

    try:
        write_latex_fragment(df, experiment, latex_path)
    except Exception as exc:
        print(f"[WARN] Could not write LaTeX fragment: {exc!r}")

    print("\n============================================================")
    print(f"{experiment.upper()} SENSITIVITY SUMMARY")
    print("CSV  :", summary_csv)
    print("XLSX :", summary_xlsx)
    print("LaTeX:", latex_path)
    print("============================================================")
    cols = [
        c
        for c in ["variant", "status", "cases", "top_k", "P", "I", "S", "ARI_mean", "NMI_mean", "description"]
        if c in df.columns
    ]
    print(df[cols].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MoCap Top-K or P/I/S sensitivity experiments.")
    parser.add_argument("--experiment", choices=["topk", "pis"], required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out-root", type=Path, default=None)

    parser.add_argument("--topk-list", default="2,4,8", help="Only for --experiment topk. Use 0/all for all branches.")
    parser.add_argument("--pis-plan", default="core", help="Only for --experiment pis: core, components, all, or comma-separated variant names.")
    parser.add_argument("--default-p", type=float, default=0.45)
    parser.add_argument("--default-i", type=float, default=0.35)
    parser.add_argument("--default-s", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.15)

    parser.add_argument("--max-series", type=int, default=None, help="Override max_series_per_dataset for quick testing.")
    parser.add_argument("--only-case-ids", default="", help="Run only selected case IDs, e.g. amc_86_14.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but do not run.")
    parser.add_argument("--summarize-only", action="store_true", help="Do not run; only summarize existing result folders.")
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    entry = args.entry.resolve()
    py_exe = args.python.resolve()

    if not base_config.exists():
        fallback = DEFAULT_FALLBACK_BASE_CONFIG.resolve()
        if fallback.exists():
            print(f"[WARN] Base config not found in mocapM, fallback to: {fallback}")
            base_config = fallback
        else:
            raise FileNotFoundError(f"Base config not found: {base_config}; fallback also not found: {fallback}")
    if not entry.exists():
        raise FileNotFoundError(f"Entry script not found: {entry}")
    if not py_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {py_exe}")

    if args.experiment == "topk":
        topks = parse_topk_list(args.topk_list)
        variants = make_topk_variants(topks, args.default_p, args.default_i, args.default_s, args.tau)
        default_out_root = entry.parent / "_topk_sensitivity"
    else:
        variants = choose_pis_variants(args.pis_plan)
        default_out_root = entry.parent / "_pis_sensitivity"

    out_root = args.out_root.resolve() if args.out_root else default_out_root.resolve()
    config_dir = out_root / "configs"
    log_dir = out_root / "logs"
    result_root = out_root / "results"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base_text = read_config(base_config)

    print("============================================================")
    print("MoCap sensitivity runner")
    print("Experiment :", args.experiment)
    print("Base config:", base_config)
    print("Entry      :", entry)
    print("Python     :", py_exe)
    print("Out root   :", out_root)
    print("Variants   :", ", ".join(v.name for v in variants))
    if args.max_series is not None:
        print("max_series :", args.max_series)
    if args.only_case_ids.strip():
        print("only cases :", args.only_case_ids)
    print("============================================================")

    summary_rows: list[dict] = []
    any_failure = False

    for variant in variants:
        result_dir = result_root / variant.name
        config_path = config_dir / f"mocap_{args.experiment}_{variant.name}.txt"
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

    write_summary(summary_rows, out_root, args.experiment)
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
