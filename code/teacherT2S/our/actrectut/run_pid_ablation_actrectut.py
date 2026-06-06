                       
r"""
PID ablation runner for ActRecTut.
Put in: D:\code\teacherT2S\our\actrectut\
"""

from __future__ import annotations
import argparse, csv, math, os, subprocess, time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DEFAULT_BASE_CONFIG = Path(r"D:\code\teacherT2S\our\actrectut\actrectut_config.txt")
DEFAULT_ENTRY = Path(r"D:\code\teacherT2S\our\actrectut\run_our_actrectut.py")
DEFAULT_PYTHON = Path(r"D:\software\anaconda\envs\multit2s_cuda\python.exe")

@dataclass(frozen=True)
class Variant:
    name: str
    desc: str
    updates: dict[str, str]

def v(name, desc, select, vote, topk="16", kp="0.45", ki="0.35", kd="0.20"):
    return Variant(name, desc, {
        "select_top_k_branches": topk,
        "branch_select_metric": select,
        "meta_vote_weight_mode": vote,
        "pid_kp": kp,
        "pid_ki": ki,
        "pid_kd": kd,
        "pid_softmax_tau": "0.15",
    })

CORE = [
    v("full_pid", "Full method: PID branch selection + PID meta voting.", "PID", "pid_weight"),
    v("no_pid_peer", "Remove PID: PEER selection + PEER reliability voting.", "PEER", "branch_reliability"),
    v("pid_select_uniform", "PID branch selection + uniform voting.", "PID", "uniform"),
    v("peer_select_pid_weight", "PEER branch selection + PID voting weights.", "PEER", "pid_weight"),
    v("all_branches_uniform", "No top-k selection; all branches vote uniformly.", "PEER", "uniform", topk="0"),
]

COMPONENTS = [
    v("pid_p_only", "P term only.", "PID", "pid_weight", kp="1.0", ki="0.0", kd="0.0"),
    v("pid_i_only", "I term only.", "PID", "pid_weight", kp="0.0", ki="1.0", kd="0.0"),
    v("pid_d_only", "D term only.", "PID", "pid_weight", kp="0.0", ki="0.0", kd="1.0"),
    v("pid_no_p", "Remove P.", "PID", "pid_weight", kp="0.0", ki="0.6363636364", kd="0.3636363636"),
    v("pid_no_i", "Remove I.", "PID", "pid_weight", kp="0.6923076923", ki="0.0", kd="0.3076923077"),
    v("pid_no_d", "Remove D.", "PID", "pid_weight", kp="0.5625", ki="0.4375", kd="0.0"),
]

def choose(plan: str) -> list[Variant]:
    plan = plan.strip().lower()
    if plan == "core": return CORE
    if plan == "components": return COMPONENTS
    if plan == "all": return CORE + COMPONENTS
    table = {x.name: x for x in CORE + COMPONENTS}
    wanted = [x.strip() for x in plan.replace(";", ",").split(",") if x.strip()]
    missing = [x for x in wanted if x not in table]
    if missing:
        raise ValueError(f"Unknown variants: {missing}. Available: {sorted(table)}")
    return [table[x] for x in wanted]

def clean(line: str) -> str:
    return line.strip().lstrip("\ufeff").lower()

def split_config(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if clean(line) == "[branches]":
            return "\n".join(lines[:i]).rstrip() + "\n", "\n".join(lines[i:]).rstrip() + "\n"
    for i, line in enumerate(lines):
        if clean(line).replace(" ", "").startswith("enabled,dataset,branch_name"):
            print("[WARN] Missing standalone [branches]; inserted it before branch CSV header.")
            return "\n".join(lines[:i]).rstrip() + "\n", "[branches]\n" + "\n".join(lines[i:]).rstrip() + "\n"
    preview = "\n".join(f"{i+1:03d}: {line}" for i, line in enumerate(lines[:120]))
    raise ValueError("Cannot find [branches] or branch CSV header. First 120 lines:\n" + preview)

def update_preamble(preamble: str, updates: dict[str, str]) -> str:
    used, out = set(), []
    for line in preamble.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line); continue
        key, _ = stripped.split("=", 1)
        norm = key.strip().lower().replace("-", "_")
        if norm in updates:
            out.append(f"{key.strip()}={updates[norm]}")
            used.add(norm)
        else:
            out.append(line)
    missing = [k for k in updates if k not in used]
    if missing:
        out += ["", "# Added by PID ablation script"]
        out += [f"{k}={updates[k]}" for k in missing]
    return "\n".join(out).rstrip() + "\n\n"

def make_config(base_text: str, variant: Variant, out_dir: Path, max_series: int | None, rounds: str, case_ids: str, skip_completed: bool) -> str:
    pre, branches = split_config(base_text)
    updates = dict(variant.updates)
    updates["out_dir"] = str(out_dir)
    updates["skip_completed"] = "1" if skip_completed else "0"
    if max_series is not None:
        updates["max_series_per_dataset"] = str(int(max_series))
    if rounds.strip():
        updates["rounds"] = rounds.strip()
    if case_ids.strip():
        updates["case_ids"] = case_ids.strip()
    header = (
        f"# PID ablation variant: {variant.name}\n"
        f"# {variant.desc}\n"
        "# Generated automatically. Dataset preprocessing and branch table are inherited from base config.\n"
        "# For formal runs, leave rounds/case_ids blank; do not use seed-gate selection.\n\n"
    )
    return header + update_preamble(pre, updates) + branches

def child_env():
    env = os.environ.copy()
    env["T2S_USE_LOCAL_DEPS"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    return env

def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*60)
    print("Running:", " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    print("Log:", log_path)
    print("="*60 + "\n")
    env = child_env()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        log.write(f"T2S_USE_LOCAL_DEPS={env.get('T2S_USE_LOCAL_DEPS')}\nPYTHONNOUSERSITE={env.get('PYTHONNOUSERSITE')}\n\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line); log.flush()
        return int(proc.wait())

def result_csv(out_dir: Path) -> Path | None:
    for pat in ("all_case_results*.csv", "case_results*.csv"):
        files = sorted(out_dir.glob(pat))
        if files: return files[0]
    return None

def summarize(variant: Variant, result_dir: Path, config_path: Path, log_path: Path) -> dict:
    path = result_csv(result_dir)
    row = {
        "variant": variant.name, "description": variant.desc, "status": "missing_result_csv",
        "cases": 0, "ARI_mean": math.nan, "NMI_mean": math.nan, "ARI_std": math.nan, "NMI_std": math.nan,
        "ARI_min": math.nan, "NMI_min": math.nan, "ARI_max": math.nan, "NMI_max": math.nan,
        "config_path": str(config_path), "result_dir": str(result_dir), "result_csv": "", "log_path": str(log_path),
    }
    if path is None or not path.exists(): return row
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip().eq("")]
    if "ARI" not in df.columns or "NMI" not in df.columns:
        row["status"] = "missing_ARI_NMI_columns"; row["result_csv"] = str(path); return row
    ari0, nmi0 = pd.to_numeric(df["ARI"], errors="coerce"), pd.to_numeric(df["NMI"], errors="coerce")
    valid = df[ari0.notna() & nmi0.notna()]
    ari, nmi = pd.to_numeric(valid["ARI"], errors="coerce"), pd.to_numeric(valid["NMI"], errors="coerce")
    row.update({
        "status": "ok", "cases": int(len(valid)),
        "ARI_mean": float(ari.mean()) if len(ari) else math.nan,
        "NMI_mean": float(nmi.mean()) if len(nmi) else math.nan,
        "ARI_std": float(ari.std(ddof=1)) if len(ari) > 1 else 0.0,
        "NMI_std": float(nmi.std(ddof=1)) if len(nmi) > 1 else 0.0,
        "ARI_min": float(ari.min()) if len(ari) else math.nan,
        "NMI_min": float(nmi.min()) if len(nmi) else math.nan,
        "ARI_max": float(ari.max()) if len(ari) else math.nan,
        "NMI_max": float(nmi.max()) if len(nmi) else math.nan,
        "result_csv": str(path),
    })
    return row

def write_summary(rows: list[dict], out_root: Path) -> None:
    if not rows: return
    csv_path = out_root / "pid_ablation_summary.csv"
    xlsx_path = out_root / "pid_ablation_summary.xlsx"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    try:
        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Excel summary failed: {exc!r}")
    print("\n" + "="*60)
    print("PID ABLATION SUMMARY")
    print("CSV :", csv_path)
    print("XLSX:", xlsx_path)
    print("="*60)
    view = pd.DataFrame(rows)
    cols = [c for c in ["variant","status","cases","ARI_mean","NMI_mean","description"] if c in view.columns]
    print(view[cols].to_string(index=False))

def main() -> int:
    p = argparse.ArgumentParser(description="Run PID ablations for ActRecTut.")
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--plan", default="core")
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--rounds", default="")
    p.add_argument("--case-ids", default="")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    base_config, entry, py_exe = args.base_config.resolve(), args.entry.resolve(), args.python.resolve()
    for name, path in [("base config", base_config), ("entry", entry), ("python", py_exe)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    out_root = args.out_root.resolve() if args.out_root else (base_config.parent / "_pid_ablation").resolve()
    config_dir, log_dir, result_root = out_root / "configs", out_root / "logs", out_root / "results"
    config_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True); result_root.mkdir(parents=True, exist_ok=True)

    base_text = base_config.read_text(encoding="utf-8-sig")
    variants = choose(args.plan)

    print("="*60)
    print("PID ablation runner for ActRecTut")
    print("Base config:", base_config)
    print("Entry      :", entry)
    print("Python     :", py_exe)
    print("Out root   :", out_root)
    print("Plan       :", args.plan)
    print("Variants   :", ", ".join(x.name for x in variants))
    if args.max_series is not None: print("max_series :", args.max_series)
    if args.rounds.strip(): print("rounds     :", args.rounds)
    if args.case_ids.strip(): print("case_ids   :", args.case_ids)
    print("="*60)

    rows, any_failure = [], False
    for variant in variants:
        result_dir = result_root / variant.name
        config_path = config_dir / f"actrectut_pid_ablation_{variant.name}.txt"
        log_path = log_dir / f"{variant.name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}.log"
        config_path.write_text(make_config(base_text, variant, result_dir, args.max_series, args.rounds, args.case_ids, args.skip_completed), encoding="utf-8-sig")
        print(f"\n[CONFIG] {variant.name}: {config_path}")

        if args.dry_run:
            continue
        if not args.summarize_only:
            code = run_cmd([str(py_exe), "-u", str(entry), "--config", str(config_path)], log_path, cwd=entry.parent)
            if code != 0:
                any_failure = True
                print(f"[ERROR] Variant {variant.name} failed with exit code {code}. Continue.")
        else:
            log_path = log_dir / f"{variant.name}_summarize_only.log"
        rows.append(summarize(variant, result_dir, config_path, log_path))

    if args.dry_run:
        print("[DONE] Configs generated only.")
        return 0
    write_summary(rows, out_root)
    return 1 if any_failure else 0

if __name__ == "__main__":
    raise SystemExit(main())
