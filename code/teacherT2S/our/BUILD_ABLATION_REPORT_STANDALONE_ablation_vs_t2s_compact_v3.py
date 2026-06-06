                       
r"""
Strict ablation-report builder for teacherT2S.

Purpose
-------
This script builds the PID-ablation report by reading case-level results from
actual result directories, instead of trusting only pre-collected summary rows.

Default policy
--------------
1) full_pid / Ours is read STRICTLY from formal Ours result files, for example:
     D:\code\teacherT2S\our\actrectut\results_actrectut_our.csv
     D:\code\teacherT2S\our\actrectut\results_actrectut_our\*.csv
     D:\code\teacherT2S\our\PAMAP2_zero\results_pamap2_zero_fullsensor_remove0\all_case_results.csv

2) Other ablation variants are read from the ablation result folders, for example:
     D:\code\teacherT2S\our\actrectut\_pid_ablation\results\pid_select_uniform\*.csv
     D:\code\teacherT2S\our\actrectut\_pid_ablation\results\all_branches_uniform\*.csv
     D:\code\teacherT2S\our\actrectut\_pid_ablation\results\peer_select_pid_weight\*.csv
     D:\code\teacherT2S\our\actrectut\_pid_ablation\results\no_pid_peer\*.csv

3) The ablation folder named full_pid is NOT used by default, because full_pid is
   required to come from formal Ours. To override this for diagnostics only, use:
     --full-pid-source ablation_folder

Outputs
-------
Default output directory:
  D:\code\teacherT2S\our\_ari_nmi_summary\report\ablationResult

Default Time2State root:
  D:\code\baseline\t2s

Main outputs only:
  - ablation_vs_t2s_mean_table.csv
  - ablation_vs_t2s_gain_plot.png

Example
-------
python BUILD_ABLATION_REPORT_STANDALONE_scan_ablation_dirs.py

python BUILD_ABLATION_REPORT_STANDALONE_scan_ablation_dirs.py ^
  --root D:\code\teacherT2S ^
  --datasets actrectut,pamap2_zero,usc-had,ucr-seg,mocap,synthetic
"""

from __future__ import annotations

import argparse
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, studentized_range
except Exception:
    ttest_rel = None
    studentized_range = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEACHERT2S_ROOT = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = DEFAULT_TEACHERT2S_ROOT.parent.parent
DEFAULT_OUR_ROOT = SCRIPT_DIR
DEFAULT_BASELINE_ROOT = DEFAULT_REPO_ROOT / "baseline"


                                                              
                  
                                                              

METRICS = ["ARI", "NMI"]

DATASET_ORDER = [
    "synthetic",
    "mocap",
    "actrectut",
    "pamap2_zero",
    "ucr-seg",
    "usc-had",
]

DATASET_ALIASES = {
    "synthetic": "Synthetic",
    "mocap": "MoCap",
    "actrectut": "ActRecTut",
    "pamap2": "PAMAP2",
    "pamap2_zero": "PAMAP2",
    "pamap2-zero": "PAMAP2",
    "ucr-seg": "UCR-SEG",
    "ucrseg": "UCR-SEG",
    "tssb": "UCR-SEG",
    "usc-had": "USC-HAD",
    "uschad": "USC-HAD",
}

ABLATION_ORDER = [
    "full_pid",
    "pid_select_uniform",
    "all_branches_uniform",
    "peer_select_pid_weight",
    "no_pid_peer",
]

VARIANT_DISPLAY = {
    "full_pid": "Full method",
    "pid_select_uniform": "PID selection + uniform fusion",
    "all_branches_uniform": "All branches + uniform fusion",
    "peer_select_pid_weight": "Peer selection + PID-weighted fusion",
    "no_pid_peer": "Peer fusion w/o PID",
}

VARIANT_FOLDER_ALIASES = {
    "full_pid": ["full_pid", "ours", "full"],
    "pid_select_uniform": ["pid_select_uniform", "pid_selected_uniform", "pid_selection_uniform", "pid_select_unif"],
    "all_branches_uniform": ["all_branches_uniform", "all_branch_uniform", "all_uniform"],
    "peer_select_pid_weight": ["peer_select_pid_weight", "peer_selected_pid_weight", "peer_select_pid", "peer_pid_weight"],
    "no_pid_peer": ["no_pid_peer", "peer_without_pid", "peer_fusion_wo_pid", "peer_fusion_without_pid"],
}

EXCLUDE_PATH_TOKENS_FOR_FORMAL_OURS = [
    "_pid_ablation",
    "ablation",
    "smoke",
    "debug",
    "old",
]

BAD_CSV_NAME_TOKENS = [
    "manifest",
    "source_manifest",
    "rank_",
    "ttest",
    "pairwise",
    "letter",
    "k_sweep",
    "summary_by",
]

                                                                             
                                                                             
                                                                             
                         
PREFERRED_CASE_CSV_NAMES = [
    "all_case_results.csv",
    "case_results.csv",
    "final_case_results.csv",
    "all_cases.csv",
    "results.csv",
]


                                                              
                  
                                                              

def norm_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x or "").strip().lower())


def norm_dataset(x: object) -> str:
    s = str(x or "").strip().lower().replace("\\", "/").split("/")[-1]
    s = s.replace("_", "-").replace(" ", "")
    if s in {"pamap2", "pamap2zero", "pamap2-zero", "pamap2-0"}:
        return "pamap2_zero"
    if s in {"uschad", "usc-had"}:
        return "usc-had"
    if s in {"ucrseg", "ucr-seg", "tssb"}:
        return "ucr-seg"
    return s


def display_dataset(x: object) -> str:
    ds = norm_dataset(x)
    return DATASET_ALIASES.get(ds, str(x))


def read_csv_any(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Cannot read {path}: {last!r}")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    mp = {norm_key(c): c for c in df.columns}
    for c in candidates:
        k = norm_key(c)
        if k in mp:
            return mp[k]
    return None


def find_metric_col(df: pd.DataFrame, metric: str) -> str | None:
    bad = {"std", "min", "max", "delta", "gain", "diff", "rank"}
    candidates = {
        "ARI": ["ARI", "ari", "adjusted_rand_score", "adjusted_rand_index", "adjustedrandindex"],
        "NMI": ["NMI", "nmi", "normalized_mutual_info", "normalized_mutual_information"],
    }[metric]
    col = find_col(df, candidates)
    if col is not None:
        return col
    key = metric.lower()
    for c in df.columns:
        nc = norm_key(c)
        if key in nc and not any(t in nc for t in bad):
            return c
    return None


def find_case_col(df: pd.DataFrame) -> str | None:
    return find_col(df, [
        "case_id", "case", "sample_id", "sample", "series_id", "series",
        "name", "file", "filename", "id", "case_name", "seq", "sequence",
    ])


def status_ok_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    status_col = find_col(out, ["status"])
    if status_col is not None:
        ok = out[status_col].fillna("").astype(str).str.strip().str.lower().isin(["", "ok", "success", "done"])
        out = out[ok].copy()
    error_col = find_col(out, ["error", "err"])
    if error_col is not None:
        noerr = out[error_col].fillna("").astype(str).str.strip().eq("")
        out = out[noerr].copy()
    return out


def canonical_case_id(dataset: object, raw_case: object) -> str:
    ds = norm_dataset(dataset)
    s = str(raw_case if raw_case is not None else "").strip()
    s = s.replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(csv|txt|dat|npy|npz|mat|4d)$", "", s, flags=re.I)
    s_low = s.lower().strip()

    if re.fullmatch(r"\d+\.0", s_low):
        s_low = s_low[:-2]

    if ds == "actrectut":
        m = re.search(r"subject\s*0*(\d+)\s*[_-]?\s*walk\s*0*(\d+)$", s_low)
        if m:
            return f"subject{int(m.group(1))}_walk{int(m.group(2))}"
        m = re.search(r"subj\s*0*(\d+)\s*[_-]?\s*walk\s*0*(\d+)$", s_low)
        if m:
            return f"subject{int(m.group(1))}_walk{int(m.group(2))}"
        m = re.search(r"subject\s*0*(\d+)\s*[_-]?\s*walk$", s_low)
        if m:
            return f"subject{int(m.group(1))}_walk"
        m = re.search(r"subj\s*0*(\d+)\s*[_-]?\s*walk$", s_low)
        if m:
            return f"subject{int(m.group(1))}_walk"

    if ds == "pamap2_zero":
        m = re.search(r"subject\s*0*(\d+)", s_low)
        if m:
            num = int(m.group(1))
            return str(num + 100) if 1 <= num <= 8 else str(num)
        m = re.fullmatch(r"0*(\d+)", s_low)
        if m:
            num = int(m.group(1))
            return str(num + 100) if 1 <= num <= 8 else str(num)

    if ds == "usc-had":
        m = re.search(r"subject\s*0*(\d+)\s*[_-]?\s*target\s*0*(\d+)", s_low)
        if m:
            return f"s{int(m.group(1))}_t{int(m.group(2))}"
        m = re.search(r"s\s*0*(\d+)\s*[_-]?\s*t\s*0*(\d+)", s_low)
        if m:
            return f"s{int(m.group(1))}_t{int(m.group(2))}"

    if ds == "mocap":
        m = re.search(r"amc\s*[_-]?\s*(\d+)\s*[_-]?\s*(\d+)", s_low)
        if m:
            return f"amc_{int(m.group(1))}_{int(m.group(2))}"

    if ds == "synthetic":
        m = re.search(r"(?:case|sample|series|synthetic)\s*[_-]?\s*0*(\d+)$", s_low)
        if m:
            return str(int(m.group(1)))
        if re.fullmatch(r"0*\d+", s_low):
            return str(int(s_low))

    s_low = s_low.replace(" ", "").replace("-", "_")
    return s_low


def should_ignore_csv(path: Path) -> bool:
    name = path.name.lower()
    return any(tok in name for tok in BAD_CSV_NAME_TOKENS)


def parse_case_metric_csv(path: Path, dataset: str, variant: str) -> pd.DataFrame | None:
    try:
        df = read_csv_any(path)
    except Exception as exc:
        print(f"[WARN] Cannot read CSV: {path} | {exc}")
        return None

    df = status_ok_filter(df)
    ari_col = find_metric_col(df, "ARI")
    nmi_col = find_metric_col(df, "NMI")
    case_col = find_case_col(df)

    if ari_col is None or nmi_col is None or case_col is None:
        return None

    out = pd.DataFrame({
        "dataset": norm_dataset(dataset),
        "dataset_display": display_dataset(dataset),
        "variant": variant,
        "variant_display": VARIANT_DISPLAY.get(variant, variant),
        "case_id_raw": df[case_col].astype(str),
        "case_id": df[case_col].map(lambda x: canonical_case_id(dataset, x)),
        "ARI": pd.to_numeric(df[ari_col], errors="coerce"),
        "NMI": pd.to_numeric(df[nmi_col], errors="coerce"),
        "source_file": str(path),
    })
    out = out[out["case_id"].astype(str).str.len().gt(0) & out["ARI"].notna() & out["NMI"].notna()].copy()
    if out.empty:
        return None

    return out


def read_case_metrics_from_paths(paths: list[Path], dataset: str, variant: str) -> tuple[pd.DataFrame | None, list[str]]:
    parts = []
    used = []
    for path in paths:
        if not path.exists() or path.is_dir() or should_ignore_csv(path):
            continue
        df = parse_case_metric_csv(path, dataset, variant)
        if df is None or df.empty:
            continue
        parts.append(df)
        used.append(str(path))

    if not parts:
        return None, []

    all_df = pd.concat(parts, ignore_index=True)
                                                                          
                                                                
    src_by_case = all_df.groupby(["dataset", "case_id", "variant"], as_index=False)["source_file"].agg(lambda x: ";".join(sorted(set(map(str, x)))))
    mean_df = all_df.groupby(["dataset", "dataset_display", "case_id", "variant", "variant_display"], as_index=False).agg({"ARI": "mean", "NMI": "mean"})
    mean_df = mean_df.merge(src_by_case, on=["dataset", "case_id", "variant"], how="left")
    return mean_df, used


def collect_csvs_under(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        all_csvs = sorted([p for p in path.rglob("*.csv") if p.is_file() and not should_ignore_csv(p)])
        preferred_names = {x.lower() for x in PREFERRED_CASE_CSV_NAMES}
        preferred = [p for p in all_csvs if p.name.lower() in preferred_names]
        if preferred:
                                                                                           
            return preferred
        return all_csvs
    return []


                                                              
                   
                                                              

def discover_dataset_dirs(our_root: Path, dataset_arg: str | None) -> list[tuple[str, Path]]:
    if dataset_arg:
        names = [x.strip() for x in dataset_arg.split(",") if x.strip()]
    else:
        names = DATASET_ORDER

    out = []
    for name in names:
        ds = norm_dataset(name)
        candidates = [
            our_root / ds,
            our_root / name,
        ]
        if ds == "pamap2_zero":
            candidates.extend([our_root / "pamap2", our_root / "pamap2-zero"])
        if ds == "usc-had":
            candidates.extend([our_root / "uschad", our_root / "USC-HAD"])
        if ds == "ucr-seg":
            candidates.extend([our_root / "ucrseg", our_root / "tssb", our_root / "UCR-SEG"])

        hit = None
        for p in candidates:
            if p.exists() and p.is_dir():
                hit = p.resolve()
                break
        if hit is None:
            print(f"[WARN] Dataset folder not found for {name}: tried {candidates}")
            continue
        out.append((ds, hit))
    return out


def formal_ours_candidates(dataset: str, dataset_dir: Path, our_root: Path) -> list[Path]:
    names = [
        f"results_{dataset}_our.csv",
        f"results_{dataset}_ours.csv",
        f"result_{dataset}_our.csv",
        f"results_{dataset}_our",
        f"results_{dataset}_ours",
        f"result_{dataset}_our",
    ]

                                                 
    alias_names = {dataset, dataset.replace("_", "-"), dataset.replace("-", "_")}
    if dataset == "pamap2_zero":
        alias_names.update(["pamap2", "pamap2zero", "pamap2_zero", "PAMAP2_zero"])
                                                                          
                                                                                   
        names.extend([
            "results_pamap2/all_case_results.csv",
            "results_pamap2/case_results.csv",
            "results_pamap2",
            "results_pamap2_zero_fullsensor_remove0/all_case_results.csv",
            "results_pamap2_zero_fullsensor_remove0/case_results.csv",
            "results_pamap2_zero_fullsensor_remove0",
            "results_PAMAP2/all_case_results.csv",
            "results_PAMAP2/case_results.csv",
            "results_PAMAP2",
            "results_PAMAP2_zero_fullsensor_remove0/all_case_results.csv",
            "results_PAMAP2_zero_fullsensor_remove0/case_results.csv",
            "results_PAMAP2_zero_fullsensor_remove0",
        ])
    if dataset == "usc-had":
        alias_names.update(["uschad", "usc_had", "usc-had"])
    if dataset == "ucr-seg":
        alias_names.update(["ucrseg", "ucr_seg", "ucr-seg", "tssb"])

    for a in sorted(alias_names):
        names.extend([
            f"results_{a}_our.csv",
            f"results_{a}_ours.csv",
            f"results_{a}_our",
            f"results_{a}_ours",
        ])

    candidates = []
    for n in names:
        candidates.append(dataset_dir / n)

                                                                                           
    for p in dataset_dir.rglob("*.csv"):
        rel_low = str(p.relative_to(dataset_dir)).lower().replace("/", "\\")
        if any(tok in rel_low for tok in EXCLUDE_PATH_TOKENS_FOR_FORMAL_OURS):
            continue
        stem = p.stem.lower()
        if "our" in stem and "result" in stem:
            candidates.append(p)

                                         
    seen = set()
    uniq = []
    for p in candidates:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def ablation_variant_candidates(dataset_dir: Path, variant: str) -> list[Path]:
    roots = [
        dataset_dir / "_pid_ablation" / "results",
        dataset_dir / "pid_ablation" / "results",
        dataset_dir / "_ablation" / "results",
    ]
    aliases = VARIANT_FOLDER_ALIASES.get(variant, [variant])
    candidates = []
    for root in roots:
        for a in aliases:
            candidates.append(root / a)
            candidates.append(root / f"{a}.csv")
    return candidates


def choose_best_source(paths: list[Path], dataset: str, variant: str) -> tuple[pd.DataFrame | None, list[str], str]:
    """Try path candidates in order. For directories, parse all metric CSVs inside.

    The first candidate that yields valid case-level metrics is used. This avoids
    accidentally mixing repeated exports from several alternative directories.
    """
    for p in paths:
        csvs = collect_csvs_under(p)
        if not csvs:
            continue
        df, used = read_case_metrics_from_paths(csvs, dataset, variant)
        if df is not None and not df.empty:
            return df, used, str(p)
    return None, [], ""


def load_strict_ablation_cases(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    manifest = []

    dataset_dirs = discover_dataset_dirs(args.our_root, args.datasets)
    if not dataset_dirs:
        raise RuntimeError(f"No dataset directories found under {args.our_root}")

    for dataset, dataset_dir in dataset_dirs:
        print("=" * 100)
        print(f"[DATASET] {dataset} -> {dataset_dir}")

        for variant in ABLATION_ORDER:
            if variant == "full_pid" and args.full_pid_source == "formal_ours":
                candidates = formal_ours_candidates(dataset, dataset_dir, args.our_root)
                source_kind = "formal_ours"
            else:
                candidates = ablation_variant_candidates(dataset_dir, variant)
                source_kind = "ablation_folder"

            df, used, chosen = choose_best_source(candidates, dataset, variant)
            if df is None or df.empty:
                print(f"[MISS] {dataset:12s} {variant:24s} source={source_kind}")
                manifest.append({
                    "dataset": dataset,
                    "variant": variant,
                    "source_kind": source_kind,
                    "status": "missing",
                    "chosen_root": "",
                    "n_cases": 0,
                    "used_files": "",
                })
                continue

            rows.append(df)
            n_cases = int(df["case_id"].nunique())
            print(f"[OK]   {dataset:12s} {variant:24s} n={n_cases:4d} source={source_kind}")
            print(f"       root: {chosen}")
            for u in used[:8]:
                print(f"       csv : {u}")
            if len(used) > 8:
                print(f"       ... {len(used) - 8} more csv files")

            manifest.append({
                "dataset": dataset,
                "variant": variant,
                "source_kind": source_kind,
                "status": "ok",
                "chosen_root": chosen,
                "n_cases": n_cases,
                "used_files": ";".join(used),
            })

        if args.full_pid_source == "formal_ours":
                                                                                                   
            ab_full_candidates = ablation_variant_candidates(dataset_dir, "full_pid")
            existing = [str(p) for p in ab_full_candidates if p.exists()]
            if existing:
                print(f"[INFO] {dataset} ablation full_pid folder exists but is ignored by default:")
                for p in existing:
                    print(f"       ignored: {p}")
                print("       reason : full_pid is required to come from formal Ours. Use --full-pid-source ablation_folder only for diagnostics.")

    if not rows:
        raise RuntimeError("No ablation case metrics were loaded. Check directory paths and CSV columns.")

    cases = pd.concat(rows, ignore_index=True)
    cases["variant"] = pd.Categorical(cases["variant"].astype(str), categories=ABLATION_ORDER, ordered=True)
    cases = cases.sort_values(["dataset", "variant", "case_id"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest)
    return cases, manifest_df


                                                              
           
                                                              

def safe_mean(values: Iterable[float]) -> float:
    arr = [float(x) for x in values if pd.notna(x) and np.isfinite(float(x))]
    return float(np.mean(arr)) if arr else math.nan


def fmt4(x: float) -> str:
    return "" if pd.isna(x) or not np.isfinite(float(x)) else f"{float(x):.4f}"


def make_ablation_summary(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.groupby(["dataset", "dataset_display", "variant", "variant_display"], observed=True).agg(
        cases=("case_id", "nunique"),
        ARI_mean=("ARI", "mean"),
        NMI_mean=("NMI", "mean"),
        ARI_std=("ARI", "std"),
        NMI_std=("NMI", "std"),
    ).reset_index()
    out["variant"] = pd.Categorical(out["variant"].astype(str), categories=ABLATION_ORDER, ordered=True)
    dataset_order = {d: i for i, d in enumerate(DATASET_ORDER)}
    out["dataset_order"] = out["dataset"].map(lambda x: dataset_order.get(str(x), 99))
    out = out.sort_values(["dataset_order", "variant"]).drop(columns=["dataset_order"])
    return out


def make_dataset_avg(summary: pd.DataFrame) -> pd.DataFrame:
    variants_by_dataset = summary.groupby("dataset", observed=True)["variant"].agg(lambda x: set(map(str, x))).to_dict()
    complete_datasets = [d for d in DATASET_ORDER if variants_by_dataset.get(d, set()) >= set(ABLATION_ORDER)]

    rows = []
    for variant in ABLATION_ORDER:
        sub = summary[(summary["variant"].astype(str) == variant) & (summary["dataset"].isin(complete_datasets))]
        rows.append({
            "Variant": variant,
            "Display": VARIANT_DISPLAY.get(variant, variant),
            "Dataset count": int(len(complete_datasets)),
            "Datasets used": ", ".join(display_dataset(d) for d in complete_datasets),
            "Dataset-Avg ARI": fmt4(safe_mean(sub["ARI_mean"])),
            "Dataset-Avg NMI": fmt4(safe_mean(sub["NMI_mean"])),
        })
    avg = pd.DataFrame(rows)
    avg["Rank"] = avg["Dataset-Avg ARI"].replace("", np.nan).astype(float).rank(ascending=False, method="min").astype("Int64")
    cols = ["Rank", "Variant", "Display", "Dataset count", "Datasets used", "Dataset-Avg ARI", "Dataset-Avg NMI"]
    return avg[cols].sort_values(["Rank", "Variant"])


                                                              
                      
                                                              

def complete_case_matrix(cases: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = cases.pivot_table(index=["dataset", "case_id"], columns="variant", values=metric, aggfunc="mean")
    cols = [v for v in ABLATION_ORDER if v in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def average_ranks(pivot: pd.DataFrame) -> pd.DataFrame:
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean(axis=0).sort_values(ascending=True)
    return pd.DataFrame({"label": avg.index.astype(str), "avg_rank": avg.values})


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    if k < 2 or n < 1:
        return math.nan
    if studentized_range is not None:
        q_alpha = float(studentized_range.ppf(1 - alpha, k, math.inf) / math.sqrt(2.0))
    else:
        table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q_alpha = table.get(k, 3.164)
    return q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))


def cd_intervals(rank_df: pd.DataFrame, cd: float) -> list[tuple[float, float]]:
    vals = rank_df["avg_rank"].to_numpy(dtype=float)
    intervals = []
    n = len(vals)
    for i in range(n):
        j = i
        while j + 1 < n and vals[j + 1] - vals[i] <= cd + 1e-12:
            j += 1
        if j > i:
            intervals.append((i, j))
    maximal = []
    for a, b in intervals:
        if not any(c <= a and b <= d and (c, d) != (a, b) for c, d in intervals):
            maximal.append((a, b))
    return [(float(vals[a]), float(vals[b])) for a, b in maximal]


def draw_cd_diagram(rank_df: pd.DataFrame, cd: float, metric: str, out_png: Path) -> None:
    rank_df = rank_df.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    k = len(rank_df)
    if k < 2:
        return

    min_rank, max_rank = 1.0, float(k)
    fig_w = max(10, 1.2 * k + 4.5)
    fig_h = max(5.0, 0.55 * k + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.set_xlim(max_rank + 0.35, min_rank - 0.35)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_axis = 0.68
    ax.hlines(y_axis, min_rank, max_rank, color="black", linewidth=1.6)
    for r in range(1, k + 1):
        ax.vlines(r, y_axis - 0.035, y_axis + 0.035, color="black", linewidth=1.4)
        ax.text(r, y_axis + 0.055, str(r), ha="center", va="bottom", fontsize=12)

    cd_y = 0.90
    cd_start = max_rank
    cd_end = max(max_rank - cd, min_rank)
    ax.hlines(cd_y, cd_start, cd_end, color="black", linewidth=2.0)
    ax.vlines([cd_start, cd_end], cd_y - 0.03, cd_y + 0.03, color="black", linewidth=1.5)
    ax.text((cd_start + cd_end) / 2, cd_y + 0.045, f"CD={cd:.3f}", ha="center", va="bottom", fontsize=13)

    left_items = rank_df[rank_df["avg_rank"] > (k + 1) / 2].sort_values("avg_rank", ascending=False).reset_index(drop=True)
    right_items = rank_df[rank_df["avg_rank"] <= (k + 1) / 2].sort_values("avg_rank", ascending=True).reset_index(drop=True)
    left_x_text = max_rank + 0.28
    right_x_text = min_rank - 0.28
    y0 = 0.48
    step = 0.085

    for i, row in left_items.iterrows():
        y = y0 - i * step
        x = float(row["avg_rank"])
        label = VARIANT_DISPLAY.get(str(row["label"]), str(row["label"]))
        ax.plot([x, left_x_text - 0.05], [y_axis, y], color="black", linewidth=1.1)
        ax.text(left_x_text, y, f"{label} {x:.3f}", ha="right", va="center", fontsize=10)

    for i, row in right_items.iterrows():
        y = y0 - i * step
        x = float(row["avg_rank"])
        label = VARIANT_DISPLAY.get(str(row["label"]), str(row["label"]))
        ax.plot([x, right_x_text + 0.05], [y_axis, y], color="black", linewidth=1.1)
        ax.text(right_x_text, y, f"{x:.3f} {label}", ha="left", va="center", fontsize=10)

    base_y = y_axis - 0.115
    for idx, (a, b) in enumerate(cd_intervals(rank_df, cd)[:8]):
        y = base_y - idx * 0.035
        ax.hlines(y, a, b, color="black", linewidth=3.0)

    ax.text((min_rank + max_rank) / 2, 0.05, "Rank (smaller is better)", ha="center", va="center", fontsize=12)
    ax.set_title(f"Critical Difference Diagram ({metric})", fontsize=16, fontweight="bold", pad=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_rank_outputs(cases: pd.DataFrame, out_dir: Path, alpha: float) -> None:
    for metric in METRICS:
        pivot = complete_case_matrix(cases, metric)
        if pivot.empty or pivot.shape[1] < 2:
            print(f"[WARN] Rank/CD skipped for {metric}: not enough complete aligned cases.")
            continue
        ranks = average_ranks(pivot)
        n_cases = int(pivot.shape[0])
        k_methods = int(pivot.shape[1])
        cd = nemenyi_cd(k_methods, n_cases, alpha=alpha)
        ranks.insert(0, "mode", "ablation")
        ranks.insert(1, "metric", metric)
        ranks.insert(2, "complete_aligned_cases", n_cases)
        ranks.insert(3, "n_methods", k_methods)
        ranks.insert(4, f"CD_alpha_{alpha}", cd)
        ranks.to_csv(out_dir / f"ablation_rank_{metric}.csv", index=False, encoding="utf-8-sig")
        draw_cd_diagram(ranks[["label", "avg_rank"]].copy(), cd, metric, out_dir / f"ablation_cd_diagram_{metric}.png")
        print(f"[OK] Rank/CD {metric}: n={n_cases}, k={k_methods}, CD={cd:.3f}")


                                                              
                          
                                                              

def safe_paired_ttest(a: pd.Series, b: pd.Series, min_n: int) -> tuple[float, int, str]:
    pair = pd.concat([a, b], axis=1).dropna()
    n = int(len(pair))
    if n < min_n:
        return math.nan, n, "too_few_common_cases"
    diff = pair.iloc[:, 0].to_numpy(dtype=float) - pair.iloc[:, 1].to_numpy(dtype=float)
    if np.allclose(diff, 0):
        return 1.0, n, "all_equal"
    if ttest_rel is None:
        return math.nan, n, "scipy_missing"
    stat, p = ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1], nan_policy="omit")
    if not np.isfinite(p):
        return math.nan, n, "nan_pvalue"
    return float(p), n, "ok"


def compact_letters(means: pd.Series, sig: pd.DataFrame) -> dict[str, str]:
    labels = means.sort_values(ascending=False).index.astype(str).tolist()
    groups: list[list[str]] = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    def can_join(group: list[str], label: str) -> bool:
        return all(not bool(sig.loc[label, other]) for other in group if other in sig.columns)

    for label in labels:
        placed = False
        for group in groups:
            if can_join(group, label):
                group.append(label)
                placed = True
        if not placed:
            groups.append([label])

    out = {label: "" for label in labels}
    for i, group in enumerate(groups):
        letter = alphabet[i] if i < len(alphabet) else f"L{i+1}"
        for label in group:
            out[label] += letter
    return out


def pairwise_tests(cases: pd.DataFrame, dataset: str, metric: str, alpha: float, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = cases[cases["dataset"].astype(str).eq(dataset)].copy()
    pivot = sub.pivot_table(index="case_id", columns="variant", values=metric, aggfunc="mean")
    labels = [v for v in ABLATION_ORDER if v in pivot.columns]
    sig = pd.DataFrame(False, index=labels, columns=labels)
    rows = []
    for a, b in combinations(labels, 2):
        p, n, status = safe_paired_ttest(pivot[a], pivot[b], min_n=min_n)
        significant = bool(np.isfinite(p) and p < alpha)
        sig.loc[a, b] = sig.loc[b, a] = significant
        rows.append({
            "dataset": dataset,
            "dataset_display": display_dataset(dataset),
            "metric": metric,
            "method_a": a,
            "method_b": b,
            "n_common_cases": n,
            "mean_a": float(pivot[a].dropna().mean()) if a in pivot else math.nan,
            "mean_b": float(pivot[b].dropna().mean()) if b in pivot else math.nan,
            "p_value": p,
            "alpha": alpha,
            "significant": significant if status in {"ok", "all_equal"} else "NA",
            "status": status,
        })
    return pd.DataFrame(rows), sig


def make_letters_outputs(cases: pd.DataFrame, out_dir: Path, alpha: float, min_n: int) -> None:
    for metric in METRICS:
        pairwise_rows = []
        letter_rows = []
        datasets = [d for d in DATASET_ORDER if d in set(cases["dataset"].astype(str))]
        datasets += sorted(set(cases["dataset"].astype(str)) - set(datasets))

        for dataset in datasets:
            pairwise, sig = pairwise_tests(cases, dataset, metric, alpha, min_n)
            if not pairwise.empty:
                pairwise_rows.append(pairwise)
            sub = cases[cases["dataset"].astype(str).eq(dataset)]
            means = sub.groupby("variant", observed=False)[metric].mean().reindex([v for v in ABLATION_ORDER if v in set(sub["variant"].astype(str))])
            letters = compact_letters(means.dropna(), sig)
            n_by = sub.groupby("variant", observed=False)["case_id"].nunique()
            for variant, mean_val in means.dropna().items():
                letter_rows.append({
                    "mode": "ablation",
                    "dataset": dataset,
                    "dataset_display": display_dataset(dataset),
                    "metric": metric,
                    "label": variant,
                    "display": VARIANT_DISPLAY.get(str(variant), str(variant)),
                    "mean": float(mean_val),
                    "n_cases": int(n_by.get(variant, 0)),
                    "letters": letters.get(str(variant), ""),
                })

        letters_df = pd.DataFrame(letter_rows)
        pairwise_df = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
        letters_df.to_csv(out_dir / f"ablation_ttest_letters_{metric}.csv", index=False, encoding="utf-8-sig")
        pairwise_df.to_csv(out_dir / f"ablation_pairwise_ttest_{metric}.csv", index=False, encoding="utf-8-sig")
        plot_t2s_letters(letters_df, metric, out_dir / f"ablation_t2s_style_ab_{metric}.png")
        print(f"[OK] T-test letters {metric}")


def plot_t2s_letters(letter_df: pd.DataFrame, metric: str, out_png: Path) -> None:
    if letter_df.empty:
        return
    datasets = [d for d in DATASET_ORDER if d in set(letter_df["dataset"].astype(str))]
    datasets += sorted(set(letter_df["dataset"].astype(str)) - set(datasets))
    labels = [v for v in ABLATION_ORDER if v in set(letter_df["label"].astype(str))]

    x = np.arange(len(datasets))
    width = min(0.14, 0.80 / max(1, len(labels)))
    fig_w = max(12, 1.45 * len(datasets) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=180)

    for i, label in enumerate(labels):
        vals = []
        letters = []
        for d in datasets:
            row = letter_df[(letter_df["dataset"].astype(str).eq(d)) & (letter_df["label"].astype(str).eq(label))]
            if row.empty:
                vals.append(np.nan)
                letters.append("")
            else:
                vals.append(float(row.iloc[0]["mean"]))
                letters.append(str(row.iloc[0]["letters"]))
        offset = (i - (len(labels) - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width=width, label=VARIANT_DISPLAY.get(label, label))
        for bar, txt in zip(bars, letters):
            h = bar.get_height()
            if np.isfinite(h) and txt:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, txt, ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Time2State-style paired t-test markers ({metric})", fontsize=16, fontweight="bold", pad=10)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([display_dataset(d) for d in datasets], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(labels), 3), frameon=False, fontsize=8)
    ax.text(
        0.01,
        -0.20,
        "Same letter = not significantly different; no shared letter = significant difference (paired t-test, alpha=0.05).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



                                                              
                                                                 
                                                              

T2S_DISPLAY = "Time2State"
FINAL_METHOD_ORDER = [
    "full_pid",
    "pid_select_uniform",
    "all_branches_uniform",
    "peer_select_pid_weight",
    "no_pid_peer",
    "Time2State",
]

                                                                 
                                                                                  
T2S_DEFAULT_ROOT = DEFAULT_BASELINE_ROOT / "t2s"


def _to_float_or_nan(x: object) -> float:
    try:
        if x is None or str(x).strip() == "":
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def _source_status_ok(row: pd.Series) -> bool:
    status = str(row.get("status", row.get("Status", "ok"))).strip().lower()
    return status in {"", "ok", "success", "done"}


def _protocol_priority_for_t2s(row: pd.Series) -> int:
    """Prefer formal Time2State rows when several T2S summaries exist."""
    protocol = str(row.get("protocol", row.get("Protocol", ""))).strip()
    source = str(row.get("source_file", row.get("Source", ""))).lower()
    if protocol == "T2S-E2U-same-param":
        return 0
    if protocol in {"T2S-fixed-param-from-E2USD-code", "Time2State-paper-fixed-param"}:
        return 1
    if protocol == "T2S-grid-mean":
        return 2
    if "diagnostic" in protocol.lower() or "_pid_ablation" in source or "s14_gate" in source:
        return 99
    return 10


def find_t2s_summary_table(args: argparse.Namespace, out_dir: Path) -> Path | None:
    """Find a table that contains dataset-level Time2State ARI/NMI means.

    The earlier version only checked a few fixed files under the report folder.
    In practice, the clean main table may be generated in a sibling/older report
    folder, so this version first checks explicit candidates and then performs a
    bounded recursive search under our/_ari_nmi_summary.
    """
    candidates: list[Path] = []
    if getattr(args, "t2s_table", None) is not None:
        candidates.append(Path(args.t2s_table))

    report_dir = out_dir.parent
    summary_dir = args.our_root / "_ari_nmi_summary"
    candidates.extend([
        report_dir / "main_results_table_clean3.csv",
        report_dir / "main_results_table.csv",
        report_dir / "all_methods_main_table.csv",
        summary_dir / "report" / "main_results_table_clean3.csv",
        summary_dir / "report" / "main_results_table.csv",
        summary_dir / "report" / "all_methods_main_table.csv",
        summary_dir / "main_results_table_clean3.csv",
        summary_dir / "main_results_table.csv",
        summary_dir / "all_methods_main_table.csv",
    ])

                                                                 
                                                                    
    patterns = [
        "main_results_table_clean*.csv",
        "main_results_table*.csv",
        "all_methods_main_table*.csv",
        "*main*result*.csv",
        "*all*method*.csv",
    ]
    if summary_dir.exists():
        for pat in patterns:
            candidates.extend(sorted(summary_dir.rglob(pat)))

    seen = set()
    for p in candidates:
        if p is None:
            continue
        p = Path(p).resolve()
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        if not (p.exists() and p.is_file()):
            continue

                                                                              
        try:
            raw = read_csv_any(p)
            method_col = find_col(raw, ["method", "Method", "label", "Label"])
            dataset_col = find_col(raw, ["dataset", "Dataset"])
            ari_col = find_col(raw, ["ARI_mean", "Dataset-Avg ARI", "ARI"])
            nmi_col = find_col(raw, ["NMI_mean", "Dataset-Avg NMI", "NMI"])
            if method_col and dataset_col and ari_col and nmi_col:
                hit = raw[method_col].astype(str).str.strip().str.lower().eq("time2state").any()
                if hit:
                    print(f"[INFO] T2S summary table found: {p}")
                    return p
        except Exception:
            continue
    return None



def infer_dataset_from_path(path: Path) -> str | None:
    """Infer benchmark dataset name from a result-file path."""
    aliases = {
        "synthetic": "synthetic",
        "synth": "synthetic",
        "mocap": "mocap",
        "actrectut": "actrectut",
        "act-rec-tut": "actrectut",
        "act_rec_tut": "actrectut",
        "pamap2": "pamap2_zero",
        "pamap2zero": "pamap2_zero",
        "pamap2_zero": "pamap2_zero",
        "pamap2-zero": "pamap2_zero",
        "ucrseg": "ucr-seg",
        "ucr_seg": "ucr-seg",
        "ucr-seg": "ucr-seg",
        "tssb": "ucr-seg",
        "uschad": "usc-had",
        "usc_had": "usc-had",
        "usc-had": "usc-had",
    }
    parts = [str(x).strip().lower() for x in path.parts]
                                                         
    for part in reversed(parts):
        clean = part.replace(" ", "").replace("-", "_")
        if clean in aliases:
            return aliases[clean]
        clean2 = clean.replace("_", "-")
        if clean2 in aliases:
            return aliases[clean2]
        for key, value in aliases.items():
            k1 = key.replace("-", "_")
            k2 = key.replace("_", "-")
            if k1 in clean or k2 in clean:
                return value
    return None


def _csv_priority_for_t2s_file(path: Path) -> int:
    """Prefer final case-level T2S result files over summaries/branch files."""
    name = path.name.lower()
    full = str(path).lower().replace("/", "\\")
    if any(tok in full for tok in ["_pid_ablation", "ablationresult", "ablation_result"]):
        return 99
    if any(tok in name for tok in ["all_branch", "branch_level", "case_branch", "manifest", "rank", "ttest", "letter"]):
        return 99
    preferred = {x.lower() for x in PREFERRED_CASE_CSV_NAMES}
    if name in preferred:
        return 0
    if name in {"dataset_summary.csv", "summary.csv"}:
        return 1
    if "case" in name and "result" in name:
        return 2
    if "result" in name:
        return 3
    return 10


def _parse_t2s_table_candidate(path: Path) -> list[dict]:
    """Parse one possible T2S result CSV into dataset-level mean rows."""
    try:
        raw = read_csv_any(path)
    except Exception:
        return []
    if raw.empty:
        return []

    dataset_col = find_col(raw, ["dataset", "Dataset", "data", "benchmark"])
    case_col = find_case_col(raw)
    ari_col = find_col(raw, ["ARI_mean", "ari_mean", "Dataset-Avg ARI", "ARI", "ari", "adjusted_rand_score", "adjusted_rand_index"])
    nmi_col = find_col(raw, ["NMI_mean", "nmi_mean", "Dataset-Avg NMI", "NMI", "nmi", "normalized_mutual_info", "normalized_mutual_information"])
    if ari_col is None or nmi_col is None:
        return []

    df = status_ok_filter(raw.copy())
    df["_ARI"] = pd.to_numeric(df[ari_col], errors="coerce")
    df["_NMI"] = pd.to_numeric(df[nmi_col], errors="coerce")
    df = df[df["_ARI"].notna() & df["_NMI"].notna()].copy()
    if df.empty:
        return []

    if dataset_col is not None:
        df["_dataset"] = df[dataset_col].map(norm_dataset)
    else:
        ds = infer_dataset_from_path(path)
        if ds is None:
            return []
        df["_dataset"] = ds

    df = df[df["_dataset"].isin(DATASET_ORDER)].copy()
    if df.empty:
        return []

    rows: list[dict] = []
    for ds, g in df.groupby("_dataset", observed=True):
        if case_col is not None:
            n_cases = int(g[case_col].map(lambda x: canonical_case_id(ds, x)).nunique())
        else:
            n_col = find_col(g, ["cases", "case_count", "n_cases", "num_cases"])
            if n_col is not None:
                n_cases = int(pd.to_numeric(g[n_col], errors="coerce").dropna().max()) if pd.to_numeric(g[n_col], errors="coerce").notna().any() else int(len(g))
            else:
                n_cases = int(len(g))
        rows.append({
            "dataset": ds,
            "dataset_display": display_dataset(ds),
            "method": "Time2State",
            "ARI_mean": float(g["_ARI"].mean()),
            "NMI_mean": float(g["_NMI"].mean()),
            "n_cases": n_cases,
            "source_file": str(path),
            "_priority": _csv_priority_for_t2s_file(path),
        })
    return rows


def scan_t2s_root_dataset_means(t2s_root: Path) -> pd.DataFrame | None:
    """Scan D:\\code\\baseline\\t2s-style result folders for Time2State means."""
    t2s_root = Path(t2s_root).resolve()
    if not t2s_root.exists():
        print(f"[WARN] T2S root does not exist: {t2s_root}")
        return None

    rows: list[dict] = []
    csvs = sorted(p for p in t2s_root.rglob("*.csv") if p.is_file() and _csv_priority_for_t2s_file(p) < 99)
    for p in csvs:
        rows.extend(_parse_t2s_table_candidate(p))

    if not rows:
        print(f"[WARN] No usable Time2State CSV was parsed under: {t2s_root}")
        return None

    df = pd.DataFrame(rows)
                                                                                  
    df = df.sort_values(["dataset", "_priority", "n_cases"], ascending=[True, True, False])
    best = df.groupby("dataset", as_index=False).first()
    best = best[best["dataset"].isin(DATASET_ORDER)].copy()
    if best.empty:
        return None
    best = best[["dataset", "dataset_display", "method", "ARI_mean", "NMI_mean", "source_file", "n_cases"]]
    print(f"[INFO] T2S result root scanned: {t2s_root}")
    for _, r in best.sort_values("dataset").iterrows():
        print(f"[T2S]  {str(r['dataset']):12s} n={int(r.get('n_cases', 0)):4d} ARI={float(r['ARI_mean']):.4f} NMI={float(r['NMI_mean']):.4f}")
        print(f"       source: {r['source_file']}")
    return best.drop(columns=["n_cases"])


def load_t2s_dataset_means(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    """Load Time2State dataset-level ARI/NMI means.

    Default source is main_results_table_clean3.csv under the report folder.
    If that file is unavailable, pass --t2s-ari and --t2s-nmi manually.
    """
    manual_ari = getattr(args, "t2s_ari", None)
    manual_nmi = getattr(args, "t2s_nmi", None)
    if manual_ari is not None and manual_nmi is not None:
        return pd.DataFrame([{
            "dataset": "Dataset-Avg",
            "dataset_display": "Dataset-Avg",
            "method": "Time2State",
            "ARI_mean": float(manual_ari),
            "NMI_mean": float(manual_nmi),
            "source_file": "manual_cli",
        }])

    table_path = find_t2s_summary_table(args, out_dir)
    if table_path is None:
        t2s_root = Path(getattr(args, "t2s_root", T2S_DEFAULT_ROOT) or T2S_DEFAULT_ROOT)
        scanned = scan_t2s_root_dataset_means(t2s_root)
        if scanned is not None and not scanned.empty:
            return scanned
        raise FileNotFoundError(
            "Cannot find a usable Time2State summary table containing method == 'Time2State', "
            "and no usable Time2State case-level result files were found under --t2s-root. "
            "You can pass --t2s-root D:\\code\\baseline\\t2s, "
            "or pass --t2s-table <main_results_table.csv>, "
            "or pass --t2s-ari <value> --t2s-nmi <value>."
        )

    raw = read_csv_any(table_path)
    method_col = find_col(raw, ["method", "Method", "label", "Label"])
    dataset_col = find_col(raw, ["dataset", "Dataset"])
    ari_col = find_col(raw, ["ARI_mean", "Dataset-Avg ARI", "ARI"])
    nmi_col = find_col(raw, ["NMI_mean", "Dataset-Avg NMI", "NMI"])

    if method_col is None or dataset_col is None or ari_col is None or nmi_col is None:
        raise ValueError(
            f"{table_path} must contain method, dataset, ARI_mean/ARI, and NMI_mean/NMI columns. "
            f"Current columns: {list(raw.columns)}"
        )

    df = raw.copy()
    df = df[df[method_col].astype(str).str.strip().str.lower().eq("time2state")].copy()
    df = df[df.apply(_source_status_ok, axis=1)].copy()
    if df.empty:
        raise RuntimeError(f"No method == 'Time2State' rows found in {table_path}")

    df["dataset"] = df[dataset_col].map(norm_dataset)
    df["dataset_display"] = df["dataset"].map(display_dataset)
    df["ARI_mean"] = pd.to_numeric(df[ari_col], errors="coerce")
    df["NMI_mean"] = pd.to_numeric(df[nmi_col], errors="coerce")
    df = df[df["ARI_mean"].notna() & df["NMI_mean"].notna()].copy()
    if df.empty:
        raise RuntimeError(f"Time2State rows in {table_path} have no valid ARI/NMI values.")

                                                                                  
    df["_priority"] = df.apply(_protocol_priority_for_t2s, axis=1)
    df["_source_table"] = str(table_path)
    df = df.sort_values(["dataset", "_priority"]).groupby("dataset", as_index=False).first()

    out = df[["dataset", "dataset_display", "ARI_mean", "NMI_mean", "_source_table"]].copy()
    out = out.rename(columns={"_source_table": "source_file"})
    out.insert(2, "method", "Time2State")
    return out


def complete_datasets_from_summary(summary: pd.DataFrame) -> list[str]:
    variants_by_dataset = summary.groupby("dataset", observed=True)["variant"].agg(lambda x: set(map(str, x))).to_dict()
    return [d for d in DATASET_ORDER if variants_by_dataset.get(d, set()) >= set(ABLATION_ORDER)]


def make_compact_vs_t2s_table(summary: pd.DataFrame, t2s_df: pd.DataFrame) -> pd.DataFrame:
    """Create the only CSV table: dataset-average ARI/NMI and gains vs T2S."""
    complete_datasets = complete_datasets_from_summary(summary)
    if not complete_datasets:
        raise RuntimeError("No dataset contains all ablation variants; cannot compute dataset-average table.")

    t2s_by_dataset = t2s_df[t2s_df["dataset"].isin(complete_datasets)].copy()
    if len(set(t2s_by_dataset["dataset"])) < len(complete_datasets):
        missing = sorted(set(complete_datasets) - set(t2s_by_dataset["dataset"]))
        raise RuntimeError(
            "Time2State summary is missing required dataset(s): "
            + ", ".join(display_dataset(d) for d in missing)
        )

    t2s_ari = safe_mean(t2s_by_dataset["ARI_mean"])
    t2s_nmi = safe_mean(t2s_by_dataset["NMI_mean"])

    rows: list[dict] = []
    for variant in ABLATION_ORDER:
        sub = summary[(summary["variant"].astype(str).eq(variant)) & summary["dataset"].isin(complete_datasets)].copy()
        ari = safe_mean(sub["ARI_mean"])
        nmi = safe_mean(sub["NMI_mean"])
        rows.append({
            "Method": VARIANT_DISPLAY.get(variant, variant),
            "Variant": variant,
            "Avg ARI": ari,
            "Avg NMI": nmi,
            "ARI gain vs T2S (%)": ((ari - t2s_ari) / t2s_ari * 100.0) if t2s_ari not in {0, math.nan} and np.isfinite(t2s_ari) else math.nan,
            "NMI gain vs T2S (%)": ((nmi - t2s_nmi) / t2s_nmi * 100.0) if t2s_nmi not in {0, math.nan} and np.isfinite(t2s_nmi) else math.nan,
            "Datasets used": ", ".join(display_dataset(d) for d in complete_datasets),
        })

                                            
    rows.append({
        "Method": T2S_DISPLAY,
        "Variant": "Time2State",
        "Avg ARI": t2s_ari,
        "Avg NMI": t2s_nmi,
        "ARI gain vs T2S (%)": 0.0,
        "NMI gain vs T2S (%)": 0.0,
        "Datasets used": ", ".join(display_dataset(d) for d in complete_datasets),
    })

    out = pd.DataFrame(rows)
    for col in ["Avg ARI", "Avg NMI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    for col in ["ARI gain vs T2S (%)", "NMI gain vs T2S (%)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out


def draw_gain_vs_t2s_plot(table: pd.DataFrame, out_png: Path) -> None:
    """Draw relative improvements over T2S; T2S is the bottom zero-gain row."""
    plot_df = table.copy()
    plot_df["ARI_gain"] = pd.to_numeric(plot_df["ARI gain vs T2S (%)"], errors="coerce")
    plot_df["NMI_gain"] = pd.to_numeric(plot_df["NMI gain vs T2S (%)"], errors="coerce")
    plot_df = plot_df[plot_df["ARI_gain"].notna() & plot_df["NMI_gain"].notna()].copy()

    if plot_df.empty:
        raise RuntimeError("No valid gain values to plot.")

                                                                                               
    labels = plot_df["Method"].astype(str).tolist()
    y = np.arange(len(labels))
    height = 0.36

    min_gain = float(np.nanmin([plot_df["ARI_gain"].min(), plot_df["NMI_gain"].min(), 0.0]))
    max_gain = float(np.nanmax([plot_df["ARI_gain"].max(), plot_df["NMI_gain"].max(), 0.0]))
    pad = max(2.0, (max_gain - min_gain) * 0.12)

    fig_h = max(4.8, 0.58 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=260)

    bars_ari = ax.barh(y - height / 2, plot_df["ARI_gain"].to_numpy(dtype=float), height=height, label="ARI gain")
    bars_nmi = ax.barh(y + height / 2, plot_df["NMI_gain"].to_numpy(dtype=float), height=height, label="NMI gain")

    ax.axvline(0, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Relative gain over Time2State (%)")
    ax.set_title("Ablation Gains over the Time2State Backbone", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(min_gain - pad, max_gain + pad)
    ax.grid(axis="x", alpha=0.28)
    ax.legend(frameon=False, loc="lower right")

    for bars in (bars_ari, bars_nmi):
        for bar in bars:
            w = float(bar.get_width())
            if not np.isfinite(w):
                continue
            offset = 0.45 if w >= 0 else -0.45
            ha = "left" if w >= 0 else "right"
            ax.text(
                w + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{w:+.2f}%",
                va="center",
                ha=ha,
                fontsize=8,
            )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


                                                              
         
                                                              

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a compact ablation-vs-T2S table and gain plot.")
    ap.add_argument("--root", type=Path, default=DEFAULT_TEACHERT2S_ROOT)
    ap.add_argument("--our-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names. Default uses built-in dataset order.")
    ap.add_argument("--full-pid-source", choices=["formal_ours", "ablation_folder"], default="formal_ours")
    ap.add_argument("--t2s-table", type=Path, default=None, help="Optional table containing Time2State dataset-level ARI/NMI means.")
    ap.add_argument("--t2s-root", type=Path, default=T2S_DEFAULT_ROOT, help="Folder containing standalone Time2State result files.")
    ap.add_argument("--t2s-ari", type=float, default=None, help="Manual Time2State dataset-average ARI, used only if --t2s-table is unavailable.")
    ap.add_argument("--t2s-nmi", type=float, default=None, help="Manual Time2State dataset-average NMI, used only if --t2s-table is unavailable.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    args.our_root = (args.our_root or (root / "our")).resolve()
    out_dir = (args.out_dir or (args.our_root / "_ari_nmi_summary" / "report" / "ablationResult")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = out_dir / "ablation_vs_t2s_mean_table.csv"
    plot_path = out_dir / "ablation_vs_t2s_gain_plot.png"

    print("============================================================")
    print("Compact ablation-vs-T2S builder")
    print("root            :", root)
    print("our_root        :", args.our_root)
    print("out_dir         :", out_dir)
    print("datasets        :", args.datasets or ",".join(DATASET_ORDER))
    print("full_pid_source :", args.full_pid_source)
    print("t2s_table       :", args.t2s_table or "auto")
    print("t2s_root        :", args.t2s_root)
    print("output table    :", table_path)
    print("output plot     :", plot_path)
    print("============================================================")

    cases, _manifest = load_strict_ablation_cases(args)
    summary = make_ablation_summary(cases)
    t2s_df = load_t2s_dataset_means(args, out_dir)

    final_table = make_compact_vs_t2s_table(summary, t2s_df)
    final_table.to_csv(table_path, index=False, encoding="utf-8-sig")
    draw_gain_vs_t2s_plot(final_table, plot_path)

    print("\n[WRITE]")
    print(" ", table_path)
    print(" ", plot_path)
    print("\n[DONE] Only two output files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
