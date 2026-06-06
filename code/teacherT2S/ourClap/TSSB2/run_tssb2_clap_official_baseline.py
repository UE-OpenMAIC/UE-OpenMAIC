from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official CLaP default baseline on bundled CLaP benchmark datasets.")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--clap-repo", type=Path, required=True)
    p.add_argument("--dataset-name", required=True, help="UTSA, SKAB, MIT-BIH, or HAS")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    p.add_argument("--case-names", default="", help="Comma-separated dataset/case names to run. Empty means all cases.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--compute-clasp-if-missing", action="store_true", help="If official segmentation is missing, compute ClaSP init cps.")
    return p.parse_args()


def add_clap_repo(clap_repo: Path) -> None:
    clap_repo = clap_repo.resolve()
    if not (clap_repo / "src" / "clap.py").exists():
        raise FileNotFoundError(f"Cannot find CLaP source: {clap_repo / 'src' / 'clap.py'}")
    if str(clap_repo) in sys.path:
        sys.path.remove(str(clap_repo))
    sys.path.insert(0, str(clap_repo))


def safe_int_array(x) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=int)
    arr = np.asarray(x)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    return arr.astype(int).reshape(-1)


def safe_ts_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_cps(cps: Iterable[int], n: int) -> np.ndarray:
    out = []
    for v in list(cps):
        try:
            x = int(v)
        except Exception:
            continue
        if 0 < x < n:
            out.append(x)
    return np.asarray(sorted(set(out)), dtype=int)


def reorder_labels(seq: np.ndarray) -> np.ndarray:
    mapping = {}
    out = []
    nxt = 0
    for v in list(seq):
        k = int(v)
        if k not in mapping:
            mapping[k] = nxt
            nxt += 1
        out.append(mapping[k])
    return np.asarray(out, dtype=int)


def labels_from_cps(cps, labels, ts_len: int) -> np.ndarray:
    cps = sanitize_cps(cps, ts_len)
    labels = [int(x) for x in list(labels)]
    n_segments = len(cps) + 1
    if len(labels) < n_segments:
        labels = labels + ([labels[-1]] * (n_segments - len(labels)) if labels else [0] * n_segments)
    elif len(labels) > n_segments:
        labels = labels[:n_segments]
    y = np.zeros(ts_len, dtype=int)
    start = 0
    for i, end in enumerate(list(cps) + [ts_len]):
        y[start:end] = int(labels[i])
        start = end
    return reorder_labels(y)


def true_labels_to_point_labels(cps_true, labels, ts_len: int, create_state_labels_func) -> np.ndarray:
    labels = safe_int_array(labels)
    if len(labels) == ts_len:
        return reorder_labels(labels)
    return reorder_labels(np.asarray(create_state_labels_func(cps_true, labels, ts_len), dtype=int))


def segmentation_covering(cps_true, cps_pred, n: int) -> float:
    cps_true = sanitize_cps(cps_true, n).tolist()
    cps_pred = sanitize_cps(cps_pred, n).tolist()
    true_bounds = [0] + cps_true + [n]
    pred_bounds = [0] + cps_pred + [n]
    pred_segments = [(pred_bounds[i], pred_bounds[i + 1]) for i in range(len(pred_bounds) - 1)]
    total = 0.0
    for i in range(len(true_bounds) - 1):
        a, b = true_bounds[i], true_bounds[i + 1]
        length = max(0, b - a)
        if length <= 0:
            continue
        best = 0.0
        for c, d in pred_segments:
            inter = max(0, min(b, d) - max(a, c))
            union = max(b, d) - min(a, c)
            if union > 0:
                best = max(best, inter / union)
        total += length * best
    return float(total / max(1, n))


def f_measure_simple(cps_true, cps_pred, margin: int) -> float:
    cps_true = list(map(int, cps_true))
    cps_pred = list(map(int, cps_pred))
    if not cps_true and not cps_pred:
        return 1.0
    if not cps_true or not cps_pred:
        return 0.0
    used = set()
    tp = 0
    for p in cps_pred:
        best_i = None
        best_dist = None
        for i, t in enumerate(cps_true):
            if i in used:
                continue
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best_i = i
                best_dist = dist
        if best_i is not None:
            used.add(best_i)
            tp += 1
    precision = tp / len(cps_pred) if cps_pred else 0.0
    recall = tp / len(cps_true) if cps_true else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_metrics(n: int, cps_true, cps_pred, y_true, y_pred) -> dict[str, float]:
    cps_true = sanitize_cps(cps_true, n)
    cps_pred = sanitize_cps(cps_pred, n)
    margin = int(n * 0.01)
    try:
        from benchmark.metrics import f_measure, covering
        f1 = float(f_measure({0: cps_true}, cps_pred, margin=margin))
        cov = float(covering({0: cps_true}, cps_pred, n))
    except Exception:
        f1 = f_measure_simple(cps_true, cps_pred, margin=margin)
        cov = segmentation_covering(cps_true, cps_pred, n)
    return {
        "f1_score": f1,
        "covering_score": cov,
        "ami_score": float(adjusted_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred, average_method="geometric")),
    }


def find_segmentation_file(clap_repo: Path, dataset_name: str) -> Path | None:
    seg_dir = clap_repo / "experiments" / "segmentation"
    if not seg_dir.exists():
        return None
    candidates = []
    keys = {
        dataset_name.lower(),
        dataset_name.lower().replace("-", ""),
        dataset_name.lower().replace("_", ""),
        dataset_name.lower().replace("-", "_")
    }
    for path in list(seg_dir.glob("*.csv")) + list(seg_dir.glob("*.csv.gz")):
        name = path.name.lower()
        compact = name.replace("-", "").replace("_", "")
        if any(k in name or k in compact for k in keys) and "clasp" in name.lower():
            candidates.append(path)
                                
    exacts = [
        seg_dir / f"{dataset_name}_ClaSP.csv.gz",
        seg_dir / f"{dataset_name}_ClaSP.csv",
        seg_dir / f"{dataset_name.replace('-', '')}_ClaSP.csv.gz",
        seg_dir / f"{dataset_name.replace('-', '_')}_ClaSP.csv.gz",
    ]
    for p in exacts:
        if p.exists():
            return p
    return sorted(candidates, key=lambda p: len(p.name))[0] if candidates else None


def load_segmentation_df(seg_path: Path | None) -> pd.DataFrame | None:
    if seg_path is None or not seg_path.exists():
        return None
    converters = {"found_cps": lambda data: np.array(eval(data), dtype=int)}
    df = pd.read_csv(seg_path, converters=converters)
    if "dataset" not in df.columns or "found_cps" not in df.columns:
        raise ValueError(f"Segmentation file missing dataset/found_cps columns: {seg_path}")
    return df[["dataset", "found_cps"]]


def compute_clasp_cps(ts, n_jobs: int) -> np.ndarray:
    from claspy.segmentation import BinaryClaSPSegmentation
    clasp = BinaryClaSPSegmentation(n_jobs=int(n_jobs))
    cps = clasp.fit_predict(ts)
    return sanitize_cps(cps, len(ts))


def load_dataset(clap_repo: Path, dataset_name: str, case_names: list[str] | None = None):
    add_clap_repo(clap_repo)
    from src.utils import load_datasets, load_has_datasets
    name_upper = dataset_name.upper()
    if name_upper in {"HAS", "HAS2023"}:
        return load_has_datasets(), "HAS"
    if name_upper in {"MITBIH", "MIT-BIH", "MIT_BIH"}:
        return load_datasets("MIT-BIH"), "MIT-BIH"
    if name_upper == "SKAB":
        return load_datasets("SKAB"), "SKAB"
    if name_upper == "UTSA":
        return load_datasets("UTSA"), "UTSA"
    if name_upper == "TSSB":
        from src.utils import load_tssb_datasets
        return load_tssb_datasets(names=case_names), "TSSB"
    return load_datasets(dataset_name), dataset_name


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.clap_repo = args.clap_repo.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    add_clap_repo(args.clap_repo)
    from src.clap import CLaP
    from src.utils import create_state_labels

    case_names = [x.strip() for x in args.case_names.split(",") if x.strip()]
    df_data, canonical_name = load_dataset(args.clap_repo, args.dataset_name, case_names=case_names or None)
    if case_names and canonical_name != "TSSB":
        df_data = df_data.loc[df_data["dataset"].isin(case_names), :]
    seg_path = find_segmentation_file(args.clap_repo, canonical_name)
    seg_df = load_segmentation_df(seg_path)
    if args.max_cases and args.max_cases > 0:
        df_data = df_data.iloc[: int(args.max_cases), :]

    print("============================================================")
    print(f"Official CLaP default baseline on {canonical_name}")
    print("Repo root :", args.repo_root)
    print("CLaP repo :", args.clap_repo)
    print("Dataset   :", args.clap_repo / "datasets" / canonical_name)
    print("Seg file  :", seg_path if seg_path is not None else "<missing; compute/fallback>")
    print("Output    :", args.out_dir)
    print("Cases     :", len(df_data))
    print("n_jobs    :", args.n_jobs)
    print("============================================================")

    rows = []
    case_csv = args.out_dir / "all_case_results.csv"
    fieldnames = [
        "dataset", "window_size", "n", "true_cps_count", "init_cps_count", "final_cps_count",
        "true_states", "pred_states", "f1_score", "covering_score", "ami_score", "ari", "nmi",
        "runtime_seconds", "status", "error", "true_cps", "init_cps", "found_cps", "found_labels",
    ]

    with case_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (_, row) in enumerate(df_data.iterrows(), start=1):
            dataset = str(row["dataset"])
            w = int(row["window_size"]) if "window_size" in row.index else 50
            cps_true = safe_int_array(row["change_points"])
            labels = safe_int_array(row["labels"])
            ts = safe_ts_array(row["time_series"])
            ts_len = int(ts.shape[0])

            print(f"[{idx}/{len(df_data)}] {dataset}", flush=True)
            t0 = time.process_time()
            status = "ok"
            error = ""
            try:
                init_cps = None
                if seg_df is not None:
                    hit = seg_df.loc[seg_df["dataset"] == dataset]
                    if not hit.empty:
                        init_cps = sanitize_cps(hit.iloc[0].found_cps, ts_len)
                if init_cps is None or len(init_cps) == 0:
                    if args.compute_clasp_if_missing:
                        init_cps = compute_clasp_cps(ts, int(args.n_jobs))
                    else:
                        raise KeyError(f"No official ClaSP found_cps for dataset={dataset!r}; rerun with --compute-clasp-if-missing")

                clap = CLaP(n_jobs=int(args.n_jobs))
                clap.fit(ts, init_cps)
                found_cps = sanitize_cps(clap.get_change_points(), ts_len)
                found_labels = safe_int_array(clap.get_segment_labels())

                y_true = true_labels_to_point_labels(cps_true, labels, ts_len, create_state_labels)
                y_pred = labels_from_cps(found_cps, found_labels, ts_len)

                metrics = evaluate_metrics(ts_len, cps_true, found_cps, y_true, y_pred)
                runtime = time.process_time() - t0

                print(
                    f"  F1={metrics['f1_score']:.3f} Covering={metrics['covering_score']:.3f} "
                    f"AMI={metrics['ami_score']:.3f} ARI={metrics['ari']:.3f} "
                    f"NMI={metrics['nmi']:.3f} K={len(np.unique(y_pred))} seconds={runtime:.1f}",
                    flush=True,
                )

                if args.save_predictions:
                    safe_name = dataset.replace("/", "_").replace("\\", "_").replace(":", "_")
                    with (pred_dir / f"{safe_name}.csv").open("w", newline="", encoding="utf-8") as pf:
                        pw = csv.writer(pf)
                        pw.writerow(["t", "true_label", "pred_label"])
                        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
                            pw.writerow([t, int(yt), int(yp)])

            except Exception as exc:
                status = "error"
                error = repr(exc)
                runtime = time.process_time() - t0
                init_cps = np.asarray([], dtype=int)
                found_cps = np.asarray([], dtype=int)
                found_labels = np.asarray([], dtype=int)
                metrics = {"f1_score": 0.0, "covering_score": 0.0, "ami_score": 0.0, "ari": 0.0, "nmi": 0.0}
                print("  ERROR:", error, flush=True)

            out_row = {
                "dataset": dataset,
                "window_size": w,
                "n": ts_len,
                "true_cps_count": len(cps_true),
                "init_cps_count": len(init_cps),
                "final_cps_count": len(found_cps),
                "true_states": int(len(np.unique(labels))) if len(labels) else 0,
                "pred_states": int(len(np.unique(found_labels))) if len(found_labels) else 0,
                "f1_score": f"{metrics['f1_score']:.10f}",
                "covering_score": f"{metrics['covering_score']:.10f}",
                "ami_score": f"{metrics['ami_score']:.10f}",
                "ari": f"{metrics['ari']:.10f}",
                "nmi": f"{metrics['nmi']:.10f}",
                "runtime_seconds": f"{runtime:.4f}",
                "status": status,
                "error": error,
                "true_cps": json.dumps([int(x) for x in list(cps_true)]),
                "init_cps": json.dumps([int(x) for x in list(init_cps)]),
                "found_cps": json.dumps([int(x) for x in list(found_cps)]),
                "found_labels": json.dumps([int(x) for x in list(found_labels)]),
            }
            writer.writerow(out_row)
            rows.append(out_row)

    ok = [r for r in rows if r["status"] == "ok"]
    def mean_col(col: str) -> float:
        vals = [float(r[col]) for r in ok]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "repo_root": str(args.repo_root),
        "clap_repo": str(args.clap_repo),
        "dataset_name": canonical_name,
        "segmentation_path": str(seg_path) if seg_path is not None else None,
        "out_dir": str(args.out_dir),
        "cases": len(rows),
        "ok": len(ok),
        "mean_f1_score": mean_col("f1_score"),
        "mean_covering_score": mean_col("covering_score"),
        "mean_ami_score": mean_col("ami_score"),
        "mean_ari": mean_col("ari"),
        "mean_nmi": mean_col("nmi"),
    }
    (args.out_dir / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("============================================================")
    print(f"OK cases: {len(ok)}/{len(rows)}")
    print(f"Mean F1={summary['mean_f1_score']:.4f}")
    print(f"Mean Covering={summary['mean_covering_score']:.4f}")
    print(f"Mean AMI={summary['mean_ami_score']:.4f}")
    print(f"Mean ARI={summary['mean_ari']:.4f}")
    print(f"Mean NMI={summary['mean_nmi']:.4f}")
    print("Saved:", case_csv)
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
