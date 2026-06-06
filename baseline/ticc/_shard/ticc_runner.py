from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ticc_preprocessing import (
    default_data_root,
    import_runtime,
    load_cases,
    normalize_dataset_key,
    parse_dataset_list,
)
from ticc_metrics import compute_metrics


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except Exception:
        pass


def read_completed(case_csv: Path) -> set[tuple[str, str]]:
    if not case_csv.exists():
        return set()
    out = set()
    with case_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                out.add((row.get("dataset", ""), row.get("case_id", "")))
    return out


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def recompute_summary(case_csv: Path, out_dir: Path) -> dict:
    import pandas as pd
    if not case_csv.exists():
        return {"total_cases": 0, "ok_cases": 0, "failed_cases": 0}
    df = pd.read_csv(case_csv)
    total = int(len(df))
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.iloc[0:0].copy()
    failed = total - int(len(ok))
    rows = []
    if not ok.empty:
        for dataset, g in ok.groupby("dataset", dropna=False):
            rows.append({
                "algorithm": "TICC",
                "dataset": dataset,
                "case_count": int(len(g)),
                "ARI_mean": float(g["ARI"].mean()),
                "NMI_mean": float(g["NMI"].mean()),
                "AMI_mean": float(g["AMI"].mean()),
                "covering_mean": float(g["covering_score"].mean()),
                "f1_mean": float(g["f1_score"].mean()),
                "seconds_sum": float(g["seconds"].sum()),
            })
        rows.append({
            "algorithm": "TICC",
            "dataset": "ALL_DATASETS_EQUAL_CASE_WEIGHT",
            "case_count": int(len(ok)),
            "ARI_mean": float(ok["ARI"].mean()),
            "NMI_mean": float(ok["NMI"].mean()),
            "AMI_mean": float(ok["AMI"].mean()),
            "covering_mean": float(ok["covering_score"].mean()),
            "f1_mean": float(ok["f1_score"].mean()),
            "seconds_sum": float(ok["seconds"].sum()),
        })
    summary_path = out_dir / "algorithm_summary.csv"
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"total_cases": total, "ok_cases": int(len(ok)), "failed_cases": failed}


def align_labels_pred(labels, pred, np):
    labels = np.asarray(labels, dtype=int)
    pred = np.asarray(pred, dtype=int)
    n = min(len(labels), len(pred))
    return labels[:n], pred[:n]


def run_ticc(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else default_data_root(repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from TICC_solver import TICC

    runtime = import_runtime(repo_root)
    np = runtime["np"]
    set_global_seeds(args.seed)

    dataset_keys = parse_dataset_list(args.datasets)
    cases = load_cases(dataset_keys, data_root, runtime, max_cases=args.max_cases)

    print("============================================================")
    print("Strict TICC baseline")
    print("Shard   :", THIS_DIR)
    print("Repo    :", repo_root)
    print("Data    :", data_root)
    print("Output  :", out_dir)
    print("Datasets:", dataset_keys)
    print("Cases   :", len(cases))
    print("Note    : number_of_clusters uses the original oracle-K baseline setting")
    print("============================================================")

    if args.dry_run:
        for c in cases[:20]:
            print(f"DRY {c.dataset}/{c.case_id}: rows={len(c.data)} labels={len(c.labels)} K={c.n_clusters} win={c.window_size}")
        return 0

    case_csv = out_dir / "case_results.csv"
    completed = read_completed(case_csv) if args.skip_completed else set()
    fieldnames = [
        "algorithm", "dataset", "case_id", "status", "error",
        "rows_raw", "rows_eval", "features", "true_states", "pred_states",
        "true_cps_count", "pred_cps_count", "window_size", "n_clusters",
        "beta", "lambda_parameter", "threshold", "max_iters", "num_proc",
        "ARI", "NMI", "AMI", "covering_score", "f1_score", "cp_margin",
        "seconds", "length_aligned", "prediction_path", "source_path", "protocol",
    ]

    ok_cases = 0
    failed_cases = 0
    total_cases = 0
    t_all = time.time()

    for idx, case in enumerate(cases, start=1):
        key = (case.dataset, case.case_id)
        if key in completed:
            print(f"[{idx}/{len(cases)}] SKIP completed {case.dataset}/{case.case_id}", flush=True)
            continue
        total_cases += 1
        win = int(args.window_size) if int(args.window_size) > 0 else int(case.window_size)
        max_iters = int(args.max_iters) if int(args.max_iters) > 0 else int(case.max_iters)
        num_proc = int(args.num_proc) if int(args.num_proc) > 0 else int(case.num_proc_default)
        print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id} rows={len(case.data)} K={case.n_clusters} win={win}", flush=True)
        try:
            set_global_seeds(args.seed + idx)
            ticc = TICC(
                window_size=win,
                number_of_clusters=int(case.n_clusters),
                lambda_parameter=float(args.lambda_parameter),
                beta=float(args.beta),
                maxIters=max_iters,
                threshold=float(args.threshold),
                write_out_file=False,
                prefix_string=str(out_dir / "_ticc_work" / f"{case.dataset}_{case.case_id}"),
                num_proc=num_proc,
            )
            t0 = time.time()
            pred, _mrf = ticc.fit_transform(case.data)
            seconds = time.time() - t0
            labels, pred = align_labels_pred(case.labels, pred, np)
            met = compute_metrics(labels, pred, args.cp_margin_ratio)
            pred_path = pred_dir / f"{case.dataset}_{case.case_id}_labels_pred.npy".replace("/", "_").replace("\\", "_")
            np.save(pred_path, np.vstack([labels, pred]))
            row = {
                "algorithm": "TICC",
                "dataset": case.dataset,
                "case_id": case.case_id,
                "status": "ok",
                "error": "",
                "rows_raw": int(len(case.data)),
                "rows_eval": int(len(labels)),
                "features": int(case.data.shape[1]) if hasattr(case.data, "shape") and len(case.data.shape) > 1 else 0,
                "true_states": int(len(np.unique(labels))),
                "pred_states": int(len(np.unique(pred))),
                "true_cps_count": int(met["true_cps_count"]),
                "pred_cps_count": int(met["pred_cps_count"]),
                "window_size": win,
                "n_clusters": int(case.n_clusters),
                "beta": float(args.beta),
                "lambda_parameter": float(args.lambda_parameter),
                "threshold": float(args.threshold),
                "max_iters": max_iters,
                "num_proc": num_proc,
                "ARI": float(met["ARI"]),
                "NMI": float(met["NMI"]),
                "AMI": float(met["AMI"]),
                "covering_score": float(met["covering_score"]),
                "f1_score": float(met["f1_score"]),
                "cp_margin": int(met["cp_margin"]),
                "seconds": float(seconds),
                "length_aligned": int(len(labels)),
                "prediction_path": str(pred_path),
                "source_path": case.source_path,
                "protocol": case.protocol,
            }
            ok_cases += 1
            print(f"  OK ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} K_pred={row['pred_states']} seconds={seconds:.1f}", flush=True)
        except Exception as exc:
            row = {
                "algorithm": "TICC", "dataset": case.dataset, "case_id": case.case_id,
                "status": "error", "error": repr(exc),
                "rows_raw": int(len(case.data)) if hasattr(case.data, "__len__") else 0,
                "rows_eval": int(len(case.labels)) if hasattr(case.labels, "__len__") else 0,
                "features": int(case.data.shape[1]) if hasattr(case.data, "shape") and len(case.data.shape) > 1 else 0,
                "true_states": int(len(set(map(int, case.labels)))) if hasattr(case.labels, "__len__") else 0,
                "pred_states": 0, "true_cps_count": 0, "pred_cps_count": 0,
                "window_size": win, "n_clusters": int(case.n_clusters),
                "beta": float(args.beta), "lambda_parameter": float(args.lambda_parameter), "threshold": float(args.threshold),
                "max_iters": max_iters, "num_proc": num_proc,
                "ARI": float("nan"), "NMI": float("nan"), "AMI": float("nan"),
                "covering_score": float("nan"), "f1_score": float("nan"), "cp_margin": 0,
                "seconds": float("nan"), "length_aligned": 0, "prediction_path": "",
                "source_path": case.source_path, "protocol": case.protocol,
            }
            failed_cases += 1
            print("  ERROR:", row["error"], flush=True)
        write_rows(case_csv, [row], fieldnames)

    status = recompute_summary(case_csv, out_dir)
    status.update({
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "datasets_requested": dataset_keys,
        "this_run_cases": total_cases,
        "this_run_ok": ok_cases,
        "this_run_failed": failed_cases,
        "total_seconds_this_run": time.time() - t_all,
        "oracle_k_note": "number_of_clusters is set from ground-truth number of states, matching original TICC baseline scripts",
        "shard_dir": str(THIS_DIR),
    })
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print("Finished strict TICC baseline")
    print("case_results.csv      :", case_csv)
    print("algorithm_summary.csv :", out_dir / "algorithm_summary.csv")
    print("run_status.json       :", out_dir / "run_status.json")
    print(f"this run ok/failed    : {ok_cases}/{failed_cases}")
    print("============================================================")
    return 0 if failed_cases == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict sharded TICC baseline runner following original Time2State TICC preprocessing.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results_ticc_strict")
    parser.add_argument("--datasets", nargs="+", default=["mocap"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--beta", type=float, default=2200.0)
    parser.add_argument("--lambda-parameter", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=0, help="0 means original per-dataset value")
    parser.add_argument("--max-iters", type=int, default=0, help="0 means original per-dataset value")
    parser.add_argument("--num-proc", type=int, default=1, help="0 means original per-dataset default; 1 is safer on Windows")
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--cp-margin-ratio", type=float, default=0.01)
    args = parser.parse_args()
    return run_ticc(args)


if __name__ == "__main__":
    raise SystemExit(main())
