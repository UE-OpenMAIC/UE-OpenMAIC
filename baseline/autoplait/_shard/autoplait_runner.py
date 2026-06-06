from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from autoplait_preprocessing import load_cases, split_values
from autoplait_metrics import (
    adjusted_rand_index,
    normalized_mutual_information,
    adjusted_mutual_information,
    segmentation_covering,
    cp_f1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict AutoPlait baseline runner.")
    p.add_argument("--repo-root", type=Path, default=Path(r"D:\code\teacherT2S"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--autoplait-bin", type=Path, required=True)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--cp-margin-ratio", type=float, default=0.01)
    p.add_argument("--segment-index-base", choices=["auto", "zero", "one"], default="auto")
    p.add_argument("--pamap2-downsample", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def import_runtime():
    missing = []
    for name in ["numpy", "pandas", "scipy"]:
        try:
            __import__(name)
        except Exception as exc:
            missing.append({"package": name, "error": repr(exc)})
    if missing:
        raise RuntimeError("Missing packages: " + json.dumps(missing, ensure_ascii=False))
    import numpy as np
    import pandas as pd
    import scipy.io
    try:
        from TSpy.dataset import load_USC_HAD as tspy_load_USC_HAD
    except Exception:
        tspy_load_USC_HAD = None
    return {"np": np, "pd": pd, "scipy_io": scipy.io, "tspy_load_USC_HAD": tspy_load_USC_HAD}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def existing_ok_keys(case_csv: Path) -> set[tuple[str, str]]:
    if not case_csv.exists():
        return set()
    out = set()
    try:
        with case_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("error", "")).strip() == "":
                    out.add((str(row.get("dataset", "")), str(row.get("case_id", ""))))
    except Exception:
        return set()
    return out


def write_autoplait_input(path: Path, data, np) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=float)

    np.savetxt(path, arr, fmt="%.10g")


def parse_autoplait_output(path: Path, n: int, *, index_base: str = "auto", np=None):
    """
    Flexible parser for SavePlait output.

    Expected core segment line format from PrintStEd:
        start end

    If non-segment lines appear between groups, each group is treated as a state/pattern.
    If there are no group markers, each segment is treated as a new state.
    """
    if not path.exists():
        raise FileNotFoundError(f"AutoPlait output file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    saw_marker = False

    seg_re = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s*$")
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                groups.append(current)
                current = []
                saw_marker = True
            continue
        m = seg_re.match(stripped)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            current.append((a, b))
        else:
            if current:
                groups.append(current)
                current = []
                saw_marker = True

    if current:
        groups.append(current)

    if not groups:

        pairs = []
        for line in raw_lines:
            ints = re.findall(r"-?\d+", line)
            if len(ints) >= 2:
                pairs.append((int(ints[0]), int(ints[1])))
        if not pairs:
            raise ValueError(f"Cannot parse any segment pairs from AutoPlait output: {path}")
        groups = [[p] for p in pairs]


    if len(groups) == 1 and not saw_marker and len(groups[0]) > 1:
        groups = [[p] for p in groups[0]]

    all_starts = [s for g in groups for s, _ in g]
    all_ends = [e for g in groups for _, e in g]
    if index_base == "one" or (index_base == "auto" and all_starts and min(all_starts) >= 1 and max(all_ends) <= n):
        groups = [[(s - 1, e - 1) for s, e in g] for g in groups]

    pred = np.full(n, -1, dtype=int)
    for label, group in enumerate(groups):
        for s, e in group:
            s = max(0, min(n - 1, int(s)))
            e = max(0, min(n - 1, int(e)))
            if e < s:
                s, e = e, s
            pred[s : e + 1] = int(label)


    last = -1
    for i in range(n):
        if pred[i] >= 0:
            last = int(pred[i])
        elif last >= 0:
            pred[i] = last
    nextv = -1
    for i in range(n - 1, -1, -1):
        if pred[i] >= 0:
            nextv = int(pred[i])
        elif nextv >= 0:
            pred[i] = nextv
    pred[pred < 0] = 0


    mapping = {}
    remapped = np.zeros(n, dtype=int)
    nxt = 0
    for i, v in enumerate(pred):
        v = int(v)
        if v not in mapping:
            mapping[v] = nxt
            nxt += 1
        remapped[i] = mapping[v]
    return remapped, groups


def run_one_case(case, args, runtime, dirs: dict[str, Path]):
    np = runtime["np"]
    safe = safe_name(f"{case.dataset}_{case.case_id}")
    input_path = dirs["inputs"] / f"{safe}.txt"
    raw_output = dirs["raw"] / f"{safe}_autoplait_segments.txt"
    stdout_path = dirs["logs"] / f"{safe}_stdout.txt"
    pred_path = dirs["pred"] / f"{safe}_labels_pred.npy"

    write_autoplait_input(input_path, case.data, np)
    dim = int(np.asarray(case.data).shape[1])

    cmd = [str(args.autoplait_bin), str(dim), str(input_path), str(raw_output)]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    seconds = time.time() - t0
    stdout_path.write_text("COMMAND:\n" + " ".join(cmd) + "\n\n" + proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"AutoPlait failed with exit code {proc.returncode}. See {stdout_path}")

    pred, groups = parse_autoplait_output(raw_output, len(case.labels), index_base=args.segment_index_base, np=np)
    labels = np.asarray(case.labels, dtype=int)
    n = min(len(labels), len(pred))
    labels = labels[:n]
    pred = pred[:n]
    np.save(pred_path, np.vstack([labels, pred]))

    margin = max(1, int(round(n * float(args.cp_margin_ratio))))
    return {
        "algorithm": "AutoPlait",
        "dataset": case.dataset,
        "case_id": case.case_id,
        "rows": int(n),
        "features": int(dim),
        "true_states": int(len(np.unique(labels))),
        "pred_states": int(len(np.unique(pred))),
        "ARI": adjusted_rand_index(labels, pred),
        "NMI": normalized_mutual_information(labels, pred),
        "AMI": adjusted_mutual_information(labels, pred),
        "Covering": segmentation_covering(labels, pred),
        "CP_F1": cp_f1(labels, pred, margin),
        "CP_margin": margin,
        "seconds": seconds,
        "autoplait_bin": str(args.autoplait_bin),
        "input_path": str(input_path),
        "raw_output": str(raw_output),
        "stdout_path": str(stdout_path),
        "prediction_path": str(pred_path),
        "segment_groups": int(len(groups)),
        "note": case.note,
        "error": "",
    }


def summarize(case_csv: Path, out_dir: Path) -> None:
    import pandas as pd
    if not case_csv.exists():
        return
    df = pd.read_csv(case_csv, encoding="utf-8-sig")
    ok = df[df["error"].fillna("").astype(str).str.strip().eq("")]
    if ok.empty:
        return
    rows = []
    for dataset, g in ok.groupby("dataset"):
        rows.append({
            "algorithm": "AutoPlait",
            "dataset": dataset,
            "cases": int(len(g)),
            "ARI": float(pd.to_numeric(g["ARI"], errors="coerce").mean()),
            "NMI": float(pd.to_numeric(g["NMI"], errors="coerce").mean()),
            "AMI": float(pd.to_numeric(g["AMI"], errors="coerce").mean()),
            "Covering": float(pd.to_numeric(g["Covering"], errors="coerce").mean()),
            "CP_F1": float(pd.to_numeric(g["CP_F1"], errors="coerce").mean()),
        })
    pd.DataFrame(rows).to_csv(out_dir / "algorithm_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    args.repo_root = Path(args.repo_root).resolve()
    args.out_dir = Path(args.out_dir).resolve()
    args.data_root = Path(args.data_root or (args.repo_root / "Time2State" / "data")).resolve()
    args.autoplait_bin = Path(args.autoplait_bin).resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "inputs": args.out_dir / "inputs",
        "raw": args.out_dir / "raw_outputs",
        "logs": args.out_dir / "logs",
        "pred": args.out_dir / "predictions",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("AutoPlait strict baseline")
    print("Repo root     :", args.repo_root)
    print("Data root     :", args.data_root)
    print("Output        :", args.out_dir)
    print("Datasets      :", args.datasets)
    print("AutoPlait bin :", args.autoplait_bin)
    print("Note          : AutoPlait C program internally applies ZnormSequence.")
    print("============================================================")

    if not args.autoplait_bin.exists():
        raise FileNotFoundError(f"AutoPlait binary not found: {args.autoplait_bin}")

    runtime = import_runtime()
    cases = load_cases(args.datasets, args.data_root, runtime, max_cases=args.max_cases, args=args)
    print(f"Loaded cases: {len(cases)}")
    if args.dry_run:
        for c in cases[:10]:
            print(" ", c.dataset, c.case_id, getattr(c.data, "shape", None), len(c.labels))
        return 0

    case_csv = args.out_dir / "case_results.csv"
    done = existing_ok_keys(case_csv) if args.skip_completed else set()
    fieldnames = [
        "algorithm", "dataset", "case_id", "rows", "features", "true_states", "pred_states",
        "ARI", "NMI", "AMI", "Covering", "CP_F1", "CP_margin", "seconds",
        "autoplait_bin", "input_path", "raw_output", "stdout_path", "prediction_path",
        "segment_groups", "note", "error",
    ]
    write_header = not case_csv.exists() or not args.skip_completed
    mode = "a" if args.skip_completed and case_csv.exists() else "w"

    ok_count = 0
    fail_count = 0
    with case_csv.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, case in enumerate(cases, start=1):
            key = (case.dataset, case.case_id)
            if key in done:
                print(f"[SKIP {idx}/{len(cases)}] {case.dataset}/{case.case_id}")
                continue
            print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id}", flush=True)
            try:
                row = run_one_case(case, args, runtime, dirs)
                ok_count += 1
                print(f"  ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} states={row['pred_states']} seconds={row['seconds']:.1f}")
            except Exception as exc:
                fail_count += 1
                row = {k: "" for k in fieldnames}
                row.update({
                    "algorithm": "AutoPlait",
                    "dataset": case.dataset,
                    "case_id": case.case_id,
                    "rows": int(len(case.labels)),
                    "features": int(getattr(case.data, "shape", [0, 0])[1]) if hasattr(case.data, "shape") and len(case.data.shape) > 1 else 1,
                    "note": case.note,
                    "error": repr(exc),
                })
                print("  [ERROR]", repr(exc), flush=True)
            writer.writerow(row)
            f.flush()

    summarize(case_csv, args.out_dir)
    print("============================================================")
    print("DONE")
    print("OK    :", ok_count)
    print("FAILED:", fail_count)
    print("CSV   :", case_csv)
    print("============================================================")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
