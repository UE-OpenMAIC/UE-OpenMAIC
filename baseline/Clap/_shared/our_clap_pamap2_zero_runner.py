from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

METRIC_BACKEND = "sklearn_adjusted_rand_score__sklearn_nmi_geometric"


@dataclass
class SeriesCase:
    dataset: str
    case_id: str
    data: object
    labels: object


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run plain single-branch CLaP baseline on PAMAP2_zero.")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["pamap2_zero"])
    p.add_argument("--clap-repo", type=Path, required=True)
    p.add_argument("--public-data-root", type=Path, default=None)
    p.add_argument("--max-series-per-dataset", type=int, default=None)
    p.add_argument("--priority-case-ids", default="")
    p.add_argument("--only-case-ids", default="")
    p.add_argument("--case-ids", default="")
    p.add_argument("--case-indexes", default="")
    p.add_argument("--rounds", default="")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--seed", type=int, default=1379)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--init-cps-source", choices=["clasp", "uniform"], default="clasp")
    p.add_argument("--fallback-uniform-segments", type=int, default=8)
    p.add_argument("--adf-sample-max", type=int, default=50000)
    p.add_argument("--normalize-input", choices=["none", "zscore", "minmax"], default="none")
    p.add_argument("--clap-window-size", default="suss")
    p.add_argument("--clap-classifier", default="rocket")
    p.add_argument("--clap-merge-score", default="cgain")
    p.add_argument("--save-predictions", action="store_true")


    p.add_argument("--pamap2-feature-mode", choices=["paper9acc", "full_sensor"], default="paper9acc")
    p.add_argument("--pamap2-remove-zero", action="store_true", default=True)
    p.add_argument("--pamap2-keep-zero", action="store_true")
    p.add_argument("--pamap2-loader-normalize", action="store_true", default=True)
    p.add_argument("--pamap2-no-loader-normalize", action="store_true")
    p.add_argument("--pamap2-subjects", default="1-8")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x for x in str(value).replace(",", " ").replace(";", " ").split() if x]


def parse_int_range_list(text: str | int) -> list[int]:
    if isinstance(text, int):
        return [text]
    out: list[int] = []
    for part in str(text).replace(",", " ").replace(";", " ").split():
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(float(part)))
    return out or list(range(1, 9))


def adjusted_rand_index(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    y = list(labels_true)
    z = list(labels_pred)
    if len(y) != len(z):
        raise ValueError("ARI inputs must have equal length")
    if len(y) < 2:
        return 1.0
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(y, z))
    except Exception:
        return _adjusted_rand_index_fallback(y, z)


def _adjusted_rand_index_fallback(y, z) -> float:
    def comb2(n: int) -> float:
        return 0.0 if n < 2 else n * (n - 1) / 2.0

    n = len(y)
    contingency = defaultdict(int)
    y_counts = Counter()
    z_counts = Counter()
    for a, b in zip(y, z):
        contingency[(int(a), int(b))] += 1
        y_counts[int(a)] += 1
        z_counts[int(b)] += 1
    sum_cells = sum(comb2(v) for v in contingency.values())
    sum_y = sum(comb2(v) for v in y_counts.values())
    sum_z = sum(comb2(v) for v in z_counts.values())
    total = comb2(n)
    expected = (sum_y * sum_z) / total if total else 0.0
    max_index = 0.5 * (sum_y + sum_z)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_cells == max_index else 0.0
    return float((sum_cells - expected) / denom)


def normalized_mutual_information(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    y = list(labels_true)
    z = list(labels_pred)
    if len(y) != len(z):
        raise ValueError("NMI inputs must have equal length")
    if len(y) == 0:
        return 0.0
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(y, z, average_method="geometric"))
    except Exception:
        return _normalized_mutual_information_fallback(y, z)


def _normalized_mutual_information_fallback(y, z) -> float:
    import math

    n = len(y)
    contingency = defaultdict(int)
    y_counts = Counter()
    z_counts = Counter()
    for a, b in zip(y, z):
        contingency[(int(a), int(b))] += 1
        y_counts[int(a)] += 1
        z_counts[int(b)] += 1
    mi = 0.0
    for (a, b), c in contingency.items():
        mi += (c / n) * math.log((c * n) / (y_counts[a] * z_counts[b]))

    def entropy(counts):
        out = 0.0
        for c in counts:
            p = c / n
            if p > 0:
                out -= p * math.log(p)
        return out

    hy = entropy(y_counts.values())
    hz = entropy(z_counts.values())
    if hy == 0.0 and hz == 0.0:
        return 1.0
    if hy == 0.0 or hz == 0.0:
        return 0.0
    return float(mi / math.sqrt(hy * hz))


def reorder_label(seq, np):
    mapping = {}
    out = []
    nxt = 0
    for value in list(seq):
        key = int(value)
        if key not in mapping:
            mapping[key] = nxt
            nxt += 1
        out.append(mapping[key])
    return np.asarray(out, dtype=int), mapping


def find_pamap2_protocol_dir(repo_root: Path, public_data_root: Path | None = None) -> Path:
    candidates = [
        repo_root / "Time2State" / "data" / "PAMAP2" / "Protocol",
        repo_root / "Time2State" / "data" / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
        repo_root / "Time2State" / "data" / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol",
        repo_root / "Time2State" / "Baselines" / "public_ts_datasets" / "extracted" / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol",
        repo_root / "Time2State" / "Baselines" / "public_ts_datasets" / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
    ]
    if public_data_root is not None:
        candidates.extend([
            public_data_root / "extracted" / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol",
            public_data_root / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
            public_data_root / "PAMAP2_Dataset" / "Protocol",
        ])
    for p in candidates:
        if (p / "subject101.dat").exists():
            return p.resolve()


    roots = [repo_root / "Time2State"]
    if public_data_root is not None:
        roots.append(public_data_root)
    for root in roots:
        if not root.exists():
            continue
        hits = list(root.rglob("subject101.dat"))
        for hit in hits:
            if hit.parent.name.lower() == "protocol":
                return hit.parent.resolve()
    raise FileNotFoundError(
        "Cannot find PAMAP2 Protocol directory. Expected subject101.dat under Time2State data/public_ts_datasets."
    )


def fill_nan_forward(data, np):
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    for j in range(arr.shape[1]):
        last = 0.0
        for i in range(arr.shape[0]):
            if np.isnan(arr[i, j]):
                arr[i, j] = last
            else:
                last = arr[i, j]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_normalize(data, np):
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    arr = (arr - mean) / std
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_pamap2_subject(protocol_dir: Path, subject_idx: int, np, pd, *, feature_mode: str, remove_zero: bool, loader_normalize: bool):
    path = protocol_dir / f"subject10{subject_idx}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing PAMAP2 file: {path}")
    df = pd.read_csv(path, sep=" ", header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if numeric.shape[1] < 2:
        raise ValueError(f"PAMAP2 file has too few columns: {path}, shape={numeric.shape}")
    labels = np.asarray(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)
    mode = str(feature_mode or "paper9acc").strip().lower()
    if mode == "full_sensor":
        data = numeric[:, 2:]
    elif mode == "paper9acc":
        if numeric.shape[1] < 41:
            raise ValueError(f"Too few columns for PAMAP2 paper9acc: {path}, shape={numeric.shape}")
        hand_acc = numeric[:, 4:7]
        chest_acc = numeric[:, 21:24]
        ankle_acc = numeric[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
    else:
        raise ValueError(f"Unsupported pamap2 feature_mode={feature_mode!r}")
    data = fill_nan_forward(data, np)
    if remove_zero:
        valid = labels > 0
        data = data[valid]
        labels = labels[valid]
    if len(labels) < 2:
        raise ValueError(f"subject10{subject_idx} has too few valid frames")
    if loader_normalize:
        data = zscore_normalize(data, np)
    labels, _ = reorder_label(labels, np)
    n = min(len(data), len(labels))
    return data[:n], labels[:n]


def load_pamap2_zero_cases(args: argparse.Namespace):
    import numpy as np
    import pandas as pd

    protocol_dir = find_pamap2_protocol_dir(args.repo_root, args.public_data_root)
    remove_zero = not bool(getattr(args, "pamap2_keep_zero", False))
    if getattr(args, "pamap2_remove_zero", False):
        remove_zero = True
    loader_normalize = not bool(getattr(args, "pamap2_no_loader_normalize", False))
    if getattr(args, "pamap2_loader_normalize", False):
        loader_normalize = True
    subjects = parse_int_range_list(getattr(args, "pamap2_subjects", "1-8"))

    cases: list[SeriesCase] = []
    for subject_idx in subjects:
        data, labels = load_pamap2_subject(
            protocol_dir,
            subject_idx,
            np,
            pd,
            feature_mode=args.pamap2_feature_mode,
            remove_zero=remove_zero,
            loader_normalize=loader_normalize,
        )
        case_id = f"10{subject_idx}"
        cases.append(SeriesCase("PAMAP2_zero" if remove_zero else "PAMAP2", case_id, data, labels))
        if args.max_series_per_dataset is not None and len(cases) >= int(args.max_series_per_dataset):
            break
    status = [{
        "dataset": "PAMAP2_zero" if remove_zero else "PAMAP2",
        "protocol_dir": str(protocol_dir),
        "subjects": subjects,
        "feature_mode": args.pamap2_feature_mode,
        "remove_zero": remove_zero,
        "loader_normalize": loader_normalize,
        "cases": len(cases),
    }]
    return cases, status


def matches_case_id(case_id: str, wanted: str) -> bool:
    c = str(case_id).strip()
    w = str(wanted).strip()
    return c == w or Path(c).stem == w or c.lower() == w.lower() or Path(c).stem.lower() == w.lower()


def apply_case_selection(cases: list[SeriesCase], args: argparse.Namespace) -> list[SeriesCase]:
    only = split_values(args.only_case_ids) + split_values(getattr(args, "case_ids", ""))
    if only:
        cases = [c for c in cases if any(matches_case_id(c.case_id, x) for x in only)]
    indexes = []
    for x in split_values(args.case_indexes or args.rounds):
        try:
            indexes.append(int(x))
        except Exception:
            pass
    if indexes:
        cases = [cases[i - 1] for i in indexes if 0 <= i - 1 < len(cases)]
    priority = split_values(args.priority_case_ids)
    if priority:
        front = [c for c in cases if any(matches_case_id(c.case_id, x) for x in priority)]
        rest = [c for c in cases if not any(matches_case_id(c.case_id, x) for x in priority)]
        cases = front + rest
    return cases


def add_clap_imports(clap_repo: Path):
    clap_repo = clap_repo.resolve()
    if not (clap_repo / "src" / "clap.py").exists():
        raise FileNotFoundError(f"CLaP source not found: {clap_repo / 'src' / 'clap.py'}")
    if str(clap_repo) in sys.path:
        sys.path.remove(str(clap_repo))
    sys.path.insert(0, str(clap_repo))
    from src.clap import CLaP
    from claspy.segmentation import BinaryClaSPSegmentation
    return CLaP, BinaryClaSPSegmentation


def clean_numeric_array(data, np):
    arr = np.asarray(data, dtype=float)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_input_array(ts, mode: str, np):
    arr = np.asarray(ts, dtype=float)
    if mode == "none":
        return arr
    work = arr.reshape(-1, 1) if arr.ndim == 1 else arr.copy()
    if mode == "zscore":
        mean = np.nanmean(work, axis=0, keepdims=True)
        std = np.nanstd(work, axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        work = (work - mean) / std
    elif mode == "minmax":
        mn = np.nanmin(work, axis=0, keepdims=True)
        mx = np.nanmax(work, axis=0, keepdims=True)
        denom = mx - mn
        denom[denom < 1e-8] = 1.0
        work = (work - mn) / denom
    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    return work.reshape(-1) if arr.ndim == 1 else work


def subsample_for_adf(x, max_len: int, np):
    x = np.asarray(x, dtype=float)
    if max_len <= 0 or len(x) <= max_len:
        return x
    idx = np.linspace(0, len(x) - 1, max_len).astype(int)
    return x[idx]


def choose_clasp_distance(ts, args, np) -> str:
    from statsmodels.tsa.stattools import adfuller
    try:
        if ts.ndim == 1:
            p = adfuller(subsample_for_adf(ts, args.adf_sample_max, np))[1]
        else:
            ps = []
            for d in range(ts.shape[1]):
                try:
                    ps.append(adfuller(subsample_for_adf(ts[:, d], args.adf_sample_max, np))[1])
                except Exception:
                    pass
            p = float(np.median(ps)) if ps else 1.0
    except Exception:
        p = 1.0
    return "znormed_euclidean_distance" if p < 0.05 else "euclidean_distance"


def uniform_change_points(n: int, n_segments: int, np):
    n_segments = max(2, int(n_segments))
    cps = np.linspace(0, n, n_segments + 1, dtype=int)[1:-1]
    return np.asarray([int(x) for x in cps if 0 < int(x) < n], dtype=int)


def safe_change_points(cps, n: int, np):
    out = []
    for x in list(cps):
        try:
            v = int(x)
        except Exception:
            continue
        if 0 < v < n:
            out.append(v)
    return np.asarray(sorted(set(out)), dtype=int)


def fit_initial_cps(ts, args, BinaryClaSPSegmentation, np):
    if args.init_cps_source == "uniform":
        return uniform_change_points(len(ts), args.fallback_uniform_segments, np), "uniform"
    try:
        distance = choose_clasp_distance(ts, args, np)
        clasp = BinaryClaSPSegmentation(distance=distance, n_jobs=int(args.n_jobs))
        cps = safe_change_points(clasp.fit_predict(ts), len(ts), np)
        if len(cps) == 0:
            raise ValueError("ClaSP returned no change points")
        return cps, f"clasp:{distance}"
    except Exception as exc:
        cps = uniform_change_points(len(ts), args.fallback_uniform_segments, np)
        return cps, f"uniform_fallback_after_clasp_error:{repr(exc)}"


def cps_labels_to_sequence(cps, seg_labels, n: int, np):
    cps = safe_change_points(cps, n, np).tolist()
    labels = [int(x) for x in list(seg_labels)]
    n_segments = len(cps) + 1
    if len(labels) < n_segments:
        labels = ([0] * n_segments) if not labels else labels + [labels[-1]] * (n_segments - len(labels))
    elif len(labels) > n_segments:
        labels = labels[:n_segments]
    out = np.zeros(n, dtype=int)
    start = 0
    for idx, end in enumerate(cps + [n]):
        out[start:end] = int(labels[idx])
        start = end
    out, _ = reorder_label(out, np)
    return out


def run_plain_clap_case(data, args, CLaP, BinaryClaSPSegmentation, np):
    ts = clean_numeric_array(data, np)
    ts = normalize_input_array(ts, args.normalize_input, np)
    init_cps, init_source = fit_initial_cps(ts, args, BinaryClaSPSegmentation, np)
    kwargs = {"n_jobs": int(args.n_jobs)}
    if args.clap_window_size:
        kwargs["window_size"] = str(args.clap_window_size)
    if args.clap_classifier:
        kwargs["classifier"] = str(args.clap_classifier)
    if args.clap_merge_score:
        kwargs["merge_score"] = str(args.clap_merge_score)
    clap = CLaP(**kwargs)
    clap.fit(ts, init_cps)
    final_cps = safe_change_points(clap.get_change_points(), len(ts), np)
    seg_labels = clap.get_segment_labels()
    pred = cps_labels_to_sequence(final_cps, seg_labels, len(ts), np)
    return init_cps, final_cps, seg_labels, pred, init_source


def read_existing_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                out.add((row.get("dataset", ""), row.get("case_id", "")))
    return out


def append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def save_prediction_csv(path: Path, labels_true, labels_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "true_label", "pred_label"])
        for i, (yt, yp) in enumerate(zip(labels_true, labels_pred)):
            w.writerow([i, int(yt), int(yp)])


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.clap_repo = args.clap_repo.resolve()
    if args.public_data_root is not None:
        args.public_data_root = args.public_data_root.resolve()

    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Cuda is available." if _cuda_available() else "Cuda is not available or torch is not installed.", flush=True)
    CLaP, BinaryClaSPSegmentation = add_clap_imports(args.clap_repo)
    cases, dataset_status = load_pamap2_zero_cases(args)
    cases = apply_case_selection(cases, args)
    if args.max_series_per_dataset is not None:
        cases = cases[: int(args.max_series_per_dataset)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = args.out_dir / "all_case_results.csv"
    status_json = args.out_dir / "run_status.json"
    pred_dir = args.out_dir / "predictions"
    done = read_existing_completed(case_csv) if args.skip_completed else set()
    fieldnames = [
        "dataset", "case_id", "ari", "nmi", "true_states", "pred_states",
        "init_source", "init_cps_count", "final_cps_count", "init_cps", "final_cps",
        "segment_labels", "rows", "features", "seconds", "status", "error",
    ]

    print("============================================================")
    print("Plain original CLaP runner for PAMAP2_zero")
    print("Repo root :", args.repo_root)
    print("CLaP repo :", args.clap_repo)
    print("Output    :", args.out_dir)
    print("Cases     :", len(cases))
    print("Feature   :", args.pamap2_feature_mode)
    print("Remove 0  :", int(not args.pamap2_keep_zero))
    print("Loader z  :", int(not args.pamap2_no_loader_normalize))
    print("Params    :", args.clap_window_size, args.clap_classifier, args.clap_merge_score, args.normalize_input)
    print("============================================================", flush=True)

    if args.dry_run:
        print("Dry run only.")
        return 0

    rows = []
    for idx, case in enumerate(cases, 1):
        if (case.dataset, case.case_id) in done:
            print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id} skipped", flush=True)
            continue
        print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id}", flush=True)
        t0 = time.time()
        status = "ok"
        error = ""
        try:
            data = clean_numeric_array(case.data, np)
            labels = np.asarray(case.labels, dtype=int)
            if args.limit_rows is not None and args.limit_rows > 0:
                n0 = min(args.limit_rows, len(data), len(labels))
                data = data[:n0]
                labels = labels[:n0]
            init_cps, final_cps, seg_labels, pred, init_source = run_plain_clap_case(data, args, CLaP, BinaryClaSPSegmentation, np)
            n = min(len(labels), len(pred))
            yt = labels[:n]
            yp = pred[:n]
            ari = adjusted_rand_index(yt, yp)
            nmi = normalized_mutual_information(yt, yp)
            true_states = int(len(np.unique(yt)))
            pred_states = int(len(np.unique(yp)))
            rows_count = int(n)
            features = int(data.shape[1]) if data.ndim > 1 else 1
            if args.save_predictions:
                safe_case = str(case.case_id).replace("/", "_").replace("\\", "_").replace(":", "_")
                save_prediction_csv(pred_dir / f"{case.dataset}__{safe_case}.csv", yt, yp)
        except Exception as exc:
            status = "error"
            error = repr(exc)
            ari = nmi = 0.0
            true_states = int(len(set(map(int, list(case.labels))))) if case.labels is not None else 0
            pred_states = 0
            rows_count = int(len(case.labels)) if case.labels is not None else 0
            features = int(case.data.shape[1]) if getattr(case.data, "ndim", 0) > 1 else 1
            init_source = "error"
            init_cps = []
            final_cps = []
            seg_labels = []
        seconds = time.time() - t0
        print(f"  ARI={ari:.4f} NMI={nmi:.4f} K={pred_states} seconds={seconds:.1f} status={status}", flush=True)
        if error:
            print("  ERROR:", error, flush=True)
        row = {
            "dataset": case.dataset,
            "case_id": case.case_id,
            "ari": f"{ari:.10f}",
            "nmi": f"{nmi:.10f}",
            "true_states": true_states,
            "pred_states": pred_states,
            "init_source": init_source,
            "init_cps_count": len(init_cps),
            "final_cps_count": len(final_cps),
            "init_cps": json.dumps([int(x) for x in list(init_cps)], ensure_ascii=False),
            "final_cps": json.dumps([int(x) for x in list(final_cps)], ensure_ascii=False),
            "segment_labels": json.dumps([int(x) for x in list(seg_labels)], ensure_ascii=False),
            "rows": rows_count,
            "features": features,
            "seconds": f"{seconds:.4f}",
            "status": status,
            "error": error,
        }
        append_rows(case_csv, [row], fieldnames)
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    ari_vals = [float(r["ari"]) for r in ok]
    nmi_vals = [float(r["nmi"]) for r in ok]
    summary = {
        "ok": bool(ari_vals),
        "repo_root": str(args.repo_root),
        "clap_repo": str(args.clap_repo),
        "out_dir": str(args.out_dir),
        "datasets": args.datasets,
        "dataset_status": dataset_status,
        "n_cases_loaded": len(cases),
        "n_ok_this_time": len(ok),
        "mean_ari_this_time": float(np.mean(ari_vals)) if ari_vals else None,
        "mean_nmi_this_time": float(np.mean(nmi_vals)) if nmi_vals else None,
        "metric_backend": METRIC_BACKEND,
        "settings": {
            "init_cps_source": args.init_cps_source,
            "normalize_input": args.normalize_input,
            "clap_window_size": args.clap_window_size,
            "clap_classifier": args.clap_classifier,
            "clap_merge_score": args.clap_merge_score,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
            "pamap2_feature_mode": args.pamap2_feature_mode,
            "pamap2_remove_zero": not args.pamap2_keep_zero,
            "pamap2_loader_normalize": not args.pamap2_no_loader_normalize,
        },
    }
    status_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if ari_vals:
        print("============================================================")
        print(f"Mean ARI={float(np.mean(ari_vals)):.4f}  Mean NMI={float(np.mean(nmi_vals)):.4f}  n={len(ari_vals)}")
        print("Saved:", case_csv)
        print("============================================================")
    else:
        print("No successful cases in this run.")
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
