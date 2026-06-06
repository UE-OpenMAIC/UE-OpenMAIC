from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MOCAP_INFO = {
    "amc_86_01.4d": {"label": {588: 0, 1200: 1, 2006: 0, 2530: 2, 3282: 0, 4048: 3, 4579: 2}},
    "amc_86_02.4d": {"label": {1009: 0, 1882: 1, 2677: 2, 3158: 3, 4688: 4, 5963: 0, 7327: 5, 8887: 6, 9632: 7, 10617: 0}},
    "amc_86_03.4d": {"label": {872: 0, 1938: 1, 2448: 2, 3470: 0, 4632: 3, 5372: 4, 6182: 5, 7089: 6, 8401: 0}},
    "amc_86_07.4d": {"label": {1060: 0, 1897: 1, 2564: 2, 3665: 1, 4405: 2, 5169: 3, 5804: 4, 6962: 0, 7806: 5, 8702: 0}},
    "amc_86_08.4d": {"label": {1062: 0, 1904: 1, 2661: 2, 3282: 3, 3963: 4, 4754: 5, 5673: 6, 6362: 4, 7144: 7, 8139: 8, 9206: 0}},
    "amc_86_09.4d": {"label": {921: 0, 1275: 1, 2139: 2, 2887: 3, 3667: 4, 4794: 0}},
    "amc_86_10.4d": {"label": {2003: 0, 3720: 1, 4981: 0, 5646: 2, 6641: 3, 7583: 0}},
    "amc_86_11.4d": {"label": {1231: 0, 1693: 1, 2332: 2, 2762: 1, 3386: 3, 4015: 2, 4665: 1, 5674: 0}},
    "amc_86_14.4d": {"label": {671: 0, 1913: 1, 2931: 0, 4134: 2, 5051: 0, 5628: 1, 6055: 2}},
}

@dataclass
class SeriesCase:
    dataset: str
    case_id: str
    data: object
    labels: object
    fit_data: object | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run plain single-branch CLaP baseline on the strict our datasets.")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["mocap"])
    p.add_argument("--clap-repo", type=Path, required=True)
    p.add_argument("--loader-runner", type=Path, default=None)
    p.add_argument("--public-data-root", type=Path, default=None)
    p.add_argument("--max-series-per-dataset", type=int, default=None)
    p.add_argument("--max-synthetic", type=int, default=None)
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
    p.add_argument("--pamap2-feature-mode", choices=["auto", "paper9acc", "full_sensor"], default="auto")
    p.add_argument("--pamap2-remove-zero", action="store_true")
    p.add_argument("--public-max-rows", type=int, default=0)
    return p.parse_args()


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x for x in str(value).replace(",", " ").replace(";", " ").split() if x]


def seg_to_label(seg_info: dict[int, int], np):
    labels = []
    start = 0
    for end in sorted(seg_info):
        if end < start:
            continue
        labels.extend([seg_info[end]] * (end - start))
        start = end
    return np.asarray(labels, dtype=int)


def reorder_label(seq, np):
    mapping = {}
    out = []
    next_id = 0
    for value in list(seq):
        key = int(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out.append(mapping[key])
    return np.asarray(out, dtype=int), mapping


def adjusted_rand_index(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("ARI inputs must have equal length")
    if len(true_list) < 2:
        return 1.0
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(true_list, pred_list))


def normalized_mutual_information(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("NMI inputs must have equal length")
    if len(true_list) == 0:
        return 0.0
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(true_list, pred_list, average_method="geometric"))


def load_mocap(repo_root: Path, max_count: int | None):
    import numpy as np
    import pandas as pd
    base = repo_root / "Time2State" / "data" / "MoCap" / "4d"
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases = []
    for path in files:
        if path.name not in MOCAP_INFO:
            continue
        df = pd.read_csv(path, sep=" ", usecols=range(0, 4))
        data = df.to_numpy(dtype=float)
        labels = seg_to_label(MOCAP_INFO[path.name]["label"], np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("MoCap", path.name, data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No MoCap .4d cases found under {base}")
    return cases


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    aliases = {
        "synthetic": "synthetic",
        "mocap": "mocap",
        "actrectut": "actrectut",
        "act-rec-tut": "actrectut",
        "pamap2": "pamap2",
        "uschad": "usc-had",
        "usc-had": "usc-had",
        "ucrseg": "tssb",
        "ucr-seg": "tssb",
        "tssb": "tssb",
    }
    return aliases.get(key, key)


def import_module_from_path(path: Path, module_name: str):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(args: argparse.Namespace) -> tuple[list[SeriesCase], list[dict[str, object]]]:
    loader_path = args.loader_runner
    if loader_path is None:
        loader_path = args.repo_root / "our" / "_shared" / "our_multit2s_runner.py"
    module = import_module_from_path(loader_path, "external_our_multit2s_runner_for_clap_baseline")
    runtime = module.import_runtime(args.repo_root)
    cases, status = module.load_cases(args, runtime)
    out: list[SeriesCase] = []
    for case in cases:
        out.append(
            SeriesCase(
                case.dataset,
                str(case.case_id),
                case.data,
                case.labels,
                getattr(case, "fit_data", None),
            )
        )
    return out, status


def matches_case_id(case_id: str, wanted: str) -> bool:
    case_text = str(case_id).strip()
    wanted_text = str(wanted).strip()
    return (
        case_text == wanted_text
        or Path(case_text).stem == wanted_text
        or case_text.lower() == wanted_text.lower()
        or Path(case_text).stem.lower() == wanted_text.lower()
    )


def apply_case_selection(cases, args):
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
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    CLaP, BinaryClaSPSegmentation = add_clap_imports(args.clap_repo)
    cases, dataset_status = load_cases(args)
    cases = apply_case_selection(cases, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = args.out_dir / "all_case_results.csv"
    status_json = args.out_dir / "run_status.json"
    pred_dir = args.out_dir / "predictions"
    done = read_existing_completed(case_csv) if args.skip_completed else set()
    fieldnames = [
        "dataset", "case_id", "ari", "nmi", "true_states", "pred_states",
        "init_source", "init_cps_count", "final_cps_count", "init_cps", "final_cps",
        "segment_labels", "seconds", "status", "error",
    ]
    print("============================================================")
    print("Plain original CLaP runner")
    print("Repo root :", args.repo_root)
    print("CLaP repo :", args.clap_repo)
    print("Loader    :", args.loader_runner or (args.repo_root / "our" / "_shared" / "our_multit2s_runner.py"))
    print("Output    :", args.out_dir)
    print("Datasets  :", " ".join(args.datasets))
    print("Cases     :", len(cases))
    print("Params    :", args.clap_window_size, args.clap_classifier, args.clap_merge_score, args.normalize_input)
    print("============================================================")
    rows = []
    for idx, case in enumerate(cases, 1):
        if (case.dataset, case.case_id) in done:
            print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id} skipped")
            continue
        print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id}")
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
            if args.save_predictions:
                safe_case = str(case.case_id).replace("/", "_").replace("\\", "_").replace(":", "_")
                save_prediction_csv(pred_dir / f"{case.dataset}__{safe_case}.csv", yt, yp)
        except Exception as exc:
            status = "error"
            error = repr(exc)
            ari = nmi = 0.0
            true_states = int(len(set(map(int, list(case.labels))))) if case.labels is not None else 0
            pred_states = 0
            init_source = "error"
            init_cps = []
            final_cps = []
            seg_labels = []
        seconds = time.time() - t0
        print(f"  ARI={ari:.4f} NMI={nmi:.4f} K={pred_states} seconds={seconds:.1f} status={status}")
        if error:
            print("  ERROR:", error)
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
        "repo_root": str(args.repo_root),
        "clap_repo": str(args.clap_repo),
        "out_dir": str(args.out_dir),
        "datasets": args.datasets,
        "dataset_status": dataset_status,
        "n_cases_loaded": len(cases),
        "n_ok_this_time": len(ok),
        "mean_ari_this_time": float(np.mean(ari_vals)) if ari_vals else None,
        "mean_nmi_this_time": float(np.mean(nmi_vals)) if nmi_vals else None,
        "settings": {
            "init_cps_source": args.init_cps_source,
            "normalize_input": args.normalize_input,
            "clap_window_size": args.clap_window_size,
            "clap_classifier": args.clap_classifier,
            "clap_merge_score": args.clap_merge_score,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
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


if __name__ == "__main__":
    raise SystemExit(main())
