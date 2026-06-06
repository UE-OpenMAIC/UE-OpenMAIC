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
from io import StringIO
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
OURCLAP_ROOT = THIS_DIR.parent
REPO_ROOT = OURCLAP_ROOT.parent
DEFAULT_OUT_DIR = OURCLAP_ROOT / "results"
METRIC_BACKEND = "sklearn_adjusted_rand_score__sklearn_nmi_geometric"


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


@dataclass
class ClapBranchConfig:
    branch_name: str
    scale_id: int
    step_id: int
    init_cps_source: str
    fallback_uniform_segments: int
    normalize_input: str
    clap_window_size: str
    clap_classifier: str
    clap_merge_score: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLaP multi-branch + PID + meta aggregation on Time2State benchmarks.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["mocap"])
    parser.add_argument("--clap-repo", type=Path, default=REPO_ROOT / "classification-label-profile-main")
    parser.add_argument("--loader-runner", type=Path, default=None)
    parser.add_argument("--public-data-root", type=Path, default=REPO_ROOT / "Time2State" / "Baselines" / "public_ts_datasets")
    parser.add_argument("--clap-branch-config-txt", type=Path, default=None)
    parser.add_argument("--max-series-per-dataset", type=int, default=None)
    parser.add_argument("--max-synthetic", type=int, default=None)
    parser.add_argument("--only-case-ids", default="")
    parser.add_argument("--priority-case-ids", default="")
    parser.add_argument("--case-indexes", default="")
    parser.add_argument("--rounds", default="")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--seed", type=int, default=1379)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--fallback-uniform-segments", type=int, default=8)
    parser.add_argument("--adf-sample-max", type=int, default=50000)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--select-top-k-branches", type=int, default=16)
    parser.add_argument("--branch-select-metric", choices=["PEER", "PID"], default="PID")
    parser.add_argument("--meta-vote-weight-mode", choices=["uniform", "branch_reliability", "pid_weight"], default="pid_weight")
    parser.add_argument("--meta-min-len", type=int, default=24)
    parser.add_argument("--state-cluster-smooth", type=int, default=9)
    parser.add_argument("--peer-health-weight", type=float, default=0.45)
    parser.add_argument("--peer-consensus-weight", type=float, default=0.55)
    parser.add_argument("--pid-kp", type=float, default=0.45)
    parser.add_argument("--pid-ki", type=float, default=0.35)
    parser.add_argument("--pid-kd", type=float, default=0.20)
    parser.add_argument("--pid-softmax-tau", type=float, default=0.15)
    parser.add_argument("--pamap2-feature-mode", choices=["auto", "paper9acc", "full_sensor"], default="auto")
    parser.add_argument("--pamap2-remove-zero", action="store_true")
    parser.add_argument("--public-max-rows", type=int, default=0)
    return parser.parse_args()


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(",", " ").replace(";", " ").strip()
    return [part for part in text.split() if part]


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


def comb2(n: int) -> float:
    return 0.0 if n < 2 else n * (n - 1) / 2.0


def adjusted_rand_index(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("ARI inputs must have equal length")
    if len(true_list) < 2:
        return 1.0
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(true_list, pred_list))
    except Exception:
        n = len(true_list)
        contingency = defaultdict(int)
        true_counts = Counter()
        pred_counts = Counter()
        for truth, pred in zip(true_list, pred_list):
            contingency[(truth, pred)] += 1
            true_counts[truth] += 1
            pred_counts[pred] += 1
        sum_cells = sum(comb2(v) for v in contingency.values())
        sum_true = sum(comb2(v) for v in true_counts.values())
        sum_pred = sum(comb2(v) for v in pred_counts.values())
        total = comb2(n)
        expected = (sum_true * sum_pred) / total if total else 0.0
        max_index = 0.5 * (sum_true + sum_pred)
        denom = max_index - expected
        if denom == 0:
            return 1.0 if sum_cells == max_index else 0.0
        return (sum_cells - expected) / denom


def normalized_mutual_information(labels_true: Iterable[object], labels_pred: Iterable[object]) -> float:
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("NMI inputs must have equal length")
    if len(true_list) == 0:
        return 0.0
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(true_list, pred_list, average_method="geometric"))
    except Exception:
        n = len(true_list)
        contingency = defaultdict(int)
        true_counts = Counter()
        pred_counts = Counter()
        for truth, pred in zip(true_list, pred_list):
            contingency[(truth, pred)] += 1
            true_counts[truth] += 1
            pred_counts[pred] += 1
        mi = 0.0
        for (truth, pred), count in contingency.items():
            mi += (count / n) * math.log((count * n) / (true_counts[truth] * pred_counts[pred]))

        def entropy(counts: Iterable[int]) -> float:
            out = 0.0
            for count in counts:
                p = count / n
                if p > 0:
                    out -= p * math.log(p)
            return out

        h_true = entropy(true_counts.values())
        h_pred = entropy(pred_counts.values())
        if h_true == 0.0 and h_pred == 0.0:
            return 1.0
        if h_true == 0.0 or h_pred == 0.0:
            return 0.0
        return mi / math.sqrt(h_true * h_pred)


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


def seg_to_label(seg_info: dict[int, int], np):
    labels = []
    start = 0
    for end in sorted(seg_info):
        if end < start:
            continue
        labels.extend([seg_info[end]] * (end - start))
        start = end
    return np.asarray(labels, dtype=int)


def align_sequence(seq, target_len, np):
    seq = np.asarray(seq, dtype=int)
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) == 0:
        return np.zeros(target_len, dtype=int)
    pad = np.full(target_len - len(seq), int(seq[-1]), dtype=int)
    return np.concatenate([seq, pad])


def _parse_int(value, default):
    value = "" if value is None else str(value).strip()
    if value == "":
        return default
    return int(float(value))


def _parse_str(value, default=""):
    value = "" if value is None else str(value).strip()
    return default if value == "" else value


def default_clap_branches(args) -> list[ClapBranchConfig]:
    rows = []
                                                                                                       
                                                                                         
    windows = ["suss", "mwf", "fft", "acf"]
    classifiers = ["rocket", "mrhydra"]
    merge_scores = ["cgain", "f1_score", "log_loss", "hamming_loss"]
    normalizers = ["none", "zscore"]
    idx = 1
    for norm in normalizers:
        for win in windows:
            for clf in classifiers:
                for merge in merge_scores:
                    scale = 96 + 16 * idx
                    rows.append(ClapBranchConfig(
                        branch_name=f"CLaP_b{idx:02d}_{win}_{clf}_{merge}_{norm}",
                        scale_id=scale,
                        step_id=max(1, scale // 4),
                        init_cps_source="clasp",
                        fallback_uniform_segments=int(args.fallback_uniform_segments),
                        normalize_input=norm,
                        clap_window_size=win,
                        clap_classifier=clf,
                        clap_merge_score=merge,
                    ))
                    idx += 1
    return rows

def load_clap_branch_config(path: Path | None, args) -> list[ClapBranchConfig]:
    if path is None:
        return default_clap_branches(args)
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CLaP branch config not found: {path}")
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        return default_clap_branches(args)
    reader = csv.DictReader(StringIO("\n".join(lines)))
    branches = []
    for row_idx, row in enumerate(reader, start=2):
        enabled = str(row.get("enabled", "1")).strip().lower()
        if enabled in {"0", "false", "no", "n", "off"}:
            continue
        name = _parse_str(row.get("branch_name"), f"CLaP_b{row_idx:02d}")
        branches.append(ClapBranchConfig(
            branch_name=name,
            scale_id=_parse_int(row.get("scale_id"), row_idx * 16),
            step_id=_parse_int(row.get("step_id"), max(1, row_idx * 4)),
            init_cps_source=_parse_str(row.get("init_cps_source"), "clasp"),
            fallback_uniform_segments=_parse_int(row.get("fallback_uniform_segments"), int(args.fallback_uniform_segments)),
            normalize_input=_parse_str(row.get("normalize_input"), "none"),
            clap_window_size=_parse_str(row.get("clap_window_size"), "suss"),
            clap_classifier=_parse_str(row.get("clap_classifier"), "rocket"),
            clap_merge_score=_parse_str(row.get("clap_merge_score"), "cgain"),
        ))
    if not branches:
        raise ValueError(f"No enabled CLaP branches in {path}")
    return branches


def load_mocap(repo_root: Path, max_count: int | None = None) -> list[SeriesCase]:
    import numpy as np
    import pandas as pd
    base = repo_root / "Time2State" / "data" / "MoCap" / "4d"
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases: list[SeriesCase] = []
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
    keys = [normalize_dataset_key(x) for x in args.datasets]
    if keys == ["mocap"]:
        cases = load_mocap(args.repo_root, args.max_series_per_dataset)
        return cases, [{"dataset": "mocap", "key": "mocap", "ok": True, "cases": len(cases), "loader": "self_contained_mocap_loader"}]

    loader_path = args.loader_runner
    if loader_path is None:
        loader_path = args.repo_root / "our" / "_shared" / "our_multit2s_runner.py"
    module = import_module_from_path(loader_path, "external_our_multit2s_runner_loader")
    runtime = module.import_runtime(args.repo_root)
    cases, status = module.load_cases(args, runtime)
    out: list[SeriesCase] = []
    for case in cases:
        out.append(SeriesCase(case.dataset, case.case_id, case.data, case.labels, getattr(case, "fit_data", None)))
    return out, status


def import_meta_module(args: argparse.Namespace):
    loader_path = args.loader_runner
    if loader_path is None:
        loader_path = args.repo_root / "our" / "_shared" / "our_multit2s_runner.py"
    return import_module_from_path(loader_path, "external_our_multit2s_runner_meta")


def matches_case_id(case_id: str, wanted: str) -> bool:
    case_text = str(case_id).strip()
    wanted_text = str(wanted).strip()
    if case_text == wanted_text:
        return True
    if Path(case_text).stem == wanted_text:
        return True
    if case_text.lower() == wanted_text.lower():
        return True
    if Path(case_text).stem.lower() == wanted_text.lower():
        return True
    return False


def parse_index_list(value: str) -> list[int]:
    out = []
    for part in split_values(value):
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def apply_case_selection(cases: list[SeriesCase], args: argparse.Namespace) -> list[SeriesCase]:
    only = split_values(args.only_case_ids)
    if only:
        cases = [case for case in cases if any(matches_case_id(case.case_id, x) for x in only)]
    indexes = parse_index_list(args.case_indexes or args.rounds)
    if indexes:
        selected = []
        for idx in indexes:
            pos = idx - 1
            if 0 <= pos < len(cases):
                selected.append(cases[pos])
        cases = selected
    priority = split_values(args.priority_case_ids)
    if priority:
        front = [case for case in cases if any(matches_case_id(case.case_id, x) for x in priority)]
        rest = [case for case in cases if not any(matches_case_id(case.case_id, x) for x in priority)]
        cases = front + rest
    return cases


def add_clap_imports(clap_repo: Path):
    clap_repo = Path(clap_repo).resolve()
    if not clap_repo.exists():
        raise FileNotFoundError(f"CLaP repository not found: {clap_repo}")
    if str(clap_repo) in sys.path:
        sys.path.remove(str(clap_repo))
    sys.path.insert(0, str(clap_repo))
    from src.clap import CLaP
    from claspy.segmentation import BinaryClaSPSegmentation
    return CLaP, BinaryClaSPSegmentation


def clean_numeric_array(data, np):
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def maybe_limit_rows(data, labels, limit_rows: int | None, np):
    if limit_rows is None or int(limit_rows) <= 0:
        return data, labels
    n = min(int(limit_rows), len(data), len(labels))
    return data[:n], labels[:n]


def normalize_input_array(ts, mode: str, np):
    mode = str(mode or "none").lower()
    arr = np.asarray(ts, dtype=float)
    if mode == "none":
        return arr
    if arr.ndim == 1:
        work = arr.reshape(-1, 1)
    else:
        work = arr.copy()
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
        raise ValueError(f"Unsupported normalize_input mode: {mode}")
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
            x = subsample_for_adf(ts, args.adf_sample_max, np)
            p = adfuller(x)[1]
        else:
            p_values = []
            for dim in range(ts.shape[1]):
                x = subsample_for_adf(ts[:, dim], args.adf_sample_max, np)
                try:
                    p_values.append(adfuller(x)[1])
                except Exception:
                    continue
            p = float(np.median(p_values)) if p_values else 1.0
    except Exception:
        p = 1.0
    return "znormed_euclidean_distance" if p < 0.05 else "euclidean_distance"


def uniform_change_points(n: int, n_segments: int, np):
    n_segments = max(2, int(n_segments))
    cps = np.linspace(0, n, n_segments + 1, dtype=int)[1:-1]
    cps = [int(x) for x in cps if 0 < int(x) < n]
    return np.asarray(sorted(set(cps)), dtype=int)


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


def fit_initial_cps(ts, branch: ClapBranchConfig, args, BinaryClaSPSegmentation, np):
    if str(branch.init_cps_source).lower() == "uniform":
        return uniform_change_points(len(ts), branch.fallback_uniform_segments, np), "uniform"
    try:
        distance = choose_clasp_distance(ts, args, np)
        clasp = BinaryClaSPSegmentation(distance=distance, n_jobs=int(args.n_jobs))
        cps = clasp.fit_predict(ts)
        cps = safe_change_points(cps, len(ts), np)
        if len(cps) == 0:
            raise ValueError("ClaSP returned no change points")
        return cps, f"clasp:{distance}"
    except Exception as exc:
        cps = uniform_change_points(len(ts), branch.fallback_uniform_segments, np)
        return cps, f"uniform_fallback_after_clasp_error:{repr(exc)}"


def make_clap_kwargs(branch: ClapBranchConfig, args) -> dict[str, object]:
    kwargs: dict[str, object] = {"n_jobs": int(args.n_jobs)}
    if str(branch.clap_window_size).strip():
        kwargs["window_size"] = str(branch.clap_window_size).strip()
    if str(branch.clap_classifier).strip():
        kwargs["classifier"] = str(branch.clap_classifier).strip()
    if str(branch.clap_merge_score).strip():
        kwargs["merge_score"] = str(branch.clap_merge_score).strip()
    return kwargs


def cps_labels_to_sequence(cps, seg_labels, n: int, np):
    cps = safe_change_points(cps, n, np).tolist()
    labels = [int(x) for x in list(seg_labels)]
    n_segments = len(cps) + 1
    if len(labels) < n_segments:
        if len(labels) == 0:
            labels = [0] * n_segments
        else:
            labels = labels + [labels[-1]] * (n_segments - len(labels))
    elif len(labels) > n_segments:
        labels = labels[:n_segments]
    out = np.zeros(n, dtype=int)
    start = 0
    for idx, end in enumerate(cps + [n]):
        out[start:end] = int(labels[idx])
        start = end
    out, _ = reorder_label(out, np)
    return out


def run_one_clap_branch(data, branch: ClapBranchConfig, args, CLaP, BinaryClaSPSegmentation, np):
    ts = clean_numeric_array(data, np)
    ts = normalize_input_array(ts, branch.normalize_input, np)
    init_cps, init_source = fit_initial_cps(ts, branch, args, BinaryClaSPSegmentation, np)
    clap = CLaP(**make_clap_kwargs(branch, args))
    clap.fit(ts, np.asarray(init_cps, dtype=int))
    final_cps = safe_change_points(clap.get_change_points(), len(ts), np)
    segment_labels = np.asarray(clap.get_segment_labels(), dtype=int)
    pred_seq = cps_labels_to_sequence(final_cps, segment_labels, len(ts), np)
    return init_cps, final_cps, segment_labels, pred_seq, init_source


def read_existing_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                done.add((row.get("dataset", ""), row.get("case_id", "")))
    return done


def append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_prediction_csv(path: Path, labels_true, labels_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "true_label", "pred_label"])
        for idx, (yt, yp) in enumerate(zip(labels_true, labels_pred)):
            writer.writerow([idx, int(yt), int(yp)])


def run_clap_pid_meta_case(case: SeriesCase, branches: list[ClapBranchConfig], args, CLaP, BinaryClaSPSegmentation, meta, meta_runtime, np):
    data = clean_numeric_array(case.data, np)
    labels = np.asarray(case.labels, dtype=int)
    data, labels = maybe_limit_rows(data, labels, args.limit_rows, np)

    branch_sequences = []
    branch_metrics = []
    raw_branch_rows = []

    for bidx, branch in enumerate(branches, start=1):
        b_start = time.time()
        b_status = "ok"
        b_error = ""
        try:
            init_cps, final_cps, seg_labels, seq, init_source = run_one_clap_branch(
                data, branch, args, CLaP, BinaryClaSPSegmentation, np
            )
            seq = align_sequence(seq, len(data), np)
            seq, _ = reorder_label(seq, np)
            states = int(len(np.unique(seq)))
            segments = int(meta.count_segments(seq))

                                                                                           
                                                                                               
                                                                                              
            n_branch_eval = min(len(labels), len(seq))
            branch_ari = adjusted_rand_index(labels[:n_branch_eval], seq[:n_branch_eval])
            branch_nmi = normalized_mutual_information(labels[:n_branch_eval], seq[:n_branch_eval])

            branch_sequences.append((branch.branch_name, int(branch.scale_id), int(branch.step_id), seq))
            branch_metrics.append({
                "branch": branch.branch_name,
                "branch_idx": bidx,
                "win": int(branch.scale_id),
                "step": int(branch.step_id),
                "states": states,
                "segments": segments,
                "init_source": init_source,
                "init_cps_count": int(len(init_cps)),
                "final_cps_count": int(len(final_cps)),
                "clap_window_size": branch.clap_window_size,
                "clap_classifier": branch.clap_classifier,
                "clap_merge_score": branch.clap_merge_score,
                "normalize_input": branch.normalize_input,
            })
        except Exception as exc:
            init_cps, final_cps, seg_labels = [], [], []
            seq = np.zeros(len(data), dtype=int)
            states = 0
            segments = 0
            branch_ari = None
            branch_nmi = None
            init_source = "error"
            b_status = "error"
            b_error = repr(exc)

        raw_branch_rows.append({
            "dataset": str(case.dataset),
            "case_id": str(case.case_id),
            "branch": branch.branch_name,
            "branch_idx": bidx,
            "scale_id": int(branch.scale_id),
            "step_id": int(branch.step_id),
            "clap_window_size": branch.clap_window_size,
            "clap_classifier": branch.clap_classifier,
            "clap_merge_score": branch.clap_merge_score,
            "normalize_input": branch.normalize_input,
            "init_source": init_source,
            "states": states,
            "segments": segments,
            "branch_ari": "" if branch_ari is None else f"{branch_ari:.10f}",
            "branch_nmi": "" if branch_nmi is None else f"{branch_nmi:.10f}",
            "init_cps_count": len(init_cps),
            "final_cps_count": len(final_cps),
            "seconds": f"{time.time() - b_start:.4f}",
            "status": b_status,
            "error": b_error,
        })

    if len(branch_sequences) == 0:
        raise RuntimeError("All CLaP branches failed.")

    peer_rows = meta.compute_peer_reliability(branch_sequences, args, meta_runtime)
    for row, peer in zip(branch_metrics, peer_rows):
        row.update(peer)
    pid_rows = meta.compute_pid_reliability(branch_sequences, branch_metrics, args, meta_runtime)
    for row, pid in zip(branch_metrics, pid_rows):
        row.update(pid)

    selected_indices, ranked_branch_metrics = meta.select_top_k_branch_indices(
        branch_metrics, int(args.select_top_k_branches), str(args.branch_select_metric)
    )

    if not selected_indices:
        selected_indices = set(range(len(branch_sequences)))

    selected_pairs = [
        (idx, seq_tuple)
        for idx, seq_tuple in enumerate(branch_sequences)
        if idx in selected_indices
    ]

    selected_sequences = [seq_tuple for _idx, seq_tuple in selected_pairs]

                
                                                                          
                                                                                    
                                                                                       
    selected_metrics = [
        branch_metrics[idx]
        for idx, _seq_tuple in selected_pairs
    ]

    if len(selected_sequences) != len(selected_metrics):
        raise RuntimeError(
            f"Selected sequence/metric mismatch: "
            f"{len(selected_sequences)} sequences vs {len(selected_metrics)} metrics"
        )

    for (_idx, seq_tuple), metric_row in zip(selected_pairs, selected_metrics):
        if str(seq_tuple[0]) != str(metric_row.get("branch", "")):
            raise RuntimeError(
                "Selected branch order mismatch: "
                f"sequence={seq_tuple[0]!r}, metric={metric_row.get('branch', '')!r}"
            )

    indicator_df, state_info_df = meta.build_indicator(selected_sequences, meta_runtime)
    weighted_state_info = meta.attach_branch_weights_to_state_info(
        state_info_df, selected_metrics, str(args.meta_vote_weight_mode)
    )
    run_state_counts = [int(len(np.unique(item[3]))) for item in selected_sequences]
    best_k, all_k = meta.choose_meta_k(
        indicator_df, weighted_state_info, run_state_counts, args, meta_runtime, int(args.meta_min_len)
    )
    meta_seq = np.asarray(best_k["seq"], dtype=int)
    meta_seq = align_sequence(meta_seq, len(labels), np)
    meta_seq, _ = reorder_label(meta_seq, np)

    n = min(len(labels), len(meta_seq))
    labels_eval = labels[:n]
    pred_eval = meta_seq[:n]
    ari = adjusted_rand_index(labels_eval, pred_eval)
    nmi = normalized_mutual_information(labels_eval, pred_eval)

    for idx, row in enumerate(raw_branch_rows):
        match = next((m for m in ranked_branch_metrics if int(m.get("branch_idx", -1)) == int(row["branch_idx"])), None)
        if match is not None:
            row.update({
                "branch_rank": match.get("branch_rank", ""),
                "selected_for_meta": match.get("selected_for_meta", 0),
                "branch_selection_score": match.get("branch_selection_score", ""),
                "health": match.get("health", ""),
                "peer_consensus": match.get("peer_consensus", ""),
                "peer_reliability_norm": match.get("peer_reliability_norm", ""),
                "pid_p": match.get("pid_p", ""),
                "pid_i": match.get("pid_i", ""),
                "pid_d": match.get("pid_d", ""),
                "pid_score_norm": match.get("pid_score_norm", ""),
                "pid_weight_norm": match.get("pid_weight_norm", ""),
            })
        else:
            row.update({
                "branch_rank": "",
                "selected_for_meta": 0,
                "branch_selection_score": "",
                "health": "",
                "peer_consensus": "",
                "peer_reliability_norm": "",
                "pid_p": "",
                "pid_i": "",
                "pid_d": "",
                "pid_score_norm": "",
                "pid_weight_norm": "",
            })

    return {
        "labels_eval": labels_eval,
        "pred_eval": pred_eval,
        "ari": ari,
        "nmi": nmi,
        "true_states": int(len(np.unique(labels_eval))),
        "pred_states": int(len(np.unique(pred_eval))),
        "selected_count": int(len(selected_sequences)),
        "candidate_count": int(len(branch_sequences)),
        "selected_branches": [x[0] for x in selected_sequences],
        "selected_K": int(best_k["K"]),
        "k_selection_score": float(best_k["selection_score"]),
        "k_dominant_ratio": float(best_k["dominant_ratio"]),
        "k_balance_entropy": float(best_k["balance_entropy"]),
        "k_mean_segment_length": float(best_k["mean_segment_length"]),
        "branch_rows": raw_branch_rows,
        "all_k_rows": all_k,
    }


def main() -> int:
    args = parse_args()
    args.repo_root = Path(args.repo_root).resolve()
    args.out_dir = Path(args.out_dir).resolve()
    args.clap_repo = Path(args.clap_repo).resolve()
    if args.loader_runner is not None:
        args.loader_runner = Path(args.loader_runner).resolve()
    if args.clap_branch_config_txt is not None:
        args.clap_branch_config_txt = Path(args.clap_branch_config_txt).resolve()

    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    CLaP, BinaryClaSPSegmentation = add_clap_imports(args.clap_repo)
    meta = import_meta_module(args)
    meta_runtime = {"np": np, "pd": pd, "AgglomerativeClustering": AgglomerativeClustering}

    branches = load_clap_branch_config(args.clap_branch_config_txt, args)
    cases, dataset_status = load_cases(args)
    cases = apply_case_selection(cases, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.out_dir / "predictions"
    case_csv = args.out_dir / "all_case_results.csv"
    branch_csv = args.out_dir / "all_branch_results.csv"
    k_csv = args.out_dir / "all_k_selection_results.csv"
    status_json = args.out_dir / "run_status.json"

    done = read_existing_completed(case_csv) if args.skip_completed else set()

    case_fieldnames = [
        "dataset", "case_id", "ari", "nmi", "true_states", "pred_states", "selected_K",
        "candidate_branches", "selected_branches_count", "selected_branches", "k_selection_score",
        "k_dominant_ratio", "k_balance_entropy", "k_mean_segment_length", "seconds", "status", "error",
    ]
    branch_fieldnames = [
        "dataset", "case_id", "branch", "branch_idx", "scale_id", "step_id", "clap_window_size",
        "clap_classifier", "clap_merge_score", "normalize_input", "init_source", "states", "segments",
        "branch_ari", "branch_nmi",
        "init_cps_count", "final_cps_count", "seconds", "status", "error", "branch_rank",
        "selected_for_meta", "branch_selection_score", "health", "peer_consensus", "peer_reliability_norm",
        "pid_p", "pid_i", "pid_d", "pid_score_norm", "pid_weight_norm",
    ]
    k_fieldnames = [
        "dataset", "case_id", "K", "selection_score", "dominant_ratio", "balance_entropy",
        "coherence", "mean_segment_length", "segment_count",
    ]

    print("============================================================")
    print("CLaP multi-branch + PID + meta runner")
    print("Repo root :", args.repo_root)
    print("CLaP repo :", args.clap_repo)
    print("Output    :", args.out_dir)
    print("Datasets  :", " ".join(args.datasets))
    print("Cases     :", len(cases))
    print("Branches  :", len(branches))
    print("Selection :", args.branch_select_metric, "top", args.select_top_k_branches)
    print("Vote mode :", args.meta_vote_weight_mode)
    print("Metric    :", METRIC_BACKEND)
    print("============================================================")

    all_case_rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        key = (str(case.dataset), str(case.case_id))
        if key in done:
            print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id} skipped")
            continue

        print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id}")
        start_time = time.time()
        status = "ok"
        error = ""
        try:
            result = run_clap_pid_meta_case(case, branches, args, CLaP, BinaryClaSPSegmentation, meta, meta_runtime, np)
            if args.save_predictions:
                safe_case = str(case.case_id).replace("/", "_").replace("\\", "_").replace(":", "_")
                save_prediction_csv(predictions_dir / f"{case.dataset}__{safe_case}.csv", result["labels_eval"], result["pred_eval"])
            append_rows(branch_csv, result["branch_rows"], branch_fieldnames)
            k_rows = []
            for kr in result["all_k_rows"]:
                k_rows.append({
                    "dataset": str(case.dataset),
                    "case_id": str(case.case_id),
                    "K": int(kr["K"]),
                    "selection_score": f"{float(kr['selection_score']):.10f}",
                    "dominant_ratio": f"{float(kr['dominant_ratio']):.10f}",
                    "balance_entropy": f"{float(kr['balance_entropy']):.10f}",
                    "coherence": f"{float(kr.get('coherence', 0.0)):.10f}",
                    "mean_segment_length": f"{float(kr['mean_segment_length']):.10f}",
                    "segment_count": int(kr["segment_count"]),
                })
            append_rows(k_csv, k_rows, k_fieldnames)
            ari = float(result["ari"])
            nmi = float(result["nmi"])
            true_states = int(result["true_states"])
            pred_states = int(result["pred_states"])
            selected_K = int(result["selected_K"])
            candidate_count = int(result["candidate_count"])
            selected_count = int(result["selected_count"])
            selected_branches = result["selected_branches"]
            k_selection_score = float(result["k_selection_score"])
            k_dominant_ratio = float(result["k_dominant_ratio"])
            k_balance_entropy = float(result["k_balance_entropy"])
            k_mean_segment_length = float(result["k_mean_segment_length"])
        except Exception as exc:
            status = "error"
            error = repr(exc)
            ari = 0.0
            nmi = 0.0
            true_states = int(len(set(map(int, list(case.labels))))) if getattr(case, "labels", None) is not None else 0
            pred_states = 0
            selected_K = 0
            candidate_count = len(branches)
            selected_count = 0
            selected_branches = []
            k_selection_score = 0.0
            k_dominant_ratio = 0.0
            k_balance_entropy = 0.0
            k_mean_segment_length = 0.0

        seconds = time.time() - start_time
        print(f"  ARI={ari:.4f} NMI={nmi:.4f} K={pred_states} metaK={selected_K} selected={selected_count}/{candidate_count} seconds={seconds:.1f} status={status}")
        if error:
            print("  ERROR:", error)

        row = {
            "dataset": str(case.dataset),
            "case_id": str(case.case_id),
            "ari": f"{ari:.10f}",
            "nmi": f"{nmi:.10f}",
            "true_states": true_states,
            "pred_states": pred_states,
            "selected_K": selected_K,
            "candidate_branches": candidate_count,
            "selected_branches_count": selected_count,
            "selected_branches": json.dumps(selected_branches, ensure_ascii=False),
            "k_selection_score": f"{k_selection_score:.10f}",
            "k_dominant_ratio": f"{k_dominant_ratio:.10f}",
            "k_balance_entropy": f"{k_balance_entropy:.10f}",
            "k_mean_segment_length": f"{k_mean_segment_length:.10f}",
            "seconds": f"{seconds:.4f}",
            "status": status,
            "error": error,
        }
        append_rows(case_csv, [row], case_fieldnames)
        all_case_rows.append(row)

    ok_rows = [r for r in all_case_rows if r["status"] == "ok"]
    ari_vals = [float(r["ari"]) for r in ok_rows]
    nmi_vals = [float(r["nmi"]) for r in ok_rows]
    summary = {
        "repo_root": str(args.repo_root),
        "clap_repo": str(args.clap_repo),
        "out_dir": str(args.out_dir),
        "datasets": args.datasets,
        "dataset_status": dataset_status,
        "n_cases_loaded": len(cases),
        "n_cases_run_this_time": len(all_case_rows),
        "n_ok_this_time": len(ok_rows),
        "mean_ari_this_time": float(np.mean(ari_vals)) if ari_vals else None,
        "mean_nmi_this_time": float(np.mean(nmi_vals)) if nmi_vals else None,
        "metric_backend": METRIC_BACKEND,
        "settings": {
            "n_branches": len(branches),
            "select_top_k_branches": args.select_top_k_branches,
            "branch_select_metric": args.branch_select_metric,
            "meta_vote_weight_mode": args.meta_vote_weight_mode,
            "meta_min_len": args.meta_min_len,
            "state_cluster_smooth": args.state_cluster_smooth,
            "pid_kp": args.pid_kp,
            "pid_ki": args.pid_ki,
            "pid_kd": args.pid_kd,
            "pid_softmax_tau": args.pid_softmax_tau,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
        },
    }
    status_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if ari_vals:
        print("============================================================")
        print(f"Mean ARI={float(np.mean(ari_vals)):.4f}  Mean NMI={float(np.mean(nmi_vals)):.4f}  n={len(ari_vals)}")
        print("Saved:", case_csv)
        print("Branches:", branch_csv)
        print("K table:", k_csv)
        print("============================================================")
    else:
        print("No successful cases in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
