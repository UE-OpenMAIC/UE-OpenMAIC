"""Run multi-scale T2S on the Time2State paper benchmark datasets."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence


THIS_DIR = Path(__file__).resolve().parent
OUR_ROOT = THIS_DIR.parent




REPO_ROOT = THIS_DIR.parents[2]
BENCH_DIR = REPO_ROOT / "multi_t2s_paper_benchmark"
TIME2STATE_ROOT = REPO_ROOT / "Time2State"
DEFAULT_OUT_DIR = OUR_ROOT / "results"
BASELINE_JSON = BENCH_DIR / "paper_baseline_values.json"
METRIC_BACKEND = "sklearn_adjusted_rand_score__sklearn_nmi_geometric"


MOCAP_INFO = {
    "amc_86_01.4d": {"n_segs": 4, "label": {588: 0, 1200: 1, 2006: 0, 2530: 2, 3282: 0, 4048: 3, 4579: 2}},
    "amc_86_02.4d": {"n_segs": 8, "label": {1009: 0, 1882: 1, 2677: 2, 3158: 3, 4688: 4, 5963: 0, 7327: 5, 8887: 6, 9632: 7, 10617: 0}},
    "amc_86_03.4d": {"n_segs": 7, "label": {872: 0, 1938: 1, 2448: 2, 3470: 0, 4632: 3, 5372: 4, 6182: 5, 7089: 6, 8401: 0}},
    "amc_86_07.4d": {"n_segs": 6, "label": {1060: 0, 1897: 1, 2564: 2, 3665: 1, 4405: 2, 5169: 3, 5804: 4, 6962: 0, 7806: 5, 8702: 0}},
    "amc_86_08.4d": {"n_segs": 9, "label": {1062: 0, 1904: 1, 2661: 2, 3282: 3, 3963: 4, 4754: 5, 5673: 6, 6362: 4, 7144: 7, 8139: 8, 9206: 0}},
    "amc_86_09.4d": {"n_segs": 5, "label": {921: 0, 1275: 1, 2139: 2, 2887: 3, 3667: 4, 4794: 0}},
    "amc_86_10.4d": {"n_segs": 4, "label": {2003: 0, 3720: 1, 4981: 0, 5646: 2, 6641: 3, 7583: 0}},
    "amc_86_11.4d": {"n_segs": 4, "label": {1231: 0, 1693: 1, 2332: 2, 2762: 1, 3386: 3, 4015: 2, 4665: 1, 5674: 0}},
    "amc_86_14.4d": {"n_segs": 3, "label": {671: 0, 1913: 1, 2931: 0, 4134: 2, 5051: 0, 5628: 1, 6055: 2}},
}


@dataclass
class SeriesCase:
    dataset: str
    case_id: str
    data: object
    labels: object




    fit_data: object | None = None


@dataclass
class BranchConfig:
    dataset: str
    branch_name: str
    win: int
    step: int
    m: int
    n: int
    out_channels: int
    nb_steps: int
    kernel_size: int | None = None
    win_type: str = "hanning"
    branch_min_len: int | None = None





    meta_min_len: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "mocap", "actrectut"])
    parser.add_argument("--branches", default="128:50,128:100,256:50,256:100,512:50,512:100")
    parser.add_argument("--branch-config-txt", type=Path, default=None, help="Optional CSV-like txt config. If provided, each dataset can define its own branches and per-branch T2S parameters.")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional case_id filter, e.g. TwoLeadECG UWaveGestureLibraryAll.")
    parser.add_argument("--only-case-ids", default="", help="Comma/space separated exact case_id values to run exclusively.")
    parser.add_argument("--priority-case-ids", default="", help="Comma/space separated case_id values to run first without filtering.")
    parser.add_argument("--rounds", default="", help="Debug only: comma/space separated 1-based loaded-case indexes to run.")
    parser.add_argument("--case-indexes", default="", help="Debug only: alias for --rounds.")
    parser.add_argument("--ucrseg-adaptive-scales", action="store_true", help="Generate UCR-SEG branches per case by scaling the TwoLeadECG base branch table according to sequence length.")
    parser.add_argument("--ucrseg-adaptive-base-len", type=int, default=471, help="Reference length for the UCR-SEG adaptive branch table. Default: TwoLeadECG rows=471.")
    parser.add_argument("--ucrseg-adaptive-min-win", type=int, default=4, help="Minimum adaptive window size after scaling.")
    parser.add_argument("--ucrseg-adaptive-min-step", type=int, default=1, help="Minimum adaptive step size after scaling.")
    parser.add_argument("--ucrseg-normal-scales", action="store_true", help="Generate UCR-SEG window ratios from a truncated normal distribution.")
    parser.add_argument("--ucrseg-normal-mu", type=float, default=0.18, help="Mean of normal window-ratio distribution.")
    parser.add_argument("--ucrseg-normal-sigma", type=float, default=0.07, help="Std of normal window-ratio distribution.")
    parser.add_argument("--ucrseg-normal-low", type=float, default=0.05, help="Lower truncation ratio for normal window ratios.")
    parser.add_argument("--ucrseg-normal-high", type=float, default=0.35, help="Upper truncation ratio for normal window ratios.")
    parser.add_argument("--ucrseg-normal-win-count", type=int, default=16, help="Number of window ratios sampled from the truncated normal distribution.")
    parser.add_argument("--ucrseg-normal-step-ratios", default="0.25,0.50,0.75,1.00", help="Comma-separated step/window ratios for normal-scale experts.")
    parser.add_argument("--ucrseg-worst-first", action="store_true", help="Run historically difficult UCR-SEG cases first.")
    parser.add_argument("--max-series-per-dataset", type=int, default=None)
    parser.add_argument("--max-synthetic", type=int, default=None, help="Override max-series-per-dataset for Synthetic.")
    parser.add_argument("--nb-steps", type=int, default=20)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--meta-min-len", type=int, default=24)
    parser.add_argument("--state-cluster-smooth", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=os.environ.get("T2S_DEVICE", "auto"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("T2S_GPU", "0")))
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug only: truncate each time series.")
    parser.add_argument("--public-data-root", type=Path, default=Path(os.environ.get("PUBLIC_TS_DATASETS_DIR", THIS_DIR / "public_ts_datasets")), help="Root directory created by download_pamap2_uschad_ucrseg.py.")
    parser.add_argument("--public-max-rows", type=int, default=0, help="Uniformly downsample public long-series cases to this many rows; set <=0 to disable.")
    parser.add_argument(
        "--pamap2-feature-mode",
        choices=["auto", "paper9acc", "full_sensor"],
        default="auto",
        help=(
            "PAMAP2 feature protocol. auto keeps the original paper9acc behavior, "
            "except when the output/branch path contains PAMAP2_zero, where it uses "
            "full_sensor for the dedicated PAMAP2_zero experiment."
        ),
    )
    parser.add_argument(
        "--pamap2-remove-zero",
        action="store_true",
        help=(
            "Remove PAMAP2 frames whose activity_id is 0. This is automatically enabled "
            "for the dedicated PAMAP2_zero experiment when the output/branch path contains PAMAP2_zero."
        ),
    )
    parser.add_argument("--skip-completed", action="store_true", help="If case_results.csv exists, skip dataset/case_id pairs already present and append only missing cases.")
    parser.add_argument("--select-top-k-branches", type=int, default=0, help="Run all candidate branches, then use only the top K branches for meta aggregation. 0 means use all branches.")
    parser.add_argument(
        "--branch-select-metric",
        choices=["PEER", "PID", "M2QD"],
        default="PEER",
        help=(
            "Metric for selecting top branches. "
            "Both choices are unsupervised and use only predicted sequences."
        ),
    )
    parser.add_argument(
        "--meta-vote-weight-mode",
        choices=["uniform", "branch_reliability", "pid_weight"],
        default="branch_reliability",
        help=(
            "Meta voting weights. uniform uses the original equal vote; "
            "branch_reliability uses unsupervised PEER reliability; "
            "pid_weight uses unsupervised PID reliability."
        ),
    )
    parser.add_argument("--peer-health-weight", type=float, default=0.45, help="PEER score weight for branch health.")
    parser.add_argument("--peer-consensus-weight", type=float, default=0.55, help="PEER score weight for leave-one-branch-out consensus.")
    parser.add_argument("--pid-kp", type=float, default=0.45, help="PID-inspired P weight: current quality.")
    parser.add_argument("--pid-ki", type=float, default=0.35, help="PID-inspired I weight: neighborhood stability.")
    parser.add_argument("--pid-kd", type=float, default=0.20, help="PID-inspired D penalty: fragmentation/instability.")
    parser.add_argument("--pid-win-ref", type=float, default=256.0, help="Deprecated: ignored when PID scale prior is disabled.")
    parser.add_argument("--pid-scale-sigma", type=float, default=0.55, help="Deprecated: ignored when PID scale prior is disabled.")
    parser.add_argument("--pid-softmax-tau", type=float, default=0.15, help="Softmax temperature for PID reliability normalization.")
    parser.add_argument("--m2qd-alpha", type=float, default=0.5, help="M2QD balance between quality and diversity; 0.5 gives equal weight.")
    parser.add_argument(
        "--fusion-backend",
        choices=["meta", "cspa", "ec_tdwm"],
        default="ec_tdwm",
        help="Final fusion backend. ec_tdwm uses an adapted EC-TDWM three-level dynamic weighting consensus.",
    )
    parser.add_argument(
        "--cspa-k-mode",
        choices=["median", "mean", "max", "min"],
        default="median",
        help="How to choose final CSPA K from selected branch state counts. Default: median, label-free.",
    )
    parser.add_argument(
        "--cspa-engine",
        choices=["auto", "agglomerative", "minibatch_kmeans"],
        default="auto",
        help="CSPA consensus solver. auto uses agglomerative for short sequences and MiniBatchKMeans for long ones.",
    )
    parser.add_argument("--cspa-agglomerative-max-rows", type=int, default=3000, help="Max rows for exact agglomerative CSPA in auto mode.")
    parser.add_argument("--cspa-batch-size", type=int, default=4096, help="MiniBatchKMeans batch size for CSPA on long sequences.")
    parser.add_argument("--cspa-apply-meta-min-len", action="store_true", help="Apply the common meta_min_len short-segment merge after CSPA consensus.")
    parser.add_argument(
        "--ec-tdwm-k-mode",
        choices=["median", "mean", "max", "min"],
        default="median",
        help="How to choose final EC-TDWM K from branch state counts. Default: median, label-free.",
    )
    parser.add_argument(
        "--ec-tdwm-engine",
        choices=["auto", "agglomerative", "minibatch_kmeans"],
        default="auto",
        help="Solver used to update the consensus labels inside EC-TDWM.",
    )
    parser.add_argument("--ec-tdwm-max-iter", type=int, default=20, help="Maximum alternating update iterations for adapted EC-TDWM.")
    parser.add_argument("--ec-tdwm-tol", type=float, default=1e-5, help="Convergence tolerance based on label-change ratio.")
    parser.add_argument("--ec-tdwm-agglomerative-max-rows", type=int, default=3000, help="Max rows for agglomerative updates in auto mode.")
    parser.add_argument("--ec-tdwm-batch-size", type=int, default=4096, help="MiniBatchKMeans batch size for EC-TDWM on long sequences.")
    parser.add_argument("--ec-tdwm-apply-meta-min-len", action="store_true", help="Apply meta_min_len short-segment merge after EC-TDWM consensus.")
    parser.add_argument("--allow-oracle-debug", action="store_true", help="Accepted for compatibility with older debug configs.")
    return parser.parse_args()


def add_time2state_imports(repo_root: Path) -> None:
    t2s_root = repo_root / "Time2State"
    deps_root = BENCH_DIR / "_deps"
    use_local_deps = str(os.environ.get("T2S_USE_LOCAL_DEPS", "1")).strip().lower() not in {"0", "false", "no"}
    paths = []
    if deps_root.exists() and use_local_deps:
        paths.append(str(deps_root))
    paths.extend([str(t2s_root), str(repo_root), str(BENCH_DIR)])


    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def import_runtime(repo_root: Path):
    add_time2state_imports(repo_root)
    missing = []
    for name in ["numpy", "pandas", "sklearn", "torch", "scipy"]:
        try:
            __import__(name)
        except Exception as exc:
            missing.append({"package": name, "error": repr(exc)})
    if missing:
        raise RuntimeError("Missing required packages: " + json.dumps(missing, ensure_ascii=False))

    import numpy as np
    import pandas as pd
    import scipy.io
    import torch
    from sklearn.cluster import AgglomerativeClustering

    from Time2State.adapers import CausalConv_LSE_Adaper
    from Time2State.clustering import DPGMM
    from Time2State.default_params import params_LSE
    from Time2State.time2state import Time2State



    try:
        from TSpy.utils import normalize as tspy_normalize
    except Exception:
        tspy_normalize = None

    try:
        from TSpy.dataset import load_USC_HAD as tspy_load_USC_HAD
    except Exception:
        tspy_load_USC_HAD = None

    return {
        "np": np,
        "pd": pd,
        "scipy_io": scipy.io,
        "torch": torch,
        "AgglomerativeClustering": AgglomerativeClustering,
        "CausalConv_LSE_Adaper": CausalConv_LSE_Adaper,
        "DPGMM": DPGMM,
        "params_LSE": params_LSE,
        "Time2State": Time2State,
        "tspy_normalize": tspy_normalize,
        "tspy_load_USC_HAD": tspy_load_USC_HAD,
    }


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

        return float(
            normalized_mutual_info_score(
                true_list,
                pred_list,
                average_method="geometric",
            )
        )
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


def merge_short_segments(seq, min_len, np):
    seq = np.asarray(seq, dtype=int).copy()
    n = len(seq)
    if n == 0:
        return seq
    changed = True
    while changed:
        changed = False
        segments = []
        start = 0
        for idx in range(1, n):
            if seq[idx] != seq[start]:
                segments.append((start, idx - 1, seq[start]))
                start = idx
        segments.append((start, n - 1, seq[start]))
        if len(segments) <= 1:
            break
        for pos, (left, right, value) in enumerate(segments):
            length = right - left + 1
            if length >= min_len:
                continue
            if pos == 0:
                fill_value = segments[pos + 1][2]
            elif pos == len(segments) - 1:
                fill_value = segments[pos - 1][2]
            else:
                prev_len = segments[pos - 1][1] - segments[pos - 1][0] + 1
                next_len = segments[pos + 1][1] - segments[pos + 1][0] + 1
                fill_value = segments[pos - 1][2] if prev_len >= next_len else segments[pos + 1][2]
            if fill_value != value:
                seq[left : right + 1] = fill_value
                changed = True
                break
    return seq


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


def parse_branches(branches: str) -> list[tuple[int, int]]:
    out = []
    for part in branches.split(","):
        part = part.strip()
        if not part:
            continue
        win, step = part.split(":")
        out.append((int(win), int(step)))
    if not out:
        raise ValueError("No branches were provided")
    return out


def count_t2s_windows(n_rows: int, win: int, step: int) -> int:
    if n_rows < win:
        return 0
    return 1 + max(0, (n_rows - win) // step)


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower()
    key = key.replace("_", "-").replace(" ", "")
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


def _parse_optional_int(value, default=None):
    value = "" if value is None else str(value).strip()
    if value == "":
        return default
    return int(float(value))


def load_branch_config_txt(path: Path | None, args) -> dict[str, list[BranchConfig]]:
    """Read per-dataset/per-branch configuration from a CSV-like txt file."""
    if path is None:
        return {}

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"branch config txt not found: {path}")

    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)

    if not lines:
        raise ValueError(f"branch config txt is empty after removing comments: {path}")

    reader = csv.DictReader(lines)
    for col in ["dataset", "win", "step"]:
        if col not in (reader.fieldnames or []):
            raise ValueError(f"branch config txt missing required column '{col}'. fieldnames={reader.fieldnames}")

    cfg: dict[str, list[BranchConfig]] = {}
    for row_idx, row in enumerate(reader, start=2):
        enabled = str(row.get("enabled", "1")).strip().lower()
        if enabled in {"0", "false", "no", "n", "off"}:
            continue

        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            raise ValueError(f"branch config row {row_idx}: empty dataset")

        win = int(float(str(row.get("win", "")).strip()))
        step = int(float(str(row.get("step", "")).strip()))
        branch_name = str(row.get("branch_name", "")).strip()
        if not branch_name:
            branch_name = f"{normalize_dataset_key(dataset)}_w{win}_s{step}"

        branch = BranchConfig(
            dataset=dataset,
            branch_name=branch_name,
            win=win,
            step=step,
            m=_parse_optional_int(row.get("m"), args.m),
            n=_parse_optional_int(row.get("n"), args.n),
            out_channels=_parse_optional_int(row.get("out_channels"), args.out_channels),
            nb_steps=_parse_optional_int(row.get("nb_steps"), args.nb_steps),
            kernel_size=_parse_optional_int(row.get("kernel_size"), args.kernel_size),
            win_type=str(row.get("win_type", "") or "hanning").strip(),
            branch_min_len=_parse_optional_int(row.get("branch_min_len"), None),
            meta_min_len=_parse_optional_int(row.get("meta_min_len"), None),
        )
        cfg.setdefault(normalize_dataset_key(dataset), []).append(branch)

    return cfg


def default_branch_configs_for_case(case: SeriesCase, args) -> list[BranchConfig]:
    branches = parse_branches(args.branches)
    out = []
    for idx, (win, step) in enumerate(branches, start=1):
        out.append(
            BranchConfig(
                dataset=case.dataset,
                branch_name=f"run{idx}_w{win}_s{step}",
                win=int(win),
                step=int(step),
                m=int(args.m),
                n=int(args.n),
                out_channels=int(args.out_channels),
                nb_steps=int(args.nb_steps),
                kernel_size=int(args.kernel_size) if args.kernel_size is not None else None,
                win_type="hanning",
                branch_min_len=None,
                meta_min_len=None,
            )
        )
    return out


def branch_configs_for_case(case: SeriesCase, args) -> list[BranchConfig]:
    key = normalize_dataset_key(case.dataset)
    config_map = getattr(args, "branch_config_map", {}) or {}
    if key in config_map and config_map[key]:
        return config_map[key]
    return default_branch_configs_for_case(case, args)



def resolve_branch_min_len(branch: BranchConfig) -> int:
    if branch.branch_min_len is None:
        return max(2, int(branch.win) // 8)
    return int(branch.branch_min_len)


def resolve_meta_min_len_for_case(branches: list[BranchConfig], args) -> int:
    vals = [branch.meta_min_len for branch in branches if branch.meta_min_len is not None]
    if vals:
        return int(vals[0])
    return int(args.meta_min_len)

def normalize_data(data, np):
    """Fallback z-score normalization.

    The strict original loader below prefers TSpy.utils.normalize, because
    exp_of_Time2State.py calls normalize(data) from TSpy.utils. This fallback is
    only used if TSpy cannot be imported in the user's environment.
    """
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    arr = (arr - mean) / std
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _original_data_root(repo_root: Path) -> Path:
    """Match exp_of_Time2State.py: script_path/../data -> Time2State/data."""
    return repo_root / "Time2State" / "data"


def _original_normalize(data, runtime):
    """Use the same normalize() symbol imported by exp_of_Time2State.py."""
    norm = runtime.get("tspy_normalize")
    if norm is not None:
        return norm(data)
    return normalize_data(data, runtime["np"])


def _count_file_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def fill_nan_original(data, np):
    """Exact PAMAP2 NaN fill logic from exp_of_Time2State.py."""
    arr = np.asarray(data, dtype=float).copy()
    x_len, y_len = arr.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(arr[x, y]):
                arr[x, y] = arr[x - 1, y]
    return arr


def _find_existing_dir(candidates: list[Path], what: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {what}. Tried: " + " | ".join(str(c) for c in candidates))


def _max_count_for_dataset(args: argparse.Namespace, key: str) -> int | None:
    if key == "synthetic" and getattr(args, "max_synthetic", None) is not None:
        return int(args.max_synthetic)
    return getattr(args, "max_series_per_dataset", None)


def load_synthetic(repo_root: Path, runtime, max_count: int | None) -> list[SeriesCase]:
    """
    Strictly match exp_on_synthetic():

        prefix = data/synthetic_data_for_segmentation3/test
        pd.read_csv(..., usecols=range(4), skiprows=1)   # default header=0
        pd.read_csv(..., usecols=[4], skiprows=1)        # default header=0
        no normalize
        for i in range(100)
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = _original_data_root(repo_root) / "synthetic_data_for_segmentation3"
    cases = []
    for i in range(100):
        path = base / f"test{i}.csv"
        if not path.exists():
            continue
        df_x = pd.read_csv(path, usecols=range(4), skiprows=1)
        df_y = pd.read_csv(path, usecols=[4], skiprows=1)
        data = df_x.to_numpy()
        labels = df_y.to_numpy(dtype=int).flatten()
        n = min(len(data), len(labels))
        cases.append(SeriesCase("Synthetic", str(i), data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No strict Synthetic files found under {base} using test0.csv...test99.csv")
    return cases


def load_mocap(repo_root: Path, runtime, max_count: int | None) -> list[SeriesCase]:
    """
    Strictly match exp_on_MoCap():

        base = data/MoCap/4d/
        f_list = os.listdir(base_path); f_list.sort()
        pd.read_csv(..., sep=' ', usecols=range(0,4))    # default header=0
        no normalize
        labels = seg_to_label(dataset_info[fname]['label'])[:-1]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = _original_data_root(repo_root) / "MoCap" / "4d"
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases = []
    for path in files:
        if path.name not in MOCAP_INFO:
            continue
        df = pd.read_csv(path, sep=" ", usecols=range(0, 4))
        data = df.to_numpy()
        labels = seg_to_label(MOCAP_INFO[path.name]["label"], np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("MoCap", path.name, data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No strict MoCap .4d cases found under {base}")
    return cases


def load_actrectut(repo_root: Path, runtime, max_count: int | None) -> list[SeriesCase]:
    """
    Strictly match exp_on_ActRecTut():

        dir_list = ['subject1_walk', 'subject2_walk']
        for each dir repeat 10 times
        labels = reorder_label(data['labels'].flatten())
        data = data['data'][:,0:10]
        data = normalize(data)
        fit on this same data
    """
    np = runtime["np"]
    scipy_io = runtime["scipy_io"]
    base = _original_data_root(repo_root) / "ActRecTut"
    names = ["subject1_walk", "subject2_walk"]
    cases = []
    for name in names:
        mat_path = base / name / "data.mat"
        mat = scipy_io.loadmat(mat_path)
        labels_raw = mat["labels"].flatten()
        labels, _ = reorder_label(labels_raw, np)
        data = mat["data"][:, 0:10]
        data = _original_normalize(data, runtime)
        n = min(len(data), len(labels))
        for rep in range(10):
            cases.append(SeriesCase("ActRecTut", f"{name}{rep}", data[:n], labels[:n]))
            if max_count is not None and len(cases) >= max_count:
                return cases
    return cases


def get_public_data_root(args: argparse.Namespace) -> Path:
    """
    Kept for compatibility with launcher.py and run_status.json.

    The strict original loader uses Time2State/data first. For datasets that
    the original script did not read from public_ts_datasets, no downsampling,
    filtering, or public benchmark reformatting is applied.
    """
    public_root = getattr(args, "public_data_root", None)
    if public_root is not None:
        return Path(public_root).resolve()
    env_root = os.environ.get("PUBLIC_TS_DATASETS_DIR", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return (THIS_DIR / "public_ts_datasets").resolve()



def _is_pamap2_zero_run(args: argparse.Namespace | None) -> bool:
    """
    Detect the dedicated PAMAP2_zero experiment without requiring launcher.py changes.

    This keeps all existing experiments unchanged. The zero/full-sensor behavior is
    activated only when the run path explicitly contains "PAMAP2_zero" / "pamap2_zero",
    or when the user passes --pamap2-feature-mode full_sensor / --pamap2-remove-zero.
    """
    if args is None:
        return False
    parts = []
    for name in ["out_dir", "branch_config_txt"]:
        value = getattr(args, name, None)
        if value is not None:
            parts.append(str(value))
    text = " ".join(parts).replace("\\", "/").lower()
    return "pamap2_zero" in text or "pamap2-zero" in text


def _resolve_pamap2_feature_mode(args: argparse.Namespace | None) -> str:
    """Resolve PAMAP2 feature mode while preserving old default behavior."""
    mode = str(getattr(args, "pamap2_feature_mode", "auto") if args is not None else "auto").strip().lower()
    if mode == "auto":
        return "full_sensor" if _is_pamap2_zero_run(args) else "paper9acc"
    return mode


def _resolve_pamap2_remove_zero(args: argparse.Namespace | None) -> bool:
    """Remove activity_id=0 only when explicitly requested or in PAMAP2_zero."""
    if args is not None and bool(getattr(args, "pamap2_remove_zero", False)):
        return True
    return _is_pamap2_zero_run(args)


def _load_pamap2_one_subject(
    protocol_dir: Path,
    subject_idx: int,
    runtime,
    feature_mode: str = "paper9acc",
    remove_zero: bool = False,
):
    """
    Load one PAMAP2 subject.

    Default behavior is unchanged from the strict Time2State-paper protocol:
        feature_mode="paper9acc", remove_zero=False.

    Dedicated PAMAP2_zero behavior:
        feature_mode="full_sensor", remove_zero=True.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    path = protocol_dir / f"subject10{subject_idx}.dat"
    df = pd.read_csv(path, sep=" ", header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if numeric.shape[1] < 2:
        raise ValueError(f"PAMAP2 file has too few columns: {path}, shape={numeric.shape}")

    labels = np.array(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)
    mode = str(feature_mode or "paper9acc").strip().lower()

    if mode == "full_sensor":


        data = numeric[:, 2:]
        if data.shape[1] == 0:
            raise ValueError(f"PAMAP2 full_sensor mode found no feature columns: {path}")
    elif mode == "paper9acc":

        if numeric.shape[1] < 41:
            raise ValueError(f"PAMAP2 file has too few columns for paper9acc: {path}, shape={numeric.shape}")
        hand_acc = numeric[:, 4:7]
        chest_acc = numeric[:, 21:24]
        ankle_acc = numeric[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
    else:
        raise ValueError(f"Unsupported PAMAP2 feature_mode={feature_mode!r}; expected paper9acc or full_sensor")


    data = fill_nan_original(data, np)

    if remove_zero:
        valid = labels > 0
        data = data[valid]
        labels = labels[valid]
        if len(labels) < 2:
            raise ValueError(f"PAMAP2 subject10{subject_idx} has too few non-zero labels after removing activity_id=0")

    data = _original_normalize(data, runtime)
    n = min(len(data), len(labels))
    return data[:n], labels[:n]


def load_pamap2(repo_root: Path, runtime, max_count: int | None, args: argparse.Namespace | None = None) -> list[SeriesCase]:
    """
    Strict PAMAP2 loader with one controlled extension.

    Default behavior, used by all existing experiments:
        - read Time2State/data/PAMAP2/Protocol/subject101..subject108.dat
        - keep activity_id exactly as ground truth, including 0
        - use 9 acceleration dimensions: hand_acc + chest_acc + ankle_acc
        - fit on subject101 via SeriesCase.fit_data

    PAMAP2_zero behavior, activated only when the run path contains PAMAP2_zero
    or when the user explicitly passes the corresponding CLI flags:
        - use all sensor columns after timestamp and activity_id
        - remove frames whose activity_id is 0

    This path-based auto switch is intentional so launcher.py does not need to
    be modified and other experiments keep exactly their old defaults.
    """
    data_root = _original_data_root(repo_root)
    protocol_dir = _find_existing_dir(
        [
            data_root / "PAMAP2" / "Protocol",
            data_root / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
        ],
        "strict PAMAP2 Protocol directory",
    )

    feature_mode = _resolve_pamap2_feature_mode(args)
    remove_zero = _resolve_pamap2_remove_zero(args)
    print(f"[PAMAP2] feature_mode={feature_mode} remove_zero={int(remove_zero)}", flush=True)

    train_data, _train_labels = _load_pamap2_one_subject(
        protocol_dir, 1, runtime, feature_mode=feature_mode, remove_zero=remove_zero
    )
    cases = []
    for i in range(1, 9):
        path = protocol_dir / f"subject10{i}.dat"
        if not path.exists():
            continue
        data, labels = _load_pamap2_one_subject(
            protocol_dir, i, runtime, feature_mode=feature_mode, remove_zero=remove_zero
        )
        cases.append(SeriesCase("PAMAP2", f"10{i}", data, labels, fit_data=train_data))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No strict PAMAP2 subject10*.dat cases found under {protocol_dir}")
    return cases


def _load_usc_had_original_case(subject: int, target: int, repo_root: Path, runtime):
    """
    Strictly match exp_on_USC_HAD(): load_USC_HAD(subject, target, data_path),
    then normalize(data). The original load_USC_HAD implementation is imported
    from TSpy.dataset when available.
    """




    import sys
    from pathlib import Path

    tspy_dev_root = Path(r"D:\code\teacherT2S\TSpy-dev")
    if tspy_dev_root.exists():
        tspy_dev_str = str(tspy_dev_root)

        if tspy_dev_str in sys.path:
            sys.path.remove(tspy_dev_str)
        sys.path.insert(0, tspy_dev_str)





        for _mod in list(sys.modules):
            if _mod == "TSpy" or _mod.startswith("TSpy."):
                sys.modules.pop(_mod, None)

        try:
            import TSpy.dataset as _tspy_dataset
            from TSpy.dataset import load_USC_HAD as _load_USC_HAD
            from TSpy.utils import normalize as _tspy_normalize

            print("[CHECK] USC-HAD loader TSpy.dataset =", _tspy_dataset.__file__)
            print("[CHECK] load_USC_HAD OK")

            runtime["tspy_load_USC_HAD"] = _load_USC_HAD
            runtime["tspy_normalize"] = _tspy_normalize

        except Exception as e:
            print("[WARN] Failed to import load_USC_HAD from TSpy-dev:", repr(e))

    load_USC_HAD = runtime.get("tspy_load_USC_HAD")
    if load_USC_HAD is None:
        raise RuntimeError(
            "TSpy.dataset.load_USC_HAD is unavailable, so strict USC-HAD loading "
            "cannot reproduce exp_of_Time2State.py. Install/enable TSpy or skip USC-HAD."
        )

    data_path = str(_original_data_root(repo_root)) + os.sep
    data, labels = load_USC_HAD(subject, target, data_path)
    data = _original_normalize(data, runtime)
    return data, labels


def load_uschad(repo_root: Path, runtime, max_count: int | None, args: argparse.Namespace | None = None) -> list[SeriesCase]:
    """
    Strictly match exp_on_USC_HAD():

        train, _ = load_USC_HAD(1, 1, data_path)
        train = normalize(train)
        t2s.fit(train, win_size, step)
        for subject in 1..14:
            for target in 1..5:
                data, groundtruth = load_USC_HAD(subject, target, data_path)
                data = normalize(data)
                t2s.predict(data, win_size, step)

    The fit-on-subject1-target1 protocol is represented through SeriesCase.fit_data.
    """
    train_data, _ = _load_usc_had_original_case(1, 1, repo_root, runtime)
    cases = []
    for subject in range(1, 15):
        for target in range(1, 6):
            data, labels = _load_usc_had_original_case(subject, target, repo_root, runtime)
            n = min(len(data), len(labels))
            cases.append(SeriesCase("USC-HAD", f"s{subject}_t{target}", data[:n], labels[:n], fit_data=train_data))
            if max_count is not None and len(cases) >= max_count:
                return cases
    return cases


def _make_labels_from_change_points(n: int, cps, np):
    cps = [int(x) for x in cps if 0 < int(x) < n]
    cps = sorted(set(cps))
    y = np.zeros(n, dtype=int)
    start = 0
    label = 0
    for cp in cps:
        y[start:cp] = label
        start = cp
        label += 1
    y[start:n] = label
    return y


def load_tssb(repo_root: Path, runtime, max_count: int | None, args: argparse.Namespace | None = None) -> list[SeriesCase]:
    """
    Strictly match exp_on_UCR_SEG():

        dataset_path = data/UCR-SEG/UCR_datasets_seg/
        fname[:-4].split('_') gives change points
        pd.read_csv(file)                           # default header=0
        data = normalize(data)
        groundtruth = seg_to_label(seg_info)[:-1]

    This function intentionally does NOT use the newer TSSB package loader,
    last-column label inference, downsampling, median filling, or re-labeling.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    dataset_path = _original_data_root(repo_root) / "UCR-SEG" / "UCR_datasets_seg"
    files = sorted([p for p in dataset_path.iterdir() if p.is_file()])
    cases = []
    for path in files:
        if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        info_list = path.name[:-4].split("_")
        if len(info_list) < 3:
            continue
        seg_info = {}
        for i, seg in enumerate(info_list[2:]):
            seg_info[int(seg)] = i
        seg_info[_count_file_lines(path)] = len(info_list[2:])
        df = pd.read_csv(path)
        data = df.to_numpy()
        data = _original_normalize(data, runtime)
        labels = seg_to_label(seg_info, np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("UCR-SEG", path.name[:-4], data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No strict UCR-SEG files found under {dataset_path}")
    return cases


def load_cases(args: argparse.Namespace, runtime) -> tuple[list[SeriesCase], list[dict[str, object]]]:
    """
    Strict original Time2State-paper preprocessing entry point.

    This replaces the previous delegation to run_multit2s_benchmark.load_cases(),
    because that shared benchmark loader uses a normalized public-dataset format.
    Here we intentionally reproduce exp_of_Time2State.py dataset preprocessing.
    """
    loaders = {
        "synthetic": load_synthetic,
        "mocap": load_mocap,
        "actrectut": load_actrectut,
        "pamap2": load_pamap2,
        "usc-had": load_uschad,
        "tssb": load_tssb,
    }

    cases: list[SeriesCase] = []
    dataset_status: list[dict[str, object]] = []
    for raw_name in args.datasets:
        key = normalize_dataset_key(raw_name)
        if key not in loaders:
            dataset_status.append({"dataset": raw_name, "key": key, "ok": False, "error": f"unsupported dataset key: {key}"})
            raise ValueError(f"Unsupported dataset for strict original loader: {raw_name!r} -> {key!r}")

        max_count = _max_count_for_dataset(args, key)
        try:
            if key in {"pamap2", "usc-had", "tssb"}:
                ds_cases = loaders[key](args.repo_root, runtime, max_count, args)
            else:
                ds_cases = loaders[key](args.repo_root, runtime, max_count)
            cases.extend(ds_cases)
            dataset_status.append(
                {
                    "dataset": raw_name,
                    "key": key,
                    "ok": True,
                    "cases": len(ds_cases),
                    "loader": "strict_exp_of_Time2State_preprocessing",
                    "note": "No public-format filtering/downsampling; preprocessing follows exp_of_Time2State.py.",
                }
            )
        except Exception as exc:
            dataset_status.append({"dataset": raw_name, "key": key, "ok": False, "error": repr(exc)})
            raise

    return cases, dataset_status


def resolve_cuda(args, runtime) -> bool:
    torch = runtime["torch"]
    if args.device == "cpu":
        return False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    return bool(torch.cuda.is_available())


def run_one_t2s_branch(data, branch: BranchConfig, args, runtime, fit_data=None):
    np = runtime["np"]
    Time2State = runtime["Time2State"]
    CausalConv_LSE_Adaper = runtime["CausalConv_LSE_Adaper"]
    DPGMM = runtime["DPGMM"]
    params = copy.deepcopy(runtime["params_LSE"])
    params["in_channels"] = int(data.shape[1])
    params["win_size"] = int(branch.win)
    params["compared_length"] = int(branch.win)
    params["M"] = int(branch.m)
    params["N"] = int(branch.n)
    params["out_channels"] = int(branch.out_channels)
    params["nb_steps"] = int(branch.nb_steps)
    params["win_type"] = str(branch.win_type or "hanning")
    params["cuda"] = resolve_cuda(args, runtime)
    params["gpu"] = int(args.gpu)
    if branch.kernel_size is not None:
        params["kernel_size"] = int(branch.kernel_size)

    train_data = data if fit_data is None else fit_data
    n_windows = count_t2s_windows(int(train_data.shape[0]), int(branch.win), int(branch.step))
    if n_windows < 2:
        raise ValueError(f"Too few Time2State windows: rows={train_data.shape[0]}, win={branch.win}, step={branch.step}, windows={n_windows}")

    import inspect

    from Time2State import encoders
    sig = inspect.signature(encoders.CausalConv_LSE.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
    }
    params = {k: v for k, v in params.items() if k in allowed}


    model = Time2State(branch.win, branch.step, CausalConv_LSE_Adaper(params), DPGMM(None))
    model.fit(train_data, branch.win, branch.step)
    if fit_data is not None:
        model.predict(data, branch.win, branch.step)
    seq = np.asarray(model.state_seq, dtype=int)
    seq = align_sequence(seq, data.shape[0], np)
    seq, _ = reorder_label(seq, np)

    branch_min_len = resolve_branch_min_len(branch)
    if branch_min_len > 0:
        seq = merge_short_segments(seq, branch_min_len, np)
        seq, _ = reorder_label(seq, np)
    return seq

def one_hot(seq, n_states, np):
    out = np.zeros((len(seq), n_states), dtype=np.float32)
    out[np.arange(len(seq)), seq.astype(int)] = 1.0
    return out


def build_indicator(branch_sequences: list[tuple[str, int, int, object]], runtime):
    np = runtime["np"]
    pd = runtime["pd"]
    mats = []
    state_rows = []
    for idx, (name, win, step, seq) in enumerate(branch_sequences, start=1):
        seq = np.asarray(seq, dtype=int)
        for state_id in sorted(np.unique(seq).astype(int).tolist()):
            mats.append((seq == state_id).astype(np.float32))
            state_rows.append(
                {
                    "global_state_name": f"{name}_state{state_id}",
                    "run_idx": idx,
                    "win": win,
                    "step": step,
                    "local_state": state_id,
                }
            )
    indicator = np.stack(mats, axis=1).astype(np.float32)
    return pd.DataFrame(indicator, columns=[r["global_state_name"] for r in state_rows]), pd.DataFrame(state_rows)


def rolling_mean_axis1(arr, win, np):
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant")
    return (cumsum[:, win:] - cumsum[:, :-win]) / float(win)


def attach_branch_weights_to_state_info(state_info_df, branch_metrics: list[dict[str, object]], mode: str) -> object:
    """
    Add branch_weight to every local state row according to its run_idx.

    mode:
    - uniform: all branches weight 1
    - branch_reliability: use unsupervised peer_reliability_norm
    - oracle_score: use supervised branch_selection_score, debug only
    """
    out = state_info_df.copy()
    mode = str(mode or "uniform").lower()
    weights = []
    for row in branch_metrics:
        if mode == "branch_reliability":
            w = _safe_float(row.get("peer_reliability_norm", row.get("peer_reliability", 0.0)), 0.0)
        elif mode == "pid_weight":
            w = _safe_float(row.get("pid_weight_norm", row.get("pid_score_norm", 0.0)), 0.0)
        elif mode == "oracle_score":
            w = _safe_float(row.get("branch_selection_score", 0.0), 0.0)
        else:
            w = 1.0
        weights.append(max(0.0, float(w)))


    if not weights or max(weights) <= 1e-12:
        weights = [1.0 for _ in weights]

    run_idx_values = out["run_idx"].astype(int).to_numpy()
    out["branch_weight"] = [weights[i - 1] if 1 <= i <= len(weights) else 1.0 for i in run_idx_values]
    return out


def cluster_state_matrix(indicator_df, state_info_df, n_clusters, smooth_win, runtime):
    np = runtime["np"]
    AgglomerativeClustering = runtime["AgglomerativeClustering"]
    H = indicator_df.to_numpy(dtype=np.float32)
    G = H.T
    G_smooth = rolling_mean_axis1(G, smooth_win, np)
    norm = np.linalg.norm(G_smooth, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    X = G_smooth / norm
    try:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    labels = clusterer.fit_predict(X).astype(int)
    labels, _ = reorder_label(labels, np)
    out_info = state_info_df.copy()
    out_info["cluster_id"] = labels
    return out_info


def meta_sequence_from_clusters(indicator_df, state_info_df, meta_min_len, runtime):
    np = runtime["np"]
    pd = runtime["pd"]
    H = indicator_df.to_numpy(dtype=np.float32)
    cluster_labels = state_info_df["cluster_id"].to_numpy(dtype=int)
    if "branch_weight" in state_info_df.columns:
        local_weights = state_info_df["branch_weight"].to_numpy(dtype=np.float32)
    else:
        local_weights = np.ones(len(cluster_labels), dtype=np.float32)
    n_clusters = int(cluster_labels.max()) + 1
    votes = np.zeros((H.shape[0], n_clusters), dtype=np.float32)
    for local_idx, cluster_id in enumerate(cluster_labels):
        votes[:, cluster_id] += float(local_weights[local_idx]) * H[:, local_idx]
    denom = votes.sum(axis=1, keepdims=True)
    denom[denom < 1e-8] = 1.0
    ratios = votes / denom
    seq = np.argmax(ratios, axis=1).astype(int)
    seq = merge_short_segments(seq, meta_min_len, np)
    seq, mapping = reorder_label(seq, np)
    new_ratios = np.zeros((len(seq), len(mapping)), dtype=np.float32)
    for old_state, new_state in mapping.items():
        if old_state < ratios.shape[1]:
            new_ratios[:, new_state] += ratios[:, old_state]
    denom2 = new_ratios.sum(axis=1, keepdims=True)
    denom2[denom2 < 1e-8] = 1.0
    new_ratios = new_ratios / denom2
    ratio_df = pd.DataFrame(new_ratios, columns=[f"meta_ratio_{i}" for i in range(new_ratios.shape[1])])
    return seq, ratio_df


def count_segments(seq) -> int:
    values = list(seq)
    if not values:
        return 0
    return 1 + sum(1 for left, right in zip(values, values[1:]) if left != right)


def normalized_entropy(seq, np) -> float:
    values, counts = np.unique(np.asarray(seq, dtype=int), return_counts=True)
    if len(values) <= 1:
        return 0.0
    probs = counts.astype(float) / counts.sum()
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent / np.log(len(values)))


def evaluate_k(indicator_df, state_info_df, k, args, runtime, meta_min_len: int):
    np = runtime["np"]
    clustered_info = cluster_state_matrix(indicator_df, state_info_df, k, args.state_cluster_smooth, runtime)
    seq, ratio_df = meta_sequence_from_clusters(indicator_df, clustered_info, meta_min_len, runtime)
    values, counts = np.unique(seq, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    tau = max(0.05, 1.0 / float(k))
    dominant_count = int(np.sum(probs >= tau))
    return {
        "K": int(k),
        "dominant_ratio": float(dominant_count) / float(k),
        "balance_entropy": normalized_entropy(seq, np),
        "mean_segment_length": float(len(seq) / max(1, count_segments(seq))),
        "segment_count": count_segments(seq),
        "seq": seq,
        "ratio_df": ratio_df,
        "state_info": clustered_info,
    }


def choose_meta_k(indicator_df, state_info_df, run_state_counts, args, runtime, meta_min_len: int):
    np = runtime["np"]
    k_low = max(2, min(run_state_counts))
    k_high = max(run_state_counts)
    k_high = min(k_high, max(2, len(state_info_df) - 1))
    candidates = list(range(k_low, k_high + 1))
    if not candidates:
        candidates = [2]
    raw = [evaluate_k(indicator_df, state_info_df, k, args, runtime, meta_min_len) for k in candidates]
    seg_vals = np.asarray([r["mean_segment_length"] for r in raw], dtype=float)
    seg_min = float(seg_vals.min())
    seg_max = float(seg_vals.max())
    scored = []
    for r in raw:
        coherence = 1.0 if seg_max - seg_min < 1e-12 else float((r["mean_segment_length"] - seg_min) / (seg_max - seg_min))
        score = 0.60 * r["dominant_ratio"] + 0.25 * r["balance_entropy"] + 0.15 * coherence
        row = dict(r)
        row["coherence"] = coherence
        row["selection_score"] = score
        scored.append(row)
    scored.sort(key=lambda r: (-r["selection_score"], r["K"], -r["mean_segment_length"]))
    return scored[0], scored



def _safe_float(x, default=0.0) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def get_segments(seq) -> list[tuple[int, int, int]]:
    values = list(seq)
    if not values:
        return []
    segments = []
    start = 0
    for idx in range(1, len(values)):
        if values[idx] != values[start]:
            segments.append((start, idx - 1, int(values[start])))
            start = idx
    segments.append((start, len(values) - 1, int(values[start])))
    return segments


def short_segment_ratio(seq, short_len: int, np) -> float:
    seq = np.asarray(seq, dtype=int)
    if len(seq) == 0:
        return 1.0
    short_points = 0
    for left, right, _state in get_segments(seq):
        seg_len = right - left + 1
        if seg_len < short_len:
            short_points += seg_len
    return float(short_points) / float(len(seq))


def branch_health_score(seq, np, median_segments: float | None = None) -> dict[str, float]:
    """
    Unsupervised self-health score for one branch prediction.

    A healthy expert should:
    - not collapse into one state;
    - not be almost entirely dominated by one state;
    - not be extremely fragmented;
    - have a reasonable segment count relative to the candidate pool.
    """
    seq = np.asarray(seq, dtype=int)
    n = int(len(seq))
    if n <= 0:
        return {
            "health": 0.0,
            "state_entropy": 0.0,
            "dominant_ratio": 1.0,
            "short_segment_ratio": 1.0,
            "segment_reasonable_score": 0.0,
            "n_states": 0.0,
            "segments": 0.0,
        }

    values, counts = np.unique(seq, return_counts=True)
    n_states = int(len(values))
    seg_count = int(count_segments(seq))
    dom = float(counts.max()) / float(n)

    if n_states < 2 or seg_count < 2:
        return {
            "health": 0.0,
            "state_entropy": 0.0,
            "dominant_ratio": dom,
            "short_segment_ratio": 1.0,
            "segment_reasonable_score": 0.0,
            "n_states": float(n_states),
            "segments": float(seg_count),
        }

    entropy = normalized_entropy(seq, np)


    if dom <= 0.75:
        dom_score = 1.0
    elif dom >= 0.98:
        dom_score = 0.0
    else:
        dom_score = (0.98 - dom) / (0.98 - 0.75)


    short_len = max(2, int(round(0.005 * n)))
    short_ratio = short_segment_ratio(seq, short_len, np)
    frag_score = max(0.0, 1.0 - min(1.0, short_ratio / 0.35))

    if median_segments is None or median_segments <= 0:
        seg_score = 1.0
    else:


        seg_score = math.exp(-abs(math.log((seg_count + 1.0) / (float(median_segments) + 1.0))))

    health = float(entropy * dom_score * frag_score * seg_score)
    health = max(0.0, min(1.0, health))

    return {
        "health": health,
        "state_entropy": float(entropy),
        "dominant_ratio": float(dom),
        "short_segment_ratio": float(short_ratio),
        "segment_reasonable_score": float(seg_score),
        "n_states": float(n_states),
        "segments": float(seg_count),
    }


def pairwise_prediction_similarity(seq_a, seq_b) -> float:
    """
    Unsupervised similarity between two expert predictions.
    It compares prediction to prediction, not prediction to ground truth.
    """
    ari = adjusted_rand_index(seq_a, seq_b)
    nmi = normalized_mutual_information(seq_a, seq_b)
    return float(0.5 * max(0.0, ari) + 0.5 * max(0.0, nmi))


def compute_peer_reliability(branch_sequences: list[tuple[str, int, int, object]], args, runtime) -> list[dict[str, float]]:
    """
    Leave-one-branch-out peer-consensus reliability.

    For branch i:
        Health_i = self-quality of its predicted segmentation.
        Consensus_i = weighted average similarity between branch i and every other healthy branch.
        Reliability_i = alpha * Health_i + beta * Consensus_i, then multiplied by Health_i
                        to suppress collapsed/fragmented branches.

    This uses no ground-truth labels.
    """
    np = runtime["np"]
    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]
    n = len(seqs)
    seg_counts = [count_segments(seq) for seq in seqs]
    median_segments = float(np.median(np.asarray(seg_counts, dtype=float))) if seg_counts else 0.0

    health_rows = [branch_health_score(seq, np, median_segments=median_segments) for seq in seqs]
    health = np.asarray([row["health"] for row in health_rows], dtype=float)

    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = pairwise_prediction_similarity(seqs[i], seqs[j])
            sim[i, j] = s
            sim[j, i] = s

    consensus = np.zeros(n, dtype=float)
    for i in range(n):
        weights = health.copy()
        weights[i] = 0.0
        denom = float(weights.sum())
        if denom <= 1e-12:

            others = [j for j in range(n) if j != i]
            consensus[i] = float(np.mean([sim[i, j] for j in others])) if others else 0.0
        else:
            consensus[i] = float((sim[i] * weights).sum() / denom)

    alpha = float(getattr(args, "peer_health_weight", 0.45))
    beta = float(getattr(args, "peer_consensus_weight", 0.55))
    denom_ab = max(1e-12, alpha + beta)
    alpha /= denom_ab
    beta /= denom_ab

    reliability = health * (alpha * health + beta * consensus)
    max_rel = float(reliability.max()) if len(reliability) else 0.0
    if max_rel > 1e-12:
        reliability_norm = reliability / max_rel
    else:
        reliability_norm = reliability

    rows = []
    for i in range(n):
        row = dict(health_rows[i])
        row["peer_consensus"] = float(consensus[i])
        row["peer_reliability"] = float(reliability[i])
        row["peer_reliability_norm"] = float(reliability_norm[i])
        rows.append(row)
    return rows



def compute_pid_reliability(branch_sequences: list[tuple[str, int, int, object]], branch_metrics: list[dict[str, object]], args, runtime) -> list[dict[str, float]]:
    """
    PID-inspired unsupervised expert reliability.

    P term: current branch quality = health + peer consensus + scale prior.
    I term: neighborhood stability = agreement with nearby-window branches.
    D term: instability penalty = short fragments + segment-count deviation + scale sensitivity.

    This function uses no ground-truth labels. It compares expert predictions to expert predictions.
    """
    np = runtime["np"]
    n = len(branch_sequences)
    if n == 0:
        return []

    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]
    wins = np.asarray([float(item[1]) for item in branch_sequences], dtype=float)
    steps = np.asarray([float(item[2]) for item in branch_sequences], dtype=float)


    if not branch_metrics or "health" not in branch_metrics[0] or "peer_consensus" not in branch_metrics[0]:
        peer_rows = compute_peer_reliability(branch_sequences, args, runtime)
        for row, peer in zip(branch_metrics, peer_rows):
            row.update(peer)

    health = np.asarray([_safe_float(row.get("health", 0.0)) for row in branch_metrics], dtype=float)
    peer_consensus = np.asarray([_safe_float(row.get("peer_consensus", 0.0)) for row in branch_metrics], dtype=float)



    scale_prior = np.zeros(n, dtype=float)


    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = pairwise_prediction_similarity(seqs[i], seqs[j])
            sim[i, j] = s
            sim[j, i] = s



    neighborhood_stability = np.zeros(n, dtype=float)
    scale_sensitivity = np.zeros(n, dtype=float)
    for i in range(n):
        log_dist = np.abs(np.log((wins + 1e-6) / (wins[i] + 1e-6)))

        neighbor_mask = (log_dist > 1e-12) & (log_dist <= 0.45)
        neighbor_idx = np.where(neighbor_mask)[0].tolist()
        if len(neighbor_idx) == 0 and n > 1:
            neighbor_idx = np.argsort(log_dist)[1:min(n, 3)].tolist()

        if len(neighbor_idx) == 0:
            neighborhood_stability[i] = 0.0
            scale_sensitivity[i] = 1.0
        else:
            vals = np.asarray([sim[i, j] for j in neighbor_idx], dtype=float)
            neighborhood_stability[i] = float(vals.mean())

            scale_sensitivity[i] = float(1.0 - vals.mean())


    seg_counts = np.asarray([count_segments(seq) for seq in seqs], dtype=float)
    median_segments = float(np.median(seg_counts)) if len(seg_counts) else 1.0
    segment_count_deviation = np.asarray(
        [abs(math.log((c + 1.0) / (median_segments + 1.0))) for c in seg_counts],
        dtype=float,
    )
    if segment_count_deviation.max() > 1e-12:
        segment_count_deviation = segment_count_deviation / segment_count_deviation.max()

    short_ratios = np.asarray(
        [branch_health_score(seq, np, median_segments=median_segments).get("short_segment_ratio", 1.0) for seq in seqs],
        dtype=float,
    )
    instability_penalty = (
        0.50 * np.clip(short_ratios, 0.0, 1.0)
        + 0.30 * np.clip(segment_count_deviation, 0.0, 1.0)
        + 0.20 * np.clip(scale_sensitivity, 0.0, 1.0)
    )






    p_term = 0.5625 * peer_consensus + 0.4375 * health
    i_term = neighborhood_stability
    d_term = instability_penalty

    kp = float(getattr(args, "pid_kp", 0.45))
    ki = float(getattr(args, "pid_ki", 0.35))
    kd = float(getattr(args, "pid_kd", 0.20))
    norm = max(1e-12, abs(kp) + abs(ki) + abs(kd))
    kp, ki, kd = kp / norm, ki / norm, kd / norm

    raw = kp * p_term + ki * i_term - kd * d_term


    for i, seq in enumerate(seqs):
        vals, counts = np.unique(seq, return_counts=True)
        dominant = float(counts.max()) / max(1.0, float(len(seq)))
        if len(vals) < 2 or count_segments(seq) < 2 or dominant > 0.98:
            raw[i] = -1e9

    finite = np.isfinite(raw) & (raw > -1e8)
    if finite.any():
        raw_min = float(raw[finite].min())
        raw_shift = raw.copy()
        raw_shift[finite] = raw_shift[finite] - raw_min
        raw_shift[~finite] = 0.0
        raw_norm = raw_shift / max(1e-12, float(raw_shift[finite].max()))
    else:
        raw_norm = np.zeros(n, dtype=float)


    tau = max(1e-6, float(getattr(args, "pid_softmax_tau", 0.15)))
    z = raw_norm / tau
    z = z - float(np.max(z)) if len(z) else z
    expz = np.exp(z)
    weights = expz / max(1e-12, float(expz.sum()))
    if weights.max() > 1e-12:
        weights_norm = weights / weights.max()
    else:
        weights_norm = weights

    rows = []
    for i in range(n):
        rows.append({
            "pid_p": float(p_term[i]),
            "pid_i": float(i_term[i]),
            "pid_d": float(d_term[i]),
            "pid_scale_prior": float(scale_prior[i]),
            "pid_neighborhood_stability": float(neighborhood_stability[i]),
            "pid_scale_sensitivity": float(scale_sensitivity[i]),
            "pid_segment_deviation": float(segment_count_deviation[i]),
            "pid_raw_score": float(raw[i]),
            "pid_score_norm": float(raw_norm[i]),
            "pid_weight": float(weights[i]),
            "pid_weight_norm": float(weights_norm[i]),
        })
    return rows




def compute_m2qd_reliability(branch_sequences: list[tuple[str, int, int, object]], args, runtime) -> list[dict[str, float]]:
    """
    Pair-wise M2QD (maximum quality-maximum diversity) selection scores.

    Adapted from cluster ensemble selection to temporal expert selection:
    - each T2S branch prediction is treated as one base clustering;
    - Quality is average NMI agreement with other candidate branch predictions;
    - Diversity is average 1-NMI against already selected branches;
    - a greedy order is computed without using ground-truth labels.

    The returned m2qd_rank_score is monotonic with the greedy selection order and
    is used by select_top_k_branch_indices. The raw M2QD score at the time of
    selection is also exported for diagnostics.
    """
    np = runtime["np"]
    n = len(branch_sequences)
    if n == 0:
        return []

    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]
    names = [str(item[0]) for item in branch_sequences]


    sim = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = normalized_mutual_information(seqs[i], seqs[j])
            if not math.isfinite(s):
                s = 0.0
            s = max(0.0, min(1.0, float(s)))
            sim[i, j] = s
            sim[j, i] = s

    if n == 1:
        return [{
            "m2qd_quality_all": 0.0,
            "m2qd_quality_when_selected": 0.0,
            "m2qd_diversity_when_selected": 0.0,
            "m2qd_score": 0.0,
            "m2qd_rank": 1.0,
            "m2qd_rank_score": 1.0,
        }]

    quality_all = np.asarray([(sim[i].sum() - 1.0) / float(n - 1) for i in range(n)], dtype=float)
    alpha = float(getattr(args, "m2qd_alpha", 0.5))
    if not math.isfinite(alpha):
        alpha = 0.5
    alpha = max(0.0, min(1.0, alpha))

    remaining = set(range(n))
    selected: list[int] = []
    rows = [None for _ in range(n)]

    while remaining:
        best = None
        for i in sorted(remaining):
            if not selected:
                quality = float(quality_all[i])
                diversity = 0.0
                score = quality
            else:
                rem_without_i = [j for j in remaining if j != i]
                if rem_without_i:
                    quality = float(np.mean([sim[i, j] for j in rem_without_i]))
                else:
                    quality = float(quality_all[i])
                diversity = float(np.mean([1.0 - sim[i, r] for r in selected]))
                score = float(alpha * quality + (1.0 - alpha) * diversity)


            key = (score, quality, diversity, -i)
            if best is None or key > best[0]:
                best = (key, i, quality, diversity, score)

        _key, i_sel, q_sel, d_sel, s_sel = best
        selected.append(i_sel)
        remaining.remove(i_sel)
        rank = len(selected)
        rows[i_sel] = {
            "m2qd_quality_all": float(quality_all[i_sel]),
            "m2qd_quality_when_selected": float(q_sel),
            "m2qd_diversity_when_selected": float(d_sel),
            "m2qd_score": float(s_sel),
            "m2qd_rank": float(rank),

            "m2qd_rank_score": float(n - rank + 1),
        }

    return rows



def choose_cspa_k_from_selected(run_state_counts: list[int], n_rows: int, args) -> int:
    """Choose CSPA final K without ground-truth labels.

    The default median rule follows the user's requested setting:
    K = round(median(number_of_states among selected branches)).
    """
    values = [int(x) for x in run_state_counts if int(x) > 0]
    if not values:
        return 2
    mode = str(getattr(args, "cspa_k_mode", "median") or "median").lower()
    import statistics
    if mode == "mean":
        raw = sum(values) / float(len(values))
    elif mode == "max":
        raw = max(values)
    elif mode == "min":
        raw = min(values)
    else:
        raw = statistics.median(values)

    k = int(math.floor(float(raw) + 0.5))
    k = max(2, k)
    k = min(k, max(2, int(n_rows)))
    return k


def cspa_consensus_sequence(branch_sequences: list[tuple[str, int, int, object]], k: int, args, runtime, meta_min_len: int):
    """CSPA-style consensus over selected temporal expert label sequences.

    Each selected T2S branch is treated as one base clustering over time points.
    We build the usual cluster-membership indicator matrix H where each row is a
    time point and each column is a local cluster from a selected branch. The
    CSPA co-association similarity is proportional to H @ H.T. For scalability,
    short sequences use agglomerative clustering on H, while long sequences use
    MiniBatchKMeans on H as a practical CSPA-style solver.

    No ground-truth labels are used. The final number of clusters k is supplied
    by choose_cspa_k_from_selected, defaulting to the median state count of the
    selected branches.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    indicator_df, state_info_df = build_indicator(branch_sequences, runtime)
    X = indicator_df.to_numpy(dtype=np.float32)
    n_rows = int(X.shape[0])
    k = int(max(2, min(int(k), n_rows)))




    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    row_norm[row_norm < 1e-8] = 1.0
    Xn = X / row_norm

    engine = str(getattr(args, "cspa_engine", "auto") or "auto").lower()
    if engine == "auto":
        engine = "agglomerative" if n_rows <= int(getattr(args, "cspa_agglomerative_max_rows", 3000)) else "minibatch_kmeans"

    if engine == "agglomerative":
        AgglomerativeClustering = runtime["AgglomerativeClustering"]
        try:
            clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        seq = clusterer.fit_predict(Xn).astype(int)
    else:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception:
            from sklearn.cluster import KMeans as MiniBatchKMeans
        seed = int(getattr(args, "seed", 0) or 0)
        batch_size = int(getattr(args, "cspa_batch_size", 4096) or 4096)

        clusterer = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=max(256, batch_size),
            n_init=10,
            max_iter=200,
        )
        seq = clusterer.fit_predict(Xn).astype(int)

    seq, mapping = reorder_label(seq, np)
    if bool(getattr(args, "cspa_apply_meta_min_len", False)) and int(meta_min_len) > 0:
        seq = merge_short_segments(seq, int(meta_min_len), np)
        seq, mapping = reorder_label(seq, np)



    n_states = int(seq.max()) + 1 if len(seq) else 0
    ratios = np.zeros((len(seq), max(1, n_states)), dtype=np.float32)
    if len(seq):
        ratios[np.arange(len(seq)), seq.astype(int)] = 1.0
    ratio_df = pd.DataFrame(ratios, columns=[f"cspa_ratio_{i}" for i in range(ratios.shape[1])])

    state_info_df = state_info_df.copy()
    state_info_df["cspa_selected"] = 1
    state_info_df["cspa_engine"] = engine
    return seq, ratio_df, state_info_df, engine


def choose_ec_tdwm_k_from_selected(run_state_counts: list[int], n_rows: int, args) -> int:
    """Choose EC-TDWM final K without ground-truth labels.

    The original EC-TDWM paper assumes the final cluster number is known in the
    static clustering benchmark. For temporal state discovery we use the same
    label-free protocol as the adapted CSPA baseline: median/mean/max/min of the
    state counts produced by candidate temporal experts.
    """
    values = [int(x) for x in run_state_counts if int(x) > 0]
    if not values:
        return 2
    mode = str(getattr(args, "ec_tdwm_k_mode", "median") or "median").lower()
    import statistics
    if mode == "mean":
        raw = sum(values) / float(len(values))
    elif mode == "max":
        raw = max(values)
    elif mode == "min":
        raw = min(values)
    else:
        raw = statistics.median(values)
    k = int(math.floor(float(raw) + 0.5))
    k = max(2, k)
    k = min(k, max(2, int(n_rows)))
    return k


def _fit_label_update(X, k: int, args, runtime, prefix: str = "ec_tdwm"):
    """Cluster rows of X to update consensus labels."""
    np = runtime["np"]
    n_rows = int(X.shape[0])
    k = int(max(2, min(k, n_rows)))
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    row_norm[row_norm < 1e-8] = 1.0
    Xn = X / row_norm
    engine = str(getattr(args, f"{prefix}_engine", "auto") or "auto").lower()
    if engine == "auto":
        max_rows = int(getattr(args, f"{prefix}_agglomerative_max_rows", 3000) or 3000)
        engine = "agglomerative" if n_rows <= max_rows else "minibatch_kmeans"
    if engine == "agglomerative":
        AgglomerativeClustering = runtime["AgglomerativeClustering"]
        try:
            clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        seq = clusterer.fit_predict(Xn).astype(int)
    else:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception:
            from sklearn.cluster import KMeans as MiniBatchKMeans
        seed = int(getattr(args, "seed", 0) or 0)
        batch_size = int(getattr(args, f"{prefix}_batch_size", 4096) or 4096)
        clusterer = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=max(256, batch_size),
            n_init=10,
            max_iter=200,
        )
        seq = clusterer.fit_predict(Xn).astype(int)
    seq, _ = reorder_label(seq, np)
    return seq, engine


def ec_tdwm_consensus_sequence(branch_sequences: list[tuple[str, int, int, object]], k: int, args, runtime, meta_min_len: int):
    """Adapted EC-TDWM three-level dynamic weighting consensus.

    The uploaded EC-TDWM paper proposes a parameter-free ensemble clustering
    framework that jointly weights base clustering results, clusters and samples,
    and directly learns a discrete consensus label matrix. This implementation
    adapts that ensemble-stage idea to temporal state discovery:

    - each selected/all T2S branch prediction is treated as one base clustering;
    - branch weights measure agreement between a branch and the current consensus;
    - cluster weights measure purity of each local state under the current consensus;
    - sample weights measure how consistently a time point is supported by the
      base clusterings;
    - a weighted binary partition matrix is repeatedly reclustered until stable.

    No ground-truth labels are used. The final K is supplied by the label-free
    branch-state-count rule in choose_ec_tdwm_k_from_selected.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    if not branch_sequences:
        raise ValueError("EC-TDWM requires at least one branch sequence")

    indicator_df, state_info_df = build_indicator(branch_sequences, runtime)
    H = indicator_df.to_numpy(dtype=np.float32)
    n_rows = int(H.shape[0])
    k = int(max(2, min(int(k), n_rows)))

    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]
    state_offsets = []
    cursor = 0
    for seq in seqs:
        states = sorted(np.unique(seq).astype(int).tolist())
        state_offsets.append((cursor, {s: cursor + j for j, s in enumerate(states)}, len(states)))
        cursor += len(states)


    init_k = k
    try:
        y, _ratio0, _info0, engine = cspa_consensus_sequence(branch_sequences, init_k, args, runtime, meta_min_len=0)
    except Exception:
        y, engine = _fit_label_update(H, init_k, args, runtime, prefix="ec_tdwm")
    y = align_sequence(y, n_rows, np)
    y, _ = reorder_label(y, np)

    max_iter = max(1, int(getattr(args, "ec_tdwm_max_iter", 20) or 20))
    tol = max(0.0, float(getattr(args, "ec_tdwm_tol", 1e-5) or 1e-5))
    eps = 1e-8
    branch_alpha = np.ones(len(seqs), dtype=np.float64) / max(1, len(seqs))
    cluster_weights = np.ones(H.shape[1], dtype=np.float64)
    sample_weights = np.ones(n_rows, dtype=np.float64)
    n_iter = 0
    final_change = 1.0

    for it in range(max_iter):
        n_iter = it + 1
        old_y = y.copy()
        Y = one_hot(y, max(k, int(y.max()) + 1), np).astype(np.float32)

        branch_scores = []
        cluster_weights[:] = 0.0
        sample_support = np.zeros(n_rows, dtype=np.float64)

        for bi, seq in enumerate(seqs):
            seq = np.asarray(seq, dtype=int)
            _offset, state_to_col, _num_states = state_offsets[bi]
            local_agreement = np.zeros(n_rows, dtype=np.float64)
            local_cluster_scores = []
            for state, col in state_to_col.items():
                idx = np.where(seq == state)[0]
                if len(idx) == 0:
                    score = 0.0
                else:
                    counts = np.bincount(y[idx].astype(int), minlength=max(k, int(y.max()) + 1)).astype(float)
                    score = float(counts.max() / max(eps, counts.sum()))
                    local_agreement[idx] = score
                cluster_weights[col] = score
                local_cluster_scores.append(score)

            try:
                nmi_to_y = normalized_mutual_information(y, seq)
            except Exception:
                nmi_to_y = 0.0
            branch_score = 0.5 * float(nmi_to_y) + 0.5 * float(np.mean(local_cluster_scores) if local_cluster_scores else 0.0)
            branch_scores.append(max(eps, branch_score))
            sample_support += local_agreement

        branch_alpha = np.asarray(branch_scores, dtype=np.float64)
        if branch_alpha.sum() <= eps:
            branch_alpha[:] = 1.0 / float(len(branch_alpha))
        else:
            branch_alpha = branch_alpha / branch_alpha.sum()

        sample_weights = sample_support / max(eps, float(len(seqs)))

        sample_weights = 0.25 + 0.75 * sample_weights
        sample_weights = sample_weights / max(eps, float(sample_weights.mean()))



        col_weights = np.zeros(H.shape[1], dtype=np.float64)
        for bi, (_offset, _state_to_col, _num_states) in enumerate(state_offsets):
            for _state, col in _state_to_col.items():
                col_weights[col] = branch_alpha[bi] * max(eps, cluster_weights[col])
        X = H.astype(np.float32, copy=True)
        X *= np.sqrt(col_weights + eps).astype(np.float32)[None, :]
        X *= np.sqrt(sample_weights + eps).astype(np.float32)[:, None]

        y, engine = _fit_label_update(X, k, args, runtime, prefix="ec_tdwm")
        y = align_sequence(y, n_rows, np)
        y, _ = reorder_label(y, np)
        final_change = float(np.mean(y != old_y)) if len(y) else 0.0
        if final_change <= tol:
            break

    if bool(getattr(args, "ec_tdwm_apply_meta_min_len", False)) and int(meta_min_len) > 0:
        y = merge_short_segments(y, int(meta_min_len), np)
        y, _ = reorder_label(y, np)

    n_states = int(y.max()) + 1 if len(y) else 0
    ratios = np.zeros((len(y), max(1, n_states)), dtype=np.float32)
    if len(y):
        ratios[np.arange(len(y)), y.astype(int)] = 1.0
    ratio_df = pd.DataFrame(ratios, columns=[f"ec_tdwm_ratio_{i}" for i in range(ratios.shape[1])])

    info = state_info_df.copy()
    info["ec_tdwm_cluster_weight"] = cluster_weights.astype(float)

    info["ec_tdwm_base_weight"] = [float(branch_alpha[int(r) - 1]) for r in info["run_idx"].astype(int).tolist()]
    info["ec_tdwm_engine"] = engine
    info["ec_tdwm_iterations"] = int(n_iter)
    info["ec_tdwm_final_change"] = float(final_change)
    return y, ratio_df, info, engine, int(n_iter), float(final_change)

def branch_selection_score(metric_row: dict[str, object], metric: str) -> float:
    """
    Score one branch for top-K branch selection.

    ARI/NMI/ARI_NMI are supervised oracle/debug scores.
    PEER is unsupervised and uses leave-one-branch-out peer consensus reliability.
    M2QD is unsupervised and uses pair-wise maximum quality-maximum diversity order.
    """
    metric = str(metric).upper()
    if metric == "PEER":
        return _safe_float(metric_row.get("peer_reliability_norm", metric_row.get("peer_reliability", 0.0)))
    if metric == "PID":
        return _safe_float(metric_row.get("pid_score_norm", metric_row.get("pid_weight_norm", 0.0)))
    if metric == "M2QD":
        return _safe_float(metric_row.get("m2qd_rank_score", 0.0))

    ari = _safe_float(metric_row.get("ARI", 0.0))
    nmi = _safe_float(metric_row.get("NMI", 0.0))
    if metric == "ARI":
        return ari
    if metric == "NMI":
        return nmi
    if metric == "ARI_NMI":
        return 0.5 * (ari + nmi)
    raise ValueError(f"Unknown branch selection metric: {metric}")


def select_top_k_branch_indices(
    branch_metrics: list[dict[str, object]],
    top_k: int,
    metric: str
) -> tuple[set[int], list[dict[str, object]]]:
    """
    Select top-K branch indices from branch_metrics.

    Clean version:
    - The primary branch_selection_score can be PEER/PID/ARI/NMI/ARI_NMI.
    - However, the deterministic tie-breaker NEVER uses ARI/NMI.
    - This completely removes the hidden label-leakage risk in tie-breaking.

    Returns:
        selected_indices: zero-based indices selected for meta aggregation.
        ranked_rows: branch metrics with branch_selection_score and selected_for_meta fields.
    """
    n = len(branch_metrics)

    if n == 0:
        return set(), []


    if top_k is None or int(top_k) <= 0 or int(top_k) >= n:
        ranked_rows = []
        for idx, row in enumerate(branch_metrics):
            new_row = dict(row)
            new_row["branch_rank"] = idx + 1
            new_row["branch_selection_score"] = branch_selection_score(new_row, metric)
            new_row["selected_for_meta"] = 1
            ranked_rows.append(new_row)
        return set(range(n)), ranked_rows

    top_k = int(top_k)
    scored = []

    for idx, row in enumerate(branch_metrics):
        score = branch_selection_score(row, metric)












        scored.append((
            idx,
            score,
            _safe_float(row.get("health", row.get("branch_health", 0.0))),
            _safe_float(row.get("peer_consensus", 0.0)),
            str(row.get("branch", "")),
        ))

    scored_sorted = sorted(
        scored,
        key=lambda x: (
            -x[1],
            -x[2],
            -x[3],
            x[4],
            x[0],
        )
    )

    selected_indices = {idx for idx, *_ in scored_sorted[:top_k]}
    rank_map = {
        idx: rank + 1
        for rank, (idx, *_rest) in enumerate(scored_sorted)
    }

    ranked_rows = []
    for idx, row in enumerate(branch_metrics):
        new_row = dict(row)
        new_row["branch_rank"] = rank_map[idx]
        new_row["branch_selection_score"] = branch_selection_score(new_row, metric)
        new_row["selected_for_meta"] = 1 if idx in selected_indices else 0
        ranked_rows.append(new_row)

    return selected_indices, ranked_rows


def run_multit2s_case(case: SeriesCase, args, runtime):
    np = runtime["np"]
    candidate_branch_sequences = []
    candidate_run_state_counts = []
    branch_metrics = []
    start = time.time()
    data = case.data
    labels = case.labels
    if args.limit_rows is not None:
        data = data[: args.limit_rows]
        labels = labels[: args.limit_rows]

    branches = branch_configs_for_case(case, args)
    meta_min_len = resolve_meta_min_len_for_case(branches, args)


    for idx, branch in enumerate(branches, start=1):
        if data.shape[0] <= branch.win + branch.step:
            print(f"  [branch skip] {case.dataset}/{case.case_id} {branch.branch_name}: too short for win={branch.win}, step={branch.step}, rows={data.shape[0]}")
            continue

        branch_name = branch.branch_name or f"run{idx}_w{branch.win}_s{branch.step}"
        branch_start = time.time()
        seq = run_one_t2s_branch(data, branch, args, runtime, fit_data=getattr(case, "fit_data", None))
        elapsed = time.time() - branch_start

        candidate_branch_sequences.append((branch_name, branch.win, branch.step, seq))
        candidate_run_state_counts.append(int(seq.max()) + 1)
        branch_metrics.append(
            {
                "branch": branch_name,
                "win": branch.win,
                "step": branch.step,
                "m": branch.m,
                "n": branch.n,
                "out_channels": branch.out_channels,
                "nb_steps": branch.nb_steps,
                "kernel_size": branch.kernel_size or "",
                "win_type": branch.win_type,
                "branch_min_len": resolve_branch_min_len(branch),
                "meta_min_len": meta_min_len,
                "states": int(seq.max()) + 1,
                "segments": count_segments(seq),
                "seconds": elapsed,
            }
        )

    if not candidate_branch_sequences:
        raise RuntimeError(f"No branches could run for {case.dataset}/{case.case_id}")


    peer_rows = compute_peer_reliability(candidate_branch_sequences, args, runtime)
    for row, peer_row in zip(branch_metrics, peer_rows):
        row.update(peer_row)

    pid_rows = compute_pid_reliability(candidate_branch_sequences, branch_metrics, args, runtime)
    for row, pid_row in zip(branch_metrics, pid_rows):
        row.update(pid_row)

    m2qd_rows = compute_m2qd_reliability(candidate_branch_sequences, args, runtime)
    for row, m2qd_row in zip(branch_metrics, m2qd_rows):
        row.update(m2qd_row)


    selected_indices, ranked_branch_metrics = select_top_k_branch_indices(
        branch_metrics,
        int(getattr(args, "select_top_k_branches", 0) or 0),
        str(getattr(args, "branch_select_metric", "PEER")),
    )
    branch_metrics = ranked_branch_metrics





    for row, (_branch_name, _win, _step, seq) in zip(branch_metrics, candidate_branch_sequences):
        row["ARI"] = adjusted_rand_index(labels, seq)
        row["NMI"] = normalized_mutual_information(labels, seq)

    branch_sequences = [seq_tuple for idx, seq_tuple in enumerate(candidate_branch_sequences) if idx in selected_indices]
    run_state_counts = [cnt for idx, cnt in enumerate(candidate_run_state_counts) if idx in selected_indices]

    if not branch_sequences:
        raise RuntimeError(f"No branches selected for meta aggregation for {case.dataset}/{case.case_id}")

    if int(getattr(args, "select_top_k_branches", 0) or 0) > 0:
        selected_names = [branch_metrics[idx]["branch"] for idx in range(len(branch_metrics)) if branch_metrics[idx]["selected_for_meta"] == 1]
        print(
            f"  selected top {len(selected_names)}/{len(candidate_branch_sequences)} branches "
            f"by {args.branch_select_metric} with vote={args.meta_vote_weight_mode}: {', '.join(selected_names)}"
        )


    selected_branch_metrics = [row for row in branch_metrics if int(row.get("selected_for_meta", 0)) == 1]
    fusion_backend = str(getattr(args, "fusion_backend", "cspa") or "cspa").lower()

    if fusion_backend == "ec_tdwm":




        ec_k = choose_ec_tdwm_k_from_selected(run_state_counts, len(labels), args)
        meta_seq_raw, ratio_df, state_info_df, ec_engine, ec_iter, ec_change = ec_tdwm_consensus_sequence(
            branch_sequences, ec_k, args, runtime, meta_min_len
        )
        meta_seq = align_sequence(meta_seq_raw, len(labels), np)
        ec_segments = count_segments(meta_seq)
        scored = [{
            "K": int(ec_k),
            "dominant_ratio": "",
            "balance_entropy": normalized_entropy(meta_seq, np),
            "mean_segment_length": float(len(meta_seq) / max(1, ec_segments)),
            "segment_count": int(ec_segments),
            "coherence": "",
            "selection_score": 0.0,
            "fusion_backend": "ec_tdwm",
            "ec_tdwm_engine": ec_engine,
            "ec_tdwm_k_mode": str(getattr(args, "ec_tdwm_k_mode", "median")),
            "ec_tdwm_iterations": int(ec_iter),
            "ec_tdwm_final_change": float(ec_change),
            "seq": meta_seq,
            "ratio_df": ratio_df,
            "state_info": state_info_df,
        }]
        best = scored[0]
        indicator_df, _tmp_state_info_df = build_indicator(branch_sequences, runtime)
    elif fusion_backend == "cspa":




        cspa_k = choose_cspa_k_from_selected(run_state_counts, len(labels), args)
        meta_seq_raw, ratio_df, state_info_df, cspa_engine = cspa_consensus_sequence(
            branch_sequences, cspa_k, args, runtime, meta_min_len
        )
        meta_seq = align_sequence(meta_seq_raw, len(labels), np)
        cspa_segments = count_segments(meta_seq)
        scored = [{
            "K": int(cspa_k),
            "dominant_ratio": "",
            "balance_entropy": normalized_entropy(meta_seq, np),
            "mean_segment_length": float(len(meta_seq) / max(1, cspa_segments)),
            "segment_count": int(cspa_segments),
            "coherence": "",
            "selection_score": 0.0,
            "fusion_backend": "cspa",
            "cspa_engine": cspa_engine,
            "cspa_k_mode": str(getattr(args, "cspa_k_mode", "median")),
            "seq": meta_seq,
            "ratio_df": ratio_df,
            "state_info": state_info_df,
        }]
        best = scored[0]
        indicator_df, _tmp_state_info_df = build_indicator(branch_sequences, runtime)
    else:

        indicator_df, state_info_df = build_indicator(branch_sequences, runtime)
        state_info_df = attach_branch_weights_to_state_info(
            state_info_df,
            selected_branch_metrics,
            str(getattr(args, "meta_vote_weight_mode", "branch_reliability")),
        )
        best, scored = choose_meta_k(indicator_df, state_info_df, run_state_counts, args, runtime, meta_min_len)
        meta_seq = align_sequence(best["seq"], len(labels), np)
        ratio_df = best["ratio_df"]
        state_info_df = best["state_info"]

    ari = adjusted_rand_index(labels, meta_seq)
    nmi = normalized_mutual_information(labels, meta_seq)
    elapsed_total = time.time() - start

    selected_branch_names = [row["branch"] for row in branch_metrics if int(row.get("selected_for_meta", 0)) == 1]

    return {
        "data": data,
        "labels": labels,
        "candidate_branch_sequences": candidate_branch_sequences,
        "branch_sequences": branch_sequences,
        "branch_metrics": branch_metrics,
        "indicator_df": indicator_df,
        "state_info": state_info_df,
        "k_scores": scored,
        "meta_seq": meta_seq,
        "meta_ratio_df": ratio_df,
        "ARI": ari,
        "NMI": nmi,
        "K": int(best["K"]),
        "n_pred_states": int(meta_seq.max()) + 1,
        "segments": count_segments(meta_seq),
        "meta_min_len": meta_min_len,
        "candidate_branches_ran": len(candidate_branch_sequences),
        "meta_branches_used": len(branch_sequences),
        "branch_select_metric": str(getattr(args, "branch_select_metric", "PEER")),
        "meta_vote_weight_mode": str(getattr(args, "meta_vote_weight_mode", "branch_reliability")),
        "fusion_backend": fusion_backend,
        "cspa_k_mode": str(getattr(args, "ec_tdwm_k_mode", getattr(args, "cspa_k_mode", "median"))) if fusion_backend == "ec_tdwm" else str(getattr(args, "cspa_k_mode", "median")),
        "selected_branch_names": "|".join(selected_branch_names),
        "seconds": elapsed_total,
    }

def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


CASE_RESULT_FIELDS = ["dataset", "case_id", "rows", "candidate_branches_ran", "meta_branches_used", "branches_ran", "selected_K", "n_pred_states", "segments", "meta_min_len", "branch_select_metric", "meta_vote_weight_mode", "fusion_backend", "cspa_k_mode", "selected_branch_names", "ARI", "NMI", "seconds"]


def load_existing_case_rows(out_dir: Path) -> list[dict[str, object]]:
    path = out_dir / "case_results.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def completed_case_keys(case_rows: Sequence[dict[str, object]]) -> set[tuple[str, str]]:
    return {(str(row.get("dataset", "")), str(row.get("case_id", ""))) for row in case_rows}


def parse_selection_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            parts.extend(parse_selection_values(item))
        return parts
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()]


def parse_case_indexes(value: object) -> list[int]:
    indexes = []
    for part in parse_selection_values(value):
        try:
            idx = int(part)
        except ValueError as exc:
            raise ValueError(f"case index filters must be integers, got: {part!r}") from exc
        if idx < 1:
            raise ValueError(f"case index filters are 1-based and must be >= 1, got: {idx}")
        indexes.append(idx)
    return indexes


def apply_case_selection(cases: list[SeriesCase], args: argparse.Namespace) -> list[SeriesCase]:
    selected_ids = set(parse_selection_values(getattr(args, "case_ids", None)))
    selected_ids.update(parse_selection_values(getattr(args, "only_case_ids", "")))
    selected_indexes = set(parse_case_indexes(getattr(args, "rounds", "") or getattr(args, "case_indexes", "")))

    if selected_ids or selected_indexes:
        loaded = ", ".join(f"{idx}:{case.dataset}/{case.case_id}" for idx, case in enumerate(cases, start=1))
        before = len(cases)
        cases = [
            case
            for idx, case in enumerate(cases, start=1)
            if (selected_ids and str(case.case_id) in selected_ids) or (idx in selected_indexes)
        ]
        print(f"[case-selection] kept {len(cases)}/{before} case(s). Loaded order: {loaded}")
        if not cases:
            raise ValueError(
                "Case selection matched no cases. "
                f"Requested case_ids={sorted(selected_ids)} indexes={sorted(selected_indexes)}. "
                f"Loaded cases are: {loaded or '<none>'}"
            )

    priority_ids = parse_selection_values(getattr(args, "priority_case_ids", ""))
    if priority_ids:
        rank = {case_id: idx for idx, case_id in enumerate(priority_ids)}
        cases = sorted(
            cases,
            key=lambda case: (
                rank.get(str(case.case_id), len(priority_ids)),
                str(case.dataset),
                str(case.case_id),
            ),
        )
        print("[priority-case-ids] first cases:", ", ".join(str(case.case_id) for case in cases[:10]))

    return cases



def export_all_k_cluster_outputs(case: SeriesCase, result: dict[str, object], args, runtime, out_dir: Path) -> None:
    """
    Export every meta-clustering K candidate, not only the selected K.

    This lets you diagnose whether:
    1) a good K/meta cluster exists but the unsupervised K-selection did not choose it; or
    2) every K candidate is poor, meaning the selected experts / meta aggregation itself is weak.

    Files written:
    - <case_id>_all_k_meta_states.csv
    - <case_id>_all_k_metrics.csv
    - all_k_clusters/<case_id>_Kxx_cluster_members.csv
    - all_k_clusters/<case_id>_Kxx_meta_ratios.csv
    """
    np = runtime["np"]
    pd = runtime["pd"]
    labels = result["labels"]
    selected_K = int(result["K"])

    all_state_df = pd.DataFrame({"true_label": labels})
    metric_rows = []
    cluster_dir = out_dir / "all_k_clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for r in result.get("k_scores", []):
        K = int(r["K"])
        seq = align_sequence(r["seq"], len(labels), np)
        seq, _ = reorder_label(seq, np)

        col = f"meta_K{K}"
        all_state_df[col] = seq

        metric_rows.append(
            {
                "K": K,
                "selected_by_rule": 1 if K == selected_K else 0,
                "ARI": adjusted_rand_index(labels, seq),
                "NMI": normalized_mutual_information(labels, seq),
                "n_pred_states": int(seq.max()) + 1 if len(seq) else 0,
                "segments": count_segments(seq),
                "dominant_ratio": r.get("dominant_ratio", ""),
                "balance_entropy": r.get("balance_entropy", ""),
                "mean_segment_length": r.get("mean_segment_length", ""),
                "segment_count_raw": r.get("segment_count", ""),
                "coherence": r.get("coherence", ""),
                "selection_score": r.get("selection_score", ""),
                "meta_min_len": result.get("meta_min_len", args.meta_min_len),
                "branch_select_metric": result.get("branch_select_metric", getattr(args, "branch_select_metric", "")),
                "meta_vote_weight_mode": result.get("meta_vote_weight_mode", getattr(args, "meta_vote_weight_mode", "")),
                "fusion_backend": result.get("fusion_backend", getattr(args, "fusion_backend", "")),
                "cspa_k_mode": result.get("cspa_k_mode", getattr(args, "cspa_k_mode", "")),
                "selected_branch_names": result.get("selected_branch_names", ""),
            }
        )


        state_info = r.get("state_info", None)
        if state_info is not None:
            state_info.to_csv(
                cluster_dir / f"{case.case_id}_K{K:02d}_cluster_members.csv",
                index=False,
                encoding="utf-8-sig",
            )


        ratio_df = r.get("ratio_df", None)
        if ratio_df is not None:
            ratio_out = pd.concat([pd.DataFrame({"true_label": labels, f"meta_K{K}": seq}), ratio_df.reset_index(drop=True)], axis=1)
            ratio_out.to_csv(
                cluster_dir / f"{case.case_id}_K{K:02d}_meta_ratios.csv",
                index=False,
                encoding="utf-8-sig",
            )

    all_state_df.to_csv(out_dir / f"{case.case_id}_all_k_meta_states.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).sort_values("K").to_csv(
        out_dir / f"{case.case_id}_all_k_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )



def save_case_outputs(case: SeriesCase, result: dict[str, object], args, runtime) -> None:
    pd = runtime["pd"]
    out_dir = args.out_dir / "predictions" / case.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"true_label": result["labels"], "meta_state": result["meta_seq"]})
    for name, _win, _step, seq in result["branch_sequences"]:
        df[name] = seq
    df.to_csv(out_dir / f"{case.case_id}_states.csv", index=False, encoding="utf-8-sig")
    result["state_info"].to_csv(out_dir / f"{case.case_id}_state_info.csv", index=False, encoding="utf-8-sig")
    k_rows = []
    for r in result.get("k_scores", []):
        k_rows.append(
            {
                "K": r.get("K", result.get("K", "")),
                "dominant_ratio": r.get("dominant_ratio", ""),
                "balance_entropy": r.get("balance_entropy", ""),
                "mean_segment_length": r.get("mean_segment_length", ""),
                "segment_count": r.get("segment_count", ""),
                "meta_min_len": result.get("meta_min_len", args.meta_min_len),
                "coherence": r.get("coherence", ""),
                "selection_score": r.get("selection_score", ""),
                "fusion_backend": r.get("fusion_backend", result.get("fusion_backend", "")),
                "cspa_k_mode": r.get("cspa_k_mode", result.get("cspa_k_mode", "")),
                "cspa_engine": r.get("cspa_engine", ""),
                "ec_tdwm_k_mode": r.get("ec_tdwm_k_mode", result.get("ec_tdwm_k_mode", "")),
                "ec_tdwm_engine": r.get("ec_tdwm_engine", ""),
                "ec_tdwm_iterations": r.get("ec_tdwm_iterations", ""),
                "ec_tdwm_final_change": r.get("ec_tdwm_final_change", ""),
            }
        )
    pd.DataFrame(k_rows).to_csv(out_dir / f"{case.case_id}_k_sweep.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["branch_metrics"]).to_csv(out_dir / f"{case.case_id}_branches.csv", index=False, encoding="utf-8-sig")


    export_all_k_cluster_outputs(case, result, args, runtime, out_dir)


def summarize(case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[row["dataset"]].append(row)
    out = []
    for dataset, rows in grouped.items():
        out.append(
            {
                "dataset": dataset,
                "case_count": len(rows),
                "mean_ARI": sum(float(r["ARI"]) for r in rows) / len(rows),
                "mean_NMI": sum(float(r["NMI"]) for r in rows) / len(rows),
                "mean_pred_states": sum(float(r["n_pred_states"]) for r in rows) / len(rows),
                "mean_segments": sum(float(r["segments"]) for r in rows) / len(rows),
                "total_seconds": sum(float(r["seconds"]) for r in rows),
            }
        )
    return sorted(out, key=lambda r: str(r["dataset"]))


def build_comparison(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    with BASELINE_JSON.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)["metrics"]
    rows = []
    for row in summary_rows:
        dataset = str(row["dataset"])
        for metric, our_col in [("ARI", "mean_ARI"), ("NMI", "mean_NMI")]:
            values = baseline.get(metric, {}).get(dataset)
            if not values:
                continue
            baseline_without_t2s = {k: v for k, v in values.items() if k != "Time2State"}
            best_name, best_value = max(baseline_without_t2s.items(), key=lambda kv: kv[1])
            our_value = float(row[our_col])
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "multi_t2s": our_value,
                    "paper_Time2State": values["Time2State"],
                    "best_paper_baseline": best_name,
                    "best_paper_baseline_value": best_value,
                    "delta_vs_paper_Time2State": our_value - float(values["Time2State"]),
                    "delta_vs_best_paper_baseline": our_value - float(best_value),
                    "beats_paper_Time2State": our_value > float(values["Time2State"]),
                    "beats_best_paper_baseline": our_value > float(best_value),
                }
            )
    return rows


def write_status(args, dataset_status, runtime_status, error: str | None = None) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "repo_root": str(args.repo_root),
        "out_dir": str(args.out_dir),
        "datasets_requested": args.datasets,
        "branches": args.branches,
        "branch_config_txt": str(args.branch_config_txt.resolve()) if args.branch_config_txt else None,
        "branch_config_counts": {k: len(v) for k, v in getattr(args, "branch_config_map", {}).items()},
        "nb_steps": args.nb_steps,
        "meta_min_len_cli_default": args.meta_min_len,
        "meta_min_len_note": "If branch_config_txt has a meta_min_len column, the first non-empty value for each dataset overrides --meta-min-len.",
        "public_data_root": str(get_public_data_root(args)),
        "public_max_rows": args.public_max_rows,
        "pamap2_feature_mode_requested": str(getattr(args, "pamap2_feature_mode", "auto")),
        "pamap2_feature_mode_resolved": _resolve_pamap2_feature_mode(args) if "pamap2" in [normalize_dataset_key(x) for x in args.datasets] else None,
        "pamap2_remove_zero": _resolve_pamap2_remove_zero(args) if "pamap2" in [normalize_dataset_key(x) for x in args.datasets] else None,
        "skip_completed": bool(args.skip_completed),
        "select_top_k_branches": int(getattr(args, "select_top_k_branches", 0) or 0),
        "branch_select_metric": str(getattr(args, "branch_select_metric", "")),
        "meta_vote_weight_mode": str(getattr(args, "meta_vote_weight_mode", "")),
        "peer_health_weight": float(getattr(args, "peer_health_weight", 0.45)),
        "peer_consensus_weight": float(getattr(args, "peer_consensus_weight", 0.55)),
        "pid_kp": float(getattr(args, "pid_kp", 0.45)),
        "pid_ki": float(getattr(args, "pid_ki", 0.35)),
        "pid_kd": float(getattr(args, "pid_kd", 0.20)),
        "pid_scale_prior_enabled": False,
        "pid_win_ref": "disabled",
        "pid_scale_sigma": "disabled",
        "pid_softmax_tau": float(getattr(args, "pid_softmax_tau", 0.15)),
        "m2qd_alpha": float(getattr(args, "m2qd_alpha", 0.5)),
        "loader_mode": "strict_exp_of_Time2State_preprocessing",
        "loader_mode_note": "Data reading, feature columns, normalization, NaN handling, and PAMAP2/USC-HAD fit-predict protocols are reproduced from exp_of_Time2State.py where applicable.",
        "branch_selection_warning": "Clean runner hard-rejects ARI/NMI/ARI_NMI branch selection and oracle_score voting. Branch ARI/NMI are written only after selection for diagnostics.",
        "all_k_export": True,
        "all_k_export_note": "For each case, *_all_k_metrics.csv, *_all_k_meta_states.csv, and all_k_clusters/* are exported to diagnose K/meta-cluster candidates.",
        "dataset_status": dataset_status,
        "runtime_status": runtime_status,
        "error": error,
    }
    (args.out_dir / "run_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")



def branch_metric_rows_for_global_csv(case: SeriesCase, result: dict[str, object]) -> list[dict[str, object]]:
    """
    Flatten this case's branch_metrics into root-level all_branch_results.csv rows.

    Each row = one branch/expert result for one case.
    This makes it easy to check, across all cases:
      - which branches were selected;
      - branch ARI/NMI;
      - PID/PEER scores;
      - whether meta is above branch mean/median/max.
    """
    rows = []
    for idx, row in enumerate(result.get("branch_metrics", []), start=1):
        out = dict(row)
        out["dataset"] = case.dataset
        out["case_id"] = case.case_id
        out["branch_order_in_case"] = idx
        out["meta_ARI"] = result.get("ARI", "")
        out["meta_NMI"] = result.get("NMI", "")
        out["meta_selected_K"] = result.get("K", "")
        out["meta_segments"] = result.get("segments", "")
        out["meta_pred_states"] = result.get("n_pred_states", "")
        out["meta_branches_used"] = result.get("meta_branches_used", "")
        out["candidate_branches_ran"] = result.get("candidate_branches_ran", "")
        out["selected_branch_names"] = result.get("selected_branch_names", "")
        rows.append(out)
    return rows


def write_combined_branch_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """
    Write all branch metrics from all cases into one CSV.
    The columns are the union of all row keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    preferred = [
        "dataset", "case_id", "branch_order_in_case",
        "branch", "win", "step",
        "selected_for_meta", "branch_rank", "branch_selection_score",
        "ARI", "NMI", "states", "segments",
        "health", "peer_consensus", "peer_reliability_norm",
        "pid_p", "pid_i", "pid_d", "pid_scale_prior",
        "pid_neighborhood_stability", "pid_raw_score", "pid_score_norm",
        "pid_weight", "pid_weight_norm",
        "meta_ARI", "meta_NMI", "meta_selected_K", "meta_segments", "meta_pred_states",
        "meta_branches_used", "candidate_branches_ran", "selected_branch_names",
        "m", "n", "out_channels", "nb_steps", "kernel_size", "win_type",
        "branch_min_len", "meta_min_len", "seconds",
    ]

    all_keys = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            all_keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    write_csv(path, rows, all_keys)


def write_case_level_branch_summary(path: Path, case_rows: list[dict[str, object]], branch_rows: list[dict[str, object]]) -> None:
    """
    Write one summary row per case:
      Meta vs selected-branch mean/median/max and all-candidate mean/median/max.
    This directly tells whether the case reaches level 1/2/3.
    """
    try:
        import pandas as _pd
    except Exception:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if not case_rows:
        write_csv(path, [], [])
        return

    bdf = _pd.DataFrame(branch_rows)
    cdf = _pd.DataFrame(case_rows)

    out_rows = []
    for _, crow in cdf.iterrows():
        dataset = str(crow.get("dataset", ""))
        case_id = str(crow.get("case_id", ""))
        meta_ari = float(crow.get("ARI", 0.0))
        meta_nmi = float(crow.get("NMI", 0.0))

        sub = bdf[(bdf["dataset"].astype(str) == dataset) & (bdf["case_id"].astype(str) == case_id)].copy()
        selected = sub[sub.get("selected_for_meta", 0).astype(int) == 1].copy() if len(sub) else sub

        def _stats(df, col):
            if df is None or len(df) == 0 or col not in df.columns:
                return {"mean": "", "median": "", "max": ""}
            vals = _pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                return {"mean": "", "median": "", "max": ""}
            return {"mean": float(vals.mean()), "median": float(vals.median()), "max": float(vals.max())}

        sel_ari = _stats(selected, "ARI")
        sel_nmi = _stats(selected, "NMI")
        all_ari = _stats(sub, "ARI")
        all_nmi = _stats(sub, "NMI")

        def _level(meta, stats):
            if stats["max"] != "" and meta > stats["max"]:
                return 3
            if stats["median"] != "" and meta > stats["median"]:
                return 2
            if stats["mean"] != "" and meta > stats["mean"]:
                return 1
            return 0

        row = dict(crow)
        row.update({
            "selected_branch_ARI_mean": sel_ari["mean"],
            "selected_branch_ARI_median": sel_ari["median"],
            "selected_branch_ARI_max": sel_ari["max"],
            "selected_branch_NMI_mean": sel_nmi["mean"],
            "selected_branch_NMI_median": sel_nmi["median"],
            "selected_branch_NMI_max": sel_nmi["max"],
            "all_branch_ARI_mean": all_ari["mean"],
            "all_branch_ARI_median": all_ari["median"],
            "all_branch_ARI_max": all_ari["max"],
            "all_branch_NMI_mean": all_nmi["mean"],
            "all_branch_NMI_median": all_nmi["median"],
            "all_branch_NMI_max": all_nmi["max"],
            "level_vs_selected_by_ARI": _level(meta_ari, sel_ari),
            "level_vs_selected_by_NMI": _level(meta_nmi, sel_nmi),
            "level_vs_all_candidates_by_ARI": _level(meta_ari, all_ari),
            "level_vs_all_candidates_by_NMI": _level(meta_nmi, all_nmi),
        })
        out_rows.append(row)

    if out_rows:
        fieldnames = list(out_rows[0].keys())
        write_csv(path, out_rows, fieldnames)


def validate_clean_evaluation_args(args: argparse.Namespace) -> None:
    metric = str(getattr(args, "branch_select_metric", "PEER")).upper()
    vote_mode = str(getattr(args, "meta_vote_weight_mode", "branch_reliability")).lower()
    supervised_metrics = {"ARI", "NMI", "ARI_NMI"}

    if metric in supervised_metrics:
        raise ValueError(
            f"Invalid clean evaluation setting: branch_select_metric={metric} uses ground-truth labels. "
            "Use PEER or PID for our method."
        )
    if vote_mode == "oracle_score":
        raise ValueError(
            "Invalid clean evaluation setting: meta_vote_weight_mode=oracle_score uses ground-truth labels. "
            "Use uniform, branch_reliability, or pid_weight for our method."
        )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.out_dir = args.out_dir.resolve()
    validate_clean_evaluation_args(args)
    random.seed(args.seed)

    runtime_status = {"ok": False}
    dataset_status: list[dict[str, object]] = []
    try:
        runtime = import_runtime(args.repo_root)
        np = runtime["np"]
        torch = runtime["torch"]
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        runtime_status = {
            "ok": True,
            "python": sys.executable,
            "numpy": runtime["np"].__version__,
            "pandas": runtime["pd"].__version__,
            "torch": torch.__version__,
        }
        cases, dataset_status = load_cases(args, runtime)
        cases = apply_case_selection(cases, args)

        if bool(getattr(args, "ucrseg_worst_first", False)):
            worst_first_ids = [
                "MedicalImages",
                "ECGFiveDays",
                "InlineSkate",
                "DistalPhalanxOutlineAgeGroup",
                "FacesUCR",
                "Haptics",
                "CricketY",
                "Crop",
                "Car",
                "CricketZ",
                "Meat",
                "FreezerRegularTrain",
                "EOGHorizontalSignal",
                "DiatomSizeReduction",
                "BirdChicken",
                "CBF",
            ]
            rank = {case_id: i for i, case_id in enumerate(worst_first_ids)}
            cases = sorted(cases, key=lambda c: (rank.get(str(c.case_id), len(rank)), str(c.case_id)))
            print("Worst-first order enabled. First cases:", ", ".join(str(c.case_id) for c in cases[:10]))

        args.branch_config_map = load_branch_config_txt(args.branch_config_txt, args)
        if args.branch_config_map:
            for ds_key, ds_branches in sorted(args.branch_config_map.items()):
                print(f"[branch-config] {ds_key}: {len(ds_branches)} branches")
        else:
            print(f"[branch-config] using fallback --branches: {args.branches}")
        existing_rows = load_existing_case_rows(args.out_dir) if args.skip_completed else []
        done = completed_case_keys(existing_rows)
        if args.skip_completed and done:
            before = len(cases)
            cases = [case for case in cases if (case.dataset, case.case_id) not in done]
            print(f"Skipping {before - len(cases)} completed cases from {args.out_dir / 'case_results.csv'}")
        case_rows = list(existing_rows)
        all_branch_rows: list[dict[str, object]] = []
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx}/{len(cases)}] {case.dataset}/{case.case_id}")
            result = run_multit2s_case(case, args, runtime)
            save_case_outputs(case, result, args, runtime)
            row = {
                "dataset": case.dataset,
                "case_id": case.case_id,
                "rows": len(result["labels"]),
                "candidate_branches_ran": result.get("candidate_branches_ran", len(result.get("branch_sequences", []))),
                "meta_branches_used": result.get("meta_branches_used", len(result.get("branch_sequences", []))),
                "branches_ran": result.get("meta_branches_used", len(result.get("branch_sequences", []))),
                "selected_K": result["K"],
                "n_pred_states": result["n_pred_states"],
                "segments": result["segments"],
                "meta_min_len": result.get("meta_min_len", args.meta_min_len),
                "branch_select_metric": result.get("branch_select_metric", getattr(args, "branch_select_metric", "")),
                "meta_vote_weight_mode": result.get("meta_vote_weight_mode", getattr(args, "meta_vote_weight_mode", "")),
                "fusion_backend": result.get("fusion_backend", getattr(args, "fusion_backend", "")),
                "cspa_k_mode": result.get("cspa_k_mode", getattr(args, "cspa_k_mode", "")),
                "selected_branch_names": result.get("selected_branch_names", ""),
                "ARI": result["ARI"],
                "NMI": result["NMI"],
                "seconds": result["seconds"],
            }
            case_rows.append(row)
            all_branch_rows.extend(branch_metric_rows_for_global_csv(case, result))
            print(f"  ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} K={row['selected_K']} seconds={row['seconds']:.1f}")

        summary_rows = summarize(case_rows)
        comparison_rows = build_comparison(summary_rows)
        write_csv(args.out_dir / "case_results.csv", case_rows, CASE_RESULT_FIELDS)

        write_csv(args.out_dir / "all_case_results.csv", case_rows, CASE_RESULT_FIELDS)
        write_combined_branch_csv(args.out_dir / "all_branch_results.csv", all_branch_rows)
        write_case_level_branch_summary(args.out_dir / "case_branch_level_summary.csv", case_rows, all_branch_rows)

        write_csv(args.out_dir / "dataset_summary.csv", summary_rows, ["dataset", "case_count", "mean_ARI", "mean_NMI", "mean_pred_states", "mean_segments", "total_seconds"])
        write_csv(args.out_dir / "baseline_comparison.csv", comparison_rows, ["dataset", "metric", "multi_t2s", "paper_Time2State", "best_paper_baseline", "best_paper_baseline_value", "delta_vs_paper_Time2State", "delta_vs_best_paper_baseline", "beats_paper_Time2State", "beats_best_paper_baseline"])
        write_status(args, dataset_status, runtime_status)
        print(f"Wrote {args.out_dir / 'all_case_results.csv'}")
        print(f"Wrote {args.out_dir / 'all_branch_results.csv'}")
        print(f"Wrote {args.out_dir / 'case_branch_level_summary.csv'}")
        print(f"Wrote {args.out_dir / 'dataset_summary.csv'}")
        print(f"Wrote {args.out_dir / 'baseline_comparison.csv'}")
    except Exception as exc:
        write_status(args, dataset_status, runtime_status, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
