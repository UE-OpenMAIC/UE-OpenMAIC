from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

THIS_DIR = Path(__file__).resolve().parent
OURCLAP_ROOT = THIS_DIR.parent
REPO_ROOT_DEFAULT = OURCLAP_ROOT.parent
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def parse_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Expected boolean value, got: {value!r}")


def parse_config(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    settings: dict[str, str] = {}
    branch_lines: list[str] = []
    in_branches = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower() in {"[clap_branches]", "[branches]"}:
            in_branches = True
            continue
        if in_branches:
            branch_lines.append(raw)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            raise ValueError(f"Config line {line_no}: unsupported section {stripped!r}")
        if "=" not in stripped:
            raise ValueError(f"Config line {line_no} should be key=value or [clap_branches]: {raw}")
        key, value = stripped.split("=", 1)
        settings[key.strip().lower().replace("-", "_")] = value.strip()

    if not branch_lines:
        raise ValueError("Missing [clap_branches] table in config")
    reader = csv.DictReader(StringIO("\n".join(branch_lines)))
    required = {"branch_name", "clap_window_size", "clap_classifier", "clap_merge_score"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"CLaP branch table is missing required columns: {sorted(missing)}")
    rows: list[dict[str, str]] = []
    for row in reader:
        enabled = str(row.get("enabled", "1")).strip().lower()
        if enabled in {"0", "false", "no", "n", "off"}:
            continue
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise ValueError("All CLaP branches are disabled")
    return settings, rows


def expand_path_value(value: str | None, *, default: Path, config_dir: Path, repo_root: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    text = str(value).strip()
    text = text.replace("{THIS_DIR}", str(config_dir))
    text = text.replace("{CONFIG_DIR}", str(config_dir))
    text = text.replace("{OURCLAP_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{OUR_ROOT}", str(OURCLAP_ROOT))
    text = text.replace("{REPO_ROOT}", str(repo_root))
    text = text.replace("{CLAP_ROOT}", str(repo_root / "classification-label-profile-main"))
    path = Path(text)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _int_setting(settings: dict[str, str], key: str, default: int) -> int:
    value = str(settings.get(key, "")).strip()
    if value == "":
        return default
    return int(float(value))


def _float_setting(settings: dict[str, str], key: str, default: float) -> float:
    value = str(settings.get(key, "")).strip()
    if value == "":
        return default
    return float(value)


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


def reorder_labels(seq) -> np.ndarray:
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


def labels_from_cps(cps, labels, n: int) -> np.ndarray:
    cps = sanitize_cps(cps, n).tolist()
    labels = [int(x) for x in list(labels)]
    n_seg = len(cps) + 1
    if len(labels) < n_seg:
        labels = labels + ([labels[-1]] * (n_seg - len(labels)) if labels else [0] * n_seg)
    elif len(labels) > n_seg:
        labels = labels[:n_seg]
    y = np.zeros(n, dtype=int)
    start = 0
    for i, end in enumerate(cps + [n]):
        y[start:end] = int(labels[i])
        start = end
    return reorder_labels(y)


def normalize_input_array(ts, mode: str) -> np.ndarray:
    arr = np.asarray(ts, dtype=float)
    if mode == "none" or mode == "":
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
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
        raise ValueError(f"Unsupported normalize_input={mode!r}")
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    return work.reshape(-1) if arr.ndim == 1 else work


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


def load_official_tssb_and_segmentation(clap_repo: Path):
    add_clap_repo(clap_repo)
    from src.utils import load_tssb_datasets
    df = load_tssb_datasets()
    seg_path = clap_repo / "experiments" / "segmentation" / "TSSB_ClaSP.csv.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Cannot find official ClaSP segmentation file: {seg_path}")
    converters = {"found_cps": lambda data: np.array(eval(data), dtype=int)}
    seg_df = pd.read_csv(seg_path, converters=converters)[["dataset", "found_cps"]]
    return df, seg_df


def run_clap_branch(ts, init_cps, branch: dict[str, str], n_jobs: int, seed: int):
    from src.clap import CLaP
    normalize = str(branch.get("normalize_input", "none") or "none").strip()
    ts_in = normalize_input_array(ts, normalize)
    kwargs = {
        "window_size": str(branch.get("clap_window_size", "suss") or "suss"),
        "classifier": str(branch.get("clap_classifier", "rocket") or "rocket"),
        "merge_score": str(branch.get("clap_merge_score", "cgain") or "cgain"),
        "n_splits": int(float(branch.get("n_splits", 5) or 5)),
        "sample_size": int(float(branch.get("sample_size", 1000) or 1000)),
        "n_jobs": int(n_jobs),
        "random_state": int(seed),
    }
    clap = CLaP(**kwargs)
    clap.fit(ts_in, init_cps)
    final_cps = sanitize_cps(clap.get_change_points(), len(ts_in))
    seg_labels = safe_int_array(clap.get_segment_labels())
    pred = labels_from_cps(final_cps, seg_labels, len(ts_in))
    return final_cps, seg_labels, pred


def count_segments(seq) -> int:
    values = list(seq)
    if not values:
        return 0
    return 1 + sum(1 for a, b in zip(values, values[1:]) if a != b)


def normalized_entropy(seq) -> float:
    seq = np.asarray(seq, dtype=int)
    values, counts = np.unique(seq, return_counts=True)
    if len(values) <= 1:
        return 0.0
    p = counts.astype(float) / counts.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / np.log(len(values)))


def short_segment_ratio(seq, short_len: int) -> float:
    seq = np.asarray(seq, dtype=int)
    if len(seq) == 0:
        return 1.0
    short = 0
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[start]:
            if i - start < short_len:
                short += i - start
            start = i
    if len(seq) - start < short_len:
        short += len(seq) - start
    return float(short) / float(len(seq))


def branch_health_score(seq, median_segments: float | None = None) -> dict[str, float]:
    seq = np.asarray(seq, dtype=int)
    n = int(len(seq))
    if n <= 0:
        return {"health": 0.0, "state_entropy": 0.0, "dominant_ratio": 1.0, "short_segment_ratio": 1.0, "segments": 0.0, "n_states": 0.0}
    values, counts = np.unique(seq, return_counts=True)
    n_states = int(len(values))
    seg_count = int(count_segments(seq))
    dom = float(counts.max()) / float(n)
    if n_states < 2 or seg_count < 2:
        return {"health": 0.0, "state_entropy": 0.0, "dominant_ratio": dom, "short_segment_ratio": 1.0, "segments": float(seg_count), "n_states": float(n_states)}
    ent = normalized_entropy(seq)
    dom_score = 1.0 if dom <= 0.75 else (0.0 if dom >= 0.98 else (0.98 - dom) / (0.98 - 0.75))
    short_len = max(2, int(round(0.005 * n)))
    short_ratio = short_segment_ratio(seq, short_len)
    frag_score = max(0.0, 1.0 - min(1.0, short_ratio / 0.35))
    if median_segments is None or median_segments <= 0:
        seg_score = 1.0
    else:
        seg_score = math.exp(-abs(math.log((seg_count + 1e-9) / (median_segments + 1e-9))))
    health = 0.40 * ent + 0.25 * dom_score + 0.20 * frag_score + 0.15 * seg_score
    return {"health": float(health), "state_entropy": float(ent), "dominant_ratio": float(dom), "short_segment_ratio": float(short_ratio), "segments": float(seg_count), "n_states": float(n_states)}


def pairwise_consensus(sequences: list[np.ndarray]) -> list[float]:
    if len(sequences) <= 1:
        return [1.0 for _ in sequences]
    out = []
    for i, seq_i in enumerate(sequences):
        vals = []
        for j, seq_j in enumerate(sequences):
            if i == j:
                continue
            vals.append(adjusted_rand_score(seq_i, seq_j))
                                                             
        out.append(float((np.mean(vals) + 1.0) / 2.0))
    return out


def softmax_weights(scores: list[float], tau: float) -> list[float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    tau = max(1e-6, float(tau))
    arr = arr - np.max(arr)
    ex = np.exp(arr / tau)
    s = float(ex.sum())
    if s <= 1e-12:
        return [1.0 / len(arr) for _ in arr]
    return (ex / s).astype(float).tolist()


def score_branches(branch_rows: list[dict[str, object]], settings: dict[str, str]) -> list[dict[str, object]]:
    seqs = [np.asarray(r["pred_seq"], dtype=int) for r in branch_rows]
    median_segments = float(np.median([count_segments(s) for s in seqs])) if seqs else 1.0
    consensus = pairwise_consensus(seqs)
    kp = _float_setting(settings, "pid_kp", 0.45)
    ki = _float_setting(settings, "pid_ki", 0.35)
    kd = _float_setting(settings, "pid_kd", 0.20)
    hw = _float_setting(settings, "peer_health_weight", 0.45)
    cw = _float_setting(settings, "peer_consensus_weight", 0.55)
    raw_scores = []
    metrics = []
    for row, cons in zip(branch_rows, consensus):
        h = branch_health_score(row["pred_seq"], median_segments)
        peer = hw * h["health"] + cw * cons
                                                                                               
        pid = kp * h["health"] + ki * cons - kd * h["short_segment_ratio"]
        raw_scores.append(pid)
        metrics.append((h, cons, peer, pid))
    weights = softmax_weights(raw_scores, _float_setting(settings, "pid_softmax_tau", 0.15))
    for idx, (row, (h, cons, peer, pid), w) in enumerate(zip(branch_rows, metrics, weights), start=1):
        row.update({
            "run_idx": idx,
            "health": h["health"],
            "state_entropy": h["state_entropy"],
            "dominant_ratio": h["dominant_ratio"],
            "short_segment_ratio": h["short_segment_ratio"],
            "segments": h["segments"],
            "n_states": h["n_states"],
            "peer_consensus": cons,
            "peer_reliability": peer,
            "pid_score": pid,
            "pid_weight_norm": w,
        })
    return branch_rows


def build_indicator(selected_rows: list[dict[str, object]]):
    mats = []
    info = []
    for run_idx, row in enumerate(selected_rows, start=1):
        seq = np.asarray(row["pred_seq"], dtype=int)
        for state_id in sorted(np.unique(seq).astype(int).tolist()):
            mats.append((seq == state_id).astype(np.float32))
            info.append({
                "global_state_name": f"{row['branch_name']}_state{state_id}",
                "run_idx": run_idx,
                "branch_name": row["branch_name"],
                "local_state": state_id,
                "branch_weight": float(row.get("pid_weight_norm", 1.0)),
            })
    H = np.stack(mats, axis=1).astype(np.float32)
    return pd.DataFrame(H, columns=[r["global_state_name"] for r in info]), pd.DataFrame(info)


def rolling_mean_axis1(arr, win):
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant")
    return (cumsum[:, win:] - cumsum[:, :-win]) / float(win)


def merge_short_segments(seq, min_len: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=int).copy()
    if min_len <= 1 or len(seq) == 0:
        return reorder_labels(seq)
    changed = True
    while changed:
        changed = False
        segs = []
        start = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[start]:
                segs.append((start, i - 1, int(seq[start])))
                start = i
        segs.append((start, len(seq) - 1, int(seq[start])))
        if len(segs) <= 1:
            break
        for pos, (l, r, val) in enumerate(segs):
            length = r - l + 1
            if length >= min_len:
                continue
            if pos == 0:
                fill = segs[pos + 1][2]
            elif pos == len(segs) - 1:
                fill = segs[pos - 1][2]
            else:
                prev_len = segs[pos - 1][1] - segs[pos - 1][0] + 1
                next_len = segs[pos + 1][1] - segs[pos + 1][0] + 1
                fill = segs[pos - 1][2] if prev_len >= next_len else segs[pos + 1][2]
            if fill != val:
                seq[l:r + 1] = fill
                changed = True
                break
    return reorder_labels(seq)


def cluster_state_matrix(indicator_df, state_info_df, n_clusters: int, smooth_win: int):
    H = indicator_df.to_numpy(dtype=np.float32)
    G = H.T
    G_smooth = rolling_mean_axis1(G, int(smooth_win))
    norm = np.linalg.norm(G_smooth, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    X = G_smooth / norm
    try:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    labels = reorder_labels(clusterer.fit_predict(X).astype(int))
    out = state_info_df.copy()
    out["cluster_id"] = labels
    return out


def meta_sequence_from_clusters(indicator_df, state_info_df, meta_min_len: int):
    H = indicator_df.to_numpy(dtype=np.float32)
    clusters = state_info_df["cluster_id"].to_numpy(dtype=int)
    weights = state_info_df.get("branch_weight", pd.Series(np.ones(len(clusters)))).to_numpy(dtype=np.float32)
    n_clusters = int(clusters.max()) + 1
    votes = np.zeros((H.shape[0], n_clusters), dtype=np.float32)
    for local_idx, cluster_id in enumerate(clusters):
        votes[:, cluster_id] += float(weights[local_idx]) * H[:, local_idx]
    seq = np.argmax(votes, axis=1).astype(int)
    return merge_short_segments(seq, int(meta_min_len))


def evaluate_k(indicator_df, state_info_df, k: int, settings: dict[str, str]):
    smooth = _int_setting(settings, "state_cluster_smooth", 9)
    meta_min_len = _int_setting(settings, "meta_min_len", 1)
    clustered = cluster_state_matrix(indicator_df, state_info_df, k, smooth)
    seq = meta_sequence_from_clusters(indicator_df, clustered, meta_min_len)
    values, counts = np.unique(seq, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    tau = max(0.05, 1.0 / float(k))
    dominant_count = int(np.sum(probs >= tau))
    mean_seg_len = float(len(seq) / max(1, count_segments(seq)))
    return {
        "K": int(k),
        "dominant_ratio": float(dominant_count) / float(k),
        "balance_entropy": normalized_entropy(seq),
        "mean_segment_length": mean_seg_len,
        "segment_count": count_segments(seq),
        "seq": seq,
        "state_info": clustered,
    }


def choose_meta_k(indicator_df, state_info_df, selected_rows, settings: dict[str, str]):
    run_state_counts = [int(len(np.unique(r["pred_seq"]))) for r in selected_rows]
    min_auto = max(2, min(run_state_counts))
    max_auto = min(max(run_state_counts), max(2, len(state_info_df) - 1))
    k_min_text = str(settings.get("meta_k_min", "auto")).strip().lower()
    k_max_text = str(settings.get("meta_k_max", "auto")).strip().lower()
    k_low = min_auto if k_min_text in {"", "auto"} else int(float(k_min_text))
    k_high = max_auto if k_max_text in {"", "auto"} else int(float(k_max_text))
    k_low = max(2, k_low)
    k_high = max(k_low, min(k_high, max(2, len(state_info_df) - 1)))
    raw = [evaluate_k(indicator_df, state_info_df, k, settings) for k in range(k_low, k_high + 1)]
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


def cps_from_label_seq(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=int)
    if len(seq) <= 1:
        return np.asarray([], dtype=int)
    return np.where(seq[:-1] != seq[1:])[0].astype(int) + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run configurable TSSB CLaP PID-Meta 12x4.")
    p.add_argument("--config", type=Path, default=THIS_DIR / "tssb_clap_pid_meta_12x4_config.txt")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    cli = parse_args()
    config_path = cli.config.resolve()
    config_dir = config_path.parent
    settings, branches = parse_config(config_path)

    repo_root = expand_path_value(settings.get("repo_root"), default=REPO_ROOT_DEFAULT, config_dir=config_dir, repo_root=REPO_ROOT_DEFAULT)
    clap_repo = expand_path_value(settings.get("clap_repo"), default=repo_root / "classification-label-profile-main", config_dir=config_dir, repo_root=repo_root)
    out_dir = expand_path_value(settings.get("out_dir"), default=config_dir / "results_tssb_clap_pid_meta_12x4_cgain", config_dir=config_dir, repo_root=repo_root)
    max_cases = _int_setting(settings, "max_cases", 0)
    n_jobs = _int_setting(settings, "n_jobs", 1)
    top_k = _int_setting(settings, "select_top_k_branches", 4)
    seed = _int_setting(settings, "seed", 1379)
    save_predictions = parse_bool(settings.get("save_predictions", "1"))

    print("============================================================")
    print("TSSB2 CLaP configurable 9x3 PID-Meta")
    print("Config   :", config_path)
    print("Repo root:", repo_root)
    print("CLaP repo:", clap_repo)
    print("Output   :", out_dir)
    print("Cases    :", "all" if max_cases <= 0 else max_cases)
    print("Branches :", len(branches))
    print("Top-k    :", top_k)
    print("n_jobs   :", n_jobs)
    print("============================================================")
    if cli.dry_run:
        print("Enabled branches:")
        for b in branches:
            print(" ", b)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    branch_pred_dir = out_dir / "branch_predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
        branch_pred_dir.mkdir(parents=True, exist_ok=True)

    add_clap_repo(clap_repo)
    from src.utils import create_state_labels
    df_data, seg_df = load_official_tssb_and_segmentation(clap_repo)
    if max_cases and max_cases > 0:
        df_data = df_data.iloc[:max_cases, :]

    case_csv = out_dir / "all_case_results.csv"
    branch_csv = out_dir / "all_branch_results.csv"
    k_csv = out_dir / "all_k_selection_results.csv"

    case_fields = [
        "dataset", "n", "true_cps_count", "init_cps_count", "pred_cps_count", "true_states", "pred_states",
        "selected_branches", "selected_count", "candidate_count", "metaK", "f1_score", "covering_score", "ami_score",
        "ari", "nmi", "runtime_seconds", "status", "error", "true_cps", "init_cps", "pred_cps",
    ]
    branch_fields = [
        "dataset", "branch_name", "selected_for_meta", "rank", "normalize_input", "clap_window_size", "clap_classifier",
        "clap_merge_score", "pred_states", "pred_cps_count", "f1_score", "covering_score", "ami_score", "ari", "nmi",
        "health", "peer_consensus", "peer_reliability", "pid_score", "pid_weight_norm", "runtime_seconds", "status", "error",
    ]
    k_fields = ["dataset", "K", "selected", "selection_score", "dominant_ratio", "balance_entropy", "coherence", "mean_segment_length", "segment_count"]

    all_case_rows = []
    all_branch_rows_out = []
    all_k_rows_out = []

    with case_csv.open("w", newline="", encoding="utf-8") as cf, branch_csv.open("w", newline="", encoding="utf-8") as bf, k_csv.open("w", newline="", encoding="utf-8") as kf:
        cw = csv.DictWriter(cf, fieldnames=case_fields)
        bw = csv.DictWriter(bf, fieldnames=branch_fields)
        kw = csv.DictWriter(kf, fieldnames=k_fields)
        cw.writeheader(); bw.writeheader(); kw.writeheader()

        for idx, (_, row) in enumerate(df_data.iterrows(), start=1):
            dataset = str(row["dataset"])
            cps_true = safe_int_array(row["change_points"])
            labels = safe_int_array(row["labels"])
            ts = safe_ts_array(row["time_series"])
            ts_len = int(ts.shape[0])
            print(f"[{idx}/{len(df_data)}] {dataset}", flush=True)
            t_case = time.process_time()
            status = "ok"; error = ""
            branch_runtime_total = 0.0
            try:
                hit = seg_df.loc[seg_df["dataset"] == dataset]
                if hit.empty:
                    raise KeyError(f"No official ClaSP found_cps for dataset={dataset!r}")
                init_cps = sanitize_cps(hit.iloc[0].found_cps, ts_len)
                y_true = create_state_labels(cps_true, labels, ts_len)
                y_true = reorder_labels(np.asarray(y_true, dtype=int))

                branch_rows: list[dict[str, object]] = []
                for b in branches:
                    b_start = time.process_time()
                    b_status = "ok"; b_error = ""
                    try:
                        final_cps, seg_labels, pred_seq = run_clap_branch(ts, init_cps, b, n_jobs=n_jobs, seed=seed)
                        b_metrics = evaluate_metrics(ts_len, cps_true, final_cps, y_true, pred_seq)
                    except Exception as exc:
                        b_status = "error"; b_error = repr(exc)
                        final_cps = np.asarray([], dtype=int)
                        seg_labels = np.asarray([], dtype=int)
                        pred_seq = np.zeros(ts_len, dtype=int)
                        b_metrics = {"f1_score": 0.0, "covering_score": 0.0, "ami_score": 0.0, "ari": 0.0, "nmi": 0.0}
                    b_seconds = time.process_time() - b_start
                    branch_runtime_total += b_seconds
                    br = {
                        "branch_name": b.get("branch_name", "branch"),
                        "branch_cfg": b,
                        "pred_seq": pred_seq,
                        "pred_cps": final_cps,
                        "pred_labels": seg_labels,
                        "branch_metrics": b_metrics,
                        "branch_seconds": b_seconds,
                        "branch_status": b_status,
                        "branch_error": b_error,
                    }
                    branch_rows.append(br)
                    if save_predictions:
                        safe_name = dataset.replace("/", "_").replace("\\", "_").replace(":", "_")
                        safe_branch = str(b.get("branch_name", "branch")).replace("/", "_").replace("\\", "_").replace(":", "_")
                        with (branch_pred_dir / f"{safe_name}__{safe_branch}.csv").open("w", newline="", encoding="utf-8") as pf:
                            pw = csv.writer(pf)
                            pw.writerow(["t", "true_label", "pred_label"])
                            for t, (yt, yp) in enumerate(zip(y_true, pred_seq)):
                                pw.writerow([t, int(yt), int(yp)])

                scored = score_branches(branch_rows, settings)
                scored_sorted = sorted(scored, key=lambda r: (-float(r["pid_score"]), str(r["branch_name"])))
                selected = scored_sorted[:max(1, min(top_k, len(scored_sorted)))]
                selected_names = [str(r["branch_name"]) for r in selected]

                                                 
                rank_map = {id(r): i for i, r in enumerate(scored_sorted, start=1)}
                selected_id = {id(r) for r in selected}
                for r in scored_sorted:
                    cfg = r["branch_cfg"]
                    bm = r["branch_metrics"]
                    outb = {
                        "dataset": dataset,
                        "branch_name": r["branch_name"],
                        "selected_for_meta": 1 if id(r) in selected_id else 0,
                        "rank": rank_map[id(r)],
                        "normalize_input": cfg.get("normalize_input", ""),
                        "clap_window_size": cfg.get("clap_window_size", ""),
                        "clap_classifier": cfg.get("clap_classifier", ""),
                        "clap_merge_score": cfg.get("clap_merge_score", ""),
                        "pred_states": int(len(np.unique(r["pred_seq"]))),
                        "pred_cps_count": int(len(r["pred_cps"])),
                        "f1_score": f"{bm['f1_score']:.10f}",
                        "covering_score": f"{bm['covering_score']:.10f}",
                        "ami_score": f"{bm['ami_score']:.10f}",
                        "ari": f"{bm['ari']:.10f}",
                        "nmi": f"{bm['nmi']:.10f}",
                        "health": f"{float(r['health']):.10f}",
                        "peer_consensus": f"{float(r['peer_consensus']):.10f}",
                        "peer_reliability": f"{float(r['peer_reliability']):.10f}",
                        "pid_score": f"{float(r['pid_score']):.10f}",
                        "pid_weight_norm": f"{float(r['pid_weight_norm']):.10f}",
                        "runtime_seconds": f"{float(r['branch_seconds']):.4f}",
                        "status": r["branch_status"],
                        "error": r["branch_error"],
                    }
                    bw.writerow(outb); all_branch_rows_out.append(outb)

                if len(selected) == 1:
                    meta_seq = np.asarray(selected[0]["pred_seq"], dtype=int)
                    meta_k = int(len(np.unique(meta_seq)))
                    k_scored = []
                else:
                    indicator_df, state_info_df = build_indicator(selected)
                    best_k, k_scored = choose_meta_k(indicator_df, state_info_df, selected, settings)
                    meta_seq = np.asarray(best_k["seq"], dtype=int)
                    meta_k = int(best_k["K"])
                    for kr in k_scored:
                        outk = {
                            "dataset": dataset,
                            "K": int(kr["K"]),
                            "selected": 1 if int(kr["K"]) == meta_k else 0,
                            "selection_score": f"{float(kr['selection_score']):.10f}",
                            "dominant_ratio": f"{float(kr['dominant_ratio']):.10f}",
                            "balance_entropy": f"{float(kr['balance_entropy']):.10f}",
                            "coherence": f"{float(kr['coherence']):.10f}",
                            "mean_segment_length": f"{float(kr['mean_segment_length']):.10f}",
                            "segment_count": int(kr["segment_count"]),
                        }
                        kw.writerow(outk); all_k_rows_out.append(outk)

                pred_cps = sanitize_cps(cps_from_label_seq(meta_seq), ts_len)
                metrics = evaluate_metrics(ts_len, cps_true, pred_cps, y_true, meta_seq)
                runtime = time.process_time() - t_case
                print(
                    f"  F1={metrics['f1_score']:.3f} Covering={metrics['covering_score']:.3f} "
                    f"AMI={metrics['ami_score']:.3f} ARI={metrics['ari']:.3f} NMI={metrics['nmi']:.3f} "
                    f"K={int(len(np.unique(meta_seq)))} metaK={meta_k} selected={len(selected)}/{len(branches)} seconds={runtime:.1f}",
                    flush=True,
                )
                if save_predictions:
                    safe_name = dataset.replace("/", "_").replace("\\", "_").replace(":", "_")
                    with (pred_dir / f"{safe_name}.csv").open("w", newline="", encoding="utf-8") as pf:
                        pw = csv.writer(pf)
                        pw.writerow(["t", "true_label", "pred_label"])
                        for t, (yt, yp) in enumerate(zip(y_true, meta_seq)):
                            pw.writerow([t, int(yt), int(yp)])
            except Exception as exc:
                status = "error"; error = repr(exc)
                runtime = time.process_time() - t_case
                init_cps = np.asarray([], dtype=int)
                pred_cps = np.asarray([], dtype=int)
                selected_names = []
                meta_k = 0
                meta_seq = np.zeros(ts_len, dtype=int)
                metrics = {"f1_score": 0.0, "covering_score": 0.0, "ami_score": 0.0, "ari": 0.0, "nmi": 0.0}
                print("  ERROR:", error, flush=True)

            case_out = {
                "dataset": dataset,
                "n": ts_len,
                "true_cps_count": int(len(cps_true)),
                "init_cps_count": int(len(init_cps)),
                "pred_cps_count": int(len(pred_cps)),
                "true_states": int(len(np.unique(labels))) if len(labels) else 0,
                "pred_states": int(len(np.unique(meta_seq))) if status == "ok" else 0,
                "selected_branches": json.dumps(selected_names, ensure_ascii=False),
                "selected_count": len(selected_names),
                "candidate_count": len(branches),
                "metaK": meta_k,
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
                "pred_cps": json.dumps([int(x) for x in list(pred_cps)]),
            }
            cw.writerow(case_out); all_case_rows.append(case_out)

    ok = [r for r in all_case_rows if r["status"] == "ok"]
    def mean_col(col: str) -> float:
        vals = [float(r[col]) for r in ok]
        return float(np.mean(vals)) if vals else 0.0
    summary = {
        "config": str(config_path),
        "repo_root": str(repo_root),
        "clap_repo": str(clap_repo),
        "out_dir": str(out_dir),
        "cases": len(all_case_rows),
        "ok": len(ok),
        "branches": len(branches),
        "select_top_k_branches": top_k,
        "mean_f1_score": mean_col("f1_score"),
        "mean_covering_score": mean_col("covering_score"),
        "mean_ami_score": mean_col("ami_score"),
        "mean_ari": mean_col("ari"),
        "mean_nmi": mean_col("nmi"),
    }
    (out_dir / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("============================================================")
    print(f"OK cases: {len(ok)}/{len(all_case_rows)}")
    print(f"Mean F1={summary['mean_f1_score']:.4f}")
    print(f"Mean Covering={summary['mean_covering_score']:.4f}")
    print(f"Mean AMI={summary['mean_ami_score']:.4f}")
    print(f"Mean ARI={summary['mean_ari']:.4f}")
    print(f"Mean NMI={summary['mean_nmi']:.4f}")
    print("Saved:", case_csv)
    print("Branches:", branch_csv)
    print("K table:", k_csv)
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
