                       

   

from __future__ import annotations

import argparse

import copy

import json

import math

import re

import time

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:

    import torch

except Exception:

    torch = None

from Time2State.time2state import Time2State

from Time2State.adapers import CausalConv_LSE_Adaper

from Time2State.clustering import DPGMM

from Time2State.default_params import params_LSE

                                                              

            

                                                              

T2S_OUTPUT_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

VISUAL_CSV_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\pose_csv")

OUT_DIRNAME_DEFAULT = "_cross_video_t2s_X1_compactY1_segment_multit2s_pid_8x4"

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

RUN_INFO_CSV = "multiscale_t2s_run_info.csv"

BRANCH_METRICS_CSV = "multiscale_branch_metrics_pid_peer.csv"

RAW_LABEL_CANDIDATES = [

    "multiscale_raw_label_matrix_Tx32.csv",

    "multiscale_raw_label_matrix.csv",

]

TIME_COL = "time_sec"

SKIP_STATE = -1

ACTION_SMOOTH = 7

                                              

T2S_M = 10

T2S_N = 4

T2S_OUT_CHANNELS = 8

T2S_NB_STEPS = 30

T2S_KERNEL_SIZE = None

T2S_WIN_TYPE = "hanning"

T2S_GPU = 0

T2S_USE_CUDA = bool(torch is not None and torch.cuda.is_available())

                        

PEER_HEALTH_WEIGHT = 0.45

PEER_CONSENSUS_WEIGHT = 0.55

PID_KP = 0.45

PID_KI = 0.35

PID_KD = 0.20

PID_SOFTMAX_TAU = 0.15

STATE_CLUSTER_SMOOTH = 9

META_MIN_LEN = 4

                                                              

         

                                                              

def ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)

def norm_video_id(x) -> str:

    s = str(x).strip().replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

    return re.sub(r"/+", "/", s).strip("/")

def tail_video_id(x) -> str:

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def remap_to_contiguous_labels(arr):

    vals = sorted(pd.unique(pd.Series(arr).dropna().astype(int)))

    mp = {int(v): i for i, v in enumerate(vals)}

    return np.array([mp[int(x)] for x in arr], dtype=int), mp

def split_segments(seq):

    seq = np.asarray(seq).astype(int)

    if len(seq) == 0:

        return []

    out = []

    s = 0

    for i in range(1, len(seq)):

        if seq[i] != seq[s]:

            out.append((s, i - 1, int(seq[s])))

            s = i

    out.append((s, len(seq) - 1, int(seq[s])))

    return out

def merge_short_segments(state_seq, min_len=4):

    seq = np.asarray(state_seq).astype(int).copy()

    n = len(seq)

    if n == 0 or min_len <= 1:

        return seq

    changed = True

    while changed:

        changed = False

        segs = split_segments(seq)

        if len(segs) <= 1:

            break

        for i, (s, e, label) in enumerate(segs):

            seg_len = e - s + 1

            if seg_len >= min_len:

                continue

            left_label = segs[i - 1][2] if i > 0 else None

            right_label = segs[i + 1][2] if i < len(segs) - 1 else None

            if left_label is None and right_label is None:

                continue

            elif left_label is None:

                seq[s:e + 1] = right_label

            elif right_label is None:

                seq[s:e + 1] = left_label

            else:

                left_len = segs[i - 1][1] - segs[i - 1][0] + 1

                right_len = segs[i + 1][1] - segs[i + 1][0] + 1

                seq[s:e + 1] = left_label if left_len >= right_len else right_label

            changed = True

            break

    return seq

def count_segments(seq) -> int:

    seq = np.asarray(seq, dtype=int)

    if len(seq) == 0:

        return 0

    return 1 + int(np.sum(seq[1:] != seq[:-1]))

def normalized_entropy(seq) -> float:

    seq = np.asarray(seq, dtype=int)

    seq = seq[seq >= 0]

    if len(seq) == 0:

        return 0.0

    vals, counts = np.unique(seq, return_counts=True)

    if len(vals) <= 1:

        return 0.0

    probs = counts.astype(np.float64) / counts.sum()

    ent = -np.sum(probs * np.log(probs + 1e-12))

    return float(ent / np.log(len(vals)))

def short_segment_ratio(seq, short_len: int) -> float:

    seq = np.asarray(seq, dtype=int)

    if len(seq) == 0:

        return 1.0

    short_points = 0

    for s, e, _ in split_segments(seq):

        seg_len = e - s + 1

        if seg_len < short_len:

            short_points += seg_len

    return float(short_points) / float(len(seq))

def dominant_ratio_from_labels(seq) -> float:

    seq = np.asarray(seq).astype(int)

    seq = seq[seq >= 0]

    if len(seq) == 0:

        return 0.0

    _, counts = np.unique(seq, return_counts=True)

    return float(counts.max() / counts.sum())

def normalized_entropy_from_labels(seq) -> float:

    return normalized_entropy(seq)

def transition_rate_from_labels(seq) -> float:

    seq = np.asarray(seq).astype(int)

    seq = seq[seq >= 0]

    if len(seq) <= 1:

        return 0.0

    return float(np.mean(seq[1:] != seq[:-1]))

def n_unique_norm_from_labels(seq, denom: float = 20.0) -> float:

    seq = np.asarray(seq).astype(int)

    seq = seq[seq >= 0]

    if len(seq) == 0:

        return 0.0

    return float(min(1.0, len(np.unique(seq)) / max(1.0, denom)))

def _safe_float(x, default=0.0) -> float:

    try:

        value = float(x)

        if math.isfinite(value):

            return value

    except Exception:

        pass

    return float(default)

def adjusted_rand_index(labels_true, labels_pred) -> float:

    try:

        return float(adjusted_rand_score(labels_true, labels_pred))

    except Exception:

        return 0.0

def normalized_mutual_information(labels_true, labels_pred) -> float:

    try:

        return float(normalized_mutual_info_score(labels_true, labels_pred, average_method="geometric"))

    except Exception:

        return 0.0

def pairwise_prediction_similarity(seq_a, seq_b) -> float:

    ari = adjusted_rand_index(seq_a, seq_b)

    nmi = normalized_mutual_information(seq_a, seq_b)

    return float(0.5 * max(0.0, ari) + 0.5 * max(0.0, nmi))

def align_sequence_to_length(seq, target_len):

    seq = np.asarray(seq).astype(int)

    if len(seq) == target_len:

        return seq

    if len(seq) <= 1:

        return np.zeros(target_len, dtype=int)

    src_x = np.linspace(0, 1, len(seq))

    dst_x = np.linspace(0, 1, target_len)

    mapped_idx = np.searchsorted(src_x, dst_x, side="left")

    mapped_idx = np.clip(mapped_idx, 0, len(seq) - 1)

    return seq[mapped_idx]

def find_raw_label_csv(case_dir: Path) -> Path | None:

    for name in RAW_LABEL_CANDIDATES:

        p = case_dir / name

        if p.exists():

            return p

    matches = sorted(case_dir.glob("multiscale_raw_label_matrix_Tx*.csv"))

    return matches[0] if matches else None

def find_case_dirs(root: Path, out_dir: Path) -> list[Path]:

    case_dirs = []

    for p in root.rglob(FINAL_META_CSV):

        case_dir = p.parent

        if out_dir in case_dir.parents or case_dir == out_dir:

            continue

        if (case_dir / RUN_INFO_CSV).exists() and find_raw_label_csv(case_dir) is not None:

            case_dirs.append(case_dir)

    return sorted(set(case_dirs))

def infer_video_id_from_case_dir(root: Path, case_dir: Path) -> str:

    try:

        return norm_video_id(str(case_dir.relative_to(root)).replace("\\", "/"))

    except Exception:

        return norm_video_id(case_dir.name)

def find_visual_csv(visual_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    parts = vid.split("/") if vid else [tail]

    candidates = []

    if len(parts) >= 1:

        candidates.append(visual_root / Path(*parts).with_suffix(".csv"))

        candidates.append(visual_root / Path(*parts[:-1]) / "teacher_visual_15d.csv")

        candidates.append(visual_root / Path(*parts) / "teacher_visual_15d.csv")

    candidates.append(visual_root / tail / f"{tail}.csv")

    candidates.append(visual_root / tail / "teacher_visual_15d.csv")

    candidates.append(visual_root / f"{tail}.csv")

    for p in candidates:

        if p.exists():

            return p

    if visual_root.exists():

        for p in visual_root.rglob("*.csv"):

            if p.stem == tail or p.name == "teacher_visual_15d.csv":

                try:

                    rel = str(p.parent.relative_to(visual_root)).replace("\\", "/")

                except Exception:

                    rel = str(p.parent).replace("\\", "/")

                if tail_video_id(rel) == tail or tail in rel.split("/"):

                    return p

    return None

                                                              

                              

                                                              

def quantize_orientation_3class(series):

    s = series.copy().clip(-2.0, 2.0)

    out = []

    for v in s:

        if pd.isna(v):

            out.append(np.nan)

        elif v <= -0.25:

            out.append(-1.0)

        elif v >= 0.25:

            out.append(1.0)

        else:

            out.append(0.0)

    return pd.Series(out, index=series.index)

def build_action_relative_features(visual_df: pd.DataFrame) -> pd.DataFrame:

    feature_cols = [

        "orientation_score",

        "left_shoulder_x", "left_shoulder_y",

        "left_elbow_x", "left_elbow_y",

        "left_wrist_x", "left_wrist_y",

        "right_shoulder_x", "right_shoulder_y",

        "right_elbow_x", "right_elbow_y",

        "right_wrist_x", "right_wrist_y",

        "center_x", "center_y",

    ]

    missing = [c for c in feature_cols if c not in visual_df.columns]

    if missing:

        raise ValueError(f"动作 CSV 缺少这些列：{missing}")

    X = visual_df[feature_cols].copy()

    cx = X["center_x"].astype(float)

    cy = X["center_y"].astype(float)

    ori3 = quantize_orientation_3class(X["orientation_score"].astype(float))

    out = pd.DataFrame(index=X.index)

    out["orientation_3class"] = ori3

    pts = [

        "left_shoulder", "left_elbow", "left_wrist",

        "right_shoulder", "right_elbow", "right_wrist",

    ]

    for p in pts:

        out[f"{p}_dx"] = X[f"{p}_x"].astype(float) - cx

        out[f"{p}_dy"] = X[f"{p}_y"].astype(float) - cy

    out = out.ffill().bfill().fillna(0.0)

    out = out.rolling(ACTION_SMOOTH, center=True, min_periods=1).median()

    return out

def align_x1_to_final_df(visual_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:

    action_X = build_action_relative_features(visual_df)

    if len(action_X) == len(final_df):

        return action_X.reset_index(drop=True)

    if TIME_COL not in visual_df.columns or TIME_COL not in final_df.columns:

        raise ValueError(

            f"visual_df 与 final_df 长度不同且缺少 {TIME_COL}: "

            f"visual={len(visual_df)}, final={len(final_df)}"

        )

    left = final_df[[TIME_COL]].copy()

    left["_row_id"] = np.arange(len(left))

    right = visual_df[[TIME_COL]].copy()

    right["_visual_idx"] = np.arange(len(right))

    merged = pd.merge_asof(

        left.sort_values(TIME_COL),

        right.sort_values(TIME_COL),

        on=TIME_COL,

        direction="nearest",

    ).sort_values("_row_id")

    idx = merged["_visual_idx"].ffill().bfill().astype(int).to_numpy()

    return action_X.iloc[idx].reset_index(drop=True)

def get_selected_run_cols(

    raw_df: pd.DataFrame,

    run_info_df: pd.DataFrame,

    branch_metrics_df: pd.DataFrame | None,

    max_selected_branches: int,

    use_all_branches: bool,

) -> list[str]:

    rows = []

    if not use_all_branches:

        if {"selected_for_meta", "col_name"}.issubset(run_info_df.columns):

            tmp = run_info_df.copy()

            tmp["_selected"] = pd.to_numeric(tmp["selected_for_meta"], errors="coerce").fillna(0).astype(int)

            tmp = tmp[tmp["_selected"].eq(1)].copy()

            if len(tmp):

                if "branch_rank" in tmp.columns:

                    tmp["_rank"] = pd.to_numeric(tmp["branch_rank"], errors="coerce").fillna(999999)

                else:

                    tmp["_rank"] = np.arange(len(tmp))

                rows = tmp.sort_values(["_rank", "col_name"]).to_dict("records")

        if not rows and branch_metrics_df is not None and "selected_for_meta" in branch_metrics_df.columns:

            bm = branch_metrics_df.copy()

            bm["_selected"] = pd.to_numeric(bm["selected_for_meta"], errors="coerce").fillna(0).astype(int)

            bm = bm[bm["_selected"].eq(1)].copy()

            if len(bm):

                if "branch_rank" in bm.columns:

                    bm["_rank"] = pd.to_numeric(bm["branch_rank"], errors="coerce").fillna(999999)

                else:

                    bm["_rank"] = np.arange(len(bm))

                if "col_name" in bm.columns:

                    rows = bm.sort_values(["_rank", "col_name"]).to_dict("records")

                elif "run_idx" in bm.columns and {"run_idx", "col_name"}.issubset(run_info_df.columns):

                    selected_run_idx = pd.to_numeric(bm.sort_values("_rank")["run_idx"], errors="coerce").dropna().astype(int).tolist()

                    ri = run_info_df.copy()

                    ri["_run_idx"] = pd.to_numeric(ri["run_idx"], errors="coerce").fillna(-999).astype(int)

                    ri["_order"] = ri["_run_idx"].map({r: i for i, r in enumerate(selected_run_idx)})

                    ri = ri[ri["_run_idx"].isin(selected_run_idx)].sort_values("_order")

                    rows = ri.to_dict("records")

    if not rows:

        if "col_name" in run_info_df.columns:

            ri = run_info_df.copy()

            if "branch_rank" in ri.columns:

                ri["_rank"] = pd.to_numeric(ri["branch_rank"], errors="coerce").fillna(999999)

            elif "run_idx" in ri.columns:

                ri["_rank"] = pd.to_numeric(ri["run_idx"], errors="coerce").fillna(999999)

            else:

                ri["_rank"] = np.arange(len(ri))

            rows = ri.sort_values(["_rank", "col_name"]).to_dict("records")

        else:

            rows = [{"col_name": c} for c in raw_df.columns if c.startswith("run") and c.endswith("_state")]

    cols = []

    for r in rows:

        c = str(r.get("col_name", ""))

        if c and c in raw_df.columns and c not in cols:

            cols.append(c)

    if max_selected_branches and max_selected_branches > 0:

        cols = cols[:int(max_selected_branches)]

    return cols

def compact_branch_signature_for_segment(raw_df: pd.DataFrame, run_cols: list[str], s: int, e: int) -> dict:

    out = {}

    doms, ents, trans, uniques, start_end_same = [], [], [], [], []

    for rank, col in enumerate(run_cols, start=1):

        seq = pd.to_numeric(raw_df.iloc[s:e + 1][col], errors="coerce").fillna(SKIP_STATE).astype(int).to_numpy()

        seq = seq[seq >= 0]

        dom = dominant_ratio_from_labels(seq)

        ent = normalized_entropy_from_labels(seq)

        tr = transition_rate_from_labels(seq)

        uq = n_unique_norm_from_labels(seq, denom=20.0)

        ses = 1.0 if len(seq) > 1 and int(seq[0]) == int(seq[-1]) else 0.0

        out[f"y1_rank{rank:02d}_dominant_ratio"] = dom

        out[f"y1_rank{rank:02d}_entropy"] = ent

        out[f"y1_rank{rank:02d}_transition_rate"] = tr

        out[f"y1_rank{rank:02d}_n_unique_norm"] = uq

        out[f"y1_rank{rank:02d}_start_end_same"] = ses

        doms.append(dom); ents.append(ent); trans.append(tr); uniques.append(uq); start_end_same.append(ses)

    def add_stats(prefix: str, vals):

        vals = np.asarray(vals, dtype=float)

        if len(vals) == 0:

            vals = np.zeros(1, dtype=float)

        out[f"{prefix}_mean"] = float(vals.mean())

        out[f"{prefix}_std"] = float(vals.std())

        out[f"{prefix}_min"] = float(vals.min())

        out[f"{prefix}_max"] = float(vals.max())

    add_stats("y1_branch_dom", doms)

    add_stats("y1_branch_entropy", ents)

    add_stats("y1_branch_transition_rate", trans)

    add_stats("y1_branch_n_unique_norm", uniques)

    add_stats("y1_branch_start_end_same", start_end_same)

    out["y1_selected_branch_count_norm"] = float(min(1.0, len(run_cols) / 16.0))

    return out

def build_case_segment_table(

    case_dir: Path,

    root: Path,

    visual_root: Path,

    min_layer1_seg_len: int,

    max_selected_branches: int,

    use_all_branches: bool,

    add_transition: bool,

) -> tuple[pd.DataFrame, dict]:

    final_csv = case_dir / FINAL_META_CSV

    raw_csv = find_raw_label_csv(case_dir)

    run_info_csv = case_dir / RUN_INFO_CSV

    branch_metrics_csv = case_dir / BRANCH_METRICS_CSV

    final_df = pd.read_csv(final_csv)

    raw_df = pd.read_csv(raw_csv)

    run_info_df = pd.read_csv(run_info_csv)

    branch_metrics_df = pd.read_csv(branch_metrics_csv) if branch_metrics_csv.exists() else None

    video_id = infer_video_id_from_case_dir(root, case_dir)

    visual_csv = find_visual_csv(visual_root, video_id)

    if visual_csv is None:

        raise FileNotFoundError(f"找不到 visual CSV: video_id={video_id}, visual_root={visual_root}")

    visual_df = pd.read_csv(visual_csv)

    X1 = align_x1_to_final_df(visual_df, final_df)

    X1.columns = [f"x1_{c}" for c in X1.columns]

    min_len = min(len(final_df), len(raw_df), len(X1))

    final_df = final_df.iloc[:min_len].reset_index(drop=True)

    raw_df = raw_df.iloc[:min_len].reset_index(drop=True)

    X1 = X1.iloc[:min_len].reset_index(drop=True)

    if "meta_state" not in final_df.columns:

        raise ValueError(f"{final_csv} 缺少 meta_state")

    meta_seq_full = pd.to_numeric(final_df["meta_state"], errors="coerce").fillna(SKIP_STATE).astype(int)

    if "is_teacher_frame" in final_df.columns:

        teacher_mask = pd.to_numeric(final_df["is_teacher_frame"], errors="coerce").fillna(1).astype(int).eq(1)

        teacher_mask = teacher_mask & meta_seq_full.ge(0)

    else:

        teacher_mask = meta_seq_full.ge(0)

    keep_idx = np.where(teacher_mask.to_numpy())[0]

    if len(keep_idx) == 0:

        raise ValueError("没有可用教师帧")

    final_t = final_df.iloc[keep_idx].reset_index(drop=True)

    raw_t = raw_df.iloc[keep_idx].reset_index(drop=True)

    X1_t = X1.iloc[keep_idx].reset_index(drop=True)

    meta_seq = meta_seq_full.iloc[keep_idx].to_numpy(dtype=int)

    meta_seq_smooth = merge_short_segments(meta_seq, min_len=int(min_layer1_seg_len))

    meta_seq_smooth, _ = remap_to_contiguous_labels(meta_seq_smooth)

    run_cols = get_selected_run_cols(

        raw_t,

        run_info_df,

        branch_metrics_df,

        max_selected_branches=int(max_selected_branches),

        use_all_branches=bool(use_all_branches),

    )

    if len(run_cols) == 0:

        raise ValueError("没有可用 run_state 列")

    segs = split_segments(meta_seq_smooth)

    rows = []

    x1_means_for_distance = []

    for seg_idx, (s, e, label) in enumerate(segs):

        sub_x = X1_t.iloc[s:e + 1].astype(float)

        row = {

            "video_id": video_id,

            "seg_idx": int(seg_idx),

            "frame_start_idx": int(s),

            "frame_end_idx": int(e),

            "duration_frames": int(e - s + 1),

            "log_duration_frames": float(math.log1p(max(0, e - s + 1))),

            "layer1_meta_state_local": int(label),

        }

        if TIME_COL in final_t.columns:

            t0 = float(final_t.iloc[s][TIME_COL])

            t1 = float(final_t.iloc[e][TIME_COL])

            row["start_sec"] = t0

            row["end_sec"] = t1

            row["duration_sec"] = max(0.0, t1 - t0)

        else:

            row["start_sec"] = np.nan

            row["end_sec"] = np.nan

            row["duration_sec"] = float(e - s + 1)

        mean_vals = sub_x.mean(axis=0)

        std_vals = sub_x.std(axis=0, ddof=0)

        for c in X1_t.columns:

            row[f"mean_{c}"] = float(mean_vals[c])

            row[f"std_{c}"] = float(std_vals[c])

        row.update(compact_branch_signature_for_segment(raw_t, run_cols, s, e))

        rows.append(row)

        x1_means_for_distance.append(mean_vals.to_numpy(dtype=float))

    seg_df = pd.DataFrame(rows)

    if add_transition and len(seg_df) > 0:

        x1_mean_arr = np.vstack(x1_means_for_distance)

        prev_dur = np.zeros(len(seg_df), dtype=float)

        next_dur = np.zeros(len(seg_df), dtype=float)

        prev_x1_dist = np.zeros(len(seg_df), dtype=float)

        next_x1_dist = np.zeros(len(seg_df), dtype=float)

        has_prev = np.zeros(len(seg_df), dtype=float)

        has_next = np.zeros(len(seg_df), dtype=float)

        for i in range(len(seg_df)):

            if i > 0:

                has_prev[i] = 1.0

                prev_dur[i] = float(math.log1p(float(seg_df.iloc[i - 1]["duration_frames"])))

                prev_x1_dist[i] = float(np.linalg.norm(x1_mean_arr[i] - x1_mean_arr[i - 1]))

            if i < len(seg_df) - 1:

                has_next[i] = 1.0

                next_dur[i] = float(math.log1p(float(seg_df.iloc[i + 1]["duration_frames"])))

                next_x1_dist[i] = float(np.linalg.norm(x1_mean_arr[i] - x1_mean_arr[i + 1]))

        seg_df["trans_has_prev"] = has_prev

        seg_df["trans_has_next"] = has_next

        seg_df["trans_prev_log_duration"] = prev_dur

        seg_df["trans_next_log_duration"] = next_dur

        seg_df["trans_prev_x1_mean_l2"] = prev_x1_dist

        seg_df["trans_next_x1_mean_l2"] = next_x1_dist

    info = {

        "video_id": video_id,

        "case_dir": str(case_dir),

        "visual_csv": str(visual_csv),

        "n_frames_used": int(len(final_t)),

        "n_segments": int(len(seg_df)),

        "selected_run_cols": run_cols,

        "selected_branch_count": int(len(run_cols)),

        "x1_dim": int(len(X1_t.columns)),

    }

    return seg_df, info

def insert_boundary_padding(segment_df: pd.DataFrame, feature_cols: list[str], boundary_pad: int):

    if boundary_pad <= 0:

        out = segment_df.copy()

        out["is_padding"] = 0

        return out, out["is_padding"].astype(bool).to_numpy()

    chunks = []

    unique_videos = list(dict.fromkeys(segment_df["video_id"].astype(str).tolist()))

    for vid_i, vid in enumerate(unique_videos):

        sub = segment_df[segment_df["video_id"].astype(str).eq(vid)].copy()

        sub["is_padding"] = 0

        chunks.append(sub)

        if vid_i < len(unique_videos) - 1:

            pad_rows = []

            for _ in range(boundary_pad):

                row = {c: 0.0 for c in feature_cols}

                row.update({

                    "video_id": f"__PAD_AFTER__{vid}",

                    "seg_idx": -1,

                    "frame_start_idx": -1,

                    "frame_end_idx": -1,

                    "start_sec": np.nan,

                    "end_sec": np.nan,

                    "duration_frames": 0,

                    "duration_sec": 0.0,

                    "log_duration_frames": 0.0,

                    "layer1_meta_state_local": -1,

                    "is_padding": 1,

                })

                pad_rows.append(row)

            chunks.append(pd.DataFrame(pad_rows))

    out = pd.concat(chunks, axis=0, ignore_index=True)

    return out, out["is_padding"].astype(int).eq(1).to_numpy(dtype=bool)

def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str], pca_dim: int, out_dir: Path):

    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler()

    X_std = scaler.fit_transform(X)

    if pca_dim and int(pca_dim) > 0 and X_std.shape[1] > int(pca_dim):

        dim = min(int(pca_dim), X_std.shape[0] - 1, X_std.shape[1])

        if dim >= 2:

            pca = PCA(n_components=dim, random_state=42)

            X_used = pca.fit_transform(X_std).astype(np.float32)

            info = {

                "used_pca": True,

                "pca_dim": int(dim),

                "original_dim": int(X_std.shape[1]),

                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),

            }

            with open(out_dir / "layer2_multit2s_pca_info.json", "w", encoding="utf-8") as f:

                json.dump(info, f, ensure_ascii=False, indent=2)

            return X_used, info

    info = {

        "used_pca": False,

        "pca_dim": 0,

        "original_dim": int(X_std.shape[1]),

        "explained_variance_ratio_sum": None,

    }

    with open(out_dir / "layer2_multit2s_pca_info.json", "w", encoding="utf-8") as f:

        json.dump(info, f, ensure_ascii=False, indent=2)

    return X_std.astype(np.float32), info

                                                              

                  

                                                              

def run_t2s_branch(data_2d, win_size, step, out_channels, branch_name, min_len):

    scaler = StandardScaler()

    data_std = scaler.fit_transform(data_2d.astype(np.float32))

    params = copy.deepcopy(params_LSE)

    params["in_channels"] = int(data_std.shape[1])

    params["win_size"] = int(win_size)

    params["compared_length"] = int(win_size)

    params["M"] = int(T2S_M)

    params["N"] = int(T2S_N)

    params["out_channels"] = int(out_channels)

    params["nb_steps"] = int(T2S_NB_STEPS)

    params["win_type"] = str(T2S_WIN_TYPE)

    params["cuda"] = bool(T2S_USE_CUDA)

    params["gpu"] = int(T2S_GPU)

    if T2S_KERNEL_SIZE is not None:

        params["kernel_size"] = int(T2S_KERNEL_SIZE)

    try:

        import inspect

        from Time2State import encoders

        sig = inspect.signature(encoders.CausalConv_LSE.__init__)

        allowed = {name for name, p in sig.parameters.items() if name != "self"}

        params_for_encoder = {k: v for k, v in params.items() if k in allowed}

    except Exception:

        params_for_encoder = params

    if data_std.shape[0] < int(win_size):

        raise ValueError(f"rows={data_std.shape[0]} < win_size={win_size}")

    n_windows = 1 + max(0, (data_std.shape[0] - int(win_size)) // int(step))

    if n_windows < 2:

        raise ValueError(

            f"Too few Time2State windows for {branch_name}: "

            f"rows={data_std.shape[0]}, win={win_size}, step={step}, windows={n_windows}"

        )

    t2s = Time2State(

        int(win_size),

        int(step),

        CausalConv_LSE_Adaper(params_for_encoder),

        DPGMM(None),

    )

    t2s.fit(data_std, int(win_size), int(step))

    state_seq = np.asarray(t2s.state_seq).astype(int)

    state_seq = align_sequence_to_length(state_seq, data_std.shape[0])

    state_seq, _ = remap_to_contiguous_labels(state_seq)

    if int(min_len) > 0:

        state_seq = merge_short_segments(state_seq, min_len=int(min_len))

        state_seq, _ = remap_to_contiguous_labels(state_seq)

    return {

        "data_std": data_std,

        "state_seq": state_seq,

    }

def branch_health_score(seq, median_segments: float | None = None) -> dict:

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

    entropy = normalized_entropy(seq)

    if dom <= 0.75:

        dom_score = 1.0

    elif dom >= 0.98:

        dom_score = 0.0

    else:

        dom_score = (0.98 - dom) / (0.98 - 0.75)

    short_len = max(2, int(round(0.005 * n)))

    short_ratio = short_segment_ratio(seq, short_len)

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

def compute_peer_reliability(branch_sequences) -> list[dict]:

    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]

    n = len(seqs)

    seg_counts = [count_segments(seq) for seq in seqs]

    median_segments = float(np.median(np.asarray(seg_counts, dtype=float))) if seg_counts else 0.0

    health_rows = [branch_health_score(seq, median_segments=median_segments) for seq in seqs]

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

    alpha = float(PEER_HEALTH_WEIGHT)

    beta = float(PEER_CONSENSUS_WEIGHT)

    denom_ab = max(1e-12, alpha + beta)

    alpha /= denom_ab

    beta /= denom_ab

    reliability = health * (alpha * health + beta * consensus)

    max_rel = float(reliability.max()) if len(reliability) else 0.0

    reliability_norm = reliability / max_rel if max_rel > 1e-12 else reliability

    rows = []

    for i in range(n):

        row = dict(health_rows[i])

        row["peer_consensus"] = float(consensus[i])

        row["peer_reliability"] = float(reliability[i])

        row["peer_reliability_norm"] = float(reliability_norm[i])

        rows.append(row)

    return rows

def compute_pid_reliability(branch_sequences, branch_metrics) -> list[dict]:

    n = len(branch_sequences)

    if n == 0:

        return []

    seqs = [np.asarray(item[3], dtype=int) for item in branch_sequences]

    wins = np.asarray([float(item[1]) for item in branch_sequences], dtype=float)

    if not branch_metrics or "health" not in branch_metrics[0] or "peer_consensus" not in branch_metrics[0]:

        peer_rows = compute_peer_reliability(branch_sequences)

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

        [branch_health_score(seq, median_segments=median_segments).get("short_segment_ratio", 1.0) for seq in seqs],

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

    kp = float(PID_KP)

    ki = float(PID_KI)

    kd = float(PID_KD)

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

    tau = max(1e-6, float(PID_SOFTMAX_TAU))

    z = raw_norm / tau

    z = z - float(np.max(z)) if len(z) else z

    expz = np.exp(z)

    weights = expz / max(1e-12, float(expz.sum()))

    weights_norm = weights / weights.max() if weights.max() > 1e-12 else weights

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

def branch_selection_score(metric_row: dict, metric: str) -> float:

    metric = str(metric).upper()

    if metric == "PEER":

        return _safe_float(metric_row.get("peer_reliability_norm", metric_row.get("peer_reliability", 0.0)))

    if metric == "PID":

        return _safe_float(metric_row.get("pid_score_norm", metric_row.get("pid_weight_norm", 0.0)))

    raise ValueError(f"Unknown branch selection metric: {metric}")

def select_top_k_branch_indices(branch_metrics: list[dict], top_k: int, metric: str):

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

    scored_sorted = sorted(scored, key=lambda x: (-x[1], -x[2], -x[3], x[4], x[0]))

    selected_indices = {idx for idx, *_ in scored_sorted[:int(top_k)]}

    rank_map = {idx: rank + 1 for rank, (idx, *_rest) in enumerate(scored_sorted)}

    ranked_rows = []

    for idx, row in enumerate(branch_metrics):

        new_row = dict(row)

        new_row["branch_rank"] = rank_map[idx]

        new_row["branch_selection_score"] = branch_selection_score(new_row, metric)

        new_row["selected_for_meta"] = 1 if idx in selected_indices else 0

        ranked_rows.append(new_row)

    return selected_indices, ranked_rows

                                                              

                                  

                                                              

def one_hot_from_state(state_seq, num_classes):

    arr = np.zeros((len(state_seq), num_classes), dtype=np.float32)

    arr[np.arange(len(state_seq)), np.asarray(state_seq).astype(int)] = 1.0

    return arr

def build_indicator_time_matrix(state_matrix_df, run_infos):

    mats = []

    state_rows = []

    for info in run_infos:

        col = info["col_name"]

        seq = state_matrix_df[col].values.astype(int)

        valid = seq[seq >= 0]

        if len(valid) == 0:

            continue

        k_num = int(valid.max()) + 1

        onehot = one_hot_from_state(seq.clip(min=0), k_num)

        for k in range(k_num):

            state_name = f"R{info['run_idx']}_W{info['win']}_S{info['step']}_state{k}"

            mats.append(onehot[:, k])

            state_rows.append({

                "global_state_name": state_name,

                "run_idx": int(info["run_idx"]),

                "win": int(info["win"]),

                "step": int(info["step"]),

                "local_state": int(k),

            })

    if not mats:

        raise RuntimeError("indicator matrix 为空")

    indicator_mat = np.stack(mats, axis=1).astype(np.float32)

    indicator_df = pd.DataFrame(indicator_mat, columns=[r["global_state_name"] for r in state_rows])

    state_info_df = pd.DataFrame(state_rows)

    return indicator_df, state_info_df

def attach_branch_weights_to_state_info(state_info_df, branch_metrics: list[dict], mode: str):

    out = state_info_df.copy()

    mode = str(mode or "pid_weight").lower()

    weight_by_run_idx = {}

    for row in branch_metrics:

        run_idx = int(row.get("run_idx", 0))

        if mode == "branch_reliability":

            w = _safe_float(row.get("peer_reliability_norm", row.get("peer_reliability", 0.0)), 0.0)

        elif mode == "pid_weight":

            w = _safe_float(row.get("pid_weight_norm", row.get("pid_score_norm", 0.0)), 0.0)

        else:

            w = 1.0

        weight_by_run_idx[run_idx] = max(0.0, float(w))

    if not weight_by_run_idx or max(weight_by_run_idx.values()) <= 1e-12:

        weight_by_run_idx = {int(row.get("run_idx", i + 1)): 1.0 for i, row in enumerate(branch_metrics)}

    out["branch_weight"] = [float(weight_by_run_idx.get(int(r), 1.0)) for r in out["run_idx"].astype(int).values]

    out["meta_vote_weight_mode"] = mode

    return out

def filter_active_indicator_states(indicator_df_full, indicator_df_valid, state_info_df, eps=1e-8):

    

       

    if indicator_df_full.shape[1] != len(state_info_df):

        raise ValueError(

            f"indicator 列数与 state_info 行数不一致："

            f"indicator_cols={indicator_df_full.shape[1]}, state_info_rows={len(state_info_df)}"

        )

    col_activity = indicator_df_valid.to_numpy(dtype=np.float32).sum(axis=0)

    active_mask = col_activity > float(eps)

    dropped_columns = indicator_df_full.columns[~active_mask].astype(str).tolist()

    if int(active_mask.sum()) < 2:

        raise ValueError(

            f"有效局部状态列少于 2 个，无法做 meta 聚合："

            f"active={int(active_mask.sum())}, total={len(active_mask)}, dropped={len(dropped_columns)}"

        )

    indicator_df_full_active = indicator_df_full.loc[:, active_mask].reset_index(drop=True)

    indicator_df_valid_active = indicator_df_valid.loc[:, active_mask].reset_index(drop=True)

    state_info_active = state_info_df.loc[active_mask].reset_index(drop=True).copy()

    return indicator_df_full_active, indicator_df_valid_active, state_info_active, dropped_columns

def cluster_state_matrix(indicator_df, state_info_df, n_clusters, smooth_win=9):

    H = indicator_df.to_numpy(dtype=np.float32)                

    G = H.T                                                    

    G_smooth = pd.DataFrame(G.T).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy().T

    norm = np.linalg.norm(G_smooth, axis=1, keepdims=True)

    zero_mask = (norm[:, 0] < 1e-8)

    if zero_mask.any():

                                                   

                                                      

        G_smooth[zero_mask, 0] = 1e-6

        norm = np.linalg.norm(G_smooth, axis=1, keepdims=True)

    norm[norm < 1e-8] = 1.0

    X = G_smooth / norm

    n_clusters = int(min(max(2, int(n_clusters)), max(2, X.shape[0] - 1)))

    try:

        clusterer = AgglomerativeClustering(

            n_clusters=n_clusters,

            metric="cosine",

            linkage="average",

        )

    except TypeError:

        clusterer = AgglomerativeClustering(

            n_clusters=n_clusters,

            affinity="cosine",

            linkage="average",

        )

    labels = clusterer.fit_predict(X).astype(int)

    labels, _ = remap_to_contiguous_labels(labels)

    out_info = state_info_df.copy()

    out_info["cluster_id"] = labels

    return out_info

def build_meta_ratio_and_sequence(indicator_df, state_info_df, meta_min_len=4):

    H = indicator_df.to_numpy(dtype=np.float32)

    cluster_labels = state_info_df["cluster_id"].to_numpy(dtype=int)

    C = int(cluster_labels.max()) + 1

    if "branch_weight" in state_info_df.columns:

        local_weights = state_info_df["branch_weight"].to_numpy(dtype=np.float32)

    else:

        local_weights = np.ones(len(cluster_labels), dtype=np.float32)

    meta_vote = np.zeros((H.shape[0], C), dtype=np.float32)

    for s_idx, c in enumerate(cluster_labels):

        meta_vote[:, c] += float(local_weights[s_idx]) * H[:, s_idx]

    denom = meta_vote.sum(axis=1, keepdims=True)

    denom[denom < 1e-8] = 1.0

    meta_ratio_raw = meta_vote / denom

    meta_state_seq_raw = np.argmax(meta_ratio_raw, axis=1).astype(int)

    meta_state_seq_smooth = merge_short_segments(meta_state_seq_raw, min_len=meta_min_len)

    meta_state_seq, state_map = remap_to_contiguous_labels(meta_state_seq_smooth)

    new_C = len(set(meta_state_seq))

    meta_ratio_new = np.zeros((H.shape[0], new_C), dtype=np.float32)

    for old_state, new_state in state_map.items():

        if old_state < meta_ratio_raw.shape[1]:

            meta_ratio_new[:, new_state] += meta_ratio_raw[:, old_state]

    denom2 = meta_ratio_new.sum(axis=1, keepdims=True)

    denom2[denom2 < 1e-8] = 1.0

    meta_ratio_new = meta_ratio_new / denom2

    meta_ratio_df = pd.DataFrame(

        meta_ratio_new,

        columns=[f"meta_ratio_{i}" for i in range(meta_ratio_new.shape[1])],

    )

    meta_seq_df = pd.DataFrame({"meta_state": meta_state_seq})

    return meta_ratio_df, meta_seq_df, state_map

def _mean_segment_length(seq):

    seq = np.asarray(seq).astype(int)

    n = len(seq)

    if n == 0:

        return 0.0

    seg_num = count_segments(seq)

    return float(n) / float(seg_num)

def _evaluate_one_k(K, indicator_df_valid, state_info_df, meta_min_len=4, smooth_win=9):

    state_info_cluster = cluster_state_matrix(

        indicator_df=indicator_df_valid,

        state_info_df=state_info_df,

        n_clusters=K,

        smooth_win=smooth_win,

    )

    _meta_ratio_df, meta_seq_df, _map = build_meta_ratio_and_sequence(

        indicator_df=indicator_df_valid,

        state_info_df=state_info_cluster,

        meta_min_len=meta_min_len,

    )

    seq = meta_seq_df["meta_state"].to_numpy(dtype=int)

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

    tau = max(0.05, 1.0 / float(K))

    dominant_count = int(np.sum(probs >= tau))

    dominant_ratio = float(dominant_count) / float(K)

    balance = normalized_entropy(seq)

    mean_seg_len = _mean_segment_length(seq)

    seg_num = count_segments(seq)

    return {

        "K": int(K),

        "dominant_threshold": float(tau),

        "dominant_count": int(dominant_count),

        "dominant_ratio": float(dominant_ratio),

        "balance_entropy": float(balance),

        "mean_segment_length": float(mean_seg_len),

        "segment_count": int(seg_num),

        "state_info_cluster": state_info_cluster,

    }

def choose_meta_cluster_k_via_sweep(run_infos, indicator_df_valid, state_info_df, meta_min_len=4, smooth_win=9, min_allowed=2, save_csv_path=None):

    if len(run_infos) == 0:

        raise ValueError("run_infos 为空，无法自动确定 meta 聚类数")

    state_counts = [int(info["n_states"]) for info in run_infos if int(info.get("n_states", 0)) > 0]

    if not state_counts:

        raise ValueError("state_counts 为空")

    k_low = max(min_allowed, min(state_counts))

    k_high = max(state_counts)

    k_high = min(k_high, max(2, len(state_info_df) - 1))

    candidate_Ks = list(range(k_low, k_high + 1))

    if not candidate_Ks:

        candidate_Ks = [2]

    raw_results = []

    for K in candidate_Ks:

        result = _evaluate_one_k(

            K=K,

            indicator_df_valid=indicator_df_valid,

            state_info_df=state_info_df,

            meta_min_len=meta_min_len,

            smooth_win=smooth_win,

        )

        raw_results.append(result)

    mean_seg_vals = np.array([r["mean_segment_length"] for r in raw_results], dtype=np.float64)

    seg_min = float(mean_seg_vals.min())

    seg_max = float(mean_seg_vals.max())

    rows = []

    for r in raw_results:

        if seg_max - seg_min < 1e-12:

            coherence = 1.0

        else:

            coherence = float((r["mean_segment_length"] - seg_min) / (seg_max - seg_min))

        score = (

            0.60 * r["dominant_ratio"]

            + 0.25 * r["balance_entropy"]

            + 0.15 * coherence

        )

        rows.append({

            "K": r["K"],

            "dominant_threshold": r["dominant_threshold"],

            "dominant_count": r["dominant_count"],

            "dominant_ratio": r["dominant_ratio"],

            "balance_entropy": r["balance_entropy"],

            "mean_segment_length": r["mean_segment_length"],

            "segment_count": r["segment_count"],

            "coherence": coherence,

            "selection_score": score,

        })

    score_df = pd.DataFrame(rows).sort_values(

        by=["selection_score", "K", "mean_segment_length"],

        ascending=[False, True, False],

    ).reset_index(drop=True)

    best_K = int(score_df.iloc[0]["K"])

    if save_csv_path is not None:

        score_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    best_raw = None

    for r in raw_results:

        if r["K"] == best_K:

            best_raw = r

            break

    return best_K, score_df, best_raw

                                                              

        

                                                              

def parse_int_list(s: str) -> list[int]:

    return [int(x.strip()) for x in str(s).split(",") if str(x).strip()]

def build_steps_from_wins(wins: list[int], step_mode: str) -> list[int]:

    if str(step_mode).lower() == "auto":

                              

                                                               

        return [max(2, int(w) // 4) for w in wins]

    vals = parse_int_list(step_mode)

    if len(vals) != len(wins):

        raise ValueError(f"--branch-steps 数量必须等于 --branch-wins 数量: {len(vals)} vs {len(wins)}")

    return vals

def get_branch_min_len(win: int) -> int:

                                    

    return max(2, int(win) // 2)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--t2s-root", type=str, default=str(T2S_OUTPUT_ROOT_DEFAULT))

    ap.add_argument("--visual-root", type=str, default=str(VISUAL_CSV_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default="")

    ap.add_argument("--branch-wins", type=str, default="32,28,24,20,16,12,10,8")

    ap.add_argument("--branch-steps", type=str, default="auto")

    ap.add_argument("--select-top-k", type=int, default=4)

    ap.add_argument("--branch-select-metric", type=str, default="PID", choices=["PID", "PEER"])

    ap.add_argument("--meta-vote-weight-mode", type=str, default="pid_weight", choices=["pid_weight", "branch_reliability", "uniform"])

    ap.add_argument("--meta-min-len", type=int, default=4)

    ap.add_argument("--state-cluster-smooth", type=int, default=9)

    ap.add_argument("--boundary-pad", type=int, default=16)

    ap.add_argument("--pca-dim", type=int, default=0)

    ap.add_argument("--min-layer1-seg-len", type=int, default=6)

    ap.add_argument("--max-selected-branches", type=int, default=16)

    ap.add_argument("--use-all-branches", action="store_true")

    ap.add_argument("--no-transition", action="store_true")

    ap.add_argument("--max-cases", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.t2s_root)

    visual_root = Path(args.visual_root)

    out_dir = Path(args.out_dir) if args.out_dir else root / OUT_DIRNAME_DEFAULT

    ensure_dir(out_dir)

    branch_wins = parse_int_list(args.branch_wins)

    branch_steps = build_steps_from_wins(branch_wins, args.branch_steps)

    print("=" * 100)

    print("Cross-video Layer-2: segment X1+compactY1 + 8-branch multi-T2S PID Top-4 meta")

    print(f"t2s_root              = {root}")

    print(f"visual_root           = {visual_root}")

    print(f"out_dir               = {out_dir}")

    print(f"branch_wins           = {branch_wins}")

    print(f"branch_steps          = {branch_steps}")

    print(f"select_top_k          = {args.select_top_k}")

    print(f"branch_select_metric  = {args.branch_select_metric}")

    print(f"meta_vote_weight_mode = {args.meta_vote_weight_mode}")

    print(f"boundary_pad          = {args.boundary_pad}")

    print(f"pca_dim               = {args.pca_dim}")

    print("说明：第二层使用 8 个段级 T2S 分支，并用 PID Top-4 + meta 聚合。")

    print("=" * 100)

    case_dirs = find_case_dirs(root, out_dir)

    if args.max_cases and args.max_cases > 0:

        case_dirs = case_dirs[:int(args.max_cases)]

    if not case_dirs:

        raise FileNotFoundError(f"没有找到第一层输出 case: {root}")

    manifest_rows = []

    seg_tables = []

    selected_branch_counts = []

    for i, case_dir in enumerate(case_dirs, start=1):

        video_id = infer_video_id_from_case_dir(root, case_dir)

        print(f"[{i}/{len(case_dirs)}] {case_dir}")

        row = {

            "video_id": video_id,

            "case_dir": str(case_dir),

            "included": 0,

            "skip_reason": "",

            "n_frames_used": 0,

            "n_segments": 0,

            "selected_branch_count": 0,

            "x1_dim": 0,

        }

        try:

            seg_df, info = build_case_segment_table(

                case_dir=case_dir,

                root=root,

                visual_root=visual_root,

                min_layer1_seg_len=int(args.min_layer1_seg_len),

                max_selected_branches=int(args.max_selected_branches),

                use_all_branches=bool(args.use_all_branches),

                add_transition=not bool(args.no_transition),

            )

            if len(seg_df) < 3:

                raise ValueError(f"segment 太少: {len(seg_df)}")

            seg_tables.append(seg_df)

            selected_branch_counts.append(info["selected_branch_count"])

            row.update({

                "included": 1,

                "n_frames_used": info["n_frames_used"],

                "n_segments": info["n_segments"],

                "selected_branch_count": info["selected_branch_count"],

                "x1_dim": info["x1_dim"],

                "visual_csv": info["visual_csv"],

                "selected_run_cols": ",".join(info["selected_run_cols"]),

            })

            print(f"  [OK] frames={row['n_frames_used']}, segments={row['n_segments']}, selected_branches={row['selected_branch_count']}, X1={row['x1_dim']}")

        except Exception as e:

            row["skip_reason"] = repr(e)

            print(f"  [SKIP] {repr(e)}")

        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_path = out_dir / "layer2_multit2s_X1compactY1_build_manifest.csv"

    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[OK] manifest: {manifest_path}")

    if not seg_tables:

        raise RuntimeError("没有可用 segment 表，无法运行第二层 T2S。")

    segment_df = pd.concat(seg_tables, axis=0, ignore_index=True)

    segment_df = segment_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta_cols = {

        "video_id", "seg_idx", "frame_start_idx", "frame_end_idx",

        "start_sec", "end_sec", "layer1_meta_state_local",

    }

    feature_cols = []

    for c in segment_df.columns:

        if c in meta_cols:

            continue

        if pd.api.types.is_numeric_dtype(segment_df[c]):

            feature_cols.append(c)

    segment_df[feature_cols] = segment_df[feature_cols].astype(float).fillna(0.0)

    no_pad_path = out_dir / "layer2_segment_input_X1compactY1_no_padding.csv"

    segment_df.to_csv(no_pad_path, index=False, encoding="utf-8-sig")

    print(f"[OK] segment input no padding: {no_pad_path}")

    print(f"第二层 segment 输入: rows={len(segment_df)}, dim={len(feature_cols)}")

    with open(out_dir / "layer2_multit2s_feature_columns.json", "w", encoding="utf-8") as f:

        json.dump({

            "feature_cols": feature_cols,

            "n_features": len(feature_cols),

            "mean_selected_branch_count": float(np.mean(selected_branch_counts)) if selected_branch_counts else 0.0,

            "branch_wins": branch_wins,

            "branch_steps": branch_steps,

            "note": (

                "X1 is reconstructed from the exact layer-1 action input. "

                "Y1 is compact branch-signature statistics. "

                "Layer-2 uses 8 segment-scale T2S branches + PID Top-4 + meta aggregation."

            ),

        }, f, ensure_ascii=False, indent=2)

             

    padded_df, is_padding = insert_boundary_padding(segment_df, feature_cols, int(args.boundary_pad))

    padded_df[feature_cols] = padded_df[feature_cols].astype(float).fillna(0.0)

    padded_path = out_dir / "layer2_segment_input_X1compactY1_with_padding.csv"

    padded_df.to_csv(padded_path, index=False, encoding="utf-8-sig")

    print(f"[OK] segment input with padding: {padded_path}")

    print(f"加入边界 padding 后: rows={len(padded_df)}, padding={int(is_padding.sum())}")

          

    X_used, pca_info = prepare_feature_matrix(padded_df, feature_cols, int(args.pca_dim), out_dir)

    print(f"二层 multi-T2S 实际输入矩阵: rows={X_used.shape[0]}, dim={X_used.shape[1]}, pca={pca_info}")

             

    state_matrix_dict = {}

    run_infos = []

    candidate_branch_sequences = []

    branch_metrics = []

    valid_mask = ~is_padding

    for i, (win, step) in enumerate(zip(branch_wins, branch_steps), start=1):

        run_name = f"l2_run{i}_w{win}_s{step}"

        print(f"\n===== Layer-2 branch {i}/{len(branch_wins)}: win={win}, step={step} =====")

        start_time = time.time()

        result = run_t2s_branch(

            X_used,

            win_size=int(win),

            step=int(step),

            out_channels=T2S_OUT_CHANNELS,

            branch_name=run_name,

            min_len=get_branch_min_len(int(win)),

        )

        elapsed = time.time() - start_time

        seq_full = align_sequence_to_length(result["state_seq"], X_used.shape[0])

        seq_full, _ = remap_to_contiguous_labels(seq_full)

        col_name = f"l2_run{i}_state"

        state_matrix_dict[col_name] = seq_full

        seq_valid = seq_full[valid_mask]

        seq_valid, _ = remap_to_contiguous_labels(seq_valid)

        run_infos.append({

            "run_idx": i,

            "win": int(win),

            "step": int(step),

            "min_len": int(get_branch_min_len(int(win))),

            "col_name": col_name,

            "n_states": int(seq_valid.max()) + 1 if len(seq_valid) else 0,

        })

        candidate_branch_sequences.append((run_name, int(win), int(step), seq_valid.copy()))

        branch_metrics.append({

            "branch": run_name,

            "run_idx": i,

            "win": int(win),

            "step": int(step),

            "m": int(T2S_M),

            "n": int(T2S_N),

            "out_channels": int(T2S_OUT_CHANNELS),

            "nb_steps": int(T2S_NB_STEPS),

            "win_type": str(T2S_WIN_TYPE),

            "branch_min_len": int(get_branch_min_len(int(win))),

            "meta_min_len": int(args.meta_min_len),

            "states": int(seq_valid.max()) + 1 if len(seq_valid) else 0,

            "segments": int(count_segments(seq_valid)),

            "seconds": float(elapsed),

            "col_name": col_name,

        })

        print(f"  OK states={branch_metrics[-1]['states']} segments={branch_metrics[-1]['segments']} seconds={elapsed:.1f}")

           

    peer_rows = compute_peer_reliability(candidate_branch_sequences)

    for row, peer_row in zip(branch_metrics, peer_rows):

        row.update(peer_row)

    pid_rows = compute_pid_reliability(candidate_branch_sequences, branch_metrics)

    for row, pid_row in zip(branch_metrics, pid_rows):

        row.update(pid_row)

    selected_indices, ranked_branch_metrics = select_top_k_branch_indices(

        branch_metrics,

        int(args.select_top_k),

        str(args.branch_select_metric),

    )

    branch_metrics = ranked_branch_metrics

    branch_metric_by_run_idx = {int(row["run_idx"]): row for row in branch_metrics}

    for info in run_infos:

        mrow = branch_metric_by_run_idx.get(int(info["run_idx"]), {})

        info["selected_for_meta"] = int(mrow.get("selected_for_meta", 0))

        info["branch_rank"] = int(mrow.get("branch_rank", 0))

        info["branch_selection_score"] = float(mrow.get("branch_selection_score", 0.0))

    selected_run_infos = [info for idx, info in enumerate(run_infos) if idx in selected_indices]

    selected_branch_metrics = [row for idx, row in enumerate(branch_metrics) if idx in selected_indices]

    metrics_path = out_dir / "layer2_multit2s_branch_metrics_pid_peer.csv"

    pd.DataFrame(branch_metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"[OK] layer2 branch metrics: {metrics_path}")

    print(

        f"已选择 top {len(selected_run_infos)}/{len(run_infos)} 个二层分支用于 meta 聚合："

        f"metric={args.branch_select_metric}, vote={args.meta_vote_weight_mode}"

    )

    print(", ".join([row["branch"] for row in selected_branch_metrics]))

                     

    state_matrix_df = padded_df[[

        "video_id", "seg_idx", "frame_start_idx", "frame_end_idx",

        "start_sec", "end_sec", "duration_frames", "duration_sec",

        "layer1_meta_state_local", "is_padding",

    ]].copy()

    for col, seq in state_matrix_dict.items():

        state_matrix_df[col] = seq

    state_matrix_path = out_dir / "layer2_multit2s_state_matrix_with_padding.csv"

    state_matrix_df.to_csv(state_matrix_path, index=False, encoding="utf-8-sig")

    print(f"[OK] layer2 state matrix: {state_matrix_path}")

                                 

    selected_state_matrix = state_matrix_df[["is_padding"] + [info["col_name"] for info in selected_run_infos]].copy()

    indicator_df, state_info_df = build_indicator_time_matrix(selected_state_matrix.drop(columns=["is_padding"]), selected_run_infos)

    state_info_df = attach_branch_weights_to_state_info(

        state_info_df,

        selected_branch_metrics,

        mode=str(args.meta_vote_weight_mode),

    )

    indicator_valid = indicator_df.loc[valid_mask].reset_index(drop=True)

           

                                           

                                                  

                                           

    indicator_df, indicator_valid, state_info_df, dropped_indicator_cols = filter_active_indicator_states(

        indicator_df_full=indicator_df,

        indicator_df_valid=indicator_valid,

        state_info_df=state_info_df,

        eps=1e-8,

    )

    print(

        f"meta 聚合前过滤全零局部状态列："

        f"dropped={len(dropped_indicator_cols)}, active={indicator_df.shape[1]}"

    )

    if dropped_indicator_cols:

        pd.DataFrame({"dropped_indicator_col": dropped_indicator_cols}).to_csv(

            out_dir / "layer2_multit2s_dropped_padding_only_indicator_cols.csv",

            index=False,

            encoding="utf-8-sig",

        )

    k_sweep_path = out_dir / "layer2_multit2s_meta_k_sweep_scores.csv"

    best_K, score_df, best_raw = choose_meta_cluster_k_via_sweep(

        run_infos=selected_run_infos,

        indicator_df_valid=indicator_valid,

        state_info_df=state_info_df,

        meta_min_len=int(args.meta_min_len),

        smooth_win=int(args.state_cluster_smooth),

        min_allowed=2,

        save_csv_path=k_sweep_path,

    )

    print(f"[OK] meta K sweep: {k_sweep_path}")

    print(f"Layer-2 meta best_K = {best_K}")

                                                              

    state_info_with_cluster = best_raw["state_info_cluster"].copy()

    state_info_path = out_dir / "layer2_multit2s_state_info_with_cluster.csv"

    state_info_with_cluster.to_csv(state_info_path, index=False, encoding="utf-8-sig")

    meta_ratio_df, meta_seq_df, _ = build_meta_ratio_and_sequence(

        indicator_df=indicator_df,

        state_info_df=state_info_with_cluster,

        meta_min_len=int(args.meta_min_len),

    )

    padded_out = state_matrix_df.copy()

    padded_out["layer2_meta_state_raw"] = meta_seq_df["meta_state"].to_numpy(dtype=int)

    for c in meta_ratio_df.columns:

        padded_out[c] = meta_ratio_df[c].to_numpy(dtype=float)

    padded_out_path = out_dir / "layer2_multit2s_meta_states_with_padding.csv"

    padded_out.to_csv(padded_out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] layer2 meta with padding: {padded_out_path}")

                      

    valid_out = padded_out.loc[~is_padding].copy().reset_index(drop=True)

    valid_states, _ = remap_to_contiguous_labels(valid_out["layer2_meta_state_raw"].astype(int).to_numpy())

    valid_out["layer2_state"] = valid_states

    out_state_path = out_dir / "layer2_cross_video_segment_states_X1compactY1_multit2s_pid.csv"

    valid_out.to_csv(out_state_path, index=False, encoding="utf-8-sig")

    print(f"[OK] final layer2 states: {out_state_path}")

        

    summary_rows = []

    total_segments = len(valid_out)

    for state, g in valid_out.groupby("layer2_state"):

        vc_video = g["video_id"].astype(str).value_counts()

        vc_l1 = g["layer1_meta_state_local"].astype(int).value_counts()

        summary_rows.append({

            "layer2_state": int(state),

            "n_segments": int(len(g)),

            "segment_ratio": float(len(g) / max(1, total_segments)),

            "support_videos": int(vc_video.size),

            "top_video": str(vc_video.index[0]) if len(vc_video) else "",

            "top_video_segments": int(vc_video.iloc[0]) if len(vc_video) else 0,

            "top_video_ratio_within_state": float(vc_video.iloc[0] / max(1, len(g))) if len(vc_video) else 0.0,

            "dominant_layer1_meta_state_local": int(vc_l1.index[0]) if len(vc_l1) else -1,

            "dominant_layer1_meta_state_segments": int(vc_l1.iloc[0]) if len(vc_l1) else 0,

            "mean_duration_frames": float(pd.to_numeric(g["duration_frames"], errors="coerce").mean()),

            "mean_duration_sec": float(pd.to_numeric(g["duration_sec"], errors="coerce").mean()) if "duration_sec" in g.columns else np.nan,

        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["support_videos", "n_segments"], ascending=[False, False])

    summary_path = out_dir / "layer2_state_summary_X1compactY1_multit2s_pid.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[OK] layer2 summary: {summary_path}")

    slim_cols = [

        "video_id", "seg_idx", "start_sec", "end_sec", "duration_frames", "duration_sec",

        "layer1_meta_state_local", "layer2_state",

    ]

    slim_cols = [c for c in slim_cols if c in valid_out.columns]

    slim_path = out_dir / "per_video_layer2_segment_sequence_slim_multit2s_pid.csv"

    valid_out[slim_cols].to_csv(slim_path, index=False, encoding="utf-8-sig")

    print(f"[OK] slim sequence: {slim_path}")

    print("\nDONE.")

if __name__ == "__main__":

    main()
