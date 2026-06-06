import os

import copy

import cv2

import pandas as pd

import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from moviepy import VideoFileClip

from TSpy.view import plot_mts

from Time2State.time2state import Time2State

from Time2State.adapers import CausalConv_LSE_Adaper

from Time2State.clustering import DPGMM

from Time2State.default_params import *

                           

          

                           

VIDEO_ROOT = r"D:\code\teacherT2S\yolo\input"

VISUAL_CSV_ROOT = r"D:\code\teacherT2S\yolo\pose_csv"

AUDIO_CSV_ROOT = r"D:\code\teacherT2S\yolo\pose_audio_csv"

                           

          

                           

OUTPUT_ROOT = r"D:\code\teacherT2S\multiscale_t2s_output_event_batch"

         

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

TIME_COL = "time_sec"

USE_NORMALIZED_COORDS = True

                           

               

                           

                      

MULTI_ACTION_WINS = [72 - 2 * i for i in range(32)]

MULTI_ACTION_STEPS = [max(8, w // 4) for w in MULTI_ACTION_WINS]

ACTION_OUT_CHANNELS = 8

def get_action_min_len(win):

    return max(18, win // 2)

                          

STATE_CLUSTER_SMOOTH = 9

META_MIN_LEN = 24

    

AUDIO_SMOOTH_MED = 11

CENTER_SMOOTH_MED = 21

CENTER_SMOOTH_MEAN = 21

ACTION_SMOOTH = 7

        

AUDIO_ENTER_VOICED = 0.32

AUDIO_EXIT_VOICED = 0.12

AUDIO_ENTER_SILENCE = 0.48

AUDIO_EXIT_SILENCE = 0.78

                

AUDIO_MIN_SEG = 15

        

MOVE_DX_LAG = 18

MOVE_MIN_SEG = 18

MOVE_ABS_MIN_TH = 0.003

                           

          

                           

COLOR_CENTER = (0, 255, 255)

COLOR_LEFT_ARM = (255, 0, 0)

COLOR_RIGHT_ARM = (0, 0, 255)

COLOR_ORI = (0, 180, 0)

COLOR_TEXT = (30, 30, 30)

COLOR_BOX = (255, 255, 255)

STATE_PALETTE = [

    (255, 0, 0),

    (0, 255, 0),

    (0, 0, 255),

    (255, 255, 0),

    (255, 0, 255),

    (0, 255, 255),

    (128, 0, 255),

    (255, 128, 0),

    (180, 120, 40),

    (40, 180, 120),

]

TIMELINE_MARGIN = 12

TIMELINE_BG_ALPHA = 0.35

                           

         

                           

def remap_to_contiguous_labels(state_seq):

    unique_states_raw = sorted(np.unique(state_seq))

    state_map = {old: new for new, old in enumerate(unique_states_raw)}

    remapped = np.array([state_map[s] for s in state_seq], dtype=int)

    return remapped, state_map

def merge_short_segments(state_seq, min_len=24):

    seq = np.asarray(state_seq).copy()

    n = len(seq)

    if n == 0:

        return seq

    changed = True

    while changed:

        changed = False

        segments = []

        start = 0

        for i in range(1, n):

            if seq[i] != seq[start]:

                segments.append((start, i - 1, seq[start]))

                start = i

        segments.append((start, n - 1, seq[start]))

        for i, (s, e, label) in enumerate(segments):

            seg_len = e - s + 1

            if seg_len >= min_len:

                continue

            left_label = segments[i - 1][2] if i > 0 else None

            right_label = segments[i + 1][2] if i < len(segments) - 1 else None

            if left_label is None and right_label is None:

                continue

            elif left_label is None:

                seq[s:e + 1] = right_label

            elif right_label is None:

                seq[s:e + 1] = left_label

            else:

                left_len = segments[i - 1][1] - segments[i - 1][0] + 1

                right_len = segments[i + 1][1] - segments[i + 1][0] + 1

                seq[s:e + 1] = left_label if left_len >= right_len else right_label

            changed = True

            break

    return seq

def one_hot_from_state(state_seq, num_classes):

    arr = np.zeros((len(state_seq), num_classes), dtype=np.float32)

    arr[np.arange(len(state_seq)), np.asarray(state_seq).astype(int)] = 1.0

    return arr

def build_timeline_strip(width, state_seq, palette, bar_h):

    strip = np.ones((bar_h, width, 3), dtype=np.uint8) * 255

    n = len(state_seq)

    if n == 0:

        return strip

    start = 0

    for i in range(1, n + 1):

        if i == n or state_seq[i] != state_seq[start]:

            s = start

            e = i - 1

            state = int(state_seq[s])

            x1 = int(round(s / n * width))

            x2 = int(round((e + 1) / n * width)) - 1

            x2 = max(x1, x2)

            color = palette[state % len(palette)]

            cv2.rectangle(strip, (x1, 0), (x2, bar_h - 1), color, -1)

            start = i

    cv2.rectangle(strip, (0, 0), (width - 1, bar_h - 1), (0, 0, 0), 1)

    return strip

def to_pixel_x(x, width):

    if pd.isna(x) or x == "":

        return None

    x = float(x)

    return int(round(x * width)) if USE_NORMALIZED_COORDS else int(round(x))

def to_pixel_y(y, height):

    if pd.isna(y) or y == "":

        return None

    y = float(y)

    return int(round(y * height)) if USE_NORMALIZED_COORDS else int(round(y))

def draw_point(frame, pt, color, radius=5):

    if pt is None:

        return

    x, y = pt

    if x is None or y is None:

        return

    cv2.circle(frame, (x, y), radius, color, -1)

def draw_line(frame, p1, p2, color, thickness=2):

    if p1 is None or p2 is None:

        return

    x1, y1 = p1

    x2, y2 = p2

    if None in [x1, y1, x2, y2]:

        return

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def get_point(row, prefix, width, height):

    x_col = f"{prefix}_x"

    y_col = f"{prefix}_y"

    if x_col not in row.index or y_col not in row.index:

        return None

    x = to_pixel_x(row[x_col], width)

    y = to_pixel_y(row[y_col], height)

    if x is None or y is None:

        return None

    return (x, y)

def draw_orientation_arrow(frame, center, ori_score):

    if center is None or pd.isna(ori_score) or ori_score == "":

        return

    cx, cy = center

    if cx is None or cy is None:

        return

    score = float(ori_score)

    score = max(min(score, 2.0), -2.0)

    length = int(40 + 30 * abs(score))

    end_pt = (cx + length, cy - 20) if score >= 0 else (cx - length, cy - 20)

    cv2.arrowedLine(frame, (cx, cy - 20), end_pt, COLOR_ORI, 3, tipLength=0.25)

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

def _count_segments(seq):

    seq = np.asarray(seq).astype(int)

    if len(seq) == 0:

        return 0

    return 1 + int(np.sum(seq[1:] != seq[:-1]))

def _mean_segment_length(seq):

    seq = np.asarray(seq).astype(int)

    n = len(seq)

    if n == 0:

        return 0.0

    seg_num = _count_segments(seq)

    return float(n) / float(seg_num)

def _normalized_entropy_from_seq(seq):

    seq = np.asarray(seq).astype(int)

    if len(seq) == 0:

        return 0.0

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

    if len(probs) <= 1:

        return 0.0

    ent = -np.sum(probs * np.log(probs + 1e-12))

    ent_max = np.log(len(probs))

    if ent_max < 1e-12:

        return 0.0

    return float(ent / ent_max)

def _evaluate_one_k(

    K,

    indicator_df,

    state_info_df,

    meta_min_len=24,

    smooth_win=9

):

    

       

    state_matrix_sxt_df, state_info_with_cluster_df = cluster_state_matrix(

        indicator_df=indicator_df,

        state_info_df=state_info_df,

        n_clusters=K,

        smooth_win=smooth_win

    )

    meta_ratio_df, meta_seq_df, meta_state_map = build_meta_ratio_and_sequence(

        indicator_df=indicator_df,

        state_info_df=state_info_with_cluster_df,

        meta_min_len=META_MIN_LEN

    )

    seq = meta_seq_df["meta_state"].to_numpy(dtype=int)

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

                             

    tau = max(0.05, 1.0 / float(K))

    dominant_count = int(np.sum(probs >= tau))

    dominant_ratio = float(dominant_count) / float(K)

    balance = _normalized_entropy_from_seq(seq)

    mean_seg_len = _mean_segment_length(seq)

    seg_num = _count_segments(seq)

    return {

        "K": int(K),

        "dominant_threshold": float(tau),

        "dominant_count": int(dominant_count),

        "dominant_ratio": float(dominant_ratio),

        "balance_entropy": float(balance),

        "mean_segment_length": float(mean_seg_len),

        "segment_count": int(seg_num),

        "state_matrix_sxt_df": state_matrix_sxt_df,

        "state_info_with_cluster_df": state_info_with_cluster_df,

        "meta_ratio_df": meta_ratio_df,

        "meta_seq_df": meta_seq_df,

    }

def choose_meta_cluster_k_via_sweep(

    run_infos,

    indicator_df,

    state_info_df,

    meta_min_len=24,

    smooth_win=9,

    min_allowed=2,

    save_csv_path=None

):

    

       

    if len(run_infos) == 0:

        raise ValueError("run_infos 为空，无法自动确定 meta 聚类数")

    state_counts = [int(info["n_states"]) for info in run_infos]

    k_low = max(min_allowed, min(state_counts))

    k_high = max(state_counts)

    candidate_Ks = list(range(k_low, k_high + 1))

    raw_results = []

    for K in candidate_Ks:

        result = _evaluate_one_k(

            K=K,

            indicator_df=indicator_df,

            state_info_df=state_info_df,

            meta_min_len=meta_min_len,

            smooth_win=smooth_win

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

            0.60 * r["dominant_ratio"] +

            0.25 * r["balance_entropy"] +

            0.15 * coherence

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

    score_df = pd.DataFrame(rows)

                                  

    score_df = score_df.sort_values(

        by=["selection_score", "K", "mean_segment_length"],

        ascending=[False, True, False]

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

                           

             

                           

def build_audio_binary_state(audio_df):

    req = ["audio_rms", "audio_voiced_ratio", "audio_silence_ratio"]

    missing = [c for c in req if c not in audio_df.columns]

    if missing:

        raise ValueError(f"音频 CSV 缺少这些列：{missing}")

    x = audio_df[req].copy()

    x = x.ffill().bfill().fillna(0.0)

    x = x.rolling(AUDIO_SMOOTH_MED, center=True, min_periods=1).median()

    rms = x["audio_rms"].values.astype(np.float32)

    voiced = x["audio_voiced_ratio"].values.astype(np.float32)

    silence = x["audio_silence_ratio"].values.astype(np.float32)

    enter_rms = max(0.006, float(np.quantile(rms, 0.58)))

    exit_rms = max(0.004, float(np.quantile(rms, 0.38)))

    state = np.zeros(len(rms), dtype=np.int32)

    cur = 0

    for i in range(len(rms)):

        if cur == 0:

            if (

                (voiced[i] >= AUDIO_ENTER_VOICED and silence[i] <= AUDIO_ENTER_SILENCE and rms[i] >= enter_rms)

                or (voiced[i] >= 0.45 and rms[i] >= exit_rms)

            ):

                cur = 1

        else:

            if (voiced[i] <= AUDIO_EXIT_VOICED and silence[i] >= AUDIO_EXIT_SILENCE and rms[i] <= exit_rms):

                cur = 0

        state[i] = cur

    state = merge_short_segments(state, min_len=AUDIO_MIN_SEG)

    state = merge_short_segments(state, min_len=AUDIO_MIN_SEG)

    return state

def build_move_direction_state(visual_df):

    req = ["center_x"]

    missing = [c for c in req if c not in visual_df.columns]

    if missing:

        raise ValueError(f"视觉 CSV 缺少这些列：{missing}")

    cx = visual_df["center_x"].astype(float).copy()

    cx = cx.ffill().bfill().fillna(0.0)

    cx = cx.rolling(CENTER_SMOOTH_MED, center=True, min_periods=1).median()

    cx = cx.rolling(CENTER_SMOOTH_MEAN, center=True, min_periods=1).mean()

    dx = cx - cx.shift(MOVE_DX_LAG)

    dx = dx.fillna(0.0).values.astype(np.float32)

    abs_dx = np.abs(dx)

    move_th = max(MOVE_ABS_MIN_TH, float(np.quantile(abs_dx, 0.88)))

    state = np.zeros(len(dx), dtype=np.int32)

    state[dx < -move_th] = 1

    state[dx > move_th] = 2

    state = merge_short_segments(state, min_len=MOVE_MIN_SEG)

    state = merge_short_segments(state, min_len=MOVE_MIN_SEG)

    return state, dx

                           

           

                           

def build_action_relative_features(visual_df):

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

        "right_shoulder", "right_elbow", "right_wrist"

    ]

    for p in pts:

        out[f"{p}_dx"] = X[f"{p}_x"].astype(float) - cx

        out[f"{p}_dy"] = X[f"{p}_y"].astype(float) - cy

    out = out.ffill().bfill().fillna(0.0)

    out = out.rolling(ACTION_SMOOTH, center=True, min_periods=1).median()

    return out

                           

        

                           

def run_t2s_branch(data_2d, win_size, step, out_channels, branch_name, min_len):

    scaler = StandardScaler()

    data_std = scaler.fit_transform(data_2d.astype(np.float32))

    params = copy.deepcopy(params_LSE)

    params["in_channels"] = data_std.shape[1]

    params["out_channels"] = out_channels

    params["nb_steps"] = 30

    params["win_size"] = win_size

    params["win_type"] = "hanning"

    t2s = Time2State(

        win_size,

        step,

        CausalConv_LSE_Adaper(params),

        DPGMM(None),

        params

    )

    t2s.fit(data_std, win_size, step)

    state_seq = np.asarray(t2s.state_seq).astype(int)

    state_seq, _ = remap_to_contiguous_labels(state_seq)

    state_seq = merge_short_segments(state_seq, min_len=min_len)

    state_seq, _ = remap_to_contiguous_labels(state_seq)

    return {

        "data_std": data_std,

        "state_seq": state_seq,

    }

def plot_branch(data_std, state_seq, output_png):

    plt.style.use("classic")

    plot_mts(data_std, state_seq)

    plt.savefig(output_png, dpi=200, bbox_inches="tight")

    plt.close()

def plot_aux_states(visual_times, audio_state_aligned, move_state, output_png):

    plt.style.use("classic")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].step(visual_times, audio_state_aligned, where="post")

    axes[0].set_ylabel("audio\n(0/1)")

    axes[0].set_title("Audio speaking state")

    axes[1].step(visual_times, move_state, where="post")

    axes[1].set_ylabel("move\n(0/1/2)")

    axes[1].set_title("Movement direction state")

    axes[1].set_xlabel("time")

    plt.tight_layout()

    plt.savefig(output_png, dpi=200, bbox_inches="tight")

    plt.close()

def plot_multiscale_states(state_matrix_df, output_png):

    run_cols = [c for c in state_matrix_df.columns if c.startswith("run")]

    n = len(run_cols)

    plt.style.use("classic")

    fig, axes = plt.subplots(n, 1, figsize=(16, max(10, 1.4 * n)), sharex=True)

    if n == 1:

        axes = [axes]

    t = state_matrix_df["time_sec"].values

    for ax, col in zip(axes, run_cols):

        ax.step(t, state_matrix_df[col].values, where="post")

        ax.set_ylabel(col.replace("_state", ""))

        ax.grid(False)

    axes[-1].set_xlabel("time")

    plt.tight_layout()

    plt.savefig(output_png, dpi=220, bbox_inches="tight")

    plt.close()

def build_raw_label_matrix(state_matrix_df, run_infos):

    raw_cols = ["time_sec"] + [info["col_name"] for info in run_infos]

    return state_matrix_df[raw_cols].copy()

def build_indicator_time_matrix(state_matrix_df, run_infos):

    mats = []

    state_rows = []

    for info in run_infos:

        col = info["col_name"]

        seq = state_matrix_df[col].values.astype(int)

        k_num = int(seq.max()) + 1

        onehot = one_hot_from_state(seq, k_num)

        for k in range(k_num):

            state_name = f"R{info['run_idx']}_W{info['win']}_S{info['step']}_state{k}"

            mats.append(onehot[:, k])

            state_rows.append({

                "global_state_name": state_name,

                "run_idx": info["run_idx"],

                "win": info["win"],

                "step": info["step"],

                "local_state": k,

            })

    indicator_mat = np.stack(mats, axis=1).astype(np.float32)

    indicator_df = pd.DataFrame(indicator_mat, columns=[r["global_state_name"] for r in state_rows])

    state_info_df = pd.DataFrame(state_rows)

    return indicator_df, state_info_df

def cluster_state_matrix(indicator_df, state_info_df, n_clusters, smooth_win=9):

    H = indicator_df.to_numpy(dtype=np.float32)                

    G = H.T                                                    

    G_smooth = pd.DataFrame(G.T).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy().T

    norm = np.linalg.norm(G_smooth, axis=1, keepdims=True)

    norm[norm < 1e-8] = 1.0

    X = G_smooth / norm

    try:

        clusterer = AgglomerativeClustering(

            n_clusters=n_clusters,

            metric="cosine",

            linkage="average"

        )

    except TypeError:

        clusterer = AgglomerativeClustering(

            n_clusters=n_clusters,

            affinity="cosine",

            linkage="average"

        )

    labels = clusterer.fit_predict(X).astype(int)

    labels, _ = remap_to_contiguous_labels(labels)

    out_info = state_info_df.copy()

    out_info["cluster_id"] = labels

    state_matrix_df = pd.DataFrame(

        G,

        index=indicator_df.columns,

        columns=[f"t{i}" for i in range(G.shape[1])]

    )

    return state_matrix_df, out_info

def build_meta_ratio_and_sequence(indicator_df, state_info_df, meta_min_len=24):

    H = indicator_df.to_numpy(dtype=np.float32)                

    cluster_labels = state_info_df["cluster_id"].to_numpy(dtype=int)

    C = int(cluster_labels.max()) + 1

    meta_vote = np.zeros((H.shape[0], C), dtype=np.float32)

    for s_idx, c in enumerate(cluster_labels):

        meta_vote[:, c] += H[:, s_idx]

    denom = meta_vote.sum(axis=1, keepdims=True)

    denom[denom < 1e-8] = 1.0

    meta_ratio_raw = meta_vote / denom

                     

    meta_state_seq_raw = np.argmax(meta_ratio_raw, axis=1).astype(int)

            

    meta_state_seq_smooth = merge_short_segments(meta_state_seq_raw, min_len=meta_min_len)

                           

    meta_state_seq, state_map = remap_to_contiguous_labels(meta_state_seq_smooth)

                            

    new_C = len(set(meta_state_seq))

    meta_ratio_new = np.zeros((H.shape[0], new_C), dtype=np.float32)

    for old_state, new_state in state_map.items():

        meta_ratio_new[:, new_state] += meta_ratio_raw[:, old_state]

           

    denom2 = meta_ratio_new.sum(axis=1, keepdims=True)

    denom2[denom2 < 1e-8] = 1.0

    meta_ratio_new = meta_ratio_new / denom2

    meta_ratio_df = pd.DataFrame(

        meta_ratio_new,

        columns=[f"meta_ratio_{i}" for i in range(meta_ratio_new.shape[1])]

    )

    meta_seq_df = pd.DataFrame({"meta_state": meta_state_seq})

    return meta_ratio_df, meta_seq_df, state_map

def plot_meta_state(meta_ratio_df, meta_state_seq, output_png):

    data_std = StandardScaler().fit_transform(meta_ratio_df.to_numpy(dtype=np.float32))

    plt.style.use("classic")

    plot_mts(data_std, meta_state_seq)

    plt.savefig(output_png, dpi=220, bbox_inches="tight")

    plt.close()

def draw_multi_timelines(frame, timeline_strips, run_infos, state_matrix_df, idx,

                         meta_timeline_strip=None, meta_seq=None):

    h, w, _ = frame.shape

    n_runs = len(timeline_strips)

    has_meta = (meta_timeline_strip is not None) and (meta_seq is not None)

    n_rows = n_runs + (1 if has_meta else 0)

    box_x1 = TIMELINE_MARGIN

    box_x2 = w - TIMELINE_MARGIN

    box_y2 = h - TIMELINE_MARGIN

    bar_h = 8

    gap = 2

    top_pad = 18

    left_label_w = 150

    total_h = top_pad + n_rows * (bar_h + gap) + 8

    box_y1 = max(0, box_y2 - total_h)

    overlay = frame.copy()

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

    frame[:] = cv2.addWeighted(overlay, TIMELINE_BG_ALPHA, frame, 1 - TIMELINE_BG_ALPHA, 0)

    title = f"{len(run_infos)}-scale T2S + meta timeline"

    cv2.putText(

        frame,

        title,

        (box_x1 + 6, box_y1 + 13),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (30, 30, 30),

        1

    )

    total_len = len(state_matrix_df)

    row_idx = 0

    if has_meta:

        y1 = box_y1 + top_pad + row_idx * (bar_h + gap)

        y2 = y1 + bar_h

        x1 = box_x1 + left_label_w

        x2 = box_x2 - 8

        target_w = x2 - x1

        target_h = y2 - y1

        strip_resized = cv2.resize(meta_timeline_strip, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        frame[y1:y2, x1:x2] = strip_resized

        cur_state = int(meta_seq[idx])

        color = STATE_PALETTE[cur_state % len(STATE_PALETTE)]

        label = f"META : {cur_state}"

        cv2.putText(

            frame,

            label,

            (box_x1 + 5, y1 + bar_h - 1),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.34,

            color,

            1

        )

        if total_len > 1:

            x_cur = x1 + int(round(idx / (total_len - 1) * (target_w - 1)))

        else:

            x_cur = x1

        cv2.line(frame, (x_cur, y1 - 1), (x_cur, y2 + 1), (255, 255, 255), 2)

        cv2.line(frame, (x_cur, y1 - 1), (x_cur, y2 + 1), (0, 0, 0), 1)

        row_idx += 1

    for i, (strip, info) in enumerate(zip(timeline_strips, run_infos)):

        y1 = box_y1 + top_pad + row_idx * (bar_h + gap)

        y2 = y1 + bar_h

        x1 = box_x1 + left_label_w

        x2 = box_x2 - 8

        target_w = x2 - x1

        target_h = y2 - y1

        strip_resized = cv2.resize(strip, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        frame[y1:y2, x1:x2] = strip_resized

        col = info["col_name"]

        cur_state = int(state_matrix_df.iloc[idx][col])

        color = STATE_PALETTE[cur_state % len(STATE_PALETTE)]

        label = f"R{i+1} W{info['win']} S{info['step']} : {cur_state}"

        cv2.putText(

            frame,

            label,

            (box_x1 + 5, y1 + bar_h - 1),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.34,

            color,

            1

        )

        if total_len > 1:

            x_cur = x1 + int(round(idx / (total_len - 1) * (target_w - 1)))

        else:

            x_cur = x1

        cv2.line(frame, (x_cur, y1 - 1), (x_cur, y2 + 1), (255, 255, 255), 2)

        cv2.line(frame, (x_cur, y1 - 1), (x_cur, y2 + 1), (0, 0, 0), 1)

        row_idx += 1

def collect_all_videos(video_root):

    video_root = Path(video_root)

    videos = [p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

    videos.sort()

    return videos

def find_corresponding_visual_csv(video_path, video_root, visual_csv_root):

    

       

    video_path = Path(video_path)

    video_root = Path(video_root)

    visual_csv_root = Path(visual_csv_root)

    rel = video_path.relative_to(video_root)

    candidates = [

        visual_csv_root / rel.with_suffix(".csv"),

        visual_csv_root / rel.parent / "teacher_visual_15d.csv",

    ]

    for p in candidates:

        if p.exists():

            return p

    return None

def find_corresponding_audio_csv(video_path, video_root, audio_csv_root):

    

       

    video_path = Path(video_path)

    video_root = Path(video_root)

    audio_csv_root = Path(audio_csv_root)

    rel = video_path.relative_to(video_root)

    candidates = [

        audio_csv_root / rel.with_suffix(".csv"),

        audio_csv_root / rel.parent / "pose_12fps_with_audio.csv",

    ]

    for p in candidates:

        if p.exists():

            return p

    return None

def build_output_paths(video_path, video_root, output_root):

    

       

    video_path = Path(video_path)

    video_root = Path(video_root)

    output_root = Path(output_root)

    rel = video_path.relative_to(video_root)

    case_dir = output_root / rel.with_suffix("")

    plot_dir = case_dir / "plots"

    case_dir.mkdir(parents=True, exist_ok=True)

    plot_dir.mkdir(parents=True, exist_ok=True)

    return {

        "CASE_DIR": case_dir,

        "PLOT_DIR": plot_dir,

        "OUTPUT_STATE_MATRIX_CSV": case_dir / "multiscale_t2s_state_matrix.csv",

        "OUTPUT_RUN_INFO_CSV": case_dir / "multiscale_t2s_run_info.csv",

        "OUTPUT_AUX_PNG": plot_dir / "audio_move_states.png",

        "OUTPUT_MULTI_PNG": plot_dir / "multiscale_t2s_states.png",

        "OUTPUT_META_PNG": plot_dir / "multiscale_meta_state.png",

        "OUTPUT_RAW_LABEL_MATRIX_CSV": case_dir / "multiscale_raw_label_matrix_Tx32.csv",

        "OUTPUT_INDICATOR_TIME_MATRIX_CSV": case_dir / "multiscale_indicator_time_matrix_TxS.csv",

        "OUTPUT_STATE_MATRIX_SXT_CSV": case_dir / "multiscale_state_matrix_SxT.csv",

        "OUTPUT_STATE_INFO_CSV": case_dir / "multiscale_state_info_with_cluster.csv",

        "OUTPUT_META_RATIO_CSV": case_dir / "multiscale_meta_ratio_TxC.csv",

        "OUTPUT_META_SEQ_CSV": case_dir / "multiscale_meta_state_seq.csv",

        "OUTPUT_FINAL_ALL_CSV": case_dir / "multiscale_t2s_with_meta.csv",

        "TEMP_VIDEO": case_dir / "multiscale_t2s_overlay_silent.mp4",

        "OUTPUT_VIDEO": case_dir / "multiscale_t2s_overlay_with_audio.mp4",

        "OUTPUT_K_SWEEP_CSV": case_dir / "meta_k_sweep_scores.csv",

    }

def choose_mode():

    print("\n请选择运行模式：")

    print("2 = 完整处理：生成结果 + 叠加视频")

    print("4 = 快速分析：只生成结果，不生成视频")

    while True:

        mode = input("请输入 2 / 4 : ").strip()

        if mode in {"2", "4"}:

            return mode

        print("输入无效，请重新输入。")

def process_one_case(video_path, visual_csv_path, audio_csv_path, outp, mode="2"):

    print("=" * 100)

    print(f"开始处理视频: {video_path}")

    print(f"视觉 CSV: {visual_csv_path}")

    print(f"音频 CSV: {audio_csv_path}")

    print(f"输出目录: {outp['CASE_DIR']}")

                               

             

                               

    audio_df = pd.read_csv(audio_csv_path)

    visual_df = pd.read_csv(visual_csv_path)

    if TIME_COL not in audio_df.columns:

        raise ValueError(f"音频 CSV 缺少 {TIME_COL}")

    if TIME_COL not in visual_df.columns:

        raise ValueError(f"动作 CSV 缺少 {TIME_COL}")

    audio_df = audio_df.sort_values(TIME_COL).reset_index(drop=True)

    visual_df = visual_df.sort_values(TIME_COL).reset_index(drop=True)

    audio_times = audio_df[TIME_COL].astype(float).values

    visual_times = visual_df[TIME_COL].astype(float).values

                               

             

                               

    audio_state = build_audio_binary_state(audio_df)

    move_state, move_dx = build_move_direction_state(visual_df)

    audio_idx_for_visual = np.searchsorted(audio_times, visual_times, side="right") - 1

    audio_idx_for_visual = np.clip(audio_idx_for_visual, 0, len(audio_state) - 1)

    audio_state_aligned = audio_state[audio_idx_for_visual]

    plot_aux_states(visual_times, audio_state_aligned, move_state, str(outp["OUTPUT_AUX_PNG"]))

                               

               

                               

    action_X = build_action_relative_features(visual_df)

    print("动作分支输入维度：", list(action_X.columns))

                               

                 

                               

    run_state_dict = {}

    run_infos = []

    timeline_strips = []

    for i, (win, step) in enumerate(zip(MULTI_ACTION_WINS, MULTI_ACTION_STEPS), start=1):

        run_name = f"run{i}_w{win}_s{step}"

        print(f"\n===== 第 {i} 次 T2S: ACTION_WIN={win}, ACTION_STEP={step} =====")

        result = run_t2s_branch(

            action_X.to_numpy(dtype=np.float32),

            win_size=win,

            step=step,

            out_channels=ACTION_OUT_CHANNELS,

            branch_name=run_name,

            min_len=get_action_min_len(win)

        )

        state_seq = align_sequence_to_length(result["state_seq"], len(visual_df))

        state_seq, _ = remap_to_contiguous_labels(state_seq)

        col_name = f"run{i}_state"

        run_state_dict[col_name] = state_seq

        out_png = outp["PLOT_DIR"] / f"{run_name}.png"

        plot_branch(result["data_std"], result["state_seq"], str(out_png))

        run_infos.append({

            "run_idx": i,

            "win": win,

            "step": step,

            "min_len": get_action_min_len(win),

            "col_name": col_name,

            "n_states": int(np.max(state_seq)) + 1,

            "plot_png": str(out_png),

        })

                               

                 

                               

    state_matrix_df = pd.DataFrame({

        "time_sec": visual_times,

        "audio_speech_state": audio_state_aligned,

        "move_direction_state": move_state,

        "move_dx": move_dx,

    })

    for info in run_infos:

        col = info["col_name"]

        state_matrix_df[col] = run_state_dict[col]

    state_matrix_df.to_csv(outp["OUTPUT_STATE_MATRIX_CSV"], index=False, encoding="utf-8-sig")

    pd.DataFrame(run_infos).to_csv(outp["OUTPUT_RUN_INFO_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出状态矩阵：{outp['OUTPUT_STATE_MATRIX_CSV']}")

    print(f"已输出运行信息：{outp['OUTPUT_RUN_INFO_CSV']}")

    plot_multiscale_states(state_matrix_df, str(outp["OUTPUT_MULTI_PNG"]))

    print(f"已输出综合状态图：{outp['OUTPUT_MULTI_PNG']}")

                               

                                                        

                               

    raw_label_df = build_raw_label_matrix(state_matrix_df, run_infos)

    raw_label_df.to_csv(outp["OUTPUT_RAW_LABEL_MATRIX_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出原始标签时间矩阵：{outp['OUTPUT_RAW_LABEL_MATRIX_CSV']}")

    indicator_df, state_info_df = build_indicator_time_matrix(state_matrix_df, run_infos)

    indicator_with_time_df = pd.concat(

        [pd.DataFrame({"time_sec": visual_times}), indicator_df],

        axis=1

    )

    indicator_with_time_df.to_csv(outp["OUTPUT_INDICATOR_TIME_MATRIX_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出指示时间矩阵：{outp['OUTPUT_INDICATOR_TIME_MATRIX_CSV']}")

    AUTO_STATE_CLUSTER_K, k_score_df, best_k_result = choose_meta_cluster_k_via_sweep(

        run_infos=run_infos,

        indicator_df=indicator_df,

        state_info_df=state_info_df,

        meta_min_len=META_MIN_LEN,

        smooth_win=STATE_CLUSTER_SMOOTH,

        min_allowed=2,

        save_csv_path=str(outp["OUTPUT_K_SWEEP_CSV"])

    )

    print(f"自动选择的 meta 聚类数（K sweep 选优）: {AUTO_STATE_CLUSTER_K}")

    print(f"K 选择评分表已保存: {outp['OUTPUT_K_SWEEP_CSV']}")

    print(k_score_df.head(10))

                  

    state_matrix_sxt_df = best_k_result["state_matrix_sxt_df"]

    state_info_with_cluster_df = best_k_result["state_info_with_cluster_df"]

    meta_ratio_df = best_k_result["meta_ratio_df"]

    meta_seq_df = best_k_result["meta_seq_df"]

    state_matrix_sxt_df.to_csv(outp["OUTPUT_STATE_MATRIX_SXT_CSV"], encoding="utf-8-sig")

    state_info_with_cluster_df.to_csv(outp["OUTPUT_STATE_INFO_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出状态矩阵：{outp['OUTPUT_STATE_MATRIX_SXT_CSV']}")

    print(f"已输出状态聚类信息：{outp['OUTPUT_STATE_INFO_CSV']}")

    meta_ratio_with_time_df = pd.concat(

        [pd.DataFrame({"time_sec": visual_times}), meta_ratio_df],

        axis=1

    )

    meta_ratio_with_time_df.to_csv(outp["OUTPUT_META_RATIO_CSV"], index=False, encoding="utf-8-sig")

    meta_seq_df_with_time = pd.concat(

        [pd.DataFrame({"time_sec": visual_times}), meta_seq_df],

        axis=1

    )

    meta_seq_df_with_time.to_csv(outp["OUTPUT_META_SEQ_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出 meta-state 比例：{outp['OUTPUT_META_RATIO_CSV']}")

    print(f"已输出 meta-state 序列：{outp['OUTPUT_META_SEQ_CSV']}")

    plot_meta_state(meta_ratio_df, meta_seq_df["meta_state"].values.astype(int), str(outp["OUTPUT_META_PNG"]))

    print(f"已输出 meta-state 图：{outp['OUTPUT_META_PNG']}")

    final_df = state_matrix_df.copy()

    final_df["meta_state"] = meta_seq_df["meta_state"].values.astype(int)

    for c in meta_ratio_df.columns:

        final_df[c] = meta_ratio_df[c].values

    final_df.to_csv(outp["OUTPUT_FINAL_ALL_CSV"], index=False, encoding="utf-8-sig")

    print(f"已输出总表：{outp['OUTPUT_FINAL_ALL_CSV']}")

    if mode == "4":

        print("模式4：已完成快速分析，跳过视频生成与音轨合并。")

        return

                               

                       

                               

    timeline_width = 1400

    for info in run_infos:

        col = info["col_name"]

        strip = build_timeline_strip(

            timeline_width,

            state_matrix_df[col].values.astype(int),

            STATE_PALETTE,

            bar_h=8

        )

        timeline_strips.append(strip)

    meta_timeline_strip = build_timeline_strip(

        timeline_width,

        meta_seq_df["meta_state"].values.astype(int),

        STATE_PALETTE,

        bar_h=8

    )

                               

              

                               

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:

        raise RuntimeError("视频 FPS 读取失败")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(outp["TEMP_VIDEO"]), fourcc, fps, (width, height))

    frame_id = 0

    total_rows = len(final_df)

    def move_text(v):

        if v == 1:

            return "left"

        if v == 2:

            return "right"

        return "still"

    def speech_text(v):

        return "speaking" if v == 1 else "silent"

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        current_time = frame_id / fps

        idx = np.searchsorted(visual_times, current_time, side="right") - 1

        idx = np.clip(idx, 0, total_rows - 1)

        row = visual_df.iloc[idx]

        state_row = final_df.iloc[idx]

        audio_state_val = int(state_row["audio_speech_state"])

        move_state_val = int(state_row["move_direction_state"])

        meta_state_val = int(state_row["meta_state"])

        center = (

            to_pixel_x(row["center_x"], width),

            to_pixel_y(row["center_y"], height)

        )

        if None in center:

            center = None

        ls = get_point(row, "left_shoulder", width, height)

        le = get_point(row, "left_elbow", width, height)

        lw = get_point(row, "left_wrist", width, height)

        rs = get_point(row, "right_shoulder", width, height)

        re = get_point(row, "right_elbow", width, height)

        rw = get_point(row, "right_wrist", width, height)

        draw_point(frame, center, COLOR_CENTER, radius=6)

        if center is not None:

            cv2.putText(frame, "center", (center[0] + 8, center[1] - 8),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 2)

        draw_line(frame, ls, le, COLOR_LEFT_ARM, 3)

        draw_line(frame, le, lw, COLOR_LEFT_ARM, 3)

        draw_point(frame, ls, COLOR_LEFT_ARM, 5)

        draw_point(frame, le, COLOR_LEFT_ARM, 5)

        draw_point(frame, lw, COLOR_LEFT_ARM, 5)

        draw_line(frame, rs, re, COLOR_RIGHT_ARM, 3)

        draw_line(frame, re, rw, COLOR_RIGHT_ARM, 3)

        draw_point(frame, rs, COLOR_RIGHT_ARM, 5)

        draw_point(frame, re, COLOR_RIGHT_ARM, 5)

        draw_point(frame, rw, COLOR_RIGHT_ARM, 5)

        draw_orientation_arrow(frame, center, row["orientation_score"])

        overlay = frame.copy()

        cv2.rectangle(overlay, (20, 20), (620, 180), COLOR_BOX, -1)

        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        cv2.putText(frame, f"{len(run_infos)}-scale action T2S", (35, 48),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2)

        cv2.putText(frame, f"time = {current_time:.2f}s", (35, 76),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT, 2)

        cv2.putText(frame, f"Audio: {speech_text(audio_state_val)}", (35, 104),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 2)

        cv2.putText(frame, f"Move: {move_text(move_state_val)}", (35, 132),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 2)

        cv2.putText(frame, f"Meta-state: {meta_state_val}", (35, 160),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.56,

                    STATE_PALETTE[meta_state_val % len(STATE_PALETTE)], 2)

        draw_multi_timelines(

            frame,

            timeline_strips,

            run_infos,

            final_df,

            idx,

            meta_timeline_strip=meta_timeline_strip,

            meta_seq=final_df["meta_state"].values.astype(int)

        )

        writer.write(frame)

        frame_id += 1

        if frame_id % 200 == 0:

            print(f"已处理 {frame_id} 帧")

    cap.release()

    writer.release()

    print(f"已生成无声叠加视频：{outp['TEMP_VIDEO']}")

                               

                

                               

    print("开始合并原视频音轨...")

    overlay_clip = VideoFileClip(str(outp["TEMP_VIDEO"]))

    orig_clip = VideoFileClip(str(video_path))

    try:

        if orig_clip.audio is not None:

            audio_clip = orig_clip.audio.subclipped(0, overlay_clip.duration)

            final_clip = overlay_clip.with_audio(audio_clip)

        else:

            print("警告：原视频没有音轨，将只输出无声视频。")

            final_clip = overlay_clip

        final_clip.write_videofile(

            str(outp["OUTPUT_VIDEO"]),

            codec="libx264",

            audio_codec="aac"

        )

    finally:

        try:

            overlay_clip.close()

        except Exception:

            pass

        try:

            orig_clip.close()

        except Exception:

            pass

        try:

            final_clip.close()

        except Exception:

            pass

    print(f"完成，带声音的视频已保存到：{outp['OUTPUT_VIDEO']}")

def main():

    print("=== demoMixMix BATCH START ===")

    print("__file__ =", __file__)

    video_root = Path(VIDEO_ROOT)

    visual_root = Path(VISUAL_CSV_ROOT)

    audio_root = Path(AUDIO_CSV_ROOT)

    output_root = Path(OUTPUT_ROOT)

    output_root.mkdir(parents=True, exist_ok=True)

    if not video_root.exists():

        raise FileNotFoundError(f"视频根目录不存在: {video_root}")

    if not visual_root.exists():

        raise FileNotFoundError(f"视觉 CSV 根目录不存在: {visual_root}")

    if not audio_root.exists():

        raise FileNotFoundError(f"音频 CSV 根目录不存在: {audio_root}")

    videos = collect_all_videos(video_root)

    if len(videos) == 0:

        print(f"没有在 {video_root} 下找到视频文件")

        return

    print(f"共找到 {len(videos)} 个视频")

    for i, v in enumerate(videos, 1):

        print(f"{i:03d}. {v}")

    mode = choose_mode()

    success_count = 0

    fail_count = 0

    skip_count = 0

    for video_path in videos:

        visual_csv_path = find_corresponding_visual_csv(video_path, video_root, visual_root)

        audio_csv_path = find_corresponding_audio_csv(video_path, video_root, audio_root)

        if visual_csv_path is None:

            skip_count += 1

            print("=" * 100)

            print(f"[跳过] 找不到对应视觉 CSV: {video_path}")

            continue

        if audio_csv_path is None:

            skip_count += 1

            print("=" * 100)

            print(f"[跳过] 找不到对应音频 CSV: {video_path}")

            continue

        outp = build_output_paths(video_path, video_root, output_root)

        try:

            process_one_case(video_path, visual_csv_path, audio_csv_path, outp, mode=mode)

            success_count += 1

        except Exception as e:

            fail_count += 1

            print("=" * 100)

            print(f"[失败] {video_path}")

            print(f"原因: {e}")

    print("=" * 100)

    print("全部处理完成")

    print(f"成功: {success_count}")

    print(f"失败: {fail_count}")

    print(f"跳过: {skip_count}")

    print(f"输出根目录: {output_root}")

    print("=== demoMixMix BATCH END ===")

if __name__ == "__main__":

    main()
