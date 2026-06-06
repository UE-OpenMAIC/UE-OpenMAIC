                       

   

from __future__ import annotations

import argparse

import copy

import inspect

import json

import math

import re

import sys

import time

import types

from pathlib import Path

from typing import Iterable

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from sklearn.decomposition import PCA

try:

    import torch

except Exception:

    torch = None

try:

    from moviepy import VideoFileClip

    HAS_MOVIEPY = True

except Exception:

    try:

        from moviepy.editor import VideoFileClip

        HAS_MOVIEPY = True

    except Exception:

        VideoFileClip = None

        HAS_MOVIEPY = False

                                                           

                                    

                                                           

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = Path(r"D:\code\teacherT2S")

TIME2STATE_ROOT = PROJECT_ROOT / "Time2State"

TIME2STATE_PACKAGE_ROOT = TIME2STATE_ROOT / "Time2State"

TSPY_ROOT = TIME2STATE_ROOT / "TSpy"

def _norm_path_for_sys_path(p: str) -> str:

    try:

        return str(Path(p or ".").resolve()).lower()

    except Exception:

        return str(p).lower()

def _prepend_sys_path(p: Path) -> None:

    if not p.exists():

        return

    ps = str(p.resolve())

    norm = ps.lower()

    sys.path[:] = [x for x in sys.path if _norm_path_for_sys_path(x) != norm]

    sys.path.insert(0, ps)

                                         

_prepend_sys_path(PROJECT_ROOT)

_prepend_sys_path(TSPY_ROOT)

_prepend_sys_path(TIME2STATE_ROOT)

_prepend_sys_path(TIME2STATE_PACKAGE_ROOT)

def _clear_conflicting_time2state_modules() -> None:

    for _m in ["utils", "Time2State"]:

        if _m in sys.modules:

            del sys.modules[_m]

def _install_time2state_package_shim() -> None:

    candidate_dirs = [

        TIME2STATE_PACKAGE_ROOT,

        TIME2STATE_ROOT,

        SCRIPT_DIR / "Time2State",

        PROJECT_ROOT / "Time2State" / "Time2State",

    ]

    pkg_dir = None

    for d in candidate_dirs:

        if (d / "time2state.py").exists():

            pkg_dir = d

            break

    if pkg_dir is None:

        return

    pkg = types.ModuleType("Time2State")

    pkg.__path__ = [str(pkg_dir.resolve())]

    init_file = pkg_dir / "__init__.py"

    pkg.__file__ = str(init_file.resolve()) if init_file.exists() else str(pkg_dir.resolve())

    sys.modules["Time2State"] = pkg

_clear_conflicting_time2state_modules()

try:

    from TSpy.view import plot_mts

except Exception:

    plot_mts = None

_clear_conflicting_time2state_modules()

try:

    from Time2State.time2state import Time2State

    from Time2State.adapers import CausalConv_LSE_Adaper

    from Time2State.clustering import DPGMM

    from Time2State.default_params import params_LSE

except Exception:

    _clear_conflicting_time2state_modules()

    _install_time2state_package_shim()

    try:

        from Time2State.time2state import Time2State

        from Time2State.adapers import CausalConv_LSE_Adaper

        from Time2State.clustering import DPGMM

        from Time2State.default_params import params_LSE

    except Exception:

        print("[IMPORT ERROR] Time2State 导入失败")

        print("PROJECT_ROOT =", PROJECT_ROOT)

        print("TIME2STATE_ROOT =", TIME2STATE_ROOT)

        print("TIME2STATE_PACKAGE_ROOT =", TIME2STATE_PACKAGE_ROOT)

        print("inner time2state.py exists =", (TIME2STATE_PACKAGE_ROOT / "time2state.py").exists())

        print("inner utils.py exists =", (TIME2STATE_PACKAGE_ROOT / "utils.py").exists())

        print("sys.path 前 12 项:")

        for i, p in enumerate(sys.path[:12]):

            print(f"  {i}: {p}")

        raise

                                                           

         

                                                           

VIDEO_ROOT = Path(r"D:\code\teacherT2S\yolo\input")

T2S_OUTPUT_ROOT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

NLP_OUTPUT_ROOT = Path(r"D:\code\teacherT2S\yolo\surprisal_batch_output")

                                               

OUTPUT_ROOT = Path(r"D:\code\teacherT2S\nlp_first_then_cross_video_output")

PER_VIDEO_ROOT = OUTPUT_ROOT / "per_video"

CROSS_VIDEO_ROOT = OUTPUT_ROOT / "_cross_video_pid_nlp_proto"

PLOT_DIR = CROSS_VIDEO_ROOT / "plots"

NLP_MODE6_TOTAL_DIR = NLP_OUTPUT_ROOT / "_mode6_total_export"

NLP_MODE6_ALL_WINDOW = NLP_MODE6_TOTAL_DIR / "ALL_window_surprisal_features.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

                                                           

                       

                                                           

MULTI_WINS = [72 - 2 * i for i in range(32)]

MULTI_STEPS = [max(8, w // 4) for w in MULTI_WINS]

SELECT_TOP_K_BRANCHES = 16

T2S_M = 10

T2S_N = 4

T2S_NB_STEPS = 30

T2S_OUT_CHANNELS = 8

T2S_KERNEL_SIZE = None

T2S_WIN_TYPE = "hanning"

T2S_GPU = 0

T2S_USE_CUDA = bool(torch is not None and torch.cuda.is_available())

PID_KP = 0.45

PID_KI = 0.35

PID_KD = 0.20

PID_SOFTMAX_TAU = 0.15

FINAL_K_MIN = 2

FINAL_K_MAX = 12

MAX_AGGLO_ROWS = 4500

SILHOUETTE_SAMPLE_ROWS = 2500

META_MIN_LEN = 24

USE_META_RATIO_FEATURES = True

PREFERRED_NLP_NUMERIC_COLS = [

    "norm_surprisal",

    "avg_surprisal",

    "std_surprisal",

    "lm_token_count",

    "text_char_count",

    "window_text_char_count",

    "token_count",

]

         

GLOBAL_K_MIN = 2

GLOBAL_K_MAX = 12

FORCE_GLOBAL_K = 0

MAX_LABEL_POINTS = 220

                                                           

         

                                                           

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

def safe_filename(x) -> str:

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    return re.sub(r"_+", "_", s).strip("_") or "video"

def video_id_match(a, b) -> bool:

    a = norm_video_id(a)

    b = norm_video_id(b)

    return a == b or tail_video_id(a) == tail_video_id(b)

def remap_to_contiguous_labels(seq):

    seq = np.asarray(seq).astype(int)

    vals = sorted(pd.unique(pd.Series(seq).dropna().astype(int)))

    mp = {int(v): i for i, v in enumerate(vals)}

    return np.array([mp[int(v)] for v in seq], dtype=int), mp

def align_sequence_to_length(seq, target_len: int) -> np.ndarray:

    seq = np.asarray(seq).astype(int)

    if len(seq) == target_len:

        return seq

    if len(seq) <= 1:

        return np.zeros(target_len, dtype=int)

    src_x = np.linspace(0, 1, len(seq))

    dst_x = np.linspace(0, 1, target_len)

    idx = np.searchsorted(src_x, dst_x, side="left")

    idx = np.clip(idx, 0, len(seq) - 1)

    return seq[idx]

def fill_teacher_sequence_to_full(seq_teacher, teacher_mask, skip_value=-1) -> np.ndarray:

    teacher_mask = np.asarray(teacher_mask).astype(bool)

    seq_teacher = np.asarray(seq_teacher).astype(int)

    if int(teacher_mask.sum()) != len(seq_teacher):

        raise ValueError(

            f"teacher_mask 数量与 seq_teacher 长度不一致: "

            f"mask={int(teacher_mask.sum())}, seq={len(seq_teacher)}"

        )

    full = np.full(len(teacher_mask), int(skip_value), dtype=int)

    full[teacher_mask] = seq_teacher

    return full

def get_segments(seq):

    values = list(np.asarray(seq, dtype=int))

    if not values:

        return []

    segs = []

    s = 0

    for i in range(1, len(values)):

        if values[i] != values[s]:

            segs.append((s, i - 1, int(values[s])))

            s = i

    segs.append((s, len(values) - 1, int(values[s])))

    return segs

def merge_short_segments(seq, min_len=24) -> np.ndarray:

    seq = np.asarray(seq, dtype=int).copy()

    if len(seq) == 0:

        return seq

    changed = True

    while changed:

        changed = False

        segs = get_segments(seq)

        for i, (s, e, lab) in enumerate(segs):

            if e - s + 1 >= min_len:

                continue

            left_lab = segs[i - 1][2] if i > 0 else None

            right_lab = segs[i + 1][2] if i < len(segs) - 1 else None

            if left_lab is None and right_lab is None:

                continue

            if left_lab is None:

                seq[s:e + 1] = right_lab

            elif right_lab is None:

                seq[s:e + 1] = left_lab

            else:

                left_len = segs[i - 1][1] - segs[i - 1][0] + 1

                right_len = segs[i + 1][1] - segs[i + 1][0] + 1

                seq[s:e + 1] = left_lab if left_len >= right_len else right_lab

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

    for s, e, _lab in get_segments(seq):

        if e - s + 1 < short_len:

            short_points += e - s + 1

    return float(short_points) / float(len(seq))

def safe_nmi(a, b) -> float:

    try:

        return float(normalized_mutual_info_score(a, b, average_method="geometric"))

    except Exception:

        return 0.0

def get_action_min_len(win: int) -> int:

    return max(18, int(win) // 2)

def safe_normalized_entropy(seq: Iterable[int]) -> float:

    seq = pd.Series(seq).dropna().astype(int).values

    if len(seq) == 0:

        return 0.0

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

    if len(probs) <= 1:

        return 0.0

    ent = -np.sum(probs * np.log(probs + 1e-12))

    ent_max = np.log(len(probs))

    return 0.0 if ent_max < 1e-12 else float(ent / ent_max)

                                                           

         

                                                           

def find_all_case_dirs() -> list[Path]:

    case_dirs = []

    for p in T2S_OUTPUT_ROOT.rglob("multiscale_t2s_with_meta.csv"):

        case_dir = p.parent

        if "_cross_video" in str(case_dir).replace("\\", "/"):

            continue

        case_dirs.append(case_dir)

    return sorted(set(case_dirs))

def find_video_file(video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    candidates = []

    for ext in VIDEO_EXTS:

        candidates.append(VIDEO_ROOT / tail / f"{tail}{ext}")

        candidates.append(VIDEO_ROOT / f"{tail}{ext}")

        candidates.append(VIDEO_ROOT / Path(*vid.split("/")) / f"{tail}{ext}")

    for p in candidates:

        if p.exists():

            return p

    if VIDEO_ROOT.exists():

        for p in VIDEO_ROOT.rglob("*"):

            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == tail:

                return p

    return None

def find_nlp_window_csv(video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    candidates = [

        NLP_OUTPUT_ROOT / Path(*vid.split("/")) / "window_surprisal_features.csv",

        NLP_OUTPUT_ROOT / tail / tail / "window_surprisal_features.csv",

        NLP_OUTPUT_ROOT / tail / "window_surprisal_features.csv",

    ]

    for p in candidates:

        if p.exists():

            return p

    if NLP_OUTPUT_ROOT.exists():

        for p in NLP_OUTPUT_ROOT.rglob("window_surprisal_features.csv"):

            pp = str(p).replace("\\", "/")

            if f"/{tail}/" in pp or p.parent.name == tail or p.parent.parent.name == tail:

                return p

    return None

                                                           

           

                                                           

def load_individual_or_mode6_nlp(video_id: str) -> pd.DataFrame:

    p = find_nlp_window_csv(video_id)

    if p is not None:

        df = pd.read_csv(p)

        df["__nlp_source_file"] = str(p)

        return df

    if NLP_MODE6_ALL_WINDOW.exists():

        all_df = pd.read_csv(NLP_MODE6_ALL_WINDOW)

        id_cols = [c for c in ["video_id", "case_id", "video", "source_video", "video_path"] if c in all_df.columns]

        if id_cols:

            mask = pd.Series(False, index=all_df.index)

            for c in id_cols:

                mask |= all_df[c].astype(str).map(lambda x: video_id_match(x, video_id))

            sub = all_df.loc[mask].copy()

            if len(sub) > 0:

                sub["__nlp_source_file"] = str(NLP_MODE6_ALL_WINDOW)

                return sub

    raise FileNotFoundError(f"找不到 video_id={video_id} 对应的 NLP window_surprisal_features.csv")

def normalize_nlp_time_columns(nlp_df: pd.DataFrame) -> pd.DataFrame:

    df = nlp_df.copy()

    if "window_start" not in df.columns:

        for c in ["start_sec", "start", "begin_sec", "band_start"]:

            if c in df.columns:

                df["window_start"] = pd.to_numeric(df[c], errors="coerce")

                break

    if "window_end" not in df.columns:

        for c in ["end_sec", "end", "finish_sec", "band_end"]:

            if c in df.columns:

                df["window_end"] = pd.to_numeric(df[c], errors="coerce")

                break

    if "window_center" not in df.columns and {"window_start", "window_end"}.issubset(df.columns):

        df["window_center"] = (

            pd.to_numeric(df["window_start"], errors="coerce") +

            pd.to_numeric(df["window_end"], errors="coerce")

        ) / 2.0

    missing = sorted({"window_start", "window_end", "window_center"} - set(df.columns))

    if missing:

        raise ValueError(f"NLP CSV 缺少时间窗列: {missing}")

    for c in ["window_start", "window_end", "window_center"]:

        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["window_center"]).sort_values("window_center").reset_index(drop=True)

    if "text_joined" not in df.columns:

        for c in ["text", "text_band", "asr_text", "window_text"]:

            if c in df.columns:

                df["text_joined"] = df[c].fillna("").astype(str)

                break

    if "text_joined" not in df.columns:

        df["text_joined"] = ""

    return df

def choose_nlp_numeric_cols(nlp_df: pd.DataFrame) -> list[str]:

    cols = [c for c in PREFERRED_NLP_NUMERIC_COLS if c in nlp_df.columns]

    for c in nlp_df.columns:

        low = c.lower()

        if c not in cols and any(k in low for k in ["surprisal", "token", "score", "entropy"]):

            if pd.to_numeric(nlp_df[c], errors="coerce").notna().sum() > 0:

                cols.append(c)

    bad = {"window_start", "window_end", "window_center", "start_sec", "end_sec", "video_id", "case_id"}

    out = []

    for c in cols:

        if c not in bad and c not in out:

            out.append(c)

    return out

def align_nlp_to_frames(frame_times: np.ndarray, nlp_df: pd.DataFrame) -> pd.DataFrame:

    nlp_df = normalize_nlp_time_columns(nlp_df)

    nlp_cols = choose_nlp_numeric_cols(nlp_df)

    out = pd.DataFrame({"time_sec": frame_times.astype(float)})

    centers = nlp_df["window_center"].astype(float).to_numpy()

    if len(centers) == 0:

        raise ValueError("NLP window rows 为空")

    for c in nlp_cols:

        y = pd.to_numeric(nlp_df[c], errors="coerce").astype(float).to_numpy()

        valid = np.isfinite(centers) & np.isfinite(y)

        if valid.sum() <= 1:

            out[f"nlp_{c}"] = 0.0

        else:

            out[f"nlp_{c}"] = np.interp(

                frame_times.astype(float),

                centers[valid],

                y[valid],

                left=float(y[valid][0]),

                right=float(y[valid][-1]),

            )

    nearest_idx = np.searchsorted(centers, frame_times, side="left")

    nearest_idx = np.clip(nearest_idx, 0, len(centers) - 1)

    out["nlp_text_joined"] = nlp_df["text_joined"].fillna("").astype(str).to_numpy()[nearest_idx]

    out["nlp_window_start"] = nlp_df["window_start"].astype(float).to_numpy()[nearest_idx]

    out["nlp_window_end"] = nlp_df["window_end"].astype(float).to_numpy()[nearest_idx]

    out["nlp_window_center"] = nlp_df["window_center"].astype(float).to_numpy()[nearest_idx]

    return out

                                                           

              

                                                           

def one_hot_series(values: pd.Series, prefix: str, ignore_negative=True) -> pd.DataFrame:

    vals = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int)

    cats = sorted([int(x) for x in pd.unique(vals) if (not ignore_negative or int(x) >= 0)])

    out = pd.DataFrame(index=values.index)

    for c in cats:

        out[f"{prefix}_{c}"] = vals.eq(c).astype(float)

    return out

def build_combined_features(frame_df: pd.DataFrame, nlp_aligned: pd.DataFrame) -> pd.DataFrame:

    parts = []

                                                              

    parts.append(one_hot_series(frame_df["meta_state"], "local_meta"))

    if USE_META_RATIO_FEATURES:

        meta_ratio_cols = [c for c in frame_df.columns if str(c).startswith("meta_ratio_")]

        if meta_ratio_cols:

            meta_ratio = frame_df[meta_ratio_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            parts.append(meta_ratio.add_prefix("l1_"))

    nlp_num_cols = [

        c for c in nlp_aligned.columns

        if c.startswith("nlp_") and c not in {"nlp_text_joined", "nlp_window_start", "nlp_window_end", "nlp_window_center"}

    ]

    if nlp_num_cols:

        nlp_num = nlp_aligned[nlp_num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        nlp_num = nlp_num.rolling(5, center=True, min_periods=1).mean()

        parts.append(nlp_num)

    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    keep_cols = [c for c in X.columns if pd.to_numeric(X[c], errors="coerce").dropna().nunique() > 1]

    X = X[keep_cols].copy()

    if X.shape[1] == 0:

        raise ValueError("组合特征全部为常量，无法运行单视频 NLP-first PID-T2S")

    return X

                                                           

                  

                                                           

def run_t2s_branch(data_2d, win_size: int, step: int, out_channels: int, branch_name: str, min_len: int):

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

        from Time2State import encoders

        sig = inspect.signature(encoders.CausalConv_LSE.__init__)

        allowed = {name for name, _p in sig.parameters.items() if name != "self"}

        params_for_encoder = {k: v for k, v in params.items() if k in allowed}

    except Exception:

        params_for_encoder = params

    if data_std.shape[0] < int(win_size):

        raise ValueError(f"rows={data_std.shape[0]} < win={win_size}")

    n_windows = 1 + max(0, (data_std.shape[0] - int(win_size)) // int(step))

    if n_windows < 2:

        raise ValueError(f"Too few windows: rows={data_std.shape[0]}, win={win_size}, step={step}, windows={n_windows}")

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

    return {"data_std": data_std, "state_seq": state_seq}

                                                           

                   

                                                           

def branch_health_score(seq, median_segments=None, short_len=18) -> dict:

    seq = np.asarray(seq, dtype=int)

    if len(seq) == 0:

        return {

            "health": 0.0,

            "state_entropy": 0.0,

            "dominant_ratio": 1.0,

            "short_segment_ratio": 1.0,

            "segment_reasonable_score": 0.0,

            "n_states": 0,

            "segments": 0,

        }

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

    ent = normalized_entropy(seq)

    dominant_ratio = float(probs.max())

    short_ratio = short_segment_ratio(seq, short_len=short_len)

    seg_count = count_segments(seq)

    if median_segments is None or median_segments <= 0:

        segment_reasonable = 1.0

    else:

        dev = abs(seg_count - float(median_segments)) / max(float(median_segments), 1.0)

        segment_reasonable = float(np.exp(-dev))

    non_collapse = 1.0 - dominant_ratio

    health = 0.40 * ent + 0.30 * non_collapse + 0.20 * (1.0 - short_ratio) + 0.10 * segment_reasonable

    return {

        "health": float(np.clip(health, 0.0, 1.0)),

        "state_entropy": float(ent),

        "dominant_ratio": float(dominant_ratio),

        "short_segment_ratio": float(short_ratio),

        "segment_reasonable_score": float(segment_reasonable),

        "n_states": int(len(vals)),

        "segments": int(seg_count),

    }

def temporal_stability_score(seq) -> float:

    seq = np.asarray(seq, dtype=int)

    if len(seq) < 10:

        return 0.0

    scores = []

    for lag in [1, 3, 5, 9]:

        if len(seq) > lag + 1:

            scores.append(safe_nmi(seq[:-lag], seq[lag:]))

    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0

def derivative_fragment_score(seq, short_len=18) -> float:

    seq = np.asarray(seq, dtype=int)

    if len(seq) == 0:

        return 0.0

    transitions = float(np.mean(seq[1:] != seq[:-1])) if len(seq) > 1 else 0.0

    short_ratio = short_segment_ratio(seq, short_len=short_len)

    return float(np.clip(1.0 - 0.5 * transitions - 0.5 * short_ratio, 0.0, 1.0))

def compute_pid_reliability(candidate_branch_sequences) -> pd.DataFrame:

    seg_counts = [count_segments(seq) for _name, _win, _step, seq in candidate_branch_sequences]

    median_segments = float(np.median(seg_counts)) if seg_counts else 0.0

    rows = []

    for name, win, step, seq in candidate_branch_sequences:

        h = branch_health_score(seq, median_segments=median_segments, short_len=get_action_min_len(win))

        p_term = h["health"]

        i_term = temporal_stability_score(seq)

        d_term = derivative_fragment_score(seq, short_len=get_action_min_len(win))

        pid_score = PID_KP * p_term + PID_KI * i_term + PID_KD * d_term

        row = {

            "branch": name,

            "win": int(win),

            "step": int(step),

            "pid_p_health": float(p_term),

            "pid_i_temporal_stability": float(i_term),

            "pid_d_fragment_score": float(d_term),

            "pid_score": float(pid_score),

        }

        row.update(h)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("pid_score", ascending=False).reset_index(drop=True)

    df["branch_rank"] = np.arange(1, len(df) + 1)

    df["selected_for_meta"] = (df["branch_rank"] <= int(SELECT_TOP_K_BRANCHES)).astype(int)

    return df

def softmax_weights(scores, tau=0.15) -> np.ndarray:

    scores = np.asarray(scores, dtype=np.float64)

    if len(scores) == 0:

        return scores

    tau = max(float(tau), 1e-6)

    z = (scores - scores.max()) / tau

    w = np.exp(z)

    if not np.isfinite(w).all() or w.sum() <= 0:

        return np.ones_like(scores) / len(scores)

    return w / w.sum()

def build_weighted_branch_indicator(selected_sequences, weights) -> pd.DataFrame:

    mats, col_names = [], []

    for bi, seq in enumerate(selected_sequences, start=1):

        seq = np.asarray(seq, dtype=int)

        states = sorted(pd.unique(pd.Series(seq).astype(int)))

        for st in states:

            mats.append(((seq == int(st)).astype(np.float32) * float(weights[bi - 1])).reshape(-1, 1))

            col_names.append(f"b{bi}_s{st}")

    if not mats:

        raise ValueError("没有可用于最终 meta 聚合的 selected branch indicators")

    return pd.DataFrame(np.hstack(mats), columns=col_names)

def choose_final_k_and_cluster(indicator_df: pd.DataFrame):

    X = StandardScaler().fit_transform(indicator_df.values.astype(np.float32))

    n = X.shape[0]

    if n < 10:

        raise ValueError(f"最终聚类样本太少: n={n}")

    k_max = min(FINAL_K_MAX, n - 1)

    k_min = min(FINAL_K_MIN, k_max)

    ks = list(range(k_min, k_max + 1))

    if n > SILHOUETTE_SAMPLE_ROWS:

        rng = np.random.default_rng(42)

        sample_idx = np.sort(rng.choice(n, size=SILHOUETTE_SAMPLE_ROWS, replace=False))

    else:

        sample_idx = np.arange(n)

    score_rows, best_k, best_score = [], None, -1e18

    for k in ks:

        try:

            if n > MAX_AGGLO_ROWS:

                km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=10)

                labels_sample = km.fit_predict(X[sample_idx])

                sil = -1.0 if len(np.unique(labels_sample)) < 2 else silhouette_score(X[sample_idx], labels_sample, metric="euclidean")

            else:

                try:

                    clus = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")

                except TypeError:

                    clus = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")

                labels_sample = clus.fit_predict(X[sample_idx])

                sil = -1.0 if len(np.unique(labels_sample)) < 2 else silhouette_score(X[sample_idx], labels_sample, metric="cosine")

            score_rows.append({"K": int(k), "silhouette": float(sil)})

            if sil > best_score:

                best_score, best_k = float(sil), int(k)

        except Exception as e:

            score_rows.append({"K": int(k), "silhouette": np.nan, "error": repr(e)})

    if best_k is None:

        best_k = min(4, k_max)

    if n > MAX_AGGLO_ROWS:

        model = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=2048, n_init=20)

        labels = model.fit_predict(X)

    else:

        try:

            model = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")

        except TypeError:

            model = AgglomerativeClustering(n_clusters=best_k, affinity="cosine", linkage="average")

        labels = model.fit_predict(X)

    labels, _ = remap_to_contiguous_labels(labels.astype(int))

    labels = merge_short_segments(labels, min_len=META_MIN_LEN)

    labels, _ = remap_to_contiguous_labels(labels)

    return labels, pd.DataFrame(score_rows), int(best_k)

def run_pid_t2s_on_combined_features(X_teacher: pd.DataFrame, out_dir: Path):

    branch_dir = out_dir / "branches"

    ensure_dir(branch_dir)

    candidate_sequences, raw_state_cols, branch_infos = [], {}, []

    X_np = X_teacher.to_numpy(dtype=np.float32)

    for i, (win, step) in enumerate(zip(MULTI_WINS, MULTI_STEPS), start=1):

        name = f"run{i}_w{win}_s{step}"

        print(f"  [{i:02d}/{len(MULTI_WINS)}] {name}")

        start = time.time()

        try:

            result = run_t2s_branch(X_np, int(win), int(step), int(T2S_OUT_CHANNELS), name, get_action_min_len(int(win)))

            seq = align_sequence_to_length(result["state_seq"], len(X_teacher))

            seq, _ = remap_to_contiguous_labels(seq)

            elapsed = time.time() - start

            raw_state_cols[f"run{i}_state"] = seq

            candidate_sequences.append((name, int(win), int(step), seq.copy()))

            branch_infos.append({

                "branch": name, "run_idx": int(i), "win": int(win), "step": int(step),

                "states": int(np.max(seq)) + 1, "segments": int(count_segments(seq)),

                "seconds": float(elapsed), "status": "ok", "error": "",

            })

            if plot_mts is not None:

                try:

                    plt.style.use("classic")

                    plot_mts(result["data_std"], seq)

                    plt.savefig(branch_dir / f"{name}.png", dpi=180, bbox_inches="tight")

                    plt.close()

                except Exception:

                    pass

        except Exception as e:

            elapsed = time.time() - start

            branch_infos.append({

                "branch": name, "run_idx": int(i), "win": int(win), "step": int(step),

                "states": 0, "segments": 0, "seconds": float(elapsed),

                "status": "failed", "error": repr(e),

            })

            print(f"    [跳过] {name}: {e}")

    if len(candidate_sequences) == 0:

        raise RuntimeError("没有任何 T2S 分支成功运行，无法继续 PID meta 聚合")

    pid_df = compute_pid_reliability(candidate_sequences)

    pid_df = pid_df.merge(pd.DataFrame(branch_infos), on=["branch", "win", "step"], how="left")

    selected_names = pid_df.loc[pid_df["selected_for_meta"].eq(1), "branch"].astype(str).tolist()

    seq_by_name = {name: seq for name, _w, _s, seq in candidate_sequences}

    selected_sequences = [seq_by_name[n] for n in selected_names if n in seq_by_name]

    scores = pid_df.loc[pid_df["selected_for_meta"].eq(1), "pid_score"].astype(float).to_numpy()

    weights = softmax_weights(scores, tau=PID_SOFTMAX_TAU)

    selected_weight_map = {name: float(w) for name, w in zip(selected_names, weights)}

    pid_df["pid_meta_weight"] = pid_df["branch"].astype(str).map(selected_weight_map).fillna(0.0)

    indicator_df = build_weighted_branch_indicator(selected_sequences, weights)

    final_teacher_seq, k_sweep_df, best_k = choose_final_k_and_cluster(indicator_df)

    raw_df = pd.DataFrame(raw_state_cols)

    raw_df.insert(0, "teacher_row_idx", np.arange(len(raw_df)))

    return final_teacher_seq, pid_df, raw_df, k_sweep_df, best_k

                                                           

           

                                                           

def plot_single_video_timeline(result_df: pd.DataFrame, out_png: Path) -> None:

    t = result_df["time_sec"].astype(float).to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(18, 9), sharex=True)

    axes[0].step(t, result_df["meta_state"].astype(int).to_numpy(), where="post")

    axes[0].set_ylabel("layer1\nlocal")

    if "nlp_norm_surprisal" in result_df.columns:

        axes[1].plot(t, result_df["nlp_norm_surprisal"].astype(float).to_numpy())

        axes[1].set_ylabel("NLP\nnorm_surprisal")

    else:

        nlp_cols = [c for c in result_df.columns if c.startswith("nlp_") and result_df[c].dtype != object]

        if nlp_cols:

            axes[1].plot(t, result_df[nlp_cols[0]].astype(float).to_numpy())

            axes[1].set_ylabel(nlp_cols[0])

        else:

            axes[1].axis("off")

    axes[2].step(t, result_df["pid_nlp_local_state"].astype(int).to_numpy(), where="post")

    axes[2].set_ylabel("PID-NLP\nlocal")

    axes[3].step(t, result_df["is_teacher_frame_used"].astype(int).to_numpy(), where="post")

    axes[3].set_ylabel("teacher\nused")

    axes[3].set_xlabel("time_sec")

    axes[0].set_title("NLP-first single-video PID-T2S local states")

    for ax in axes:

        ax.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(out_png, dpi=220, bbox_inches="tight")

    plt.close()

def get_bgr_for_state(state: int):

    palette = [

        (255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 160, 0),

        (160, 0, 255), (0, 180, 180), (180, 120, 0), (120, 0, 180),

        (60, 160, 220), (220, 60, 100), (100, 100, 100), (20, 180, 80),

        (180, 20, 80), (80, 20, 180),

    ]

    if int(state) < 0:

        return (160, 160, 160)

    r, g, b = palette[int(state) % len(palette)]

    return (b, g, r)

def shorten_text(text: str, max_chars=70) -> str:

    text = str(text).replace("\n", " ").replace("\t", " ").strip()

    text = re.sub(r"\s+", " ", text)

    return text if len(text) <= max_chars else text[:max_chars] + "..."

def build_timeline_strip(label_seq, width=1600, height=26):

    seq = np.asarray(label_seq, dtype=int)

    strip = np.ones((height, width, 3), dtype=np.uint8) * 255

    n = len(seq)

    if n == 0:

        return strip

    start = 0

    for i in range(1, n + 1):

        if i == n or seq[i] != seq[start]:

            s, e = start, i - 1

            x1 = int(round(s / n * width))

            x2 = max(x1, int(round((e + 1) / n * width)) - 1)

            cv2.rectangle(strip, (x1, 0), (x2, height - 1), get_bgr_for_state(seq[start]), -1)

            start = i

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)

    return strip

def draw_timeline_on_frame(frame, timeline_strip, idx, total_len):

    h, w, _ = frame.shape

    x1, x2 = 30, w - 30

    y2, y1 = h - 25, h - 25 - 78

    overlay = frame.copy()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)

    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, "NLP-first local PID-T2S timeline", (x1 + 10, y1 + 22),

                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    strip_w = max(10, x2 - x1 - 20)

    resized = cv2.resize(timeline_strip, (strip_w, 26), interpolation=cv2.INTER_NEAREST)

    sy, sx = y1 + 38, x1 + 10

    frame[sy:sy + 26, sx:sx + strip_w] = resized

    px = sx + int(round(idx / max(total_len - 1, 1) * (strip_w - 1)))

    cv2.line(frame, (px, sy - 3), (px, sy + 30), (0, 0, 0), 2)

def render_video_overlay(result_df: pd.DataFrame, video_path: Path, out_dir: Path) -> Path:

    out_silent = out_dir / f"{safe_filename(video_path.stem)}_nlp_first_local_overlay_silent.mp4"

    out_audio = out_dir / f"{safe_filename(video_path.stem)}_nlp_first_local_overlay_with_audio.mp4"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_silent), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    times = result_df["time_sec"].astype(float).to_numpy()

    local_states = result_df["pid_nlp_local_state"].astype(int).to_numpy()

    timeline_strip = build_timeline_strip(local_states, width=1600, height=26)

    norm_sur_col = "nlp_norm_surprisal" if "nlp_norm_surprisal" in result_df.columns else None

    total_rows = len(result_df)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        t = frame_idx / float(fps)

        idx = int(np.searchsorted(times, t, side="left"))

        idx = min(max(idx, 0), total_rows - 1)

        r = result_df.iloc[idx]

        state = int(r["pid_nlp_local_state"])

        layer1_state = int(r["meta_state"])

        color = get_bgr_for_state(state)

        cv2.rectangle(frame, (25, 25), (min(width - 25, 1040), 155), (255, 255, 255), -1)

        cv2.rectangle(frame, (25, 25), (min(width - 25, 1040), 155), color, 4)

        cv2.putText(frame, f"NLP-first local state: {state}", (45, 62),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Layer1 local meta_state: {layer1_state}", (45, 95),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)

        if norm_sur_col is not None:

            cv2.putText(frame, f"NLP norm_surprisal: {float(r.get(norm_sur_col, 0.0)):.4f}", (45, 126),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)

        text = shorten_text(r.get("nlp_text_joined", ""), max_chars=85)

        if text:

            cv2.putText(frame, f"NLP text: {text}", (45, 150),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

        draw_timeline_on_frame(frame, timeline_strip, idx, total_rows)

        writer.write(frame)

        frame_idx += 1

        if frame_idx % 300 == 0:

            print(f"    video overlay frames: {frame_idx}")

    cap.release()

    writer.release()

    if HAS_MOVIEPY:

        try:

            print("    合并原视频音频...")

            src = VideoFileClip(str(video_path))

            dst = VideoFileClip(str(out_silent))

            if src.audio is not None:

                dst = dst.with_audio(src.audio)

                dst.write_videofile(str(out_audio), codec="libx264", audio_codec="aac")

                try:

                    src.close()

                    dst.close()

                except Exception:

                    pass

                return out_audio

        except Exception as e:

            print(f"    [提示] 音频合并失败，保留无声视频: {e}")

    return out_silent

                                                           

                         

                                                           

def process_one_case(case_dir: Path, generate_video=False) -> dict:

    video_id = norm_video_id(str(case_dir.relative_to(T2S_OUTPUT_ROOT)).replace("\\", "/"))

    out_dir = PER_VIDEO_ROOT / Path(*video_id.split("/"))

    ensure_dir(out_dir)

    print("=" * 90)

    print(f"单视频 NLP-first 聚类: {video_id}")

    print(f"case_dir: {case_dir}")

    print(f"out_dir : {out_dir}")

    frame_df = pd.read_csv(case_dir / "multiscale_t2s_with_meta.csv")

    if "time_sec" not in frame_df.columns or "meta_state" not in frame_df.columns:

        raise ValueError("multiscale_t2s_with_meta.csv 缺少 time_sec/meta_state")

    frame_df["time_sec"] = pd.to_numeric(frame_df["time_sec"], errors="coerce")

    frame_df["meta_state"] = pd.to_numeric(frame_df["meta_state"], errors="coerce").fillna(-1).astype(int)

    frame_df = frame_df.dropna(subset=["time_sec"]).reset_index(drop=True)

    if "is_teacher_frame" in frame_df.columns:

        teacher_mask = pd.to_numeric(frame_df["is_teacher_frame"], errors="coerce").fillna(0).astype(int).eq(1)

    else:

        teacher_mask = pd.Series(True, index=frame_df.index)

    teacher_mask &= frame_df["meta_state"].ge(0)

    frame_df["is_teacher_frame_used"] = teacher_mask.astype(int)

    if int(teacher_mask.sum()) < max(MULTI_WINS) + 10:

        raise RuntimeError(f"教师有效帧太少，无法运行 NLP-first PID-T2S: {int(teacher_mask.sum())} frames")

    nlp_df = load_individual_or_mode6_nlp(video_id)

    nlp_aligned = align_nlp_to_frames(frame_df["time_sec"].astype(float).to_numpy(), nlp_df)

    X_full = build_combined_features(frame_df, nlp_aligned)

    X_teacher = X_full.loc[teacher_mask].reset_index(drop=True)

    X_full.to_csv(out_dir / "combined_nlp_first_features_full.csv", index=False, encoding="utf-8-sig")

    X_teacher.to_csv(out_dir / "combined_nlp_first_features_teacher_only.csv", index=False, encoding="utf-8-sig")

    final_teacher_seq, branch_metrics_df, raw_branch_df, k_sweep_df, best_k = run_pid_t2s_on_combined_features(X_teacher, out_dir)

    final_full_seq = fill_teacher_sequence_to_full(final_teacher_seq, teacher_mask.to_numpy(), skip_value=-1)

    result_df = frame_df.copy()

    for c in nlp_aligned.columns:

        if c != "time_sec":

            result_df[c] = nlp_aligned[c].values

    result_df["pid_nlp_local_state"] = final_full_seq

    result_csv = out_dir / "nlp_first_local_pid_t2s_final.csv"

    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    branch_metrics_df.to_csv(out_dir / "branch_metrics_nlp_first_pid.csv", index=False, encoding="utf-8-sig")

    raw_branch_df.to_csv(out_dir / "raw_branch_matrix_nlp_first_teacher_only.csv", index=False, encoding="utf-8-sig")

    k_sweep_df.to_csv(out_dir / "local_nlp_first_k_sweep_scores.csv", index=False, encoding="utf-8-sig")

    timeline_png = out_dir / "timeline_nlp_first_local_pid_t2s.png"

    plot_single_video_timeline(result_df, timeline_png)

    overlay_path = ""

    if generate_video:

        video_path = find_video_file(video_id)

        if video_path is None:

            print(f"  [提示] 找不到原视频，跳过视频叠加: {video_id}")

        else:

            overlay_path = str(render_video_overlay(result_df, video_path, out_dir))

    cfg = {

        "video_id": video_id,

        "case_dir": str(case_dir),

        "result_csv": str(result_csv),

        "timeline_png": str(timeline_png),

        "overlay_video": overlay_path,

        "n_frames": int(len(result_df)),

        "n_teacher_used": int(teacher_mask.sum()),

        "n_features": int(X_teacher.shape[1]),

        "best_local_K": int(best_k),

        "nlp_source": str(nlp_df["__nlp_source_file"].iloc[0]) if "__nlp_source_file" in nlp_df.columns else "",

        "T2S_USE_CUDA": bool(T2S_USE_CUDA),

    }

    with open(out_dir / "run_config_nlp_first_local_pid_t2s.json", "w", encoding="utf-8") as f:

        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"完成单视频: {video_id}")

    print(f"  result_csv  : {result_csv}")

    print(f"  timeline_png: {timeline_png}")

    if overlay_path:

        print(f"  overlay     : {overlay_path}")

    return {

        "video_id": video_id,

        "status": "ok",

        "case_dir": str(case_dir),

        "per_video_dir": str(out_dir),

        "result_csv": str(result_csv),

        "timeline_png": str(timeline_png),

        "overlay_video": overlay_path,

        "n_frames": int(len(result_df)),

        "n_teacher_used": int(teacher_mask.sum()),

        "n_features": int(X_teacher.shape[1]),

        "best_local_K": int(best_k),

        "error": "",

    }

                                                           

                         

                                                           

def split_state_segments(state_seq: Iterable[int]) -> list[tuple[int, int, int]]:

    seq = np.asarray(state_seq).astype(int)

    if len(seq) == 0:

        return []

    segs = []

    s = 0

    for i in range(1, len(seq)):

        if seq[i] != seq[s]:

            segs.append((s, i - 1, int(seq[s])))

            s = i

    segs.append((s, len(seq) - 1, int(seq[s])))

    return segs

def collect_successful_per_video_dirs(manifest_df: pd.DataFrame) -> list[Path]:

    dirs = []

    for _, r in manifest_df.iterrows():

        if str(r.get("status", "")) != "ok":

            continue

        d = Path(str(r.get("per_video_dir", "")))

        if (d / "nlp_first_local_pid_t2s_final.csv").exists():

            dirs.append(d)

    return sorted(set(dirs))

def choose_prototype_feature_columns(proto_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

    id_cols = {

        "video_id",

        "per_video_dir",

        "local_pid_nlp_state",

        "cluster_type",

    }

    candidate_cols = [c for c in proto_df.columns if c not in id_cols]

    numeric = pd.DataFrame(index=proto_df.index)

    feature_cols = []

    for c in candidate_cols:

        s = pd.to_numeric(proto_df[c], errors="coerce")

        if s.notna().sum() == 0:

            continue

        if s.dropna().nunique() <= 1:

            continue

        numeric[c] = s

        feature_cols.append(c)

    if not feature_cols:

        raise ValueError("没有可用于跨视频聚合的 prototype 数值特征")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    med = numeric.median(numeric_only=True)

    numeric = numeric.fillna(med).fillna(0.0)

    pd.DataFrame({"feature_col": feature_cols}).to_csv(

        CROSS_VIDEO_ROOT / "feature_cols_used.csv",

        index=False,

        encoding="utf-8-sig",

    )

    return numeric, feature_cols

def extract_single_video_prototypes(per_video_dirs: list[Path]) -> pd.DataFrame:

    rows = []

    for d in per_video_dirs:

        result_csv = d / "nlp_first_local_pid_t2s_final.csv"

        raw_branch_csv = d / "raw_branch_matrix_nlp_first_teacher_only.csv"

        result_df = pd.read_csv(result_csv)

        raw_branch_df = pd.read_csv(raw_branch_csv) if raw_branch_csv.exists() else None

        video_id = norm_video_id(str(d.relative_to(PER_VIDEO_ROOT)).replace("\\", "/"))

        print(f"提取 prototype: {video_id}")

        if "pid_nlp_local_state" not in result_df.columns:

            print(f"  [跳过] 缺少 pid_nlp_local_state: {result_csv}")

            continue

        if "is_teacher_frame_used" in result_df.columns:

            keep = pd.to_numeric(result_df["is_teacher_frame_used"], errors="coerce").fillna(0).astype(int).eq(1)

        else:

            keep = pd.Series(True, index=result_df.index)

        keep &= pd.to_numeric(result_df["pid_nlp_local_state"], errors="coerce").fillna(-1).astype(int).ge(0)

        if int(keep.sum()) <= 0:

            print(f"  [跳过] 无有效教师帧: {video_id}")

            continue

        df = result_df.loc[keep].reset_index(drop=True)

        state_seq = pd.to_numeric(df["pid_nlp_local_state"], errors="coerce").fillna(-1).astype(int).to_numpy()

        all_segs = split_state_segments(state_seq)

        n_total = len(df)

        raw_teacher = None

        raw_cols = []

        if raw_branch_df is not None and len(raw_branch_df) == n_total:

            raw_teacher = raw_branch_df.copy()

            raw_cols = [c for c in raw_teacher.columns if c.startswith("run") and c.endswith("_state")]

        elif raw_branch_df is not None:

            print(f"  [提示] raw branch 行数不匹配，跳过 branch signature: raw={len(raw_branch_df)}, teacher={n_total}")

        nlp_num_cols = [

            c for c in df.columns

            if c.startswith("nlp_") and c not in {"nlp_text_joined", "nlp_window_start", "nlp_window_end", "nlp_window_center"}

        ]

        meta_ratio_cols = [c for c in df.columns if str(c).startswith("meta_ratio_")]

        for st in sorted(pd.unique(pd.Series(state_seq).astype(int))):

            if int(st) < 0:

                continue

            idx = np.where(state_seq == int(st))[0]

            if len(idx) == 0:

                continue

            seg_lens = [e - s + 1 for s, e, lab in all_segs if lab == int(st)]

            seg_lens = np.asarray(seg_lens, dtype=float) if seg_lens else np.asarray([], dtype=float)

            row = {

                "video_id": video_id,

                "per_video_dir": str(d),

                "local_pid_nlp_state": int(st),

                "num_frames": int(len(idx)),

                "state_ratio": float(len(idx) / max(1, n_total)),

                "seg_count": int(len(seg_lens)),

                "seg_len_mean": float(seg_lens.mean()) if len(seg_lens) else 0.0,

                "seg_len_std": float(seg_lens.std(ddof=0)) if len(seg_lens) else 0.0,

                "seg_len_max": float(seg_lens.max()) if len(seg_lens) else 0.0,

            }

                                          

            if "meta_state" in df.columns:

                meta_sub = pd.to_numeric(df.iloc[idx]["meta_state"], errors="coerce").dropna().astype(int).to_numpy()

                if len(meta_sub):

                    vals, counts = np.unique(meta_sub, return_counts=True)

                    probs = counts.astype(float) / counts.sum()

                    row["l1_meta_unique_states"] = int(len(vals))

                    row["l1_meta_dominant_ratio"] = float(probs.max())

                    row["l1_meta_entropy"] = float(safe_normalized_entropy(meta_sub))

                    for mv, pr in zip(vals, probs):

                        row[f"l1_meta_ratio_state_{int(mv)}"] = float(pr)

                           

            for c in meta_ratio_cols:

                s = pd.to_numeric(df.iloc[idx][c], errors="coerce")

                row[f"{c}_mean"] = float(s.mean()) if s.notna().any() else 0.0

                row[f"{c}_std"] = float(s.std(ddof=0)) if s.notna().any() else 0.0

                        

            for c in nlp_num_cols:

                s = pd.to_numeric(df.iloc[idx][c], errors="coerce")

                row[f"{c}_mean"] = float(s.mean()) if s.notna().any() else 0.0

                row[f"{c}_std"] = float(s.std(ddof=0)) if s.notna().any() else 0.0

                                      

            if raw_teacher is not None and raw_cols:

                for c in raw_cols:

                    run_sub = pd.to_numeric(raw_teacher.iloc[idx][c], errors="coerce").dropna().astype(int).to_numpy()

                    run_sub = run_sub[run_sub >= 0]

                    prefix = c.replace("_state", "")

                    if len(run_sub) == 0:

                        row[f"{prefix}_unique_states"] = np.nan

                        row[f"{prefix}_dominant_ratio"] = np.nan

                        row[f"{prefix}_entropy"] = np.nan

                    else:

                        vals, counts = np.unique(run_sub, return_counts=True)

                        probs = counts.astype(float) / counts.sum()

                        row[f"{prefix}_unique_states"] = int(len(vals))

                        row[f"{prefix}_dominant_ratio"] = float(probs.max())

                        row[f"{prefix}_entropy"] = float(safe_normalized_entropy(run_sub))

            rows.append(row)

    proto_df = pd.DataFrame(rows)

    if len(proto_df) == 0:

        raise ValueError("没有成功提取任何 NLP-first local prototype")

    out_csv = CROSS_VIDEO_ROOT / "local_nlp_first_state_prototypes.csv"

    proto_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"已输出: {out_csv}")

    return proto_df

def choose_global_k_and_fit(proto_df: pd.DataFrame, force_global_k: int = 0):

    X, feature_cols = choose_prototype_feature_columns(proto_df)

    X_std = StandardScaler().fit_transform(X.values.astype(np.float32))

    n = len(proto_df)

    if n < 3:

        raise ValueError(f"prototype 数量太少，无法做跨视频聚合: n={n}")

    k_max = min(GLOBAL_K_MAX, n - 1)

    k_min = min(GLOBAL_K_MIN, k_max)

    if int(force_global_k) > 0:

        best_k = int(force_global_k)

        if best_k < 2 or best_k >= n:

            raise ValueError(f"force_global_k={best_k} 不合法，prototype n={n}")

        score_df = pd.DataFrame([{"K": best_k, "silhouette_cosine": np.nan, "forced": 1}])

    else:

        ks = list(range(k_min, k_max + 1))

        score_rows = []

        best_k = None

        best_score = -1e18

        for k in ks:

            try:

                try:

                    clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")

                except TypeError:

                    clusterer = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")

                labels = clusterer.fit_predict(X_std)

                sil = -1.0 if len(np.unique(labels)) < 2 else silhouette_score(X_std, labels, metric="cosine")

                score_rows.append({"K": int(k), "silhouette_cosine": float(sil), "forced": 0})

                if sil > best_score:

                    best_score = float(sil)

                    best_k = int(k)

            except Exception as e:

                score_rows.append({"K": int(k), "silhouette_cosine": np.nan, "forced": 0, "error": repr(e)})

        score_df = pd.DataFrame(score_rows)

        if best_k is None:

            raise ValueError("无法自动选择 global K")

    score_df.to_csv(CROSS_VIDEO_ROOT / "global_k_sweep_scores.csv", index=False, encoding="utf-8-sig")

    print(f"跨视频 global K = {best_k}")

    try:

        clusterer = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")

    except TypeError:

        clusterer = AgglomerativeClustering(n_clusters=best_k, affinity="cosine", linkage="average")

    labels = clusterer.fit_predict(X_std).astype(int)

    labels, _ = remap_to_contiguous_labels(labels)

    out_df = proto_df.copy()

    out_df["global_pid_nlp_proto_id"] = labels

    centers = {}

    for g in sorted(pd.unique(labels)):

        idx = np.where(labels == g)[0]

        centers[int(g)] = X_std[idx].mean(axis=0)

    dists = []

    for i, g in enumerate(labels):

        dists.append(float(np.linalg.norm(X_std[i] - centers[int(g)])))

    out_df["dist_to_cluster_center"] = dists

    return out_df, X_std, feature_cols, best_k, score_df

def summarize_global_prototypes(aligned_df: pd.DataFrame) -> pd.DataFrame:

    summary = aligned_df.groupby("global_pid_nlp_proto_id").agg(

        support_local_states=("local_pid_nlp_state", "count"),

        support_videos=("video_id", "nunique"),

        mean_state_ratio=("state_ratio", "mean"),

        mean_num_frames=("num_frames", "mean"),

        mean_seg_count=("seg_count", "mean"),

        mean_seg_len=("seg_len_mean", "mean"),

        mean_dist_to_center=("dist_to_cluster_center", "mean"),

    ).reset_index()

    def typ(nv: int) -> str:

        if nv >= 5:

            return "shared"

        if nv >= 2:

            return "variant"

        return "private"

    summary["cluster_type"] = summary["support_videos"].apply(lambda x: typ(int(x)))

    aligned_df = aligned_df.merge(

        summary[["global_pid_nlp_proto_id", "support_videos", "cluster_type"]],

        on="global_pid_nlp_proto_id",

        how="left",

    )

    aligned_df.to_csv(CROSS_VIDEO_ROOT / "local_to_global_nlp_first_pid_proto.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(CROSS_VIDEO_ROOT / "global_pid_nlp_proto_summary.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {CROSS_VIDEO_ROOT / 'local_to_global_nlp_first_pid_proto.csv'}")

    print(f"已输出: {CROSS_VIDEO_ROOT / 'global_pid_nlp_proto_summary.csv'}")

    return summary

def plot_cross_video_pca(aligned_df: pd.DataFrame, X_std: np.ndarray):

    ensure_dir(PLOT_DIR)

    pca = PCA(n_components=2, random_state=42)

    X2 = pca.fit_transform(X_std)

    plot_df = aligned_df.copy()

    plot_df["pca_x"] = X2[:, 0]

    plot_df["pca_y"] = X2[:, 1]

    plt.figure(figsize=(12, 9))

    sc = plt.scatter(

        plot_df["pca_x"],

        plot_df["pca_y"],

        c=plot_df["global_pid_nlp_proto_id"],

        cmap="tab20",

        s=55,

        alpha=0.85,

    )

    plt.colorbar(sc, label="global_pid_nlp_proto_id")

    plt.xlabel("PCA-1")

    plt.ylabel("PCA-2")

    plt.title("NLP-first cross-video prototype alignment")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "nlp_first_cross_video_pca.png", dpi=220, bbox_inches="tight")

    plt.close()

    plt.figure(figsize=(14, 10))

    sc = plt.scatter(

        plot_df["pca_x"],

        plot_df["pca_y"],

        c=plot_df["global_pid_nlp_proto_id"],

        cmap="tab20",

        s=60,

        alpha=0.9,

    )

    plt.colorbar(sc, label="global_pid_nlp_proto_id")

    if len(plot_df) <= MAX_LABEL_POINTS:

        label_idx = range(len(plot_df))

    else:

        label_idx = []

        for _g, sub in plot_df.groupby("global_pid_nlp_proto_id"):

            sub = sub.sort_values("dist_to_cluster_center").head(6)

            label_idx.extend(sub.index.tolist())

    for i in label_idx:

        r = plot_df.loc[i]

        txt = f"{tail_video_id(r['video_id'])}-s{int(r['local_pid_nlp_state'])}"

        plt.text(r["pca_x"], r["pca_y"], txt, fontsize=7)

    plt.xlabel("PCA-1")

    plt.ylabel("PCA-2")

    plt.title("NLP-first cross-video prototype alignment: labeled PCA")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "nlp_first_cross_video_pca_labeled.png", dpi=220, bbox_inches="tight")

    plt.close()

    plot_df.to_csv(CROSS_VIDEO_ROOT / "nlp_first_cross_video_pca_coordinates.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {PLOT_DIR / 'nlp_first_cross_video_pca.png'}")

def plot_cross_video_cluster_sizes(summary_df: pd.DataFrame):

    ensure_dir(PLOT_DIR)

    x = summary_df["global_pid_nlp_proto_id"].astype(int).values

    y = summary_df["support_local_states"].astype(int).values

    plt.figure(figsize=(11, 6))

    plt.bar(x, y)

    for xi, yi in zip(x, y):

        plt.text(xi, yi + 0.1, str(int(yi)), ha="center", va="bottom", fontsize=8)

    plt.xlabel("global_pid_nlp_proto_id")

    plt.ylabel("support_local_states")

    plt.title("NLP-first cross-video global prototype sizes")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "nlp_first_cross_video_cluster_sizes.png", dpi=220, bbox_inches="tight")

    plt.close()

def write_frame_level_global_results(aligned_df: pd.DataFrame):

    

       

    for video_id, sub in aligned_df.groupby("video_id"):

        out_dir = PER_VIDEO_ROOT / Path(*norm_video_id(video_id).split("/"))

        result_csv = out_dir / "nlp_first_local_pid_t2s_final.csv"

        if not result_csv.exists():

            continue

        df = pd.read_csv(result_csv)

        mapping = {}

        for _, r in sub.iterrows():

            mapping[int(r["local_pid_nlp_state"])] = int(r["global_pid_nlp_proto_id"])

        df["global_pid_nlp_proto_id"] = (

            pd.to_numeric(df["pid_nlp_local_state"], errors="coerce")

            .fillna(-1).astype(int)

            .map(mapping).fillna(-1).astype(int)

        )

        out_csv = out_dir / "nlp_first_local_with_global_proto.csv"

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

                                

        try:

            t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy()

            fig, axes = plt.subplots(3, 1, figsize=(18, 7), sharex=True)

            axes[0].step(t, df["pid_nlp_local_state"].astype(int), where="post")

            axes[0].set_ylabel("local\nPID-NLP")

            axes[1].step(t, df["global_pid_nlp_proto_id"].astype(int), where="post")

            axes[1].set_ylabel("global\nproto")

            if "nlp_norm_surprisal" in df.columns:

                axes[2].plot(t, df["nlp_norm_surprisal"].astype(float))

                axes[2].set_ylabel("NLP\nsurprisal")

            else:

                axes[2].axis("off")

            axes[2].set_xlabel("time_sec")

            for ax in axes:

                ax.grid(alpha=0.25)

            plt.tight_layout()

            plt.savefig(out_dir / "timeline_nlp_first_local_and_global.png", dpi=220, bbox_inches="tight")

            plt.close()

        except Exception as e:

            print(f"[提示] 写回 {video_id} 的 global timeline 图失败: {e}")

def run_cross_video_aggregation(manifest_df: pd.DataFrame, force_global_k: int = 0) -> dict:

    ensure_dir(CROSS_VIDEO_ROOT)

    ensure_dir(PLOT_DIR)

    per_video_dirs = collect_successful_per_video_dirs(manifest_df)

    if not per_video_dirs:

        raise ValueError("没有成功的 per-video 输出，无法跨视频聚合")

    print("\n" + "=" * 90)

    print("开始跨视频聚合：NLP-first local states -> global_pid_nlp_proto_id")

    print(f"成功视频数: {len(per_video_dirs)}")

    proto_df = extract_single_video_prototypes(per_video_dirs)

    aligned_df, X_std, feature_cols, best_k, score_df = choose_global_k_and_fit(proto_df, force_global_k=force_global_k)

    summary_df = summarize_global_prototypes(aligned_df)

    plot_cross_video_pca(aligned_df, X_std)

    plot_cross_video_cluster_sizes(summary_df)

    write_frame_level_global_results(aligned_df)

    cfg = {

        "OUTPUT_ROOT": str(OUTPUT_ROOT),

        "PER_VIDEO_ROOT": str(PER_VIDEO_ROOT),

        "CROSS_VIDEO_ROOT": str(CROSS_VIDEO_ROOT),

        "GLOBAL_K_MIN": int(GLOBAL_K_MIN),

        "GLOBAL_K_MAX": int(GLOBAL_K_MAX),

        "FORCE_GLOBAL_K": int(force_global_k),

        "selected_global_K": int(best_k),

        "feature_cols": feature_cols,

    }

    with open(CROSS_VIDEO_ROOT / "run_config_cross_video_nlp_first.json", "w", encoding="utf-8") as f:

        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return {

        "status": "ok",

        "n_videos": int(len(per_video_dirs)),

        "n_prototypes": int(len(proto_df)),

        "selected_global_K": int(best_k),

        "summary_csv": str(CROSS_VIDEO_ROOT / "global_pid_nlp_proto_summary.csv"),

    }

                                                           

         

                                                           

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video-id", default="all", help="all 或某个视频编号，例如 21；兼容 21/21。")

    parser.add_argument("--no-video", action="store_true", help="只生成 CSV/PNG，不生成视频 overlay。")

    parser.add_argument("--skip-per-video", action="store_true", help="跳过单视频阶段，直接使用已有 per_video 输出做跨视频聚合。")

    parser.add_argument("--skip-cross-video", action="store_true", help="只跑单视频阶段，不做跨视频聚合。")

    parser.add_argument("--force-global-k", type=int, default=0, help="强制跨视频 global K；0 表示自动 silhouette 选 K。")

    args = parser.parse_args()

    global FORCE_GLOBAL_K

    FORCE_GLOBAL_K = int(args.force_global_k)

    ensure_dir(OUTPUT_ROOT)

    ensure_dir(PER_VIDEO_ROOT)

    ensure_dir(CROSS_VIDEO_ROOT)

    ensure_dir(PLOT_DIR)

    manifest_rows = []

    if not args.skip_per_video:

        case_dirs = find_all_case_dirs()

        if not case_dirs:

            raise FileNotFoundError(f"找不到第一阶段 multiscale_t2s_with_meta.csv: {T2S_OUTPUT_ROOT}")

        if args.video_id.lower() != "all":

            target = norm_video_id(args.video_id)

            case_dirs = [

                p for p in case_dirs

                if video_id_match(str(p.relative_to(T2S_OUTPUT_ROOT)).replace("\\", "/"), target)

            ]

            if not case_dirs:

                raise FileNotFoundError(f"找不到 video_id={args.video_id} 对应的 case_dir")

        print(f"待处理单视频 case 数: {len(case_dirs)}")

        for case_dir in case_dirs:

            try:

                manifest_rows.append(process_one_case(case_dir, generate_video=(not args.no_video)))

            except Exception as e:

                rel = str(case_dir.relative_to(T2S_OUTPUT_ROOT)).replace("\\", "/")

                print(f"[失败] {rel}: {e}")

                manifest_rows.append({

                    "video_id": rel,

                    "status": "failed",

                    "case_dir": str(case_dir),

                    "per_video_dir": "",

                    "result_csv": "",

                    "timeline_png": "",

                    "overlay_video": "",

                    "n_frames": 0,

                    "n_teacher_used": 0,

                    "n_features": 0,

                    "best_local_K": 0,

                    "error": repr(e),

                })

        manifest_df = pd.DataFrame(manifest_rows)

        manifest_csv = OUTPUT_ROOT / "ALL_processing_manifest_nlp_first_local.csv"

        manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

        print(f"\n单视频阶段完成，manifest: {manifest_csv}")

    else:

        manifest_csv = OUTPUT_ROOT / "ALL_processing_manifest_nlp_first_local.csv"

        if not manifest_csv.exists():

            raise FileNotFoundError(f"找不到已有单视频 manifest: {manifest_csv}")

        manifest_df = pd.read_csv(manifest_csv)

        print(f"读取已有单视频 manifest: {manifest_csv}")

    if not args.skip_cross_video:

        try:

            cross_info = run_cross_video_aggregation(manifest_df, force_global_k=int(args.force_global_k))

            with open(CROSS_VIDEO_ROOT / "cross_video_done.json", "w", encoding="utf-8") as f:

                json.dump(cross_info, f, ensure_ascii=False, indent=2)

        except Exception as e:

            print(f"[跨视频聚合失败] {e}")

            raise

    print("\n全部处理结束。")

    print(f"总输出目录: {OUTPUT_ROOT}")

    print("单视频重点看：")

    print("  PER_VIDEO_ROOT/<video>/<video>/nlp_first_local_pid_t2s_final.csv")

    print("  PER_VIDEO_ROOT/<video>/<video>/branch_metrics_nlp_first_pid.csv")

    print("  PER_VIDEO_ROOT/<video>/<video>/local_nlp_first_k_sweep_scores.csv")

    print("  PER_VIDEO_ROOT/<video>/<video>/timeline_nlp_first_local_pid_t2s.png")

    print("跨视频重点看：")

    print("  _cross_video_pid_nlp_proto/local_nlp_first_state_prototypes.csv")

    print("  _cross_video_pid_nlp_proto/local_to_global_nlp_first_pid_proto.csv")

    print("  _cross_video_pid_nlp_proto/global_pid_nlp_proto_summary.csv")

    print("  _cross_video_pid_nlp_proto/global_k_sweep_scores.csv")

    print("  _cross_video_pid_nlp_proto/plots/nlp_first_cross_video_pca.png")

if __name__ == "__main__":

    main()
