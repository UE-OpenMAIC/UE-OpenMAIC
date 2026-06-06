import copy

import cv2

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from TSpy.view import plot_mts

import matplotlib.pyplot as plt

from moviepy import VideoFileClip

from Time2State.time2state import Time2State

from Time2State.adapers import CausalConv_LSE_Adaper

from Time2State.clustering import DPGMM

from Time2State.default_params import *

                           

       

                           

AUDIO_CSV = r"D:\code\teacherT2S\yolo\pose_12fps_with_audio.csv"

VISUAL_CSV = r"D:\code\teacherT2S\yolo\teacher_visual_15d.csv"

INPUT_VIDEO = r"D:\code\teacherT2S\yolo\input.mp4"

OUTPUT_CSV = r"D:\code\teacherT2S\yolo\schemeC_stable_audio_move_fusion_with_state.csv"

OUTPUT_ACTION_PNG = r"D:\code\teacherT2S\yolo\schemeC_action_branch_t2s.png"

OUTPUT_FUSION_PNG = r"D:\code\teacherT2S\yolo\schemeC_fusion_branch_t2s.png"

OUTPUT_AUX_PNG = r"D:\code\teacherT2S\yolo\schemeC_audio_move_states.png"

TEMP_VIDEO = r"D:\code\teacherT2S\yolo\schemeC_fusion_overlay_silent.mp4"

OUTPUT_VIDEO = r"D:\code\teacherT2S\yolo\schemeC_fusion_overlay_with_audio.mp4"

TIME_COL = "time_sec"

USE_NORMALIZED_COORDS = True

                           

       

                           

                     

ACTION_WIN = 96

ACTION_STEP = 24

ACTION_OUT_CHANNELS = 8

ACTION_MIN_LEN = 36        

         

FUSION_WIN = 96

FUSION_STEP = 24

FUSION_OUT_CHANNELS = 6

FUSION_MIN_LEN = 48        

    

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

             

RATIO_WIN = 24             

                           

          

                           

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

]

TIMELINE_BAR_HEIGHT = 22

TIMELINE_MARGIN = 12

TIMELINE_BOX_HEIGHT = 48

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

def rolling_ratio_from_state(state_seq, num_classes, win):

    onehot = one_hot_from_state(state_seq, num_classes)

    df = pd.DataFrame(onehot)

    df = df.rolling(win, center=True, min_periods=1).mean()

    return df.to_numpy(dtype=np.float32)

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

def draw_timeline_on_frame(frame, timeline_strip, idx, total_len, state_color, cur_state):

    h, w, _ = frame.shape

    box_x1 = TIMELINE_MARGIN

    box_x2 = w - TIMELINE_MARGIN

    box_y2 = h - TIMELINE_MARGIN

    box_y1 = box_y2 - TIMELINE_BOX_HEIGHT

    bar_x1 = box_x1 + 8

    bar_x2 = box_x2 - 8

    bar_y1 = box_y1 + 16

    bar_y2 = bar_y1 + TIMELINE_BAR_HEIGHT

    overlay = frame.copy()

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

    frame[:] = cv2.addWeighted(overlay, TIMELINE_BG_ALPHA, frame, 1 - TIMELINE_BG_ALPHA, 0)

    target_w = bar_x2 - bar_x1

    target_h = bar_y2 - bar_y1

    strip_resized = cv2.resize(timeline_strip, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    frame[bar_y1:bar_y2, bar_x1:bar_x2] = strip_resized

    if total_len > 1:

        x_cur = bar_x1 + int(round(idx / (total_len - 1) * (target_w - 1)))

    else:

        x_cur = bar_x1

    cv2.line(frame, (x_cur, bar_y1 - 4), (x_cur, bar_y2 + 4), (255, 255, 255), 2)

    cv2.line(frame, (x_cur, bar_y1 - 4), (x_cur, bar_y2 + 4), (0, 0, 0), 1)

    cv2.putText(

        frame,

        "fusion state timeline",

        (bar_x1, box_y1 + 12),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (30, 30, 30),

        1

    )

    cv2.putText(

        frame,

        f"S{cur_state}",

        (bar_x2 - 38, box_y1 + 12),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        state_color,

        1

    )

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

                or

                (voiced[i] >= 0.45 and rms[i] >= exit_rms)

            ):

                cur = 1

        else:

            if (

                (voiced[i] <= AUDIO_EXIT_VOICED and silence[i] >= AUDIO_EXIT_SILENCE and rms[i] <= exit_rms)

            ):

                cur = 0

        state[i] = cur

               

    state = merge_short_segments(state, min_len=AUDIO_MIN_SEG)

    state = merge_short_segments(state, min_len=AUDIO_MIN_SEG)

    return state, one_hot_from_state(state, 2)

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

    return state, one_hot_from_state(state, 3), dx

                           

                      

                           

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

                           

           

                           

def extract_t2s_embeddings(t2s_obj, branch_name):

    candidates = [

        "embeddings",

        "embedding",

        "embedding_seq",

        "embedding_sequence",

        "embs",

        "emb",

        "E",

        "_embeddings",

    ]

    for name in candidates:

        if hasattr(t2s_obj, name):

            value = getattr(t2s_obj, name)

            if value is not None:

                arr = np.asarray(value)

                if arr.ndim == 2 and arr.shape[0] > 1:

                    print(f"[{branch_name}] 使用 embedding 属性: {name}, shape={arr.shape}")

                    return arr

    emb_like = [name for name in dir(t2s_obj) if "emb" in name.lower()]

    raise RuntimeError(

        f"{branch_name} 分支没有找到可用的 embedding 序列。\n"

        f"带 emb 的属性有: {emb_like}\n"

        f"把这个报错贴给我，我给你按你这份 Time2State 实现改一行。"

    )

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

    state_seq, state_map = remap_to_contiguous_labels(state_seq)

    state_seq = merge_short_segments(state_seq, min_len=min_len)

    state_seq, state_map = remap_to_contiguous_labels(state_seq)

    emb_seq = extract_t2s_embeddings(t2s, branch_name)

    return {

        "data_std": data_std,

        "t2s": t2s,

        "state_seq": state_seq,

        "state_map": state_map,

        "emb_seq": emb_seq,

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

                           

         

                           

audio_df = pd.read_csv(AUDIO_CSV)

visual_df = pd.read_csv(VISUAL_CSV)

if TIME_COL not in audio_df.columns:

    raise ValueError(f"音频 CSV 缺少 {TIME_COL}")

if TIME_COL not in visual_df.columns:

    raise ValueError(f"动作 CSV 缺少 {TIME_COL}")

audio_df = audio_df.sort_values(TIME_COL).reset_index(drop=True)

visual_df = visual_df.sort_values(TIME_COL).reset_index(drop=True)

audio_times = audio_df[TIME_COL].astype(float).values

visual_times = visual_df[TIME_COL].astype(float).values

                           

                 

                           

audio_state, _audio_onehot = build_audio_binary_state(audio_df)

move_state, _move_onehot, move_dx = build_move_direction_state(visual_df)

action_X = build_action_relative_features(visual_df)

print("动作分支输入维度：", list(action_X.columns))

action_branch = run_t2s_branch(

    action_X.to_numpy(dtype=np.float32),

    ACTION_WIN,

    ACTION_STEP,

    ACTION_OUT_CHANNELS,

    "action",

    ACTION_MIN_LEN

)

plot_branch(action_branch["data_std"], action_branch["state_seq"], OUTPUT_ACTION_PNG)

              

audio_idx_for_visual = np.searchsorted(audio_times, visual_times, side="right") - 1

audio_idx_for_visual = np.clip(audio_idx_for_visual, 0, len(audio_state) - 1)

audio_state_aligned = audio_state[audio_idx_for_visual]

                               

audio_ratio_feat = rolling_ratio_from_state(audio_state_aligned, 2, RATIO_WIN)

move_ratio_feat = rolling_ratio_from_state(move_state, 3, RATIO_WIN)

plot_aux_states(visual_times, audio_state_aligned, move_state, OUTPUT_AUX_PNG)

                           

                                    

                           

action_emb = action_branch["emb_seq"]

action_emb_times = np.linspace(visual_times[0], visual_times[-1], len(action_emb))

action_emb_idx = np.searchsorted(action_emb_times, visual_times, side="right") - 1

action_emb_idx = np.clip(action_emb_idx, 0, len(action_emb) - 1)

action_emb_aligned = action_emb[action_emb_idx]

fusion_input = np.concatenate(

    [

        action_emb_aligned.astype(np.float32),

        audio_ratio_feat.astype(np.float32),

        move_ratio_feat.astype(np.float32),

    ],

    axis=1

)

fusion_branch = run_t2s_branch(

    fusion_input,

    FUSION_WIN,

    FUSION_STEP,

    FUSION_OUT_CHANNELS,

    "fusion",

    FUSION_MIN_LEN

)

plot_branch(fusion_branch["data_std"], fusion_branch["state_seq"], OUTPUT_FUSION_PNG)

                           

            

                           

action_state_point = action_branch["state_seq"].copy()

fusion_state_point = fusion_branch["state_seq"].copy()

fusion_state_point = merge_short_segments(fusion_state_point, min_len=FUSION_MIN_LEN)

fusion_state_point, fusion_state_map = remap_to_contiguous_labels(fusion_state_point)

out_df = visual_df.copy()

out_df["audio_speech_state"] = audio_state_aligned

out_df["move_direction_state"] = move_state

out_df["action_t2s_state"] = action_state_point

out_df["fusion_t2s_state"] = fusion_state_point

out_df["move_dx"] = move_dx

out_df["audio_speech_ratio"] = audio_ratio_feat[:, 1]

out_df["audio_silence_ratio"] = audio_ratio_feat[:, 0]

out_df["move_still_ratio"] = move_ratio_feat[:, 0]

out_df["move_left_ratio"] = move_ratio_feat[:, 1]

out_df["move_right_ratio"] = move_ratio_feat[:, 2]

out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"已输出：{OUTPUT_CSV}")

df_vis = out_df

sample_times = visual_times

                           

           

                           

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():

    raise RuntimeError(f"无法打开视频: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:

    raise RuntimeError("视频 FPS 读取失败")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (width, height))

unique_states, counts_states = np.unique(fusion_state_point, return_counts=True)

fusion_state_count_dict = {int(s): int(c) for s, c in zip(unique_states, counts_states)}

timeline_width = width - 2 * TIMELINE_MARGIN - 16

timeline_strip = build_timeline_strip(

    timeline_width,

    fusion_state_point,

    STATE_PALETTE,

    TIMELINE_BAR_HEIGHT

)

frame_id = 0

total_rows = len(df_vis)

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

    idx = np.searchsorted(sample_times, current_time, side="right") - 1

    idx = np.clip(idx, 0, total_rows - 1)

    row = df_vis.iloc[idx]

    fusion_state = int(row["fusion_t2s_state"])

    action_state = int(row["action_t2s_state"])

    audio_state_val = int(row["audio_speech_state"])

    move_state_val = int(row["move_direction_state"])

    state_color = STATE_PALETTE[fusion_state % len(STATE_PALETTE)]

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

    cv2.rectangle(overlay, (20, 20), (760, 270), COLOR_BOX, -1)

    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, f"Fusion state: S{fusion_state}", (35, 55),

                cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

    cv2.putText(frame, f"Action state: S{action_state}", (35, 85),

                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2)

    cv2.putText(frame, f"Audio: {speech_text(audio_state_val)}", (35, 115),

                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2)

    cv2.putText(frame, f"Move: {move_text(move_state_val)}", (35, 145),

                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 60, 60), 2)

    cv2.putText(frame, f"audio_speech_ratio = {float(row['audio_speech_ratio']):.2f}", (35, 175),

                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT, 2)

    cv2.putText(frame, f"move(still/left/right)=({float(row['move_still_ratio']):.2f}, {float(row['move_left_ratio']):.2f}, {float(row['move_right_ratio']):.2f})", (35, 200),

                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1)

    cv2.putText(frame, f"time = {current_time:.2f}s", (35, 225),

                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT, 2)

    cv2.putText(frame,

                f"count(fusion state {fusion_state}) = {fusion_state_count_dict.get(fusion_state, 0)}",

                (35, 250),

                cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_TEXT, 2)

    cv2.rectangle(frame, (width - 120, 30), (width - 40, 90), state_color, -1)

    cv2.rectangle(frame, (width - 120, 30), (width - 40, 90), (0, 0, 0), 2)

    cv2.putText(frame, f"S{fusion_state}", (width - 105, 72),

                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    draw_timeline_on_frame(

        frame,

        timeline_strip,

        idx,

        total_rows,

        state_color,

        fusion_state

    )

    writer.write(frame)

    frame_id += 1

    if frame_id % 200 == 0:

        print(f"已处理 {frame_id} 帧")

cap.release()

writer.release()

print(f"已生成无声叠加视频：{TEMP_VIDEO}")

                           

             

                           

print("开始合并原视频音轨...")

overlay_clip = VideoFileClip(TEMP_VIDEO)

orig_clip = VideoFileClip(INPUT_VIDEO)

try:

    if orig_clip.audio is not None:

        audio_clip = orig_clip.audio.subclipped(0, overlay_clip.duration)

        final_clip = overlay_clip.with_audio(audio_clip)

    else:

        print("警告：原视频没有音轨，将只输出无声视频。")

        final_clip = overlay_clip

    final_clip.write_videofile(

        OUTPUT_VIDEO,

        codec="libx264",

        audio_codec="aac"

    )

finally:

    try:

        overlay_clip.close()

    except:

        pass

    try:

        orig_clip.close()

    except:

        pass

    try:

        final_clip.close()

    except:

        pass

print(f"完成，带声音的视频已保存到：{OUTPUT_VIDEO}")
