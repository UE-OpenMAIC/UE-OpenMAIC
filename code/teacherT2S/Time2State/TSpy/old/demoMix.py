import os

import copy

import cv2

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from moviepy import VideoFileClip

from TSpy.view import plot_mts

from Time2State.time2state import Time2State

from Time2State.adapers import CausalConv_LSE_Adaper

from Time2State.clustering import DPGMM

from Time2State.default_params import *

                           

         

                           

AUDIO_CSV = r"D:\code\teacherT2S\yolo\pose_12fps_with_audio.csv"

VISUAL_CSV = r"D:\code\teacherT2S\yolo\teacher_visual_15d.csv"

INPUT_VIDEO = r"D:\code\teacherT2S\yolo\input.mp4"

                           

                          

                           

OUTPUT_DIR = r"/multiscale_t2s_output"

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(PLOT_DIR, exist_ok=True)

OUTPUT_STATE_MATRIX_CSV = os.path.join(OUTPUT_DIR, "multiscale_t2s_state_matrix.csv")

OUTPUT_RUN_INFO_CSV = os.path.join(OUTPUT_DIR, "multiscale_t2s_run_info.csv")

OUTPUT_AUX_PNG = os.path.join(PLOT_DIR, "audio_move_states.png")

OUTPUT_MULTI_PNG = os.path.join(PLOT_DIR, "multiscale_t2s_states.png")

TEMP_VIDEO = os.path.join(OUTPUT_DIR, "multiscale_t2s_overlay_silent.mp4")

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "multiscale_t2s_overlay_with_audio.mp4")

TIME_COL = "time_sec"

USE_NORMALIZED_COORDS = True

                           

               

                           

                   

MULTI_ACTION_WINS = [72 - 2 * i for i in range(32)]                                 

MULTI_ACTION_STEPS = [max(8, w // 4) for w in MULTI_ACTION_WINS]

ACTION_OUT_CHANNELS = 8

                     

def get_action_min_len(win):

    return max(18, win // 2)

    

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

    fig, axes = plt.subplots(n, 1, figsize=(16, 2 * n), sharex=True)

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

def draw_multi_timelines(frame, timeline_strips, run_infos, state_matrix_df, idx):

    h, w, _ = frame.shape

    n_runs = len(timeline_strips)

    box_x1 = TIMELINE_MARGIN

    box_x2 = w - TIMELINE_MARGIN

    box_y2 = h - TIMELINE_MARGIN

    bar_h = 10

    gap = 4

    top_pad = 18

    left_label_w = 120

    total_h = top_pad + n_runs * (bar_h + gap) + 8

    box_y1 = max(0, box_y2 - total_h)

    overlay = frame.copy()

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

    frame[:] = cv2.addWeighted(overlay, TIMELINE_BG_ALPHA, frame, 1 - TIMELINE_BG_ALPHA, 0)

    cv2.putText(

        frame,

        "9-scale T2S timelines",

        (box_x1 + 6, box_y1 + 13),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (30, 30, 30),

        1

    )

    total_len = len(state_matrix_df)

    for i, (strip, info) in enumerate(zip(timeline_strips, run_infos)):

        y1 = box_y1 + top_pad + i * (bar_h + gap)

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

                           

          

                           

audio_state = build_audio_binary_state(audio_df)

move_state, move_dx = build_move_direction_state(visual_df)

              

audio_idx_for_visual = np.searchsorted(audio_times, visual_times, side="right") - 1

audio_idx_for_visual = np.clip(audio_idx_for_visual, 0, len(audio_state) - 1)

audio_state_aligned = audio_state[audio_idx_for_visual]

plot_aux_states(visual_times, audio_state_aligned, move_state, OUTPUT_AUX_PNG)

                           

                     

                           

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

    out_png = os.path.join(PLOT_DIR, f"{run_name}.png")

    plot_branch(result["data_std"], result["state_seq"], out_png)

    run_infos.append({

        "run_idx": i,

        "win": win,

        "step": step,

        "min_len": get_action_min_len(win),

        "col_name": col_name,

        "n_states": int(np.max(state_seq)) + 1,

        "plot_png": out_png,

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

state_matrix_df.to_csv(OUTPUT_STATE_MATRIX_CSV, index=False, encoding="utf-8-sig")

pd.DataFrame(run_infos).to_csv(OUTPUT_RUN_INFO_CSV, index=False, encoding="utf-8-sig")

print(f"\n已输出状态矩阵：{OUTPUT_STATE_MATRIX_CSV}")

print(f"已输出运行信息：{OUTPUT_RUN_INFO_CSV}")

       

plot_multiscale_states(state_matrix_df, OUTPUT_MULTI_PNG)

print(f"已输出综合状态图：{OUTPUT_MULTI_PNG}")

                    

timeline_width = 1400

for info in run_infos:

    col = info["col_name"]

    strip = build_timeline_strip(

        timeline_width,

        state_matrix_df[col].values.astype(int),

        STATE_PALETTE,

        bar_h=10

    )

    timeline_strips.append(strip)

                           

           

                           

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

frame_id = 0

total_rows = len(state_matrix_df)

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

    state_row = state_matrix_df.iloc[idx]

    audio_state_val = int(state_row["audio_speech_state"])

    move_state_val = int(state_row["move_direction_state"])

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

    cv2.rectangle(overlay, (20, 20), (500, 150), COLOR_BOX, -1)

    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, "9-scale action T2S", (35, 50),

                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)

    cv2.putText(frame, f"time = {current_time:.2f}s", (35, 80),

                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT, 2)

    cv2.putText(frame, f"Audio: {speech_text(audio_state_val)}", (35, 108),

                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT, 2)

    cv2.putText(frame, f"Move: {move_text(move_state_val)}", (35, 136),

                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT, 2)

    draw_multi_timelines(frame, timeline_strips, run_infos, state_matrix_df, idx)

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
