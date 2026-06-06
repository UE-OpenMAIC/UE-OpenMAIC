from Time2State.time2state import Time2State

from Time2State.adapers import CausalConv_LSE_Adaper

from Time2State.clustering import DPGMM

from Time2State.default_params import *

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from TSpy.view import plot_mts

import matplotlib.pyplot as plt

import cv2

from moviepy import VideoFileClip

                           

           

                           

INPUT_CSV = r"D:\code\teacherT2S\yolo\teacher_visual_15d.csv"

INPUT_VIDEO = r"D:\code\teacherT2S\yolo\input.mp4"

OUTPUT_CSV = r"D:\code\teacherT2S\yolo\teacher_visual_15d_with_state.csv"

OUTPUT_PNG = r"D:\code\teacherT2S\yolo\teacher_visual_15d_t2s.png"

TEMP_VIDEO = r"D:\code\teacherT2S\yolo\teacher_visual_15d_state_overlay_silent.mp4"

OUTPUT_VIDEO = r"D:\code\teacherT2S\yolo\teacher_visual_15d_state_overlay_with_audio.mp4"

TIME_COL = "time_sec"

USE_NORMALIZED_COORDS = True

                           

           

                           

           

            

            

win_size = 96

step = 24

           

               

           

                           

             

                           

feature_cols = [

    "center_x",

    "center_y",

    "orientation_score",

    "left_shoulder_x", "left_shoulder_y",

    "left_elbow_x", "left_elbow_y",

    "left_wrist_x", "left_wrist_y",

    "right_shoulder_x", "right_shoulder_y",

    "right_elbow_x", "right_elbow_y",

    "right_wrist_x", "right_wrist_y",

]

REQUIRED_COLS = [TIME_COL] + feature_cols

                           

       

                           

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

    if score >= 0:

        end_pt = (cx + length, cy - 20)

    else:

        end_pt = (cx - length, cy - 20)

    cv2.arrowedLine(frame, (cx, cy - 20), end_pt, COLOR_ORI, 3, tipLength=0.25)

def remap_to_contiguous_labels(state_seq):

    unique_states_raw = sorted(np.unique(state_seq))

    state_map = {old: new for new, old in enumerate(unique_states_raw)}

    remapped = np.array([state_map[s] for s in state_seq], dtype=int)

    return remapped, state_map

def merge_short_segments(state_seq, min_len=36):

    

       

    seq = state_seq.copy()

    n = len(seq)

    if n == 0:

        return seq

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

    return seq

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

        "state timeline",

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

                           

                

                           

df = pd.read_csv(INPUT_CSV)

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

if missing_cols:

    raise ValueError(f"CSV 缺少这些列：{missing_cols}")

df = df.sort_values(TIME_COL).reset_index(drop=True)

sample_times = df[TIME_COL].astype(float).values

X = df[feature_cols].copy()

                               

if "orientation_score" in X.columns:

    ori = X["orientation_score"].copy()

    ori = ori.clip(-2.0, 2.0)

          

    ori_three = []

    for v in ori:

        if pd.isna(v):

            ori_three.append(np.nan)

        elif v <= -0.25:

            ori_three.append(-1.0)

        elif v >= 0.25:

            ori_three.append(1.0)

        else:

            ori_three.append(0.0)

    X["orientation_score"] = ori_three

                              

                     

def safe_sub(a, b):

    if pd.isna(a) or pd.isna(b):

        return np.nan

    return float(a) - float(b)

def safe_dist(x1, y1, x2, y2):

    vals = [x1, y1, x2, y2]

    if any(pd.isna(v) for v in vals):

        return np.nan

    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def safe_angle(ax, ay, bx, by, cx, cy):

    vals = [ax, ay, bx, by, cx, cy]

    if any(pd.isna(v) for v in vals):

        return np.nan

    ba = np.array([ax - bx, ay - by], dtype=np.float32)

    bc = np.array([cx - bx, cy - by], dtype=np.float32)

    nba = np.linalg.norm(ba)

    nbc = np.linalg.norm(bc)

    if nba < 1e-6 or nbc < 1e-6:

        return np.nan

    cosang = np.dot(ba, bc) / (nba * nbc)

    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.arccos(cosang))

X_beh = pd.DataFrame(index=X.index)

X_beh["center_x"] = X["center_x"]

X_beh["center_y"] = X["center_y"]

X_beh["orientation_3class"] = X["orientation_score"]

X_beh["left_wrist_height_rel"] = [

    safe_sub(lsy, lwy)

    for lsy, lwy in zip(X["left_shoulder_y"], X["left_wrist_y"])

]

X_beh["right_wrist_height_rel"] = [

    safe_sub(rsy, rwy)

    for rsy, rwy in zip(X["right_shoulder_y"], X["right_wrist_y"])

]

X_beh["left_arm_extension"] = [

    safe_dist(lsx, lsy, lwx, lwy)

    for lsx, lsy, lwx, lwy in zip(

        X["left_shoulder_x"], X["left_shoulder_y"],

        X["left_wrist_x"], X["left_wrist_y"]

    )

]

X_beh["right_arm_extension"] = [

    safe_dist(rsx, rsy, rwx, rwy)

    for rsx, rsy, rwx, rwy in zip(

        X["right_shoulder_x"], X["right_shoulder_y"],

        X["right_wrist_x"], X["right_wrist_y"]

    )

]

X_beh["left_elbow_angle"] = [

    safe_angle(lsx, lsy, lex, ley, lwx, lwy)

    for lsx, lsy, lex, ley, lwx, lwy in zip(

        X["left_shoulder_x"], X["left_shoulder_y"],

        X["left_elbow_x"], X["left_elbow_y"],

        X["left_wrist_x"], X["left_wrist_y"]

    )

]

X_beh["right_elbow_angle"] = [

    safe_angle(rsx, rsy, rex, rey, rwx, rwy)

    for rsx, rsy, rex, rey, rwx, rwy in zip(

        X["right_shoulder_x"], X["right_shoulder_y"],

        X["right_elbow_x"], X["right_elbow_y"],

        X["right_wrist_x"], X["right_wrist_y"]

    )

]

X_beh["hands_distance"] = [

    safe_dist(lwx, lwy, rwx, rwy)

    for lwx, lwy, rwx, rwy in zip(

        X["left_wrist_x"], X["left_wrist_y"],

        X["right_wrist_x"], X["right_wrist_y"]

    )

]

       

X_beh = X_beh.ffill().bfill().fillna(0.0)

         

X_beh = X_beh.rolling(10, center=True, min_periods=1).median()

     

data = X_beh.to_numpy(dtype=np.float32)

scaler = StandardScaler()

data = scaler.fit_transform(data)

print("输入数据形状:", data.shape)

print("使用维度：", list(X_beh.columns))

params_LSE["in_channels"] = data.shape[1]

params_LSE["out_channels"] = 8

params_LSE["nb_steps"] = 30

params_LSE["win_size"] = win_size

params_LSE["win_type"] = "hanning"

t2s = Time2State(

    win_size,

    step,

    CausalConv_LSE_Adaper(params_LSE),

    DPGMM(None),

    params_LSE

)

t2s.fit(data, win_size, step)

state_seq = np.asarray(t2s.state_seq).astype(int)

if len(state_seq) != len(df):

    raise RuntimeError(f"状态序列长度 {len(state_seq)} 与数据行数 {len(df)} 不一致")

        

state_seq, state_map = remap_to_contiguous_labels(state_seq)

            

state_seq = merge_short_segments(state_seq, min_len=36)

            

state_seq, state_map = remap_to_contiguous_labels(state_seq)

print("状态映射关系：", state_map)

            

df["t2s_state"] = state_seq

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"已输出：{OUTPUT_CSV}")

          

plt.style.use("classic")

plot_mts(data, state_seq)

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

plt.close()

print(f"已保存图像：{OUTPUT_PNG}")

                           

                         

                           

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

unique_states, counts_states = np.unique(state_seq, return_counts=True)

state_count_dict = {int(s): int(c) for s, c in zip(unique_states, counts_states)}

frame_id = 0

total_rows = len(df)

            

timeline_width = width - 2 * TIMELINE_MARGIN - 16

timeline_strip = build_timeline_strip(

    timeline_width,

    state_seq,

    STATE_PALETTE,

    TIMELINE_BAR_HEIGHT

)

while True:

    ret, frame = cap.read()

    if not ret:

        break

    current_time = frame_id / fps

    idx = np.searchsorted(sample_times, current_time, side="right") - 1

    if idx < 0:

        idx = 0

    elif idx >= total_rows:

        idx = total_rows - 1

    row = df.iloc[idx]

    cur_state = int(state_seq[idx])

    state_color = STATE_PALETTE[cur_state % len(STATE_PALETTE)]

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

        cv2.putText(

            frame,

            "center",

            (center[0] + 8, center[1] - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            COLOR_CENTER,

            2

        )

         

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

    cv2.rectangle(overlay, (20, 20), (560, 210), COLOR_BOX, -1)

    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(

        frame,

        f"Latent visual state: {cur_state}",

        (35, 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        state_color,

        2

    )

    cv2.putText(

        frame,

        f"time = {current_time:.2f}s",

        (35, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        COLOR_TEXT,

        2

    )

    cv2.putText(

        frame,

        f"sample_idx = {idx}",

        (35, 112),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        COLOR_TEXT,

        2

    )

    ori = row["orientation_score"]

    ori_text = "NaN" if pd.isna(ori) or ori == "" else f"{float(ori):.3f}"

    cv2.putText(

        frame,

        f"orientation_score = {ori_text}",

        (35, 139),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        COLOR_TEXT,

        2

    )

    cv2.putText(

        frame,

        f"count(state {cur_state}) = {state_count_dict.get(cur_state, 0)}",

        (35, 166),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        COLOR_TEXT,

        2

    )

    cv2.putText(

        frame,

        "yellow=center, blue=left arm, red=right arm, green=orientation",

        (35, 193),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        COLOR_TEXT,

        2

    )

             

    cv2.rectangle(frame, (width - 120, 30), (width - 40, 90), state_color, -1)

    cv2.rectangle(frame, (width - 120, 30), (width - 40, 90), (0, 0, 0), 2)

    cv2.putText(

        frame,

        f"S{cur_state}",

        (width - 105, 72),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (255, 255, 255),

        2

    )

               

    draw_timeline_on_frame(

        frame,

        timeline_strip,

        idx,

        total_rows,

        state_color,

        cur_state

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
