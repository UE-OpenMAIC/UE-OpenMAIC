                       

   

from __future__ import annotations

import argparse

import json

import re

import shutil

import subprocess

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

                                                              

      

                                                              

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

VIDEO_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\input")

VISUAL_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\pose_csv")

GLOBAL_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_prototype_alignment_X1_only"

GLOBAL_CSV_DEFAULT = GLOBAL_DIR_DEFAULT / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

OUT_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_expert_visualize_skeleton_local_before_global_after"

LAYER1_CSV_NAME = "multiscale_t2s_with_meta.csv"

TIME_COL = "time_sec"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

POSE_COLS_REQUIRED = [

    "orientation_score",

    "left_shoulder_x", "left_shoulder_y",

    "left_elbow_x", "left_elbow_y",

    "left_wrist_x", "left_wrist_y",

    "right_shoulder_x", "right_shoulder_y",

    "right_elbow_x", "right_elbow_y",

    "right_wrist_x", "right_wrist_y",

    "center_x", "center_y",

]

                                                              

      

                                                              

def norm_video_id(x) -> str:

    s = str(x).strip().replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

    return re.sub(r"/+", "/", s).strip("/")

def tail_video_id(x) -> str:

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def safe_name(x) -> str:

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s or "video"

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def parse_video_ids(s: str | None):

    if not s:

        return None

    return [norm_video_id(x) for x in str(s).split(",") if str(x).strip()]

def load_state_names(json_path: str | None) -> dict:

    

       

    if not json_path:

        return {"local": {}, "global": {}}

    p = Path(json_path)

    if not p.exists():

        print(f"[WARN] state-name-json not found: {p}")

        return {"local": {}, "global": {}}

    with open(p, "r", encoding="utf-8") as f:

        data = json.load(f)

    data.setdefault("local", data.get("layer1", {}))

    data.setdefault("global", data.get("layer2", {}))

    data["local"] = {str(k): str(v) for k, v in data["local"].items()}

    data["global"] = {str(k): str(v) for k, v in data["global"].items()}

    return data

def state_display_name(layer: str, state: int, names: dict) -> str:

    try:

        state = int(state)

    except Exception:

        return "NA"

    if state < 0:

        return "NA"

    key = str(state)

    if layer in names and key in names[layer]:

        return f"{state}:{names[layer][key]}"

    return f"{state}"

def color_for_state(state: int, layer: str = "global"):

    try:

        s = int(state)

    except Exception:

        s = -1

    if s < 0:

        return (120, 120, 120)

                                 

    offset = 17 if layer == "local" else 83

    hue = int((s * 37 + offset) % 180)

    hsv = np.uint8([[[hue, 180, 255]]])

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

    return tuple(int(x) for x in bgr.tolist())

def draw_filled_rect_alpha(img, x1, y1, x2, y2, color, alpha=0.55):

    h, w = img.shape[:2]

    x1, x2 = max(0, x1), min(w, x2)

    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:

        return img

    overlay = img.copy()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    img[y1:y2, x1:x2] = cv2.addWeighted(

        overlay[y1:y2, x1:x2], alpha,

        img[y1:y2, x1:x2], 1 - alpha,

        0

    )

    return img

def put_text(img, text, x, y, scale=0.65, color=(255, 255, 255), thickness=2):

    cv2.putText(

        img,

        str(text),

        (int(x), int(y)),

        cv2.FONT_HERSHEY_SIMPLEX,

        float(scale),

        color,

        int(thickness),

        cv2.LINE_AA,

    )

                                                              

      

                                                              

def find_video_file(video_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    parts = vid.split("/") if vid else [tail]

    candidates = []

    for ext in VIDEO_EXTS:

        candidates.append(video_root / tail / f"{tail}{ext}")

    if parts:

        for ext in VIDEO_EXTS:

            candidates.append(video_root / Path(*parts).with_suffix(ext))

            candidates.append(video_root / Path(*parts) / f"{tail}{ext}")

    for ext in VIDEO_EXTS:

        candidates.append(video_root / f"{tail}{ext}")

    for p in candidates:

        if p.exists():

            return p

    if video_root.exists():

        for ext in VIDEO_EXTS:

            matches = sorted(video_root.rglob(f"{tail}{ext}"))

            if matches:

                return matches[0]

    return None

def find_layer1_csv(t2s_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    parts = vid.split("/") if vid else [tail]

    candidates = [

        t2s_root / Path(*parts) / LAYER1_CSV_NAME,

        t2s_root / tail / tail / LAYER1_CSV_NAME,

        t2s_root / tail / LAYER1_CSV_NAME,

    ]

    for p in candidates:

        if p.exists():

            return p

    if t2s_root.exists():

        matches = sorted(t2s_root.rglob(LAYER1_CSV_NAME))

        key = f"/{tail}/{tail}/"

        for p in matches:

            if key in str(p).replace("\\", "/"):

                return p

        for p in matches:

            if f"/{tail}/" in str(p).replace("\\", "/"):

                return p

    return None

def find_visual_csv(visual_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    parts = vid.split("/") if vid else [tail]

    candidates = []

    if parts:

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

                                                              

      

                                                              

def load_local_sequence(layer1_csv: Path, fps: float):

    df = pd.read_csv(layer1_csv)

    if "meta_state" not in df.columns:

        raise ValueError(f"{layer1_csv} 缺少 meta_state 列")

    state = pd.to_numeric(df["meta_state"], errors="coerce").fillna(-1).astype(int).to_numpy()

    if TIME_COL in df.columns:

        t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)

        if np.isfinite(t).sum() < len(t) * 0.8:

            t = np.arange(len(df), dtype=float) / max(1e-6, fps)

    else:

        t = np.arange(len(df), dtype=float) / max(1e-6, fps)

    order = np.argsort(t)

    return {"time": t[order], "state": state[order], "df": df}

def local_state_at_time(local_obj, current_sec: float) -> int:

    if local_obj is None:

        return -1

    t = local_obj["time"]

    s = local_obj["state"]

    if len(t) == 0:

        return -1

    idx = np.searchsorted(t, current_sec, side="left")

    if idx <= 0:

        return int(s[0])

    if idx >= len(t):

        return int(s[-1])

    if abs(t[idx] - current_sec) < abs(t[idx - 1] - current_sec):

        return int(s[idx])

    return int(s[idx - 1])

def prepare_global_segments(global_df: pd.DataFrame, video_id: str) -> pd.DataFrame:

    vid = norm_video_id(video_id)

    sub = global_df[global_df["video_id"].astype(str).map(norm_video_id).eq(vid)].copy()

    if len(sub) == 0:

        tail = tail_video_id(vid)

        sub = global_df[global_df["video_id"].astype(str).map(tail_video_id).eq(tail)].copy()

    if len(sub) == 0:

        return sub

            

    if "global_state" not in sub.columns and "layer2_state" in sub.columns:

        sub["global_state"] = sub["layer2_state"]

    if "layer2_state" not in sub.columns and "global_state" in sub.columns:

        sub["layer2_state"] = sub["global_state"]

    if "layer1_meta_state_local" not in sub.columns and "local_meta_state" in sub.columns:

        sub["layer1_meta_state_local"] = sub["local_meta_state"]

    for c in ["start_sec", "end_sec", "global_state", "layer2_state", "layer1_meta_state_local"]:

        if c in sub.columns:

            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sort_cols = [c for c in ["start_sec", "end_sec", "seg_idx"] if c in sub.columns]

    sub = sub.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return sub

def global_state_at_time(global_sub: pd.DataFrame, current_sec: float, last_idx: int = 0):

    

       

    if global_sub is None or len(global_sub) == 0:

        return -1, -1, -1, 0

    n = len(global_sub)

    i = min(max(0, int(last_idx)), n - 1)

    while i < n - 1 and current_sec > float(global_sub.iloc[i]["end_sec"]):

        i += 1

    while i > 0 and current_sec < float(global_sub.iloc[i]["start_sec"]):

        i -= 1

    row = global_sub.iloc[i]

    gs = int(row["global_state"]) if "global_state" in row and pd.notna(row["global_state"]) else -1

    ls = int(row["layer1_meta_state_local"]) if "layer1_meta_state_local" in row and pd.notna(row["layer1_meta_state_local"]) else -1

    seg_idx = int(row["seg_idx"]) if "seg_idx" in row and pd.notna(row["seg_idx"]) else -1

    start = float(row.get("start_sec", np.nan))

    end = float(row.get("end_sec", np.nan))

    if not (np.isfinite(start) and np.isfinite(end) and start <= current_sec <= end):

                                                               

        return gs, ls, seg_idx, i

    return gs, ls, seg_idx, i

                                                              

           

                                                              

def load_visual_action_sequence(visual_csv: Path, fps: float):

    df = pd.read_csv(visual_csv)

    missing = [c for c in POSE_COLS_REQUIRED if c not in df.columns]

    if missing:

        raise ValueError(f"{visual_csv} 缺少动作列：{missing}")

    if TIME_COL in df.columns:

        t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)

        if np.isfinite(t).sum() < len(t) * 0.8:

            t = np.arange(len(df), dtype=float) / max(1e-6, fps)

    else:

        t = np.arange(len(df), dtype=float) / max(1e-6, fps)

    order = np.argsort(t)

    df = df.iloc[order].reset_index(drop=True)

    t = t[order]

    return {"time": t, "df": df, "visual_csv": str(visual_csv)}

def visual_row_at_time(visual_obj, current_sec: float):

    if visual_obj is None:

        return None

    t = visual_obj["time"]

    df = visual_obj["df"]

    if len(t) == 0:

        return None

    idx = np.searchsorted(t, current_sec, side="left")

    if idx <= 0:

        return df.iloc[0]

    if idx >= len(t):

        return df.iloc[-1]

    if abs(t[idx] - current_sec) < abs(t[idx - 1] - current_sec):

        return df.iloc[idx]

    return df.iloc[idx - 1]

def _coord_to_pixel(x, y, w, h):

    try:

        x = float(x)

        y = float(y)

    except Exception:

        return None

    if not (np.isfinite(x) and np.isfinite(y)):

        return None

                  

    if -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:

        px = int(round(x * w))

        py = int(round(y * h))

    else:

        px = int(round(x))

        py = int(round(y))

    if px < -w or px > 2 * w or py < -h or py > 2 * h:

        return None

    return (px, py)

def orientation_3class_value(v: float):

    try:

        v = float(v)

    except Exception:

        return 0

    if v <= -0.25:

        return -1

    if v >= 0.25:

        return 1

    return 0

def draw_skeleton(frame, visual_row, args):

    if visual_row is None:

        return frame, {}

    h, w = frame.shape[:2]

    pts = {}

    for name in [

        "center",

        "left_shoulder", "left_elbow", "left_wrist",

        "right_shoulder", "right_elbow", "right_wrist",

    ]:

        x_col = f"{name}_x"

        y_col = f"{name}_y"

        if x_col in visual_row.index and y_col in visual_row.index:

            pts[name] = _coord_to_pixel(visual_row[x_col], visual_row[y_col], w, h)

        else:

            pts[name] = None

    left_color = (255, 180, 80)

    right_color = (80, 220, 255)

    mid_color = (210, 210, 210)

    center_color = (255, 255, 255)

    if not args.no_skeleton:

        for a, b, color in [

            ("left_shoulder", "left_elbow", left_color),

            ("left_elbow", "left_wrist", left_color),

            ("right_shoulder", "right_elbow", right_color),

            ("right_elbow", "right_wrist", right_color),

            ("left_shoulder", "right_shoulder", mid_color),

            ("center", "left_shoulder", mid_color),

            ("center", "right_shoulder", mid_color),

        ]:

            if pts.get(a) is not None and pts.get(b) is not None:

                cv2.line(frame, pts[a], pts[b], color, int(args.skeleton_thickness), cv2.LINE_AA)

        for name, p in pts.items():

            if p is None:

                continue

            color = center_color

            if name.startswith("left"):

                color = left_color

            elif name.startswith("right"):

                color = right_color

            radius = int(args.joint_radius if name != "center" else args.joint_radius + 2)

            cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)

                       

    ori = float(visual_row.get("orientation_score", np.nan))

    ori3 = orientation_3class_value(ori)

    return frame, {

        "orientation_score": ori,

        "orientation_3class": ori3,

    }

                                                              

                

                                                              

def build_timeline_image(width: int, duration_sec: float, local_obj, global_sub):

    

       

    h = 58

    img = np.zeros((h, width, 3), dtype=np.uint8)

    img[:] = (30, 30, 30)

    if duration_sec <= 0:

        duration_sec = 1.0

    ptr = 0

    for x in range(width):

        t = (x / max(1, width - 1)) * duration_sec

        local_state = local_state_at_time(local_obj, t)

        global_state, _seg_local, _seg_idx, ptr = global_state_at_time(global_sub, t, ptr)

        img[0:22, x:x + 1] = color_for_state(local_state, "local")

        img[30:52, x:x + 1] = color_for_state(global_state, "global")

    put_text(img, "Before", 5, 16, scale=0.42, color=(255, 255, 255), thickness=1)

    put_text(img, "After", 5, 47, scale=0.42, color=(255, 255, 255), thickness=1)

    return img

def draw_overlay(

    frame,

    current_sec,

    duration_sec,

    local_state,

    global_state,

    segment_local_state,

    global_seg_idx,

    names,

    timeline_img,

    action_info,

    args,

):

    h, w = frame.shape[:2]

    panel_h = 168

    draw_filled_rect_alpha(frame, 0, 0, w, panel_h, (0, 0, 0), alpha=0.62)

    local_color = color_for_state(local_state, "local")

    global_color = color_for_state(global_state, "global")

    cv2.rectangle(frame, (18, 44), (50, 76), local_color, -1)

    cv2.rectangle(frame, (18, 92), (50, 124), global_color, -1)

    put_text(frame, f"time {current_sec:7.2f}s / {duration_sec:7.2f}s", 18, 28, scale=0.70)

    put_text(

        frame,

        f"BEFORE cross-video clustering  | local meta_state: {state_display_name('local', local_state, names)}",

        62,

        69,

        scale=0.68,

    )

    put_text(

        frame,

        f"AFTER  cross-video clustering  | global_state: {state_display_name('global', global_state, names)}",

        62,

        116,

        scale=0.68,

    )

    put_text(frame, f"segment local: {segment_local_state}", w - 300, 74, scale=0.55)

    put_text(frame, f"global seg: {global_seg_idx}", w - 300, 104, scale=0.55)

    if args.show_orientation and action_info:

        ori = action_info.get("orientation_score", np.nan)

        ori3 = action_info.get("orientation_3class", 0)

        put_text(

            frame,

            f"skeleton/action: orientation_score={ori:+.2f}, orientation_3class={ori3:+d}",

            18,

            152,

            scale=0.50,

            color=(230, 230, 230),

            thickness=1,

        )

    if timeline_img is not None:

        th = timeline_img.shape[0]

        y0 = h - th - 8

        if y0 > panel_h + 5:

            frame[y0:y0 + th, 0:w] = timeline_img

            x_cur = int(np.clip((current_sec / max(1e-6, duration_sec)) * (w - 1), 0, w - 1))

            cv2.line(frame, (x_cur, y0), (x_cur, y0 + th), (255, 255, 255), 2)

            put_text(

                frame,

                "Before = local states inside each video / After = cross-video global states",

                80,

                y0 + th - 7,

                scale=0.43,

                color=(255, 255, 255),

                thickness=1,

            )

    return frame

def make_global_legend(global_df: pd.DataFrame, out_path: Path, names: dict):

    if "global_state" not in global_df.columns and "layer2_state" in global_df.columns:

        global_df = global_df.copy()

        global_df["global_state"] = global_df["layer2_state"]

    rows = []

    total = len(global_df)

    for s, g in global_df.groupby("global_state"):

        s = int(s)

        rows.append({

            "state": s,

            "n_segments": int(len(g)),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "ratio": float(len(g) / max(1, total)),

        })

    rows = sorted(rows, key=lambda r: (-r["support_videos"], -r["n_segments"], r["state"]))

    row_h = 38

    w = 900

    h = max(120, 52 + row_h * len(rows))

    img = np.zeros((h, w, 3), dtype=np.uint8)

    img[:] = (245, 245, 245)

    put_text(img, "After cross-video clustering: global_state legend", 20, 32, scale=0.76, color=(20, 20, 20), thickness=2)

    y = 62

    for r in rows:

        color = color_for_state(r["state"], "global")

        cv2.rectangle(img, (22, y - 20), (55, y + 8), color, -1)

        text = (

            f"global_state {state_display_name('global', r['state'], names):<12} "

            f"segments={r['n_segments']:<4} "

            f"support_videos={r['support_videos']:<2} "

            f"ratio={r['ratio']:.3f}"

        )

        put_text(img, text, 70, y, scale=0.53, color=(20, 20, 20), thickness=1)

        y += row_h

    cv2.imwrite(str(out_path), img)

                                                              

      

                                                              

def try_mux_audio(video_noaudio: Path, original_video: Path, final_video: Path) -> bool:

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:

        return False

    cmd = [

        ffmpeg, "-y",

        "-i", str(video_noaudio),

        "-i", str(original_video),

        "-map", "0:v:0",

        "-map", "1:a:0?",

        "-c:v", "copy",

        "-c:a", "aac",

        "-shortest",

        str(final_video),

    ]

    try:

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        return final_video.exists() and final_video.stat().st_size > 0

    except Exception:

        return False

                                                              

       

                                                              

def render_one_video(

    video_id: str,

    video_file: Path,

    layer1_csv: Path | None,

    visual_csv: Path | None,

    global_sub: pd.DataFrame,

    out_video: Path,

    args,

    names: dict,

):

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():

        raise RuntimeError(f"Cannot open video: {video_file}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)

    if not src_fps or not np.isfinite(src_fps) or src_fps <= 1e-6:

        src_fps = float(args.fps_fallback)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_sec = frame_count / max(1e-6, src_fps)

    if args.resize_width and args.resize_width > 0 and src_w > args.resize_width:

        out_w = int(args.resize_width)

        out_h = int(round(src_h * (out_w / src_w)))

    else:

        out_w, out_h = src_w, src_h

    local_obj = None

    if layer1_csv is not None and layer1_csv.exists():

        try:

            local_obj = load_local_sequence(layer1_csv, fps=src_fps)

        except Exception as e:

            print(f"  [WARN] failed to load layer1/local csv: {layer1_csv} -> {repr(e)}")

    else:

        print("  [WARN] layer1/local csv not found; before-clustering state will be NA")

    visual_obj = None

    if visual_csv is not None and visual_csv.exists():

        try:

            visual_obj = load_visual_action_sequence(visual_csv, fps=src_fps)

        except Exception as e:

            print(f"  [WARN] failed to load visual/action csv: {visual_csv} -> {repr(e)}")

    else:

        print("  [WARN] visual/action csv not found; skeleton will be NA")

    timeline_img = build_timeline_image(out_w, duration_sec, local_obj, global_sub)

    ensure_dir(out_video.parent)

    tmp_video = out_video.with_suffix(".noaudio.mp4") if not args.no_audio else out_video

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(tmp_video), fourcc, float(src_fps), (out_w, out_h))

    if not writer.isOpened():

        raise RuntimeError(f"Cannot create writer: {tmp_video}")

    ptr_global = 0

    frame_idx = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        if out_w != src_w or out_h != src_h:

            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        current_sec = frame_idx / max(1e-6, src_fps)

                  

        visual_row = visual_row_at_time(visual_obj, current_sec)

        frame, action_info = draw_skeleton(frame, visual_row, args)

                                     

        local_state = local_state_at_time(local_obj, current_sec)

                                

        global_state, segment_local_state, global_seg_idx, ptr_global = global_state_at_time(

            global_sub,

            current_sec,

            ptr_global,

        )

        frame = draw_overlay(

            frame=frame,

            current_sec=current_sec,

            duration_sec=duration_sec,

            local_state=local_state,

            global_state=global_state,

            segment_local_state=segment_local_state,

            global_seg_idx=global_seg_idx,

            names=names,

            timeline_img=timeline_img,

            action_info=action_info,

            args=args,

        )

        writer.write(frame)

        frame_idx += 1

        if args.max_seconds and args.max_seconds > 0 and current_sec >= float(args.max_seconds):

            break

    cap.release()

    writer.release()

    if not args.no_audio:

        mux_ok = try_mux_audio(tmp_video, video_file, out_video)

        if mux_ok:

            try:

                tmp_video.unlink()

            except Exception:

                pass

        else:

            if tmp_video != out_video:

                if out_video.exists():

                    try:

                        out_video.unlink()

                    except Exception:

                        pass

                tmp_video.rename(out_video)

    return {

        "video_id": norm_video_id(video_id),

        "video_file": str(video_file),

        "layer1_csv": str(layer1_csv) if layer1_csv else "",

        "visual_csv": str(visual_csv) if visual_csv else "",

        "out_video": str(out_video),

        "fps": float(src_fps),

        "src_w": int(src_w),

        "src_h": int(src_h),

        "out_w": int(out_w),

        "out_h": int(out_h),

        "frame_count": int(frame_count),

        "duration_sec": float(duration_sec),

        "rendered_frames": int(frame_idx),

    }

                                                              

     

                                                              

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--global-csv", "--layer2-csv", dest="global_csv", type=str, default=str(GLOBAL_CSV_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--video-root", type=str, default=str(VIDEO_ROOT_DEFAULT))

    ap.add_argument("--visual-root", type=str, default=str(VISUAL_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--video-ids", type=str, default="", help="comma-separated video ids, e.g. 21/21,22/22. Default: all videos in global csv.")

    ap.add_argument("--resize-width", type=int, default=1280, help="resize output width; 0 means keep original")

    ap.add_argument("--fps-fallback", type=float, default=12.0)

    ap.add_argument("--state-name-json", type=str, default="", help="optional JSON: {'global': {'0':'...'}, 'local': {'0':'...'}}")

    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--no-audio", action="store_true")

    ap.add_argument("--max-videos", type=int, default=0)

    ap.add_argument("--max-seconds", type=float, default=0.0)

    ap.add_argument("--no-skeleton", action="store_true")

    ap.add_argument("--skeleton-thickness", type=int, default=4)

    ap.add_argument("--joint-radius", type=int, default=5)

    ap.add_argument("--show-orientation", action="store_true", help="show orientation numeric value in top panel")

    args = ap.parse_args()

    global_csv = Path(args.global_csv)

    t2s_root = Path(args.t2s_root)

    video_root = Path(args.video_root)

    visual_root = Path(args.visual_root)

    out_dir = Path(args.out_dir)

    if not global_csv.exists():

        raise FileNotFoundError(f"global csv not found: {global_csv}")

    ensure_dir(out_dir)

    video_out_dir = out_dir / "annotated_videos"

    ensure_dir(video_out_dir)

    names = load_state_names(args.state_name_json)

    global_df = pd.read_csv(global_csv)

    if "video_id" not in global_df.columns:

        raise ValueError(f"{global_csv} 缺少 video_id 列")

    if "global_state" not in global_df.columns:

        if "layer2_state" in global_df.columns:

            global_df["global_state"] = global_df["layer2_state"]

        else:

            raise ValueError(f"{global_csv} 缺少 global_state/layer2_state 列")

    if "layer1_meta_state_local" not in global_df.columns and "local_meta_state" in global_df.columns:

        global_df["layer1_meta_state_local"] = global_df["local_meta_state"]

    global_df["video_id"] = global_df["video_id"].astype(str).map(norm_video_id)

    selected_ids = parse_video_ids(args.video_ids)

    if selected_ids is None:

        video_ids = sorted(global_df["video_id"].unique().tolist(), key=lambda x: (tail_video_id(x), x))

    else:

        video_ids = selected_ids

    if args.max_videos and args.max_videos > 0:

        video_ids = video_ids[:int(args.max_videos)]

          

    index_df = global_df.copy()

    index_df["video_id_norm"] = index_df["video_id"].map(norm_video_id)

    index_df["video_tail"] = index_df["video_id"].map(tail_video_id)

    index_df["annotated_video"] = index_df["video_id"].map(

        lambda v: str(video_out_dir / f"{safe_name(v)}_skeleton_local_before_global_after.mp4")

    )

    index_cols = [

        "global_state", "video_id", "video_tail", "seg_idx",

        "start_sec", "end_sec", "duration_sec",

        "layer1_meta_state_local", "local_meta_state",

        "prototype_id", "annotated_video",

    ]

    index_cols = [c for c in index_cols if c in index_df.columns]

    expert_index_path = out_dir / "expert_segment_index_by_global_state.csv"

    index_df[index_cols].sort_values(["global_state", "video_id", "start_sec"]).to_csv(

        expert_index_path,

        index=False,

        encoding="utf-8-sig",

    )

    legend_path = out_dir / "global_state_legend.png"

    make_global_legend(global_df, legend_path, names)

    print("=" * 100)

    print("Visualize skeleton + BEFORE local states + AFTER global states")

    print(f"global_csv : {global_csv}")

    print(f"t2s_root   : {t2s_root}")

    print(f"video_root : {video_root}")

    print(f"visual_root: {visual_root}")

    print(f"out_dir    : {out_dir}")

    print(f"videos     : {len(video_ids)}")

    print(f"index      : {expert_index_path}")

    print(f"legend     : {legend_path}")

    print("=" * 100)

    manifest_rows = []

    for i, vid in enumerate(video_ids, start=1):

        vid = norm_video_id(vid)

        out_video = video_out_dir / f"{safe_name(vid)}_skeleton_local_before_global_after.mp4"

        print(f"\n[{i}/{len(video_ids)}] {vid}")

        if out_video.exists() and not args.overwrite:

            print(f"  [SKIP existing] {out_video}")

            manifest_rows.append({

                "video_id": vid,

                "status": "skipped_existing",

                "out_video": str(out_video),

            })

            continue

        video_file = find_video_file(video_root, vid)

        layer1_csv = find_layer1_csv(t2s_root, vid)

        visual_csv = find_visual_csv(visual_root, vid)

        global_sub = prepare_global_segments(global_df, vid)

        row = {

            "video_id": vid,

            "status": "",

            "video_file": str(video_file) if video_file else "",

            "layer1_csv": str(layer1_csv) if layer1_csv else "",

            "visual_csv": str(visual_csv) if visual_csv else "",

            "global_segments": int(len(global_sub)),

            "out_video": str(out_video),

        }

        if video_file is None:

            row["status"] = "missing_video"

            print("  [MISS] video file not found")

            manifest_rows.append(row)

            continue

        if len(global_sub) == 0:

            row["status"] = "missing_global_segments"

            print("  [MISS] no global-state segments")

            manifest_rows.append(row)

            continue

        if layer1_csv is None:

            print("  [WARN] layer1 csv not found; before-clustering local state will be NA")

        if visual_csv is None:

            print("  [WARN] visual csv not found; skeleton will be NA")

        try:

            render_info = render_one_video(

                video_id=vid,

                video_file=video_file,

                layer1_csv=layer1_csv,

                visual_csv=visual_csv,

                global_sub=global_sub,

                out_video=out_video,

                args=args,

                names=names,

            )

            row.update(render_info)

            row["status"] = "ok"

            print(f"  [OK] {out_video}")

        except Exception as e:

            row["status"] = "failed"

            row["error"] = repr(e)

            print(f"  [FAILED] {repr(e)}")

        manifest_rows.append(row)

        pd.DataFrame(manifest_rows).to_csv(out_dir / "expert_video_manifest.csv", index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "expert_video_manifest.csv"

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\nDONE.")

    print(f"Annotated videos: {video_out_dir}")

    print(f"Manifest        : {manifest_path}")

    print(f"Segment index   : {expert_index_path}")

    print(f"Legend          : {legend_path}")

if __name__ == "__main__":

    main()
