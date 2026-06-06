                       

   

from __future__ import annotations

import argparse

import json

import math

import os

import re

import shutil

import subprocess

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

                                                              

      

                                                              

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

VIDEO_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\input")

LAYER2_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_t2s_X1_compactY1_segment_multit2s_pid_8x4"

LAYER2_CSV_DEFAULT = LAYER2_DIR_DEFAULT / "layer2_cross_video_segment_states_X1compactY1_multit2s_pid.csv"

OUT_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_expert_visualize_layer1_local_and_layer2_8x4"

LAYER1_CSV_NAME = "multiscale_t2s_with_meta.csv"

TIME_COL = "time_sec"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

                                                              

      

                                                              

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

        return {"layer1": {}, "layer2": {}}

    p = Path(json_path)

    if not p.exists():

        print(f"[WARN] state-name-json not found: {p}")

        return {"layer1": {}, "layer2": {}}

    with open(p, "r", encoding="utf-8") as f:

        data = json.load(f)

    data.setdefault("layer1", {})

    data.setdefault("layer2", {})

                                  

    data["layer1"] = {str(k): str(v) for k, v in data["layer1"].items()}

    data["layer2"] = {str(k): str(v) for k, v in data["layer2"].items()}

    return data

def state_display_name(layer: str, state: int, names: dict) -> str:

    if state is None or int(state) < 0:

        return "NA"

    key = str(int(state))

    if layer in names and key in names[layer]:

        return f"{int(state)}:{names[layer][key]}"

    return f"{int(state)}"

def color_for_state(state: int, layer: str = "layer2"):

    

       

    try:

        s = int(state)

    except Exception:

        s = -1

    if s < 0:

        return (120, 120, 120)

    offset = 17 if layer == "layer1" else 83

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

    candidates = []

    candidates.append(t2s_root / Path(*parts) / LAYER1_CSV_NAME)

    candidates.append(t2s_root / tail / tail / LAYER1_CSV_NAME)

    candidates.append(t2s_root / tail / LAYER1_CSV_NAME)

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

                                                              

      

                                                              

def load_layer1_sequence(layer1_csv: Path, fps: float):

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

    t = t[order]

    state = state[order]

    return {

        "time": t,

        "state": state,

        "df": df,

    }

def layer1_state_at_time(layer1_obj, current_sec: float) -> int:

    if layer1_obj is None:

        return -1

    t = layer1_obj["time"]

    s = layer1_obj["state"]

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

def prepare_layer2_segments(layer2_df: pd.DataFrame, video_id: str) -> pd.DataFrame:

    vid = norm_video_id(video_id)

    sub = layer2_df[layer2_df["video_id"].astype(str).map(norm_video_id).eq(vid)].copy()

    if len(sub) == 0:

                          

        tail = tail_video_id(vid)

        sub = layer2_df[layer2_df["video_id"].astype(str).map(tail_video_id).eq(tail)].copy()

    if len(sub) == 0:

        return sub

    for c in ["start_sec", "end_sec", "layer2_state", "layer1_meta_state_local"]:

        if c in sub.columns:

            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub = sub.sort_values(["start_sec", "end_sec", "seg_idx"], na_position="last").reset_index(drop=True)

    return sub

def layer2_at_time(layer2_sub: pd.DataFrame, current_sec: float, last_idx: int = 0):

    

       

    if layer2_sub is None or len(layer2_sub) == 0:

        return -1, -1, 0.0, -1, 0

    n = len(layer2_sub)

    i = min(max(0, int(last_idx)), n - 1)

                                                    

    while i < n - 1 and current_sec > float(layer2_sub.iloc[i]["end_sec"]):

        i += 1

                                

    while i > 0 and current_sec < float(layer2_sub.iloc[i]["start_sec"]):

        i -= 1

    row = layer2_sub.iloc[i]

    start = float(row.get("start_sec", np.nan))

    end = float(row.get("end_sec", np.nan))

    if not (np.isfinite(start) and np.isfinite(end) and start <= current_sec <= end):

                                                                               

                                                                  

        l2 = int(row["layer2_state"]) if "layer2_state" in row else -1

        l1seg = int(row["layer1_meta_state_local"]) if "layer1_meta_state_local" in row else -1

        return l2, l1seg, 0.0, int(row.get("seg_idx", -1)), i

    l2 = int(row["layer2_state"]) if "layer2_state" in row else -1

    l1seg = int(row["layer1_meta_state_local"]) if "layer1_meta_state_local" in row else -1

    ratio_cols = [c for c in layer2_sub.columns if c.startswith("meta_ratio_")]

    conf = 0.0

    if ratio_cols:

        vals = pd.to_numeric(row[ratio_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        conf = float(np.max(vals)) if len(vals) else 0.0

    return l2, l1seg, conf, int(row.get("seg_idx", -1)), i

                                                              

                   

                                                              

def build_timeline_image(width: int, duration_sec: float, layer1_obj, layer2_sub, names: dict):

    

       

    h = 54

    img = np.zeros((h, width, 3), dtype=np.uint8)

    img[:] = (30, 30, 30)

    if duration_sec <= 0:

        duration_sec = 1.0

    ptr = 0

    for x in range(width):

        t = (x / max(1, width - 1)) * duration_sec

        l1 = layer1_state_at_time(layer1_obj, t)

        l2, _l1seg, _conf, _seg_idx, ptr = layer2_at_time(layer2_sub, t, ptr)

        img[0:20, x:x + 1] = color_for_state(l1, "layer1")

        img[26:46, x:x + 1] = color_for_state(l2, "layer2")

    put_text(img, "L1", 5, 16, scale=0.45, color=(255, 255, 255), thickness=1)

    put_text(img, "L2", 5, 42, scale=0.45, color=(255, 255, 255), thickness=1)

    return img

def draw_overlay(frame, current_sec, duration_sec, layer1_state, layer1_seg_state, layer2_state, layer2_conf, layer2_seg_idx, names, timeline_img):

    h, w = frame.shape[:2]

               

    panel_h = 150

    draw_filled_rect_alpha(frame, 0, 0, w, panel_h, (0, 0, 0), alpha=0.58)

    l1_color = color_for_state(layer1_state, "layer1")

    l2_color = color_for_state(layer2_state, "layer2")

                 

    cv2.rectangle(frame, (18, 44), (48, 74), l1_color, -1)

    cv2.rectangle(frame, (18, 90), (48, 120), l2_color, -1)

    put_text(frame, f"time {current_sec:7.2f}s / {duration_sec:7.2f}s", 18, 28, scale=0.70)

    put_text(frame, f"L1 local meta_state: {state_display_name('layer1', layer1_state, names)}", 60, 68, scale=0.72)

    put_text(frame, f"L2 cross-video state: {state_display_name('layer2', layer2_state, names)}", 60, 114, scale=0.72)

                            

    put_text(frame, f"L1-seg: {layer1_seg_state}", w - 260, 68, scale=0.62)

    put_text(frame, f"L2-seg: {layer2_seg_idx}", w - 260, 96, scale=0.62)

    put_text(frame, f"L2-conf: {layer2_conf:.2f}", w - 260, 124, scale=0.62)

                     

    if timeline_img is not None:

        th = timeline_img.shape[0]

        y0 = h - th - 8

        if y0 > panel_h + 5:

            frame[y0:y0 + th, 0:w] = timeline_img

            x_cur = int(np.clip((current_sec / max(1e-6, duration_sec)) * (w - 1), 0, w - 1))

            cv2.line(frame, (x_cur, y0), (x_cur, y0 + th), (255, 255, 255), 2)

            put_text(frame, "Layer-1 local timeline / Layer-2 cross-video timeline", 70, y0 + th - 7, scale=0.45, color=(255, 255, 255), thickness=1)

    return frame

def make_layer2_legend(layer2_df: pd.DataFrame, out_path: Path, names: dict):

    if "layer2_state" not in layer2_df.columns:

        return

    rows = []

    total = len(layer2_df)

    for s, g in layer2_df.groupby("layer2_state"):

        s = int(s)

        rows.append({

            "state": s,

            "n_segments": int(len(g)),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "ratio": float(len(g) / max(1, total)),

        })

    rows = sorted(rows, key=lambda r: (-r["support_videos"], -r["n_segments"], r["state"]))

    row_h = 38

    w = 820

    h = max(120, 52 + row_h * len(rows))

    img = np.zeros((h, w, 3), dtype=np.uint8)

    img[:] = (245, 245, 245)

    put_text(img, "Layer-2 cross-video state legend", 20, 32, scale=0.8, color=(20, 20, 20), thickness=2)

    y = 62

    for r in rows:

        color = color_for_state(r["state"], "layer2")

        cv2.rectangle(img, (22, y - 20), (55, y + 8), color, -1)

        text = (

            f"state {state_display_name('layer2', r['state'], names):<12} "

            f"segments={r['n_segments']:<4} "

            f"support_videos={r['support_videos']:<2} "

            f"ratio={r['ratio']:.3f}"

        )

        put_text(img, text, 70, y, scale=0.55, color=(20, 20, 20), thickness=1)

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

                                                              

       

                                                              

def render_one_video(video_id: str, video_file: Path, layer1_csv: Path | None, layer2_sub: pd.DataFrame, out_video: Path, args, names: dict):

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

    if layer1_csv is not None and layer1_csv.exists():

        try:

            layer1_obj = load_layer1_sequence(layer1_csv, fps=src_fps)

        except Exception as e:

            print(f"  [WARN] failed to load layer1 csv: {layer1_csv} -> {repr(e)}")

            layer1_obj = None

    else:

        layer1_obj = None

    timeline_img = build_timeline_image(out_w, duration_sec, layer1_obj, layer2_sub, names)

    ensure_dir(out_video.parent)

    tmp_video = out_video.with_suffix(".noaudio.mp4") if not args.no_audio else out_video

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(tmp_video), fourcc, float(src_fps), (out_w, out_h))

    if not writer.isOpened():

        raise RuntimeError(f"Cannot create writer: {tmp_video}")

    ptr_l2 = 0

    frame_idx = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        if out_w != src_w or out_h != src_h:

            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        current_sec = frame_idx / max(1e-6, src_fps)

        l1_state = layer1_state_at_time(layer1_obj, current_sec)

        l2_state, l1_seg_state, l2_conf, l2_seg_idx, ptr_l2 = layer2_at_time(layer2_sub, current_sec, ptr_l2)

        frame = draw_overlay(

            frame,

            current_sec=current_sec,

            duration_sec=duration_sec,

            layer1_state=l1_state,

            layer1_seg_state=l1_seg_state,

            layer2_state=l2_state,

            layer2_conf=l2_conf,

            layer2_seg_idx=l2_seg_idx,

            names=names,

            timeline_img=timeline_img,

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

    ap.add_argument("--layer2-csv", type=str, default=str(LAYER2_CSV_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--video-root", type=str, default=str(VIDEO_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--video-ids", type=str, default="", help="comma-separated video ids, e.g. 21/21,22/22. Default: all videos in layer2 csv.")

    ap.add_argument("--resize-width", type=int, default=1280, help="resize output width; 0 means keep original")

    ap.add_argument("--fps-fallback", type=float, default=12.0)

    ap.add_argument("--state-name-json", type=str, default="", help="optional JSON: {'layer1': {'0':'...'}, 'layer2': {'0':'...'}}")

    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--no-audio", action="store_true", help="do not try to mux original audio into annotated mp4")

    ap.add_argument("--max-videos", type=int, default=0)

    ap.add_argument("--max-seconds", type=float, default=0.0, help="debug only: render first N seconds of each video")

    args = ap.parse_args()

    layer2_csv = Path(args.layer2_csv)

    t2s_root = Path(args.t2s_root)

    video_root = Path(args.video_root)

    out_dir = Path(args.out_dir)

    if not layer2_csv.exists():

        raise FileNotFoundError(f"layer2 csv not found: {layer2_csv}")

    ensure_dir(out_dir)

    video_out_dir = out_dir / "annotated_videos"

    ensure_dir(video_out_dir)

    names = load_state_names(args.state_name_json)

    layer2_df = pd.read_csv(layer2_csv)

    if "video_id" not in layer2_df.columns:

        raise ValueError(f"{layer2_csv} 缺少 video_id 列")

    if "layer2_state" not in layer2_df.columns:

        raise ValueError(f"{layer2_csv} 缺少 layer2_state 列")

    layer2_df["video_id"] = layer2_df["video_id"].astype(str).map(norm_video_id)

    selected_ids = parse_video_ids(args.video_ids)

    if selected_ids is None:

        video_ids = sorted(layer2_df["video_id"].unique().tolist(), key=lambda x: (tail_video_id(x), x))

    else:

        video_ids = selected_ids

    if args.max_videos and args.max_videos > 0:

        video_ids = video_ids[:int(args.max_videos)]

                               

    index_df = layer2_df.copy()

    index_df["video_id_norm"] = index_df["video_id"].map(norm_video_id)

    index_df["video_tail"] = index_df["video_id"].map(tail_video_id)

    index_df["annotated_video"] = index_df["video_id"].map(

        lambda v: str(video_out_dir / f"{safe_name(v)}_layer1_local_layer2_cross.mp4")

    )

    index_cols = [

        "video_id", "video_tail", "seg_idx", "start_sec", "end_sec", "duration_sec",

        "layer1_meta_state_local", "layer2_state", "annotated_video",

    ]

    index_cols = [c for c in index_cols if c in index_df.columns]

    expert_index_path = out_dir / "expert_segment_index_layer2.csv"

    index_df[index_cols].sort_values(["layer2_state", "video_id", "start_sec"]).to_csv(

        expert_index_path, index=False, encoding="utf-8-sig"

    )

            

    legend_path = out_dir / "layer2_state_legend.png"

    make_layer2_legend(layer2_df, legend_path, names)

    print("=" * 100)

    print("Visualize Layer-1 local states + Layer-2 cross-video states")

    print(f"layer2_csv : {layer2_csv}")

    print(f"t2s_root   : {t2s_root}")

    print(f"video_root : {video_root}")

    print(f"out_dir    : {out_dir}")

    print(f"videos     : {len(video_ids)}")

    print(f"index      : {expert_index_path}")

    print(f"legend     : {legend_path}")

    print("=" * 100)

    manifest_rows = []

    for i, vid in enumerate(video_ids, start=1):

        vid = norm_video_id(vid)

        out_video = video_out_dir / f"{safe_name(vid)}_layer1_local_layer2_cross.mp4"

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

        layer2_sub = prepare_layer2_segments(layer2_df, vid)

        row = {

            "video_id": vid,

            "status": "",

            "video_file": str(video_file) if video_file else "",

            "layer1_csv": str(layer1_csv) if layer1_csv else "",

            "layer2_segments": int(len(layer2_sub)),

            "out_video": str(out_video),

        }

        if video_file is None:

            row["status"] = "missing_video"

            print("  [MISS] video file not found")

            manifest_rows.append(row)

            continue

        if len(layer2_sub) == 0:

            row["status"] = "missing_layer2_segments"

            print("  [MISS] no layer2 segments")

            manifest_rows.append(row)

            continue

        if layer1_csv is None:

            print("  [WARN] layer1 csv not found; L1 overlay will be NA")

        try:

            render_info = render_one_video(

                video_id=vid,

                video_file=video_file,

                layer1_csv=layer1_csv,

                layer2_sub=layer2_sub,

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
