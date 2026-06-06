                       

   

from __future__ import annotations

import argparse

import re

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

                                                              

         

                                                              

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

VISUAL_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\pose_csv")

VIDEO_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\input")

LAYER2_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_prototype_alignment_X1_only_anchor_k5_map_12_2_111_111"

LAYER2_SEGMENT_CSV_DEFAULT = LAYER2_DIR_DEFAULT / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

LAYER2_MANIFEST_DEFAULT = LAYER2_DIR_DEFAULT / "prototype_alignment_anchor_map_build_manifest.csv"

OUT_DIR_DEFAULT = LAYER2_DIR_DEFAULT / "_viz_local_global_skeleton_all_pose_fixed"

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

TIME_COL = "time_sec"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"]

POSE_X_COLS = [

    "center_x",

    "left_shoulder_x", "left_elbow_x", "left_wrist_x",

    "right_shoulder_x", "right_elbow_x", "right_wrist_x",

]

POSE_Y_COLS = [

    "center_y",

    "left_shoulder_y", "left_elbow_y", "left_wrist_y",

    "right_shoulder_y", "right_elbow_y", "right_wrist_y",

]

POSE_COLS = [

    "center_x", "center_y",

    "left_shoulder_x", "left_shoulder_y",

    "left_elbow_x", "left_elbow_y",

    "left_wrist_x", "left_wrist_y",

    "right_shoulder_x", "right_shoulder_y",

    "right_elbow_x", "right_elbow_y",

    "right_wrist_x", "right_wrist_y",

]

                                                              

         

                                                              

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

def strip_leading_zero_token(s: str) -> str:

    s = str(s).strip()

    if s.isdigit():

        return str(int(s))

    return s

def safe_name(x) -> str:

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s or "item"

def video_id_aliases(x) -> set[str]:

    vid = norm_video_id(x)

    parts = vid.split("/") if vid else []

    aliases = set()

    if vid:

        aliases.add(vid)

    if parts:

        aliases.add(parts[0])

        aliases.add(parts[-1])

        aliases.add(strip_leading_zero_token(parts[0]))

        aliases.add(strip_leading_zero_token(parts[-1]))

        aliases.add("/".join([strip_leading_zero_token(p) for p in parts]))

    t = tail_video_id(vid)

    if t:

        aliases.add(t)

        aliases.add(strip_leading_zero_token(t))

    return {a for a in aliases if a != ""}

def parse_video_id_set(s: str | None) -> set[str]:

    out = set()

    if not s:

        return out

    for x in str(s).replace("，", ",").split(","):

        x = norm_video_id(x)

        if not x:

            continue

        out.update(video_id_aliases(x))

    return out

def video_selected(video_id: str, selected_aliases: set[str]) -> bool:

    if not selected_aliases:

        return True

    return len(video_id_aliases(video_id) & selected_aliases) > 0

def infer_video_id_from_case_dir(root: Path, case_dir: Path) -> str:

    try:

        return norm_video_id(str(case_dir.relative_to(root)).replace("\\", "/"))

    except Exception:

        return norm_video_id(case_dir.name)

def find_case_dir(root: Path, video_id: str) -> Path | None:

    target_alias = video_id_aliases(video_id)

    for p in root.rglob(FINAL_META_CSV):

        case_dir = p.parent

        vid = infer_video_id_from_case_dir(root, case_dir)

        if video_id_aliases(vid) & target_alias:

            return case_dir

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

    for alias in video_id_aliases(vid):

        candidates.append(visual_root / alias / f"{alias}.csv")

        candidates.append(visual_root / alias / "teacher_visual_15d.csv")

        candidates.append(visual_root / f"{alias}.csv")

    for p in candidates:

        if p.exists():

            return p

    if visual_root.exists():

        target_aliases = video_id_aliases(vid)

        for p in visual_root.rglob("*.csv"):

            try:

                rel_no_suffix = norm_video_id(str(p.relative_to(visual_root).with_suffix("")))

            except Exception:

                rel_no_suffix = norm_video_id(p.stem)

            aliases = video_id_aliases(rel_no_suffix)

            aliases.add(p.stem)

            aliases.add(strip_leading_zero_token(p.stem))

            if aliases & target_aliases:

                return p

            if p.name == "teacher_visual_15d.csv":

                parent_aliases = video_id_aliases(p.parent.name)

                if parent_aliases & target_aliases:

                    return p

    return None

def find_video_file(video_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    parts = vid.split("/") if vid else [tail]

    candidates = []

    if parts:

        for ext in VIDEO_EXTS:

            candidates.append(video_root / Path(*parts) / f"{parts[-1]}{ext}")

            candidates.append(video_root / Path(*parts[:-1]) / f"{parts[-1]}{ext}")

    for ext in VIDEO_EXTS:

        candidates.append(video_root / tail / f"{tail}{ext}")

        candidates.append(video_root / f"{tail}{ext}")

    for alias in video_id_aliases(vid):

        for ext in VIDEO_EXTS:

            candidates.append(video_root / alias / f"{alias}{ext}")

            candidates.append(video_root / f"{alias}{ext}")

    seen = set()

    uniq = []

    for p in candidates:

        sp = str(p).lower()

        if sp not in seen:

            uniq.append(p)

            seen.add(sp)

    for p in uniq:

        if p.exists():

            return p

    if video_root.exists():

        target_aliases = video_id_aliases(vid)

        for ext in VIDEO_EXTS:

            for p in video_root.rglob(f"*{ext}"):

                aliases = video_id_aliases(p.stem)

                try:

                    rel = norm_video_id(str(p.relative_to(video_root).with_suffix("")))

                    aliases |= video_id_aliases(rel)

                except Exception:

                    pass

                if aliases & target_aliases:

                    return p

    return None

def auto_find_layer2_outputs(t2s_root: Path):

    

       

    t2s_root = Path(t2s_root)

    candidates = []

    if not t2s_root.exists():

        return None, None, None

    for p in t2s_root.rglob("layer2_cross_video_prototype_aligned_segments_X1_only.csv"):

        p = Path(p)

        parent = p.parent

        parent_name = parent.name

        lower_parts = [x.lower() for x in parent.parts]

        if any("_viz" in x or "_archive" in x for x in lower_parts):

            continue

        score = 0

        if "anchor" in parent_name:

            score += 100

        if "map_12_2_111" in parent_name:

            score += 50

        if parent_name.endswith("_111_111"):

            score += 30

        if "X1_only" in parent_name:

            score += 10

        try:

            mtime = p.stat().st_mtime

        except Exception:

            mtime = 0

        candidates.append((score, mtime, parent, p))

    if not candidates:

        return None, None, None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    parent = candidates[0][2]

    segment_csv = candidates[0][3]

    manifest = parent / "prototype_alignment_anchor_map_build_manifest.csv"

    if not manifest.exists():

        manifest = parent / "prototype_alignment_X1_only_build_manifest.csv"

    return parent, segment_csv, manifest

def merge_asof_index(left_times: np.ndarray, right_times: np.ndarray) -> np.ndarray:

    left_times = np.asarray(left_times, dtype=float)

    right_times = np.asarray(right_times, dtype=float)

    if len(right_times) == 0:

        return np.zeros(len(left_times), dtype=int)

    pos = np.searchsorted(right_times, left_times, side="left")

    pos = np.clip(pos, 0, len(right_times) - 1)

    prev_pos = np.clip(pos - 1, 0, len(right_times) - 1)

    choose_prev = np.abs(left_times - right_times[prev_pos]) <= np.abs(left_times - right_times[pos])

    return np.where(choose_prev, prev_pos, pos).astype(int)

                                                              

              

                                                              

def infer_pose_transform(

    visual_df: pd.DataFrame,

    draw_w: int,

    draw_h: int,

    raw_w: int,

    raw_h: int,

    pose_coord_mode: str = "auto",

    pose_source_width: int = 0,

    pose_source_height: int = 0,

):

    

       

    missing = [c for c in POSE_COLS if c not in visual_df.columns]

    if missing:

        return {

            "mode": "missing",

            "missing": missing,

            "x_min": None, "x_max": None,

            "y_min": None, "y_max": None,

            "sx": 1.0, "sy": 1.0,

        }

    xs = visual_df[POSE_X_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()

    ys = visual_df[POSE_Y_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()

    xs = xs[np.isfinite(xs)]

    ys = ys[np.isfinite(ys)]

    if len(xs) == 0 or len(ys) == 0:

        return {

            "mode": "empty",

            "missing": [],

            "x_min": None, "x_max": None,

            "y_min": None, "y_max": None,

            "sx": 1.0, "sy": 1.0,

        }

    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))

    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))

    max_abs = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    mode = str(pose_coord_mode).lower().strip()

    if mode == "auto":

        if max_abs <= 2.0:

            mode = "normalized"

        elif pose_source_width > 0 and pose_source_height > 0:

                                

            mode = "source_pixel"

        elif x_max <= draw_w * 1.25 and y_max <= draw_h * 1.25 and x_min >= -draw_w * 0.25 and y_min >= -draw_h * 0.25:

                                                             

            mode = "draw_pixel"

        elif x_max <= raw_w * 1.25 and y_max <= raw_h * 1.25 and x_min >= -raw_w * 0.25 and y_min >= -raw_h * 0.25:

                        

            mode = "original_pixel"

        else:

                                          

            mode = "draw_pixel"

    if mode == "normalized":

        sx, sy = float(draw_w), float(draw_h)

    elif mode == "source_pixel":

        sw = float(pose_source_width if pose_source_width > 0 else draw_w)

        sh = float(pose_source_height if pose_source_height > 0 else draw_h)

        sx, sy = float(draw_w) / max(1.0, sw), float(draw_h) / max(1.0, sh)

    elif mode == "original_pixel":

        sx, sy = float(draw_w) / max(1.0, float(raw_w)), float(draw_h) / max(1.0, float(raw_h))

    elif mode == "draw_pixel":

        sx, sy = 1.0, 1.0

    else:

                               

        sx, sy = 1.0, 1.0

    return {

        "mode": mode,

        "missing": [],

        "x_min": x_min, "x_max": x_max,

        "y_min": y_min, "y_max": y_max,

        "sx": sx, "sy": sy,

        "draw_w": int(draw_w), "draw_h": int(draw_h),

        "raw_w": int(raw_w), "raw_h": int(raw_h),

        "pose_source_width": int(pose_source_width),

        "pose_source_height": int(pose_source_height),

    }

def transform_pose_point(px, py, frame_w: int, frame_h: int, pose_tf: dict):

    try:

        px = float(px)

        py = float(py)

    except Exception:

        return None

    if not np.isfinite(px) or not np.isfinite(py):

        return None

    mode = str(pose_tf.get("mode", "draw_pixel"))

    if mode in {"missing", "empty"}:

        return None

    sx = float(pose_tf.get("sx", 1.0))

    sy = float(pose_tf.get("sy", 1.0))

    x = int(round(px * sx))

    y = int(round(py * sy))

                     

    if x < -100 or x > frame_w + 100 or y < -100 or y > frame_h + 100:

        return None

                   

    x = max(0, min(frame_w - 1, x))

    y = max(0, min(frame_h - 1, y))

    return (x, y)

def draw_skeleton(frame, row, pose_tf: dict):

    

       

    h, w = frame.shape[:2]

    for k in POSE_COLS:

        if k not in row:

            return frame

    points = {

        "center": transform_pose_point(row["center_x"], row["center_y"], w, h, pose_tf),

        "ls": transform_pose_point(row["left_shoulder_x"], row["left_shoulder_y"], w, h, pose_tf),

        "le": transform_pose_point(row["left_elbow_x"], row["left_elbow_y"], w, h, pose_tf),

        "lw": transform_pose_point(row["left_wrist_x"], row["left_wrist_y"], w, h, pose_tf),

        "rs": transform_pose_point(row["right_shoulder_x"], row["right_shoulder_y"], w, h, pose_tf),

        "re": transform_pose_point(row["right_elbow_x"], row["right_elbow_y"], w, h, pose_tf),

        "rw": transform_pose_point(row["right_wrist_x"], row["right_wrist_y"], w, h, pose_tf),

    }

    bones = [

        ("ls", "le"),

        ("le", "lw"),

        ("rs", "re"),

        ("re", "rw"),

        ("ls", "rs"),

        ("center", "ls"),

        ("center", "rs"),

    ]

                      

    for a, b in bones:

        if points[a] is not None and points[b] is not None:

            cv2.line(frame, points[a], points[b], (0, 255, 255), 4, cv2.LINE_AA)

    for k, p in points.items():

        if p is None:

            continue

        color = (255, 255, 255) if k == "center" else (0, 255, 0)

        radius = 6 if k == "center" else 8

        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)

        cv2.circle(frame, p, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

                                                              

            

                                                              

def color_for_state(state: int, scheme: str = "global") -> tuple[int, int, int]:

    if int(state) < 0:

        return (80, 80, 80)

    global_palette = [

        (82, 168, 50), (48, 124, 218), (220, 180, 0), (170, 70, 180), (52, 189, 235),

        (60, 60, 220), (200, 90, 30), (130, 130, 40), (80, 200, 200), (200, 120, 180),

    ]

    local_palette = [

        (40, 180, 240), (40, 220, 120), (240, 180, 40), (200, 80, 220), (80, 100, 240),

        (60, 200, 200), (220, 80, 80), (160, 180, 60), (180, 120, 200), (100, 220, 180),

        (200, 160, 60), (100, 140, 220), (220, 120, 120), (150, 90, 200), (120, 200, 120),

        (220, 100, 160),

    ]

    pal = global_palette if scheme == "global" else local_palette

    return pal[int(state) % len(pal)]

def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1, bg=True):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(str(text), font, scale, thickness)

    if bg:

        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 4, y + baseline + 3), (0, 0, 0), -1)

    cv2.putText(img, str(text), (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_state_badge(img, label, state, x, y, scheme="global"):

    c = color_for_state(int(state), scheme=scheme)

    cv2.rectangle(img, (x, y - 22), (x + 150, y + 6), (30, 30, 30), -1)

    cv2.rectangle(img, (x, y - 22), (x + 26, y + 6), c, -1)

    draw_text(img, f"{label}: {state}", x + 34, y, scale=0.6, color=(255, 255, 255), thickness=2, bg=False)

def draw_state_timeline(canvas, states, current_idx, x, y, w, h, label, scheme="global"):

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 35), -1)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)

    states = np.asarray(states, dtype=int)

    n = len(states)

    if n == 0:

        draw_text(canvas, f"{label}: empty", x + 6, y + 18, scale=0.55)

        return

    for px in range(w):

        idx = min(n - 1, int(px / max(1, w) * n))

        st = int(states[idx])

        color = color_for_state(st, scheme=scheme)

        cv2.line(canvas, (x + px, y + 20), (x + px, y + h - 4), color, 1)

    draw_text(canvas, label, x + 6, y + 16, scale=0.55, color=(255, 255, 255), thickness=2, bg=False)

    frac = 0.0 if n <= 1 else float(current_idx) / float(max(1, n - 1))

    cur_x = int(x + frac * (w - 1))

    cv2.line(canvas, (cur_x, y + 18), (cur_x, y + h - 2), (255, 255, 255), 2)

                                                              

         

                                                              

def build_final_global_state(final_df: pd.DataFrame, seg_df_one_video: pd.DataFrame) -> np.ndarray:

    meta_seq_full = pd.to_numeric(final_df["meta_state"], errors="coerce").fillna(-1).astype(int)

    if "is_teacher_frame" in final_df.columns:

        teacher_mask = pd.to_numeric(final_df["is_teacher_frame"], errors="coerce").fillna(1).astype(int).eq(1)

        teacher_mask = teacher_mask & meta_seq_full.ge(0)

    else:

        teacher_mask = meta_seq_full.ge(0)

    keep_idx = np.where(teacher_mask.to_numpy())[0]

    out = np.full(len(final_df), -1, dtype=int)

    for _, r in seg_df_one_video.iterrows():

        try:

            s = int(r["frame_start_idx_teacher"])

            e = int(r["frame_end_idx_teacher"])

            gs = int(r["global_state"])

        except Exception:

            continue

        if len(keep_idx) == 0:

            continue

        s = max(0, min(s, len(keep_idx) - 1))

        e = max(0, min(e, len(keep_idx) - 1))

        if e < s:

            s, e = e, s

        out[keep_idx[s:e + 1]] = gs

    return out

def align_final_to_visual(visual_df: pd.DataFrame, final_df: pd.DataFrame, global_state_final: np.ndarray):

    local_final = pd.to_numeric(final_df["meta_state"], errors="coerce").fillna(-1).astype(int).to_numpy()

    if "is_teacher_frame" in final_df.columns:

        teacher_final = pd.to_numeric(final_df["is_teacher_frame"], errors="coerce").fillna(1).astype(int).to_numpy()

    else:

        teacher_final = (local_final >= 0).astype(int)

    if len(visual_df) == len(final_df):

        idx = np.arange(len(visual_df), dtype=int)

    else:

        if TIME_COL in visual_df.columns:

            visual_t = pd.to_numeric(visual_df[TIME_COL], errors="coerce").ffill().bfill().to_numpy(dtype=float)

        else:

            visual_t = np.arange(len(visual_df), dtype=float) / 12.0

        if TIME_COL in final_df.columns:

            final_t = pd.to_numeric(final_df[TIME_COL], errors="coerce").ffill().bfill().to_numpy(dtype=float)

        else:

            final_t = np.arange(len(final_df), dtype=float) / 12.0

        idx = merge_asof_index(visual_t, final_t)

    out = visual_df.copy()

    out["_final_idx"] = idx

    out["local_state"] = local_final[idx].astype(int)

    out["global_state"] = global_state_final[idx].astype(int)

    out["is_teacher_frame_aligned"] = teacher_final[idx].astype(int)

    return out

def build_frame_to_visual_index(video_fps: float, n_frames: int, visual_times: np.ndarray) -> np.ndarray:

    if not np.isfinite(video_fps) or video_fps <= 1e-6:

        video_fps = 25.0

    frame_times = np.arange(n_frames, dtype=float) / float(video_fps)

    return merge_asof_index(frame_times, visual_times)

                                                              

          

                                                              

def process_one_video(

    video_id,

    role,

    t2s_root,

    visual_root,

    video_root,

    seg_df_all,

    out_dir,

    max_frame_width=1280,

    skip_existing=False,

    pose_coord_mode="auto",

    pose_source_width=0,

    pose_source_height=0,

    debug_pose=False,

):

    video_id = norm_video_id(video_id)

    out_subdir = out_dir / safe_name(video_id)

    ensure_dir(out_subdir)

    out_path = out_subdir / f"{safe_name(video_id)}_local_global_skeleton.mp4"

    if skip_existing and out_path.exists():

        return {"video_id": video_id, "status": "skip_existing", "out_path": str(out_path)}

    case_dir = find_case_dir(t2s_root, video_id)

    if case_dir is None:

        return {"video_id": video_id, "status": "skip", "reason": "case_dir_not_found"}

    final_csv = case_dir / FINAL_META_CSV

    if not final_csv.exists():

        return {"video_id": video_id, "status": "skip", "reason": "final_meta_missing"}

    visual_csv = find_visual_csv(visual_root, video_id)

    if visual_csv is None:

        return {"video_id": video_id, "status": "skip", "reason": "visual_csv_not_found"}

    video_file = find_video_file(video_root, video_id)

    if video_file is None:

        return {"video_id": video_id, "status": "skip", "reason": "video_file_not_found"}

    final_df = pd.read_csv(final_csv)

    visual_df = pd.read_csv(visual_csv)

    seg_df_one = seg_df_all[seg_df_all["video_id"].astype(str).map(norm_video_id).eq(video_id)].copy()

    global_state_final = build_final_global_state(final_df, seg_df_one)

    visual_aligned = align_final_to_visual(visual_df, final_df, global_state_final)

    if TIME_COL in visual_aligned.columns:

        visual_times = pd.to_numeric(visual_aligned[TIME_COL], errors="coerce").ffill().bfill().to_numpy(dtype=float)

    else:

        visual_times = np.arange(len(visual_aligned), dtype=float) / 12.0

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():

        return {"video_id": video_id, "status": "skip", "reason": "video_open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 1e-6:

        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:

        total_frames = max(1, len(visual_aligned))

    frame_to_visual = build_frame_to_visual_index(fps, total_frames, visual_times)

    ok, frame0 = cap.read()

    if not ok or frame0 is None:

        cap.release()

        return {"video_id": video_id, "status": "skip", "reason": "video_first_frame_failed"}

    raw_h, raw_w = frame0.shape[:2]

    scale_xy = min(1.0, float(max_frame_width) / float(max(1, raw_w)))

    draw_w = int(round(raw_w * scale_xy))

    draw_h = int(round(raw_h * scale_xy))

    pose_tf = infer_pose_transform(

        visual_aligned,

        draw_w=draw_w,

        draw_h=draw_h,

        raw_w=raw_w,

        raw_h=raw_h,

        pose_coord_mode=pose_coord_mode,

        pose_source_width=int(pose_source_width),

        pose_source_height=int(pose_source_height),

    )

    if debug_pose:

        print(f"  [POSE] video={video_id}")

        print(f"         visual_csv={visual_csv}")

        print(f"         video_file={video_file}")

        print(f"         raw_size={raw_w}x{raw_h}, draw_size={draw_w}x{draw_h}")

        print(

            f"         coord_range: "

            f"x=[{pose_tf.get('x_min')}, {pose_tf.get('x_max')}], "

            f"y=[{pose_tf.get('y_min')}, {pose_tf.get('y_max')}]"

        )

        print(

            f"         inferred_mode={pose_tf.get('mode')}, "

            f"sx={pose_tf.get('sx')}, sy={pose_tf.get('sy')}"

        )

    panel_w = 420

    bottom_h = 170

    canvas_w = draw_w + panel_w

    canvas_h = draw_h + bottom_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    if not writer.isOpened():

        cap.release()

        return {"video_id": video_id, "status": "skip", "reason": "writer_open_failed"}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    local_states_all = visual_aligned["local_state"].astype(int).to_numpy()

    global_states_all = visual_aligned["global_state"].astype(int).to_numpy()

    processed = 0

    while True:

        ok, frame = cap.read()

        if not ok or frame is None:

            break

        fi = processed

        if fi >= len(frame_to_visual):

            break

        vi = int(frame_to_visual[fi])

        vi = max(0, min(vi, len(visual_aligned) - 1))

        row = visual_aligned.iloc[vi]

        if scale_xy != 1.0:

            frame = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_AREA)

        else:

            frame = frame.copy()

        draw_skeleton(frame, row, pose_tf)

        local_state = int(row["local_state"])

        global_state = int(row["global_state"])

        is_teacher = int(row.get("is_teacher_frame_aligned", 1))

        draw_state_badge(frame, "local", local_state, 14, 30, scheme="local")

        draw_state_badge(frame, "global", global_state, 14, 66, scheme="global")

        draw_text(frame, f"teacher={is_teacher}", 14, 98, scale=0.55)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        canvas[:draw_h, :draw_w] = frame

        px = draw_w + 12

        draw_text(canvas, f"video_id: {video_id}", px, 28, scale=0.65, thickness=2)

        draw_text(canvas, f"role: {role}", px, 58, scale=0.62, thickness=2)

        draw_text(canvas, f"frame: {fi+1}/{total_frames}", px, 92, scale=0.60, thickness=2)

        draw_text(canvas, f"time: {fi / max(1e-6, fps):.2f}s", px, 122, scale=0.60, thickness=2)

        draw_text(canvas, f"visual_row: {vi+1}/{len(visual_aligned)}", px, 152, scale=0.60, thickness=2)

        draw_text(canvas, f"pose_mode: {pose_tf.get('mode')}", px, 182, scale=0.56, thickness=2)

        draw_text(canvas, f"pose_s: {pose_tf.get('sx'):.3f},{pose_tf.get('sy'):.3f}", px, 210, scale=0.52, thickness=2)

        draw_state_badge(canvas, "local", local_state, px, 250, scheme="local")

        draw_state_badge(canvas, "global", global_state, px, 286, scheme="global")

        if str(role).strip().lower().startswith("target"):

            draw_text(canvas, "posterior-mapped target video", px, 324, scale=0.56, color=(0, 255, 255), thickness=2)

        draw_state_timeline(canvas, local_states_all, vi, 18, draw_h + 16, canvas_w - 36, 60, "local state timeline", scheme="local")

        draw_state_timeline(canvas, global_states_all, vi, 18, draw_h + 88, canvas_w - 36, 60, "global state timeline", scheme="global")

        writer.write(canvas)

        processed += 1

    writer.release()

    cap.release()

    return {

        "video_id": video_id,

        "status": "ok",

        "frames_written": processed,

        "fps": float(fps),

        "role": role,

        "out_path": str(out_path),

        "video_file": str(video_file),

        "visual_csv": str(visual_csv),

        "final_csv": str(final_csv),

        "pose_mode": str(pose_tf.get("mode")),

        "pose_sx": float(pose_tf.get("sx", 1.0)),

        "pose_sy": float(pose_tf.get("sy", 1.0)),

        "pose_x_min": pose_tf.get("x_min"),

        "pose_x_max": pose_tf.get("x_max"),

        "pose_y_min": pose_tf.get("y_min"),

        "pose_y_max": pose_tf.get("y_max"),

    }

                                                              

        

                                                              

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--visual-root", type=str, default=str(VISUAL_ROOT_DEFAULT))

    ap.add_argument("--video-root", type=str, default=str(VIDEO_ROOT_DEFAULT))

    ap.add_argument("--layer2-segment-csv", type=str, default=str(LAYER2_SEGMENT_CSV_DEFAULT))

    ap.add_argument("--layer2-manifest", type=str, default=str(LAYER2_MANIFEST_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--video-ids", type=str, default="", help="逗号分隔；为空则默认全部视频")

    ap.add_argument("--skip-existing", action="store_true")

    ap.add_argument("--max-frame-width", type=int, default=1280)

              

    ap.add_argument(

        "--pose-coord-mode",

        type=str,

        default="auto",

        choices=["auto", "normalized", "draw_pixel", "original_pixel", "source_pixel"],

        help="pose_csv 坐标模式。默认 auto。若骨架仍不出现，可手动指定 normalized/draw_pixel/original_pixel/source_pixel。",

    )

    ap.add_argument(

        "--pose-source-width",

        type=int,

        default=0,

        help="当 --pose-coord-mode source_pixel 时使用，表示 pose_csv 坐标对应的视频宽度，例如 1280。",

    )

    ap.add_argument(

        "--pose-source-height",

        type=int,

        default=0,

        help="当 --pose-coord-mode source_pixel 时使用，表示 pose_csv 坐标对应的视频高度，例如 720。",

    )

    ap.add_argument("--debug-pose", action="store_true", help="打印每个视频的 pose 坐标范围和推断模式。")

    args = ap.parse_args()

    t2s_root = Path(args.t2s_root)

    visual_root = Path(args.visual_root)

    video_root = Path(args.video_root)

    layer2_segment_csv = Path(args.layer2_segment_csv)

    layer2_manifest = Path(args.layer2_manifest)

    out_dir = Path(args.out_dir)

                                    

    if not layer2_segment_csv.exists():

        found_dir, found_segment, found_manifest = auto_find_layer2_outputs(Path(args.t2s_root))

        if found_segment is not None and found_segment.exists():

            print("[AUTO] 默认 layer2 segment csv 不存在，已自动切换到：")

            print(f"       layer2_dir         = {found_dir}")

            print(f"       layer2_segment_csv = {found_segment}")

            print(f"       layer2_manifest    = {found_manifest}")

            layer2_segment_csv = found_segment

            layer2_manifest = found_manifest

            if str(args.out_dir) == str(OUT_DIR_DEFAULT):

                out_dir = found_dir / "_viz_local_global_skeleton_all_pose_fixed"

        else:

            raise FileNotFoundError(f"找不到 layer2 segment csv: {layer2_segment_csv}")

    ensure_dir(out_dir)

    seg_df_all = pd.read_csv(layer2_segment_csv)

    seg_df_all["video_id"] = seg_df_all["video_id"].astype(str).map(norm_video_id)

    role_map = {}

    if layer2_manifest.exists():

        manifest_df = pd.read_csv(layer2_manifest)

        if "video_id" in manifest_df.columns and "role" in manifest_df.columns:

            for _, r in manifest_df.iterrows():

                role_map[norm_video_id(r["video_id"])] = str(r["role"])

    all_video_ids = sorted(pd.unique(seg_df_all["video_id"]).tolist(), key=lambda x: (tail_video_id(x), x))

    selected_aliases = parse_video_id_set(args.video_ids)

    run_video_ids = [vid for vid in all_video_ids if video_selected(vid, selected_aliases)]

    print("=" * 100)

    print("Visualize local state + global state + skeleton - POSE FIXED")

    print(f"n_all_videos      = {len(all_video_ids)}")

    print(f"n_run_videos      = {len(run_video_ids)}")

    print(f"out_dir           = {out_dir}")

    print(f"video_root        = {video_root}")

    print(f"pose_coord_mode   = {args.pose_coord_mode}")

    print(f"pose_source_size  = {args.pose_source_width}x{args.pose_source_height}")

    print("=" * 100)

    results = []

    for i, video_id in enumerate(run_video_ids, start=1):

        role = role_map.get(norm_video_id(video_id), "unknown")

        print(f"[{i}/{len(run_video_ids)}] {video_id} role={role}")

        res = process_one_video(

            video_id=video_id,

            role=role,

            t2s_root=t2s_root,

            visual_root=visual_root,

            video_root=video_root,

            seg_df_all=seg_df_all,

            out_dir=out_dir,

            max_frame_width=int(args.max_frame_width),

            skip_existing=bool(args.skip_existing),

            pose_coord_mode=str(args.pose_coord_mode),

            pose_source_width=int(args.pose_source_width),

            pose_source_height=int(args.pose_source_height),

            debug_pose=bool(args.debug_pose),

        )

        print(f"  -> {res.get('status')} {res.get('reason', '')}")

        if res.get("status") == "ok":

            print(f"     pose_mode={res.get('pose_mode')}, sx={res.get('pose_sx')}, sy={res.get('pose_sy')}")

        results.append(res)

    result_df = pd.DataFrame(results)

    result_csv = out_dir / "visualize_local_global_skeleton_manifest.csv"

    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)

    print("DONE.")

    print(f"result manifest = {result_csv}")

    if len(result_df):

        print(result_df["status"].value_counts(dropna=False).to_string())

    print("=" * 100)

if __name__ == "__main__":

    main()
