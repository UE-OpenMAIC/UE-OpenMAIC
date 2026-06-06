                       

   

from __future__ import annotations

import argparse

import json

import math

import re

from collections import Counter, defaultdict

from pathlib import Path

from typing import Any

import cv2

import numpy as np

import pandas as pd

                           

      

                           

VIDEO_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\input")

ASR_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\surprisal_batch_output")

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch_orientation8")

LAYER2_SEGMENT_CSV_DEFAULT = (

    T2S_ROOT_DEFAULT

    / "_cross_video_prototype_alignment_X1_only_orientation8"

    / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

)

DOC_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\doc\digitalAction")

LOCAL_MOCAP_MAP_XLSX_DEFAULT = DOC_ROOT_DEFAULT / "15动作_local到动捕动作映射表_LLM学习版.xlsx"

MOCAP_DICT_XLSX_DEFAULT = DOC_ROOT_DEFAULT / "15动作_粗类G归属与动捕动作总表.xlsx"

OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\mocap_action_rag")

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

                                            

MARKER_ROI_X = 0

MARKER_ROI_Y = 0

MARKER_ROI_W = 20

MARKER_ROI_H = 20

MARKER_DOMINANT_MIN = 120.0

MARKER_DOMINANT_RATIO = 1.35

                           

      

                           

def ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)

def clean(x: Any) -> str:

    if x is None:

        return ""

    if isinstance(x, float) and math.isnan(x):

        return ""

    return " ".join(str(x).replace("\r", " ").replace("\n", " ").replace("\t", " ").split()).strip()

def norm_video_id(x: Any) -> str:

    s = clean(x).replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

    s = re.sub(r"/+", "/", s).strip("/")

    return s

def tail_video_id(x: Any) -> str:

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def video_aliases(video_id: Any) -> set[str]:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    aliases = {vid, tail}

    if tail.isdigit():

        aliases.add(str(int(tail)))

        aliases.add(f"{int(tail):02d}")

        aliases.add(f"{int(tail)}/{int(tail)}")

        aliases.add(f"{int(tail):02d}/{int(tail):02d}")

    return {a for a in aliases if a}

def safe_float(x: Any, default: float = 0.0) -> float:

    try:

        if pd.isna(x):

            return float(default)

        value = float(x)

        return value if math.isfinite(value) else float(default)

    except Exception:

        return float(default)

def safe_int(x: Any, default: int = -1) -> int:

    try:

        if pd.isna(x):

            return int(default)

        return int(float(x))

    except Exception:

        return int(default)

def parse_time_to_sec(s: Any) -> float:

    if s is None:

        return np.nan

    s = clean(s).replace("：", ":")

    s = s.replace("~", "-").replace("—", "-").replace("－", "-")

    if not s:

        return np.nan

    parts = s.split(":")

    try:

        if len(parts) == 1:

            return float(parts[0])

        if len(parts) == 2:

            return float(parts[0]) * 60.0 + float(parts[1])

        if len(parts) == 3:

            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])

    except Exception:

        return np.nan

    return np.nan

def overlap_seconds(a0, a1, b0, b1) -> float:

    if pd.isna(a0) or pd.isna(a1) or pd.isna(b0) or pd.isna(b1):

        return 0.0

    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))

def join_top_texts(texts, max_items=8, max_len=900) -> str:

    arr = [clean(t) for t in texts if clean(t)]

    out = " | ".join(arr[:max_items])

    return out if len(out) <= max_len else out[:max_len] + "..."

def local_key_aliases(video_id: Any, local_state: Any) -> set[str]:

    ls = safe_int(local_state, -999999)

    aliases = set()

    for v in video_aliases(video_id):

        aliases.add(f"{v}-L{ls}")

        aliases.add(f"{v}/L{ls}")

    tail = tail_video_id(video_id)

    if tail.isdigit():

        aliases.add(f"{int(tail)}/{int(tail)}-L{ls}")

        aliases.add(f"{int(tail)}-L{ls}")

        aliases.add(f"{int(tail):02d}/{int(tail):02d}-L{ls}")

    return {norm_local_key(x) for x in aliases if x}

def norm_local_key(x: Any) -> str:

    s = clean(x).replace("\\", "/")

    s = re.sub(r"/+", "/", s).strip("/")

                  

    m = re.match(r"^(.*?)[-/]L(\d+)$", s, flags=re.IGNORECASE)

    if m:

        return f"{m.group(1)}-L{int(m.group(2))}"

    return s

def make_safe_name(x: Any) -> str:

    s = norm_video_id(x)

    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)

    return s or "unknown"

def canonical_video_id(x: Any) -> str:

    """把 21、21/21、input/21/21.mp4 这类形式尽量统一成 21/21。"""

    vid = norm_video_id(x)

    tail = tail_video_id(vid)

    if tail.isdigit():

        n = int(tail)

        return f"{n}/{n}"

    return vid

                           

      

                           

def find_existing_file_with_keywords(root: Path, preferred: Path, keywords: list[str], suffixes: tuple[str, ...]) -> Path:

    if preferred.exists():

        return preferred

    if root.exists():

        candidates = []

        for p in root.rglob("*"):

            if not p.is_file() or p.suffix.lower() not in suffixes:

                continue

            name = p.name.lower()

            if all(k.lower() in name for k in keywords):

                candidates.append(p)

        if candidates:

            return sorted(candidates, key=lambda p: len(str(p)))[0]

    return preferred

def collect_all_videos(video_root: Path) -> list[Path]:

    if not video_root.exists():

        return []

    videos = [p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

    return sorted(videos)

def find_video_path(video_root: Path, video_id: Any, video_index: dict[str, Path] | None = None) -> Path | None:

    aliases = video_aliases(video_id)

    tail = tail_video_id(video_id)

    candidates = []

    for a in aliases:

        candidates += [

            video_root / a / f"{tail}.mp4",

            video_root / a / f"{a}.mp4",

            video_root / f"{a}.mp4",

        ]

    if tail:

        candidates += [video_root / tail / f"{tail}.mp4", video_root / tail / f"{tail}.avi"]

    for p in candidates:

        if p.exists():

            return p

    if video_index:

        for a in aliases:

            if a in video_index:

                return video_index[a]

    return None

def build_video_index(video_root: Path) -> dict[str, Path]:

    index = {}

    for p in collect_all_videos(video_root):

        keys = {p.stem, p.parent.name, f"{p.parent.name}/{p.stem}"}

        try:

            rel = norm_video_id(str(p.relative_to(video_root).with_suffix("")))

            keys.add(rel)

        except Exception:

            pass

        for k in keys:

            for a in video_aliases(k):

                index.setdefault(a, p)

    return index

def find_asr_txt(asr_root: Path, video_id: Any) -> Path | None:

    aliases = video_aliases(video_id)

    candidates = []

    for a in aliases:

        candidates += [

            asr_root / a / "asr_segments_editable.txt",

            asr_root / a / "asr_segments.txt",

            asr_root / a / "peak_texts.txt",

            asr_root / a / f"{tail_video_id(a)}.txt",

            asr_root / f"{a}.txt",

        ]

    for p in candidates:

        if p.exists():

            return p

    if asr_root.exists():

        editable_hits = []

        any_hits = []

        for p in asr_root.rglob("*.txt"):

            parent_aliases = video_aliases(p.parent.name) | video_aliases(p.stem)

            if aliases & parent_aliases:

                any_hits.append(p)

                if p.name == "asr_segments_editable.txt":

                    editable_hits.append(p)

        if editable_hits:

            return sorted(editable_hits)[0]

        if any_hits:

            return sorted(any_hits)[0]

    return None

def find_final_meta_csv(t2s_root: Path, video_id: Any) -> Path | None:

    aliases = video_aliases(video_id)

    candidates = []

    for a in aliases:

        candidates += [t2s_root / a / a / FINAL_META_CSV, t2s_root / a / FINAL_META_CSV]

    for p in candidates:

        if p.exists():

            return p

    if t2s_root.exists():

        for p in t2s_root.rglob(FINAL_META_CSV):

            try:

                rel = norm_video_id(str(p.parent.relative_to(t2s_root)))

            except Exception:

                rel = norm_video_id(p.parent.name)

            if video_aliases(rel) & aliases or video_aliases(p.parent.name) & aliases:

                return p

    return None

                           

             

                           

def parse_stage_line(line: str):

    line = clean(line)

    if not line:

        return None

    m = re.match(r"^(.+?)[（(]\s*([^~\-—－]+)\s*[~\-—－]\s*([^)）]+)\s*[)）]\s*$", line)

    if not m:

        return None

    start_s = parse_time_to_sec(m.group(2))

    end_s = parse_time_to_sec(m.group(3))

    if pd.isna(start_s) or pd.isna(end_s):

        return None

    return {"stage": m.group(1).strip(), "stage_start_sec": float(start_s), "stage_end_sec": float(end_s)}

def parse_asr_txt(txt_path: Path, video_id: Any) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    stages = []

    rows = []

    for line in raw:

        line = line.strip()

        if not line:

            continue

        if line.lower().replace(" ", "") in {"start_timeend_timetext", "start_time\tend_time\ttext"}:

            continue

        stg = parse_stage_line(line)

        if stg:

            stages.append(stg)

            continue

        parts = re.split(r"\t+", line)

        if len(parts) < 3:

                               

            m = re.match(r"^([^\s~\-—－]+)\s*[~\-—－]\s*([^\s]+)\s+(.+)$", line)

            if m:

                parts = [m.group(1), m.group(2), m.group(3)]

        if len(parts) >= 3:

            start_s = parse_time_to_sec(parts[0])

            end_s = parse_time_to_sec(parts[1])

            text = "\t".join(parts[2:]).strip()

            if not pd.isna(start_s) and not pd.isna(end_s) and text:

                rows.append({

                    "video_id": norm_video_id(video_id),

                    "asr_txt": str(txt_path),

                    "text_start_sec": float(start_s),

                    "text_end_sec": float(end_s),

                    "text_mid_sec": float((start_s + end_s) / 2.0),

                    "text_duration_sec": float(max(0.0, end_s - start_s)),

                    "text": text,

                })

    df = pd.DataFrame(rows)

    stage_df = pd.DataFrame(stages)

    if stage_df.empty:

        if not df.empty:

            df["stage"] = "未知阶段"

            df["stage_start_sec"] = np.nan

            df["stage_end_sec"] = np.nan

        return df, stage_df

    stage_df["video_id"] = norm_video_id(video_id)

    stage_df["stage_duration_sec"] = (stage_df["stage_end_sec"] - stage_df["stage_start_sec"]).clip(lower=0)

    def assign_stage(mid, field="stage"):

        for _, s in stage_df.iterrows():

            if float(s["stage_start_sec"]) <= float(mid) <= float(s["stage_end_sec"]):

                return s[field]

        return "未知阶段" if field == "stage" else np.nan

    if not df.empty:

        df["stage"] = df["text_mid_sec"].map(lambda x: assign_stage(x, "stage"))

        df["stage_start_sec"] = df["text_mid_sec"].map(lambda x: assign_stage(x, "stage_start_sec"))

        df["stage_end_sec"] = df["text_mid_sec"].map(lambda x: assign_stage(x, "stage_end_sec"))

    return df, stage_df

                           

                       

                           

def classify_top_left_marker_bgr(frame: np.ndarray) -> tuple[str, float, float, float]:

    h, w = frame.shape[:2]

    x1 = max(0, min(MARKER_ROI_X, w - 1))

    y1 = max(0, min(MARKER_ROI_Y, h - 1))

    x2 = max(x1 + 1, min(x1 + MARKER_ROI_W, w))

    y2 = max(y1 + 1, min(y1 + MARKER_ROI_H, h))

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:

        return "none", np.nan, np.nan, np.nan

    b, g, r = roi.mean(axis=(0, 1))

    is_red = (

        r >= MARKER_DOMINANT_MIN

        and r >= MARKER_DOMINANT_RATIO * max(g, 1.0)

        and r >= MARKER_DOMINANT_RATIO * max(b, 1.0)

    )

    is_blue = (

        b >= MARKER_DOMINANT_MIN

        and b >= MARKER_DOMINANT_RATIO * max(g, 1.0)

        and b >= MARKER_DOMINANT_RATIO * max(r, 1.0)

    )

    is_green = (

        g >= MARKER_DOMINANT_MIN

        and g >= MARKER_DOMINANT_RATIO * max(r, 1.0)

        and g >= MARKER_DOMINANT_RATIO * max(b, 1.0)

    )

    if is_red:

        return "red", float(b), float(g), float(r)

    if is_blue:

        return "blue", float(b), float(g), float(r)

    if is_green:

        return "green", float(b), float(g), float(r)

    return "none", float(b), float(g), float(r)

def scan_video_marker_samples(video_path: Path, sample_fps: float, skip_colors: set[str]) -> pd.DataFrame:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps and fps > 0 else 0.0

    if duration <= 0:

                           

        duration = 0.0

    step = 1.0 / max(1e-6, float(sample_fps))

    times = np.arange(0.0, max(0.0, duration), step) if duration > 0 else np.array([0.0])

    rows = []

    for idx, t in enumerate(times):

        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)

        ret, frame = cap.read()

        if not ret:

            marker, b, g, r = "read_fail", np.nan, np.nan, np.nan

        else:

            marker, b, g, r = classify_top_left_marker_bgr(frame)

        rows.append({

            "sample_idx": int(idx),

            "time_sec": float(t),

            "marker_type": marker,

            "skip_by_marker": int(marker in skip_colors),

            "roi_mean_b": b,

            "roi_mean_g": g,

            "roi_mean_r": r,

        })

    cap.release()

    return pd.DataFrame(rows)

def marker_at_time(marker_df: pd.DataFrame, t: float) -> str:

    if marker_df.empty:

        return "none"

    idx = int(np.searchsorted(marker_df["time_sec"].to_numpy(dtype=float), float(t), side="right") - 1)

    idx = max(0, min(idx, len(marker_df) - 1))

    return str(marker_df.iloc[idx]["marker_type"])

def filter_asr_by_marker(asr_df: pd.DataFrame, marker_df: pd.DataFrame, overlap_threshold: float, drop_if_midpoint_marked: bool) -> tuple[pd.DataFrame, pd.DataFrame]:

    if asr_df.empty or marker_df.empty:

        out = asr_df.copy()

        out["marker_skip_ratio"] = 0.0

        out["marker_mid_type"] = "none"

        out["dropped_by_marker"] = 0

        return out, pd.DataFrame()

    sample_times = marker_df["time_sec"].to_numpy(dtype=float)

    sample_skip = marker_df["skip_by_marker"].to_numpy(dtype=int)

    rows_keep = []

    rows_drop = []

    for _, r in asr_df.iterrows():

        t0 = safe_float(r["text_start_sec"])

        t1 = safe_float(r["text_end_sec"])

        mid = safe_float(r["text_mid_sec"], (t0 + t1) / 2.0)

        mask = (sample_times >= t0) & (sample_times <= t1)

        if mask.any():

            skip_ratio = float(sample_skip[mask].mean())

        else:

            skip_ratio = 1.0 if marker_at_time(marker_df, mid) in {"red", "blue"} else 0.0

        mid_marker = marker_at_time(marker_df, mid)

        should_drop = skip_ratio >= float(overlap_threshold)

        if drop_if_midpoint_marked and mid_marker in {"red", "blue"}:

            should_drop = True

        item = r.to_dict()

        item.update({

            "marker_skip_ratio": float(skip_ratio),

            "marker_mid_type": mid_marker,

            "dropped_by_marker": int(should_drop),

        })

        if should_drop:

            rows_drop.append(item)

        else:

            rows_keep.append(item)

    return pd.DataFrame(rows_keep), pd.DataFrame(rows_drop)

                           

          

                           

def load_layer2_segments(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    df = pd.read_csv(path)

    required = ["video_id", "local_meta_state", "global_state", "start_sec", "end_sec"]

    missing = [c for c in required if c not in df.columns]

    if missing:

        raise ValueError(f"layer2 segment csv 缺少列: {missing}")

    df = df.copy()

    df["video_id"] = df["video_id"].astype(str).map(norm_video_id)

    df["local_meta_state"] = pd.to_numeric(df["local_meta_state"], errors="coerce").fillna(-1).astype(int)

    df["global_state"] = pd.to_numeric(df["global_state"], errors="coerce").fillna(-1).astype(int)

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")

    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    df["duration_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=0)

    return df

def compress_final_meta_csv(final_csv: Path, video_id: Any) -> pd.DataFrame:

    df = pd.read_csv(final_csv)

    if "time_sec" not in df.columns or "meta_state" not in df.columns:

        return pd.DataFrame()

    df = df.sort_values("time_sec").reset_index(drop=True)

    rows = []

    times = df["time_sec"].to_numpy(dtype=float)

    states = pd.to_numeric(df["meta_state"], errors="coerce").fillna(-1).astype(int).to_numpy()

    if len(times) == 0:

        return pd.DataFrame()

    start = 0

    for i in range(1, len(states) + 1):

        if i == len(states) or states[i] != states[start]:

            st = int(states[start])

            if st >= 0:

                s0 = float(times[start])

                s1 = float(times[i - 1]) if i - 1 < len(times) else float(times[-1])

                                  

                if i < len(times):

                    s1 = float(times[i])

                elif len(times) >= 2:

                    s1 = float(times[-1] + np.median(np.diff(times)))

                rows.append({

                    "video_id": norm_video_id(video_id),

                    "local_meta_state": st,

                    "global_state": -1,

                    "start_sec": s0,

                    "end_sec": max(s0, s1),

                    "duration_sec": max(0.0, s1 - s0),

                    "prototype_id": "",

                    "segment_source": "final_meta_csv",

                })

            start = i

    return pd.DataFrame(rows)

def get_segments_for_video(layer2_df: pd.DataFrame, t2s_root: Path, video_id: Any) -> pd.DataFrame:

    vid = norm_video_id(video_id)

    if not layer2_df.empty:

        mask = layer2_df["video_id"].astype(str).map(norm_video_id).isin(video_aliases(vid))

        seg = layer2_df.loc[mask].copy()

        if not seg.empty:

            seg["segment_source"] = "layer2_cross_video"

            return seg

    final_csv = find_final_meta_csv(t2s_root, vid)

    if final_csv is not None:

        return compress_final_meta_csv(final_csv, vid)

    return pd.DataFrame()

def align_text_to_segments(asr_df: pd.DataFrame, seg_df_video: pd.DataFrame, min_overlap_ratio: float) -> pd.DataFrame:

    rows = []

    if asr_df.empty or seg_df_video.empty:

        return pd.DataFrame(rows)

    seg_df_video = seg_df_video.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    for _, t in asr_df.iterrows():

        t0 = safe_float(t["text_start_sec"])

        t1 = safe_float(t["text_end_sec"])

        dur = max(1e-6, t1 - t0)

        overlaps = []

        for _, s in seg_df_video.iterrows():

            ov = overlap_seconds(t0, t1, s["start_sec"], s["end_sec"])

            if ov > 0:

                overlaps.append((ov, s))

        if not overlaps:

            continue

        overlaps.sort(key=lambda x: x[0], reverse=True)

        best_ov, best_s = overlaps[0]

        overlap_ratio = float(best_ov / dur)

        if overlap_ratio < float(min_overlap_ratio):

            continue

        counter = Counter()

        for ov, s in overlaps:

            counter[f"G{safe_int(s.get('global_state', -1))}_L{safe_int(s.get('local_meta_state', -1))}"] += float(ov)

        item = t.to_dict()

        item.update({

            "global_state": safe_int(best_s.get("global_state", -1)),

            "local_meta_state": safe_int(best_s.get("local_meta_state", -1)),

            "prototype_id": clean(best_s.get("prototype_id", "")),

            "matched_seg_start_sec": safe_float(best_s.get("start_sec")),

            "matched_seg_end_sec": safe_float(best_s.get("end_sec")),

            "overlap_sec": float(best_ov),

            "overlap_ratio": overlap_ratio,

            "all_overlapped_states": json.dumps(dict(counter), ensure_ascii=False),

            "segment_source": clean(best_s.get("segment_source", "")),

        })

        rows.append(item)

    return pd.DataFrame(rows)

                           

            

                           

def read_excel_sheet_by_columns(path: Path, required_cols: list[str]) -> pd.DataFrame:

                                 

    sheets = pd.read_excel(path, sheet_name=None, dtype=str)

    for name, df in sheets.items():

        df = df.copy()

        df.columns = [clean(c) for c in df.columns]

        if all(c in df.columns for c in required_cols):

            df["__sheet_name"] = name

            return df

    raise ValueError(f"{path} 中找不到包含列 {required_cols} 的工作表")

def load_local_mocap_map(path: Path) -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict]]:

    df = read_excel_sheet_by_columns(path, ["local编号", "G编号", "推荐动捕编号", "推荐动捕动作"])

    for col in ["视频编号", "local编号", "G编号", "G名称", "粗类", "专家描述", "专家粗类语义", "状态", "推荐动捕编号", "推荐动捕动作", "LLM学习标签", "是否相悖"]:

        if col not in df.columns:

            df[col] = ""

    df["local_key_norm"] = df["local编号"].map(norm_local_key)

    df["推荐动捕编号"] = df["推荐动捕编号"].map(clean)

    df["推荐动捕动作"] = df["推荐动捕动作"].map(clean)

    df["G编号"] = df["G编号"].map(clean)

    df["is_mocap_usable"] = (

        df["推荐动捕编号"].str.match(r"^MC\d+", na=False)

        & ~df["LLM学习标签"].astype(str).str.contains("NO_MOCAP", na=False)

    ).astype(int)

    exact = {}

    for _, r in df.iterrows():

        if int(r["is_mocap_usable"]) != 1:

            continue

        exact.setdefault(r["local_key_norm"], r.to_dict())

                                         

    by_g = {}

    for gid, g in df[df["is_mocap_usable"] == 1].groupby("G编号"):

        g2 = g[~g["是否相悖"].astype(str).str.contains("是", na=False)].copy()

        if g2.empty:

            g2 = g.copy()

        counts = Counter(g2["推荐动捕编号"].astype(str).tolist())

        if not counts:

            continue

        mc = counts.most_common(1)[0][0]

        row = g2[g2["推荐动捕编号"].astype(str).eq(mc)].iloc[0].to_dict()

        row["g_fallback_count"] = counts[mc]

        by_g[gid] = row

    return df, exact, by_g

def load_mocap_action_dict(path: Path) -> pd.DataFrame:

    df = read_excel_sheet_by_columns(path, ["动捕编号", "动捕动作名称"])

    for col in ["动捕编号", "动捕动作名称", "所属粗类", "说明", "代表local与专家描述", "支撑local数"]:

        if col not in df.columns:

            df[col] = ""

    return df[["动捕编号", "动捕动作名称", "所属粗类", "说明", "代表local与专家描述", "支撑local数"]].copy()

def attach_mocap_mapping(aligned_df: pd.DataFrame, exact_map: dict[str, dict], g_map: dict[str, dict], action_dict_df: pd.DataFrame) -> pd.DataFrame:

    if aligned_df.empty:

        return aligned_df

    action_dict = {clean(r["动捕编号"]): r.to_dict() for _, r in action_dict_df.iterrows()}

    rows = []

    for _, r in aligned_df.iterrows():

        item = r.to_dict()

        vid = norm_video_id(item.get("video_id", ""))

        local_state = safe_int(item.get("local_meta_state", -1))

        global_state = safe_int(item.get("global_state", -1))

        mapping_row = None

        source = "none"

        for key in local_key_aliases(vid, local_state):

            if key in exact_map:

                mapping_row = exact_map[key]

                source = "exact_local"

                break

        if mapping_row is None and global_state >= 0:

            gid = f"G{global_state}"

            if gid in g_map:

                mapping_row = g_map[gid]

                source = "global_majority_fallback"

        if mapping_row is not None:

            mc_id = clean(mapping_row.get("推荐动捕编号", ""))

            d = action_dict.get(mc_id, {})

            item.update({

                "local_key_for_mapping": f"{vid}-L{local_state}",

                "mocap_id": mc_id,

                "mocap_action": clean(mapping_row.get("推荐动捕动作", "")) or clean(d.get("动捕动作名称", "")),

                "mocap_coarse": clean(d.get("所属粗类", mapping_row.get("粗类", ""))),

                "mocap_description": clean(d.get("说明", "")),

                "mocap_representative_locals": clean(d.get("代表local与专家描述", "")),

                "mapping_source": source,

                "mapping_G编号": clean(mapping_row.get("G编号", "")),

                "mapping_G名称": clean(mapping_row.get("G名称", "")),

                "mapping_粗类": clean(mapping_row.get("粗类", "")),

                "mapping_专家描述": clean(mapping_row.get("专家描述", "")),

                "mapping_专家粗类语义": clean(mapping_row.get("专家粗类语义", "")),

                "mapping_是否相悖": clean(mapping_row.get("是否相悖", "")),

            })

        else:

            item.update({

                "local_key_for_mapping": f"{vid}-L{local_state}",

                "mocap_id": "",

                "mocap_action": "",

                "mocap_coarse": "",

                "mocap_description": "",

                "mocap_representative_locals": "",

                "mapping_source": "none",

                "mapping_G编号": f"G{global_state}" if global_state >= 0 else "",

                "mapping_G名称": "",

                "mapping_粗类": "",

                "mapping_专家描述": "",

                "mapping_专家粗类语义": "",

                "mapping_是否相悖": "",

            })

        rows.append(item)

    return pd.DataFrame(rows)

                           

          

                           

def build_action_examples_df(aligned_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if aligned_df.empty:

        return pd.DataFrame(rows)

    usable = aligned_df[aligned_df["mocap_id"].astype(str).str.match(r"^MC\d+", na=False)].copy()

    for mc_id, g in usable.groupby("mocap_id"):

        rows.append({

            "mocap_id": mc_id,

            "mocap_action": clean(g["mocap_action"].iloc[0]),

            "mocap_coarse": clean(g["mocap_coarse"].iloc[0]),

            "n_sentences": int(len(g)),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "top_stages": json.dumps(Counter(g["stage"].astype(str).tolist()).most_common(8), ensure_ascii=False),

            "top_global_states": json.dumps(Counter(g["global_state"].astype(str).tolist()).most_common(8), ensure_ascii=False),

            "example_texts": join_top_texts(g.sort_values("overlap_ratio", ascending=False)["text"].astype(str).tolist(), max_items=14, max_len=1500),

            "example_expert_descriptions": join_top_texts(g["mapping_专家描述"].astype(str).tolist(), max_items=12, max_len=1000),

        })

    return pd.DataFrame(rows).sort_values(["n_sentences", "support_videos"], ascending=[False, False]).reset_index(drop=True)

def build_mocap_transition_df(aligned_df: pd.DataFrame) -> pd.DataFrame:

    raw = []

    if aligned_df.empty:

        return pd.DataFrame(), pd.DataFrame()

    usable = aligned_df[aligned_df["mocap_id"].astype(str).str.match(r"^MC\d+", na=False)].copy()

    for vid, g in usable.groupby("video_id"):

        g = g.sort_values(["text_start_sec", "text_end_sec"]).reset_index(drop=True)

        for i in range(len(g) - 1):

            a, b = g.iloc[i], g.iloc[i + 1]

            raw.append({

                "video_id": norm_video_id(vid),

                "from_mocap_id": clean(a["mocap_id"]),

                "from_mocap_action": clean(a["mocap_action"]),

                "to_mocap_id": clean(b["mocap_id"]),

                "to_mocap_action": clean(b["mocap_action"]),

                "transition_key": f"{clean(a['mocap_id'])}->{clean(b['mocap_id'])}",

                "stage": clean(b.get("stage", "")),

                "nearby_texts": join_top_texts([a.get("text", ""), b.get("text", "")], max_items=2, max_len=500),

            })

    raw_df = pd.DataFrame(raw)

    rows = []

    if raw_df.empty:

        return raw_df, pd.DataFrame()

    for key, g in raw_df.groupby(["from_mocap_id", "to_mocap_id", "transition_key"]):

        rows.append({

            "from_mocap_id": key[0],

            "to_mocap_id": key[1],

            "transition_key": key[2],

            "count": int(len(g)),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "top_stages": json.dumps(Counter(g["stage"].astype(str).tolist()).most_common(8), ensure_ascii=False),

            "example_texts": join_top_texts(g["nearby_texts"].astype(str).tolist(), max_items=8, max_len=1000),

        })

    return raw_df, pd.DataFrame(rows).sort_values(["count", "support_videos"], ascending=[False, False]).reset_index(drop=True)

def write_jsonl(items: list[dict], path: Path) -> None:

    with path.open("w", encoding="utf-8") as f:

        for item in items:

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def build_rag_docs(aligned_df: pd.DataFrame, action_examples_df: pd.DataFrame, transition_df: pd.DataFrame, action_dict_df: pd.DataFrame) -> list[dict]:

    docs = []

    def add_doc(doc_type: str, title: str, content: str, metadata: dict):

        content = clean(content)

        if not content:

            return

        docs.append({

            "doc_id": f"{doc_type}_{len(docs):08d}",

            "doc_type": doc_type,

            "title": clean(title),

            "content": content[:7000],

            "metadata": metadata,

        })

    usable = aligned_df[aligned_df["mocap_id"].astype(str).str.match(r"^MC\d+", na=False)].copy()

    for _, r in usable.iterrows():

        title = f"文本-动捕样本 | {clean(r.get('mocap_id'))} | {clean(r.get('stage'))} | {norm_video_id(r.get('video_id'))}"

        content = (

            f"教师文本：{clean(r.get('text'))}\n"

            f"课程阶段：{clean(r.get('stage'))}\n"

            f"动捕编号：{clean(r.get('mocap_id'))}\n"

            f"动捕动作：{clean(r.get('mocap_action'))}\n"

            f"动捕粗类：{clean(r.get('mocap_coarse'))}\n"

            f"动捕说明：{clean(r.get('mocap_description'))}\n"

            f"原始动作专家描述：{clean(r.get('mapping_专家描述'))}\n"

            f"G/local：G{safe_int(r.get('global_state'))}/L{safe_int(r.get('local_meta_state'))}\n"

            f"视频与时间：{norm_video_id(r.get('video_id'))} {safe_float(r.get('text_start_sec')):.2f}-{safe_float(r.get('text_end_sec')):.2f}s\n"

            f"过滤说明：已排除红/蓝标记文本；marker_skip_ratio={safe_float(r.get('marker_skip_ratio')):.3f}\n"

            f"映射来源：{clean(r.get('mapping_source'))}；是否相悖：{clean(r.get('mapping_是否相悖'))}"

        )

        add_doc("sentence_mocap", title, content, {

            "video_id": norm_video_id(r.get("video_id")),

            "stage": clean(r.get("stage")),

            "mocap_id": clean(r.get("mocap_id")),

            "mocap_action": clean(r.get("mocap_action")),

            "mocap_coarse": clean(r.get("mocap_coarse")),

            "global_state": safe_int(r.get("global_state")),

            "local_meta_state": safe_int(r.get("local_meta_state")),

            "mapping_source": clean(r.get("mapping_source")),

            "overlap_ratio": safe_float(r.get("overlap_ratio")),

        })

    for _, r in action_examples_df.iterrows():

        title = f"动捕动作语义汇总 | {clean(r.get('mocap_id'))} {clean(r.get('mocap_action'))}"

        content = (

            f"动捕编号：{clean(r.get('mocap_id'))}\n"

            f"动捕动作：{clean(r.get('mocap_action'))}\n"

            f"所属粗类：{clean(r.get('mocap_coarse'))}\n"

            f"支持句子数：{safe_int(r.get('n_sentences'),0)}\n"

            f"支持视频数：{safe_int(r.get('support_videos'),0)}\n"

            f"常见阶段：{clean(r.get('top_stages'))}\n"

            f"真实教师文本样例：{clean(r.get('example_texts'))}\n"

            f"专家动作描述样例：{clean(r.get('example_expert_descriptions'))}"

        )

        add_doc("mocap_summary", title, content, {

            "mocap_id": clean(r.get("mocap_id")),

            "mocap_action": clean(r.get("mocap_action")),

            "mocap_coarse": clean(r.get("mocap_coarse")),

            "n_sentences": safe_int(r.get("n_sentences"), 0),

        })

    for _, r in transition_df.iterrows():

        title = f"动捕转移语法 | {clean(r.get('transition_key'))}"

        content = (

            f"动捕转移：{clean(r.get('transition_key'))}\n"

            f"出现次数：{safe_int(r.get('count'),0)}\n"

            f"支持视频数：{safe_int(r.get('support_videos'),0)}\n"

            f"常见阶段：{clean(r.get('top_stages'))}\n"

            f"附近文本样例：{clean(r.get('example_texts'))}"

        )

        add_doc("mocap_transition", title, content, {

            "from_mocap_id": clean(r.get("from_mocap_id")),

            "to_mocap_id": clean(r.get("to_mocap_id")),

            "transition_key": clean(r.get("transition_key")),

        })

    for _, r in action_dict_df.iterrows():

        mc = clean(r.get("动捕编号"))

        if not mc.startswith("MC"):

            continue

        title = f"15动作字典 | {mc} {clean(r.get('动捕动作名称'))}"

        content = (

            f"动捕编号：{mc}\n"

            f"动捕动作：{clean(r.get('动捕动作名称'))}\n"

            f"所属粗类：{clean(r.get('所属粗类'))}\n"

            f"说明：{clean(r.get('说明'))}\n"

            f"代表local与专家描述：{clean(r.get('代表local与专家描述'))}"

        )

        add_doc("mocap_dictionary", title, content, {

            "mocap_id": mc,

            "mocap_action": clean(r.get("动捕动作名称")),

            "mocap_coarse": clean(r.get("所属粗类")),

        })

    return docs

def write_outputs(out_dir: Path, aligned_all: pd.DataFrame, dropped_all: pd.DataFrame, marker_manifest: pd.DataFrame, action_dict_df: pd.DataFrame) -> None:

    ensure_dir(out_dir)

    aligned_all.to_csv(out_dir / "teacher_text_mocap_alignment_filtered.csv", index=False, encoding="utf-8-sig")

    dropped_all.to_csv(out_dir / "teacher_text_dropped_by_red_blue_marker.csv", index=False, encoding="utf-8-sig")

    marker_manifest.to_csv(out_dir / "marker_scan_manifest.csv", index=False, encoding="utf-8-sig")

    action_dict_df.to_csv(out_dir / "mocap_action_dictionary.csv", index=False, encoding="utf-8-sig")

    action_examples_df = build_action_examples_df(aligned_all)

    action_examples_df.to_csv(out_dir / "mocap_action_text_examples.csv", index=False, encoding="utf-8-sig")

    transition_raw_df, transition_df = build_mocap_transition_df(aligned_all)

    transition_raw_df.to_csv(out_dir / "mocap_transition_raw_examples.csv", index=False, encoding="utf-8-sig")

    transition_df.to_csv(out_dir / "mocap_transition_grammar.csv", index=False, encoding="utf-8-sig")

    docs = build_rag_docs(aligned_all, action_examples_df, transition_df, action_dict_df)

    write_jsonl(docs, out_dir / "mocap_action_rag_docs.jsonl")

    pd.DataFrame([{k: v for k, v in d.items() if k != "content"} for d in docs]).to_csv(out_dir / "mocap_action_rag_doc_index.csv", index=False, encoding="utf-8-sig")

                           

     

                           

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--video-root", type=str, default=str(VIDEO_ROOT_DEFAULT))

    ap.add_argument("--asr-root", type=str, default=str(ASR_ROOT_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--layer2-segment-csv", type=str, default=str(LAYER2_SEGMENT_CSV_DEFAULT))

    ap.add_argument("--doc-root", type=str, default=str(DOC_ROOT_DEFAULT))

    ap.add_argument("--local-mocap-map-xlsx", type=str, default=str(LOCAL_MOCAP_MAP_XLSX_DEFAULT))

    ap.add_argument("--mocap-dict-xlsx", type=str, default=str(MOCAP_DICT_XLSX_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--video-ids", type=str, default="", help="可选，只处理这些视频 ID，用逗号分隔，如 3/3,21/21")

    ap.add_argument("--marker-sample-fps", type=float, default=12.0)

    ap.add_argument("--skip-marker-colors", type=str, default="red,blue", help="默认跳过红/蓝；如需要也跳绿，可写 red,blue,green")

    ap.add_argument("--drop-marker-overlap-threshold", type=float, default=0.15)

    ap.add_argument("--drop-if-midpoint-marked", action="store_true", default=True)

    ap.add_argument("--no-drop-if-midpoint-marked", dest="drop_if_midpoint_marked", action="store_false")

    ap.add_argument("--min-text-action-overlap-ratio", type=float, default=0.10)

    ap.add_argument("--save-marker-masks", action="store_true")

    ap.add_argument("--rebuild", action="store_true", help="兼容批处理参数；当前脚本每次都会重建输出")

    return ap.parse_args()

def main():

    args = parse_args()

    video_root = Path(args.video_root)

    asr_root = Path(args.asr_root)

    t2s_root = Path(args.t2s_root)

    doc_root = Path(args.doc_root)

    out_dir = Path(args.out_dir)

    marker_dir = out_dir / "marker_masks"

    ensure_dir(out_dir)

    if args.save_marker_masks:

        ensure_dir(marker_dir)

    local_map_xlsx = find_existing_file_with_keywords(

        doc_root,

        Path(args.local_mocap_map_xlsx),

        ["local", "动捕", "映射"],

        (".xlsx", ".xls"),

    )

    mocap_dict_xlsx = find_existing_file_with_keywords(

        doc_root,

        Path(args.mocap_dict_xlsx),

        ["15", "动捕", "总表"],

        (".xlsx", ".xls"),

    )

    print("=" * 100)

    print("Build mocap-action RAG library with red/blue marker filtering")

    print(f"VIDEO_ROOT        = {video_root}")

    print(f"ASR_ROOT          = {asr_root}")

    print(f"T2S_ROOT          = {t2s_root}")

    print(f"LAYER2_SEGMENT_CSV= {Path(args.layer2_segment_csv)}")

    print(f"LOCAL_MAP_XLSX    = {local_map_xlsx}")

    print(f"MOCAP_DICT_XLSX   = {mocap_dict_xlsx}")

    print(f"OUT_DIR           = {out_dir}")

    print("=" * 100)

    if not local_map_xlsx.exists():

        raise FileNotFoundError(f"找不到 local->动捕映射表: {local_map_xlsx}")

    if not mocap_dict_xlsx.exists():

        raise FileNotFoundError(f"找不到 15 动作总表/字典: {mocap_dict_xlsx}")

    local_map_df, exact_map, g_map = load_local_mocap_map(local_map_xlsx)

    local_map_df.to_csv(out_dir / "local_to_mocap_map_used.csv", index=False, encoding="utf-8-sig")

    action_dict_df = load_mocap_action_dict(mocap_dict_xlsx)

    layer2_df = load_layer2_segments(Path(args.layer2_segment_csv))

    video_index = build_video_index(video_root)

    if args.video_ids.strip():

        all_video_ids = [canonical_video_id(x) for x in args.video_ids.split(",") if norm_video_id(x)]

    else:

                                                    

                                                                    

        video_ids_from_files = []

        for vp in collect_all_videos(video_root):

            try:

                video_ids_from_files.append(canonical_video_id(str(vp.relative_to(video_root).with_suffix(""))))

            except Exception:

                video_ids_from_files.append(canonical_video_id(vp.stem))

        video_ids_from_layer2 = []

        if not layer2_df.empty:

            video_ids_from_layer2 = [canonical_video_id(x) for x in layer2_df["video_id"].astype(str).tolist()]

        all_video_ids = sorted(set(video_ids_from_files) | set(video_ids_from_layer2), key=lambda x: (tail_video_id(x), x))

    if not all_video_ids:

        raise RuntimeError("没有找到任何待处理视频 ID。请检查 layer2 CSV 或 --video-root。")

    skip_colors = {clean(x).lower() for x in args.skip_marker_colors.split(",") if clean(x)}

    aligned_tables = []

    dropped_tables = []

    manifest_rows = []

    for idx, vid in enumerate(all_video_ids, start=1):

        print(f"[{idx}/{len(all_video_ids)}] video_id={vid}")

        row = {

            "video_id": vid,

            "video_path": "",

            "asr_txt": "",

            "n_asr_raw": 0,

            "n_asr_kept_after_marker": 0,

            "n_asr_dropped_by_marker": 0,

            "n_aligned_text": 0,

            "n_mocap_usable_text": 0,

            "skip_reason": "",

        }

        try:

            video_path = find_video_path(video_root, vid, video_index)

            if video_path is None:

                raise FileNotFoundError(f"找不到视频文件: {vid}")

            row["video_path"] = str(video_path)

            asr_txt = find_asr_txt(asr_root, vid)

            if asr_txt is None:

                raise FileNotFoundError(f"找不到 ASR txt: {vid}")

            row["asr_txt"] = str(asr_txt)

            asr_df, _stage_df = parse_asr_txt(asr_txt, vid)

            if asr_df.empty:

                raise ValueError(f"ASR 解析为空: {asr_txt}")

            row["n_asr_raw"] = int(len(asr_df))

            marker_df = scan_video_marker_samples(video_path, float(args.marker_sample_fps), skip_colors=skip_colors)

            if args.save_marker_masks:

                marker_df.to_csv(marker_dir / f"{make_safe_name(vid)}_marker_samples.csv", index=False, encoding="utf-8-sig")

            asr_keep, asr_drop = filter_asr_by_marker(

                asr_df,

                marker_df,

                overlap_threshold=float(args.drop_marker_overlap_threshold),

                drop_if_midpoint_marked=bool(args.drop_if_midpoint_marked),

            )

            row["n_asr_kept_after_marker"] = int(len(asr_keep))

            row["n_asr_dropped_by_marker"] = int(len(asr_drop))

            if not asr_drop.empty:

                dropped_tables.append(asr_drop)

            seg_df = get_segments_for_video(layer2_df, t2s_root, vid)

            if seg_df.empty:

                raise ValueError(f"找不到可对齐的 layer2/final_meta 动作段: {vid}")

            aligned = align_text_to_segments(asr_keep, seg_df, min_overlap_ratio=float(args.min_text_action_overlap_ratio))

            row["n_aligned_text"] = int(len(aligned))

            if aligned.empty:

                raise ValueError("红/蓝过滤后，文本与动作段对齐为空")

            aligned = attach_mocap_mapping(aligned, exact_map, g_map, action_dict_df)

            usable = aligned[aligned["mocap_id"].astype(str).str.match(r"^MC\d+", na=False)].copy()

            row["n_mocap_usable_text"] = int(len(usable))

            aligned_tables.append(aligned)

            print(

                f"  [OK] asr_raw={row['n_asr_raw']}, kept={row['n_asr_kept_after_marker']}, "

                f"drop={row['n_asr_dropped_by_marker']}, aligned={row['n_aligned_text']}, mocap={row['n_mocap_usable_text']}"

            )

        except Exception as e:

            row["skip_reason"] = repr(e)

            print(f"  [SKIP] {repr(e)}")

        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(out_dir / "build_manifest.csv", index=False, encoding="utf-8-sig")

    if not aligned_tables:

        raise RuntimeError("没有成功生成任何文本-动捕对齐数据，无法构建 RAG 库。")

    aligned_all = pd.concat(aligned_tables, axis=0, ignore_index=True)

    dropped_all = pd.concat(dropped_tables, axis=0, ignore_index=True) if dropped_tables else pd.DataFrame()

    write_outputs(out_dir, aligned_all, dropped_all, manifest_df, action_dict_df)

    print("\n" + "=" * 100)

    print("DONE")

    print(f"RAG library: {out_dir}")

    print("主要输出：")

    print("  - teacher_text_mocap_alignment_filtered.csv")

    print("  - teacher_text_dropped_by_red_blue_marker.csv")

    print("  - mocap_action_text_examples.csv")

    print("  - mocap_transition_grammar.csv")

    print("  - mocap_action_rag_docs.jsonl")

    print("  - mocap_action_rag_doc_index.csv")

    print("下一步：")

    print("  python .\\rag_mocap_action_selector.py plan --text-file .\\demo_lesson.txt")

    print("=" * 100)

if __name__ == "__main__":

    main()
