                       

   

from __future__ import annotations

import argparse

import math

import re

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

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

try:

    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True

except Exception:

    Image = None

    ImageDraw = None

    ImageFont = None

    HAS_PIL = False

DEFAULT_RESULT_ROOT = Path(r"D:\code\teacherT2S\layer12_nlp_pid_t2s_output")

DEFAULT_VIDEO_ROOT = Path(r"D:\code\teacherT2S\yolo\input")

DEFAULT_OUT_ROOT = Path(r"D:\code\teacherT2S\layer12_nlp_pid_t2s_visualized_video")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

PALETTE_RGB = [

    (230, 57, 70), (42, 157, 143), (69, 123, 157), (244, 162, 97),

    (131, 56, 236), (0, 180, 216), (255, 183, 3), (6, 214, 160),

    (239, 71, 111), (17, 138, 178), (118, 120, 237), (40, 167, 69),

    (255, 127, 80), (128, 128, 128), (89, 13, 34), (58, 134, 255),

    (255, 0, 110), (28, 99, 48), (155, 93, 229), (255, 214, 10),

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

def safe_filename(x) -> str:

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    return re.sub(r"_+", "_", s).strip("_") or "video"

def video_id_match(a, b) -> bool:

    a = norm_video_id(a)

    b = norm_video_id(b)

    return a == b or tail_video_id(a) == tail_video_id(b)

def get_bgr(label: int) -> tuple[int, int, int]:

    if int(label) < 0:

        return (160, 160, 160)

    r, g, b = PALETTE_RGB[int(label) % len(PALETTE_RGB)]

    return int(b), int(g), int(r)

def shorten(s: object, max_chars: int = 70) -> str:

    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)

    s = s.replace("\n", " ").replace("\t", " ").strip()

    s = re.sub(r"\s+", " ", s)

    return s if len(s) <= max_chars else s[:max_chars] + "..."

def find_result_csv(result_root: Path, video_id: str, result_name: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    candidates = [

        result_root / Path(*vid.split("/")) / result_name,

        result_root / tail / tail / result_name,

        result_root / tail / result_name,

    ]

    for p in candidates:

        if p.exists():

            return p

    if result_root.exists():

        for p in result_root.rglob(result_name):

            try:

                rel = str(p.parent.relative_to(result_root)).replace("\\", "/")

            except Exception:

                rel = str(p.parent).replace("\\", "/")

            if video_id_match(rel, vid) or p.parent.name == tail or p.parent.parent.name == tail:

                return p

    return None

def find_all_result_csvs(result_root: Path, result_name: str) -> list[Path]:

    if not result_root.exists():

        return []

    return sorted(result_root.rglob(result_name))

def video_id_from_result_csv(result_root: Path, result_csv: Path) -> str:

    rel = result_csv.parent.relative_to(result_root)

    return norm_video_id(str(rel).replace("\\", "/"))

def find_video_file(video_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    candidates = []

    for ext in VIDEO_EXTS:

        candidates.append(video_root / tail / f"{tail}{ext}")

        candidates.append(video_root / f"{tail}{ext}")

        candidates.append(video_root / Path(*vid.split("/")) / f"{tail}{ext}")

    for p in candidates:

        if p.exists():

            return p

    if video_root.exists():

        for p in video_root.rglob("*"):

            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == tail:

                return p

    return None

def load_chinese_font(size: int = 24):

    if not HAS_PIL:

        return None

    candidates = [

        Path(r"C:\Windows\Fonts\msyh.ttc"),

        Path(r"C:\Windows\Fonts\msyhbd.ttc"),

        Path(r"C:\Windows\Fonts\simhei.ttf"),

        Path(r"C:\Windows\Fonts\simsun.ttc"),

        Path(r"C:\Windows\Fonts\arial.ttf"),

    ]

    for p in candidates:

        try:

            if p.exists():

                return ImageFont.truetype(str(p), size=size)

        except Exception:

            pass

    try:

        return ImageFont.load_default()

    except Exception:

        return None

FONT_SMALL = load_chinese_font(22)

FONT_MID = load_chinese_font(26)

FONT_BIG = load_chinese_font(32)

def draw_text(frame_bgr: np.ndarray, text: str, xy: tuple[int, int], font=None,

              fill=(30, 30, 30), stroke_fill=(255, 255, 255), stroke_width=1) -> np.ndarray:

    text = str(text)

    x, y = xy

    if HAS_PIL and font is not None:

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb)

        draw = ImageDraw.Draw(img)

        draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.putText(frame_bgr, text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fill[::-1], 2, cv2.LINE_AA)

    return frame_bgr

def build_timeline_strip(label_seq, width=1600, height=28) -> np.ndarray:

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

            cv2.rectangle(strip, (x1, 0), (x2, height - 1), get_bgr(seq[start]), -1)

            start = i

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)

    return strip

def save_timeline_png(df: pd.DataFrame, out_png: Path, state_col: str) -> None:

    t = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0).to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(18, 9), sharex=True)

    axes[0].step(t, pd.to_numeric(df[state_col], errors="coerce").fillna(-1).astype(int), where="post")

    axes[0].set_ylabel(state_col)

    if "global_proto_id" in df.columns:

        axes[1].step(t, pd.to_numeric(df["global_proto_id"], errors="coerce").fillna(-1).astype(int), where="post")

        axes[1].set_ylabel("global\nproto")

    else:

        axes[1].axis("off")

    if "nlp_norm_surprisal" in df.columns:

        axes[2].plot(t, pd.to_numeric(df["nlp_norm_surprisal"], errors="coerce").fillna(0))

        axes[2].set_ylabel("NLP\nsurprisal")

    else:

        nlp_cols = [c for c in df.columns if c.startswith("nlp_") and c != "nlp_text_joined"]

        if nlp_cols:

            axes[2].plot(t, pd.to_numeric(df[nlp_cols[0]], errors="coerce").fillna(0))

            axes[2].set_ylabel(nlp_cols[0])

        else:

            axes[2].axis("off")

    if "is_teacher_frame_used" in df.columns:

        axes[3].step(t, pd.to_numeric(df["is_teacher_frame_used"], errors="coerce").fillna(0).astype(int), where="post")

        axes[3].set_ylabel("teacher\nused")

    else:

        axes[3].axis("off")

    axes[-1].set_xlabel("time_sec")

    fig.suptitle(f"Cluster timeline: {state_col}", fontsize=15)

    for ax in axes:

        ax.grid(alpha=0.25)

    ensure_dir(out_png.parent)

    plt.tight_layout()

    plt.savefig(out_png, dpi=220, bbox_inches="tight")

    plt.close(fig)

def draw_bottom_timeline(frame: np.ndarray, final_strip: np.ndarray, global_strip: np.ndarray | None,

                         idx: int, total_len: int, state_col: str) -> np.ndarray:

    h, w, _ = frame.shape

    x1, x2 = 30, w - 30

    y2 = h - 24

    y1 = y2 - 108 if global_strip is not None else y2 - 78

    overlay = frame.copy()

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)

    frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

    frame = draw_text(frame, f"Timeline: {state_col}", (x1 + 10, y1 + 8), FONT_SMALL,

                      fill=(20, 20, 20), stroke_width=0)

    strip_w = max(10, x2 - x1 - 20)

    sx = x1 + 10

    fy = y1 + 38

    resized_final = cv2.resize(final_strip, (strip_w, 26), interpolation=cv2.INTER_NEAREST)

    frame[fy:fy + 26, sx:sx + strip_w] = resized_final

    frame = draw_text(frame, "Final", (sx + 4, fy - 24), FONT_SMALL,

                      fill=(20, 20, 20), stroke_width=0)

    gy = None

    if global_strip is not None:

        gy = fy + 38

        resized_global = cv2.resize(global_strip, (strip_w, 26), interpolation=cv2.INTER_NEAREST)

        frame[gy:gy + 26, sx:sx + strip_w] = resized_global

        frame = draw_text(frame, "Global", (sx + 4, gy - 24), FONT_SMALL,

                          fill=(20, 20, 20), stroke_width=0)

    px = sx + int(round(idx / max(total_len - 1, 1) * (strip_w - 1)))

    cv2.line(frame, (px, fy - 4), (px, (gy + 30) if gy is not None else (fy + 30)), (0, 0, 0), 2)

    return frame

def render_one(result_csv: Path, video_path: Path, out_dir: Path, *,

               state_col: str = "pid_nlp_state",

               global_col: str = "global_proto_id",

               no_audio: bool = False,

               max_text_chars: int = 80) -> dict:

    ensure_dir(out_dir)

    df = pd.read_csv(result_csv)

    if "time_sec" not in df.columns:

        raise ValueError(f"{result_csv} 缺少 time_sec 列")

    if state_col not in df.columns:

        raise ValueError(f"{result_csv} 缺少 {state_col} 列；可用列：{list(df.columns)}")

    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0.0)

    df[state_col] = pd.to_numeric(df[state_col], errors="coerce").fillna(-1).astype(int)

    if global_col in df.columns:

        df[global_col] = pd.to_numeric(df[global_col], errors="coerce").fillna(-1).astype(int)

    stem = safe_filename(video_path.stem)

    silent_mp4 = out_dir / f"{stem}_{state_col}_overlay_silent.mp4"

    audio_mp4 = out_dir / f"{stem}_{state_col}_overlay_with_audio.mp4"

    timeline_png = out_dir / f"{stem}_{state_col}_timeline.png"

    save_timeline_png(df, timeline_png, state_col=state_col)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(

        str(silent_mp4),

        cv2.VideoWriter_fourcc(*"mp4v"),

        float(fps),

        (width, height),

    )

    times = df["time_sec"].to_numpy(dtype=float)

    states = df[state_col].to_numpy(dtype=int)

    final_strip = build_timeline_strip(states, width=1600, height=28)

    global_strip = None

    if global_col in df.columns:

        global_strip = build_timeline_strip(df[global_col].to_numpy(dtype=int), width=1600, height=28)

    total_rows = len(df)

    frame_idx = 0

    print("=" * 90)

    print(f"[render] result_csv: {result_csv}")

    print(f"[render] video     : {video_path}")

    print(f"[render] out_dir   : {out_dir}")

    print(f"[render] fps={fps:.3f}, size={width}x{height}, video_frames={n_video_frames}, csv_rows={total_rows}")

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        t = frame_idx / float(fps)

        idx = int(np.searchsorted(times, t, side="left"))

        idx = min(max(idx, 0), total_rows - 1)

        r = df.iloc[idx]

        state = int(r[state_col])

        gproto = int(r[global_col]) if global_col in df.columns else -1

        local_meta = int(r["meta_state"]) if "meta_state" in df.columns and pd.notna(r["meta_state"]) else -1

        color = get_bgr(state)

        panel_w = min(width - 50, 1120)

        panel_h = 190

        x0, y0 = 25, 25

        overlay = frame.copy()

        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)

        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), color, 5)

        line_y = y0 + 18

        frame = draw_text(frame, f"{state_col}: {state}", (x0 + 20, line_y), FONT_BIG,

                          fill=(20, 20, 20), stroke_width=0)

        frame = draw_text(frame, f"time={t:.2f}s | frame={frame_idx} | csv_row={idx}", (x0 + 20, line_y + 42), FONT_SMALL,

                          fill=(30, 30, 30), stroke_width=0)

        if global_col in df.columns:

            frame = draw_text(frame, f"{global_col}: {gproto} | local meta_state: {local_meta}", (x0 + 20, line_y + 72), FONT_SMALL,

                              fill=(30, 30, 30), stroke_width=0)

        if "nlp_norm_surprisal" in df.columns:

            sur = pd.to_numeric(pd.Series([r.get("nlp_norm_surprisal")]), errors="coerce").fillna(0.0).iloc[0]

            frame = draw_text(frame, f"NLP norm_surprisal: {float(sur):.4f}", (x0 + 20, line_y + 102), FONT_SMALL,

                              fill=(30, 30, 30), stroke_width=0)

        if "nlp_text_joined" in df.columns:

            txt = shorten(r.get("nlp_text_joined", ""), max_chars=max_text_chars)

            if txt:

                frame = draw_text(frame, f"NLP text: {txt}", (x0 + 20, line_y + 132), FONT_SMALL,

                                  fill=(30, 30, 30), stroke_width=0)

        frame = draw_bottom_timeline(frame, final_strip, global_strip, idx, total_rows, state_col=state_col)

        writer.write(frame)

        frame_idx += 1

        if frame_idx % 300 == 0:

            print(f"  rendered frames: {frame_idx}")

    cap.release()

    writer.release()

    final_mp4 = silent_mp4

    if (not no_audio) and HAS_MOVIEPY:

        try:

            print("[audio] 合并原视频音频...")

            src = VideoFileClip(str(video_path))

            dst = VideoFileClip(str(silent_mp4))

            if src.audio is not None:

                dst = dst.with_audio(src.audio)

                dst.write_videofile(str(audio_mp4), codec="libx264", audio_codec="aac")

                final_mp4 = audio_mp4

            try:

                src.close()

                dst.close()

            except Exception:

                pass

        except Exception as e:

            print(f"[audio] 音频合并失败，保留无声视频: {e}")

    print(f"[OK] video   : {final_mp4}")

    print(f"[OK] timeline: {timeline_png}")

    return {

        "result_csv": str(result_csv),

        "video_path": str(video_path),

        "out_dir": str(out_dir),

        "silent_mp4": str(silent_mp4),

        "final_mp4": str(final_mp4),

        "timeline_png": str(timeline_png),

        "frames_rendered": int(frame_idx),

        "status": "ok",

        "error": "",

    }

def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--video-id", default="21", help="视频编号，例如 21 或 21/21；all 表示批量处理全部结果")

    ap.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)

    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)

    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)

    ap.add_argument("--result-name", default="layer12_nlp_pid_t2s_final.csv")

    ap.add_argument("--state-col", default="pid_nlp_state", help="要渲染的状态列，例如 pid_nlp_state 或 global_proto_id")

    ap.add_argument("--global-col", default="global_proto_id")

    ap.add_argument("--no-audio", action="store_true", help="不合并原视频音频，只输出 silent mp4")

    ap.add_argument("--max-text-chars", type=int, default=80)

    args = ap.parse_args()

    result_root = args.result_root.resolve()

    video_root = args.video_root.resolve()

    out_root = args.out_root.resolve()

    ensure_dir(out_root)

    tasks = []

    if args.video_id.lower() == "all":

        csvs = find_all_result_csvs(result_root, args.result_name)

        if not csvs:

            raise FileNotFoundError(f"没有找到任何 {args.result_name}: {result_root}")

        for csv_path in csvs:

            vid = video_id_from_result_csv(result_root, csv_path)

            video_path = find_video_file(video_root, vid)

            if video_path is None:

                print(f"[SKIP] 找不到原视频: video_id={vid}, result={csv_path}")

                continue

            out_dir = out_root / Path(*norm_video_id(vid).split("/"))

            tasks.append((csv_path, video_path, out_dir, vid))

    else:

        vid = norm_video_id(args.video_id)

        csv_path = find_result_csv(result_root, vid, args.result_name)

        if csv_path is None:

            raise FileNotFoundError(f"找不到结果 CSV: video_id={vid}, result_root={result_root}, result_name={args.result_name}")

        video_path = find_video_file(video_root, vid)

        if video_path is None:

            raise FileNotFoundError(f"找不到原视频: video_id={vid}, video_root={video_root}")

        out_dir = out_root / Path(*norm_video_id(vid).split("/"))

        tasks.append((csv_path, video_path, out_dir, vid))

    manifest = []

    for csv_path, video_path, out_dir, vid in tasks:

        try:

            manifest.append(render_one(

                csv_path,

                video_path,

                out_dir,

                state_col=args.state_col,

                global_col=args.global_col,

                no_audio=args.no_audio,

                max_text_chars=args.max_text_chars,

            ))

        except Exception as e:

            print(f"[FAILED] {vid}: {e}")

            manifest.append({

                "result_csv": str(csv_path),

                "video_path": str(video_path),

                "out_dir": str(out_dir),

                "silent_mp4": "",

                "final_mp4": "",

                "timeline_png": "",

                "frames_rendered": 0,

                "status": "failed",

                "error": repr(e),

            })

    manifest_csv = out_root / "visualize_layer12_result_manifest.csv"

    pd.DataFrame(manifest).to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    print("\n全部完成。")

    print(f"输出目录: {out_root}")

    print(f"manifest: {manifest_csv}")

    return 0

if __name__ == "__main__":

    raise SystemExit(main())
