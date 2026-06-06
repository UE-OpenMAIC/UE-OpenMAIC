import argparse

import re

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

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

                                                           

                     

                                                           

         

VIDEO_ROOT = r"D:\code\teacherT2S\yolo\input"

                    

                                                                                              

T2S_RESULT_ROOT = r"D:\code\teacherT2S\multiscale_t2s_output_event_batch"

           

                                

SECOND_STAGE_ROOT = (

    r"D:\code\teacherT2S\multiscale_t2s_output_event_batch"

    r"\_cross_video_proto_alignment_t2s_only"

    r"\_second_stage_combined_map"

)

      

OUTPUT_ROOT = r"D:\code\teacherT2S\second_stage_video_visualization"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

TIME_COL = "time_sec"

DEFAULT_FRAME_STATE_COL = "meta_state"

DEFAULT_ASSIGN_STATE_COL = "local_state"

FONT = cv2.FONT_HERSHEY_SIMPLEX

                                                           

                      

                                                           

LABEL_COLOR_MAP = {

    "F0-S0": (220, 120, 40),

    "F0-S1": (40, 40, 220),

    "F1-S0": (180, 80, 220),

    "F1-S1": (220, 200, 40),

    "SKIP": (150, 150, 150),

    "UNKNOWN": (100, 100, 100),

}

FALLBACK_COLORS = [

    (220, 120, 40),

    (40, 40, 220),

    (180, 80, 220),

    (220, 200, 40),

    (0, 180, 0),

    (0, 180, 180),

    (180, 120, 0),

    (120, 0, 180),

    (80, 80, 220),

    (220, 80, 80),

]

                                                           

         

                                                           

def ensure_dir(p):

    Path(p).mkdir(parents=True, exist_ok=True)

def norm_video_id(x):

    

       

    s = str(x).strip().replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

                 

    s = re.sub(r"/+", "/", s).strip("/")

    return s

def tail_video_id(x):

    """取 video_id 的最后一级。例如 21/21 -> 21。"""

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def safe_filename(x):

    

       

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s or "video"

def video_id_match(a, b):

    

       

    a = norm_video_id(a)

    b = norm_video_id(b)

    if a == b:

        return True

    return tail_video_id(a) == tail_video_id(b)

def find_video_file(video_id, video_root):

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    root = Path(video_root)

    candidates = []

    for ext in VIDEO_EXTS:

                               

        candidates.append(root / tail / f"{tail}{ext}")

                            

        candidates.append(root / f"{tail}{ext}")

                                        

        candidates.append(root / Path(*vid.split("/")) / f"{tail}{ext}")

    for p in candidates:

        if p.exists():

            return p

                             

    for p in root.rglob("*"):

        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == tail:

            return p

    raise FileNotFoundError(

        f"找不到 video_id={video_id} 对应的视频。\n"

        f"已尝试 tail={tail}，搜索根目录: {root}"

    )

def find_frame_csv(video_id, t2s_result_root):

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    root = Path(t2s_result_root)

    path_from_vid = root / Path(*vid.split("/"))

    candidates = [

                                                          

        path_from_vid / "multiscale_t2s_with_meta.csv",

        path_from_vid / "multiscale_meta_state_seq.csv",

                                                       

        root / tail / tail / "multiscale_t2s_with_meta.csv",

        root / tail / tail / "multiscale_meta_state_seq.csv",

                                                    

        root / tail / "multiscale_t2s_with_meta.csv",

        root / tail / "multiscale_meta_state_seq.csv",

    ]

    for p in candidates:

        if p.exists():

            return p

                                          

    for pattern in ["multiscale_t2s_with_meta.csv", "multiscale_meta_state_seq.csv"]:

        for p in root.rglob(pattern):

            parts = [str(x) for x in p.parts]

            if tail in parts or vid in "/".join(parts).replace("\\", "/"):

                return p

    raise FileNotFoundError(

        f"找不到 video_id={video_id} 对应的逐帧 T2S 结果 CSV。\n"

        f"搜索根目录: {root}\n"

        f"建议手动指定 --frame_csv_path"

    )

def find_assign_csv(second_stage_root):

    root = Path(second_stage_root)

    p = root / "second_stage_assignments.csv"

    if p.exists():

        return p

    all_csv = list(root.rglob("second_stage_assignments.csv")) if root.exists() else []

    if len(all_csv) > 0:

        return all_csv[0]

    raise FileNotFoundError(

        f"找不到 second_stage_assignments.csv，搜索目录: {root}\n"

        f"建议手动指定 --assign_csv_path"

    )

def auto_find_label_col(assign_df):

    

       

    candidates = [

        "subcluster_name",

        "second_stage_label",

        "combined_label",

        "stage2_label",

        "cluster_label",

        "final_label",

        "sub_label",

        "state_label",

        "group_label",

        "label",

        "second_stage_id",

        "sub_proto_id",

    ]

    for c in candidates:

        if c in assign_df.columns:

            return c

    if "F" in assign_df.columns and "S" in assign_df.columns:

        assign_df["combined_label"] = assign_df.apply(

            lambda r: f"F{int(r['F'])}-S{int(r['S'])}", axis=1

        )

        return "combined_label"

    if "parent_cluster" in assign_df.columns and "sub_cluster" in assign_df.columns:

        assign_df["combined_label"] = assign_df.apply(

            lambda r: f"F{int(r['parent_cluster'])}-S{int(r['sub_cluster'])}", axis=1

        )

        return "combined_label"

    raise ValueError(

        "无法自动识别二阶段聚类标签列。\n"

        f"当前列名为: {list(assign_df.columns)}\n"

        "建议使用：--assign_label_col subcluster_name 或 --assign_label_col second_stage_id"

    )

def get_color_for_label(label, label_to_color):

    label = str(label)

    if label in label_to_color:

        return label_to_color[label]

    return LABEL_COLOR_MAP.get("UNKNOWN", (100, 100, 100))

def build_label_color_map(labels):

    labels = [str(x) for x in labels if pd.notna(x)]

    unique_labels = sorted(set(labels))

    label_to_color = {}

    for i, lab in enumerate(unique_labels):

        if lab in LABEL_COLOR_MAP:

            label_to_color[lab] = LABEL_COLOR_MAP[lab]

        else:

            label_to_color[lab] = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]

    label_to_color["SKIP"] = LABEL_COLOR_MAP["SKIP"]

    label_to_color["UNKNOWN"] = LABEL_COLOR_MAP["UNKNOWN"]

    return label_to_color

def put_text_with_bg(frame, text, org, font_scale=0.65, color=(30, 30, 30), thickness=2):

    x, y = org

    cv2.putText(frame, text, (x, y), FONT, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)

    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)

def build_timeline_strip(label_seq, label_to_color, width=1600, height=24):

    strip = np.ones((height, width, 3), dtype=np.uint8) * 255

    n = len(label_seq)

    if n == 0:

        return strip

    start = 0

    for i in range(1, n + 1):

        if i == n or label_seq[i] != label_seq[start]:

            s = start

            e = i - 1

            lab = str(label_seq[start])

            x1 = int(round(s / n * width))

            x2 = int(round((e + 1) / n * width)) - 1

            x2 = max(x1, x2)

            color = get_color_for_label(lab, label_to_color)

            cv2.rectangle(strip, (x1, 0), (x2, height - 1), color, -1)

            start = i

    cv2.rectangle(strip, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)

    return strip

def draw_timeline_on_frame(frame, timeline_strip, idx, total_len):

    h, w, _ = frame.shape

    margin_x = 30

    margin_y = 28

    label_w = 150

    box_x1 = margin_x

    box_x2 = w - margin_x

    box_y2 = h - margin_y

    box_y1 = box_y2 - 75

    overlay = frame.copy()

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    put_text_with_bg(

        frame,

        "Second-stage timeline",

        (box_x1 + 10, box_y1 + 25),

        font_scale=0.55,

        color=(30, 30, 30),

        thickness=1,

    )

    x1 = box_x1 + label_w

    x2 = box_x2 - 10

    y1 = box_y1 + 36

    y2 = y1 + 24

    target_w = max(10, x2 - x1)

    target_h = max(8, y2 - y1)

    strip_resized = cv2.resize(timeline_strip, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = strip_resized

    if total_len > 1:

        x_cur = x1 + int(round(idx / (total_len - 1) * (target_w - 1)))

    else:

        x_cur = x1

    cv2.line(frame, (x_cur, y1 - 4), (x_cur, y2 + 4), (255, 255, 255), 3)

    cv2.line(frame, (x_cur, y1 - 4), (x_cur, y2 + 4), (0, 0, 0), 1)

def merge_audio(original_video, silent_video, output_video):

    if not HAS_MOVIEPY:

        print("未检测到 moviepy，跳过音频合并。")

        print(f"无声视频已输出: {silent_video}")

        return False

    print("开始合并原视频音轨...")

    overlay_clip = None

    orig_clip = None

    final_clip = None

    try:

        overlay_clip = VideoFileClip(str(silent_video))

        orig_clip = VideoFileClip(str(original_video))

        if orig_clip.audio is not None:

            if hasattr(orig_clip.audio, "subclipped"):

                audio_clip = orig_clip.audio.subclipped(0, overlay_clip.duration)

            else:

                audio_clip = orig_clip.audio.subclip(0, overlay_clip.duration)

            if hasattr(overlay_clip, "with_audio"):

                final_clip = overlay_clip.with_audio(audio_clip)

            else:

                final_clip = overlay_clip.set_audio(audio_clip)

        else:

            print("原视频没有音轨，将输出无声视频。")

            final_clip = overlay_clip

        final_clip.write_videofile(str(output_video), codec="libx264", audio_codec="aac")

        return True

    except Exception as e:

        print(f"音频合并失败: {e}")

        return False

    finally:

        for clip in [overlay_clip, orig_clip, final_clip]:

            try:

                if clip is not None:

                    clip.close()

            except Exception:

                pass

                                                           

                       

                                                           

def attach_second_stage_labels(

    frame_df,

    assign_df,

    video_id,

    frame_state_col,

    assign_state_col,

    assign_label_col,

):

    video_id = norm_video_id(video_id)

    if TIME_COL not in frame_df.columns:

        raise ValueError(f"逐帧 CSV 缺少时间列: {TIME_COL}")

    if frame_state_col not in frame_df.columns:

        raise ValueError(

            f"逐帧 CSV 缺少状态列: {frame_state_col}\n"

            f"当前逐帧 CSV 列名为: {list(frame_df.columns)}\n"

            "可以尝试 --frame_state_col meta_state 或 --frame_state_col local_state"

        )

    if "video_id" not in assign_df.columns:

        raise ValueError(

            "second_stage_assignments.csv 缺少 video_id 列。\n"

            f"当前列名为: {list(assign_df.columns)}"

        )

    if assign_state_col not in assign_df.columns:

        raise ValueError(

            f"second_stage_assignments.csv 缺少状态列: {assign_state_col}\n"

            f"当前列名为: {list(assign_df.columns)}"

        )

    if assign_label_col not in assign_df.columns:

        raise ValueError(

            f"second_stage_assignments.csv 缺少标签列: {assign_label_col}\n"

            f"当前列名为: {list(assign_df.columns)}"

        )

    frame_df = frame_df.copy()

    assign_df = assign_df.copy()

    assign_df["video_id_str"] = assign_df["video_id"].map(norm_video_id)

                                                      

    sub_assign = assign_df[

        assign_df["video_id_str"].map(lambda x: video_id_match(x, video_id))

    ].copy()

    if len(sub_assign) == 0:

        raise ValueError(

            f"second_stage_assignments.csv 中找不到 video_id={video_id} 的记录。\n"

            f"可用 video_id 示例: {assign_df['video_id_str'].drop_duplicates().head(20).tolist()}"

        )

    sub_assign[assign_state_col] = pd.to_numeric(sub_assign[assign_state_col], errors="coerce")

    sub_assign = sub_assign.dropna(subset=[assign_state_col])

    sub_assign[assign_state_col] = sub_assign[assign_state_col].astype(int)

    state_to_label = {}

    for _, r in sub_assign.iterrows():

        state_to_label[int(r[assign_state_col])] = str(r[assign_label_col])

    frame_df[frame_state_col] = pd.to_numeric(frame_df[frame_state_col], errors="coerce")

    label_seq = []

    for s in frame_df[frame_state_col].values:

        if pd.isna(s):

            label_seq.append("UNKNOWN")

            continue

        s_int = int(s)

        if s_int < 0:

            label_seq.append("SKIP")

        else:

            label_seq.append(state_to_label.get(s_int, "UNKNOWN"))

    frame_df["second_stage_label"] = label_seq

    print("=" * 80)

    print(f"video_id={video_id}")

    print(f"逐帧状态列: {frame_state_col}")

    print(f"二阶段匹配列: {assign_state_col}")

    print(f"二阶段标签列: {assign_label_col}")

    print("local_state -> second_stage_label 映射：")

    for k in sorted(state_to_label.keys()):

        print(f"  {k} -> {state_to_label[k]}")

    print("当前视频逐帧标签分布：")

    print(frame_df["second_stage_label"].value_counts(dropna=False).to_string())

    unknown_count = int((frame_df["second_stage_label"] == "UNKNOWN").sum())

    if unknown_count > 0:

        print(f"警告：有 {unknown_count} 行未匹配到二阶段标签，已标记为 UNKNOWN。")

    return frame_df

                                                           

          

                                                           

def visualize_one_video(

    video_id,

    video_root,

    t2s_result_root,

    second_stage_root,

    output_root,

    frame_state_col=DEFAULT_FRAME_STATE_COL,

    assign_state_col=DEFAULT_ASSIGN_STATE_COL,

    assign_label_col=None,

    video_path=None,

    frame_csv_path=None,

    assign_csv_path=None,

    merge_audio_flag=True,

):

    video_id = norm_video_id(video_id)

    display_video_id = video_id

    safe_video_id = safe_filename(video_id)

    ensure_dir(output_root)

    if video_path is None:

        video_path = find_video_file(video_id, video_root)

    else:

        video_path = Path(video_path)

    if frame_csv_path is None:

        frame_csv_path = find_frame_csv(video_id, t2s_result_root)

    else:

        frame_csv_path = Path(frame_csv_path)

    if assign_csv_path is None:

        assign_csv_path = find_assign_csv(second_stage_root)

    else:

        assign_csv_path = Path(assign_csv_path)

    print("=" * 80)

    print(f"指定可视化 video_id: {display_video_id}")

    print(f"安全输出名: {safe_video_id}")

    print(f"视频文件: {video_path}")

    print(f"逐帧 T2S CSV: {frame_csv_path}")

    print(f"二阶段 assignments: {assign_csv_path}")

    if not video_path.exists():

        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if not frame_csv_path.exists():

        raise FileNotFoundError(f"逐帧 T2S CSV 不存在: {frame_csv_path}")

    if not assign_csv_path.exists():

        raise FileNotFoundError(f"二阶段 assignments 不存在: {assign_csv_path}")

    frame_df = pd.read_csv(frame_csv_path)

    assign_df = pd.read_csv(assign_csv_path)

    if assign_label_col is None:

        assign_label_col = auto_find_label_col(assign_df)

        print(f"自动识别二阶段标签列: {assign_label_col}")

    frame_df = frame_df.sort_values(TIME_COL).reset_index(drop=True)

    frame_df = attach_second_stage_labels(

        frame_df=frame_df,

        assign_df=assign_df,

        video_id=video_id,

        frame_state_col=frame_state_col,

        assign_state_col=assign_state_col,

        assign_label_col=assign_label_col,

    )

    label_to_color = build_label_color_map(frame_df["second_stage_label"].tolist())

    time_values = frame_df[TIME_COL].astype(float).values

    label_seq = frame_df["second_stage_label"].astype(str).values

    state_seq = frame_df[frame_state_col].values

    total_rows = len(frame_df)

    if total_rows == 0:

        raise ValueError("逐帧 CSV 为空，无法可视化。")

    timeline_strip = build_timeline_strip(

        label_seq=label_seq,

        label_to_color=label_to_color,

        width=1600,

        height=24,

    )

                                                

    out_dir = Path(output_root) / safe_video_id

    ensure_dir(out_dir)

    out_silent = out_dir / f"{safe_video_id}_second_stage_overlay_silent.mp4"

    out_video = out_dir / f"{safe_video_id}_second_stage_overlay_with_audio.mp4"

    out_frame_csv = out_dir / f"{safe_video_id}_second_stage_frame_labels.csv"

    frame_df.to_csv(out_frame_csv, index=False, encoding="utf-8-sig")

    print(f"已输出逐帧标签表: {out_frame_csv}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:

        cap.release()

        raise RuntimeError("视频 FPS 读取失败")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(out_silent), fourcc, fps, (width, height))

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(f"VideoWriter 打开失败，无法写入: {out_silent}")

    frame_id = 0

    print("开始生成二阶段聚类可视化视频...")

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        current_time = frame_id / fps

        idx = np.searchsorted(time_values, current_time, side="right") - 1

        idx = int(np.clip(idx, 0, total_rows - 1))

        cur_label = str(label_seq[idx])

        cur_state = state_seq[idx]

        cur_color = get_color_for_label(cur_label, label_to_color)

        overlay = frame.copy()

        cv2.rectangle(overlay, (25, 25), (850, 195), (255, 255, 255), -1)

        frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

        cv2.rectangle(frame, (45, 45), (90, 90), cur_color, -1)

        cv2.rectangle(frame, (45, 45), (90, 90), (0, 0, 0), 1)

        put_text_with_bg(

            frame,

            "Second-stage clustering visualization",

            (110, 58),

            font_scale=0.72,

            color=(20, 20, 20),

            thickness=2,

        )

        put_text_with_bg(

            frame,

            f"video_id = {display_video_id}",

            (45, 118),

            font_scale=0.62,

            color=(30, 30, 30),

            thickness=2,

        )

        put_text_with_bg(

            frame,

            f"time = {current_time:.2f}s    row = {idx}/{total_rows - 1}",

            (45, 146),

            font_scale=0.58,

            color=(30, 30, 30),

            thickness=2,

        )

        put_text_with_bg(

            frame,

            f"{frame_state_col} = {cur_state}    second_stage = {cur_label}",

            (45, 174),

            font_scale=0.58,

            color=cur_color,

            thickness=2,

        )

        cv2.rectangle(frame, (width - 250, 35), (width - 35, 95), cur_color, -1)

        cv2.rectangle(frame, (width - 250, 35), (width - 35, 95), (0, 0, 0), 2)

        put_text_with_bg(frame, cur_label, (width - 225, 75), font_scale=0.9, color=(0, 0, 0), thickness=2)

        draw_timeline_on_frame(

            frame=frame,

            timeline_strip=timeline_strip,

            idx=idx,

            total_len=total_rows,

        )

        writer.write(frame)

        frame_id += 1

        if frame_id % 200 == 0:

            print(f"已处理 {frame_id} 帧，当前时间 {current_time:.2f}s")

    cap.release()

    writer.release()

    print(f"已输出无声视频: {out_silent}")

    if merge_audio_flag:

        ok = merge_audio(video_path, out_silent, out_video)

        if ok:

            print(f"已输出带声音视频: {out_video}")

        else:

            print(f"音频合并失败或未安装 moviepy，无声视频保留在: {out_silent}")

    else:

        print("已按参数跳过音频合并。")

    print("全部完成。")

    print(f"输出目录: {out_dir}")

                                                           

          

                                                           

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_id", type=str, required=True, help="要可视化的视频 ID，例如 21、21/21、24、53")

    parser.add_argument("--video_root", type=str, default=VIDEO_ROOT, help="原始视频根目录")

    parser.add_argument("--t2s_result_root", type=str, default=T2S_RESULT_ROOT, help="第一阶段/多尺度 T2S 输出根目录")

    parser.add_argument("--second_stage_root", type=str, default=SECOND_STAGE_ROOT, help="二阶段聚类输出目录，里面应有 second_stage_assignments.csv")

    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT, help="可视化视频输出目录")

    parser.add_argument("--frame_state_col", type=str, default=DEFAULT_FRAME_STATE_COL, help="逐帧 CSV 中用于匹配二阶段 local_state 的列名，默认 meta_state")

    parser.add_argument("--assign_state_col", type=str, default=DEFAULT_ASSIGN_STATE_COL, help="second_stage_assignments.csv 中的状态列名，默认 local_state")

    parser.add_argument("--assign_label_col", type=str, default=None, help="second_stage_assignments.csv 中的二阶段标签列名；不填则自动识别")

    parser.add_argument("--video_path", type=str, default=None, help="手动指定视频路径；不填则根据 video_id 自动查找")

    parser.add_argument("--frame_csv_path", type=str, default=None, help="手动指定逐帧 T2S CSV；不填则自动查找")

    parser.add_argument("--assign_csv_path", type=str, default=None, help="手动指定 second_stage_assignments.csv；不填则自动查找")

    parser.add_argument("--no_audio", action="store_true", help="只生成无声视频，不合并原音频")

    return parser.parse_args()

def main():

    args = parse_args()

    visualize_one_video(

        video_id=args.video_id,

        video_root=args.video_root,

        t2s_result_root=args.t2s_result_root,

        second_stage_root=args.second_stage_root,

        output_root=args.output_root,

        frame_state_col=args.frame_state_col,

        assign_state_col=args.assign_state_col,

        assign_label_col=args.assign_label_col,

        video_path=args.video_path,

        frame_csv_path=args.frame_csv_path,

        assign_csv_path=args.assign_csv_path,

        merge_audio_flag=(not args.no_audio),

    )

if __name__ == "__main__":

    main()
