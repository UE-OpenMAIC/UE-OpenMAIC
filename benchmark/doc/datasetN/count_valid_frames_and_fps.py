# -*- coding: utf-8 -*-
"""
count_valid_frames_and_fps.py

放置路径：
D:\\code\\teacherT2S\\doc\\datasetN\\count_valid_frames_and_fps.py

功能：
1. 自动扫描 multiscale_t2s_output_event_batch_orientation8 下所有 case 文件夹；
2. 统计每个 case 的总帧数、有效教师帧数、无效/滤除帧数；
3. 根据 time_sec 估计采样 FPS；
4. 统计每个视频总时长、有效教师时长、滤除时长；
5. 统计所有视频累计总时长、累计有效教师时长、累计滤除时长；
6. 输出 by-case 汇总表和 total 汇总表。
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 脚本位于 D:\code\teacherT2S\doc\datasetN
# 所以项目根目录是 D:\code\teacherT2S
PROJECT_ROOT = SCRIPT_DIR.parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "multiscale_t2s_output_event_batch_orientation8"

REPORT_DIR = SCRIPT_DIR / "reports_valid_frames"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 只统计某一个 case 时，例如：
# ONLY_CASE = Path(r"3\3")
# 统计全部时保持 None
ONLY_CASE = None

CASE_FILE_CANDIDATES = [
    "student_marker_mask.csv",
    "multiscale_t2s_with_meta.csv",
    "multiscale_meta_state_seq.csv",
    "multiscale_raw_label_matrix_Tx32.csv",
    "multiscale_t2s_state_matrix.csv",
]

TIME_COL_CANDIDATES = [
    "time_sec",
    "timestamp",
    "time",
    "t",
]

META_STATE_COL_CANDIDATES = [
    "meta_state",
    "meta",
    "global_state",
]


# ============================================================
# 2. 工具函数
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def seconds_to_hms(seconds):
    """
    秒数转成 HH:MM:SS.xxx。
    """
    if seconds is None or pd.isna(seconds):
        return ""

    seconds = float(seconds)
    if seconds < 0:
        seconds = 0.0

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:06.3f}"


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """
    兼容 1/0、True/False、true/false、yes/no 等写法。
    """
    if s.dtype == bool:
        return s.fillna(False)

    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() > 0:
        return num.fillna(0).astype(float) != 0

    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["1", "true", "t", "yes", "y"])


def find_case_dirs(root: Path):
    """
    找到包含结果 CSV 的 case 文件夹。
    """
    if ONLY_CASE is not None:
        case_dir = root / ONLY_CASE
        if not case_dir.exists():
            raise FileNotFoundError(f"指定 case 不存在：{case_dir}")
        return [case_dir]

    case_dirs = []

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        if any((p / name).exists() for name in CASE_FILE_CANDIDATES):
            case_dirs.append(p)

    case_dirs = sorted(set(case_dirs), key=lambda x: str(x.relative_to(root)))
    return case_dirs


def find_first_existing_file(case_dir: Path, names):
    for name in names:
        p = case_dir / name
        if p.exists():
            return p
    return None


def find_time_df(case_dir: Path):
    """
    优先从 student_marker_mask.csv 读取 time_sec；
    如果没有，再从其他文件读取。
    """
    for name in CASE_FILE_CANDIDATES:
        p = case_dir / name
        if not p.exists():
            continue

        df = read_csv_safely(p)

        for c in TIME_COL_CANDIDATES:
            if c in df.columns:
                return df, p, c

    return None, None, None


def estimate_fps_and_duration_from_time(time_values):
    """
    根据 time_sec 估计采样 FPS 和总时长。

    返回：
    duration_sec: 总时长，近似为 len(t) * median_dt
    frame_dt: 每帧时间间隔
    fps_median_dt: 1 / median_dt
    fps_count_based: len(t) / duration_sec
    """
    t = pd.to_numeric(pd.Series(time_values), errors="coerce").dropna().to_numpy(dtype=float)

    if len(t) < 2:
        return np.nan, np.nan, np.nan, np.nan

    t = np.sort(t)
    diffs = np.diff(t)
    diffs = diffs[diffs > 1e-9]

    if len(diffs) == 0:
        return np.nan, np.nan, np.nan, np.nan

    frame_dt = float(np.median(diffs))
    duration_sec = float(len(t) * frame_dt)

    if duration_sec <= 0 or frame_dt <= 0:
        return np.nan, np.nan, np.nan, np.nan

    fps_median_dt = 1.0 / frame_dt
    fps_count_based = len(t) / duration_sec

    return duration_sec, frame_dt, fps_median_dt, fps_count_based


def count_one_case(case_dir: Path, root: Path):
    """
    统计单个 case。
    """
    rel_case = str(case_dir.relative_to(root))

    count_file = find_first_existing_file(case_dir, CASE_FILE_CANDIDATES)

    if count_file is None:
        return None

    df = read_csv_safely(count_file)

    total_frames = int(len(df))
    valid_frames = np.nan
    valid_method = "unknown"

    # --------------------------------------------------------
    # 1. 优先用 is_teacher_frame
    # --------------------------------------------------------
    if "is_teacher_frame" in df.columns:
        is_teacher = normalize_bool_series(df["is_teacher_frame"])
        valid_frames = int(is_teacher.sum())
        valid_method = "is_teacher_frame == True"

    # --------------------------------------------------------
    # 2. 其次用 skip_t2s
    # skip_t2s = 0 表示参与 T2S
    # skip_t2s = 1 表示跳过
    # --------------------------------------------------------
    elif "skip_t2s" in df.columns:
        skip = normalize_bool_series(df["skip_t2s"])
        valid_frames = int((~skip).sum())
        valid_method = "skip_t2s == 0"

    # --------------------------------------------------------
    # 3. 再用 meta_state >= 0
    # --------------------------------------------------------
    else:
        meta_col = None

        for c in META_STATE_COL_CANDIDATES:
            if c in df.columns:
                meta_col = c
                break

        if meta_col is not None:
            meta = pd.to_numeric(df[meta_col], errors="coerce")
            valid_frames = int((meta >= 0).sum())
            valid_method = f"{meta_col} >= 0"
        else:
            valid_frames = total_frames
            valid_method = "no mask column, all rows treated as valid"

    invalid_frames = int(total_frames - valid_frames)
    valid_ratio = valid_frames / total_frames if total_frames > 0 else np.nan
    filtered_ratio = invalid_frames / total_frames if total_frames > 0 else np.nan

    # --------------------------------------------------------
    # skip_t2s 统计
    # --------------------------------------------------------
    skip_t2s_0 = np.nan
    skip_t2s_1 = np.nan

    if "skip_t2s" in df.columns:
        skip = normalize_bool_series(df["skip_t2s"])
        skip_t2s_1 = int(skip.sum())
        skip_t2s_0 = int((~skip).sum())

    # --------------------------------------------------------
    # meta_state 有效统计
    # --------------------------------------------------------
    meta_valid_frames = np.nan
    meta_invalid_frames = np.nan
    meta_col_used = ""

    for c in META_STATE_COL_CANDIDATES:
        if c in df.columns:
            meta_col_used = c
            meta = pd.to_numeric(df[c], errors="coerce")
            meta_valid_frames = int((meta >= 0).sum())
            meta_invalid_frames = int((meta < 0).sum())
            break

    # --------------------------------------------------------
    # marker_type 统计
    # --------------------------------------------------------
    marker_counts = {}

    if "marker_type" in df.columns:
        vc = df["marker_type"].astype(str).value_counts()

        for k, v in vc.items():
            safe_k = str(k).replace(" ", "_").replace("-", "_")
            marker_counts[f"marker_{safe_k}"] = int(v)

    # --------------------------------------------------------
    # FPS 和时长统计
    # --------------------------------------------------------
    time_df, time_file, time_col = find_time_df(case_dir)

    duration_sec = np.nan
    frame_dt = np.nan
    fps_median_dt = np.nan
    fps_count_based = np.nan

    valid_duration_sec = np.nan
    filtered_duration_sec = np.nan
    valid_frames_per_video_sec = np.nan

    if time_df is not None and time_col is not None:
        duration_sec, frame_dt, fps_median_dt, fps_count_based = estimate_fps_and_duration_from_time(
            time_df[time_col]
        )

        if not np.isnan(frame_dt):
            valid_duration_sec = valid_frames * frame_dt
            filtered_duration_sec = invalid_frames * frame_dt

        if not np.isnan(duration_sec) and duration_sec > 0:
            valid_frames_per_video_sec = valid_frames / duration_sec

    duration_min = duration_sec / 60.0 if not pd.isna(duration_sec) else np.nan
    valid_duration_min = valid_duration_sec / 60.0 if not pd.isna(valid_duration_sec) else np.nan
    filtered_duration_min = filtered_duration_sec / 60.0 if not pd.isna(filtered_duration_sec) else np.nan

    row = {
        "case": rel_case,
        "count_file": count_file.name,
        "valid_method": valid_method,

        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "filtered_frames": invalid_frames,
        "invalid_frames": invalid_frames,
        "valid_ratio": valid_ratio,
        "filtered_ratio": filtered_ratio,

        "skip_t2s_0_valid": skip_t2s_0,
        "skip_t2s_1_skipped": skip_t2s_1,

        "meta_col_used": meta_col_used,
        "meta_valid_frames": meta_valid_frames,
        "meta_invalid_frames": meta_invalid_frames,

        "time_file": time_file.name if time_file is not None else "",
        "time_col": time_col if time_col is not None else "",

        "frame_dt_sec": frame_dt,
        "fps_median_dt": fps_median_dt,
        "fps_count_based": fps_count_based,

        "duration_sec": duration_sec,
        "duration_min": duration_min,
        "duration_hms": seconds_to_hms(duration_sec),

        "valid_duration_sec": valid_duration_sec,
        "valid_duration_min": valid_duration_min,
        "valid_duration_hms": seconds_to_hms(valid_duration_sec),

        "filtered_duration_sec": filtered_duration_sec,
        "filtered_duration_min": filtered_duration_min,
        "filtered_duration_hms": seconds_to_hms(filtered_duration_sec),

        "valid_frames_per_video_sec": valid_frames_per_video_sec,
    }

    row.update(marker_counts)
    return row


def main():
    print("=" * 80)
    print("[INFO] Project root:", PROJECT_ROOT)
    print("[INFO] Output root :", OUTPUT_ROOT)
    print("[INFO] Report dir  :", REPORT_DIR)
    print("=" * 80)

    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"输出根目录不存在：{OUTPUT_ROOT}")

    case_dirs = find_case_dirs(OUTPUT_ROOT)

    if not case_dirs:
        raise RuntimeError(f"没有找到任何 case 文件夹：{OUTPUT_ROOT}")

    print(f"[INFO] 找到 case 数量：{len(case_dirs)}")

    rows = []

    for case_dir in case_dirs:
        row = count_one_case(case_dir, OUTPUT_ROOT)

        if row is not None:
            rows.append(row)

            print(
                f"[OK] {row['case']} | "
                f"total={row['total_frames']} | "
                f"valid={row['valid_frames']} | "
                f"filtered={row['filtered_frames']} | "
                f"duration={row['duration_hms']} | "
                f"filtered_duration={row['filtered_duration_hms']} | "
                f"fps≈{row['fps_median_dt']:.3f}"
            )

    summary = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 汇总统计
    # --------------------------------------------------------
    total_frames = int(summary["total_frames"].sum())
    valid_frames = int(summary["valid_frames"].sum())
    filtered_frames = int(summary["filtered_frames"].sum())
    invalid_frames = filtered_frames

    valid_ratio = valid_frames / total_frames if total_frames > 0 else np.nan
    filtered_ratio = filtered_frames / total_frames if total_frames > 0 else np.nan

    total_duration_sec = summary["duration_sec"].dropna().sum()
    total_valid_duration_sec = summary["valid_duration_sec"].dropna().sum()
    total_filtered_duration_sec = summary["filtered_duration_sec"].dropna().sum()

    if total_duration_sec > 0:
        overall_fps_count_based = total_frames / total_duration_sec
        overall_valid_frames_per_sec = valid_frames / total_duration_sec
    else:
        overall_fps_count_based = np.nan
        overall_valid_frames_per_sec = np.nan

    mean_case_fps = summary["fps_median_dt"].dropna().mean()
    median_case_fps = summary["fps_median_dt"].dropna().median()

    total_row = pd.DataFrame([{
        "num_cases": len(summary),

        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "filtered_frames": filtered_frames,
        "invalid_frames": invalid_frames,

        "valid_ratio": valid_ratio,
        "filtered_ratio": filtered_ratio,

        "total_video_duration_sec": total_duration_sec,
        "total_video_duration_min": total_duration_sec / 60.0 if total_duration_sec > 0 else np.nan,
        "total_video_duration_hms": seconds_to_hms(total_duration_sec),

        "total_valid_duration_sec": total_valid_duration_sec,
        "total_valid_duration_min": total_valid_duration_sec / 60.0 if total_valid_duration_sec > 0 else np.nan,
        "total_valid_duration_hms": seconds_to_hms(total_valid_duration_sec),

        "total_filtered_duration_sec": total_filtered_duration_sec,
        "total_filtered_duration_min": total_filtered_duration_sec / 60.0 if total_filtered_duration_sec > 0 else np.nan,
        "total_filtered_duration_hms": seconds_to_hms(total_filtered_duration_sec),

        "overall_fps_count_based": overall_fps_count_based,
        "overall_valid_frames_per_sec": overall_valid_frames_per_sec,
        "mean_case_fps_median_dt": mean_case_fps,
        "median_case_fps_median_dt": median_case_fps,
    }])

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    by_case_csv = REPORT_DIR / "valid_frame_fps_duration_summary_by_case.csv"
    total_csv = REPORT_DIR / "valid_frame_fps_duration_summary_total.csv"

    summary.to_csv(by_case_csv, index=False, encoding="utf-8-sig")
    total_row.to_csv(total_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[TOTAL]")
    print(f"case 数量              : {len(summary)}")
    print(f"总帧数                 : {total_frames}")
    print(f"有效教师帧数           : {valid_frames}")
    print(f"滤除/无效帧数          : {filtered_frames}")
    print(f"有效帧比例             : {valid_ratio:.2%}")
    print(f"滤除帧比例             : {filtered_ratio:.2%}")
    print("-" * 80)
    print(f"所有视频总时长          : {seconds_to_hms(total_duration_sec)}  ({total_duration_sec:.3f} sec)")
    print(f"所有有效教师区间总时长  : {seconds_to_hms(total_valid_duration_sec)}  ({total_valid_duration_sec:.3f} sec)")
    print(f"所有滤除区间总时长      : {seconds_to_hms(total_filtered_duration_sec)}  ({total_filtered_duration_sec:.3f} sec)")
    print("-" * 80)
    print(f"整体采样 FPS           : {overall_fps_count_based:.3f}")
    print(f"整体每秒有效教师帧数    : {overall_valid_frames_per_sec:.3f}")
    print(f"case 平均 FPS          : {mean_case_fps:.3f}")
    print(f"case 中位 FPS          : {median_case_fps:.3f}")
    print("=" * 80)

    print(f"\n[OUT] by-case summary: {by_case_csv}")
    print(f"[OUT] total summary  : {total_csv}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()