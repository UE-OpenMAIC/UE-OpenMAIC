                       

   

from __future__ import annotations

import argparse

import re

from pathlib import Path

from typing import Optional

from itertools import combinations

import numpy as np

import pandas as pd

                                                              

         

                                                              

DOC_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\doc")

EXPERT_NAMES_DEFAULT = "expertB,expertC,expertD"

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch_orientation8")

MODEL_SEGMENTS_DEFAULT = (

    T2S_ROOT_DEFAULT

    / "_cross_video_prototype_alignment_X1_only_orientation8"

    / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

)

OUT_DIR_NAME = "_eval_against_global_model_3class_teacher_mask_grid"

COMBINED_OUT_DIR_NAME = "_eval_BCD_global_3class_teacher_mask_grid"

FINAL_TABLE_OUT_DIR_NAME = "_final_expert_validation_table"

          

CLASSES = ["自然讲授", "侧向讲授", "板书书写"]

                                                              

                             

                                                              

GLOBAL_TO_COARSE = {

                 

                 

    0: "自然讲授",

    2: "自然讲授",

              

                

              

                

                 

    1: "侧向讲授",

    6: "侧向讲授",

    7: "侧向讲授",

    8: "侧向讲授",

    9: "侧向讲授",

             

                     

               

                                         

    3: "板书书写",

    4: "板书书写",

    5: "板书书写",

    10: "板书书写",

}

                                                              

         

                                                              

def norm_video_id(x) -> str:

    s = str(x).strip().replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

    s = re.sub(r"/+", "/", s).strip("/")

    return s

def tail_video_id(x) -> str:

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def video_aliases(x) -> set[str]:

    vid = norm_video_id(x)

    tail = tail_video_id(vid)

    out = {vid, tail}

    if tail.isdigit():

        out.add(str(int(tail)))

        out.add(f"{int(tail):02d}")

        out.add(f"{int(tail)}/{int(tail)}")

        out.add(f"{int(tail):02d}/{int(tail):02d}")

    return {v for v in out if v}

def infer_video_id_from_txt(txt_path: Path) -> str:

    

       

    stem = txt_path.stem.strip()

                                             

    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)

    m = re.match(r"^(\d+)[_\-](\d+)$", stem)

    if m:

        return f"{int(m.group(1))}/{int(m.group(2))}"

    if stem.isdigit():

        return str(int(stem))

    return norm_video_id(stem)

def _seconds_from_dot_parts(left: str, right: str) -> Optional[float]:

    

       

    if not re.fullmatch(r"\d+", left) or not re.fullmatch(r"\d{1,2}", right):

        return None

    mm = int(left)

    if len(right) == 1:

                                     

                             

        ss = int(right) * 10

    else:

        ss = int(right)

    if 0 <= ss < 60:

        return float(mm * 60 + ss)

    return None

def _parse_time_token_basic(s: str) -> float:

    

       

    s = str(s).strip().replace("：", ":")

    if not s:

        return np.nan

    if ":" in s:

        parts = s.split(":")

        try:

            nums = [float(p) for p in parts]

        except Exception:

            return np.nan

        if len(nums) == 2:

            return nums[0] * 60.0 + nums[1]

        if len(nums) == 3:

            return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]

        return np.nan

    if "." in s:

        left, right = s.split(".", 1)

        val = _seconds_from_dot_parts(left, right)

        if val is not None:

            return val

    if re.fullmatch(r"\d{3,4}", s):

                                   

                                         

        if len(s) == 4 and int(s[:2]) >= 60 and int(s[-2:]) < 60:

            mm = int(s[0])

            ss = int(s[-2:])

        else:

            mm = int(s[:-2])

            ss = int(s[-2:])

        if 0 <= ss < 60:

            return float(mm * 60 + ss)

    try:

        return float(s)

    except Exception:

        return np.nan

def parse_time_range(start_token: str, end_token: str, max_repair_segment_sec: float = 180.0) -> tuple[float, float, str]:

    

       

    start_s = _parse_time_token_basic(start_token)

    end_s = _parse_time_token_basic(end_token)

    repair_note = ""

    if pd.isna(start_s) or pd.isna(end_s):

        return start_s, end_s, repair_note

    duration = float(end_s - start_s)

                                               

    if duration <= 0 or duration > float(max_repair_segment_sec):

        e = str(end_token).strip()

        start_min = int(float(start_s) // 60)

        candidate = None

        if "." in e:

            left, right = e.split(".", 1)

            if re.fullmatch(r"\d+", left) and re.fullmatch(r"\d{1,2}", right):

                                            

                if len(right) == 1:

                    ss = int(right) * 10

                else:

                    ss = int(right)

                if 0 <= ss < 60:

                    c1 = start_min * 60 + ss

                    c2 = (start_min + 1) * 60 + ss

                    if c1 > start_s:

                        candidate = c1

                    elif c2 > start_s:

                        candidate = c2

        elif re.fullmatch(r"\d{3,4}", e):

                                               

            ss = int(e[-2:])

            if 0 <= ss < 60:

                if len(e) == 4 and int(e[:2]) >= 60:

                    mm = int(e[0])

                    candidate = mm * 60 + ss

                else:

                    mm = int(e[:-2])

                    candidate = mm * 60 + ss

        if candidate is not None:

            cand_duration = float(candidate - start_s)

            if 0 < cand_duration <= float(max_repair_segment_sec):

                repair_note = f"end_time_repaired:{end_token}->{candidate:.2f}s"

                end_s = float(candidate)

    return float(start_s), float(end_s), repair_note

def normalize_expert_label(label: str) -> str:

    

       

    s = str(label).strip()

    s = s.replace(" ", "").replace("　", "")

    s = s.replace("：", ":").replace("，", "").replace(",", "")

    s = s.replace("。", "").replace(".", "")

    if not s:

        return "未知"

          

    if s in {"自", "正"}:

        return "自然讲授"

    if s in {"侧"}:

        return "侧向讲授"

    if s in {"板"}:

        return "板书书写"

                          

    if s in {"板书书写", "板书", "书写", "写板书", "板书讲解", "板书讲授", "板书授课"}:

        return "板书书写"

    if "板书" in s or "书写" in s or s.startswith("板"):

        return "板书书写"

         

    if s in {

        "侧向讲授", "侧向讲解", "侧身讲解", "侧身讲授",

        "侧立讲解", "侧立讲授", "侧面讲授", "侧面讲解",

        "侧身", "侧面", "侧向", "侧立"

    }:

        return "侧向讲授"

    if "侧向" in s or "侧身" in s or "侧立" in s or "侧面" in s or s.startswith("侧"):

        return "侧向讲授"

                 

    if "提问" in s or "互动" in s:

        return "侧向讲授"

               

    if s in {

        "自然讲授", "自然讲解", "自然", "正面讲授", "正面讲解",

        "正向讲授", "正向讲解", "正向讲授讲解", "正面"

    }:

        return "自然讲授"

    if "自然" in s or "正面" in s or "正向" in s or s.startswith("自") or s.startswith("正"):

        return "自然讲授"

    return "未知"

def sec_to_mmss(sec: float) -> str:

    sec = float(sec)

    mm = int(sec // 60)

    ss = sec - mm * 60

    return f"{mm}:{ss:05.2f}"

                                                              

           

                                                              

def parse_expert_txt(

    txt_path: Path,

    fill_small_gaps_sec: float = 1.0,

    max_repair_segment_sec: float = 180.0,

    expert_name: str = "",

) -> pd.DataFrame:

    video_id = infer_video_id_from_txt(txt_path)

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    rows = []

    for line_idx, line in enumerate(lines, start=1):

        raw = line.strip()

        if not raw:

            continue

                                                 

        m = re.match(r"^\s*([0-9:.]+)\s*[-—~－]\s*([0-9:.]+)\s*(.+?)\s*$", raw)

        if not m:

            print(f"[WARN] 无法解析专家行: {txt_path.name}:{line_idx}: {raw}")

            continue

        start_s, end_s, repair_note = parse_time_range(

            m.group(1),

            m.group(2),

            max_repair_segment_sec=max_repair_segment_sec,

        )

        label_raw = m.group(3).strip()

        label = normalize_expert_label(label_raw)

        if pd.isna(start_s) or pd.isna(end_s) or end_s <= start_s:

            print(f"[WARN] 时间异常: {txt_path.name}:{line_idx}: {raw} -> {start_s}, {end_s}")

            continue

        rows.append({

            "expert_name": expert_name,

            "video_id": norm_video_id(video_id),

            "source_file": str(txt_path),

            "line_idx": int(line_idx),

            "expert_start_sec": float(start_s),

            "expert_end_sec": float(end_s),

            "expert_duration_sec": float(end_s - start_s),

            "expert_label_raw": label_raw,

            "expert_label": label,

            "time_repair_note": repair_note,

            "raw_line": raw,

        })

    df = pd.DataFrame(rows)

    if df.empty:

        return df

    df = df.sort_values(["expert_start_sec", "expert_end_sec"]).reset_index(drop=True)

          

    if fill_small_gaps_sec is not None and fill_small_gaps_sec > 0 and len(df) > 1:

        starts = df["expert_start_sec"].to_numpy()

        ends = df["expert_end_sec"].to_numpy().copy()

        for i in range(len(df) - 1):

            gap = starts[i + 1] - ends[i]

            if 0 < gap <= float(fill_small_gaps_sec):

                ends[i] = starts[i + 1]

        df["expert_end_sec"] = ends

        df["expert_duration_sec"] = df["expert_end_sec"] - df["expert_start_sec"]

    return df

def load_all_expert_labels(expert_dir: Path, fill_small_gaps_sec: float, max_repair_segment_sec: float, expert_name: str) -> pd.DataFrame:

    txt_files = sorted(expert_dir.glob("*.txt"))

    if not txt_files:

        raise FileNotFoundError(f"专家目录下没有 txt 文件: {expert_dir}")

    dfs = []

    for p in txt_files:

        df = parse_expert_txt(

            p,

            fill_small_gaps_sec=fill_small_gaps_sec,

            max_repair_segment_sec=max_repair_segment_sec,

            expert_name=expert_name,

        )

        if not df.empty:

            dfs.append(df)

    if not dfs:

        raise RuntimeError(f"没有成功解析任何专家标注: {expert_dir}")

    out = pd.concat(dfs, axis=0, ignore_index=True)

    unknown = out[out["expert_label"].eq("未知")]

    if not unknown.empty:

        print("[WARN] 出现无法归一化的专家标签：")

        print(unknown[["expert_name", "video_id", "source_file", "line_idx", "expert_label_raw", "raw_line"]].to_string(index=False))

    repairs = out[out["time_repair_note"].astype(str).ne("")]

    if not repairs.empty:

        print("[INFO] 已自动修复疑似异常时间：")

        print(repairs[["expert_name", "video_id", "source_file", "line_idx", "raw_line", "time_repair_note"]].to_string(index=False))

    return out

                                                              

                                

                                                              

def load_model_segments(model_segments_csv: Path) -> pd.DataFrame:

    if not model_segments_csv.exists():

        raise FileNotFoundError(f"找不到模型 segment 文件: {model_segments_csv}")

    df = pd.read_csv(model_segments_csv)

    required = ["video_id", "start_sec", "end_sec", "global_state"]

    missing = [c for c in required if c not in df.columns]

    if missing:

        raise ValueError(f"模型 segment 文件缺少列: {missing}")

    df = df.copy()

    df["video_id"] = df["video_id"].astype(str).map(norm_video_id)

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")

    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    df["global_state"] = pd.to_numeric(df["global_state"], errors="coerce").fillna(-1).astype(int)

    df["model_label"] = df["global_state"].map(lambda x: GLOBAL_TO_COARSE.get(int(x), "未知"))

    return df.dropna(subset=["start_sec", "end_sec"]).reset_index(drop=True)

def select_model_video(model_df: pd.DataFrame, expert_video_id: str) -> pd.DataFrame:

    aliases = video_aliases(expert_video_id)

    mask = model_df["video_id"].map(lambda v: bool(video_aliases(v) & aliases))

    return model_df.loc[mask].copy().sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

def find_marker_csv(t2s_root: Path, video_id: str) -> Path | None:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    candidates = [

        t2s_root / Path(vid) / "student_marker_mask.csv",

        t2s_root / tail / tail / "student_marker_mask.csv",

        t2s_root / tail / "student_marker_mask.csv",

    ]

    if tail.isdigit():

        two = f"{int(tail):02d}"

        candidates.extend([

            t2s_root / two / two / "student_marker_mask.csv",

            t2s_root / two / "student_marker_mask.csv",

            t2s_root / f"{int(tail)}" / f"{int(tail)}" / "student_marker_mask.csv",

        ])

    for p in candidates:

        if p.exists():

            return p

    return None

def load_marker_intervals(marker_csv: Path) -> pd.DataFrame:

    df = pd.read_csv(marker_csv)

    if "time_sec" not in df.columns:

        raise ValueError(f"marker 文件缺少 time_sec: {marker_csv}")

    df = df.copy().sort_values("time_sec").reset_index(drop=True)

    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")

    df = df.dropna(subset=["time_sec"]).reset_index(drop=True)

    if "is_teacher_frame" not in df.columns:

        if "skip_t2s" in df.columns:

            df["is_teacher_frame"] = 1 - pd.to_numeric(df["skip_t2s"], errors="coerce").fillna(0).astype(int)

        else:

            raise ValueError(f"marker 文件缺少 is_teacher_frame/skip_t2s: {marker_csv}")

    df["is_teacher_frame"] = pd.to_numeric(df["is_teacher_frame"], errors="coerce").fillna(1).astype(int)

    t = df["time_sec"].to_numpy(dtype=float)

    if len(t) < 2:

        raise ValueError(f"marker 时间点太少: {marker_csv}")

    dt = np.diff(t)

    good_dt = dt[dt > 0]

    median_dt = float(np.median(good_dt)) if len(good_dt) else 1.0 / 12.0

    start = t.copy()

    end = np.empty_like(start)

    end[:-1] = t[1:]

    end[-1] = t[-1] + median_dt

    out = pd.DataFrame({

        "grid_start_sec": start,

        "grid_end_sec": end,

        "grid_mid_sec": (start + end) / 2.0,

        "grid_duration_sec": np.maximum(0.0, end - start),

        "is_teacher_frame": df["is_teacher_frame"].astype(int).to_numpy(),

    })

    if "marker_type" in df.columns:

        out["marker_type"] = df["marker_type"].astype(str).to_numpy()

    else:

        out["marker_type"] = np.where(out["is_teacher_frame"].eq(1), "teacher", "skip")

    return out

                                                              

                     

                                                              

def label_at_time_from_segments(seg_df: pd.DataFrame, t: float):

    hit = seg_df[(seg_df["start_sec"] <= t) & (seg_df["end_sec"] > t)]

    if hit.empty:

        return -1, "未匹配"

    hit = hit.copy()

    hit["dur"] = hit["end_sec"] - hit["start_sec"]

    row = hit.sort_values("dur").iloc[0]

    gs = int(row["global_state"])

    lab = str(row["model_label"])

    return gs, lab

def label_at_time_from_expert(ex_df: pd.DataFrame, t: float):

    hit = ex_df[(ex_df["expert_start_sec"] <= t) & (ex_df["expert_end_sec"] > t)]

    if hit.empty:

        return "未标注", ""

                    

    hit = hit.copy()

    hit["dur"] = hit["expert_end_sec"] - hit["expert_start_sec"]

    row = hit.sort_values("dur").iloc[0]

    return str(row["expert_label"]), str(row["expert_label_raw"])

def confusion_from_detail(detail_df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:

    conf = pd.DataFrame(0.0, index=classes, columns=classes)

    for _, r in detail_df.iterrows():

        t = str(r["expert_label"])

        p = str(r["model_label"])

        if t in classes and p in classes:

            conf.loc[t, p] += float(r["duration_sec"])

    return conf

def metrics_from_confusion(conf: pd.DataFrame, classes: list[str]) -> dict:

    total = float(conf.to_numpy().sum())

    correct = sum(float(conf.loc[c, c]) for c in classes)

    accuracy = correct / total if total > 0 else np.nan

    rows = []

    f1s = []

    weights = []

    for c in classes:

        tp = float(conf.loc[c, c])

        fp = float(conf[c].sum() - tp)

        fn = float(conf.loc[c].sum() - tp)

        support = float(conf.loc[c].sum())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0

        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        rows.append({

            "class": c,

            "support_sec": support,

            "precision": precision,

            "recall": recall,

            "f1": f1,

        })

        if support > 0:

            f1s.append(f1)

            weights.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else np.nan

    weighted_f1 = float(np.average(f1s, weights=weights)) if f1s and sum(weights) > 0 else np.nan

    return {

        "total_sec": total,

        "correct_sec": correct,

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "weighted_f1": weighted_f1,

        "per_class": pd.DataFrame(rows),

    }

def evaluate_teacher_mask_grid(

    expert_df: pd.DataFrame,

    model_df: pd.DataFrame,

    t2s_root: Path,

    classes: list[str],

):

    detail_rows = []

    skipped_rows = []

    missing_marker_rows = []

    expert_name = str(expert_df["expert_name"].iloc[0]) if "expert_name" in expert_df.columns and len(expert_df) else ""

    for video_id, ex_g in expert_df.groupby("video_id"):

        mod_g = select_model_video(model_df, video_id)

        marker_path = find_marker_csv(t2s_root, video_id)

        if marker_path is None:

            missing_marker_rows.append({

                "expert_name": expert_name,

                "video_id": video_id,

                "reason": "student_marker_mask.csv not found",

            })

            print(f"[WARN] 找不到 marker mask，跳过该视频 teacher-mask 评价: {expert_name} {video_id}")

            continue

        grid = load_marker_intervals(marker_path)

        ex_min = float(ex_g["expert_start_sec"].min())

        ex_max = float(ex_g["expert_end_sec"].max())

        grid = grid[(grid["grid_mid_sec"] >= ex_min) & (grid["grid_mid_sec"] < ex_max)].copy()

        for _, gr in grid.iterrows():

            tmid = float(gr["grid_mid_sec"])

            dur = float(gr["grid_duration_sec"])

            if dur <= 0:

                continue

            exp_label, exp_raw = label_at_time_from_expert(ex_g, tmid)

            if int(gr["is_teacher_frame"]) != 1:

                skipped_rows.append({

                    "expert_name": expert_name,

                    "video_id": video_id,

                    "start_sec": float(gr["grid_start_sec"]),

                    "end_sec": float(gr["grid_end_sec"]),

                    "duration_sec": dur,

                    "reason": "non_teacher_marker",

                    "marker_type": str(gr.get("marker_type", "")),

                    "expert_label": exp_label,

                    "expert_label_raw": exp_raw,

                })

                continue

            if exp_label not in classes:

                skipped_rows.append({

                    "expert_name": expert_name,

                    "video_id": video_id,

                    "start_sec": float(gr["grid_start_sec"]),

                    "end_sec": float(gr["grid_end_sec"]),

                    "duration_sec": dur,

                    "reason": "expert_label_not_in_classes",

                    "marker_type": str(gr.get("marker_type", "")),

                    "expert_label": exp_label,

                    "expert_label_raw": exp_raw,

                })

                continue

            gs, model_label = label_at_time_from_segments(mod_g, tmid)

            if int(gs) < 0 or model_label not in classes:

                skipped_rows.append({

                    "expert_name": expert_name,

                    "video_id": video_id,

                    "start_sec": float(gr["grid_start_sec"]),

                    "end_sec": float(gr["grid_end_sec"]),

                    "duration_sec": dur,

                    "reason": "invalid_or_unmatched_model_state",

                    "marker_type": str(gr.get("marker_type", "")),

                    "expert_label": exp_label,

                    "expert_label_raw": exp_raw,

                    "global_state": int(gs),

                    "model_label": model_label,

                })

                continue

            detail_rows.append({

                "expert_name": expert_name,

                "video_id": video_id,

                "start_sec": float(gr["grid_start_sec"]),

                "end_sec": float(gr["grid_end_sec"]),

                "mid_sec": tmid,

                "duration_sec": dur,

                "expert_label": exp_label,

                "expert_label_raw": exp_raw,

                "global_state": int(gs),

                "model_label": model_label,

                "match": int(exp_label == model_label),

            })

    detail_df = pd.DataFrame(detail_rows)

    skipped_df = pd.DataFrame(skipped_rows)

    missing_marker_df = pd.DataFrame(missing_marker_rows)

    if detail_df.empty:

        raise RuntimeError(f"{expert_name} 没有任何有效 teacher-frame 评价样本，请检查 marker/model/expert 文件。")

    conf = confusion_from_detail(detail_df, classes)

    met = metrics_from_confusion(conf, classes)

    per_video_rows = []

    per_video_per_class = []

    for vid, g in detail_df.groupby("video_id"):

        c = confusion_from_detail(g, classes)

        m = metrics_from_confusion(c, classes)

        per_video_rows.append({

            "expert_name": expert_name,

            "video_id": vid,

            "total_sec": m["total_sec"],

            "correct_sec": m["correct_sec"],

            "accuracy": m["accuracy"],

            "macro_f1": m["macro_f1"],

            "weighted_f1": m["weighted_f1"],

        })

        pc = m["per_class"].copy()

        pc.insert(0, "video_id", vid)

        pc.insert(0, "expert_name", expert_name)

        per_video_per_class.append(pc)

    skipped_total_sec = float(skipped_df["duration_sec"].sum()) if not skipped_df.empty else 0.0

    skipped_by_reason = (

        skipped_df.groupby(["expert_name", "reason"])["duration_sec"].sum().reset_index()

        if not skipped_df.empty else pd.DataFrame(columns=["expert_name", "reason", "duration_sec"])

    )

    overall = pd.DataFrame([{

        "expert_name": expert_name,

        "n_videos": int(detail_df["video_id"].nunique()),

        "total_sec": met["total_sec"],

        "correct_sec": met["correct_sec"],

        "accuracy": met["accuracy"],

        "macro_f1": met["macro_f1"],

        "weighted_f1": met["weighted_f1"],

        "skipped_total_sec": skipped_total_sec,

    }])

    return {

        "detail": detail_df,

        "skipped": skipped_df,

        "missing_marker": missing_marker_df,

        "skipped_by_reason": skipped_by_reason,

        "confusion": conf,

        "overall_metrics": overall,

        "overall_per_class": met["per_class"],

        "per_video_metrics": pd.DataFrame(per_video_rows).sort_values(["expert_name", "video_id"]).reset_index(drop=True),

        "per_video_per_class": pd.concat(per_video_per_class, axis=0, ignore_index=True) if per_video_per_class else pd.DataFrame(),

    }

def summarize_detail(detail_df: pd.DataFrame, skipped_df: pd.DataFrame, classes: list[str]) -> dict:

    conf = confusion_from_detail(detail_df, classes)

    met = metrics_from_confusion(conf, classes)

    skipped_total_sec = float(skipped_df["duration_sec"].sum()) if skipped_df is not None and not skipped_df.empty else 0.0

    overall = pd.DataFrame([{

        "expert_name": "BCD_pooled",

        "n_experts": int(detail_df["expert_name"].nunique()) if "expert_name" in detail_df.columns else np.nan,

        "n_videos": int(detail_df["video_id"].nunique()),

        "total_sec": met["total_sec"],

        "correct_sec": met["correct_sec"],

        "accuracy": met["accuracy"],

        "macro_f1": met["macro_f1"],

        "weighted_f1": met["weighted_f1"],

        "skipped_total_sec": skipped_total_sec,

    }])

    return {

        "confusion": conf,

        "overall_metrics": overall,

        "overall_per_class": met["per_class"],

    }

                                                              

         

                                                              

def write_results(out_dir: Path, expert_df: pd.DataFrame, results: dict, expert_name: str) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = f"{expert_name}_normalized_segments.csv" if expert_name else "expert_normalized_segments.csv"

    expert_df.to_csv(out_dir / normalized_name, index=False, encoding="utf-8-sig")

                                                           

    if expert_name == "expertB":

        expert_df.to_csv(out_dir / "expertB_normalized_segments.csv", index=False, encoding="utf-8-sig")

    results["detail"].to_csv(out_dir / "teacher_mask_grid_detail.csv", index=False, encoding="utf-8-sig")

    results["skipped"].to_csv(out_dir / "teacher_mask_grid_skipped_intervals.csv", index=False, encoding="utf-8-sig")

    results["skipped_by_reason"].to_csv(out_dir / "teacher_mask_grid_skipped_by_reason.csv", index=False, encoding="utf-8-sig")

    results["missing_marker"].to_csv(out_dir / "missing_marker_videos.csv", index=False, encoding="utf-8-sig")

    results["confusion"].to_csv(out_dir / "confusion_seconds.csv", encoding="utf-8-sig")

    results["overall_metrics"].to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    results["overall_per_class"].to_csv(out_dir / "overall_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    results["per_video_metrics"].to_csv(out_dir / "per_video_metrics.csv", index=False, encoding="utf-8-sig")

    results["per_video_per_class"].to_csv(out_dir / "per_video_per_class_metrics.csv", index=False, encoding="utf-8-sig")

def write_combined_results(out_dir: Path, all_expert_df: pd.DataFrame, all_results: list[dict]) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.concat([r["detail"] for r in all_results], axis=0, ignore_index=True)

    skipped = pd.concat([r["skipped"] for r in all_results], axis=0, ignore_index=True)

    missing = pd.concat([r["missing_marker"] for r in all_results], axis=0, ignore_index=True)

    per_video = pd.concat([r["per_video_metrics"] for r in all_results], axis=0, ignore_index=True)

    per_video_per_class = pd.concat([r["per_video_per_class"] for r in all_results], axis=0, ignore_index=True)

    per_expert_overall = pd.concat([r["overall_metrics"] for r in all_results], axis=0, ignore_index=True)

    combined = summarize_detail(detail, skipped, CLASSES)

    all_expert_df.to_csv(out_dir / "BCD_normalized_segments.csv", index=False, encoding="utf-8-sig")

    detail.to_csv(out_dir / "BCD_teacher_mask_grid_detail.csv", index=False, encoding="utf-8-sig")

    skipped.to_csv(out_dir / "BCD_teacher_mask_grid_skipped_intervals.csv", index=False, encoding="utf-8-sig")

    missing.to_csv(out_dir / "BCD_missing_marker_videos.csv", index=False, encoding="utf-8-sig")

    per_expert_overall.to_csv(out_dir / "BCD_per_expert_overall_metrics.csv", index=False, encoding="utf-8-sig")

    per_video.to_csv(out_dir / "BCD_per_expert_per_video_metrics.csv", index=False, encoding="utf-8-sig")

    per_video_per_class.to_csv(out_dir / "BCD_per_expert_per_video_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    combined["confusion"].to_csv(out_dir / "BCD_pooled_confusion_seconds.csv", encoding="utf-8-sig")

    combined["overall_metrics"].to_csv(out_dir / "BCD_pooled_overall_metrics.csv", index=False, encoding="utf-8-sig")

    combined["overall_per_class"].to_csv(out_dir / "BCD_pooled_overall_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    if not skipped.empty:

        skipped_by_reason = skipped.groupby(["expert_name", "reason"])["duration_sec"].sum().reset_index()

        skipped_by_reason.to_csv(out_dir / "BCD_skipped_by_reason.csv", index=False, encoding="utf-8-sig")

                                                              

                                    

                                                              

def _detail_with_time_key(detail_df: pd.DataFrame, time_round: int = 3) -> pd.DataFrame:

    out = detail_df.copy()

    if "time_key" not in out.columns:

        if "mid_sec" in out.columns:

            base = pd.to_numeric(out["mid_sec"], errors="coerce")

        elif "start_sec" in out.columns:

            base = pd.to_numeric(out["start_sec"], errors="coerce")

        else:

            raise ValueError("detail_df 缺少 mid_sec/start_sec，无法构造 time_key")

        out["time_key"] = base.round(int(time_round)).map(lambda x: f"{float(x):.{int(time_round)}f}")

    return out

def confusion_from_rows(rows_df: pd.DataFrame, true_col: str, pred_col: str, classes: list[str]) -> pd.DataFrame:

    conf = pd.DataFrame(0.0, index=classes, columns=classes)

    if rows_df is None or rows_df.empty:

        return conf

    for _, r in rows_df.iterrows():

        true_label = str(r.get(true_col, ""))

        pred_label = str(r.get(pred_col, ""))

        if true_label in classes and pred_label in classes:

            conf.loc[true_label, pred_label] += float(r.get("duration_sec", 0.0))

    return conf

def compute_pairwise_kappa(detail: pd.DataFrame, classes: list[str]) -> pd.DataFrame:

    rows = []

    if detail is None or detail.empty:

        return pd.DataFrame(columns=["expert_a", "expert_b", "total_sec", "agreement", "cohen_kappa"])

    experts = sorted(detail["expert_name"].dropna().astype(str).unique().tolist())

    for a, b in combinations(experts, 2):

        sub = detail[detail["expert_name"].isin([a, b])].copy()

        piv = sub.pivot_table(

            index=["video_id", "time_key"],

            columns="expert_name",

            values="expert_label",

            aggfunc="first",

        ).reset_index()

        if a not in piv.columns or b not in piv.columns:

            continue

        dur = sub.groupby(["video_id", "time_key"])["duration_sec"].first().reset_index()

        piv = piv.merge(dur, on=["video_id", "time_key"], how="left")

        piv = piv[piv[a].isin(classes) & piv[b].isin(classes)].copy()

        total = float(piv["duration_sec"].sum()) if not piv.empty else 0.0

        agree = float(piv.loc[piv[a].eq(piv[b]), "duration_sec"].sum()) if total > 0 else 0.0

        po = agree / total if total > 0 else np.nan

        pe = 0.0

        for c in classes:

            pa = float(piv.loc[piv[a].eq(c), "duration_sec"].sum()) / total if total > 0 else 0.0

            pb = float(piv.loc[piv[b].eq(c), "duration_sec"].sum()) / total if total > 0 else 0.0

            pe += pa * pb

        kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-12 else np.nan

        rows.append({

            "expert_a": a,

            "expert_b": b,

            "total_sec": total,

            "agreement": po,

            "cohen_kappa": kappa,

        })

    return pd.DataFrame(rows)

def compute_fleiss_kappa(detail: pd.DataFrame, classes: list[str]) -> float:

    if detail is None or detail.empty:

        return np.nan

    experts = sorted(detail["expert_name"].dropna().astype(str).unique().tolist())

    n_raters = len(experts)

    if n_raters < 2:

        return np.nan

    piv = detail.pivot_table(

        index=["video_id", "time_key"],

        columns="expert_name",

        values="expert_label",

        aggfunc="first",

    ).reset_index()

    for e in experts:

        if e not in piv.columns:

            return np.nan

        piv = piv[piv[e].isin(classes)].copy()

    if piv.empty:

        return np.nan

    dur = detail.groupby(["video_id", "time_key"])["duration_sec"].first().reset_index()

    piv = piv.merge(dur, on=["video_id", "time_key"], how="left")

    weights = piv["duration_sec"].astype(float).to_numpy()

    total_sec = float(weights.sum())

    if total_sec <= 0:

        return np.nan

    item_agree = []

    vote_weight = {c: 0.0 for c in classes}

    total_vote_weight = 0.0

    for _, row in piv.iterrows():

        counts = {c: 0 for c in classes}

        for e in experts:

            counts[str(row[e])] += 1

        n = n_raters

        Pi = sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))

        item_agree.append(Pi)

        dur_i = float(row["duration_sec"])

        for c in classes:

            vote_weight[c] += counts[c] * dur_i

        total_vote_weight += n * dur_i

    Pbar = float(np.average(np.asarray(item_agree), weights=weights))

    priors = {c: vote_weight[c] / total_vote_weight if total_vote_weight > 0 else 0.0 for c in classes}

    Pe = sum(v * v for v in priors.values())

    return (Pbar - Pe) / (1.0 - Pe) if (1.0 - Pe) > 1e-12 else np.nan

def build_consensus(detail: pd.DataFrame, classes: list[str], min_experts: int = 2) -> pd.DataFrame:

    rows = []

    if detail is None or detail.empty:

        return pd.DataFrame()

    for (video_id, time_key), g in detail.groupby(["video_id", "time_key"], sort=True):

        duration = float(g["duration_sec"].iloc[0])

        model_label = str(g["model_label"].iloc[0])

        global_state = int(g["global_state"].iloc[0])

        expert_labels = {}

        for _, r in g.iterrows():

            eid = str(r["expert_name"])

            lab = str(r["expert_label"])

            if lab in classes and eid not in expert_labels:

                expert_labels[eid] = lab

        n_exp = len(expert_labels)

        votes = {c: 0 for c in classes}

        for lab in expert_labels.values():

            votes[lab] += 1

        sorted_votes = sorted(votes.items(), key=lambda x: (-x[1], x[0]))

        top_label, top_n = sorted_votes[0]

        second_n = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

        if n_exp >= int(min_experts) and top_n > second_n and top_n >= 2:

            consensus_label = top_label

            status = "consensus"

        elif n_exp >= int(min_experts):

            consensus_label = ""

            status = "disputed"

        else:

            consensus_label = ""

            status = "single_or_missing"

        row = {

            "video_id": video_id,

            "time_key": time_key,

            "duration_sec": duration,

            "global_state": global_state,

            "model_label": model_label,

            "n_experts": n_exp,

            "consensus_label": consensus_label,

            "consensus_status": status,

            "match": int(model_label == consensus_label) if status == "consensus" else np.nan,

        }

        for c in classes:

            row[f"votes_{c}"] = votes[c]

        rows.append(row)

    return pd.DataFrame(rows)

def fmt_pct(x) -> str:

    return f"{100.0 * float(x):.1f}%" if pd.notna(x) else ""

def fmt3(x) -> str:

    return f"{float(x):.3f}" if pd.notna(x) else ""

def dataframe_to_markdown(df: pd.DataFrame) -> str:

    headers = list(df.columns)

    lines = []

    lines.append("| " + " | ".join(headers) + " |")

    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():

        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    return "\n".join(lines) + "\n"

def latex_escape(s: str) -> str:

    return (

        str(s)

        .replace("\\", r"\textbackslash{}")

        .replace("%", r"\%")

        .replace("&", r"\&")

        .replace("_", r"\_")

        .replace("#", r"\#")

    )

def dataframe_to_tex(df: pd.DataFrame) -> str:

    lines = []

    lines.append(r"\begin{tabular}{ll}")

    lines.append(r"\toprule")

    lines.append(r"评价项 & 指标 \\")

    lines.append(r"\midrule")

    for _, row in df.iterrows():

        a = latex_escape(row["评价项"])

        b = latex_escape(row["指标"])

        lines.append(f"{a} & {b} \\")

    lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")

    return "\n".join(lines) + "\n"

def write_final_expert_validation_table(

    out_dir: Path,

    all_results: list[dict],

    classes: list[str],

    min_experts: int = 2,

    time_round: int = 3,

) -> dict:

    """生成 final_expert_validation_3row_table.csv/md/tex。"""

    out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.concat([r["detail"] for r in all_results], axis=0, ignore_index=True)

    skipped = pd.concat([r["skipped"] for r in all_results], axis=0, ignore_index=True)

    detail = _detail_with_time_key(detail, time_round=int(time_round))

    detail = detail[detail["expert_label"].isin(classes) & detail["model_label"].isin(classes)].copy()

                         

    pooled_conf = confusion_from_rows(detail, true_col="expert_label", pred_col="model_label", classes=classes)

    pooled = metrics_from_confusion(pooled_conf, classes)

                                                           

    pairwise = compute_pairwise_kappa(detail, classes)

    fleiss_k = compute_fleiss_kappa(detail, classes)

    pair_min = float(pairwise["cohen_kappa"].min()) if not pairwise.empty else np.nan

    pair_max = float(pairwise["cohen_kappa"].max()) if not pairwise.empty else np.nan

                               

    consensus_all = build_consensus(detail, classes=classes, min_experts=int(min_experts))

    consensus_valid = consensus_all[consensus_all["consensus_status"].eq("consensus")].copy()

    disputed_sec = float(consensus_all.loc[consensus_all["consensus_status"].eq("disputed"), "duration_sec"].sum()) if not consensus_all.empty else 0.0

    consensus_conf = confusion_from_rows(consensus_valid, true_col="consensus_label", pred_col="model_label", classes=classes)

    consensus = metrics_from_confusion(consensus_conf, classes)

    f1_map = dict(zip(consensus["per_class"]["class"], consensus["per_class"]["f1"]))

    class_f1 = "/".join(fmt3(f1_map.get(c, np.nan)) for c in classes)

    class_order_text = "/".join(classes)

    final_table = pd.DataFrame([

        {

            "评价项": "专家一致性",

            "指标": f"κ={fmt3(fleiss_k)}",

        },

        {

            "评价项": "共识标签评价",

            "指标": f"Acc. {fmt_pct(consensus['accuracy'])} / Macro-F1 {fmt3(consensus['macro_f1'])} / W-F1 {fmt3(consensus['weighted_f1'])}",

        },

        {

            "评价项": "保守合并评价",

            "指标": f"Acc. {fmt_pct(pooled['accuracy'])} / Macro-F1 {fmt3(pooled['macro_f1'])} / W-F1 {fmt3(pooled['weighted_f1'])}",

        },

    ])

    skipped_total_sec = float(skipped["duration_sec"].sum()) if skipped is not None and not skipped.empty else 0.0

    table_note = (

        f"注：κ 为三专家在共同有效教师区间上的 Fleiss' κ；两两 Cohen's κ 范围为 "

        f"{fmt3(pair_min)}–{fmt3(pair_max)}。共识标签由多数投票得到，disputed 区间"

        f"（{disputed_sec:.2f}s）与非教师/SKIP 区间不纳入共识评价。类别 F1 按"

        f"“{class_order_text}”顺序为 {class_f1}。保守合并评价指直接合并全部专家标注后的模型结果。"

        f"teacher-mask 或模型无效区间合计跳过 {skipped_total_sec:.2f}s。"

    )

    csv_path = out_dir / "final_expert_validation_3row_table.csv"

    md_path = out_dir / "final_expert_validation_3row_table.md"

    tex_path = out_dir / "final_expert_validation_3row_table.tex"

    final_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path.write_text(dataframe_to_markdown(final_table) + "\n" + table_note + "\n", encoding="utf-8")

    tex_path.write_text(dataframe_to_tex(final_table) + "\n% " + table_note.replace("\n", " ") + "\n", encoding="utf-8")

    return {

        "final_table": final_table,

        "table_note": table_note,

        "csv_path": csv_path,

        "md_path": md_path,

        "tex_path": tex_path,

        "pairwise": pairwise,

        "consensus_all": consensus_all,

        "pooled_confusion": pooled_conf,

        "consensus_confusion": consensus_conf,

    }

                                                              

        

                                                              

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--doc-root", type=str, default=str(DOC_ROOT_DEFAULT))

    ap.add_argument("--expert-names", type=str, default=EXPERT_NAMES_DEFAULT,

                    help="逗号分隔，例如 expertB,expertC,expertD；只跑 B 则填 expertB")

    ap.add_argument("--expert-dir", type=str, default="",

                    help="若指定，则只读取该目录，忽略 --doc-root 和 --expert-names")

    ap.add_argument("--model-segments", type=str, default=str(MODEL_SEGMENTS_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--combined-out-dir", type=str, default="")

    ap.add_argument("--final-table-out-dir", type=str, default="",

                    help="论文三行表输出目录；默认 D:\\code\\teacherT2S\\doc\\_final_expert_validation_table")

    ap.add_argument("--min-experts", type=int, default=2,

                    help="生成共识标签时所需的最少专家数，默认 2")

    ap.add_argument("--time-round", type=int, default=3,

                    help="B/C/D 专家网格对齐时 mid_sec 的小数保留位数，默认 3")

    ap.add_argument("--fill-small-gaps-sec", type=float, default=1.0)

    ap.add_argument("--max-repair-segment-sec", type=float, default=180.0)

    return ap.parse_args()

def main():

    args = parse_args()

    doc_root = Path(args.doc_root)

    model_segments = Path(args.model_segments)

    t2s_root = Path(args.t2s_root)

    model_df = load_model_segments(model_segments)

    if args.expert_dir:

        expert_dirs = [(Path(args.expert_dir).parent.parent.name or "expert", Path(args.expert_dir))]

    else:

        names = [x.strip() for x in str(args.expert_names).replace("，", ",").split(",") if x.strip()]

        expert_dirs = [(name, doc_root / name / "global") for name in names]

    print("=" * 100)

    print("Teacher-mask-grid evaluation: B/C/D expert labels vs model global states")

    print(f"doc_root       = {doc_root}")

    print(f"expert_dirs    = {expert_dirs}")

    print(f"model_segments = {model_segments}")

    print(f"t2s_root       = {t2s_root}")

    print(f"classes        = {CLASSES}")

    print(f"global mapping = {GLOBAL_TO_COARSE}")

    print("Policy         = evaluate only teacher frames using student_marker_mask.csv; exclude global_state=-1")

    print("=" * 100)

    all_expert_dfs = []

    all_results = []

    for expert_name, expert_dir in expert_dirs:

        print("\n" + "=" * 100)

        print(f"START {expert_name}")

        print(f"expert_dir = {expert_dir}")

        expert_df = load_all_expert_labels(

            expert_dir=expert_dir,

            fill_small_gaps_sec=float(args.fill_small_gaps_sec),

            max_repair_segment_sec=float(args.max_repair_segment_sec),

            expert_name=expert_name,

        )

        results = evaluate_teacher_mask_grid(

            expert_df=expert_df,

            model_df=model_df,

            t2s_root=t2s_root,

            classes=CLASSES,

        )

        out_dir = expert_dir / OUT_DIR_NAME

        write_results(out_dir, expert_df, results, expert_name)

        all_expert_dfs.append(expert_df)

        all_results.append(results)

        print("\nOVERALL")

        print(results["overall_metrics"].to_string(index=False))

        print("\nPER CLASS")

        print(results["overall_per_class"].to_string(index=False))

        print("\nCONFUSION_SECONDS")

        print(results["confusion"].round(2).to_string())

        print("\nOUTPUT")

        print(out_dir)

    if len(all_results) > 1:

        combined_out = Path(args.combined_out_dir) if args.combined_out_dir else doc_root / COMBINED_OUT_DIR_NAME

        all_expert_df = pd.concat(all_expert_dfs, axis=0, ignore_index=True)

        write_combined_results(combined_out, all_expert_df, all_results)

        final_table_out = Path(args.final_table_out_dir) if args.final_table_out_dir else doc_root / FINAL_TABLE_OUT_DIR_NAME

        final_pack = write_final_expert_validation_table(

            out_dir=final_table_out,

            all_results=all_results,

            classes=CLASSES,

            min_experts=int(args.min_experts),

            time_round=int(args.time_round),

        )

        pooled = summarize_detail(

            pd.concat([r["detail"] for r in all_results], axis=0, ignore_index=True),

            pd.concat([r["skipped"] for r in all_results], axis=0, ignore_index=True),

            CLASSES,

        )

        print("\n" + "=" * 100)

        print("BCD POOLED OVERALL")

        print(pooled["overall_metrics"].to_string(index=False))

        print("\nBCD POOLED PER CLASS")

        print(pooled["overall_per_class"].to_string(index=False))

        print("\nBCD POOLED CONFUSION_SECONDS")

        print(pooled["confusion"].round(2).to_string())

        print("\nCOMBINED OUTPUT")

        print(combined_out)

        print("\nFINAL 3-ROW TABLE")

        print(final_pack["final_table"].to_string(index=False))

        print("\nFINAL TABLE NOTE")

        print(final_pack["table_note"])

        print("\nFINAL TABLE OUTPUTS")

        print(final_pack["csv_path"])

        print(final_pack["md_path"])

        print(final_pack["tex_path"])

    print("=" * 100)

    print("DONE")

if __name__ == "__main__":

    main()
