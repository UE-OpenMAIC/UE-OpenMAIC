                       

   

from __future__ import annotations

import argparse

import re

from pathlib import Path

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

OUT_DIR_DEFAULT = DOC_ROOT_DEFAULT / "_final_expert_validation_table"

CLASSES = ["正向讲授", "侧向讲授", "板书书写"]

GLOBAL_TO_COARSE = {

    0: "正向讲授",

    2: "正向讲授",

    1: "侧向讲授",

    6: "侧向讲授",

    7: "侧向讲授",

    8: "侧向讲授",

    9: "侧向讲授",

    3: "板书书写",

    4: "板书书写",

    5: "板书书写",

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

    m = re.match(r"^(\d+)[_\-](\d+)$", stem)

    if m:

        return f"{int(m.group(1))}/{int(m.group(2))}"

    if stem.isdigit():

        return str(int(stem))

    return norm_video_id(stem)

def parse_mmss_to_sec(s: str) -> float:

    

       

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

        if re.fullmatch(r"\d+", left) and re.fullmatch(r"\d{1,2}", right):

            mm = int(left)

            ss = int(right)

            if 0 <= ss < 60:

                return float(mm * 60 + ss)

    if re.fullmatch(r"\d{3,4}", s):

        mm = int(s[:-2])

        ss = int(s[-2:])

        if 0 <= ss < 60:

            return float(mm * 60 + ss)

    try:

        return float(s)

    except Exception:

        return np.nan

def normalize_expert_label(label: str) -> str:

    s = str(label).strip()

    s = s.replace(" ", "").replace("　", "")

    if not s:

        return "未知"

    if s in {"正", "正向", "正向讲授", "正向讲解", "正向讲授/讲解", "正面讲授", "正面讲解"}:

        return "正向讲授"

    if s in {

        "侧", "侧向", "侧身", "侧面", "侧立",

        "侧向讲授", "侧向讲解", "侧身讲解", "侧身讲授",

        "侧立讲解", "侧立讲授", "侧面讲授", "侧面讲解"

    }:

        return "侧向讲授"

    if s in {"板", "板书书写", "板书", "书写", "写板书", "板书讲解"}:

        return "板书书写"

                      

    if s in {"互动提问", "提问", "前伸递手提问", "学生互动", "互动"}:

        return "侧向讲授"

    if "正向" in s or "正面" in s:

        return "正向讲授"

    if "侧向" in s or "侧身" in s or "侧立" in s or "侧面" in s:

        return "侧向讲授"

    if "板书" in s or "书写" in s:

        return "板书书写"

    if "提问" in s or "互动" in s:

        return "侧向讲授"

    return "未知"

                                                              

           

                                                              

def parse_expert_txt(txt_path: Path, expert_id: str, fill_small_gaps_sec: float = 1.0) -> pd.DataFrame:

    video_id = infer_video_id_from_txt(txt_path)

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    rows = []

    for line_idx, line in enumerate(lines, start=1):

        raw = line.strip()

        if not raw:

            continue

        m = re.match(r"^\s*([0-9:.]+)\s*[-—~－]\s*([0-9:.]+)\s*(.+?)\s*$", raw)

        if not m:

            print(f"[WARN] 无法解析专家行: {expert_id}/{txt_path.name}:{line_idx}: {raw}")

            continue

        start_s = parse_mmss_to_sec(m.group(1))

        end_s = parse_mmss_to_sec(m.group(2))

        label_raw = m.group(3).strip()

        label = normalize_expert_label(label_raw)

        if pd.isna(start_s) or pd.isna(end_s) or end_s <= start_s:

            print(f"[WARN] 时间异常: {expert_id}/{txt_path.name}:{line_idx}: {raw}")

            continue

        rows.append({

            "expert_id": expert_id,

            "video_id": norm_video_id(video_id),

            "source_file": str(txt_path),

            "line_idx": int(line_idx),

            "expert_start_sec": float(start_s),

            "expert_end_sec": float(end_s),

            "expert_label_raw": label_raw,

            "expert_label": label,

            "raw_line": raw,

        })

    df = pd.DataFrame(rows)

    if df.empty:

        return df

    df = df.sort_values(["expert_start_sec", "expert_end_sec"]).reset_index(drop=True)

    if fill_small_gaps_sec and fill_small_gaps_sec > 0 and len(df) > 1:

        starts = df["expert_start_sec"].to_numpy()

        ends = df["expert_end_sec"].to_numpy().copy()

        for i in range(len(df) - 1):

            gap = starts[i + 1] - ends[i]

            if 0 < gap <= float(fill_small_gaps_sec):

                ends[i] = starts[i + 1]

        df["expert_end_sec"] = ends

    return df

def load_all_experts(doc_root: Path, expert_names: list[str], global_subdir: str, fill_small_gaps_sec: float):

    dfs = []

    summary = []

    for expert_id in expert_names:

        expert_dir = doc_root / expert_id / global_subdir

        if not expert_dir.exists():

            summary.append({"expert_id": expert_id, "expert_dir": str(expert_dir), "n_txt": 0, "status": "missing_dir"})

            print(f"[WARN] 专家目录不存在: {expert_dir}")

            continue

        txt_files = sorted(expert_dir.glob("*.txt"))

        summary.append({"expert_id": expert_id, "expert_dir": str(expert_dir), "n_txt": len(txt_files), "status": "ok"})

        for p in txt_files:

            df = parse_expert_txt(p, expert_id=expert_id, fill_small_gaps_sec=fill_small_gaps_sec)

            if not df.empty:

                dfs.append(df)

    if not dfs:

        raise RuntimeError("没有读取到任何专家标注。")

    return pd.concat(dfs, axis=0, ignore_index=True), pd.DataFrame(summary)

                                                              

                       

                                                              

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

def select_model_video(model_df: pd.DataFrame, video_id: str) -> pd.DataFrame:

    aliases = video_aliases(video_id)

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

    return pd.DataFrame({

        "grid_start_sec": start,

        "grid_end_sec": end,

        "grid_mid_sec": (start + end) / 2.0,

        "duration_sec": np.maximum(0.0, end - start),

        "is_teacher_frame": df["is_teacher_frame"].astype(int).to_numpy(),

    })

def label_at_time_from_segments(seg_df: pd.DataFrame, t: float):

    hit = seg_df[(seg_df["start_sec"] <= t) & (seg_df["end_sec"] > t)]

    if hit.empty:

        return -1, "未匹配"

    hit = hit.copy()

    hit["dur"] = hit["end_sec"] - hit["start_sec"]

    row = hit.sort_values("dur").iloc[0]

    return int(row["global_state"]), str(row["model_label"])

def label_at_time_from_expert(ex_df: pd.DataFrame, t: float):

    hit = ex_df[(ex_df["expert_start_sec"] <= t) & (ex_df["expert_end_sec"] > t)]

    if hit.empty:

        return "未标注"

    return str(hit.iloc[0]["expert_label"])

                                                              

             

                                                              

def build_teacher_grid_detail(expert_df: pd.DataFrame, model_df: pd.DataFrame, t2s_root: Path, time_round: int):

    rows = []

    skipped_sec = 0.0

    for (expert_id, video_id), ex_g in expert_df.groupby(["expert_id", "video_id"]):

        marker_path = find_marker_csv(t2s_root, video_id)

        if marker_path is None:

            print(f"[WARN] 找不到 marker，跳过: {expert_id} {video_id}")

            continue

        mod_g = select_model_video(model_df, video_id)

        grid = load_marker_intervals(marker_path)

        ex_min = float(ex_g["expert_start_sec"].min())

        ex_max = float(ex_g["expert_end_sec"].max())

        grid = grid[(grid["grid_mid_sec"] >= ex_min) & (grid["grid_mid_sec"] < ex_max)].copy()

        for _, gr in grid.iterrows():

            dur = float(gr["duration_sec"])

            if dur <= 0:

                continue

            if int(gr["is_teacher_frame"]) != 1:

                skipped_sec += dur

                continue

            tmid = float(gr["grid_mid_sec"])

            exp_lab = label_at_time_from_expert(ex_g, tmid)

            if exp_lab not in CLASSES:

                skipped_sec += dur

                continue

            gs, model_lab = label_at_time_from_segments(mod_g, tmid)

            if gs < 0 or model_lab not in CLASSES:

                skipped_sec += dur

                continue

            rows.append({

                "expert_id": expert_id,

                "video_id": video_id,

                "start_sec": float(gr["grid_start_sec"]),

                "end_sec": float(gr["grid_end_sec"]),

                "mid_sec": tmid,

                "time_key": f"{video_id}__{round(tmid, time_round)}",

                "duration_sec": dur,

                "expert_label": exp_lab,

                "global_state": gs,

                "model_label": model_lab,

                "match": int(exp_lab == model_lab),

            })

    detail = pd.DataFrame(rows)

    if detail.empty:

        raise RuntimeError("没有有效评价网格，请检查专家标注、marker 和模型 segment。")

    return detail, skipped_sec

                                                              

         

                                                              

def confusion_from_rows(df: pd.DataFrame, true_col: str, pred_col: str, dur_col: str = "duration_sec") -> pd.DataFrame:

    conf = pd.DataFrame(0.0, index=CLASSES, columns=CLASSES)

    for _, r in df.iterrows():

        t = str(r[true_col])

        p = str(r[pred_col])

        if t in CLASSES and p in CLASSES:

            conf.loc[t, p] += float(r[dur_col])

    return conf

def metrics_from_confusion(conf: pd.DataFrame):

    total = float(conf.to_numpy().sum())

    correct = sum(float(conf.loc[c, c]) for c in CLASSES)

    accuracy = correct / total if total > 0 else np.nan

    rows = []

    f1s = []

    weights = []

    for c in CLASSES:

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

    return {

        "total_sec": total,

        "correct_sec": correct,

        "accuracy": accuracy,

        "macro_f1": float(np.mean(f1s)) if f1s else np.nan,

        "weighted_f1": float(np.average(f1s, weights=weights)) if f1s and sum(weights) > 0 else np.nan,

        "per_class": pd.DataFrame(rows),

    }

def compute_pairwise_kappa(detail: pd.DataFrame):

    rows = []

    experts = sorted(detail["expert_id"].unique().tolist())

    for a, b in combinations(experts, 2):

        sub = detail[detail["expert_id"].isin([a, b])].copy()

        piv = sub.pivot_table(index=["video_id", "time_key"], columns="expert_id", values="expert_label", aggfunc="first").reset_index()

        if a not in piv.columns or b not in piv.columns:

            continue

        dur = sub.groupby(["video_id", "time_key"])["duration_sec"].first().reset_index()

        piv = piv.merge(dur, on=["video_id", "time_key"], how="left")

        piv = piv[piv[a].isin(CLASSES) & piv[b].isin(CLASSES)].copy()

        total = float(piv["duration_sec"].sum())

        agree = float(piv.loc[piv[a].eq(piv[b]), "duration_sec"].sum())

        po = agree / total if total > 0 else np.nan

        pe = 0.0

        for c in CLASSES:

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

def compute_fleiss_kappa(detail: pd.DataFrame):

    experts = sorted(detail["expert_id"].unique().tolist())

    n_raters = len(experts)

    piv = detail.pivot_table(index=["video_id", "time_key"], columns="expert_id", values="expert_label", aggfunc="first").reset_index()

    for e in experts:

        if e not in piv.columns:

            return np.nan

                      

    for e in experts:

        piv = piv[piv[e].isin(CLASSES)].copy()

    dur = detail.groupby(["video_id", "time_key"])["duration_sec"].first().reset_index()

    piv = piv.merge(dur, on=["video_id", "time_key"], how="left")

    if piv.empty:

        return np.nan

    weights = piv["duration_sec"].astype(float).to_numpy()

    total_sec = float(weights.sum())

    item_agree = []

    vote_weight = {c: 0.0 for c in CLASSES}

    total_vote_weight = 0.0

    for _, row in piv.iterrows():

        counts = {c: 0 for c in CLASSES}

        for e in experts:

            counts[str(row[e])] += 1

        n = n_raters

        Pi = sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))

        item_agree.append(Pi)

        dur_i = float(row["duration_sec"])

        for c in CLASSES:

            vote_weight[c] += counts[c] * dur_i

        total_vote_weight += n * dur_i

    Pbar = float(np.average(np.asarray(item_agree), weights=weights)) if total_sec > 0 else np.nan

    priors = {c: vote_weight[c] / total_vote_weight if total_vote_weight > 0 else 0.0 for c in CLASSES}

    Pe = sum(v * v for v in priors.values())

    return (Pbar - Pe) / (1.0 - Pe) if (1.0 - Pe) > 1e-12 else np.nan

def build_consensus(detail: pd.DataFrame, min_experts: int):

    rows = []

    for (video_id, time_key), g in detail.groupby(["video_id", "time_key"], sort=True):

        duration = float(g["duration_sec"].iloc[0])

        model_label = str(g["model_label"].iloc[0])

        global_state = int(g["global_state"].iloc[0])

        expert_labels = {}

        for _, r in g.iterrows():

            eid = str(r["expert_id"])

            lab = str(r["expert_label"])

            if lab in CLASSES and eid not in expert_labels:

                expert_labels[eid] = lab

        n_exp = len(expert_labels)

        votes = {c: 0 for c in CLASSES}

        for lab in expert_labels.values():

            votes[lab] += 1

        sorted_votes = sorted(votes.items(), key=lambda x: (-x[1], x[0]))

        top_label, top_n = sorted_votes[0]

        second_n = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

        if n_exp >= min_experts and top_n > second_n and top_n >= 2:

            consensus_label = top_label

            status = "consensus"

        elif n_exp >= min_experts:

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

        for c in CLASSES:

            row[f"votes_{c}"] = votes[c]

        rows.append(row)

    return pd.DataFrame(rows)

def fmt_pct(x):

    return f"{100.0 * float(x):.1f}%" if pd.notna(x) else ""

def fmt3(x):

    return f"{float(x):.3f}" if pd.notna(x) else ""

def dataframe_to_markdown(df: pd.DataFrame) -> str:

    headers = list(df.columns)

    lines = []

    lines.append("| " + " | ".join(headers) + " |")

    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():

        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    return "\n".join(lines) + "\n"

def dataframe_to_tex(df: pd.DataFrame) -> str:

                              

    lines = []

    lines.append(r"\begin{tabular}{ll}")

    lines.append(r"\toprule")

    lines.append(r"评价项 & 指标 \\")

    lines.append(r"\midrule")

    for _, row in df.iterrows():

        a = str(row["评价项"])

        b = str(row["指标"]).replace("%", r"\%")

        lines.append(f"{a} & {b} \\\\")

    lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")

    return "\n".join(lines) + "\n"

                                                              

        

                                                              

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--doc-root", type=str, default=str(DOC_ROOT_DEFAULT))

    ap.add_argument("--expert-names", type=str, default=EXPERT_NAMES_DEFAULT)

    ap.add_argument("--global-subdir", type=str, default="global")

    ap.add_argument("--model-segments", type=str, default=str(MODEL_SEGMENTS_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--fill-small-gaps-sec", type=float, default=1.0)

    ap.add_argument("--min-experts", type=int, default=2)

    ap.add_argument("--time-round", type=int, default=3)

    return ap.parse_args()

def main():

    args = parse_args()

    doc_root = Path(args.doc_root)

    expert_names = [x.strip() for x in str(args.expert_names).split(",") if x.strip()]

    model_segments = Path(args.model_segments)

    t2s_root = Path(args.t2s_root)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)

    print("One-click final expert validation table")

    print(f"doc_root       = {doc_root}")

    print(f"experts        = {expert_names}")

    print(f"model_segments = {model_segments}")

    print(f"t2s_root       = {t2s_root}")

    print(f"out_dir        = {out_dir}")

    print("=" * 100)

    expert_df, file_summary = load_all_experts(

        doc_root=doc_root,

        expert_names=expert_names,

        global_subdir=str(args.global_subdir),

        fill_small_gaps_sec=float(args.fill_small_gaps_sec),

    )

    model_df = load_model_segments(model_segments)

    detail, skipped_sec = build_teacher_grid_detail(

        expert_df=expert_df,

        model_df=model_df,

        t2s_root=t2s_root,

        time_round=int(args.time_round),

    )

            

    pooled_conf = confusion_from_rows(detail, true_col="expert_label", pred_col="model_label")

    pooled = metrics_from_confusion(pooled_conf)

           

    pairwise = compute_pairwise_kappa(detail)

    fleiss_k = compute_fleiss_kappa(detail)

    pair_min = float(pairwise["cohen_kappa"].min()) if not pairwise.empty else np.nan

    pair_max = float(pairwise["cohen_kappa"].max()) if not pairwise.empty else np.nan

            

    consensus_all = build_consensus(detail, min_experts=int(args.min_experts))

    consensus_valid = consensus_all[consensus_all["consensus_status"].eq("consensus")].copy()

    disputed_sec = float(consensus_all.loc[consensus_all["consensus_status"].eq("disputed"), "duration_sec"].sum())

    consensus_conf = confusion_from_rows(consensus_valid, true_col="consensus_label", pred_col="model_label")

    consensus = metrics_from_confusion(consensus_conf)

           

    f1_map = dict(zip(consensus["per_class"]["class"], consensus["per_class"]["f1"]))

    class_f1 = "/".join(fmt3(f1_map.get(c, np.nan)) for c in CLASSES)

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

    table_note = (

        f"注：κ 为三专家在共同有效教师区间上的 Fleiss' κ；两两 Cohen's κ 范围为 "

        f"{fmt3(pair_min)}–{fmt3(pair_max)}。共识标签由多数投票得到，disputed 区间"

        f"（{disputed_sec:.2f}s）与非教师/SKIP 区间不纳入共识评价。类别 F1 按"

        f"“正向讲授/侧向讲授/板书书写”顺序为 {class_f1}。保守合并评价指直接合并全部专家标注后的模型结果。"

    )

                       

    csv_path = out_dir / "final_expert_validation_3row_table.csv"

    md_path = out_dir / "final_expert_validation_3row_table.md"

    tex_path = out_dir / "final_expert_validation_3row_table.tex"

    final_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_text = dataframe_to_markdown(final_table) + "\n" + table_note + "\n"

    md_path.write_text(md_text, encoding="utf-8")

    tex_text = dataframe_to_tex(final_table) + "\n% " + table_note.replace("\n", " ") + "\n"

    tex_path.write_text(tex_text, encoding="utf-8")

    print("\nFINAL TABLE")

    print(final_table.to_string(index=False))

    print("\nTABLE NOTE")

    print(table_note)

    print("\nOUTPUTS")

    print(f"  {csv_path}")

    print(f"  {md_path}")

    print(f"  {tex_path}")

    print("=" * 100)

if __name__ == "__main__":

    main()
