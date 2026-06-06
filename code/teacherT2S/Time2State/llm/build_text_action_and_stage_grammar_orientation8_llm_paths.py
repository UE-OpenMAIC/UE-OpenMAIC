

                       

   

from __future__ import annotations

import argparse

import json

import re

from pathlib import Path

from collections import Counter

import numpy as np

import pandas as pd

ASR_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\surprisal_batch_output")

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch_orientation8")

LAYER2_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_prototype_alignment_X1_only_orientation8"

LAYER2_SEGMENT_CSV_DEFAULT = LAYER2_DIR_DEFAULT / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

ACTION_OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\text_action_grammar_orientation8")

STAGE_OUT_DIR_DEFAULT = Path(r"D:\code\teacherT2S\Time2State\llm\course_stage_grammar_orientation8")

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

                                                              

      

                                                              

def ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)

def norm_video_id(x) -> str:

    s = str(x).strip().replace("\\", "/")

    if s.endswith(".0"):

        s = s[:-2]

    s = re.sub(r"/+", "/", s).strip("/")

    return s

def tail_video_id(x) -> str:

    s = norm_video_id(x)

    return s.split("/")[-1] if s else s

def video_aliases(video_id: str) -> set[str]:

    vid = norm_video_id(video_id)

    tail = tail_video_id(vid)

    aliases = {vid, tail}

    if tail.isdigit():

        aliases.add(str(int(tail)))

        aliases.add(f"{int(tail):02d}")

    return {x for x in aliases if x}

def parse_time_to_sec(s) -> float:

    if s is None:

        return np.nan

    s = str(s).strip().replace("：", ":")

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

def join_top_texts(texts, max_items=8, max_len=800) -> str:

    clean = []

    for t in texts:

        t = str(t).strip()

        if t:

            clean.append(t)

    out = " | ".join(clean[:max_items])

    return out if len(out) <= max_len else out[:max_len] + "..."

def safe_float(x, default=0.0):

    try:

        if pd.isna(x):

            return float(default)

        return float(x)

    except Exception:

        return float(default)

def safe_int(x, default=-1):

    try:

        if pd.isna(x):

            return int(default)

        return int(float(x))

    except Exception:

        return int(default)

                                                              

           

                                                              

def find_asr_txt(asr_root: Path, video_id: str) -> Path | None:

    aliases = video_aliases(video_id)

    candidates = []

    for a in aliases:

        candidates += [

            asr_root / a / "asr_segments_editable.txt",

            asr_root / a / "asr_segments.txt",

            asr_root / a / "peak_texts.txt",

            asr_root / a / f"{a}.txt",

        ]

    for p in candidates:

        if p.exists():

            return p

    if asr_root.exists():

        editable_hits = []

        any_hits = []

        for p in asr_root.rglob("*.txt"):

            if aliases & video_aliases(p.parent.name):

                any_hits.append(p)

                if p.name == "asr_segments_editable.txt":

                    editable_hits.append(p)

        if editable_hits:

            return sorted(editable_hits)[0]

        if any_hits:

            return sorted(any_hits)[0]

    return None

def parse_stage_line(line: str):

    line = str(line).strip()

    if not line:

        return None

    m = re.match(r"^(.+?)[（(]\s*([^~\-—－]+)\s*[~\-—－]\s*([^)）]+)\s*[)）]\s*$", line)

    if not m:

        return None

    stage = m.group(1).strip()

    start_s = parse_time_to_sec(m.group(2))

    end_s = parse_time_to_sec(m.group(3))

    if pd.isna(start_s) or pd.isna(end_s):

        return None

    return {"stage": stage, "stage_start_sec": float(start_s), "stage_end_sec": float(end_s)}

def parse_asr_txt(txt_path: Path, video_id: str):

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

    if df.empty:

        return df, pd.DataFrame(stages)

    def assign_stage(mid):

        for s in stages:

            if s["stage_start_sec"] <= mid <= s["stage_end_sec"]:

                return s["stage"]

        return "未知阶段"

    def assign_stage_start(mid):

        for s in stages:

            if s["stage_start_sec"] <= mid <= s["stage_end_sec"]:

                return s["stage_start_sec"]

        return np.nan

    def assign_stage_end(mid):

        for s in stages:

            if s["stage_start_sec"] <= mid <= s["stage_end_sec"]:

                return s["stage_end_sec"]

        return np.nan

    df["stage"] = df["text_mid_sec"].map(assign_stage)

    df["stage_start_sec"] = df["text_mid_sec"].map(assign_stage_start)

    df["stage_end_sec"] = df["text_mid_sec"].map(assign_stage_end)

    stage_df = pd.DataFrame(stages)

    if not stage_df.empty:

        stage_df["video_id"] = norm_video_id(video_id)

        stage_df["stage_duration_sec"] = (stage_df["stage_end_sec"] - stage_df["stage_start_sec"]).clip(lower=0)

    return df, stage_df

                                                              

                 

                                                              

def find_final_meta_csv(t2s_root: Path, video_id: str) -> Path | None:

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

def load_layer2_segments(layer2_segment_csv: Path) -> pd.DataFrame:

    if not layer2_segment_csv.exists():

        raise FileNotFoundError(f"找不到跨视频 layer2 segment CSV: {layer2_segment_csv}")

    df = pd.read_csv(layer2_segment_csv)

    df["video_id"] = df["video_id"].astype(str).map(norm_video_id)

    required = ["video_id", "local_meta_state", "global_state", "start_sec", "end_sec"]

    missing = [c for c in required if c not in df.columns]

    if missing:

        raise ValueError(f"layer2 segment csv 缺少列: {missing}")

    df["local_meta_state"] = pd.to_numeric(df["local_meta_state"], errors="coerce").fillna(-1).astype(int)

    df["global_state"] = pd.to_numeric(df["global_state"], errors="coerce").fillna(-1).astype(int)

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")

    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")

    df["duration_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=0)

    return df

                                                              

           

                                                              

def align_text_to_layer2(asr_df: pd.DataFrame, seg_df_video: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if asr_df.empty or seg_df_video.empty:

        return pd.DataFrame(rows)

    seg_df_video = seg_df_video.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    for _, t in asr_df.iterrows():

        t0 = float(t["text_start_sec"])

        t1 = float(t["text_end_sec"])

        overlaps = []

        for _, s in seg_df_video.iterrows():

            ov = overlap_seconds(t0, t1, s["start_sec"], s["end_sec"])

            if ov > 0:

                overlaps.append((ov, s))

        base = t.to_dict()

        if not overlaps:

            base.update({

                "global_state": -1,

                "local_meta_state": -1,

                "prototype_id": "",

                "matched_seg_start_sec": np.nan,

                "matched_seg_end_sec": np.nan,

                "overlap_sec": 0.0,

                "overlap_ratio": 0.0,

                "all_overlapped_states": "{}",

            })

            rows.append(base)

            continue

        overlaps.sort(key=lambda x: x[0], reverse=True)

        best_ov, best_s = overlaps[0]

        dur = max(1e-6, t1 - t0)

        state_counter = Counter()

        for ov, s in overlaps:

            key = f"G{int(s['global_state'])}_L{int(s['local_meta_state'])}"

            state_counter[key] += float(ov)

        base.update({

            "global_state": int(best_s["global_state"]),

            "local_meta_state": int(best_s["local_meta_state"]),

            "prototype_id": str(best_s.get("prototype_id", "")),

            "matched_seg_start_sec": float(best_s["start_sec"]),

            "matched_seg_end_sec": float(best_s["end_sec"]),

            "overlap_sec": float(best_ov),

            "overlap_ratio": float(best_ov / dur),

            "all_overlapped_states": json.dumps(dict(state_counter), ensure_ascii=False),

        })

        rows.append(base)

    return pd.DataFrame(rows)

                                                              

       

                                                              

def build_global_local_text_summary(aligned_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if aligned_df.empty:

        return pd.DataFrame(rows)

    for (gs, vid, ls), g in aligned_df.groupby(["global_state", "video_id", "local_meta_state"]):

        texts = g["text"].astype(str).tolist()

        stages = g["stage"].astype(str).tolist()

        rows.append({

            "global_state": int(gs),

            "video_id": norm_video_id(vid),

            "local_meta_state": int(ls),

            "local_class_key": f"{norm_video_id(vid)}__local{int(ls)}",

            "n_text_segments": int(len(g)),

            "total_text_duration_sec": float(g["text_duration_sec"].sum()),

            "top_stages": json.dumps(Counter(stages).most_common(8), ensure_ascii=False),

            "example_texts": join_top_texts(texts, max_items=10, max_len=800),

        })

    return pd.DataFrame(rows).sort_values(

        ["global_state", "n_text_segments", "total_text_duration_sec"],

        ascending=[True, False, False],

    ).reset_index(drop=True)

def build_global_summary(local_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if local_df.empty:

        return pd.DataFrame(rows)

    for gs, g in local_df.groupby("global_state"):

        vids = g["video_id"].astype(str).map(norm_video_id).unique().tolist()

        rows.append({

            "global_state": int(gs),

            "n_local_classes": int(len(g)),

            "support_videos": int(len(vids)),

            "video_ids": ",".join(sorted(vids, key=lambda x: (tail_video_id(x), x))),

            "local_classes": ",".join(g["local_class_key"].astype(str).tolist()[:100]),

            "example_texts": join_top_texts(g["example_texts"].astype(str).tolist(), max_items=8, max_len=1200),

        })

    return pd.DataFrame(rows).sort_values(["support_videos", "n_local_classes"], ascending=[False, False]).reset_index(drop=True)

def summarize_transition_raw(raw_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:

    rows = []

    if raw_df.empty:

        return pd.DataFrame(rows)

    for key, g in raw_df.groupby(group_cols):

        if not isinstance(key, tuple):

            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}

        row.update({

            "count": int(len(g)),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()) if "video_id" in g.columns else np.nan,

            "example_texts": join_top_texts(g["nearby_texts"].astype(str).tolist(), max_items=8, max_len=1000)

            if "nearby_texts" in g.columns else "",

        })

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["count", "support_videos"], ascending=[False, False]).reset_index(drop=True)

def build_segment_transition_tables(layer2_df: pd.DataFrame, aligned_df: pd.DataFrame):

    global_rows, local_rows, stage_rows = [], [], []

    text_by_video = {vid: g.sort_values("text_start_sec").reset_index(drop=True) for vid, g in aligned_df.groupby("video_id")}

    def nearby_texts(vid, start, end, max_items=5):

        if vid not in text_by_video:

            return ""

        tdf = text_by_video[vid]

        mask = (tdf["text_end_sec"] >= start) & (tdf["text_start_sec"] <= end)

        return join_top_texts(tdf.loc[mask, "text"].astype(str).tolist(), max_items=max_items, max_len=500)

    def dominant_stage(vid, start, end):

        if vid not in text_by_video:

            return "未知阶段"

        tdf = text_by_video[vid]

        mask = (tdf["text_end_sec"] >= start) & (tdf["text_start_sec"] <= end)

        if not mask.any():

            return "未知阶段"

        return Counter(tdf.loc[mask, "stage"].astype(str).tolist()).most_common(1)[0][0]

    for vid, g in layer2_df.groupby("video_id"):

        vid = norm_video_id(vid)

        g = g.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

        for i in range(len(g) - 1):

            a, b = g.iloc[i], g.iloc[i + 1]

            fg, tg = int(a["global_state"]), int(b["global_state"])

            fl, tl = int(a["local_meta_state"]), int(b["local_meta_state"])

            if fg < 0 or tg < 0:

                continue

            start, end = float(a["start_sec"]), float(b["end_sec"])

            stg = dominant_stage(vid, start, end)

            txt = nearby_texts(vid, start, end)

            global_rows.append({

                "video_id": vid, "stage": stg,

                "from_global_state": fg, "to_global_state": tg,

                "transition_key": f"G{fg}->G{tg}",

                "from_start_sec": float(a["start_sec"]), "from_end_sec": float(a["end_sec"]),

                "to_start_sec": float(b["start_sec"]), "to_end_sec": float(b["end_sec"]),

                "nearby_texts": txt,

            })

            local_rows.append({

                "video_id": vid, "stage": stg,

                "from_global_state": fg, "to_global_state": tg,

                "from_local_meta_state": fl, "to_local_meta_state": tl,

                "transition_key": f"{vid}__L{fl}->L{tl}",

                "global_transition_key": f"G{fg}->G{tg}",

                "from_start_sec": float(a["start_sec"]), "from_end_sec": float(a["end_sec"]),

                "to_start_sec": float(b["start_sec"]), "to_end_sec": float(b["end_sec"]),

                "nearby_texts": txt,

            })

            stage_rows.append({

                "stage": stg,

                "from_global_state": fg,

                "to_global_state": tg,

                "transition_key": f"{stg}::G{fg}->G{tg}",

                "video_id": vid,

                "nearby_texts": txt,

            })

    global_raw = pd.DataFrame(global_rows)

    local_raw = pd.DataFrame(local_rows)

    stage_raw = pd.DataFrame(stage_rows)

    return (

        global_raw,

        summarize_transition_raw(global_raw, ["from_global_state", "to_global_state", "transition_key"]),

        local_raw,

        summarize_transition_raw(local_raw, ["video_id", "from_global_state", "to_global_state", "from_local_meta_state", "to_local_meta_state", "transition_key", "global_transition_key"]),

        stage_raw,

        summarize_transition_raw(stage_raw, ["stage", "from_global_state", "to_global_state", "transition_key"]),

    )

def write_action_rag_corpus(aligned_df: pd.DataFrame, out_path: Path):

    with out_path.open("w", encoding="utf-8") as f:

        for _, r in aligned_df.iterrows():

            item = {

                "doc_type": "sentence_action",

                "video_id": norm_video_id(r["video_id"]),

                "stage": str(r.get("stage", "")),

                "text_start_sec": safe_float(r["text_start_sec"]),

                "text_end_sec": safe_float(r["text_end_sec"]),

                "text": str(r["text"]),

                "global_state": safe_int(r.get("global_state", -1)),

                "local_meta_state": safe_int(r.get("local_meta_state", -1)),

                "local_class_key": f"{norm_video_id(r['video_id'])}__local{safe_int(r.get('local_meta_state', -1))}",

                "prototype_id": str(r.get("prototype_id", "")),

                "overlap_ratio": safe_float(r.get("overlap_ratio", 0.0)),

            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def write_action_outputs(action_out: Path, aligned_df: pd.DataFrame, layer2_df: pd.DataFrame):

    ensure_dir(action_out)

    aligned_df.to_csv(action_out / "sentence_text_action_alignment.csv", index=False, encoding="utf-8-sig")

    local_df = build_global_local_text_summary(aligned_df)

    local_df.to_csv(action_out / "local_state_text_examples.csv", index=False, encoding="utf-8-sig")

    global_df = build_global_summary(local_df)

    global_df.to_csv(action_out / "global_to_local_state_text_summary.csv", index=False, encoding="utf-8-sig")

    global_raw, global_trans, local_raw, local_trans, stage_raw, stage_trans = build_segment_transition_tables(layer2_df, aligned_df)

    global_raw.to_csv(action_out / "global_transition_raw_examples.csv", index=False, encoding="utf-8-sig")

    global_trans.to_csv(action_out / "global_transition_grammar.csv", index=False, encoding="utf-8-sig")

    local_raw.to_csv(action_out / "local_transition_raw_examples.csv", index=False, encoding="utf-8-sig")

    local_trans.to_csv(action_out / "local_transition_grammar.csv", index=False, encoding="utf-8-sig")

    stage_raw.to_csv(action_out / "stage_conditioned_transition_raw_examples.csv", index=False, encoding="utf-8-sig")

    stage_trans.to_csv(action_out / "stage_conditioned_transition_grammar.csv", index=False, encoding="utf-8-sig")

    write_action_rag_corpus(aligned_df, action_out / "action_primitive_rag_corpus.jsonl")

                                                              

       

                                                              

def infer_stage_role(stage: str) -> str:

    s = str(stage)

    if "导入" in s:

        return "引题、激活旧知、提出核心问题、建立学习目标"

    if "新课" in s or "讲授" in s:

        return "围绕核心问题推进讲解、提问、文本细读、例证分析"

    if "反思" in s or "总结" in s or "升华" in s:

        return "总结规律、价值升华、迁移反思、形成结论"

    if "结束" in s or "收束" in s:

        return "课堂收束、课后任务、简短告别"

    return "未知阶段功能，需要结合上下文判断"

def build_stage_block_corpus(asr_all: pd.DataFrame, stage_all: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if asr_all.empty:

        return pd.DataFrame(rows)

    for (vid, stage), g in asr_all.groupby(["video_id", "stage"]):

        g = g.sort_values("text_start_sec")

        full_text = " ".join(g["text"].astype(str).tolist())

        rows.append({

            "video_id": norm_video_id(vid),

            "stage": str(stage),

            "stage_role": infer_stage_role(str(stage)),

            "stage_start_sec": safe_float(g["text_start_sec"].min()),

            "stage_end_sec": safe_float(g["text_end_sec"].max()),

            "stage_duration_sec": safe_float(g["text_end_sec"].max() - g["text_start_sec"].min()),

            "n_sentences": int(len(g)),

            "opening_texts": join_top_texts(g["text"].astype(str).tolist(), max_items=5, max_len=500),

            "closing_texts": join_top_texts(g["text"].astype(str).tolist()[-5:], max_items=5, max_len=500),

            "full_stage_text": full_text[:5000],

        })

    return pd.DataFrame(rows).sort_values(["video_id", "stage_start_sec"]).reset_index(drop=True)

def build_stage_sequence_summary(stage_block_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if stage_block_df.empty:

        return pd.DataFrame(rows)

    for vid, g in stage_block_df.groupby("video_id"):

        g = g.sort_values("stage_start_sec")

        total = max(1e-6, float(g["stage_duration_sec"].sum()))

        seq = []

        for _, r in g.iterrows():

            seq.append(f"{r['stage']}({safe_float(r['stage_duration_sec']):.1f}s,{safe_float(r['stage_duration_sec'])/total:.2%})")

        rows.append({

            "video_id": norm_video_id(vid),

            "stage_sequence": " -> ".join(seq),

            "n_stages": int(len(g)),

            "total_duration_sec": float(total),

            "stage_names": " -> ".join(g["stage"].astype(str).tolist()),

        })

    return pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)

def build_stage_transition_summary(stage_block_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    if stage_block_df.empty:

        return pd.DataFrame(rows)

    raw = []

    for vid, g in stage_block_df.groupby("video_id"):

        g = g.sort_values("stage_start_sec").reset_index(drop=True)

        for i in range(len(g) - 1):

            a, b = g.iloc[i], g.iloc[i + 1]

            raw.append({

                "video_id": norm_video_id(vid),

                "from_stage": str(a["stage"]),

                "to_stage": str(b["stage"]),

                "transition_key": f"{a['stage']}->{b['stage']}",

                "nearby_texts": join_top_texts([str(a["closing_texts"]), str(b["opening_texts"])], max_items=2, max_len=700),

            })

    raw_df = pd.DataFrame(raw)

    if raw_df.empty:

        return raw_df

    for key, g in raw_df.groupby(["from_stage", "to_stage", "transition_key"]):

        rows.append({

            "from_stage": key[0],

            "to_stage": key[1],

            "transition_key": key[2],

            "count": int(len(g)),

            "support_videos": int(g["video_id"].nunique()),

            "example_texts": join_top_texts(g["nearby_texts"].astype(str).tolist(), max_items=8, max_len=1200),

        })

    return pd.DataFrame(rows).sort_values(["count", "support_videos"], ascending=[False, False]).reset_index(drop=True)

def write_stage_rag_corpus(stage_out: Path, asr_all: pd.DataFrame, stage_block_df: pd.DataFrame, seq_df: pd.DataFrame, trans_df: pd.DataFrame):

    with (stage_out / "stage_arrangement_rag_corpus.jsonl").open("w", encoding="utf-8") as f:

        for _, r in stage_block_df.iterrows():

            item = {

                "doc_type": "stage_block",

                "video_id": norm_video_id(r["video_id"]),

                "stage": str(r["stage"]),

                "stage_role": str(r["stage_role"]),

                "stage_start_sec": safe_float(r["stage_start_sec"]),

                "stage_end_sec": safe_float(r["stage_end_sec"]),

                "stage_duration_sec": safe_float(r["stage_duration_sec"]),

                "n_sentences": safe_int(r["n_sentences"], 0),

                "opening_texts": str(r["opening_texts"]),

                "closing_texts": str(r["closing_texts"]),

                "full_stage_text": str(r["full_stage_text"]),

            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        for _, r in seq_df.iterrows():

            item = {

                "doc_type": "stage_sequence",

                "video_id": norm_video_id(r["video_id"]),

                "stage_sequence": str(r["stage_sequence"]),

                "stage_names": str(r["stage_names"]),

                "total_duration_sec": safe_float(r["total_duration_sec"]),

                "n_stages": safe_int(r["n_stages"], 0),

            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        for _, r in trans_df.iterrows():

            item = {

                "doc_type": "stage_transition",

                "from_stage": str(r["from_stage"]),

                "to_stage": str(r["to_stage"]),

                "transition_key": str(r["transition_key"]),

                "count": safe_int(r["count"], 0),

                "support_videos": safe_int(r["support_videos"], 0),

                "example_texts": str(r["example_texts"]),

            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def write_stage_outputs(stage_out: Path, asr_all: pd.DataFrame, stage_all: pd.DataFrame):

    ensure_dir(stage_out)

    asr_all.to_csv(stage_out / "stage_sentence_corpus.csv", index=False, encoding="utf-8-sig")

    stage_all.to_csv(stage_out / "stage_boundaries_from_asr.csv", index=False, encoding="utf-8-sig")

    stage_block_df = build_stage_block_corpus(asr_all, stage_all)

    stage_block_df.to_csv(stage_out / "stage_block_corpus.csv", index=False, encoding="utf-8-sig")

    seq_df = build_stage_sequence_summary(stage_block_df)

    seq_df.to_csv(stage_out / "stage_sequence_summary.csv", index=False, encoding="utf-8-sig")

    trans_df = build_stage_transition_summary(stage_block_df)

    trans_df.to_csv(stage_out / "stage_transition_summary.csv", index=False, encoding="utf-8-sig")

    write_stage_rag_corpus(stage_out, asr_all, stage_block_df, seq_df, trans_df)

                                                              

     

                                                              

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--asr-root", type=str, default=str(ASR_ROOT_DEFAULT))

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--layer2-segment-csv", type=str, default=str(LAYER2_SEGMENT_CSV_DEFAULT))

    ap.add_argument("--action-out-dir", type=str, default=str(ACTION_OUT_DIR_DEFAULT))

    ap.add_argument("--stage-out-dir", type=str, default=str(STAGE_OUT_DIR_DEFAULT))

    return ap.parse_args()

def main():

    args = parse_args()

    asr_root = Path(args.asr_root)

    t2s_root = Path(args.t2s_root)

    layer2_segment_csv = Path(args.layer2_segment_csv)

    action_out = Path(args.action_out_dir)

    stage_out = Path(args.stage_out_dir)

    ensure_dir(action_out)

    ensure_dir(stage_out)

    print("=" * 100)

    print("Build TWO libraries:")

    print(f"1) action library: {action_out}")

    print(f"2) stage  library: {stage_out}")

    print(f"ASR_ROOT          = {asr_root}")

    print(f"T2S_ROOT          = {t2s_root}")

    print(f"LAYER2_SEGMENT_CSV= {layer2_segment_csv}")

    print("=" * 100)

    layer2_df = load_layer2_segments(layer2_segment_csv)

    all_video_ids = sorted(layer2_df["video_id"].astype(str).map(norm_video_id).unique().tolist(),

                           key=lambda x: (tail_video_id(x), x))

    aligned_tables = []

    asr_tables = []

    stage_tables = []

    manifest_rows = []

    for i, vid in enumerate(all_video_ids, start=1):

        print(f"[{i}/{len(all_video_ids)}] video_id={vid}")

        asr_txt = find_asr_txt(asr_root, vid)

        final_csv = find_final_meta_csv(t2s_root, vid)

        row = {

            "video_id": vid,

            "asr_txt": str(asr_txt) if asr_txt else "",

            "final_meta_csv": str(final_csv) if final_csv else "",

            "included_action": 0,

            "included_stage": 0,

            "skip_reason": "",

        }

        try:

            if asr_txt is None:

                raise FileNotFoundError(f"找不到 ASR txt: {vid}")

            asr_df, stage_df = parse_asr_txt(asr_txt, vid)

            if asr_df.empty:

                raise ValueError(f"ASR txt 解析为空: {asr_txt}")

            asr_tables.append(asr_df)

            if not stage_df.empty:

                stage_tables.append(stage_df)

            row["included_stage"] = 1

            row["n_asr_segments"] = int(len(asr_df))

            row["n_stage_boundaries"] = int(len(stage_df)) if not stage_df.empty else 0

            if final_csv is None:

                raise FileNotFoundError(f"找不到 multiscale_t2s_with_meta.csv: {vid}")

            seg_df_video = layer2_df[layer2_df["video_id"].astype(str).map(norm_video_id).eq(norm_video_id(vid))].copy()

            if seg_df_video.empty:

                raise ValueError(f"layer2 中没有该视频 segments: {vid}")

            aligned = align_text_to_layer2(asr_df, seg_df_video)

            if aligned.empty:

                raise ValueError("文本-state 对齐为空")

            aligned_tables.append(aligned)

            row["included_action"] = 1

            row["n_aligned_segments"] = int(len(aligned))

            row["n_layer2_segments"] = int(len(seg_df_video))

            print(f"  [OK] asr={len(asr_df)}, aligned={len(aligned)}, layer2_segments={len(seg_df_video)}")

        except Exception as e:

            row["skip_reason"] = repr(e)

            print(f"  [SKIP/PARTIAL] {repr(e)}")

        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(action_out / "build_manifest.csv", index=False, encoding="utf-8-sig")

    manifest_df.to_csv(stage_out / "build_manifest.csv", index=False, encoding="utf-8-sig")

    if asr_tables:

        asr_all = pd.concat(asr_tables, axis=0, ignore_index=True)

        stage_all = pd.concat(stage_tables, axis=0, ignore_index=True) if stage_tables else pd.DataFrame()

        write_stage_outputs(stage_out, asr_all, stage_all)

    else:

        raise RuntimeError("没有成功解析任何 ASR，无法生成阶段库。")

    if aligned_tables:

        aligned_df = pd.concat(aligned_tables, axis=0, ignore_index=True)

        write_action_outputs(action_out, aligned_df, layer2_df)

    else:

        print("[WARN] 没有成功生成动作对齐库，只生成了阶段库。")

    print("\n" + "=" * 100)

    print("DONE.")

    print(f"ACTION library: {action_out}")

    print(f"  - sentence_text_action_alignment.csv")

    print(f"  - local_state_text_examples.csv")

    print(f"  - global_to_local_state_text_summary.csv")

    print(f"  - stage_conditioned_transition_grammar.csv")

    print(f"  - action_primitive_rag_corpus.jsonl")

    print(f"STAGE library: {stage_out}")

    print(f"  - stage_sentence_corpus.csv")

    print(f"  - stage_block_corpus.csv")

    print(f"  - stage_sequence_summary.csv")

    print(f"  - stage_transition_summary.csv")

    print(f"  - stage_arrangement_rag_corpus.jsonl")

    print("=" * 100)

if __name__ == "__main__":

    main()
