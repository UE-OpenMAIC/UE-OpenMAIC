                       

   

from __future__ import annotations

import argparse

import json

import math

import re

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.metrics import silhouette_score

                                                              

         

                                                              

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch_orientation8")

VISUAL_CSV_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\pose_csv")

OUT_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_prototype_alignment_X1_only_orientation8"

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

TIME_COL = "time_sec"

SKIP_STATE = -1

ACTION_SMOOTH = 7

                                                              

         

                                                              

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

def safe_name(x) -> str:

    s = norm_video_id(x)

    s = re.sub(r'[\\/:*?"<>|]+', "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s or "item"

def split_segments(seq):

    seq = np.asarray(seq).astype(int)

    if len(seq) == 0:

        return []

    out = []

    s = 0

    for i in range(1, len(seq)):

        if seq[i] != seq[s]:

            out.append((s, i - 1, int(seq[s])))

            s = i

    out.append((s, len(seq) - 1, int(seq[s])))

    return out

def find_case_dirs(root: Path, out_dir: Path) -> list[Path]:

    case_dirs = []

    for p in root.rglob(FINAL_META_CSV):

        case_dir = p.parent

        if out_dir in case_dir.parents or case_dir == out_dir:

            continue

        case_dirs.append(case_dir)

    return sorted(set(case_dirs))

def infer_video_id_from_case_dir(root: Path, case_dir: Path) -> str:

    try:

        return norm_video_id(str(case_dir.relative_to(root)).replace("\\", "/"))

    except Exception:

        return norm_video_id(case_dir.name)

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

                                                              

             

                                                              

def quantize_orientation_8class(series):

    

       

    s = pd.to_numeric(series, errors="coerce").copy()

    s = s.clip(-2.0, 2.0)

    out = []

    for v in s:

        if pd.isna(v):

            out.append(np.nan)

        else:

            cls = int(np.floor((float(v) + 2.0) / 4.0 * 8.0))

            cls = max(0, min(7, cls))

            out.append(float(cls))

    return pd.Series(out, index=series.index)

def build_action_relative_features(visual_df: pd.DataFrame) -> pd.DataFrame:

    

       

    feature_cols = [

        "orientation_score",

        "left_shoulder_x", "left_shoulder_y",

        "left_elbow_x", "left_elbow_y",

        "left_wrist_x", "left_wrist_y",

        "right_shoulder_x", "right_shoulder_y",

        "right_elbow_x", "right_elbow_y",

        "right_wrist_x", "right_wrist_y",

        "center_x", "center_y",

    ]

    missing = [c for c in feature_cols if c not in visual_df.columns]

    if missing:

        raise ValueError(f"动作 CSV 缺少这些列：{missing}")

    X = visual_df[feature_cols].copy()

    cx = X["center_x"].astype(float)

    cy = X["center_y"].astype(float)

    ori8 = quantize_orientation_8class(X["orientation_score"].astype(float))

    out = pd.DataFrame(index=X.index)

    out["orientation_8class"] = ori8

    pts = [

        "left_shoulder", "left_elbow", "left_wrist",

        "right_shoulder", "right_elbow", "right_wrist",

    ]

    for p in pts:

        out[f"{p}_dx"] = X[f"{p}_x"].astype(float) - cx

        out[f"{p}_dy"] = X[f"{p}_y"].astype(float) - cy

    out = out.ffill().bfill().fillna(0.0)

    out = out.rolling(ACTION_SMOOTH, center=True, min_periods=1).median()

    return out

def align_x1_to_final_df(visual_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:

    

       

    action_X = build_action_relative_features(visual_df)

    if len(action_X) == len(final_df):

        return action_X.reset_index(drop=True)

    if TIME_COL not in visual_df.columns or TIME_COL not in final_df.columns:

        raise ValueError(

            f"visual_df 与 final_df 长度不同且缺少 {TIME_COL}: "

            f"visual={len(visual_df)}, final={len(final_df)}"

        )

    left = final_df[[TIME_COL]].copy()

    left["_row_id"] = np.arange(len(left))

    right = visual_df[[TIME_COL]].copy()

    right["_visual_idx"] = np.arange(len(right))

    merged = pd.merge_asof(

        left.sort_values(TIME_COL),

        right.sort_values(TIME_COL),

        on=TIME_COL,

        direction="nearest",

    ).sort_values("_row_id")

    idx = merged["_visual_idx"].ffill().bfill().astype(int).to_numpy()

    return action_X.iloc[idx].reset_index(drop=True)

                                                              

                         

                                                              

def compute_x1_distribution_descriptor(X1_sub: pd.DataFrame, use_quantiles: bool = True) -> dict:

    

       

    X1_sub = X1_sub.astype(float)

    row = {}

    means = X1_sub.mean(axis=0)

    stds = X1_sub.std(axis=0, ddof=0)

    medians = X1_sub.median(axis=0)

    for c in X1_sub.columns:

        row[f"mean_{c}"] = float(means[c])

        row[f"std_{c}"] = float(stds[c])

        row[f"median_{c}"] = float(medians[c])

    if use_quantiles:

        q25 = X1_sub.quantile(0.25, axis=0)

        q75 = X1_sub.quantile(0.75, axis=0)

        for c in X1_sub.columns:

            row[f"q25_{c}"] = float(q25[c])

            row[f"q75_{c}"] = float(q75[c])

            row[f"iqr_{c}"] = float(q75[c] - q25[c])

    return row

def build_case_prototypes_x1_only(

    case_dir: Path,

    root: Path,

    visual_root: Path,

    use_quantiles: bool,

):

    final_csv = case_dir / FINAL_META_CSV

    final_df = pd.read_csv(final_csv)

    video_id = infer_video_id_from_case_dir(root, case_dir)

    visual_csv = find_visual_csv(visual_root, video_id)

    if visual_csv is None:

        raise FileNotFoundError(f"找不到 visual CSV: video_id={video_id}, visual_root={visual_root}")

    visual_df = pd.read_csv(visual_csv)

    X1 = align_x1_to_final_df(visual_df, final_df)

    X1.columns = [f"x1_{c}" for c in X1.columns]

    min_len = min(len(final_df), len(X1))

    final_df = final_df.iloc[:min_len].reset_index(drop=True)

    X1 = X1.iloc[:min_len].reset_index(drop=True)

    if "meta_state" not in final_df.columns:

        raise ValueError(f"{final_csv} 缺少 meta_state")

    meta_seq_full = pd.to_numeric(final_df["meta_state"], errors="coerce").fillna(SKIP_STATE).astype(int)

    if "is_teacher_frame" in final_df.columns:

        teacher_mask = pd.to_numeric(final_df["is_teacher_frame"], errors="coerce").fillna(1).astype(int).eq(1)

        teacher_mask = teacher_mask & meta_seq_full.ge(0)

    else:

        teacher_mask = meta_seq_full.ge(0)

    keep_idx = np.where(teacher_mask.to_numpy())[0]

    if len(keep_idx) == 0:

        raise ValueError("没有可用教师帧")

    final_t = final_df.iloc[keep_idx].reset_index(drop=True)

    X1_t = X1.iloc[keep_idx].reset_index(drop=True)

    meta_seq = meta_seq_full.iloc[keep_idx].to_numpy(dtype=int)

    if TIME_COL in final_t.columns:

        time_arr = pd.to_numeric(final_t[TIME_COL], errors="coerce").to_numpy(dtype=float)

    else:

        time_arr = np.arange(len(final_t), dtype=float)

    local_states = sorted([int(x) for x in pd.unique(pd.Series(meta_seq)) if int(x) >= 0])

    proto_rows = []

    segment_rows = []

    for local_state in local_states:

        idxs = np.where(meta_seq == local_state)[0]

        if len(idxs) == 0:

            continue

        row = {

            "video_id": video_id,

            "local_meta_state": int(local_state),

            "prototype_id": f"{safe_name(video_id)}__local{int(local_state)}",

                                                    

            "n_frames_meta": int(len(idxs)),

            "frame_ratio_in_video_meta": float(len(idxs) / max(1, len(meta_seq))),

        }

        row.update(compute_x1_distribution_descriptor(X1_t.iloc[idxs], use_quantiles=use_quantiles))

        proto_rows.append(row)

                                                            

        for local_seg_idx, (s, e, label) in enumerate(split_segments(meta_seq)):

            if int(label) != int(local_state):

                continue

            start_sec = float(time_arr[s]) if 0 <= s < len(time_arr) else float(s)

            end_sec = float(time_arr[e]) if 0 <= e < len(time_arr) else float(e)

            segment_rows.append({

                "video_id": video_id,

                "seg_idx": len(segment_rows),

                "local_seg_idx": int(local_seg_idx),

                "frame_start_idx_teacher": int(s),

                "frame_end_idx_teacher": int(e),

                "start_sec": start_sec,

                "end_sec": end_sec,

                "duration_sec": max(0.0, end_sec - start_sec),

                "duration_frames": int(e - s + 1),

                "layer1_meta_state_local": int(local_state),

                "local_meta_state": int(local_state),

                "prototype_id": f"{safe_name(video_id)}__local{int(local_state)}",

            })

    info = {

        "video_id": video_id,

        "case_dir": str(case_dir),

        "visual_csv": str(visual_csv),

        "n_teacher_frames": int(len(meta_seq)),

        "n_local_states": int(len(local_states)),

    }

    return pd.DataFrame(proto_rows), pd.DataFrame(segment_rows), info

                                                              

             

                                                              

def normalized_entropy_counts(counts) -> float:

    counts = np.asarray(counts, dtype=float)

    counts = counts[counts > 0]

    if len(counts) <= 1:

        return 0.0

    probs = counts / counts.sum()

    ent = -np.sum(probs * np.log(probs + 1e-12))

    return float(ent / np.log(len(counts)))

def build_feature_matrix(proto_df: pd.DataFrame, feature_cols: list[str], args):

    X = proto_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler()

    X_std = scaler.fit_transform(X)

    pca_info = {

        "used_pca": False,

        "pca_dim": 0,

        "original_dim": int(X_std.shape[1]),

        "explained_variance_ratio_sum": None,

    }

    if int(args.pca_dim) > 0 and X_std.shape[1] > int(args.pca_dim):

        dim = min(int(args.pca_dim), X_std.shape[0] - 1, X_std.shape[1])

        if dim >= 2:

            pca = PCA(n_components=dim, random_state=int(args.random_state))

            X_used = pca.fit_transform(X_std).astype(np.float32)

            pca_info = {

                "used_pca": True,

                "pca_dim": int(dim),

                "original_dim": int(X_std.shape[1]),

                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),

            }

            return X_used, pca_info

    return X_std.astype(np.float32), pca_info

def agglomerative_cluster(X, k: int, metric: str, linkage: str):

    try:

        model = AgglomerativeClustering(n_clusters=int(k), metric=metric, linkage=linkage)

    except TypeError:

        model = AgglomerativeClustering(n_clusters=int(k), affinity=metric, linkage=linkage)

    labels = model.fit_predict(X).astype(int)

    return labels

def cluster_labels(X, k: int, method: str, metric: str, linkage: str, random_state: int):

    method = str(method).lower()

    if method == "kmeans":

        model = KMeans(n_clusters=int(k), random_state=int(random_state), n_init=20)

        return model.fit_predict(X).astype(int)

    return agglomerative_cluster(X, k=int(k), metric=metric, linkage=linkage)

def evaluate_k(X, proto_df: pd.DataFrame, labels: np.ndarray, metric_for_silhouette: str):

    labels = np.asarray(labels).astype(int)

    n = len(labels)

    k = len(np.unique(labels))

    sil = np.nan

    if 1 < k < n:

        try:

            sil = float(silhouette_score(X, labels, metric=metric_for_silhouette))

        except Exception:

            sil = np.nan

    counts = np.asarray([np.sum(labels == c) for c in sorted(np.unique(labels))], dtype=float)

    balance = normalized_entropy_counts(counts)

    support_videos = []

    for c in sorted(np.unique(labels)):

        g = proto_df.loc[labels == c]

        support_videos.append(g["video_id"].astype(str).map(norm_video_id).nunique())

    support_videos = np.asarray(support_videos, dtype=float)

    mean_support = float(support_videos.mean()) if len(support_videos) else 0.0

    high_support_cluster_ratio = float(np.mean(support_videos >= 3)) if len(support_videos) else 0.0

    sil_for_score = sil if np.isfinite(sil) else -1.0

                                             

                                                                                             

    score = (

        sil_for_score

        + 0.03 * balance

        + 0.02 * high_support_cluster_ratio

    )

    return {

        "K": int(k),

        "silhouette": sil,

        "balance_entropy": float(balance),

        "mean_support_videos": mean_support,

        "high_support_cluster_ratio": float(high_support_cluster_ratio),

        "selection_score": float(score),

    }

def choose_k_and_cluster(X, proto_df: pd.DataFrame, args):

    n = X.shape[0]

    if n < 2:

        raise ValueError("prototype 数量少于 2，无法聚类")

    local_counts = proto_df.groupby("video_id")["local_meta_state"].nunique().to_numpy(dtype=float)

    median_local_k = int(round(float(np.median(local_counts)))) if len(local_counts) else 2

    if int(args.k) > 0:

        labels = cluster_labels(

            X,

            k=int(args.k),

            method=args.cluster_method,

            metric=args.cluster_metric,

            linkage=args.linkage,

            random_state=int(args.random_state),

        )

        score = evaluate_k(X, proto_df, labels, metric_for_silhouette=args.silhouette_metric)

        score["median_local_k"] = int(median_local_k)

        return int(args.k), labels, pd.DataFrame([score])

    low = max(2, median_local_k - int(args.k_radius))

    high = median_local_k + int(args.k_radius)

    if int(args.min_k) > 0:

        low = max(low, int(args.min_k))

    if int(args.max_k) > 0:

        high = min(high, int(args.max_k))

    high = min(high, n - 1)

    low = min(low, high)

    if high < 2:

        high = min(n - 1, 2)

        low = 2

    candidate_ks = list(range(low, high + 1))

    if not candidate_ks:

        candidate_ks = [min(max(2, median_local_k), n - 1)]

    rows = []

    label_by_k = {}

    for k in candidate_ks:

        labels = cluster_labels(

            X,

            k=int(k),

            method=args.cluster_method,

            metric=args.cluster_metric,

            linkage=args.linkage,

            random_state=int(args.random_state),

        )

        label_by_k[int(k)] = labels

        row = evaluate_k(X, proto_df, labels, metric_for_silhouette=args.silhouette_metric)

        row["median_local_k"] = int(median_local_k)

        row["k_low"] = int(low)

        row["k_high"] = int(high)

        rows.append(row)

    score_df = pd.DataFrame(rows).sort_values(

        ["selection_score", "silhouette", "balance_entropy"],

        ascending=[False, False, False],

    ).reset_index(drop=True)

    best_k = int(score_df.iloc[0]["K"])

    return best_k, label_by_k[best_k], score_df

def remap_cluster_labels_by_support(proto_df: pd.DataFrame, labels: np.ndarray):

    tmp = proto_df.copy()

    tmp["_label"] = np.asarray(labels).astype(int)

    order_rows = []

    for lab, g in tmp.groupby("_label"):

        order_rows.append({

            "old_label": int(lab),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "n_prototypes": int(len(g)),

            "total_frames_meta": int(g["n_frames_meta"].sum()) if "n_frames_meta" in g.columns else int(len(g)),

        })

    order_df = pd.DataFrame(order_rows).sort_values(

        ["support_videos", "n_prototypes", "total_frames_meta", "old_label"],

        ascending=[False, False, False, True],

    ).reset_index(drop=True)

    mapping = {int(row.old_label): int(i) for i, row in order_df.iterrows()}

    new_labels = np.array([mapping[int(x)] for x in labels], dtype=int)

    return new_labels, mapping, order_df

                                                              

         

                                                              

def build_global_summary(proto_df: pd.DataFrame, segment_df: pd.DataFrame):

    rows = []

    total_proto = max(1, len(proto_df))

    total_segments = max(1, len(segment_df))

    total_frames = max(1.0, float(segment_df["duration_frames"].sum())) if "duration_frames" in segment_df.columns else 1.0

    for gs, g in proto_df.groupby("global_state"):

        gs = int(gs)

        seg_g = segment_df[segment_df["global_state"].astype(int).eq(gs)].copy()

        vc = g["video_id"].astype(str).map(norm_video_id).value_counts()

        rows.append({

            "global_state": gs,

            "layer2_state": gs,

            "n_prototypes": int(len(g)),

            "prototype_ratio": float(len(g) / total_proto),

            "support_videos": int(g["video_id"].astype(str).map(norm_video_id).nunique()),

            "n_segments": int(len(seg_g)),

            "segment_ratio": float(len(seg_g) / total_segments),

            "total_duration_frames": int(seg_g["duration_frames"].sum()) if "duration_frames" in seg_g.columns else 0,

            "frame_ratio": float(seg_g["duration_frames"].sum() / total_frames) if "duration_frames" in seg_g.columns else 0.0,

            "top_video": str(vc.index[0]) if len(vc) else "",

            "top_video_prototypes": int(vc.iloc[0]) if len(vc) else 0,

            "top_video_ratio_within_state": float(vc.iloc[0] / max(1, len(g))) if len(vc) else 0.0,

        })

    return pd.DataFrame(rows).sort_values(

        ["support_videos", "n_prototypes"],

        ascending=[False, False],

    ).reset_index(drop=True)

                                                              

        

                                                              

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--visual-root", type=str, default=str(VISUAL_CSV_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--k", type=int, default=0, help="fixed K. 0 means auto median-sweep")

    ap.add_argument("--k-radius", type=int, default=5)

    ap.add_argument("--min-k", type=int, default=0)

    ap.add_argument("--max-k", type=int, default=0)

    ap.add_argument("--cluster-method", type=str, default="agglomerative", choices=["agglomerative", "kmeans"])

    ap.add_argument("--cluster-metric", type=str, default="euclidean", help="for agglomerative")

    ap.add_argument("--silhouette-metric", type=str, default="euclidean")

    ap.add_argument("--linkage", type=str, default="ward", help="ward/average/complete/single. ward requires euclidean.")

    ap.add_argument("--pca-dim", type=int, default=0)

    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--no-quantiles", action="store_true", help="only use mean/std/median; omit q25/q75/iqr")

    ap.add_argument("--max-cases", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.t2s_root)

    visual_root = Path(args.visual_root)

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    if str(args.linkage).lower() == "ward" and str(args.cluster_metric).lower() != "euclidean":

        print("[WARN] linkage=ward requires euclidean metric; forcing cluster_metric=euclidean")

        args.cluster_metric = "euclidean"

    print("=" * 100)

    print("X1-only cross-video prototype clustering for Layer-1 local states")

    print(f"t2s_root    = {root}")

    print(f"visual_root = {visual_root}")

    print(f"out_dir     = {out_dir}")

    print("聚类输入：仅使用每个 (video_id, local_meta_state) 内 X1 的分布统计。")

    print("不使用：branch signature / meta confidence / duration / frequency / NLP / audio。")

    print("硬约束：同一个 video_id + local_meta_state 只映射到一个 global_state。")

    print("=" * 100)

    case_dirs = find_case_dirs(root, out_dir)

    if args.max_cases and args.max_cases > 0:

        case_dirs = case_dirs[:int(args.max_cases)]

    if not case_dirs:

        raise FileNotFoundError(f"没有找到第一层输出 case: {root}")

    manifest_rows = []

    proto_tables = []

    segment_tables = []

    for i, case_dir in enumerate(case_dirs, start=1):

        video_id = infer_video_id_from_case_dir(root, case_dir)

        print(f"[{i}/{len(case_dirs)}] {case_dir}")

        row = {

            "video_id": video_id,

            "case_dir": str(case_dir),

            "included": 0,

            "skip_reason": "",

            "n_teacher_frames": 0,

            "n_local_states": 0,

        }

        try:

            proto_df, seg_df, info = build_case_prototypes_x1_only(

                case_dir=case_dir,

                root=root,

                visual_root=visual_root,

                use_quantiles=not bool(args.no_quantiles),

            )

            if len(proto_df) < 1:

                raise ValueError("prototype 表为空")

            proto_tables.append(proto_df)

            segment_tables.append(seg_df)

            row.update({

                "included": 1,

                "n_teacher_frames": info["n_teacher_frames"],

                "n_local_states": info["n_local_states"],

                "visual_csv": info["visual_csv"],

            })

            print(f"  [OK] frames={row['n_teacher_frames']}, local_states={row['n_local_states']}")

        except Exception as e:

            row["skip_reason"] = repr(e)

            print(f"  [SKIP] {repr(e)}")

        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_path = out_dir / "prototype_alignment_X1_only_build_manifest.csv"

    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[OK] manifest: {manifest_path}")

    if not proto_tables:

        raise RuntimeError("没有可用 prototype，无法跨视频聚类。")

    proto_df = pd.concat(proto_tables, axis=0, ignore_index=True)

    segment_df = pd.concat(segment_tables, axis=0, ignore_index=True)

                                                  

                                                                                                 

    feature_cols = [

        c for c in proto_df.columns

        if (

            c.startswith("mean_x1_")

            or c.startswith("std_x1_")

            or c.startswith("median_x1_")

            or c.startswith("q25_x1_")

            or c.startswith("q75_x1_")

            or c.startswith("iqr_x1_")

        )

        and pd.api.types.is_numeric_dtype(proto_df[c])

    ]

    if not feature_cols:

        raise RuntimeError("没有找到 X1 分布特征列，无法聚类。")

    proto_df[feature_cols] = proto_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X, pca_info = build_feature_matrix(proto_df, feature_cols, args)

    best_k, labels_raw, sweep_df = choose_k_and_cluster(X, proto_df, args)

    labels, label_map, label_order_df = remap_cluster_labels_by_support(proto_df, labels_raw)

    proto_df["global_state"] = labels.astype(int)

    proto_df["layer2_state"] = proto_df["global_state"].astype(int)

                                                            

    mapping = {}

    for _, row in proto_df.iterrows():

        mapping[(norm_video_id(row["video_id"]), int(row["local_meta_state"]))] = int(row["global_state"])

    segment_df["video_id"] = segment_df["video_id"].astype(str).map(norm_video_id)

    segment_df["global_state"] = [

        mapping.get((norm_video_id(v), int(s)), -1)

        for v, s in zip(segment_df["video_id"], segment_df["local_meta_state"])

    ]

    segment_df["layer2_state"] = segment_df["global_state"].astype(int)

    segment_df["layer1_meta_state_local"] = segment_df["local_meta_state"].astype(int)

             

    proto_path = out_dir / "prototype_table_with_global_state_X1_only.csv"

    proto_df.to_csv(proto_path, index=False, encoding="utf-8-sig")

    mapping_path = out_dir / "local_to_global_state_mapping_X1_only.csv"

    proto_df[[

        "video_id", "local_meta_state", "prototype_id", "global_state", "layer2_state",

        "n_frames_meta", "frame_ratio_in_video_meta",

    ]].sort_values(["global_state", "video_id", "local_meta_state"]).to_csv(

        mapping_path,

        index=False,

        encoding="utf-8-sig",

    )

    segment_path = out_dir / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

    segment_df.sort_values(["video_id", "start_sec", "end_sec"]).to_csv(segment_path, index=False, encoding="utf-8-sig")

    expert_index_path = out_dir / "expert_segment_index_by_global_state_X1_only.csv"

    idx_cols = [

        "global_state", "layer2_state", "video_id", "seg_idx", "local_seg_idx",

        "start_sec", "end_sec", "duration_sec", "duration_frames",

        "local_meta_state", "layer1_meta_state_local", "prototype_id",

    ]

    idx_cols = [c for c in idx_cols if c in segment_df.columns]

    segment_df[idx_cols].sort_values(["global_state", "video_id", "start_sec"]).to_csv(

        expert_index_path,

        index=False,

        encoding="utf-8-sig",

    )

    summary_df = build_global_summary(proto_df, segment_df)

    summary_path = out_dir / "global_state_summary_X1_only.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    sweep_path = out_dir / "prototype_k_sweep_scores_X1_only.csv"

    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    label_order_path = out_dir / "cluster_label_remap_order_X1_only.csv"

    label_order_df.to_csv(label_order_path, index=False, encoding="utf-8-sig")

    feature_info_path = out_dir / "prototype_feature_columns_X1_only.json"

    with open(feature_info_path, "w", encoding="utf-8") as f:

        json.dump({

            "feature_cols": feature_cols,

            "n_features": len(feature_cols),

            "pca_info": pca_info,

            "best_k": int(best_k),

            "use_quantiles": not bool(args.no_quantiles),

            "cluster_method": str(args.cluster_method),

            "cluster_metric": str(args.cluster_metric),

            "linkage": str(args.linkage),

            "silhouette_metric": str(args.silhouette_metric),

            "excluded_from_clustering": [

                "selected branch signature",

                "branch rank",

                "meta confidence",

                "duration",

                "n_frames",

                "frame_ratio",

                "video_id",

                "local_meta_state numerical id",

                "NLP",

                "audio_csv",

                "raw video image",

            ],

            "hard_constraint": "One (video_id, local_meta_state) maps to exactly one global_state.",

        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)

    print("DONE.")

    print(f"prototypes : {len(proto_df)}")

    print(f"segments   : {len(segment_df)}")

    print(f"features   : {len(feature_cols)}")

    print(f"best_K     : {best_k}")

    print(f"prototype table : {proto_path}")

    print(f"mapping         : {mapping_path}")

    print(f"segment mapping : {segment_path}")

    print(f"summary         : {summary_path}")

    print(f"k sweep         : {sweep_path}")

    print(f"expert index    : {expert_index_path}")

    print("=" * 100)

if __name__ == "__main__":

    main()
