                       

   

from __future__ import annotations

import argparse

import json

import re

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.metrics import silhouette_score

                                                              

         

                                                              

T2S_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

VISUAL_CSV_ROOT_DEFAULT = Path(r"D:\code\teacherT2S\yolo\pose_csv")

OUT_DIR_DEFAULT = T2S_ROOT_DEFAULT / "_cross_video_prototype_alignment_X1_only_anchor_k5_map_12_2"

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

def strip_leading_zero_token(s: str) -> str:

    s = str(s).strip()

    if s.isdigit():

        return str(int(s))

    return s

def video_id_aliases(x) -> set[str]:

    

       

    vid = norm_video_id(x)

    parts = vid.split("/") if vid else []

    aliases = set()

    if vid:

        aliases.add(vid)

    if parts:

        aliases.add(parts[-1])

        aliases.add(strip_leading_zero_token(parts[-1]))

        aliases.add(parts[0])

        aliases.add(strip_leading_zero_token(parts[0]))

        stripped_parts = [strip_leading_zero_token(p) for p in parts]

        aliases.add("/".join(stripped_parts))

             

    t = tail_video_id(vid)

    if t:

        aliases.add(t)

        aliases.add(strip_leading_zero_token(t))

    return {a for a in aliases if a != ""}

def parse_video_id_set(s: str) -> set[str]:

    out = set()

    if s is None:

        return out

    for x in str(s).replace("，", ",").split(","):

        x = norm_video_id(x)

        if not x:

            continue

        out.update(video_id_aliases(x))

    return out

def is_target_video(video_id: str, target_aliases: set[str]) -> bool:

    aliases = video_id_aliases(video_id)

    return len(aliases & target_aliases) > 0

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

                                                

        parts_lower = [part.lower() for part in case_dir.parts]

        if any(

            part.startswith("_cross")

            or part.startswith("_expert")

            or part.startswith("_archive")

            or part in {".tmp", ".numba_cache"}

            for part in parts_lower

        ):

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

                                                              

             

                                                              

def quantize_orientation_3class(series):

    s = series.copy().clip(-2.0, 2.0)

    out = []

    for v in s:

        if pd.isna(v):

            out.append(np.nan)

        elif v <= -0.25:

            out.append(-1.0)

        elif v >= 0.25:

            out.append(1.0)

        else:

            out.append(0.0)

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

    ori3 = quantize_orientation_3class(X["orientation_score"].astype(float))

    out = pd.DataFrame(index=X.index)

    out["orientation_3class"] = ori3

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

def build_feature_cols(proto_df: pd.DataFrame) -> list[str]:

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

    return feature_cols

def maybe_apply_pca(X_std, args):

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

            return X_used, pca, pca_info

    return X_std.astype(np.float32), None, pca_info

def agglomerative_cluster(X, k: int, metric: str, linkage: str):

    try:

        model = AgglomerativeClustering(n_clusters=int(k), metric=metric, linkage=linkage)

    except TypeError:

        model = AgglomerativeClustering(n_clusters=int(k), affinity=metric, linkage=linkage)

    return model.fit_predict(X).astype(int)

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

    score = sil_for_score + 0.03 * balance + 0.02 * high_support_cluster_ratio

    return {

        "K": int(k),

        "silhouette": sil,

        "balance_entropy": float(balance),

        "mean_support_videos": mean_support,

        "high_support_cluster_ratio": float(high_support_cluster_ratio),

        "selection_score": float(score),

    }

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

    order_df["new_label"] = order_df["old_label"].map(mapping).astype(int)

    order_df = order_df.sort_values("new_label").reset_index(drop=True)

    return new_labels, mapping, order_df

def compute_centroids(X_base_used: np.ndarray, labels: np.ndarray):

    centroids = {}

    labels = np.asarray(labels).astype(int)

    for gs in sorted(np.unique(labels)):

        centroids[int(gs)] = X_base_used[labels == gs].mean(axis=0)

    return centroids

def assign_targets_to_centroids(X_target_used: np.ndarray, centroids: dict[int, np.ndarray]):

    states = sorted(centroids.keys())

    C = np.vstack([centroids[s] for s in states]).astype(np.float32)

                                             

    diff = X_target_used[:, None, :] - C[None, :, :]

    dist = np.sqrt(np.sum(diff * diff, axis=2))

    nearest_idx = np.argmin(dist, axis=1)

    assigned = np.array([states[i] for i in nearest_idx], dtype=int)

    nearest_dist = dist[np.arange(len(nearest_idx)), nearest_idx]

                                       

    if dist.shape[1] >= 2:

        sorted_dist = np.sort(dist, axis=1)

        margin = sorted_dist[:, 1] - sorted_dist[:, 0]

        ratio = sorted_dist[:, 0] / (sorted_dist[:, 1] + 1e-12)

    else:

        margin = np.full(len(nearest_dist), np.nan)

        ratio = np.full(len(nearest_dist), np.nan)

    return assigned, nearest_dist, margin, ratio, dist, states

                                                              

         

                                                              

def build_global_summary(proto_df: pd.DataFrame, segment_df: pd.DataFrame):

    rows = []

    total_proto = max(1, len(proto_df))

    total_segments = max(1, len(segment_df))

    total_frames = max(1.0, float(segment_df["duration_frames"].sum())) if "duration_frames" in segment_df.columns else 1.0

    for gs, g in proto_df.groupby("global_state"):

        gs = int(gs)

        seg_g = segment_df[segment_df["global_state"].astype(int).eq(gs)].copy()

        vc = g["video_id"].astype(str).map(norm_video_id).value_counts()

        target_proto_count = int(g.get("is_target_posterior", pd.Series([0] * len(g))).sum()) if len(g) else 0

        rows.append({

            "global_state": gs,

            "layer2_state": gs,

            "n_prototypes": int(len(g)),

            "n_base_prototypes": int(len(g) - target_proto_count),

            "n_target_prototypes": int(target_proto_count),

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

def build_video_global_state_count(segment_df: pd.DataFrame):

    rows = []

    for vid, g in segment_df.groupby("video_id"):

        states = sorted([int(x) for x in pd.unique(g["global_state"]) if int(x) >= 0])

        rows.append({

            "video_id": vid,

            "n_segments": int(len(g)),

            "n_global_states": int(len(states)),

            "global_states": ",".join(map(str, states)),

            "total_duration_frames": int(g["duration_frames"].sum()) if "duration_frames" in g.columns else 0,

            "total_duration_sec": float(g["duration_sec"].sum()) if "duration_sec" in g.columns else 0.0,

            "is_degenerate_single_global_state": int(len(states) <= 1),

        })

    return pd.DataFrame(rows).sort_values(["n_global_states", "video_id"]).reset_index(drop=True)

                                                              

        

                                                              

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--t2s-root", type=str, default=str(T2S_ROOT_DEFAULT))

    ap.add_argument("--visual-root", type=str, default=str(VISUAL_CSV_ROOT_DEFAULT))

    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--target-video-ids", type=str, default="12,2",

                    help="后验映射目标视频。默认 12,2；其中 2 会匹配 2/02。")

    ap.add_argument("--base-k", type=int, default=5)

    ap.add_argument("--cluster-method", type=str, default="agglomerative", choices=["agglomerative", "kmeans"])

    ap.add_argument("--cluster-metric", type=str, default="euclidean")

    ap.add_argument("--silhouette-metric", type=str, default="euclidean")

    ap.add_argument("--linkage", type=str, default="ward")

    ap.add_argument("--pca-dim", type=int, default=0)

    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--no-quantiles", action="store_true")

    ap.add_argument("--max-cases", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.t2s_root)

    visual_root = Path(args.visual_root)

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    if str(args.linkage).lower() == "ward" and str(args.cluster_metric).lower() != "euclidean":

        print("[WARN] linkage=ward requires euclidean metric; forcing cluster_metric=euclidean")

        args.cluster_metric = "euclidean"

    target_aliases = parse_video_id_set(args.target_video_ids)

    print("=" * 100)

    print("X1-only anchor clustering + posterior mapping")

    print(f"t2s_root       = {root}")

    print(f"visual_root    = {visual_root}")

    print(f"out_dir        = {out_dir}")

    print(f"target ids     = {args.target_video_ids}")

    print(f"target aliases = {sorted(target_aliases)}")

    print(f"base K         = {args.base_k}")

    print("base 视频参与建簇；target 视频不参与建簇，只后验映射到最近 global centroid。")

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

        role = "target_posterior" if is_target_video(video_id, target_aliases) else "base_anchor"

        print(f"[{i}/{len(case_dirs)}] {case_dir}  role={role}")

        row = {

            "video_id": video_id,

            "case_dir": str(case_dir),

            "role": role,

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

            proto_df["role"] = role

            proto_df["is_target_posterior"] = int(role == "target_posterior")

            seg_df["role"] = role

            seg_df["is_target_posterior"] = int(role == "target_posterior")

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

    manifest_path = out_dir / "prototype_alignment_anchor_map_build_manifest.csv"

    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    if not proto_tables:

        raise RuntimeError("没有可用 prototype，无法跨视频聚类。")

    all_proto_df = pd.concat(proto_tables, axis=0, ignore_index=True)

    all_segment_df = pd.concat(segment_tables, axis=0, ignore_index=True)

    base_proto_df = all_proto_df[all_proto_df["role"].eq("base_anchor")].copy().reset_index(drop=True)

    target_proto_df = all_proto_df[all_proto_df["role"].eq("target_posterior")].copy().reset_index(drop=True)

    base_segment_df = all_segment_df[all_segment_df["role"].eq("base_anchor")].copy().reset_index(drop=True)

    target_segment_df = all_segment_df[all_segment_df["role"].eq("target_posterior")].copy().reset_index(drop=True)

    if len(base_proto_df) < int(args.base_k):

        raise RuntimeError(f"base prototype 数量不足以聚 K={args.base_k}: {len(base_proto_df)}")

    if len(target_proto_df) == 0:

        print("[WARN] 没有找到 target prototype。请检查 --target-video-ids 是否匹配。")

    feature_cols = build_feature_cols(all_proto_df)

    if not feature_cols:

        raise RuntimeError("没有找到 X1 分布特征列，无法聚类。")

    for df in [base_proto_df, target_proto_df, all_proto_df]:

        if len(df) > 0:

            df[feature_cols] = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                                                            

    scaler = StandardScaler()

    X_base_std = scaler.fit_transform(base_proto_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)

    if len(target_proto_df) > 0:

        X_target_std = scaler.transform(target_proto_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)

    else:

        X_target_std = np.zeros((0, X_base_std.shape[1]), dtype=np.float32)

                                            

    pca_info = {

        "used_pca": False,

        "pca_dim": 0,

        "original_dim": int(X_base_std.shape[1]),

        "explained_variance_ratio_sum": None,

    }

    pca_model = None

    X_base_used = X_base_std

    X_target_used = X_target_std

    if int(args.pca_dim) > 0 and X_base_std.shape[1] > int(args.pca_dim):

        dim = min(int(args.pca_dim), X_base_std.shape[0] - 1, X_base_std.shape[1])

        if dim >= 2:

            pca_model = PCA(n_components=dim, random_state=int(args.random_state))

            X_base_used = pca_model.fit_transform(X_base_std).astype(np.float32)

            X_target_used = pca_model.transform(X_target_std).astype(np.float32) if len(target_proto_df) else X_target_std

            pca_info = {

                "used_pca": True,

                "pca_dim": int(dim),

                "original_dim": int(X_base_std.shape[1]),

                "explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),

            }

             

    base_labels_raw = cluster_labels(

        X_base_used,

        k=int(args.base_k),

        method=args.cluster_method,

        metric=args.cluster_metric,

        linkage=args.linkage,

        random_state=int(args.random_state),

    )

    base_score = evaluate_k(X_base_used, base_proto_df, base_labels_raw, metric_for_silhouette=args.silhouette_metric)

    base_labels, label_map, label_order_df = remap_cluster_labels_by_support(base_proto_df, base_labels_raw)

    base_proto_df["global_state"] = base_labels.astype(int)

    base_proto_df["layer2_state"] = base_proto_df["global_state"].astype(int)

                                        

    centroids = compute_centroids(X_base_used, base_labels)

                 

    if len(target_proto_df) > 0:

        target_labels, nearest_dist, margin, ratio, dist_matrix, centroid_states = assign_targets_to_centroids(

            X_target_used,

            centroids,

        )

        target_proto_df["global_state"] = target_labels.astype(int)

        target_proto_df["layer2_state"] = target_proto_df["global_state"].astype(int)

        target_proto_df["posterior_nearest_dist"] = nearest_dist.astype(float)

        target_proto_df["posterior_margin_second_minus_first"] = margin.astype(float)

        target_proto_df["posterior_nearest_to_second_ratio"] = ratio.astype(float)

        for j, s in enumerate(centroid_states):

            target_proto_df[f"dist_to_global_{int(s)}"] = dist_matrix[:, j].astype(float)

    else:

        target_proto_df["global_state"] = []

        target_proto_df["layer2_state"] = []

                  

    combined_proto_df = pd.concat([base_proto_df, target_proto_df], axis=0, ignore_index=True)

    combined_proto_df["global_state"] = combined_proto_df["global_state"].astype(int)

    combined_proto_df["layer2_state"] = combined_proto_df["global_state"].astype(int)

                

    mapping = {}

    for _, row in combined_proto_df.iterrows():

        mapping[(norm_video_id(row["video_id"]), int(row["local_meta_state"]))] = int(row["global_state"])

    combined_segment_df = pd.concat([base_segment_df, target_segment_df], axis=0, ignore_index=True)

    combined_segment_df["video_id"] = combined_segment_df["video_id"].astype(str).map(norm_video_id)

    combined_segment_df["global_state"] = [

        mapping.get((norm_video_id(v), int(s)), -1)

        for v, s in zip(combined_segment_df["video_id"], combined_segment_df["local_meta_state"])

    ]

    combined_segment_df["layer2_state"] = combined_segment_df["global_state"].astype(int)

    combined_segment_df["layer1_meta_state_local"] = combined_segment_df["local_meta_state"].astype(int)

                              

    base_segment_df["video_id"] = base_segment_df["video_id"].astype(str).map(norm_video_id)

    base_segment_df["global_state"] = [

        mapping.get((norm_video_id(v), int(s)), -1)

        for v, s in zip(base_segment_df["video_id"], base_segment_df["local_meta_state"])

    ]

    base_segment_df["layer2_state"] = base_segment_df["global_state"].astype(int)

    base_segment_df["layer1_meta_state_local"] = base_segment_df["local_meta_state"].astype(int)

                               

    if len(target_segment_df) > 0:

        target_segment_df["video_id"] = target_segment_df["video_id"].astype(str).map(norm_video_id)

        target_segment_df["global_state"] = [

            mapping.get((norm_video_id(v), int(s)), -1)

            for v, s in zip(target_segment_df["video_id"], target_segment_df["local_meta_state"])

        ]

        target_segment_df["layer2_state"] = target_segment_df["global_state"].astype(int)

        target_segment_df["layer1_meta_state_local"] = target_segment_df["local_meta_state"].astype(int)

        

    base_proto_path = out_dir / "base_prototype_table_with_global_state_X1_only.csv"

    target_proto_path = out_dir / "target_posterior_mapping_X1_only.csv"

    combined_proto_path = out_dir / "combined_prototype_table_with_global_state_X1_only.csv"

    base_proto_df.to_csv(base_proto_path, index=False, encoding="utf-8-sig")

    target_proto_df.to_csv(target_proto_path, index=False, encoding="utf-8-sig")

    combined_proto_df.to_csv(combined_proto_path, index=False, encoding="utf-8-sig")

    base_segment_path = out_dir / "base_layer2_cross_video_prototype_aligned_segments_X1_only.csv"

    target_segment_path = out_dir / "target_posterior_aligned_segments_X1_only.csv"

    combined_segment_path = out_dir / "layer2_cross_video_prototype_aligned_segments_X1_only.csv"

    base_segment_df.sort_values(["video_id", "start_sec", "end_sec"]).to_csv(base_segment_path, index=False, encoding="utf-8-sig")

    target_segment_df.sort_values(["video_id", "start_sec", "end_sec"]).to_csv(target_segment_path, index=False, encoding="utf-8-sig")

    combined_segment_df.sort_values(["video_id", "start_sec", "end_sec"]).to_csv(combined_segment_path, index=False, encoding="utf-8-sig")

    mapping_path = out_dir / "local_to_global_state_mapping_X1_only.csv"

    map_cols = [

        "role", "is_target_posterior", "video_id", "local_meta_state",

        "prototype_id", "global_state", "layer2_state",

        "n_frames_meta", "frame_ratio_in_video_meta",

        "posterior_nearest_dist", "posterior_margin_second_minus_first",

        "posterior_nearest_to_second_ratio",

    ]

    map_cols = [c for c in map_cols if c in combined_proto_df.columns]

    combined_proto_df[map_cols].sort_values(["role", "global_state", "video_id", "local_meta_state"]).to_csv(

        mapping_path,

        index=False,

        encoding="utf-8-sig",

    )

    summary_path = out_dir / "global_state_summary_X1_only.csv"

    summary_df = build_global_summary(combined_proto_df, combined_segment_df)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    base_summary_path = out_dir / "base_global_state_summary_X1_only.csv"

    base_summary_df = build_global_summary(base_proto_df, base_segment_df)

    base_summary_df.to_csv(base_summary_path, index=False, encoding="utf-8-sig")

    video_count_path = out_dir / "video_global_state_count_X1_only.csv"

    video_count_df = build_video_global_state_count(combined_segment_df)

    video_count_df.to_csv(video_count_path, index=False, encoding="utf-8-sig")

    degenerate_path = out_dir / "degenerate_videos_X1_only.csv"

    degenerate_df = video_count_df[video_count_df["n_global_states"].le(1)].copy()

    degenerate_df.to_csv(degenerate_path, index=False, encoding="utf-8-sig")

    target_report_path = out_dir / "target_mapping_distance_report_X1_only.csv"

    if len(target_proto_df) > 0:

        report_cols = [

            "video_id", "local_meta_state", "prototype_id",

            "global_state", "n_frames_meta", "frame_ratio_in_video_meta",

            "posterior_nearest_dist", "posterior_margin_second_minus_first",

            "posterior_nearest_to_second_ratio",

        ]

        report_cols += [c for c in target_proto_df.columns if c.startswith("dist_to_global_")]

        report_cols = [c for c in report_cols if c in target_proto_df.columns]

        target_proto_df[report_cols].sort_values(["video_id", "global_state", "local_meta_state"]).to_csv(

            target_report_path,

            index=False,

            encoding="utf-8-sig",

        )

    else:

        pd.DataFrame().to_csv(target_report_path, index=False, encoding="utf-8-sig")

    expert_index_path = out_dir / "expert_segment_index_by_global_state_X1_only.csv"

    idx_cols = [

        "global_state", "layer2_state", "role", "is_target_posterior",

        "video_id", "seg_idx", "local_seg_idx",

        "start_sec", "end_sec", "duration_sec", "duration_frames",

        "local_meta_state", "layer1_meta_state_local", "prototype_id",

    ]

    idx_cols = [c for c in idx_cols if c in combined_segment_df.columns]

    combined_segment_df[idx_cols].sort_values(["global_state", "role", "video_id", "start_sec"]).to_csv(

        expert_index_path,

        index=False,

        encoding="utf-8-sig",

    )

             

    score_path = out_dir / "base_k_score_X1_only.csv"

    pd.DataFrame([base_score]).to_csv(score_path, index=False, encoding="utf-8-sig")

    label_order_path = out_dir / "cluster_label_remap_order_X1_only.csv"

    label_order_df.to_csv(label_order_path, index=False, encoding="utf-8-sig")

    feature_info_path = out_dir / "prototype_feature_columns_X1_only.json"

    with open(feature_info_path, "w", encoding="utf-8") as f:

        json.dump({

            "feature_cols": feature_cols,

            "n_features": len(feature_cols),

            "base_k": int(args.base_k),

            "target_video_ids": str(args.target_video_ids),

            "target_aliases": sorted(target_aliases),

            "n_base_prototypes": int(len(base_proto_df)),

            "n_target_prototypes": int(len(target_proto_df)),

            "pca_info": pca_info,

            "use_quantiles": not bool(args.no_quantiles),

            "cluster_method": str(args.cluster_method),

            "cluster_metric": str(args.cluster_metric),

            "linkage": str(args.linkage),

            "silhouette_metric": str(args.silhouette_metric),

            "base_scaler_fit_only_on_base": True,

            "target_assignment": "nearest global centroid in base-standardized feature space",

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

    print(f"base videos      : {base_proto_df['video_id'].astype(str).map(norm_video_id).nunique()}")

    print(f"target videos    : {target_proto_df['video_id'].astype(str).map(norm_video_id).nunique() if len(target_proto_df) else 0}")

    print(f"base prototypes  : {len(base_proto_df)}")

    print(f"target prototypes: {len(target_proto_df)}")

    print(f"combined segments: {len(combined_segment_df)}")

    print(f"feature dim      : {len(feature_cols)}")

    print(f"base K           : {args.base_k}")

    print(f"base silhouette  : {base_score.get('silhouette', np.nan)}")

    print(f"degenerate videos n<=1: {len(degenerate_df)}")

    if len(degenerate_df) > 0:

        print(degenerate_df[["video_id", "n_segments", "n_global_states", "global_states"]].to_string(index=False))

    print("\nOutput:")

    print(f"manifest         : {manifest_path}")

    print(f"base proto       : {base_proto_path}")

    print(f"target mapping   : {target_proto_path}")

    print(f"combined proto   : {combined_proto_path}")

    print(f"combined segments: {combined_segment_path}")

    print(f"summary          : {summary_path}")

    print(f"video counts     : {video_count_path}")

    print(f"degenerate list  : {degenerate_path}")

    print(f"target distances : {target_report_path}")

    print(f"expert index     : {expert_index_path}")

    print("=" * 100)

if __name__ == "__main__":

    main()
