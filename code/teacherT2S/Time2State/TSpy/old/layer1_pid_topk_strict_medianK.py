

   

from __future__ import annotations

import json

from pathlib import Path

from typing import Iterable

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

try:

    from scipy.cluster.hierarchy import linkage, dendrogram

    SCIPY_OK = True

except Exception:

    SCIPY_OK = False

                                                           

        

                                                           

T2S_OUTPUT_ROOT = Path(r"D:\code\teacherT2S\multiscale_t2s_output_event_batch")

OUTPUT_ROOT = T2S_OUTPUT_ROOT / "_cross_video_proto_alignment_pid_topk_strict"

PLOT_DIR = OUTPUT_ROOT / "plots"

GLOBAL_K_MIN = 2

GLOBAL_K_MAX = 20

            

                                                               

                                          

                                                                        

                                          

GLOBAL_K_FROM_LOCAL_MEDIAN = True

GLOBAL_K_MARGIN = 1

SAVE_LABELED_PCA = True

MAX_LABEL_POINTS = 180

SHARED_MIN_SUPPORT_VIDEOS = 5

VARIANT_MIN_SUPPORT_VIDEOS = 2

        

STRICT_TEACHER_ONLY = True                                    

DROP_SKIP_META_STATE = True                            

USE_SELECTED_BRANCHES_ONLY = True                                             

MIN_FRAMES_PER_LOCAL_STATE = 1                                       

     

FINAL_META_CSV = "multiscale_t2s_with_meta.csv"

RUN_INFO_CSV = "multiscale_t2s_run_info.csv"

BRANCH_METRICS_CSV = "multiscale_branch_metrics_pid_peer.csv"

RAW_LABEL_CANDIDATES = [

    "multiscale_raw_label_matrix_Tx32.csv",

    "multiscale_raw_label_matrix.csv",

]

                                                           

         

                                                           

def ensure_dirs() -> None:

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

def remap_to_contiguous_labels(arr: Iterable[int]) -> tuple[np.ndarray, dict[int, int]]:

    vals = sorted(pd.unique(pd.Series(arr).dropna().astype(int)))

    mp = {int(v): i for i, v in enumerate(vals)}

    return np.array([mp[int(x)] for x in arr], dtype=int), mp

def split_state_segments(state_seq: Iterable[int]) -> list[tuple[int, int, int]]:

    seq = np.asarray(state_seq).astype(int)

    if len(seq) == 0:

        return []

    segs = []

    s = 0

    for i in range(1, len(seq)):

        if seq[i] != seq[s]:

            segs.append((s, i - 1, int(seq[s])))

            s = i

    segs.append((s, len(seq) - 1, int(seq[s])))

    return segs

def safe_normalized_entropy(seq: Iterable[int]) -> float:

    seq = pd.Series(seq).dropna().astype(int).values

    if len(seq) == 0:

        return 0.0

    vals, counts = np.unique(seq, return_counts=True)

    probs = counts.astype(np.float64) / counts.sum()

    if len(probs) <= 1:

        return 0.0

    ent = -np.sum(probs * np.log(probs + 1e-12))

    ent_max = np.log(len(probs))

    if ent_max < 1e-12:

        return 0.0

    return float(ent / ent_max)

def find_raw_label_csv(case_dir: Path) -> Path | None:

    for name in RAW_LABEL_CANDIDATES:

        p = case_dir / name

        if p.exists():

            return p

    matches = sorted(case_dir.glob("multiscale_raw_label_matrix_Tx*.csv"))

    if matches:

        return matches[0]

    return None

def find_all_case_dirs(root: Path) -> list[Path]:

                                                                   

    case_dirs = []

    for final_csv in root.rglob(FINAL_META_CSV):

        case_dir = final_csv.parent

        raw_csv = find_raw_label_csv(case_dir)

        run_info_csv = case_dir / RUN_INFO_CSV

        if raw_csv is not None and run_info_csv.exists():

            case_dirs.append(case_dir)

                      

    case_dirs = [p for p in case_dirs if OUTPUT_ROOT not in p.parents and p != OUTPUT_ROOT]

    return sorted(set(case_dirs))

def bool_int_series(series: pd.Series, default: int = 0) -> pd.Series:

    return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)

def select_run_columns(raw_df: pd.DataFrame, run_info_df: pd.DataFrame, branch_metrics_df: pd.DataFrame | None) -> tuple[list[str], pd.DataFrame, str]:

    

       

    run_info_use = run_info_df.copy()

    source = "all_run_info"

    if USE_SELECTED_BRANCHES_ONLY:

        if "selected_for_meta" in run_info_df.columns:

            flag = bool_int_series(run_info_df["selected_for_meta"], default=0)

            selected = run_info_df.loc[flag.eq(1)].copy()

            if len(selected) > 0:

                run_info_use = selected

                source = "run_info_selected_for_meta"

        elif branch_metrics_df is not None and "selected_for_meta" in branch_metrics_df.columns:

            flag = bool_int_series(branch_metrics_df["selected_for_meta"], default=0)

            selected_metrics = branch_metrics_df.loc[flag.eq(1)].copy()

            if len(selected_metrics) > 0:

                if "run_idx" in selected_metrics.columns and "run_idx" in run_info_df.columns:

                    selected_run_idx = set(pd.to_numeric(selected_metrics["run_idx"], errors="coerce").dropna().astype(int).tolist())

                    run_info_use = run_info_df.loc[

                        pd.to_numeric(run_info_df["run_idx"], errors="coerce").fillna(-999).astype(int).isin(selected_run_idx)

                    ].copy()

                    source = "branch_metrics_selected_for_meta"

                elif "col_name" in selected_metrics.columns and "col_name" in run_info_df.columns:

                    selected_cols = set(selected_metrics["col_name"].astype(str).tolist())

                    run_info_use = run_info_df.loc[run_info_df["col_name"].astype(str).isin(selected_cols)].copy()

                    source = "branch_metrics_selected_for_meta"

    if "col_name" in run_info_use.columns:

        run_cols = run_info_use["col_name"].astype(str).tolist()

    elif "col_name" in run_info_df.columns:

        run_cols = run_info_df["col_name"].astype(str).tolist()

        source = "all_run_info_no_selected_col"

    else:

        run_cols = [c for c in raw_df.columns if c != "time_sec"]

        source = "raw_all_non_time_columns"

    run_cols = [c for c in run_cols if c in raw_df.columns]

    if len(run_cols) == 0:

        run_cols = [c for c in raw_df.columns if c != "time_sec"]

        run_info_use = run_info_df.copy()

        source = "fallback_raw_all_non_time_columns"

    return run_cols, run_info_use, source

def build_keep_mask(final_df: pd.DataFrame) -> pd.Series:

    keep = pd.Series(True, index=final_df.index)

    if DROP_SKIP_META_STATE:

        meta_num = pd.to_numeric(final_df["meta_state"], errors="coerce").fillna(-1).astype(int)

        keep &= meta_num.ge(0)

    if STRICT_TEACHER_ONLY and "is_teacher_frame" in final_df.columns:

        keep &= bool_int_series(final_df["is_teacher_frame"], default=0).eq(1)

    return keep

                                                           

                              

                                                           

def extract_local_state_prototypes_from_t2s() -> pd.DataFrame:

    case_dirs = find_all_case_dirs(T2S_OUTPUT_ROOT)

    if len(case_dirs) == 0:

        raise FileNotFoundError(f"没有找到完整的 T2S case 输出目录: {T2S_OUTPUT_ROOT}")

    print(f"共找到 {len(case_dirs)} 个候选 case")

    rows = []

    manifest_rows = []

    for case_dir in case_dirs:

        rel_case = case_dir.relative_to(T2S_OUTPUT_ROOT)

        video_id = str(rel_case).replace("\\", "/")

        print("=" * 80)

        print(f"处理 case: {video_id}")

        final_csv = case_dir / FINAL_META_CSV

        raw_csv = find_raw_label_csv(case_dir)

        run_info_csv = case_dir / RUN_INFO_CSV

        branch_metrics_csv = case_dir / BRANCH_METRICS_CSV

        manifest = {

            "video_id": video_id,

            "case_dir": str(case_dir),

            "final_csv": str(final_csv),

            "raw_csv": str(raw_csv) if raw_csv is not None else "",

            "run_info_csv": str(run_info_csv),

            "branch_metrics_csv": str(branch_metrics_csv) if branch_metrics_csv.exists() else "",

            "included": 0,

            "skip_reason": "",

            "n_rows_final": 0,

            "n_rows_raw": 0,

            "n_rows_kept": 0,

            "n_local_states": 0,

            "n_run_cols_used": 0,

            "run_col_source": "",

        }

        try:

            final_df = pd.read_csv(final_csv)

            raw_df = pd.read_csv(raw_csv) if raw_csv is not None else None

            run_info_df = pd.read_csv(run_info_csv)

            branch_metrics_df = pd.read_csv(branch_metrics_csv) if branch_metrics_csv.exists() else None

        except Exception as e:

            manifest["skip_reason"] = f"read_failed: {repr(e)}"

            manifest_rows.append(manifest)

            print(f"[跳过] 读取失败: {e}")

            continue

        manifest["n_rows_final"] = int(len(final_df))

        manifest["n_rows_raw"] = int(len(raw_df)) if raw_df is not None else 0

        if raw_df is None:

            manifest["skip_reason"] = "missing_raw_label_matrix"

            manifest_rows.append(manifest)

            print("[跳过] 缺少 raw label matrix")

            continue

        required_final_cols = {"time_sec", "meta_state"}

        missing_final = sorted(required_final_cols - set(final_df.columns))

        if missing_final:

            manifest["skip_reason"] = f"missing_final_columns: {missing_final}"

            manifest_rows.append(manifest)

            print(f"[跳过] final_df 缺少列: {missing_final}")

            continue

        if len(final_df) != len(raw_df):

            manifest["skip_reason"] = f"length_mismatch: final={len(final_df)}, raw={len(raw_df)}"

            manifest_rows.append(manifest)

            print(f"[跳过] 行数不一致: final={len(final_df)}, raw={len(raw_df)}")

            continue

        keep_mask = build_keep_mask(final_df)

        n_kept = int(keep_mask.sum())

        manifest["n_rows_kept"] = n_kept

        if n_kept <= 0:

            manifest["skip_reason"] = "no_rows_after_teacher_skip_filter"

            manifest_rows.append(manifest)

            print("[跳过] 过滤 SKIP/非教师区间后无数据")

            continue

        final_df = final_df.loc[keep_mask].reset_index(drop=True)

        raw_df = raw_df.loc[keep_mask].reset_index(drop=True)

        run_cols, run_info_use, run_col_source = select_run_columns(raw_df, run_info_df, branch_metrics_df)

        manifest["n_run_cols_used"] = int(len(run_cols))

        manifest["run_col_source"] = run_col_source

        if len(run_cols) == 0:

            manifest["skip_reason"] = "no_run_columns_after_selection"

            manifest_rows.append(manifest)

            print("[跳过] 没有可用分支列")

            continue

        meta_orig = pd.to_numeric(final_df["meta_state"], errors="coerce").astype(int).values

        meta_contig, state_map_old_to_new = remap_to_contiguous_labels(meta_orig)

        inverse_state_map = {new: old for old, new in state_map_old_to_new.items()}

        final_df["local_state"] = meta_contig

        local_states = sorted(pd.unique(final_df["local_state"]))

        n_total = len(final_df)

        segs_all = split_state_segments(meta_contig)

        local_count = 0

        for local_st in local_states:

            idx = np.where(meta_contig == int(local_st))[0]

            if len(idx) < int(MIN_FRAMES_PER_LOCAL_STATE):

                continue

            orig_st = int(inverse_state_map[int(local_st)])

            row = {

                "video_id": video_id,

                "case_dir": str(case_dir),

                "local_state": int(local_st),

                "local_state_original": orig_st,

                "num_frames": int(len(idx)),

                "state_ratio": float(len(idx) / n_total),

                "run_col_source": run_col_source,

                "n_run_cols_used": int(len(run_cols)),

            }

                                            

            seg_lens = [e - s + 1 for s, e, lab in segs_all if lab == int(local_st)]

            seg_lens = np.array(seg_lens, dtype=float) if len(seg_lens) > 0 else np.array([], dtype=float)

            row["seg_count"] = int(len(seg_lens))

            row["seg_len_mean"] = float(seg_lens.mean()) if len(seg_lens) > 0 else 0.0

            row["seg_len_std"] = float(seg_lens.std(ddof=0)) if len(seg_lens) > 0 else 0.0

            row["seg_len_max"] = float(seg_lens.max()) if len(seg_lens) > 0 else 0.0

                                                   

            target_col = f"meta_ratio_{orig_st}"

            if target_col in final_df.columns:

                conf = pd.to_numeric(final_df.loc[idx, target_col], errors="coerce")

                row["meta_conf_mean"] = float(conf.mean())

                row["meta_conf_std"] = float(conf.std(ddof=0))

            else:

                                                                                               

                fallback_col = f"meta_ratio_{int(local_st)}"

                if fallback_col in final_df.columns:

                    conf = pd.to_numeric(final_df.loc[idx, fallback_col], errors="coerce")

                    row["meta_conf_mean"] = float(conf.mean())

                    row["meta_conf_std"] = float(conf.std(ddof=0))

                else:

                    row["meta_conf_mean"] = np.nan

                    row["meta_conf_std"] = np.nan

                                                              

                                                                                              

                                                       

            for run_col in run_cols:

                run_seq = pd.to_numeric(raw_df[run_col], errors="coerce")

                run_sub = run_seq.iloc[idx].dropna().astype(int).values

                                                                                        

                run_sub = run_sub[run_sub >= 0]

                prefix = run_col.replace("_state", "")

                if len(run_sub) == 0:

                    row[f"{prefix}_unique_states"] = np.nan

                    row[f"{prefix}_dominant_ratio"] = np.nan

                    row[f"{prefix}_entropy"] = np.nan

                    continue

                vals, counts = np.unique(run_sub, return_counts=True)

                probs = counts.astype(np.float64) / counts.sum()

                row[f"{prefix}_unique_states"] = int(len(vals))

                row[f"{prefix}_dominant_ratio"] = float(probs.max())

                row[f"{prefix}_entropy"] = float(safe_normalized_entropy(run_sub))

            rows.append(row)

            local_count += 1

        manifest["n_local_states"] = int(local_count)

        if local_count == 0:

            manifest["skip_reason"] = "no_local_states_after_min_frame_filter"

        else:

            manifest["included"] = 1

        manifest_rows.append(manifest)

    manifest_df = pd.DataFrame(manifest_rows)

    manifest_df.to_csv(OUTPUT_ROOT / "case_processing_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"\n已输出: {OUTPUT_ROOT / 'case_processing_manifest.csv'}")

    proto_df = pd.DataFrame(rows)

    if len(proto_df) == 0:

        raise ValueError("没有成功提取任何 T2S prototype；请检查 manifest 中的 skip_reason")

    out_csv = OUTPUT_ROOT / "local_state_prototypes_pid_topk_strict.csv"

    proto_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"已输出: {out_csv}")

    return proto_df

                                                           

         

                                                           

def choose_feature_columns(proto_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

    id_cols = {

        "video_id",

        "case_dir",

        "local_state",

        "local_state_original",

        "run_col_source",

    }

    candidate_cols = [c for c in proto_df.columns if c not in id_cols]

    numeric = pd.DataFrame(index=proto_df.index)

    feature_cols = []

    for c in candidate_cols:

        s = pd.to_numeric(proto_df[c], errors="coerce")

        if s.notna().sum() == 0:

            continue

                      

        if s.dropna().nunique() <= 1:

            continue

        numeric[c] = s

        feature_cols.append(c)

    if not feature_cols:

        raise ValueError("没有可用的数值 feature columns")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    med = numeric.median(numeric_only=True)

    numeric = numeric.fillna(med).fillna(0.0)

    pd.DataFrame({"feature_col": feature_cols}).to_csv(

        OUTPUT_ROOT / "feature_cols_used.csv",

        index=False,

        encoding="utf-8-sig",

    )

    print(f"已输出: {OUTPUT_ROOT / 'feature_cols_used.csv'}")

    return numeric, feature_cols

def infer_global_k_range_from_local_median(proto_df: pd.DataFrame) -> tuple[int, int, int, pd.DataFrame]:

    

       

    if "video_id" not in proto_df.columns:

        raise ValueError("proto_df 缺少 video_id，无法根据单视频 local K 中位数确定 global K 范围")

    if "local_state_original" in proto_df.columns:

        local_k_df = (

            proto_df.groupby("video_id")["local_state_original"]

            .nunique()

            .reset_index(name="local_K")

        )

    elif "local_state" in proto_df.columns:

        local_k_df = (

            proto_df.groupby("video_id")["local_state"]

            .nunique()

            .reset_index(name="local_K")

        )

    else:

                                                                                 

        local_k_df = (

            proto_df.groupby("video_id")

            .size()

            .reset_index(name="local_K")

        )

    local_k_df["local_K"] = pd.to_numeric(local_k_df["local_K"], errors="coerce").fillna(0).astype(int)

    local_k_df = local_k_df.loc[local_k_df["local_K"].gt(0)].copy()

    if len(local_k_df) == 0:

        raise ValueError("没有有效的单视频 local_K，无法根据中位数确定 global K 范围")

    k_med = int(round(float(local_k_df["local_K"].median())))

    k_min = max(2, k_med - int(GLOBAL_K_MARGIN))

    k_max = min(len(proto_df) - 1, k_med + int(GLOBAL_K_MARGIN))

    if k_max < k_min:

        k_max = k_min

    local_k_df.to_csv(OUTPUT_ROOT / "local_k_per_video.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {OUTPUT_ROOT / 'local_k_per_video.csv'}")

    print(

        f"根据单视频 local_K 中位数限制 global K: "

        f"K_med={k_med}, range=[{k_min}, {k_max}], "

        f"margin={GLOBAL_K_MARGIN}"

    )

    return int(k_min), int(k_max), int(k_med), local_k_df

def choose_best_global_k(proto_df: pd.DataFrame):

    X, feature_cols = choose_feature_columns(proto_df)

    scaler = StandardScaler()

    X_std = scaler.fit_transform(X.values.astype(np.float32))

    n = len(proto_df)

    if n < 3:

        raise ValueError(f"prototype 数量太少，无法做 K sweep: n={n}")

    local_k_median = None

    if GLOBAL_K_FROM_LOCAL_MEDIAN:

        k_min, k_max, local_k_median, _local_k_df = infer_global_k_range_from_local_median(proto_df)

    else:

        k_max = min(GLOBAL_K_MAX, n - 1)

        k_min = min(GLOBAL_K_MIN, k_max)

    ks = list(range(int(k_min), int(k_max) + 1))

    if len(ks) == 0:

        raise ValueError(f"global K 搜索范围为空: k_min={k_min}, k_max={k_max}, n={n}")

    score_rows = []

    best_k = None

    best_score = -1e18

    for k in ks:

        try:

            try:

                clusterer = AgglomerativeClustering(

                    n_clusters=k,

                    metric="cosine",

                    linkage="average",

                )

            except TypeError:

                clusterer = AgglomerativeClustering(

                    n_clusters=k,

                    affinity="cosine",

                    linkage="average",

                )

            labels = clusterer.fit_predict(X_std)

            if len(np.unique(labels)) < 2:

                sil = -1.0

            else:

                sil = silhouette_score(X_std, labels, metric="cosine")

            score_rows.append({"K": int(k), "silhouette_cosine": float(sil)})

            if sil > best_score:

                best_score = sil

                best_k = int(k)

        except Exception as e:

            score_rows.append({"K": int(k), "silhouette_cosine": np.nan, "error": repr(e)})

    score_df = pd.DataFrame(score_rows)

    score_df["k_search_min"] = int(k_min)

    score_df["k_search_max"] = int(k_max)

    score_df["local_k_median"] = int(local_k_median) if local_k_median is not None else np.nan

    score_df["k_from_local_median"] = int(bool(GLOBAL_K_FROM_LOCAL_MEDIAN))

    score_df.to_csv(OUTPUT_ROOT / "global_k_sweep_scores.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {OUTPUT_ROOT / 'global_k_sweep_scores.csv'}")

    if best_k is None:

        raise ValueError("无法确定 global K")

    return best_k, score_df, X_std, feature_cols

def fit_global_clusters(proto_df: pd.DataFrame, best_k: int, X_std: np.ndarray):

    try:

        clusterer = AgglomerativeClustering(

            n_clusters=best_k,

            metric="cosine",

            linkage="average",

        )

    except TypeError:

        clusterer = AgglomerativeClustering(

            n_clusters=best_k,

            affinity="cosine",

            linkage="average",

        )

    labels = clusterer.fit_predict(X_std).astype(int)

    labels, _ = remap_to_contiguous_labels(labels)

    out_df = proto_df.copy()

    out_df["global_proto_id"] = labels

    centers = {}

    for g in sorted(pd.unique(labels)):

        idx = np.where(labels == g)[0]

        centers[int(g)] = X_std[idx].mean(axis=0)

    dists = []

    for i, g in enumerate(labels):

        d = np.linalg.norm(X_std[i] - centers[int(g)])

        dists.append(float(d))

    out_df["dist_to_cluster_center"] = dists

    out_df.to_csv(OUTPUT_ROOT / "local_to_global_proto_pid_topk_strict.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {OUTPUT_ROOT / 'local_to_global_proto_pid_topk_strict.csv'}")

    return out_df, centers

def summarize_global_clusters(aligned_df: pd.DataFrame):

    grp = aligned_df.groupby("global_proto_id")

    summary = grp.agg(

        support_local_states=("local_state", "count"),

        support_videos=("video_id", "nunique"),

        mean_state_ratio=("state_ratio", "mean"),

        mean_num_frames=("num_frames", "mean"),

        mean_seg_count=("seg_count", "mean"),

        mean_seg_len=("seg_len_mean", "mean"),

        mean_meta_conf=("meta_conf_mean", "mean"),

        mean_dist_to_center=("dist_to_cluster_center", "mean"),

    ).reset_index()

    def cluster_type(nv: int) -> str:

        if nv >= SHARED_MIN_SUPPORT_VIDEOS:

            return "shared"

        if nv >= VARIANT_MIN_SUPPORT_VIDEOS:

            return "variant"

        return "private"

    summary["cluster_type"] = summary["support_videos"].apply(cluster_type)

    aligned_df = aligned_df.merge(

        summary[["global_proto_id", "support_videos", "cluster_type"]],

        on="global_proto_id",

        how="left",

    )

    aligned_df.to_csv(OUTPUT_ROOT / "local_to_global_proto_pid_topk_strict.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(OUTPUT_ROOT / "global_proto_summary_pid_topk_strict.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {OUTPUT_ROOT / 'global_proto_summary_pid_topk_strict.csv'}")

    return aligned_df, summary

                                                           

        

                                                           

def plot_pca_clusters(aligned_df: pd.DataFrame, X_std: np.ndarray):

    pca = PCA(n_components=2, random_state=42)

    X_2d = pca.fit_transform(X_std)

    plot_df = aligned_df.copy()

    plot_df["pca_x"] = X_2d[:, 0]

    plot_df["pca_y"] = X_2d[:, 1]

    plt.figure(figsize=(12, 9))

    sc = plt.scatter(

        plot_df["pca_x"],

        plot_df["pca_y"],

        c=plot_df["global_proto_id"],

        cmap="tab20",

        s=55,

        alpha=0.85,

    )

    plt.colorbar(sc, label="global_proto_id")

    plt.xlabel("PCA-1")

    plt.ylabel("PCA-2")

    plt.title("PID-topK strict cross-video prototype alignment")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "pid_topk_strict_pca.png", dpi=220, bbox_inches="tight")

    plt.close()

    if SAVE_LABELED_PCA:

        plt.figure(figsize=(14, 10))

        sc = plt.scatter(

            plot_df["pca_x"],

            plot_df["pca_y"],

            c=plot_df["global_proto_id"],

            cmap="tab20",

            s=60,

            alpha=0.9,

        )

        plt.colorbar(sc, label="global_proto_id")

        n_points = len(plot_df)

        if n_points <= MAX_LABEL_POINTS:

            label_idx = range(n_points)

        else:

            label_idx = []

            for _g, sub in plot_df.groupby("global_proto_id"):

                sub = sub.sort_values("dist_to_cluster_center").head(5)

                label_idx.extend(sub.index.tolist())

        for i in label_idx:

            r = plot_df.loc[i]

            txt = f"{Path(str(r['video_id'])).parts[-1]}-s{int(r['local_state_original'])}"

            plt.text(r["pca_x"], r["pca_y"], txt, fontsize=7)

        plt.xlabel("PCA-1")

        plt.ylabel("PCA-2")

        plt.title("PID-topK strict clustering: labeled PCA view")

        plt.tight_layout()

        plt.savefig(PLOT_DIR / "pid_topk_strict_pca_labeled.png", dpi=220, bbox_inches="tight")

        plt.close()

    plot_df.to_csv(OUTPUT_ROOT / "pid_topk_strict_pca_coordinates.csv", index=False, encoding="utf-8-sig")

    print(f"已输出: {OUTPUT_ROOT / 'pid_topk_strict_pca_coordinates.csv'}")

    return plot_df

def plot_cluster_sizes(summary_df: pd.DataFrame) -> None:

    x = summary_df["global_proto_id"].astype(int).values

    y = summary_df["support_local_states"].astype(int).values

    plt.figure(figsize=(11, 6))

    plt.bar(x, y)

    for xi, yi in zip(x, y):

        plt.text(xi, yi + 0.1, str(int(yi)), ha="center", va="bottom", fontsize=8)

    plt.xlabel("global_proto_id")

    plt.ylabel("support_local_states")

    plt.title("PID-topK strict cluster sizes")

    plt.tight_layout()

    plt.savefig(PLOT_DIR / "pid_topk_strict_cluster_sizes.png", dpi=220, bbox_inches="tight")

    plt.close()

def plot_dendrogram_if_possible(X_std: np.ndarray, aligned_df: pd.DataFrame) -> None:

    if not SCIPY_OK:

        print("[提示] scipy 不可用，跳过树状图")

        return

    try:

        Z = linkage(X_std, method="average", metric="cosine")

        labels = [

            f"{Path(str(v)).parts[-1]}-s{int(s)}"

            for v, s in zip(aligned_df["video_id"], aligned_df["local_state_original"])

        ]

        plt.figure(figsize=(16, 8))

        dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=6)

        plt.title("PID-topK strict cross-video dendrogram")

        plt.tight_layout()

        plt.savefig(PLOT_DIR / "pid_topk_strict_dendrogram.png", dpi=220, bbox_inches="tight")

        plt.close()

    except Exception as e:

        print(f"[提示] 树状图绘制失败，已跳过: {e}")

                                                           

        

                                                           

def main() -> None:

    ensure_dirs()

    print("开始提取 PID-topK 严格对齐的 local state prototypes ...")

    proto_df = extract_local_state_prototypes_from_t2s()

    print("\n开始选择最优 global K ...")

    best_k, score_df, X_std, feature_cols = choose_best_global_k(proto_df)

    print(f"自动选择的 global K = {best_k}")

    print(score_df)

    print("\n开始全局聚类 ...")

    aligned_df, _centers = fit_global_clusters(proto_df, best_k, X_std)

    print("\n开始汇总 ...")

    aligned_df, summary_df = summarize_global_clusters(aligned_df)

    print("\n开始绘图 ...")

    plot_pca_clusters(aligned_df, X_std)

    plot_cluster_sizes(summary_df)

    plot_dendrogram_if_possible(X_std, aligned_df)

    cfg = {

        "T2S_OUTPUT_ROOT": str(T2S_OUTPUT_ROOT),

        "OUTPUT_ROOT": str(OUTPUT_ROOT),

        "GLOBAL_K_MIN": int(GLOBAL_K_MIN),

        "GLOBAL_K_MAX": int(GLOBAL_K_MAX),

        "GLOBAL_K_FROM_LOCAL_MEDIAN": bool(GLOBAL_K_FROM_LOCAL_MEDIAN),

        "GLOBAL_K_MARGIN": int(GLOBAL_K_MARGIN),

        "selected_global_K": int(best_k),

        "actual_k_search_min": int(score_df["k_search_min"].iloc[0]) if "k_search_min" in score_df.columns and len(score_df) else None,

        "actual_k_search_max": int(score_df["k_search_max"].iloc[0]) if "k_search_max" in score_df.columns and len(score_df) else None,

        "local_k_median": int(score_df["local_k_median"].dropna().iloc[0]) if "local_k_median" in score_df.columns and score_df["local_k_median"].notna().any() else None,

        "SHARED_MIN_SUPPORT_VIDEOS": int(SHARED_MIN_SUPPORT_VIDEOS),

        "VARIANT_MIN_SUPPORT_VIDEOS": int(VARIANT_MIN_SUPPORT_VIDEOS),

        "STRICT_TEACHER_ONLY": bool(STRICT_TEACHER_ONLY),

        "DROP_SKIP_META_STATE": bool(DROP_SKIP_META_STATE),

        "USE_SELECTED_BRANCHES_ONLY": bool(USE_SELECTED_BRANCHES_ONLY),

        "MIN_FRAMES_PER_LOCAL_STATE": int(MIN_FRAMES_PER_LOCAL_STATE),

        "feature_cols": feature_cols,

    }

    with open(OUTPUT_ROOT / "run_config.json", "w", encoding="utf-8") as f:

        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n全部完成。")

    print(f"输出目录: {OUTPUT_ROOT}")

    print("重点看：")

    print("  1) case_processing_manifest.csv")

    print("  2) local_state_prototypes_pid_topk_strict.csv")

    print("  3) local_to_global_proto_pid_topk_strict.csv")

    print("  4) global_proto_summary_pid_topk_strict.csv")

    print("  5) global_k_sweep_scores.csv")

    print("  6) plots/pid_topk_strict_pca.png")

    print("  7) plots/pid_topk_strict_pca_labeled.png")

if __name__ == "__main__":

    main()
