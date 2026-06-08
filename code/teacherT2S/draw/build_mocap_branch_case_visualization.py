# -*- coding: utf-8 -*-
"""
build_mocap_amc86_lowpid_gt_svg_v6_fix_branch_config.py

功能：
1. 重新跑 MoCap 的 amc_86_03.4d；
2. 保存全部 candidate branch 的 state；
3. 自动选择：
   - low-P branch
   - low-I branch
   - high-D-penalty branch（pid_d 最大）
4. 生成 5 行图：
   (a) low-P
   (b) low-I
   (c) high-D-penalty
   (d) meta-state
   (e) ground truth
5. 每个分支显示 P/I/D/R/ARI/NMI；meta 行显示 ARI/NMI。

关键修复：
    设置 T2S_USE_LOCAL_DEPS=0，避免 D:\code\teacherT2S\multi_t2s_paper_benchmark\_deps
    里的旧 sklearn/scipy 覆盖当前 conda 环境。

运行：
    conda deactivate
    conda activate multit2s_cuda
    cd /d D:\code\teacherT2S\draw
    python build_mocap_amc86_lowpid_gt_svg_v6_fix_branch_config.py
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

# ============================================================
# 0. 环境修复：必须在导入 runner/import_runtime 之前设置
# ============================================================

# 禁用 multi_t2s_paper_benchmark\_deps，避免里面的 cp312 sklearn/scipy
# 覆盖当前 Python 3.10 conda 环境。
os.environ["T2S_USE_LOCAL_DEPS"] = "0"

REQUIRED_CONDA_ENV = "multit2s_cuda"
current_env = os.environ.get("CONDA_DEFAULT_ENV", "")

if current_env != REQUIRED_CONDA_ENV:
    raise SystemExit(
        "\n[ERROR] 当前 Conda 环境不对。\n"
        f"当前环境: {current_env or '<unknown>'}\n"
        f"需要环境: {REQUIRED_CONDA_ENV}\n\n"
        "请执行：\n"
        "  conda deactivate\n"
        f"  conda activate {REQUIRED_CONDA_ENV}\n"
        "  cd /d D:\\code\\teacherT2S\\draw\n"
        "  python build_mocap_amc86_lowpid_gt_svg_v6_fix_branch_config.py\n"
    )

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 基本路径与参数
# ============================================================

REPO_ROOT = Path(r"D:\code\teacherT2S")
DRAW_DIR = REPO_ROOT / "draw"
OUR_ROOT = REPO_ROOT / "our"
OUR_MOCAP_DIR = OUR_ROOT / "mocap"
RESULT_DIR = OUR_MOCAP_DIR / "results_mocap_our"
PUBLIC_DATA_ROOT = REPO_ROOT / r"Time2State\Baselines\public_ts_datasets"

# 注意：共享 runner 在 our\_shared，不在 our\mocap\_shared。
RUNNER_PATH = OUR_ROOT / "_shared" / "our_multit2s_runner.py"
CONFIG_PATH = OUR_MOCAP_DIR / "mocap_config.txt"

CASE_ID = "amc_86_03.4d"
DATASET_NAME = "mocap"

SEED = 2125282736
DEVICE = "cuda"
GPU = 0
M = 10
N = 4
OUT_CHANNELS = 4
NB_STEPS = 20
META_MIN_LEN = 24
STATE_CLUSTER_SMOOTH = 9
PUBLIC_MAX_ROWS = 12000

SELECT_TOP_K = 16
BRANCH_SELECT_METRIC = "PID"
META_VOTE_WEIGHT_MODE = "pid_weight"

PID_KP = 0.45
PID_KI = 0.35
PID_KD = 0.20
PID_SOFTMAX_TAU = 0.15
PEER_HEALTH_WEIGHT = 0.45
PEER_CONSENSUS_WEIGHT = 0.55
MAX_SERIES_PER_DATASET = 9

READ_RATIO = 1.0
REMAP_STATE_LABELS = True
SAVE_DPI = 1200
FIG_W = 3.45
FONT_SIZE = 6.5
LINE_WIDTH = 0.75

OUT_STEM = DRAW_DIR / f"mocap_{CASE_ID.replace('.', '_')}_lowP_lowI_highD_meta_gt"
ALL_STATES_CSV = DRAW_DIR / f"mocap_{CASE_ID.replace('.', '_')}_all_candidate_states.csv"
BRANCH_METRICS_CSV = DRAW_DIR / f"mocap_{CASE_ID.replace('.', '_')}_branch_metrics.csv"
SELECTED_SUMMARY_CSV = DRAW_DIR / f"mocap_{CASE_ID.replace('.', '_')}_selected_branches.csv"
TEMP_BRANCH_TABLE = DRAW_DIR / "_tmp_mocap_branch_table.csv"


# ============================================================
# 2. 工具函数
# ============================================================

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 0.5,
        "ytick.labelsize": FONT_SIZE - 0.5,
        "axes.linewidth": 0.45,
        "lines.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def assert_required_paths() -> None:
    missing = []
    for p in [REPO_ROOT, DRAW_DIR, OUR_ROOT, OUR_MOCAP_DIR, PUBLIC_DATA_ROOT, RUNNER_PATH, CONFIG_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("以下路径不存在：\n" + "\n".join(missing))


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_branch_table(config_path: Path, out_csv: Path) -> Path:
    text = config_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "[branches]":
            start = i + 1
            break
    if start is None:
        raise ValueError(f"找不到 [branches] 段：{config_path}")

    kept = []
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        kept.append(line)

    if not kept:
        raise ValueError(f"[branches] 段为空：{config_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(kept) + "\n", encoding="utf-8-sig")
    return out_csv


def parse_runner_args(runner_mod, argv: list[str]):
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        args = runner_mod.parse_args()
    finally:
        sys.argv = old_argv
    return args


def remap_dense_by_first_appearance(seq: np.ndarray) -> np.ndarray:
    mapping = {}
    out = []
    nxt = 0
    for x in seq:
        x = int(x)
        if x not in mapping:
            mapping[x] = nxt
            nxt += 1
        out.append(mapping[x])
    return np.asarray(out, dtype=int)


def choose_one_by_metric(df: pd.DataFrame, metric: str, ascending: bool, used: set[str]) -> pd.Series:
    sub = df.sort_values([metric, "branch"], ascending=[ascending, True]).reset_index(drop=True)
    for _, row in sub.iterrows():
        b = str(row["branch"])
        if b not in used:
            used.add(b)
            return row
    raise RuntimeError(f"无法按 {metric} 选择分支。")


# ============================================================
# 3. 跑单个 MoCap case
# ============================================================

def run_single_case_and_collect():
    print(f"[INFO] importing runner: {RUNNER_PATH}", flush=True)
    runner = import_module_from_path("our_multit2s_runner_local", RUNNER_PATH)

    # 给 runner 的单分支函数打一个轻量进度补丁。
    # 不改算法，只打印每个分支开始/结束，避免长时间看起来像卡死。
    _orig_run_one_t2s_branch = runner.run_one_t2s_branch
    def _run_one_t2s_branch_with_progress(data, branch, args, runtime, fit_data=None):
        import time
        bname = getattr(branch, "branch_name", "")
        win = getattr(branch, "win", "")
        step = getattr(branch, "step", "")
        print(f"[BRANCH START] {bname} win={win} step={step}", flush=True)
        t0 = time.time()
        seq = _orig_run_one_t2s_branch(data, branch, args, runtime, fit_data=fit_data)
        print(f"[BRANCH DONE ] {bname} elapsed={time.time() - t0:.1f}s", flush=True)
        return seq
    runner.run_one_t2s_branch = _run_one_t2s_branch_with_progress

    print("[INFO] extracting branch table...", flush=True)
    branch_csv = extract_branch_table(CONFIG_PATH, TEMP_BRANCH_TABLE)

    argv = [
        str(RUNNER_PATH),
        "--repo-root", str(REPO_ROOT),
        "--out-dir", str(RESULT_DIR),
        "--datasets", DATASET_NAME,
        "--branch-config-txt", str(branch_csv),
        "--public-data-root", str(PUBLIC_DATA_ROOT),
        "--max-series-per-dataset", str(MAX_SERIES_PER_DATASET),
        "--only-case-ids", CASE_ID,
        "--seed", str(SEED),
        "--device", DEVICE,
        "--gpu", str(GPU),
        "--m", str(M),
        "--n", str(N),
        "--out-channels", str(OUT_CHANNELS),
        "--nb-steps", str(NB_STEPS),
        "--meta-min-len", str(META_MIN_LEN),
        "--state-cluster-smooth", str(STATE_CLUSTER_SMOOTH),
        "--public-max-rows", str(PUBLIC_MAX_ROWS),
        "--select-top-k-branches", str(SELECT_TOP_K),
        "--branch-select-metric", BRANCH_SELECT_METRIC,
        "--meta-vote-weight-mode", META_VOTE_WEIGHT_MODE,
        "--pid-kp", str(PID_KP),
        "--pid-ki", str(PID_KI),
        "--pid-kd", str(PID_KD),
        "--pid-softmax-tau", str(PID_SOFTMAX_TAU),
        "--peer-health-weight", str(PEER_HEALTH_WEIGHT),
        "--peer-consensus-weight", str(PEER_CONSENSUS_WEIGHT),
    ]

    print("[INFO] parsing runner args...", flush=True)
    args = parse_runner_args(runner, argv)
    args.repo_root = args.repo_root.resolve()
    args.out_dir = args.out_dir.resolve()
    runner.validate_clean_evaluation_args(args)

    # IMPORTANT:
    # 直接调用 run_multit2s_case 时，必须手动加载 branch_config_map。
    # runner.main() 内部本来会做这一步；但本脚本绕过 main() 调函数，
    # 如果漏掉，就会回退到默认 6 个 branches：
    # 128:50,128:100,256:50,256:100,512:50,512:100。
    args.branch_config_map = runner.load_branch_config_txt(args.branch_config_txt, args)
    if args.branch_config_map:
        print("[INFO] branch-config loaded:", {k: len(v) for k, v in args.branch_config_map.items()}, flush=True)
    else:
        raise RuntimeError("branch_config_map is empty; the script would fall back to default 6 branches.")

    random.seed(args.seed)
    print("[INFO] importing runtime packages: numpy/pandas/sklearn/torch/scipy/Time2State...", flush=True)
    runtime = runner.import_runtime(args.repo_root)
    print("[INFO] runtime imported OK", flush=True)
    runtime["np"].random.seed(args.seed)
    runtime["torch"].manual_seed(args.seed)

    print("[INFO] loading MoCap cases...", flush=True)
    cases, _dataset_status = runner.load_cases(args, runtime)
    print(f"[INFO] loaded cases before filter: {len(cases)}", flush=True)
    cases = runner.apply_case_selection(cases, args)
    print(f"[INFO] cases after filter: {len(cases)}", flush=True)
    if not cases:
        raise RuntimeError(f"没有加载到 case={CASE_ID}")

    target_case = None
    for case in cases:
        if str(case.case_id) == CASE_ID:
            target_case = case
            break
    if target_case is None:
        raise RuntimeError(f"找不到 case={CASE_ID}")

    print(f"[INFO] running: {target_case.dataset}/{target_case.case_id}", flush=True)
    print("[INFO] now running 64 candidate branches; this can take a long time.", flush=True)
    result = runner.run_multit2s_case(target_case, args, runtime)
    runner.save_case_outputs(target_case, result, args, runtime)

    return runner, result


# ============================================================
# 4. 保存全部 state 和 metrics
# ============================================================

def save_all_candidate_outputs(result):
    DRAW_DIR.mkdir(parents=True, exist_ok=True)

    all_state_df = pd.DataFrame({
        "true_label": np.asarray(result["labels"], dtype=int),
        "meta_state": np.asarray(result["meta_seq"], dtype=int),
    })

    for name, _win, _step, seq in result["candidate_branch_sequences"]:
        all_state_df[name] = np.asarray(seq, dtype=int)

    all_state_df.to_csv(ALL_STATES_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved all states : {ALL_STATES_CSV}")

    metrics_df = pd.DataFrame(result["branch_metrics"]).copy()
    metrics_df.to_csv(BRANCH_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved metrics    : {BRANCH_METRICS_CSV}")

    return all_state_df, metrics_df


def pick_representative_branches(metrics_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["branch", "pid_p", "pid_i", "pid_d", "pid_score_norm", "ARI", "NMI"]
    missing = [c for c in needed if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"branch metrics 缺少列：{missing}")

    used = set()
    low_p = choose_one_by_metric(metrics_df, "pid_p", ascending=True, used=used)
    low_i = choose_one_by_metric(metrics_df, "pid_i", ascending=True, used=used)
    high_d = choose_one_by_metric(metrics_df, "pid_d", ascending=False, used=used)

    out = pd.DataFrame([
        {"panel": "(a) Low-$P$ branch", **low_p.to_dict()},
        {"panel": "(b) Low-$I$ branch", **low_i.to_dict()},
        {"panel": "(c) High-$D$-penalty branch", **high_d.to_dict()},
    ])
    out.to_csv(SELECTED_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved selected   : {SELECTED_SUMMARY_CSV}")
    print(out[["panel", "branch", "pid_p", "pid_i", "pid_d", "pid_score_norm", "ARI", "NMI"]].to_string(index=False))
    return out


# ============================================================
# 5. 绘图
# ============================================================

def draw_state_only_figure(all_state_df: pd.DataFrame, selected_df: pd.DataFrame, result: dict) -> None:
    n_total = len(all_state_df)
    ratio = READ_RATIO if READ_RATIO <= 1 else READ_RATIO / 100.0
    ratio = min(max(ratio, 1e-8), 1.0)
    keep_n = max(1, int(round(n_total * ratio)))

    x = np.arange(keep_n, dtype=float)

    gt = all_state_df["true_label"].to_numpy(dtype=int)[:keep_n]
    meta = all_state_df["meta_state"].to_numpy(dtype=int)[:keep_n]

    meta_ari = float(result.get("ARI", np.nan))
    meta_nmi = float(result.get("NMI", np.nan))

    if REMAP_STATE_LABELS:
        gt_show = remap_dense_by_first_appearance(gt)
        meta_show = remap_dense_by_first_appearance(meta)
    else:
        gt_show = gt
        meta_show = meta

    branch_seqs = []
    branch_meta = []
    for _, row in selected_df.iterrows():
        branch = str(row["branch"])
        seq = all_state_df[branch].to_numpy(dtype=int)[:keep_n]
        if REMAP_STATE_LABELS:
            seq = remap_dense_by_first_appearance(seq)
        branch_seqs.append(seq)
        branch_meta.append(row)

    nrows = 5
    fig_h = 0.78 * nrows + 0.28
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(FIG_W, fig_h), sharex=True)
    axes = list(axes)

    for ax, row, seq in zip(axes[:3], branch_meta, branch_seqs):
        ax.step(x, seq, where="post", linewidth=LINE_WIDTH)
        title = (
            f"{row['panel']}  {row['branch']}\n"
            f"P={float(row['pid_p']):.3f}, "
            f"I={float(row['pid_i']):.3f}, "
            f"D={float(row['pid_d']):.3f}, "
            f"R={float(row['pid_score_norm']):.3f}, "
            f"ARI={float(row['ARI']):.3f}, "
            f"NMI={float(row['NMI']):.3f}"
        )
        ax.set_title(title, loc="left", pad=1.5)
        ax.set_ylabel("state", labelpad=2)
        ax.tick_params(axis="y", left=True, labelleft=False, length=2, width=0.4)
        ax.tick_params(axis="x", length=2, width=0.4)
        ax.margins(x=0.002, y=0.08)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)

    ax = axes[3]
    ax.step(x, meta_show, where="post", linewidth=LINE_WIDTH)
    ax.set_title(
        f"(d) Meta-state fusion\nARI={meta_ari:.3f}, NMI={meta_nmi:.3f}",
        loc="left",
        pad=1.5
    )
    ax.set_ylabel("state", labelpad=2)
    ax.tick_params(axis="y", left=True, labelleft=False, length=2, width=0.4)
    ax.tick_params(axis="x", length=2, width=0.4)
    ax.margins(x=0.002, y=0.08)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)

    ax = axes[4]
    ax.step(x, gt_show, where="post", linewidth=LINE_WIDTH)
    ax.set_title("(e) Ground truth", loc="left", pad=1.5)
    ax.set_ylabel("state", labelpad=2)
    ax.set_xlabel("time")
    ax.tick_params(axis="y", left=True, labelleft=False, length=2, width=0.4)
    ax.tick_params(axis="x", length=2, width=0.4)
    ax.margins(x=0.002, y=0.08)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.988, bottom=0.07, hspace=0.88)

    for fmt in ["svg", "png", "pdf"]:
        out_path = OUT_STEM.with_suffix(f".{fmt}")
        if fmt == "svg":
            fig.savefig(out_path, format="svg")
        else:
            fig.savefig(out_path, dpi=SAVE_DPI)
        print(f"[OK] saved figure     : {out_path}")

    plt.close(fig)


def main() -> None:
    print(f"[INFO] conda env: {current_env}", flush=True)
    print("[INFO] T2S_USE_LOCAL_DEPS =", os.environ.get("T2S_USE_LOCAL_DEPS"), flush=True)
    print("[INFO] checking paths...", flush=True)
    assert_required_paths()
    print("[INFO] paths OK; setting matplotlib...", flush=True)
    setup_matplotlib()

    _runner, result = run_single_case_and_collect()
    all_state_df, metrics_df = save_all_candidate_outputs(result)
    selected_df = pick_representative_branches(metrics_df)
    draw_state_only_figure(all_state_df, selected_df, result)

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
