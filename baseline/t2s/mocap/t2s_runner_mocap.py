from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path



MOCAP_INFO = {
    "amc_86_01.4d": {"n_segs": 4, "label": {588: 0, 1200: 1, 2006: 0, 2530: 2, 3282: 0, 4048: 3, 4579: 2}},
    "amc_86_02.4d": {"n_segs": 8, "label": {1009: 0, 1882: 1, 2677: 2, 3158: 3, 4688: 4, 5963: 0, 7327: 5, 8887: 6, 9632: 7, 10617: 0}},
    "amc_86_03.4d": {"n_segs": 7, "label": {872: 0, 1938: 1, 2448: 2, 3470: 0, 4632: 3, 5372: 4, 6182: 5, 7089: 6, 8401: 0}},
    "amc_86_07.4d": {"n_segs": 6, "label": {1060: 0, 1897: 1, 2564: 2, 3665: 1, 4405: 2, 5169: 3, 5804: 4, 6962: 0, 7806: 5, 8702: 0}},
    "amc_86_08.4d": {"n_segs": 9, "label": {1062: 0, 1904: 1, 2661: 2, 3282: 3, 3963: 4, 4754: 5, 5673: 6, 6362: 4, 7144: 7, 8139: 8, 9206: 0}},
    "amc_86_09.4d": {"n_segs": 5, "label": {921: 0, 1275: 1, 2139: 2, 2887: 3, 3667: 4, 4794: 0}},
    "amc_86_10.4d": {"n_segs": 4, "label": {2003: 0, 3720: 1, 4981: 0, 5646: 2, 6641: 3, 7583: 0}},
    "amc_86_11.4d": {"n_segs": 4, "label": {1231: 0, 1693: 1, 2332: 2, 2762: 1, 3386: 3, 4015: 2, 4665: 1, 5674: 0}},
    "amc_86_14.4d": {"n_segs": 3, "label": {671: 0, 1913: 1, 2931: 0, 4134: 2, 5051: 0, 5628: 1, 6055: 2}},
}


def add_project_imports(repo_root: Path) -> None:
    """Add project imports without using bundled binary dependencies.

    Important on Windows: multi_t2s_paper_benchmark/_deps may contain a
    NumPy wheel compiled for a different Python version, e.g. cp312, while the
    conda environment may run Python 3.10. Putting that _deps directory before
    site-packages causes ImportError: numpy._core._multiarray_umath.

    Therefore this baseline intentionally uses NumPy/Pandas/SciPy/Torch from
    the active conda environment, and only adds source-code directories.
    """
    blocked = str(repo_root / "multi_t2s_paper_benchmark" / "_deps").lower().replace("/", "\\")
    cleaned = []
    for item in sys.path:
        norm = str(item).lower().replace("/", "\\")
        if norm == blocked or norm.startswith(blocked + "\\"):
            continue
        cleaned.append(item)
    sys.path[:] = cleaned

    candidates = [
        repo_root / "Time2State",
        repo_root,
        repo_root / "multi_t2s_paper_benchmark",
    ]
    for p in reversed(candidates):
        if p.exists():
            s = str(p)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)


def set_seeds(seed: int, torch_obj=None) -> None:
    random.seed(int(seed))
    try:
        import numpy as _np
        _np.random.seed(int(seed))
    except Exception:
        pass
    if torch_obj is not None:
        try:
            torch_obj.manual_seed(int(seed))
            if torch_obj.cuda.is_available():
                torch_obj.cuda.manual_seed_all(int(seed))
        except Exception:
            pass


def parse_int_list(text: str | int) -> list[int]:
    if isinstance(text, int):
        return [int(text)]
    vals = []
    for part in str(text).replace(",", " ").replace(";", " ").split():
        part = part.strip()
        if part:
            vals.append(int(float(part)))
    if not vals:
        raise ValueError(f"Empty integer list: {text!r}")
    return vals


def adjusted_rand_index(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_true, labels_pred))
    except Exception:
        y = list(labels_true)
        z = list(labels_pred)
        if len(y) != len(z):
            raise ValueError("ARI inputs must have equal length")
        if len(y) < 2:
            return 1.0

        def comb2(n: int) -> float:
            return 0.0 if n < 2 else n * (n - 1) / 2.0

        n = len(y)
        contingency = defaultdict(int)
        y_counts = Counter()
        z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1
            z_counts[int(b)] += 1
        sum_cells = sum(comb2(v) for v in contingency.values())
        sum_y = sum(comb2(v) for v in y_counts.values())
        sum_z = sum(comb2(v) for v in z_counts.values())
        total = comb2(n)
        expected = (sum_y * sum_z) / total if total else 0.0
        max_index = 0.5 * (sum_y + sum_z)
        denom = max_index - expected
        if denom == 0:
            return 1.0 if sum_cells == max_index else 0.0
        return float((sum_cells - expected) / denom)


def normalized_mutual_information(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_true, labels_pred, average_method="geometric"))
    except Exception:
        y = list(labels_true)
        z = list(labels_pred)
        if len(y) != len(z):
            raise ValueError("NMI inputs must have equal length")
        if len(y) == 0:
            return 0.0
        n = len(y)
        contingency = defaultdict(int)
        y_counts = Counter()
        z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1
            z_counts[int(b)] += 1
        mi = 0.0
        for (a, b), c in contingency.items():
            mi += (c / n) * math.log((c * n) / (y_counts[a] * z_counts[b]))

        def entropy(counts):
            out = 0.0
            for c in counts:
                p = c / n
                if p > 0:
                    out -= p * math.log(p)
            return out

        hy = entropy(y_counts.values())
        hz = entropy(z_counts.values())
        if hy == 0.0 and hz == 0.0:
            return 1.0
        if hy == 0.0 or hz == 0.0:
            return 0.0
        return float(mi / math.sqrt(hy * hz))


def adjusted_mutual_information(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import adjusted_mutual_info_score
        return float(adjusted_mutual_info_score(labels_true, labels_pred, average_method="arithmetic"))
    except Exception:
        return float("nan")


def segments_from_labels(seq) -> list[tuple[int, int, int]]:
    seq = list(map(int, seq))
    if not seq:
        return []
    out = []
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[start]:
            out.append((start, i, seq[start]))
            start = i
    out.append((start, len(seq), seq[start]))
    return out


def segmentation_covering(labels_true, labels_pred) -> float:
    """Weighted segment covering using max IoU between each true segment and predicted segments."""
    true_segments = segments_from_labels(labels_true)
    pred_segments = segments_from_labels(labels_pred)
    n = len(labels_true)
    if n == 0:
        return 0.0
    if not true_segments or not pred_segments:
        return 0.0

    total = 0.0
    for ts, te, _ in true_segments:
        best = 0.0
        for ps, pe, _ in pred_segments:
            inter = max(0, min(te, pe) - max(ts, ps))
            if inter <= 0:
                continue
            union = max(te, pe) - min(ts, ps)
            best = max(best, inter / union)
        total += (te - ts) * best
    return float(total / n)


def change_points(seq) -> list[int]:
    seq = list(map(int, seq))
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


def cp_f1(labels_true, labels_pred, margin: int) -> float:
    true_cps = change_points(labels_true)
    pred_cps = change_points(labels_pred)
    if not true_cps and not pred_cps:
        return 1.0
    if not true_cps or not pred_cps:
        return 0.0

    used = set()
    tp = 0
    for p in pred_cps:
        best_j = None
        best_dist = None
        for j, t in enumerate(true_cps):
            if j in used:
                continue
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_j = j
        if best_j is not None:
            used.add(best_j)
            tp += 1

    precision = tp / len(pred_cps) if pred_cps else 0.0
    recall = tp / len(true_cps) if true_cps else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def seg_to_label(seg_info: dict[int, int], np):
    labels = []
    start = 0
    for end in sorted(seg_info):
        end = int(end)
        if end < start:
            continue
        labels.extend([int(seg_info[end])] * (end - start))
        start = end
    return np.asarray(labels, dtype=int)


def align_sequence(seq, target_len: int, np):
    seq = np.asarray(seq, dtype=int)
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) == 0:
        return np.zeros(target_len, dtype=int)
    pad = np.full(target_len - len(seq), int(seq[-1]), dtype=int)
    return np.concatenate([seq, pad])


def filter_lse_params(params: dict) -> dict:
    try:
        from Time2State import encoders
        sig = inspect.signature(encoders.CausalConv_LSE.__init__)
        allowed = {name for name in sig.parameters if name != "self"}
        return {k: v for k, v in params.items() if k in allowed}
    except Exception:
        return params


def find_mocap_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "Time2State" / "data" / "MoCap" / "4d",
        repo_root / "Time2State" / "Baselines" / "public_ts_datasets" / "MoCap" / "4d",
        repo_root / "multi_t2s_paper_benchmark" / "public_ts_datasets" / "MoCap" / "4d",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find MoCap/4d directory. Tried: " + " | ".join(str(x) for x in candidates))


def load_mocap_cases(repo_root: Path, np, pd, max_cases: int | None = None):
    base = find_mocap_dir(repo_root)
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases = []
    for path in files:
        if path.name not in MOCAP_INFO:
            continue



        df = pd.read_csv(path, sep=" ", usecols=range(0, 4))
        data = df.to_numpy()
        labels = seg_to_label(MOCAP_INFO[path.name]["label"], np)[:-1]
        n = min(len(data), len(labels))
        cases.append((path.name, data[:n], labels[:n]))

        if max_cases is not None and len(cases) >= max_cases:
            break

    if not cases:
        raise FileNotFoundError(f"No strict MoCap .4d cases found under {base}")
    return base, cases


def run_t2s_on_one_case(data, labels, *, win_size: int, step: int, args, case_seed: int, runtime):
    np = runtime["np"]
    torch = runtime["torch"]
    Time2State = runtime["Time2State"]
    CausalConv_LSE_Adaper = runtime["CausalConv_LSE_Adaper"]
    DPGMM = runtime["DPGMM"]
    params_LSE = runtime["params_LSE"]

    if len(data) < win_size:
        raise ValueError(f"rows={len(data)} is shorter than win_size={win_size}")

    set_seeds(case_seed, torch)

    params = dict(params_LSE)
    params["in_channels"] = int(data.shape[1])
    params["compared_length"] = int(win_size)
    params["win_size"] = int(win_size)
    params["out_channels"] = int(args.out_channels)
    params["M"] = int(args.m)
    params["N"] = int(args.n)
    params["nb_steps"] = int(args.nb_steps)
    params["win_type"] = str(args.win_type)
    params["cuda"] = bool(torch.cuda.is_available() and args.device != "cpu")
    params["gpu"] = int(args.gpu)
    if args.kernel_size is not None:
        params["kernel_size"] = int(args.kernel_size)
    params = filter_lse_params(params)

    model = Time2State(win_size, step, CausalConv_LSE_Adaper(params), DPGMM(None))

    t0 = time.time()
    model.fit(data, win_size, step)
    seconds = time.time() - t0

    pred = align_sequence(model.state_seq, len(labels), np)
    return pred, seconds


def run_t2s_mocap(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    add_project_imports(repo_root)

    import numpy as np
    import pandas as pd
    import torch
    from Time2State.adapers import CausalConv_LSE_Adaper
    from Time2State.clustering import DPGMM
    from Time2State.default_params import params_LSE
    from Time2State.time2state import Time2State

    runtime = {
        "np": np,
        "pd": pd,
        "torch": torch,
        "CausalConv_LSE_Adaper": CausalConv_LSE_Adaper,
        "DPGMM": DPGMM,
        "params_LSE": params_LSE,
        "Time2State": Time2State,
    }

    base, cases = load_mocap_cases(repo_root, np, pd, max_cases=args.max_cases)
    win_sizes = parse_int_list(args.win_size)
    steps = parse_int_list(args.step)
    grid = [(w, s) for w in win_sizes for s in steps]

    print("============================================================")
    print("Time2State baseline on MoCap")
    print("Protocol : fit Time2State separately on each MoCap sequence; no multi-branch, no PEER/PID, no meta clustering")
    print(f"Repo root: {repo_root}")
    print(f"Data dir : {base}")
    print(f"Output   : {out_dir}")
    print(f"Cases    : {len(cases)}")
    print(f"Grid     : {grid}")
    print(f"M/N      : {args.m}/{args.n}")
    print(f"Device   : {args.device}, gpu={args.gpu}")
    print("============================================================")

    if args.dry_run:
        return 0

    all_rows = []
    start_all = time.time()

    for grid_idx, (win_size, step) in enumerate(grid, start=1):
        print(f"\n[GRID {grid_idx}/{len(grid)}] win_size={win_size}, step={step}")
        for case_idx, (case_id, data, labels) in enumerate(cases, start=1):

            case_seed = int(args.seed) + grid_idx * 100000 + case_idx
            try:
                pred, seconds = run_t2s_on_one_case(
                    data,
                    labels,
                    win_size=int(win_size),
                    step=int(step),
                    args=args,
                    case_seed=case_seed,
                    runtime=runtime,
                )
                error = ""
            except Exception as exc:
                pred = np.zeros(len(labels), dtype=int)
                seconds = float("nan")
                error = repr(exc)

            ari = adjusted_rand_index(labels, pred) if not error else float("nan")
            nmi = normalized_mutual_information(labels, pred) if not error else float("nan")
            ami = adjusted_mutual_information(labels, pred) if not error else float("nan")
            covering = segmentation_covering(labels, pred) if not error else float("nan")
            margin = max(1, int(round(len(labels) * float(args.cp_margin_ratio))))
            f1 = cp_f1(labels, pred, margin) if not error else float("nan")

            k_pred = int(len(np.unique(pred))) if not error else 0
            k_true = int(len(np.unique(labels)))

            safe_case = case_id.replace("/", "_").replace("\\", "_").replace(".", "_")
            np.save(pred_dir / f"Time2State_MoCap_{safe_case}_w{win_size}_s{step}_labels_pred.npy", np.vstack([labels, pred]))

            row = {
                "algorithm": "Time2State",
                "dataset": "MoCap",
                "case_id": case_id,
                "rows": int(len(labels)),
                "features": int(data.shape[1]),
                "true_states": k_true,
                "pred_states": k_pred,
                "ARI": ari,
                "NMI": nmi,
                "AMI": ami,
                "Covering": covering,
                "CP_F1": f1,
                "CP_margin": margin,
                "seconds": seconds,
                "win_size": int(win_size),
                "step": int(step),
                "m": int(args.m),
                "n": int(args.n),
                "out_channels": int(args.out_channels),
                "nb_steps": int(args.nb_steps),
                "win_type": str(args.win_type),
                "seed": int(case_seed),
                "protocol": "fit_each_case_predict_same_case",
                "error": error,
            }
            all_rows.append(row)

            if error:
                print(f"  MoCap/{case_id}: ERROR {error}")
            else:
                print(f"  MoCap/{case_id}: ARI={ari:.4f} NMI={nmi:.4f} AMI={ami:.4f} Covering={covering:.4f} K={k_pred} seconds={seconds:.1f}")

    case_csv = out_dir / "case_results.csv"
    with case_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = []
    for win_size, step in grid:
        rows = [r for r in all_rows if r["win_size"] == win_size and r["step"] == step and not r["error"]]
        if rows:
            mean_ari = float(np.mean([r["ARI"] for r in rows]))
            mean_nmi = float(np.mean([r["NMI"] for r in rows]))
            mean_ami = float(np.mean([r["AMI"] for r in rows]))
            mean_covering = float(np.mean([r["Covering"] for r in rows]))
            mean_f1 = float(np.mean([r["CP_F1"] for r in rows]))
            total_seconds = float(np.nansum([r["seconds"] for r in rows]))
        else:
            mean_ari = mean_nmi = mean_ami = mean_covering = mean_f1 = float("nan")
            total_seconds = float("nan")

        summaries.append({
            "algorithm": "Time2State",
            "dataset": "MoCap",
            "cases": len(rows),
            "mean_ARI": mean_ari,
            "mean_NMI": mean_nmi,
            "mean_AMI": mean_ami,
            "mean_Covering": mean_covering,
            "mean_CP_F1": mean_f1,
            "total_seconds": total_seconds,
            "win_size": int(win_size),
            "step": int(step),
            "m": int(args.m),
            "n": int(args.n),
            "out_channels": int(args.out_channels),
            "nb_steps": int(args.nb_steps),
            "win_type": str(args.win_type),
            "protocol": "fit_each_case_predict_same_case",
            "metric_backend": "sklearn_ARI__sklearn_NMI_geometric__sklearn_AMI_arithmetic__segment_covering__cp_f1_margin_1pct",
        })

    summary_csv = out_dir / "algorithm_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    valid_summaries = [s for s in summaries if not math.isnan(s["mean_ARI"])]
    best_by_ari = max(valid_summaries, key=lambda x: x["mean_ARI"]) if valid_summaries else None
    best_by_nmi = max(valid_summaries, key=lambda x: x["mean_NMI"]) if valid_summaries else None

    if best_by_ari is not None:
        best_rows = [
            r for r in all_rows
            if r["win_size"] == best_by_ari["win_size"] and r["step"] == best_by_ari["step"]
        ]
        best_csv = out_dir / "best_by_mean_ARI_case_results.csv"
        with best_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
    else:
        best_csv = None

    status = {
        "ok": True,
        "algorithm": "Time2State",
        "dataset": "MoCap",
        "repo_root": str(repo_root),
        "data_dir": str(base),
        "out_dir": str(out_dir),
        "case_results_csv": str(case_csv),
        "algorithm_summary_csv": str(summary_csv),
        "best_by_mean_ARI_case_results_csv": str(best_csv) if best_csv else "",
        "best_by_mean_ARI": best_by_ari,
        "best_by_mean_NMI": best_by_nmi,
        "grid": [{"win_size": w, "step": s} for w, s in grid],
        "total_wall_seconds": float(time.time() - start_all),
        "protocol": "fit Time2State separately on each MoCap sequence; strict MoCap 4d loader; no normalization; no multi-branch/meta stage",
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n============================================================")
    print("SUMMARY")
    for s in summaries:
        print(
            f"w={s['win_size']:>3}, step={s['step']:>3} | "
            f"ARI={s['mean_ARI']:.6f}, NMI={s['mean_NMI']:.6f}, "
            f"AMI={s['mean_AMI']:.6f}, Covering={s['mean_Covering']:.6f}, CP_F1={s['mean_CP_F1']:.6f}"
        )
    if best_by_ari:
        print(
            f"BEST by mean ARI: w={best_by_ari['win_size']}, step={best_by_ari['step']}, "
            f"ARI={best_by_ari['mean_ARI']:.6f}, NMI={best_by_ari['mean_NMI']:.6f}"
        )
    print(f"case_results.csv      : {case_csv}")
    print(f"algorithm_summary.csv : {summary_csv}")
    if best_csv:
        print(f"best_by_mean_ARI_case_results.csv : {best_csv}")
    print("============================================================")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="mocap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)



    parser.add_argument("--win-size", default="128,256,512")
    parser.add_argument("--step", default="50,100")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--nb-steps", type=int, default=20)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--win-type", default="hanning")

    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--cp-margin-ratio", type=float, default=0.01)


    parser.add_argument("--feature-mode", default="")
    parser.add_argument("--train-subject", default="")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--remove-zero", action="store_true")

    args = parser.parse_args()
    if str(args.dataset).lower().replace("_", "-") not in {"mocap", "mo-cap"}:
        raise ValueError(f"This runner supports MoCap only, got {args.dataset!r}")
    return run_t2s_mocap(args)


if __name__ == "__main__":
    raise SystemExit(main())
