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


def add_project_imports(repo_root: Path) -> None:
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


def fill_nan_original(data, np):
    arr = np.asarray(data, dtype=float).copy()
    x_len, y_len = arr.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(arr[x, y]):
                arr[x, y] = arr[x - 1, y]
    return arr


def fallback_normalize(data, np):
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    arr = (arr - mean) / std
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_normalize():
    try:
        from TSpy.utils import normalize
        return normalize
    except Exception:
        return None


def find_pamap2_protocol_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "Time2State" / "data" / "PAMAP2" / "Protocol",
        repo_root / "Time2State" / "data" / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find PAMAP2 Protocol directory. Tried: " + " | ".join(str(x) for x in candidates))


def load_pamap2_subject(protocol_dir: Path, subject_idx: int, np, pd, normalize_fn, feature_mode: str, remove_zero: bool):
    path = protocol_dir / f"subject10{subject_idx}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing PAMAP2 file: {path}")

    df = pd.read_csv(path, sep=" ", header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if numeric.shape[1] < 2:
        raise ValueError(f"PAMAP2 file has too few columns: {path}, shape={numeric.shape}")

    labels = np.asarray(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)
    mode = str(feature_mode or "paper9acc").strip().lower()

    if mode == "full_sensor":
        data = numeric[:, 2:]
        if data.shape[1] <= 0:
            raise ValueError(f"No full_sensor columns found in {path}")
    elif mode == "paper9acc":
        if numeric.shape[1] < 41:
            raise ValueError(f"Too few columns for paper9acc in {path}, shape={numeric.shape}")
        hand_acc = numeric[:, 4:7]
        chest_acc = numeric[:, 21:24]
        ankle_acc = numeric[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
    else:
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}")

    data = fill_nan_original(data, np)

    if remove_zero:
        valid = labels > 0
        data = data[valid]
        labels = labels[valid]
        if len(labels) < 2:
            raise ValueError(f"subject10{subject_idx} has too few non-zero frames")

    if normalize_fn is not None:
        data = normalize_fn(data)
    else:
        data = fallback_normalize(data, np)

    n = min(len(data), len(labels))
    return data[:n], labels[:n]


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


def parse_subjects(text: str) -> list[int]:
    vals = []
    for part in str(text).replace(",", " ").replace(";", " ").split():
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(list(range(int(a), int(b) + 1)))
        else:
            vals.append(int(part))
    return vals or list(range(1, 9))


def run_t2s_pamap2(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    add_project_imports(repo_root)

    import numpy as np
    import pandas as pd
    import torch

    from Time2State.adapers import CausalConv_LSE_Adaper
    from Time2State.clustering import DPGMM
    from Time2State.default_params import params_LSE
    from Time2State.time2state import Time2State

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = find_pamap2_protocol_dir(repo_root)
    normalize_fn = load_normalize()

    subjects = parse_subjects(args.subjects)
    train_subject = int(args.train_subject)
    win_size = int(args.win_size)
    step = int(args.step)
    feature_mode = str(args.feature_mode)
    remove_zero = bool(args.remove_zero)

    print("============================================================")
    print("Time2State baseline on PAMAP2_zero")
    print(f"Protocol : fit subject10{train_subject}.dat once, predict selected subjects")
    print(f"Subjects : {subjects}")
    print(f"Feature  : {feature_mode}")
    print(f"Remove 0 : {int(remove_zero)}")
    print(f"Repo root: {repo_root}")
    print(f"Data dir : {protocol_dir}")
    print(f"Output   : {out_dir}")
    print(f"win/step : {win_size}/{step}")
    print("============================================================")

    if args.dry_run:
        return 0

    train_data, _ = load_pamap2_subject(
        protocol_dir,
        train_subject,
        np,
        pd,
        normalize_fn,
        feature_mode=feature_mode,
        remove_zero=remove_zero,
    )

    params = dict(params_LSE)
    params["in_channels"] = int(train_data.shape[1])
    params["compared_length"] = win_size
    params["win_size"] = win_size
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

    start_all = time.time()
    model = Time2State(win_size, step, CausalConv_LSE_Adaper(params), DPGMM(None))

    print(f"[1/2] Fitting Time2State on PAMAP2 subject10{train_subject}.dat ...")
    t0 = time.time()
    model.fit(train_data, win_size, step)
    fit_seconds = time.time() - t0
    print(f"[OK] fit seconds = {fit_seconds:.2f}")

    rows = []
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("[2/2] Predicting subjects")
    for subject_idx in subjects:
        case_id = f"10{subject_idx}"
        data, labels = load_pamap2_subject(
            protocol_dir,
            subject_idx,
            np,
            pd,
            normalize_fn,
            feature_mode=feature_mode,
            remove_zero=remove_zero,
        )

        t0 = time.time()
        model.predict(data, win_size, step)
        predict_seconds = time.time() - t0

        pred = align_sequence(model.state_seq, len(labels), np)

        ari = adjusted_rand_index(labels, pred)
        nmi = normalized_mutual_information(labels, pred)
        k_pred = int(len(np.unique(pred)))
        k_true = int(len(np.unique(labels)))

        np.save(pred_dir / f"PAMAP2_zero_{case_id}_labels_pred.npy", np.vstack([labels, pred]))

        row = {
            "algorithm": "Time2State",
            "dataset": "PAMAP2_zero",
            "case_id": case_id,
            "rows": int(len(labels)),
            "features": int(data.shape[1]),
            "true_states": k_true,
            "pred_states": k_pred,
            "ARI": float(ari),
            "NMI": float(nmi),
            "fit_seconds_shared": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "win_size": win_size,
            "step": step,
            "m": int(args.m),
            "n": int(args.n),
            "out_channels": int(args.out_channels),
            "nb_steps": int(args.nb_steps),
            "feature_mode": feature_mode,
            "remove_zero": int(remove_zero),
            "train_case": f"10{train_subject}",
            "protocol": f"fit_subject10{train_subject}_predict_selected_subjects",
        }
        rows.append(row)

        print(f"  PAMAP2_zero/{case_id}: ARI={ari:.4f} NMI={nmi:.4f} K={k_pred} seconds={predict_seconds:.1f}")

    total_seconds = time.time() - start_all

    case_csv = out_dir / "case_results.csv"
    with case_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_ari = float(np.mean([r["ARI"] for r in rows]))
    mean_nmi = float(np.mean([r["NMI"] for r in rows]))

    summary = {
        "algorithm": "Time2State",
        "dataset": "PAMAP2_zero",
        "cases": len(rows),
        "mean_ARI": mean_ari,
        "mean_NMI": mean_nmi,
        "fit_seconds_shared": float(fit_seconds),
        "total_seconds": float(total_seconds),
        "feature_mode": feature_mode,
        "remove_zero": int(remove_zero),
        "train_case": f"10{train_subject}",
        "protocol": f"fit_subject10{train_subject}_predict_selected_subjects",
        "metric_backend": "sklearn_adjusted_rand_score__sklearn_nmi_geometric",
    }

    summary_csv = out_dir / "algorithm_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    status = {
        "ok": True,
        "algorithm": "Time2State",
        "dataset": "PAMAP2_zero",
        "repo_root": str(repo_root),
        "protocol_dir": str(protocol_dir),
        "out_dir": str(out_dir),
        "case_results_csv": str(case_csv),
        "algorithm_summary_csv": str(summary_csv),
        "mean_ARI": mean_ari,
        "mean_NMI": mean_nmi,
        "total_seconds": total_seconds,
        "feature_mode": feature_mode,
        "remove_zero": remove_zero,
        "protocol": f"fit subject10{train_subject}.dat once, predict selected subjects",
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print(f"AVG ---- ARI: {mean_ari:.6f}, NMI: {mean_nmi:.6f}")
    print(f"case_results.csv      : {case_csv}")
    print(f"algorithm_summary.csv : {summary_csv}")
    print(f"total seconds         : {total_seconds:.2f}")
    print("============================================================")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="pamap2_zero")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--feature-mode", choices=["paper9acc", "full_sensor"], default="full_sensor")
    parser.add_argument("--remove-zero", action="store_true")
    parser.add_argument("--train-subject", type=int, default=1)
    parser.add_argument("--subjects", default="1-8")

    parser.add_argument("--win-size", type=int, default=512)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--nb-steps", type=int, default=20)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--win-type", default="hanning")

    args = parser.parse_args()
    if str(args.dataset).lower() not in {"pamap2_zero", "pamap2", "pamap2-zero"}:
        raise ValueError(f"This runner currently supports PAMAP2/PAMAP2_zero only, got {args.dataset!r}")
    return run_t2s_pamap2(args)


if __name__ == "__main__":
    raise SystemExit(main())
