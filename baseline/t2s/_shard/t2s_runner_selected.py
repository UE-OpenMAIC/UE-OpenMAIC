from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SeriesCase:
    dataset: str
    case_id: str
    data: object
    labels: object
    fit_data: object | None = None


def add_project_imports(repo_root: Path) -> None:
    """Add project source directories without importing bundled binary wheels.

    The project may contain multi_t2s_paper_benchmark/_deps with compiled NumPy
    files for another Python version. Do not add that directory to sys.path.
    """
    deps = str(repo_root / "multi_t2s_paper_benchmark" / "_deps").lower().replace("/", "\\")
    sys.path[:] = [p for p in sys.path if not str(p).lower().replace("/", "\\").startswith(deps)]

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
        if part.strip():
            vals.append(int(float(part)))
    if not vals:
        raise ValueError(f"Empty integer list: {text!r}")
    return vals


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    aliases = {
        "synthetic": "synthetic",
        "actrectut": "actrectut",
        "act-rec-tut": "actrectut",
        "usc-had": "usc-had",
        "uschad": "usc-had",
        "ucrseg": "ucr-seg",
        "ucr-seg": "ucr-seg",
        "tssb": "ucr-seg",
    }
    return aliases.get(key, key)


def original_data_root(repo_root: Path) -> Path:
    return repo_root / "Time2State" / "data"


def fallback_normalize(data, np):
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    arr = (arr - mean) / std
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def original_normalize(data, runtime):
    norm = runtime.get("tspy_normalize")
    if norm is not None:
        return norm(data)
    return fallback_normalize(data, runtime["np"])


def reorder_label(seq, np):
    mapping = {}
    out = []
    next_id = 0
    for value in list(seq):
        key = int(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out.append(mapping[key])
    return np.asarray(out, dtype=int), mapping


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


def count_file_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def load_synthetic(repo_root: Path, runtime, max_count: int | None = None) -> list[SeriesCase]:
    """Strict Time2State synthetic protocol: test0.csv...test99.csv, first 4 cols as data, col 4 as label, no normalize."""
    pd = runtime["pd"]
    base = original_data_root(repo_root) / "synthetic_data_for_segmentation3"
    cases = []
    for i in range(100):
        path = base / f"test{i}.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=range(4), skiprows=1).to_numpy()
        labels = pd.read_csv(path, usecols=[4], skiprows=1).to_numpy(dtype=int).flatten()
        n = min(len(data), len(labels))
        cases.append(SeriesCase("Synthetic", str(i), data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No Synthetic files found under {base}, expected test0.csv...test99.csv")
    return cases


def load_actrectut(repo_root: Path, runtime, max_count: int | None = None) -> list[SeriesCase]:
    """Strict Time2State ActRecTut protocol: subject1_walk and subject2_walk, repeat each 10 times."""
    np = runtime["np"]
    scipy_io = runtime["scipy_io"]
    base = original_data_root(repo_root) / "ActRecTut"
    cases = []
    for name in ["subject1_walk", "subject2_walk"]:
        mat_path = base / name / "data.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing ActRecTut file: {mat_path}")
        mat = scipy_io.loadmat(mat_path)
        labels, _ = reorder_label(mat["labels"].flatten(), np)
        data = mat["data"][:, 0:10]
        data = original_normalize(data, runtime)
        n = min(len(data), len(labels))
        for rep in range(10):
            cases.append(SeriesCase("ActRecTut", f"{name}{rep}", data[:n], labels[:n]))
            if max_count is not None and len(cases) >= max_count:
                return cases
    return cases


def ensure_tspy_for_uschad(repo_root: Path, runtime) -> None:
    """Prefer the complete local TSpy-dev package for load_USC_HAD."""
    candidates = [
        repo_root / "TSpy-dev",
        Path(r"D:\code\teacherT2S\TSpy-dev"),
        repo_root.parent / "TSpy-dev",
    ]
    for root in candidates:
        if not root.exists():
            continue
        s = str(root)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
        for mod in list(sys.modules):
            if mod == "TSpy" or mod.startswith("TSpy."):
                sys.modules.pop(mod, None)
        try:
            import TSpy.dataset as tspy_dataset
            from TSpy.dataset import load_USC_HAD
            from TSpy.utils import normalize
            runtime["tspy_load_USC_HAD"] = load_USC_HAD
            runtime["tspy_normalize"] = normalize
            print("[CHECK] USC-HAD TSpy.dataset =", getattr(tspy_dataset, "__file__", ""), flush=True)
            return
        except Exception as exc:
            print(f"[WARN] TSpy-dev candidate failed: {root} -> {exc!r}", flush=True)


def load_uschad_case(subject: int, target: int, repo_root: Path, runtime):
    ensure_tspy_for_uschad(repo_root, runtime)
    load_USC_HAD = runtime.get("tspy_load_USC_HAD")
    if load_USC_HAD is None:
        raise RuntimeError(
            "TSpy.dataset.load_USC_HAD is unavailable. Put TSpy-dev at D:\\code\\teacherT2S\\TSpy-dev "
            "or under repo root, then rerun USC-HAD baseline."
        )
    data_path = str(original_data_root(repo_root)) + os.sep
    data, labels = load_USC_HAD(subject, target, data_path)
    data = original_normalize(data, runtime)
    return data, labels


def load_uschad(repo_root: Path, runtime, max_count: int | None = None) -> list[SeriesCase]:
    """Strict Time2State USC-HAD protocol: fit on subject 1 target 1, predict subject 1..14 target 1..5."""
    train_data, _ = load_uschad_case(1, 1, repo_root, runtime)
    cases = []
    for subject in range(1, 15):
        for target in range(1, 6):
            data, labels = load_uschad_case(subject, target, repo_root, runtime)
            n = min(len(data), len(labels))
            cases.append(SeriesCase("USC-HAD", f"s{subject}_t{target}", data[:n], labels[:n], fit_data=train_data))
            if max_count is not None and len(cases) >= max_count:
                return cases
    return cases


def load_ucrseg(repo_root: Path, runtime, max_count: int | None = None) -> list[SeriesCase]:
    """Strict Time2State UCR-SEG protocol: read change points from filename, normalize data."""
    np = runtime["np"]
    pd = runtime["pd"]
    dataset_path = original_data_root(repo_root) / "UCR-SEG" / "UCR_datasets_seg"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing UCR-SEG directory: {dataset_path}")
    cases = []
    for path in sorted([p for p in dataset_path.iterdir() if p.is_file()]):
        if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        parts = path.name[:-4].split("_")
        if len(parts) < 3:
            continue
        seg_info = {}
        for i, cp in enumerate(parts[2:]):
            seg_info[int(cp)] = i
        seg_info[count_file_lines(path)] = len(parts[2:])
        data = pd.read_csv(path).to_numpy()
        data = original_normalize(data, runtime)
        labels = seg_to_label(seg_info, np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("UCR-SEG", path.name[:-4], data[:n], labels[:n]))
        if max_count is not None and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No UCR-SEG files found under {dataset_path}")
    return cases


def load_cases(dataset_key: str, repo_root: Path, runtime, max_count: int | None = None) -> list[SeriesCase]:
    key = normalize_dataset_key(dataset_key)
    if key == "synthetic":
        return load_synthetic(repo_root, runtime, max_count)
    if key == "actrectut":
        return load_actrectut(repo_root, runtime, max_count)
    if key == "usc-had":
        return load_uschad(repo_root, runtime, max_count)
    if key == "ucr-seg":
        return load_ucrseg(repo_root, runtime, max_count)
    raise ValueError(f"Unsupported dataset={dataset_key!r}; this runner supports synthetic/actrectut/uschad/ucrseg only")


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
        y_counts = Counter(); z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1; z_counts[int(b)] += 1
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
        y = list(labels_true); z = list(labels_pred)
        if len(y) != len(z):
            raise ValueError("NMI inputs must have equal length")
        if len(y) == 0:
            return 0.0
        n = len(y)
        contingency = defaultdict(int)
        y_counts = Counter(); z_counts = Counter()
        for a, b in zip(y, z):
            contingency[(int(a), int(b))] += 1
            y_counts[int(a)] += 1; z_counts[int(b)] += 1
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
        hy = entropy(y_counts.values()); hz = entropy(z_counts.values())
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
    true_segments = segments_from_labels(labels_true)
    pred_segments = segments_from_labels(labels_pred)
    n = len(labels_true)
    if n == 0 or not true_segments or not pred_segments:
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
    used = set(); tp = 0
    for p in pred_cps:
        best_j = None; best_dist = None
        for j, t in enumerate(true_cps):
            if j in used:
                continue
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best_dist = dist; best_j = j
        if best_j is not None:
            used.add(best_j); tp += 1
    precision = tp / len(pred_cps) if pred_cps else 0.0
    recall = tp / len(true_cps) if true_cps else 0.0
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def align_sequence(seq, target_len: int, np):
    seq = np.asarray(seq, dtype=int)
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) == 0:
        return np.zeros(target_len, dtype=int)
    return np.concatenate([seq, np.full(target_len - len(seq), int(seq[-1]), dtype=int)])


def filter_lse_params(params: dict) -> dict:
    try:
        from Time2State import encoders
        sig = inspect.signature(encoders.CausalConv_LSE.__init__)
        allowed = {name for name in sig.parameters if name != "self"}
        return {k: v for k, v in params.items() if k in allowed}
    except Exception:
        return params


def make_model(*, win_size: int, step: int, in_channels: int, args, runtime):
    torch = runtime["torch"]
    Time2State = runtime["Time2State"]
    CausalConv_LSE_Adaper = runtime["CausalConv_LSE_Adaper"]
    DPGMM = runtime["DPGMM"]
    params_LSE = runtime["params_LSE"]

    params = dict(params_LSE)
    params["in_channels"] = int(in_channels)
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
    return Time2State(win_size, step, CausalConv_LSE_Adaper(params), DPGMM(None))


def run_fit_predict_same_case(case: SeriesCase, *, win_size: int, step: int, args, runtime, seed: int):
    np = runtime["np"]
    torch = runtime["torch"]
    if len(case.data) < win_size:
        raise ValueError(f"rows={len(case.data)} is shorter than win_size={win_size}")
    set_seeds(seed, torch)
    model = make_model(win_size=win_size, step=step, in_channels=int(case.data.shape[1]), args=args, runtime=runtime)
    t0 = time.time()
    model.fit(case.data, win_size, step)
    seconds = time.time() - t0
    return align_sequence(model.state_seq, len(case.labels), np), seconds, 0.0


def evaluate(labels, pred, args):
    np = None
    ari = adjusted_rand_index(labels, pred)
    nmi = normalized_mutual_information(labels, pred)
    ami = adjusted_mutual_information(labels, pred)
    covering = segmentation_covering(labels, pred)
    margin = max(1, int(round(len(labels) * float(args.cp_margin_ratio))))
    f1 = cp_f1(labels, pred, margin)
    return ari, nmi, ami, covering, f1, margin


def run_selected_t2s(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    add_project_imports(repo_root)

    import numpy as np
    import pandas as pd
    import scipy.io
    import torch
    from Time2State.adapers import CausalConv_LSE_Adaper
    from Time2State.clustering import DPGMM
    from Time2State.default_params import params_LSE
    from Time2State.time2state import Time2State
    try:
        from TSpy.utils import normalize as tspy_normalize
    except Exception:
        tspy_normalize = None

    runtime = {
        "np": np,
        "pd": pd,
        "scipy_io": scipy.io,
        "torch": torch,
        "CausalConv_LSE_Adaper": CausalConv_LSE_Adaper,
        "DPGMM": DPGMM,
        "params_LSE": params_LSE,
        "Time2State": Time2State,
        "tspy_normalize": tspy_normalize,
        "tspy_load_USC_HAD": None,
    }

    key = normalize_dataset_key(args.dataset)
    cases = load_cases(key, repo_root, runtime, max_count=args.max_cases)
    win_sizes = parse_int_list(args.win_size)
    steps = parse_int_list(args.step)
    grid = [(w, s) for w in win_sizes for s in steps]

    print("============================================================")
    print(f"Time2State baseline on {args.dataset}")
    print("Protocol : paper-grid single Time2State; no multi-branch, no PEER/PID, no meta clustering")
    print(f"Repo root: {repo_root}")
    print(f"Output   : {out_dir}")
    print(f"Cases    : {len(cases)}")
    print(f"Grid     : {grid}")
    print(f"M/N      : {args.m}/{args.n}")
    print(f"Device   : {args.device}, gpu={args.gpu}")
    print("============================================================", flush=True)

    if args.dry_run:
        return 0

    all_rows = []
    start_all = time.time()

    for grid_idx, (win_size, step) in enumerate(grid, start=1):
        print(f"\n[GRID {grid_idx}/{len(grid)}] win_size={win_size}, step={step}", flush=True)


        shared_model = None
        shared_fit_seconds = 0.0
        shared_fit_error = ""
        if cases and all(c.fit_data is not None for c in cases):
            fit_data = cases[0].fit_data
            try:
                if len(fit_data) < win_size:
                    raise ValueError(f"fit rows={len(fit_data)} is shorter than win_size={win_size}")
                set_seeds(int(args.seed) + grid_idx * 100000, runtime["torch"])
                shared_model = make_model(win_size=win_size, step=step, in_channels=int(fit_data.shape[1]), args=args, runtime=runtime)
                t0 = time.time()
                shared_model.fit(fit_data, win_size, step)
                shared_fit_seconds = time.time() - t0
                print(f"  [shared fit] seconds={shared_fit_seconds:.2f}", flush=True)
            except Exception as exc:
                shared_fit_error = repr(exc)
                print(f"  [shared fit] ERROR {shared_fit_error}", flush=True)

        for case_idx, case in enumerate(cases, start=1):
            seed = int(args.seed) + grid_idx * 100000 + case_idx
            fit_seconds = 0.0
            predict_seconds = 0.0
            try:
                if shared_model is not None:
                    t0 = time.time()
                    shared_model.predict(case.data, win_size, step)
                    predict_seconds = time.time() - t0
                    pred = align_sequence(shared_model.state_seq, len(case.labels), np)
                    fit_seconds = shared_fit_seconds
                    error = ""
                elif shared_fit_error:
                    raise RuntimeError(shared_fit_error)
                else:
                    pred, fit_seconds, predict_seconds = run_fit_predict_same_case(
                        case, win_size=win_size, step=step, args=args, runtime=runtime, seed=seed
                    )
                    error = ""
            except Exception as exc:
                pred = np.zeros(len(case.labels), dtype=int)
                fit_seconds = float("nan")
                predict_seconds = float("nan")
                error = repr(exc)

            if error:
                ari = nmi = ami = covering = f1 = float("nan")
                margin = max(1, int(round(len(case.labels) * float(args.cp_margin_ratio))))
            else:
                ari, nmi, ami, covering, f1, margin = evaluate(case.labels, pred, args)

            safe_case = case.case_id.replace("/", "_").replace("\\", "_").replace(".", "_")
            safe_ds = case.dataset.replace("/", "_").replace(" ", "_")
            np.save(pred_dir / f"Time2State_{safe_ds}_{safe_case}_w{win_size}_s{step}_labels_pred.npy", np.vstack([case.labels, pred]))

            row = {
                "algorithm": "Time2State",
                "dataset": case.dataset,
                "case_id": case.case_id,
                "rows": int(len(case.labels)),
                "features": int(case.data.shape[1]),
                "true_states": int(len(np.unique(case.labels))),
                "pred_states": int(len(np.unique(pred))) if not error else 0,
                "ARI": ari,
                "NMI": nmi,
                "AMI": ami,
                "Covering": covering,
                "CP_F1": f1,
                "CP_margin": int(margin),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "seconds": (fit_seconds if not math.isnan(fit_seconds) else 0.0) + (predict_seconds if not math.isnan(predict_seconds) else 0.0),
                "win_size": int(win_size),
                "step": int(step),
                "m": int(args.m),
                "n": int(args.n),
                "out_channels": int(args.out_channels),
                "nb_steps": int(args.nb_steps),
                "win_type": str(args.win_type),
                "seed": int(seed),
                "protocol": "fit_train_predict_cases" if case.fit_data is not None else "fit_each_case_predict_same_case",
                "error": error,
            }
            all_rows.append(row)
            if error:
                print(f"  {case.dataset}/{case.case_id}: ERROR {error}", flush=True)
            else:
                print(
                    f"  {case.dataset}/{case.case_id}: ARI={ari:.4f} NMI={nmi:.4f} "
                    f"AMI={ami:.4f} Covering={covering:.4f} K={row['pred_states']} seconds={row['seconds']:.1f}",
                    flush=True,
                )

    if not all_rows:
        raise RuntimeError("No case rows were produced")

    case_csv = out_dir / "case_results.csv"
    with case_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)

    summaries = []
    for win_size, step in grid:
        rows = [r for r in all_rows if r["win_size"] == win_size and r["step"] == step and not r["error"]]
        def mean_col(col: str):
            return float(np.mean([r[col] for r in rows])) if rows else float("nan")
        summaries.append({
            "algorithm": "Time2State",
            "dataset": cases[0].dataset if cases else args.dataset,
            "cases": len(rows),
            "mean_ARI": mean_col("ARI"),
            "mean_NMI": mean_col("NMI"),
            "mean_AMI": mean_col("AMI"),
            "mean_Covering": mean_col("Covering"),
            "mean_CP_F1": mean_col("CP_F1"),
            "total_seconds": float(np.nansum([r["seconds"] for r in rows])) if rows else float("nan"),
            "win_size": int(win_size),
            "step": int(step),
            "m": int(args.m),
            "n": int(args.n),
            "out_channels": int(args.out_channels),
            "nb_steps": int(args.nb_steps),
            "win_type": str(args.win_type),
            "protocol": "fit_train_predict_cases" if cases and cases[0].fit_data is not None else "fit_each_case_predict_same_case",
            "metric_backend": "sklearn_ARI__sklearn_NMI_geometric__sklearn_AMI_arithmetic__segment_covering__cp_f1_margin_1pct",
        })

    summary_csv = out_dir / "algorithm_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)

    valid = [s for s in summaries if not math.isnan(s["mean_ARI"])]
    best_by_ari = max(valid, key=lambda x: x["mean_ARI"]) if valid else None
    best_by_nmi = max(valid, key=lambda x: x["mean_NMI"]) if valid else None
    best_csv = None
    if best_by_ari is not None:
        best_rows = [r for r in all_rows if r["win_size"] == best_by_ari["win_size"] and r["step"] == best_by_ari["step"]]
        best_csv = out_dir / "best_by_mean_ARI_case_results.csv"
        with best_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader(); writer.writerows(best_rows)

    status = {
        "ok": True,
        "algorithm": "Time2State",
        "dataset": cases[0].dataset if cases else args.dataset,
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "case_results_csv": str(case_csv),
        "algorithm_summary_csv": str(summary_csv),
        "best_by_mean_ARI_case_results_csv": str(best_csv) if best_csv else "",
        "best_by_mean_ARI": best_by_ari,
        "best_by_mean_NMI": best_by_nmi,
        "grid": [{"win_size": w, "step": s} for w, s in grid],
        "total_wall_seconds": float(time.time() - start_all),
        "protocol": "single Time2State paper-grid baseline; strict preprocessing; no multi-branch/meta/PEER/PID",
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
    parser.add_argument("--dataset", required=True)
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
    return run_selected_t2s(args)


if __name__ == "__main__":
    raise SystemExit(main())
