from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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


@dataclass
class SeriesCase:
    dataset: str
    case_id: str
    data: object
    labels: object


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    aliases = {
        "synthetic": "synthetic",
        "mocap": "mocap",
        "actrectut": "actrectut",
        "act-rec-tut": "actrectut",
        "pamap2": "pamap2",
        "pamap2zero": "pamap2_zero",
        "pamap2-zero": "pamap2_zero",
        "uschad": "usc-had",
        "usc-had": "usc-had",
        "ucrseg": "ucrseg",
        "ucr-seg": "ucrseg",
        "tssb": "ucrseg",
    }
    return aliases.get(key, key)


def split_ints(text: str | None, default: Sequence[int]) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(default)
    parts = str(text).replace(",", " ").replace(";", " ").split()
    return [int(float(p)) for p in parts]


def seg_to_label(seg_info: dict[int, int], np):
    labels = []
    start = 0
    for end in sorted(seg_info):
        if end < start:
            continue
        labels.extend([seg_info[end]] * max(0, end - start))
        start = end
    return np.asarray(labels, dtype=int)


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
    return np.asarray(out, dtype=int)


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


def fill_nan_forward(data, np):
    arr = np.asarray(data, dtype=float).copy()
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if np.isnan(arr[r, c]):
                arr[r, c] = arr[r - 1, c] if r > 0 else 0.0
    return arr


def fallback_normalize(data, np):
    arr = np.asarray(data, dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def find_existing_dir(candidates: Iterable[Path], description: str) -> Path:
    tried = []
    for path in candidates:
        path = Path(path)
        tried.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {description}. Tried:\n  " + "\n  ".join(tried))


def detect_e2usd_root(repo_root: Path, explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None and str(explicit).strip() != "":
        candidates.append(Path(explicit))
    candidates.extend([
        repo_root / "E2Usd-main",
        repo_root / "E2USD-main",
        repo_root / "E2USD",
        repo_root / "Baselines" / "E2USD",
        repo_root,
    ])
    for root in candidates:
        root = root.resolve()
        if (root / "E2USD").exists():
            return root
        if root.name == "E2USD" and (root / "e2usd.py").exists():
            return root.parent
    raise FileNotFoundError(
        "Cannot locate E2USD package root. Expected a directory containing an E2USD/ folder, e.g. D:\\code\\teacherT2S\\E2Usd-main."
    )


def add_runtime_imports(repo_root: Path, e2usd_root: Path) -> None:
    candidates = [
        e2usd_root,
        repo_root,
        repo_root / "TSpy-dev",
        repo_root / "Time2State",
        repo_root / "multi_t2s_paper_benchmark",
    ]
    for path in reversed(candidates):
        if path.exists():
            s = str(path)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)


def import_runtime(repo_root: Path, e2usd_root: Path):
    add_runtime_imports(repo_root, e2usd_root)
    import numpy as np
    import pandas as pd
    import scipy.io
    import torch
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

    from E2USD.e2usd import E2USD
    from E2USD.adapers import E2USD_Adaper
    from E2USD.clustering import DPGMM
    from E2USD.params import params as e2usd_params

    try:
        from E2USD.utils import normalize as e2usd_normalize
    except Exception:
        e2usd_normalize = None

    try:
        from TSpy.dataset import load_USC_HAD as tspy_load_USC_HAD
    except Exception:
        tspy_load_USC_HAD = None

    return {
        "np": np,
        "pd": pd,
        "scipy_io": scipy.io,
        "torch": torch,
        "adjusted_rand_score": adjusted_rand_score,
        "normalized_mutual_info_score": normalized_mutual_info_score,
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "E2USD": E2USD,
        "E2USD_Adaper": E2USD_Adaper,
        "DPGMM": DPGMM,
        "e2usd_params": e2usd_params,
        "e2usd_normalize": e2usd_normalize,
        "tspy_load_USC_HAD": tspy_load_USC_HAD,
    }


def normalize_data(data, runtime):
    np = runtime["np"]
    norm = runtime.get("e2usd_normalize")
    if norm is not None:
        try:
            return norm(np.asarray(data, dtype=float))
        except Exception:
            pass
    return fallback_normalize(data, np)


def default_data_roots(repo_root: Path, e2usd_root: Path, explicit_data_root: Path | None, public_data_root: Path | None) -> list[Path]:
    """Strict data-root priority aligned with the user's our_multit2s_runner.py.

    Default: prefer <repo_root>/Time2State/data, because the user's method and the
    strict Time2State baseline both reproduce exp_of_Time2State.py preprocessing.
    E2USD's own data folder and public_data_root are only fallback candidates.
    """
    roots: list[Path] = []
    if explicit_data_root is not None and str(explicit_data_root).strip():
        roots.append(Path(explicit_data_root))
    roots.extend([
        repo_root / "Time2State" / "data",
        e2usd_root / "data",
        e2usd_root.parent / "data",
    ])
    if public_data_root is not None and str(public_data_root).strip():
        roots.append(Path(public_data_root))
    roots.extend([
        repo_root / "Time2State" / "Baselines" / "public_ts_datasets",
        repo_root / "multi_t2s_paper_benchmark" / "public_ts_datasets",
    ])
    dedup = []
    seen = set()
    for r in roots:
        s = str(Path(r).resolve())
        if s not in seen:
            seen.add(s)
            dedup.append(Path(s))
    return dedup


def strict_data_root(data_roots: list[Path]) -> Path:

    return find_existing_dir(data_roots, "strict Time2State/data root")


def tspy_normalize(data, runtime):
    """Prefer TSpy.utils.normalize, matching the user's strict Time2State loader."""
    try:
        from TSpy.utils import normalize as _normalize
        return _normalize(data)
    except Exception:
        return normalize_data(data, runtime)


def load_mocap(data_roots: list[Path], runtime, max_count=None) -> list[SeriesCase]:
    """Strictly match our_multit2s_runner.load_mocap / exp_on_MoCap.

    - root: Time2State/data/MoCap/4d
    - pd.read_csv(..., sep=' ', usecols=range(0,4)), default header=0
    - no external normalize
    - labels from MOCAP_INFO seg_to_label(... )[:-1]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = find_existing_dir([r / "MoCap" / "4d" for r in data_roots], "MoCap/4d directory")
    cases = []
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".4d") or fname not in MOCAP_INFO:
            continue
        df = pd.read_csv(base / fname, sep=" ", usecols=range(0, 4))
        data = df.to_numpy(dtype=float)
        labels = seg_to_label(MOCAP_INFO[fname]["label"], np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("mocap", fname, data[:n], labels[:n]))
        if max_count and len(cases) >= max_count:
            break
    if not cases:
        raise RuntimeError(f"No MoCap .4d cases loaded from {base}")
    return cases


def load_actrectut(data_roots: list[Path], runtime, max_count=None) -> list[SeriesCase]:
    """Strictly match our_multit2s_runner.load_actrectut.

    - two dirs: subject1_walk, subject2_walk
    - repeat each sequence 10 times
    - labels = reorder_label(labels)
    - data = data[:,0:10]
    - data = TSpy normalize(data), fallback to local normalize
    """
    np = runtime["np"]
    scipy_io = runtime["scipy_io"]
    base = find_existing_dir([r / "ActRecTut" for r in data_roots], "ActRecTut directory")
    cases = []
    for dir_name in ["subject1_walk", "subject2_walk"]:
        mat_path = base / dir_name / "data.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing ActRecTut file: {mat_path}")
        mat = scipy_io.loadmat(mat_path)
        labels0 = reorder_label(mat["labels"].flatten(), np)
        data0 = np.asarray(mat["data"][:, 0:10], dtype=float)
        data0 = tspy_normalize(data0, runtime)
        for rep in range(10):
            cases.append(SeriesCase("actrectut", f"{dir_name}{rep}", data0.copy(), labels0.copy()))
            if max_count and len(cases) >= max_count:
                return cases
    return cases


def load_synthetic(data_roots: list[Path], runtime, max_count=None) -> list[SeriesCase]:
    """Strictly match our_multit2s_runner.load_synthetic / exp_on_synthetic.

    - root: Time2State/data/synthetic_data_for_segmentation3
    - files: test0.csv ... test99.csv
    - pd.read_csv(..., usecols=range(4), skiprows=1), default header=0
    - no external normalize
    """
    pd = runtime["pd"]
    base = find_existing_dir([r / "synthetic_data_for_segmentation3" for r in data_roots], "synthetic_data_for_segmentation3 directory")
    cases = []
    for i in range(100):
        path = base / f"test{i}.csv"
        if not path.exists():
            continue
        df_x = pd.read_csv(path, usecols=range(4), skiprows=1)
        df_y = pd.read_csv(path, usecols=[4], skiprows=1)
        data = df_x.to_numpy(dtype=float)
        labels = df_y.to_numpy(dtype=int).flatten()
        n = min(len(data), len(labels))
        cases.append(SeriesCase("synthetic", str(i), data[:n], labels[:n]))
        if max_count and len(cases) >= max_count:
            break
    if not cases:
        raise FileNotFoundError(f"No strict synthetic files found under {base}")
    return cases


def load_pamap2(data_roots: list[Path], runtime, max_count=None, feature_mode="paper9acc", remove_zero: bool = False, dataset_name: str = "pamap2") -> list[SeriesCase]:
    """Strict default PAMAP2 preprocessing aligned with our_multit2s_runner.

    Default:
    - Time2State/data/PAMAP2/Protocol or PAMAP2_Dataset/Protocol
    - keep activity_id=0 by default; PAMAP2_zero sets remove_zero=True
    - use paper9acc: hand/chest/ankle acceleration columns 4:7, 21:24, 38:41
    - fill NaN forward exactly as original
    - normalize with TSpy normalize fallback
    - run protocol handled later: fit on first case then predict all cases
    """
    np = runtime["np"]
    pd = runtime["pd"]
    protocol = find_existing_dir(
        [r / "PAMAP2" / "Protocol" for r in data_roots] +
        [r / "PAMAP2" / "PAMAP2_Dataset" / "Protocol" for r in data_roots],
        "PAMAP2 Protocol directory",
    )
    cases = []
    for i in range(1, 9):
        path = protocol / f"subject10{i}.dat"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=" ", header=None)
        numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        labels = np.asarray(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)
        mode = str(feature_mode).lower()
        if mode == "full_sensor":
            data = numeric[:, 2:]
        else:
            data = np.hstack([numeric[:, 4:7], numeric[:, 21:24], numeric[:, 38:41]])
        data = fill_nan_forward(data, np)
        if remove_zero:
            valid = labels > 0
            data = data[valid]
            labels = labels[valid]
            if len(labels) < 2:
                raise ValueError(f"subject10{i} has too few non-zero frames after remove_zero")
        data = tspy_normalize(data, runtime)
        n = min(len(data), len(labels))
        cases.append(SeriesCase(dataset_name, f"10{i}", data[:n], labels[:n]))
        if max_count and len(cases) >= max_count:
            break
    if not cases:
        raise RuntimeError(f"No PAMAP2 subject files found in {protocol}")
    return cases


def _force_tspy_dev(repo_root: Path | None = None):
    """Prefer D:/code/teacherT2S/TSpy-dev when present, matching user's strict runner."""
    candidates = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / "TSpy-dev")
    candidates.append(Path(r"D:\code\teacherT2S\TSpy-dev"))
    for root in candidates:
        if root.exists():
            s = str(root)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
            for mod in list(sys.modules):
                if mod == "TSpy" or mod.startswith("TSpy."):
                    sys.modules.pop(mod, None)
            break


def load_uschad(data_roots: list[Path], runtime, max_count=None) -> list[SeriesCase]:
    """Strictly match our_multit2s_runner.load_uschad.

    - use TSpy-dev load_USC_HAD when available
    - data_path is Time2State/data
    - normalize every sequence
    - run protocol handled later: fit on s1_t1, then predict all 70 cases
    """
    np = runtime["np"]
    data_root = strict_data_root(data_roots)

    try:

        repo_root = data_root.parent.parent if data_root.name == "data" else None
        _force_tspy_dev(repo_root)
        from TSpy.dataset import load_USC_HAD as loader
    except Exception:
        loader = runtime.get("tspy_load_USC_HAD")
    if loader is None:
        raise RuntimeError("TSpy.dataset.load_USC_HAD is unavailable. Install/activate TSpy-dev or use the same env as your T2S baseline.")
    cases = []
    for subject in range(1, 15):
        for target in range(1, 6):
            data, labels = loader(subject, target, str(data_root) + os.sep)
            data = tspy_normalize(data, runtime)
            labels = np.asarray(labels, dtype=int)
            n = min(len(data), len(labels))
            cases.append(SeriesCase("usc-had", f"s{subject}_t{target}", data[:n], labels[:n]))
            if max_count and len(cases) >= max_count:
                return cases
    return cases


def _count_file_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def load_ucrseg(data_roots: list[Path], runtime, max_count=None, dirname=None) -> list[SeriesCase]:
    """Strictly match our_multit2s_runner.load_tssb / exp_on_UCR_SEG.

    - default root: Time2State/data/UCR-SEG/UCR_datasets_seg
    - parse change points from filename
    - pd.read_csv(file), default header=0
    - normalize(data)
    - labels = seg_to_label(seg_info)[:-1]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    candidates = []
    if dirname:
        candidates.extend([r / dirname for r in data_roots])
    candidates.extend([r / "UCR-SEG" / "UCR_datasets_seg" for r in data_roots])
    base = find_existing_dir(candidates, "UCR-SEG/UCR_datasets_seg directory")
    cases = []
    for path in sorted([p for p in base.iterdir() if p.is_file()]):
        if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        info = path.name[:-4].split("_")
        if len(info) < 3:
            continue
        seg_info = {}
        for k, seg in enumerate(info[2:]):
            seg_info[int(seg)] = k
        seg_info[_count_file_lines(path)] = len(info[2:])
        df = pd.read_csv(path)
        data = df.to_numpy(dtype=float)
        data = tspy_normalize(data, runtime)
        labels = seg_to_label(seg_info, np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("ucrseg", path.name[:-4], data[:n], labels[:n]))
        if max_count and len(cases) >= max_count:
            break
    if not cases:
        raise RuntimeError(f"No UCR-SEG files loaded from {base}")
    return cases


def load_cases(key: str, data_roots: list[Path], runtime, args) -> list[SeriesCase]:
    if key == "mocap":
        return load_mocap(data_roots, runtime, args.max_cases)
    if key == "actrectut":
        return load_actrectut(data_roots, runtime, args.max_cases)
    if key == "synthetic":
        return load_synthetic(data_roots, runtime, args.max_cases)
    if key == "pamap2":
        return load_pamap2(data_roots, runtime, args.max_cases, feature_mode=args.pamap2_feature_mode, remove_zero=False, dataset_name="pamap2")
    if key == "pamap2_zero":
        return load_pamap2(data_roots, runtime, args.max_cases, feature_mode=args.pamap2_feature_mode, remove_zero=bool(getattr(args, "pamap2_remove_zero", False)), dataset_name="pamap2_zero")
    if key == "usc-had":
        return load_uschad(data_roots, runtime, args.max_cases)
    if key == "ucrseg":
        return load_ucrseg(data_roots, runtime, args.max_cases, dirname=args.ucrseg_dirname)
    raise ValueError(f"Unsupported dataset: {key}")


def true_pred_segments(labels):
    labels = list(labels)
    if not labels:
        return []
    out = []
    start = 0
    last = labels[0]
    for i, v in enumerate(labels[1:], start=1):
        if v != last:
            out.append((start, i, last))
            start = i
            last = v
    out.append((start, len(labels), last))
    return out


def segment_covering(labels_true, labels_pred) -> float:
    true_seg = true_pred_segments(labels_true)
    pred_seg = true_pred_segments(labels_pred)
    n = len(labels_true)
    if n == 0:
        return 0.0
    total = 0.0
    for ts, te, _ in true_seg:
        best = 0.0
        for ps, pe, _ in pred_seg:
            inter = max(0, min(te, pe) - max(ts, ps))
            if inter <= 0:
                continue
            union = max(te, pe) - min(ts, ps)
            best = max(best, inter / union if union > 0 else 0.0)
        total += (te - ts) * best
    return float(total / n)


def change_points(labels):
    labels = list(labels)
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def cp_f1(labels_true, labels_pred, margin: int) -> float:
    true_cp = change_points(labels_true)
    pred_cp = change_points(labels_pred)
    used = set()
    tp = 0
    for t in true_cp:
        best_j = None
        best_d = None
        for j, p in enumerate(pred_cp):
            if j in used:
                continue
            d = abs(p - t)
            if d <= margin and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is not None:
            used.add(best_j)
            tp += 1
    fp = len(pred_cp) - tp
    fn = len(true_cp) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def compute_metrics(y_true, y_pred, runtime, cp_margin_ratio: float) -> dict[str, float]:
    np = runtime["np"]
    y_true = np.asarray(y_true, dtype=int)
    y_pred = align_sequence(y_pred, len(y_true), np)
    ari = float(runtime["adjusted_rand_score"](y_true, y_pred))
    nmi = float(runtime["normalized_mutual_info_score"](y_true, y_pred, average_method="geometric"))
    ami = float(runtime["adjusted_mutual_info_score"](y_true, y_pred))
    cov = segment_covering(y_true, y_pred)
    margin = max(1, int(round(len(y_true) * cp_margin_ratio)))
    f1 = cp_f1(y_true, y_pred, margin)
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "Covering": cov, "CP_F1": f1, "cp_margin": margin}


def seed_everything(seed: int, runtime) -> None:
    random.seed(seed)
    try:
        runtime["np"].random.seed(seed)
    except Exception:
        pass
    try:
        runtime["torch"].manual_seed(seed)
        if runtime["torch"].cuda.is_available():
            runtime["torch"].cuda.manual_seed_all(seed)
    except Exception:
        pass


def prepare_params(runtime, data_dim: int, win_size: int, args, dataset_key: str):
    torch = runtime["torch"]
    p = copy.deepcopy(runtime["e2usd_params"])


    p["win_size"] = int(win_size)
    p["compared_length"] = int(win_size)
    p["in_channels"] = int(data_dim)
    p["M"] = int(args.m)
    p["N"] = int(args.n)
    p["nb_steps"] = int(args.nb_steps)
    p["kernel_size"] = int(args.kernel_size)
    p["win_type"] = str(args.win_type)
    p["gpu"] = int(args.gpu)
    if str(args.device).lower() == "cuda":
        p["cuda"] = bool(torch.cuda.is_available())
    elif str(args.device).lower() == "cpu":
        p["cuda"] = False
    else:
        p["cuda"] = bool(torch.cuda.is_available())
    if args.out_channels is not None:
        p["out_channels"] = int(args.out_channels)
    else:
        p["out_channels"] = 9 if dataset_key == "pamap2" else 4
    return p


def run_one_case(case: SeriesCase, win_size: int, step: int, runtime, args, dataset_key: str):
    E2USD = runtime["E2USD"]
    E2USD_Adaper = runtime["E2USD_Adaper"]
    DPGMM = runtime["DPGMM"]
    p = prepare_params(runtime, case.data.shape[1], win_size, args, dataset_key)
    model = E2USD(win_size, step, E2USD_Adaper(p), DPGMM(None)).fit(case.data, win_size, step)
    return model.state_seq, model


def run_with_prefit(train_case: SeriesCase, cases: list[SeriesCase], win_size: int, step: int, runtime, args, dataset_key: str):
    E2USD = runtime["E2USD"]
    E2USD_Adaper = runtime["E2USD_Adaper"]
    DPGMM = runtime["DPGMM"]
    p = prepare_params(runtime, train_case.data.shape[1], win_size, args, dataset_key)
    model = E2USD(win_size, step, E2USD_Adaper(p), DPGMM(None)).fit(train_case.data, win_size, step)
    preds = []
    for case in cases:
        model.predict(case.data, win_size, step)
        preds.append(model.state_seq.copy())
    return preds


def append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(case_results_path: Path, summary_path: Path, best_path: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(case_results_path)
    metrics = ["ARI", "NMI", "AMI", "Covering", "CP_F1"]
    group_cols = ["dataset", "win_size", "step", "M", "N", "out_channels", "nb_steps", "kernel_size", "win_type"]
    agg = df.groupby(group_cols, dropna=False).agg(
        cases=("case_id", "count"),
        **{f"mean_{m}": (m, "mean") for m in metrics},
        **{f"std_{m}": (m, "std") for m in metrics},
        mean_seconds=("seconds", "mean"),
    ).reset_index()
    agg = agg.sort_values(["mean_ARI", "mean_NMI"], ascending=[False, False])
    agg.to_csv(summary_path, index=False)
    best = agg.iloc[0]
    mask = (df["win_size"] == best["win_size"]) & (df["step"] == best["step"])
    df[mask].to_csv(best_path, index=False)
    grid_mean = {m: float(df[m].mean()) for m in metrics}
    best_dict = {m: float(best[f"mean_{m}"]) for m in metrics}
    return {
        "grid_mean": grid_mean,
        "best_by_mean_ARI": {
            "win_size": int(best["win_size"]),
            "step": int(best["step"]),
            **best_dict,
        },
        "n_rows": int(len(df)),
    }


def paper_fixed_for_dataset(key: str) -> tuple[list[int], list[int]]:

    mapping = {
        "synthetic": ([256], [50]),
        "mocap": ([256], [50]),
        "actrectut": ([128], [1]),
        "pamap2": ([512], [100]),
        "pamap2_zero": ([512], [100]),
        "usc-had": ([512], [50]),
        "ucrseg": ([512], [50]),
    }
    return mapping[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2USD baseline on selected state-detection datasets.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--e2usd-root", type=Path, default=None, help="Directory containing the E2USD/ package, e.g. D:/code/teacherT2S/E2Usd-main")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional root containing data folders used by E2USD.")
    parser.add_argument("--public-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--win-size", default="128,256,512")
    parser.add_argument("--step", default="50,100")
    parser.add_argument("--paper-fixed", action="store_true", help="Use the fixed settings in uploaded E2USD train.py main block instead of a grid.")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--nb-steps", type=int, default=20)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--win-type", default="rect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the same setting multiple times for stochastic E2USD runs. Original train.py loops 10 rounds.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--cp-margin-ratio", type=float, default=0.01)
    parser.add_argument("--ucrseg-dirname", default="")
    parser.add_argument("--pamap2-feature-mode", choices=["paper9acc", "full_sensor"], default="paper9acc")
    parser.add_argument("--pamap2-remove-zero", action="store_true", help="For PAMAP2_zero: remove frames whose activity_id is 0.")
    return parser.parse_args()


def run_selected_e2usd(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    e2usd_root = detect_e2usd_root(repo_root, args.e2usd_root).resolve()
    runtime = import_runtime(repo_root, e2usd_root)
    data_roots = default_data_roots(repo_root, e2usd_root, args.data_root, args.public_data_root)
    key = normalize_dataset_key(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    if args.paper_fixed:
        win_sizes, steps = paper_fixed_for_dataset(key)
    else:
        win_sizes = split_ints(args.win_size, [128, 256, 512])
        steps = split_ints(args.step, [50, 100])

    print("E2USD root:", e2usd_root)
    print("Dataset   :", key)
    print("Win sizes :", win_sizes)
    print("Steps     :", steps)
    print("Repeats   :", args.repeats)
    print("Paper fixed:", bool(args.paper_fixed))
    print("PAMAP2 feature_mode:", getattr(args, "pamap2_feature_mode", ""))
    print("PAMAP2 remove_zero :", int(bool(getattr(args, "pamap2_remove_zero", False))))
    print("Data roots searched:")
    for r in data_roots:
        print("  -", r)

    cases = load_cases(key, data_roots, runtime, args)
    print(f"Loaded {len(cases)} case(s).")

    case_csv = args.out_dir / "case_results.csv"
    summary_csv = args.out_dir / "algorithm_summary.csv"
    best_csv = args.out_dir / "best_by_mean_ARI_case_results.csv"
    status_json = args.out_dir / "run_status.json"

    fieldnames = [
        "dataset", "case_id", "repeat", "win_size", "step", "M", "N", "out_channels", "nb_steps", "kernel_size", "win_type",
        "true_states", "pred_states", "seconds", "ARI", "NMI", "AMI", "Covering", "CP_F1", "cp_margin",
    ]

    existing = set()
    if args.skip_completed and case_csv.exists():
        import pandas as pd
        old = pd.read_csv(case_csv)
        for _, row in old.iterrows():
            existing.add((str(row["case_id"]), int(row["repeat"]), int(row["win_size"]), int(row["step"])))

    grid_idx = 0
    for win in win_sizes:
        for step in steps:
            grid_idx += 1
            print(f"\n[GRID {grid_idx}/{len(win_sizes) * len(steps)}] win_size={win}, step={step}")
            for rep in range(max(1, int(args.repeats))):
                seed_everything(int(args.seed) + rep * 10000 + grid_idx, runtime)
                rows = []

                if key in {"pamap2", "pamap2_zero", "usc-had"} and cases:
                    train_case = cases[0]
                    start = time.time()
                    preds = run_with_prefit(train_case, cases, win, step, runtime, args, key)
                    total_elapsed = time.time() - start
                    for case, pred in zip(cases, preds):
                        if (case.case_id, rep, win, step) in existing:
                            continue
                        metrics = compute_metrics(case.labels, pred, runtime, args.cp_margin_ratio)
                        pred_aligned = align_sequence(pred, len(case.labels), runtime["np"])
                        sec = total_elapsed / max(1, len(cases))
                        rows.append({
                            "dataset": key,
                            "case_id": case.case_id,
                            "repeat": rep,
                            "win_size": win,
                            "step": step,
                            "M": args.m,
                            "N": args.n,
                            "out_channels": 9 if (args.out_channels is None and key == "pamap2") else (args.out_channels or 4),
                            "nb_steps": args.nb_steps,
                            "kernel_size": args.kernel_size,
                            "win_type": args.win_type,
                            "true_states": len(set(map(int, case.labels))),
                            "pred_states": len(set(map(int, pred_aligned))),
                            "seconds": sec,
                            **metrics,
                        })
                        print(f"  {key}/{case.case_id}: ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} AMI={metrics['AMI']:.4f} Covering={metrics['Covering']:.4f} K={rows[-1]['pred_states']} seconds={sec:.1f}")
                        if args.save_predictions:
                            runtime["np"].savez_compressed(args.out_dir / "predictions" / f"{key}_{case.case_id}_r{rep}_w{win}_s{step}.npz", y_true=case.labels, y_pred=pred_aligned)
                else:
                    for case in cases:
                        if (case.case_id, rep, win, step) in existing:
                            continue
                        start = time.time()
                        pred, _model = run_one_case(case, win, step, runtime, args, key)
                        elapsed = time.time() - start
                        metrics = compute_metrics(case.labels, pred, runtime, args.cp_margin_ratio)
                        pred_aligned = align_sequence(pred, len(case.labels), runtime["np"])
                        rows.append({
                            "dataset": key,
                            "case_id": case.case_id,
                            "repeat": rep,
                            "win_size": win,
                            "step": step,
                            "M": args.m,
                            "N": args.n,
                            "out_channels": args.out_channels or 4,
                            "nb_steps": args.nb_steps,
                            "kernel_size": args.kernel_size,
                            "win_type": args.win_type,
                            "true_states": len(set(map(int, case.labels))),
                            "pred_states": len(set(map(int, pred_aligned))),
                            "seconds": elapsed,
                            **metrics,
                        })
                        print(f"  {key}/{case.case_id}: ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} AMI={metrics['AMI']:.4f} Covering={metrics['Covering']:.4f} K={rows[-1]['pred_states']} seconds={elapsed:.1f}")
                        if args.save_predictions:
                            runtime["np"].savez_compressed(args.out_dir / "predictions" / f"{key}_{case.case_id}_r{rep}_w{win}_s{step}.npz", y_true=case.labels, y_pred=pred_aligned)
                if rows:
                    append_csv(case_csv, rows, fieldnames)
                    summary = summarize(case_csv, summary_csv, best_csv)
                    status_json.write_text(json.dumps({
                        "status": "running",
                        "dataset": key,
                        "e2usd_root": str(e2usd_root),
                        "win_sizes": win_sizes,
                        "steps": steps,
                        "repeats": args.repeats,
                        "current_summary": summary,
                    }, indent=2), encoding="utf-8")

    summary = summarize(case_csv, summary_csv, best_csv)
    status_json.write_text(json.dumps({
        "status": "ok",
        "dataset": key,
        "e2usd_root": str(e2usd_root),
        "win_sizes": win_sizes,
        "steps": steps,
        "repeats": args.repeats,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print("\nDONE")
    print("case_results:", case_csv)
    print("summary     :", summary_csv)
    print("best rows   :", best_csv)
    return 0


def main() -> int:
    args = parse_args()
    return run_selected_e2usd(args)


if __name__ == "__main__":
    raise SystemExit(main())
