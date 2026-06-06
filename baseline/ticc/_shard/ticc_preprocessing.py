from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any



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
class TICCCase:
    dataset: str
    case_id: str
    data: Any
    labels: Any
    n_clusters: int
    window_size: int
    max_iters: int
    num_proc_default: int
    source_path: str
    protocol: str


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    aliases = {
        "mocap": "mocap",
        "synthetic": "synthetic",
        "synthetic2": "synthetic",
        "actrectut": "actrectut",
        "act-rec-tut": "actrectut",
        "pamap2": "pamap2",
        "pamap2-zero": "pamap2-zero",
        "pamap2zero": "pamap2-zero",
        "pamap2_zero": "pamap2-zero",
        "usc-had": "usc-had",
        "uschad": "usc-had",
        "ucrseg": "ucr-seg",
        "ucr-seg": "ucr-seg",
        "tssb": "ucr-seg",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset={name!r}. Supported: mocap, synthetic, actrectut, pamap2, uschad, ucrseg")
    return aliases[key]


def parse_dataset_list(text) -> list[str]:
    """Parse dataset names from either a command-line string or argparse list.

    argparse with nargs="+" returns a list such as ["mocap"].  The old
    implementation converted that list with str(text), producing "['mocap']"
    and then failing normalize_dataset_key().  Keep this function tolerant so
    both launcher styles work.
    """
    if text is None:
        parts = []
    elif isinstance(text, (list, tuple, set)):
        parts = []
        for item in text:
            parts.extend(str(item).replace(",", " ").replace(";", " ").split())
    else:
        parts = str(text).replace(",", " ").replace(";", " ").split()

    parts = [p.strip().strip("'\"[]()") for p in parts if p and p.strip()]
    if not parts:
        raise ValueError("Empty dataset list")
    out = []
    for p in parts:
        key = normalize_dataset_key(p)
        if key not in out:
            out.append(key)
    return out


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


def fill_nan_original(data, np):
    """Exact PAMAP2 NaN fill logic from the original TICC experiment."""
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


def import_runtime(repo_root: Path):
    """Import numeric/data dependencies and optional TSpy utilities."""


    candidates = [
        repo_root / "TSpy-dev",
        Path(r"D:\code\teacherT2S\TSpy-dev"),
        repo_root.parent / "TSpy-dev",
    ]
    for cand in candidates:
        if cand.exists():
            s = str(cand)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
            break

    import numpy as np
    import pandas as pd
    import scipy.io

    try:
        from TSpy.utils import normalize as tspy_normalize
    except Exception:
        tspy_normalize = None

    try:
        from TSpy.dataset import load_USC_HAD as tspy_load_USC_HAD
    except Exception:
        tspy_load_USC_HAD = None

    return {
        "np": np,
        "pd": pd,
        "scipy_io": scipy.io,
        "tspy_normalize": tspy_normalize,
        "tspy_load_USC_HAD": tspy_load_USC_HAD,
    }


def original_normalize(data, runtime):
    norm = runtime.get("tspy_normalize")
    if norm is not None:
        return norm(data)
    return fallback_normalize(data, runtime["np"])


def default_data_root(repo_root: Path) -> Path:
    return repo_root / "Time2State" / "data"


def find_existing_dir(candidates: list[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {what}. Tried: " + " | ".join(str(p) for p in candidates))


def load_mocap(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC MoCap preprocessing.

    Original code:
      pd.read_csv(file, sep=' ', usecols=range(0,4))
      window_size=5; number_of_clusters=n_segs; maxIters=10
      groundtruth = seg_to_label(label)[:-5]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = data_root / "MoCap" / "4d"
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases: list[TICCCase] = []
    for path in files:
        if path.name not in MOCAP_INFO:
            continue
        data = pd.read_csv(path, sep=" ", usecols=range(0, 4)).to_numpy()
        labels = seg_to_label(MOCAP_INFO[path.name]["label"], np)[:-5]
        cases.append(TICCCase(
            dataset="MoCap", case_id=path.name, data=data, labels=labels,
            n_clusters=int(MOCAP_INFO[path.name]["n_segs"]), window_size=5,
            max_iters=10, num_proc_default=10, source_path=str(path),
            protocol="strict_TICC_MoCap_4d_usecols_0_3_oracle_K_labels_minus_5",
        ))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No MoCap .4d cases found under {base}")
    return cases


def load_actrectut(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC ActRecTut preprocessing.

    Original code:
      dir_list = ['subject1_walk', 'subject2_walk']; repeat each 10 times
      labels = data['labels'].flatten()[:-2]
      data = data['data'][:,0:10]
      window_size=3; number_of_clusters=len(set(groundtruth)); maxIters=10
    """
    np = runtime["np"]
    scipy_io = runtime["scipy_io"]
    base = data_root / "ActRecTut"
    cases: list[TICCCase] = []
    for name in ["subject1_walk", "subject2_walk"]:
        mat_path = base / name / "data.mat"
        mat = scipy_io.loadmat(mat_path)
        labels = np.asarray(mat["labels"].flatten()[:-2], dtype=int)
        data = mat["data"][:, 0:10]
        n_state = int(len(set(map(int, labels))))
        for rep in range(10):
            cases.append(TICCCase(
                dataset="ActRecTut", case_id=f"{name}{rep}", data=data, labels=labels,
                n_clusters=n_state, window_size=3, max_iters=10, num_proc_default=10,
                source_path=str(mat_path), protocol="strict_TICC_ActRecTut_raw_first10dims_repeat10_labels_minus_2",
            ))
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def load_ucrseg(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC UCR-SEG preprocessing.

    Original code:
      change points are parsed from fname[:-4].split('_')[2:]
      pd.read_csv(file); no normalize
      window_size=3; number_of_clusters=len(seg_info); maxIters=10
      groundtruth = seg_to_label(seg_info)[win_size:]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = data_root / "UCR-SEG" / "UCR_datasets_seg"
    files = sorted([p for p in base.iterdir() if p.is_file()], key=lambda p: p.name)
    cases: list[TICCCase] = []
    for path in files:
        if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        info_list = path.name[:-4].split("_")
        if len(info_list) < 3:
            continue
        seg_info: dict[int, int] = {}
        for i, seg in enumerate(info_list[2:]):
            seg_info[int(seg)] = i
        seg_info[count_file_lines(path)] = len(info_list[2:])
        data = pd.read_csv(path).to_numpy()
        win_size = 3
        labels = seg_to_label(seg_info, np)[win_size:]
        cases.append(TICCCase(
            dataset="UCR-SEG", case_id=path.name[:-4], data=data, labels=labels,
            n_clusters=int(len(seg_info)), window_size=win_size, max_iters=10,
            num_proc_default=10, source_path=str(path),
            protocol="strict_TICC_UCRSEG_change_points_from_filename_labels_from_win_size_to_end",
        ))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No UCR-SEG cases found under {base}")
    return cases


def load_pamap2(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC PAMAP2 preprocessing.

    Original code:
      df = pd.read_csv(subject10i.dat, sep=' ', header=None)
      groundtruth = data[:,1]
      features = hand_acc 4:7 + chest_acc 21:24 + ankle_acc 38:41
      fill_nan(data); data=data[::20]; groundtruth=groundtruth[::20][:-2]
      window_size=3; maxIters=3; num_proc=1
    """
    np = runtime["np"]
    pd = runtime["pd"]
    protocol_dir = find_existing_dir([
        data_root / "PAMAP2" / "Protocol",
        data_root / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
    ], "PAMAP2 Protocol directory")
    cases: list[TICCCase] = []
    for i in range(1, 9):
        path = protocol_dir / f"subject10{i}.dat"
        if not path.exists():
            continue
        raw = pd.read_csv(path, sep=" ", header=None).to_numpy()
        labels_all = np.asarray(raw[:, 1], dtype=int)
        hand_acc = raw[:, 4:7]
        chest_acc = raw[:, 21:24]
        ankle_acc = raw[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan_original(data, np)
        data = data[::20, :]
        labels = labels_all[::20][:-2]
        n_state = int(len(set(map(int, labels))))
        cases.append(TICCCase(
            dataset="PAMAP2", case_id=str(i), data=data, labels=labels,
            n_clusters=n_state, window_size=3, max_iters=3, num_proc_default=1,
            source_path=str(path), protocol="strict_TICC_PAMAP2_9acc_downsample20_labels_minus_2_oracle_K",
        ))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No PAMAP2 subjects found under {protocol_dir}")
    return cases


def load_pamap2_zero(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """PAMAP2_zero extension used by the user's zero-label experiment.

    Compared with the strict original TICC PAMAP2 loader, this controlled
    variant uses all sensor columns after timestamp/activity_id and removes
    activity_id=0 before the original downsampling-by-20 step. It is separated
    under the dataset key pamap2_zero / pamap2-zero so the original pamap2
    loader remains unchanged.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    protocol_dir = find_existing_dir([
        data_root / "PAMAP2" / "Protocol",
        data_root / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
    ], "PAMAP2 Protocol directory")
    cases: list[TICCCase] = []
    for i in range(1, 9):
        path = protocol_dir / f"subject10{i}.dat"
        if not path.exists():
            continue
        raw = pd.read_csv(path, sep=" ", header=None)
        numeric = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        labels_all = np.asarray(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)
        data = numeric[:, 2:]
        data = fill_nan_original(data, np)
        valid = labels_all > 0
        data = data[valid]
        labels_all = labels_all[valid]
        data = data[::20, :]
        labels = labels_all[::20]
        if len(labels) > 2:
            labels = labels[:-2]
        n_state = int(len(set(map(int, labels)))) if len(labels) else 0
        if n_state < 1:
            raise ValueError(f"PAMAP2_zero subject10{i} has no valid states after removing activity_id=0")
        cases.append(TICCCase(
            dataset="PAMAP2_zero", case_id=str(i), data=data, labels=labels,
            n_clusters=n_state, window_size=3, max_iters=3, num_proc_default=1,
            source_path=str(path), protocol="TICC_PAMAP2_zero_full_sensor_remove_activity0_downsample20_oracle_K",
        ))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No PAMAP2_zero subjects found under {protocol_dir}")
    return cases


def load_synthetic(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC synthetic2 preprocessing.

    Original code:
      base_path = data/synthetic_data_for_segmentation2/
      pd.read_csv(file, usecols=range(0,4)); label=usecols=[4]
      window_size=5; number_of_clusters=len(set(label)); maxIters=10
      groundtruth = label[:-4]
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = data_root / "synthetic_data_for_segmentation2"
    cases: list[TICCCase] = []
    for i in range(100):
        path = base / f"test{i}.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=range(0, 4)).to_numpy()
        label = pd.read_csv(path, usecols=[4]).to_numpy().flatten()
        labels = np.asarray(label[:-4], dtype=int)
        n_state = int(len(set(map(int, label))))
        cases.append(TICCCase(
            dataset="Synthetic", case_id=str(i), data=data, labels=labels,
            n_clusters=n_state, window_size=5, max_iters=10, num_proc_default=10,
            source_path=str(path), protocol="strict_TICC_synthetic2_first4cols_label_col4_labels_minus_4_oracle_K",
        ))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No synthetic test*.csv files found under {base}")
    return cases


def load_uschad(data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    """Strict original TICC USC-HAD preprocessing.

    Original code:
      for subject in 1..14, target in 1..5:
        data, groundtruth = load_USC_HAD(subject, target, data_path)
        data = normalize(data)
        groundtruth = groundtruth[:-2]
        window_size=3; number_of_clusters=len(set(groundtruth)); maxIters=10
    """
    np = runtime["np"]
    load_USC_HAD = runtime.get("tspy_load_USC_HAD")
    if load_USC_HAD is None:
        raise RuntimeError("TSpy.dataset.load_USC_HAD is unavailable. Install/enable TSpy-dev for strict USC-HAD TICC baseline.")
    data_path = str(data_root) + os.sep
    cases: list[TICCCase] = []
    for subject in range(1, 15):
        for target in range(1, 6):
            data, labels_all = load_USC_HAD(subject, target, data_path)
            data = original_normalize(data, runtime)
            labels = np.asarray(labels_all[:-2], dtype=int)
            n_state = int(len(set(map(int, labels))))
            cases.append(TICCCase(
                dataset="USC-HAD", case_id=f"s{subject}_t{target}", data=data, labels=labels,
                n_clusters=n_state, window_size=3, max_iters=10, num_proc_default=1,
                source_path=f"load_USC_HAD({subject},{target},{data_path})",
                protocol="strict_TICC_USCHAD_TSpy_load_USC_HAD_normalize_labels_minus_2_oracle_K",
            ))
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


LOADERS = {
    "mocap": load_mocap,
    "actrectut": load_actrectut,
    "ucr-seg": load_ucrseg,
    "pamap2": load_pamap2,
    "pamap2-zero": load_pamap2_zero,
    "synthetic": load_synthetic,
    "usc-had": load_uschad,
}


def load_cases(dataset_keys: list[str], data_root: Path, runtime, max_cases: int | None = None) -> list[TICCCase]:
    cases: list[TICCCase] = []
    for key in dataset_keys:
        key = normalize_dataset_key(key)
        loader = LOADERS[key]
        ds_cases = loader(data_root, runtime, max_cases=max_cases)
        cases.extend(ds_cases)
    return cases
