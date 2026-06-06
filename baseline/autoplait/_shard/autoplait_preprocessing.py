from __future__ import annotations

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
class SeriesCase:
    dataset: str
    case_id: str
    data: Any
    labels: Any
    note: str = ""


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    aliases = {
        "mocap": "mocap",
        "mo-cap": "mocap",
        "synthetic": "synthetic",
        "ucrseg": "ucrseg",
        "ucr-seg": "ucrseg",
        "tssb": "ucrseg",
        "actrectut": "actrectut",
        "actrec-tut": "actrectut",
        "usc-had": "uschad",
        "uschad": "uschad",
        "pamap2": "pamap2",
        "pamap2-zero": "pamap2-zero",
        "pamap2_zero": "pamap2-zero",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset={name!r}")
    return aliases[key]


def split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).replace(",", " ").replace(";", " ").strip()
    return [part for part in text.split() if part]


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


def fill_nan_forward(data, np):
    arr = np.asarray(data, dtype=float).copy()
    if arr.ndim != 2:
        arr = arr.reshape(-1, 1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(arr[i, j]):
                arr[i, j] = arr[i - 1, j] if i > 0 else 0.0
    return arr


def find_existing_dir(candidates: list[Path], what: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {what}. Tried: " + " | ".join(str(x) for x in candidates))


def count_file_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def load_synthetic(data_root: Path, runtime: dict, max_cases: int | None = None) -> list[SeriesCase]:
    """
    Synthetic path used in your existing Time2State runs:
      Time2State/data/synthetic_data_for_segmentation3/test0.csv ... test99.csv

    The C AutoPlait program internally calls ZnormSequence(), so this loader does
    not normalize data in Python.
    """
    np = runtime["np"]
    pd = runtime["pd"]
    base = find_existing_dir([
        data_root / "synthetic_data_for_segmentation3",
        data_root / "synthetic_data_for_segmentation",
        data_root / "synthetic_data_for_segmentation2",
    ], "Synthetic data directory")
    cases: list[SeriesCase] = []
    for i in range(100):
        path = base / f"test{i}.csv"
        if not path.exists():
            continue
        df_x = pd.read_csv(path, usecols=range(4), skiprows=1)
        df_y = pd.read_csv(path, usecols=[4], skiprows=1)
        data = df_x.to_numpy(dtype=float)
        labels = df_y.to_numpy(dtype=int).flatten()
        n = min(len(data), len(labels))
        cases.append(SeriesCase("Synthetic", str(i), data[:n], labels[:n], note=str(path)))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No synthetic test*.csv found under {base}")
    return cases


def load_mocap(data_root: Path, runtime: dict, max_cases: int | None = None) -> list[SeriesCase]:
    np = runtime["np"]
    pd = runtime["pd"]
    base = find_existing_dir([
        data_root / "MoCap" / "4d",
        data_root / "mocap" / "4d",
    ], "MoCap/4d directory")
    files = sorted(base.glob("*.4d"), key=lambda p: p.name)
    cases: list[SeriesCase] = []
    for path in files:
        if path.name not in MOCAP_INFO:
            continue
        df = pd.read_csv(path, sep=" ", usecols=range(0, 4))
        data = df.to_numpy(dtype=float)

        labels = seg_to_label(MOCAP_INFO[path.name]["label"], np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("MoCap", path.name, data[:n], labels[:n], note=str(path)))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No MoCap cases found under {base}")
    return cases


def load_ucrseg(data_root: Path, runtime: dict, max_cases: int | None = None) -> list[SeriesCase]:
    np = runtime["np"]
    pd = runtime["pd"]
    base = find_existing_dir([
        data_root / "UCR-SEG" / "UCR_datasets_seg",
        data_root / "UCR-SEG",
        data_root / "UCR_SEG",
    ], "UCR-SEG directory")
    files = sorted([p for p in base.iterdir() if p.is_file()], key=lambda p: p.name)
    cases: list[SeriesCase] = []
    for path in files:
        if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        info = path.stem.split("_")
        if len(info) < 3:
            continue
        seg_info = {}
        for i, seg in enumerate(info[2:]):
            try:
                seg_info[int(seg)] = i
            except ValueError:
                pass
        if not seg_info:
            continue
        seg_info[count_file_lines(path)] = len(seg_info)
        df = pd.read_csv(path)
        data = df.to_numpy(dtype=float)
        labels = seg_to_label(seg_info, np)[:-1]
        n = min(len(data), len(labels))
        cases.append(SeriesCase("UCR-SEG", path.stem, data[:n], labels[:n], note=str(path)))
        if max_cases is not None and len(cases) >= max_cases:
            break
    if not cases:
        raise FileNotFoundError(f"No UCR-SEG cases found under {base}")
    return cases


def load_actrectut(data_root: Path, runtime: dict, max_cases: int | None = None) -> list[SeriesCase]:
    np = runtime["np"]
    scipy_io = runtime["scipy_io"]
    base = find_existing_dir([data_root / "ActRecTut", data_root / "ActRecTut"], "ActRecTut directory")
    names = ["subject1_walk", "subject2_walk"]
    cases: list[SeriesCase] = []
    for name in names:
        mat_path = base / name / "data.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing ActRecTut mat file: {mat_path}")
        mat = scipy_io.loadmat(mat_path)
        labels_raw = mat["labels"].flatten()
        labels, _ = reorder_label(labels_raw, np)
        data = mat["data"][:, 0:10].astype(float)
        n = min(len(data), len(labels))



        for rep in range(10):
            cases.append(SeriesCase("ActRecTut", f"{name}{rep}", data[:n], labels[:n], note=str(mat_path)))
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def load_pamap2(data_root: Path, runtime: dict, max_cases: int | None, *, zero: bool, downsample: int = 1) -> list[SeriesCase]:
    """
    Robust PAMAP2 / PAMAP2_zero loader.

    Supports these layouts:
      1) data_root/PAMAP2/Protocol/subject101.dat
      2) data_root/PAMAP/Protocol/subject101.dat
      3) data_root/PAMAP2/subject101.dat
      4) recursive search under data_root/PAMAP2
    """
    np = runtime["np"]
    pd = runtime["pd"]

    candidates = [
        data_root / "PAMAP2" / "Protocol",
        data_root / "PAMAP" / "Protocol",
        data_root / "PAMAP2",
        data_root / "PAMAP",
        data_root,
    ]

    subject_paths: dict[int, Path] = {}

    for base in candidates:
        if not base.exists():
            continue


        for i in range(1, 9):
            p = base / f"subject10{i}.dat"
            if p.exists():
                subject_paths[i] = p


        for p in base.rglob("subject10*.dat"):
            name = p.name.lower()
            for i in range(1, 9):
                if name == f"subject10{i}.dat":
                    subject_paths.setdefault(i, p)

        if subject_paths:
            break

    if not subject_paths:
        raise FileNotFoundError(
            "Cannot find PAMAP2 subject101.dat ~ subject108.dat. Tried: "
            + " | ".join(str(x) for x in candidates)
        )

    print("[PAMAP2] Found subject files:", flush=True)
    for i in sorted(subject_paths):
        print(f"  subject10{i}: {subject_paths[i]}", flush=True)

    cases: list[SeriesCase] = []

    for i in range(1, 9):
        path = subject_paths.get(i)
        if path is None:
            continue

        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        labels = np.asarray(np.nan_to_num(numeric[:, 1], nan=0.0), dtype=int)

        if zero:

            data = numeric[:, 2:]
            valid = labels > 0
            data = data[valid]
            labels = labels[valid]
        else:

            hand_acc = numeric[:, 4:7]
            chest_acc = numeric[:, 21:24]
            ankle_acc = numeric[:, 38:41]
            data = np.hstack([hand_acc, chest_acc, ankle_acc])

        data = fill_nan_forward(data, np)

        if downsample and int(downsample) > 1:
            data = data[:: int(downsample), :]
            labels = labels[:: int(downsample)]

        labels, _ = reorder_label(labels, np)

        n = min(len(data), len(labels))
        if n <= 1:
            continue

        cases.append(
            SeriesCase(
                "PAMAP2_zero" if zero else "PAMAP2",
                f"subject10{i}",
                data[:n],
                labels[:n],
                note=str(path),
            )
        )

        if max_cases is not None and len(cases) >= max_cases:
            break

    if not cases:
        raise FileNotFoundError("PAMAP2 files were found, but no valid cases were loaded.")

    return cases


def load_uschad(data_root: Path, runtime: dict, max_cases: int | None = None) -> list[SeriesCase]:
    """
    USC-HAD loader for AutoPlait baseline.

    E2USD official train.py uses:
        load_USC_HAD(subject, target, data_path)

    where data_path is the root data directory, not data_path/USC-HAD.
    Therefore this loader first tries data_root itself.
    """
    np = runtime["np"]
    loader = runtime.get("tspy_load_USC_HAD")

    if loader is None:
        raise RuntimeError(
            "TSpy.dataset.load_USC_HAD is not available. "
            "Check TSpy-dev installation and import priority."
        )






    candidates = [
        data_root,
        data_root.parent / "Baselines" / "public_ts_datasets",
        data_root / "USC-HAD",
        data_root / "USC_HAD",
        data_root.parent / "Baselines" / "public_ts_datasets" / "USC-HAD",
        data_root.parent / "Baselines" / "public_ts_datasets" / "USC_HAD",
    ]

    dataset_paths = []
    seen = set()
    for p in candidates:
        p = Path(p)
        if p.exists():
            key = str(p.resolve()).lower()
            if key not in seen:
                dataset_paths.append(p)
                seen.add(key)

    if not dataset_paths:
        raise FileNotFoundError(
            "Cannot find USC-HAD-related data directory. Tried: "
            + " | ".join(str(p) for p in candidates)
        )

    cases: list[SeriesCase] = []
    errors: list[str] = []

    def add_case(data, labels, case_id: str, note: str) -> bool:
        try:
            arr = np.asarray(data, dtype=float)
            y = np.asarray(labels).flatten()

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)


            if arr.ndim == 2 and arr.shape[0] != len(y) and arr.shape[1] == len(y):
                arr = arr.T

            y, _ = reorder_label(y, np)

            n = min(len(arr), len(y))
            if n <= 1:
                return False

            cases.append(
                SeriesCase(
                    "USC-HAD",
                    case_id,
                    arr[:n],
                    y[:n],
                    note=note,
                )
            )
            return True
        except Exception as exc:
            errors.append(f"add_case failed for {case_id}: {repr(exc)}")
            return False

    def consume_loaded(loaded, subject: int, target: int, dataset_path: Path) -> bool:
        before = len(cases)
        note = f"TSpy.load_USC_HAD(subject={subject}, target={target}, dataset_path={dataset_path})"
        case_prefix = f"subject{subject}_target{target}"

        if loaded is None:
            return False


        if isinstance(loaded, tuple) and len(loaded) >= 2:
            data, labels = loaded[0], loaded[1]

            if isinstance(data, (list, tuple)) and isinstance(labels, (list, tuple)) and len(data) == len(labels):
                for j, (xj, yj) in enumerate(zip(data, labels), start=1):
                    add_case(xj, yj, f"{case_prefix}_case{j}", note)
            else:
                add_case(data, labels, case_prefix, note)


        elif isinstance(loaded, dict):
            for key, value in loaded.items():
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    add_case(value[0], value[1], f"{case_prefix}_{key}", note)


        elif isinstance(loaded, list):
            for j, item in enumerate(loaded, start=1):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    add_case(item[0], item[1], f"{case_prefix}_case{j}", note)

        return len(cases) > before


    for subject in range(1, 15):
        for target in range(1, 6):
            if max_cases is not None and len(cases) >= max_cases:
                return cases[:max_cases]

            added = False

            for dataset_path in dataset_paths:
                try:
                    loaded = loader(subject, target, str(dataset_path))
                    added = consume_loaded(loaded, subject, target, dataset_path)
                    if added:
                        print(
                            f"[USCHAD] loaded subject={subject}, target={target}, data_root={dataset_path}",
                            flush=True,
                        )
                        break
                except Exception as exc:
                    errors.append(
                        f"subject={subject}, target={target}, dataset_path={dataset_path}: {repr(exc)}"
                    )


            if not added:
                continue

    if max_cases is not None:
        cases = cases[:max_cases]

    if not cases:
        raise RuntimeError(
            "Could not load USC-HAD through "
            "TSpy.dataset.load_USC_HAD(subject, target, dataset_path). "
            "Recent errors:\n"
            + "\n".join(errors[-40:])
        )

    print(f"[USCHAD] total loaded cases: {len(cases)}", flush=True)
    return cases

    consecutive_misses = 0


    for target in range(1, 71):
        if max_cases is not None and len(cases) >= max_cases:
            break

        added_this_target = False

        for dataset_path in dataset_paths:
            try:
                loaded = loader(target, str(dataset_path))
                added_this_target = consume_loaded(loaded, target, dataset_path)
                if added_this_target:
                    break
            except Exception as exc:
                errors.append(
                    f"target={target}, dataset_path={dataset_path}: {repr(exc)}"
                )

        if added_this_target:
            consecutive_misses = 0
        else:
            consecutive_misses += 1


        if cases and consecutive_misses >= 10:
            break

    if max_cases is not None:
        cases = cases[:max_cases]

    if not cases:
        raise RuntimeError(
            "Could not load USC-HAD through TSpy.dataset.load_USC_HAD(target, dataset_path). "
            "Recent errors:\n"
            + "\n".join(errors[-20:])
        )

    return cases


def load_cases(dataset_keys: list[str], data_root: Path, runtime: dict, max_cases: int | None = None, args=None) -> list[SeriesCase]:
    out: list[SeriesCase] = []
    for raw in dataset_keys:
        key = normalize_dataset_key(raw)
        if key == "synthetic":
            out.extend(load_synthetic(data_root, runtime, max_cases))
        elif key == "mocap":
            out.extend(load_mocap(data_root, runtime, max_cases))
        elif key == "ucrseg":
            out.extend(load_ucrseg(data_root, runtime, max_cases))
        elif key == "actrectut":
            out.extend(load_actrectut(data_root, runtime, max_cases))
        elif key == "pamap2":
            downsample = int(getattr(args, "pamap2_downsample", 20) if args is not None else 20)
            out.extend(load_pamap2(data_root, runtime, max_cases, zero=False, downsample=downsample))
        elif key == "pamap2-zero":
            downsample = int(getattr(args, "pamap2_downsample", 1) if args is not None else 1)
            out.extend(load_pamap2(data_root, runtime, max_cases, zero=True, downsample=downsample))
        elif key == "uschad":
            out.extend(load_uschad(data_root, runtime, max_cases))
        else:
            raise ValueError(f"Unsupported dataset key={key}")
    return out
