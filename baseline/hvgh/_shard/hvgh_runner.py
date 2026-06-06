
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

from hvgh_preprocessing import import_runtime, load_cases, reorder_label, align_prediction
from prepare_hvgh_impl import build as prepare_hvgh_impl


FIELDNAMES = [
    "dataset", "case_id", "ARI", "NMI", "states", "seconds", "attempts", "status", "error",
    "prediction_file", "raw_file", "log_file", "hvgh_win_size", "hvgh_epoch", "hvgh_iteration",
]

HISTORY_FIELDNAMES = [
    "time", "dataset", "case_id", "attempt", "status", "ARI", "NMI", "states", "seconds",
    "error", "prediction_file", "raw_file", "log_file", "hvgh_win_size", "hvgh_epoch", "hvgh_iteration",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return default


def parse_args():
    p = argparse.ArgumentParser(description="HVGH strict baseline runner with retry/accumulate support")
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--skip-completed", action="store_true", help="Compatibility flag. Successful cases are accumulated/skipped by default unless --rerun-completed is used.")
    p.add_argument("--rerun-completed", action="store_true", help="Ignore existing successful case_results rows and rerun all loaded cases.")
    p.add_argument("--max-retries", type=int, default=_env_int("HVGH_MAX_RETRIES", 5), help="Retry each failed case up to this many attempts. Default: env HVGH_MAX_RETRIES or 5.")
    p.add_argument("--eval-existing-only", action="store_true", help="Do not train; only evaluate existing HVGHlearn outputs.")
    p.add_argument("--epoch", type=int, default=None, help="Override HVGH epoch for all cases.")
    p.add_argument("--iteration", type=int, default=None, help="Override HVGH mutual-learning iterations for all cases.")
    p.add_argument("--win-size", type=int, default=None, help="Override HVGH subsampling/window dilation size for all cases.")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--initial-class", type=int, default=None)
    return p.parse_args()


def safe_case_id(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out)


def case_key(dataset: str, case_id: str) -> tuple[str, str]:
    return (str(dataset), str(case_id))


def dilate_label(label, factor: int, max_len: int, np):
    label = np.asarray(label, dtype=int).flatten()
    if len(label) == 0:
        return np.zeros(max_len, dtype=int)
    pred = np.repeat(label, int(factor))
    return align_prediction(pred, max_len, np)


def read_hvgh_raw_labels(raw_file: Path, np):
    arr = np.loadtxt(raw_file)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        raw = arr.astype(int)
    else:
        raw = arr[:, 0].astype(int)
    return raw


def evaluate_case(case, pred, runtime):
    np = runtime["np"]
    y_true = np.asarray(case.labels, dtype=int).flatten()
    y_pred = align_prediction(pred, len(y_true), np)
    y_pred = reorder_label(y_pred, np)
    ari = float(runtime["adjusted_rand_score"](y_true, y_pred))
    nmi = float(runtime["normalized_mutual_info_score"](y_true, y_pred, average_method="geometric"))
    return ari, nmi, y_pred


def import_hvgh_class():
    """
    Import HVGH as a real package so that relative imports such as
    from .hvgh_gp import GP work correctly. The generated implementation is
    refreshed before import to avoid stale modules from earlier failed runs.
    """
    gen_dir = prepare_hvgh_impl()
    try:
        import pyximport
    except Exception as exc:
        raise RuntimeError("Cython/pyximport is required for HVGH. Install with: pip install cython") from exc

    import numpy as np
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

    parent = str(gen_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    for name in list(sys.modules):
        if name == gen_dir.name or name.startswith(gen_dir.name + "."):
            sys.modules.pop(name, None)

    import importlib
    mod = importlib.import_module(f"{gen_dir.name}.hvgh_model")
    return mod.HVGH


def _case_paths(case, args, attempt: int | None = None):
    out_dir = args.out_dir
    cid = safe_case_id(case.case_id)
    dataset = safe_case_id(case.dataset)
    suffix = "" if attempt is None else f"_try{attempt:02d}"

    raw_work_root = out_dir / "raw_hvgh_work"
    raw_case_dir = raw_work_root / "HVGHlearn" / dataset / cid
    raw_file = raw_case_dir / "001" / "segm000.txt"
    log_file = out_dir / "logs" / f"{dataset}_{cid}{suffix}_stdout.txt"
    pred_file = out_dir / "predictions" / f"{dataset}_{cid}_pred.csv"
    return raw_work_root, raw_case_dir, raw_file, log_file, pred_file


def run_one_case(case, args, runtime, HVGH_cls, attempt: int = 1):
    np = runtime["np"]
    out_dir = args.out_dir
    raw_work_root, raw_case_dir, raw_file, log_file, pred_file = _case_paths(case, args, attempt)
    raw_work_root.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    epoch = args.epoch if args.epoch is not None else case.hvgh_epoch
    iteration = args.iteration if args.iteration is not None else case.hvgh_iteration
    win_size = args.win_size if args.win_size is not None else case.hvgh_win_size
    gamma = args.gamma if args.gamma is not None else case.hvgh_gamma
    eta = args.eta if args.eta is not None else case.hvgh_eta
    initial_class = args.initial_class if args.initial_class is not None else case.hvgh_initial_class

    with log_file.open("w", encoding="utf-8", newline="\n") as lf, redirect_stdout(lf), redirect_stderr(lf):
        print("CASE", case.dataset, case.case_id)
        print("attempt", attempt)
        print("data_shape", getattr(case.data, "shape", None), "labels", len(case.labels))
        print("hvgh", {"epoch": epoch, "iteration": iteration, "win_size": win_size, "gamma": gamma, "eta": eta, "initial_class": initial_class, "input_dim": case.hvgh_input_dim})
        print("raw_file", raw_file)

        if not args.eval_existing_only:
            if raw_case_dir.exists():
                shutil.rmtree(raw_case_dir, ignore_errors=True)

            cwd = os.getcwd()
            os.chdir(raw_work_root)
            try:
                hvgh = HVGH_cls(epoch=epoch, iteration=iteration, gamma=gamma, eta=eta, initial_class=initial_class)
                hvgh.fit(case.data, f"{safe_case_id(case.dataset)}/{safe_case_id(case.case_id)}", win_size=int(win_size), input_dim=case.hvgh_input_dim)
            finally:
                os.chdir(cwd)
        else:
            print("eval_existing_only=1; skipping training")

        if not raw_file.exists():
            raise FileNotFoundError(f"HVGH raw output not found: {raw_file}")

        raw_labels = read_hvgh_raw_labels(raw_file, np)
        point_pred = dilate_label(raw_labels, int(win_size), len(case.labels), np)
        ari, nmi, y_pred = evaluate_case(case, point_pred, runtime)
        states = len(set(map(int, y_pred.tolist())))
        print("ARI", ari, "NMI", nmi, "states", states)

    with pred_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "y_pred"])
        for i, (yt, yp) in enumerate(zip(list(case.labels), list(y_pred))):
            w.writerow([i, int(yt), int(yp)])

    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "ARI": ari,
        "NMI": nmi,
        "states": states,
        "seconds": None,
        "attempts": attempt,
        "status": "ok",
        "error": "",
        "prediction_file": str(pred_file.relative_to(out_dir)),
        "raw_file": str(raw_file.relative_to(out_dir)),
        "log_file": str(log_file.relative_to(out_dir)),
        "hvgh_win_size": win_size,
        "hvgh_epoch": epoch,
        "hvgh_iteration": iteration,
    }


def read_existing_results(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                k = case_key(row.get("dataset", ""), row.get("case_id", ""))
                old = rows.get(k)
                if old is None:
                    rows[k] = dict(row)
                    continue
                old_ok = str(old.get("status", "")).lower() == "ok"
                row_ok = str(row.get("status", "")).lower() == "ok"
                if row_ok or not old_ok:
                    rows[k] = dict(row)
    except Exception:
        return {}
    return rows


def ordered_rows(row_by_case: dict[tuple[str, str], dict], all_cases: list) -> list[dict]:
    order = [case_key(c.dataset, c.case_id) for c in all_cases]
    seen = set(order)
    out = [row_by_case[k] for k in order if k in row_by_case]
    for k in sorted(row_by_case):
        if k not in seen:
            out.append(row_by_case[k])
    return out


def write_case_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def append_attempt_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HISTORY_FIELDNAMES})


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    by = {}
    for r in ok:
        by.setdefault(r["dataset"], []).append(r)
    fieldnames = ["algorithm", "dataset", "cases", "mean_ARI", "mean_NMI", "failed"]
    with (out_dir / "algorithm_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        all_datasets = sorted({r.get("dataset", "") for r in rows if r.get("dataset")})
        for ds in all_datasets:
            vals = by.get(ds, [])
            failed = sum(1 for r in rows if r.get("dataset") == ds and r.get("status") != "ok")
            if vals:
                ari = sum(float(v["ARI"]) for v in vals) / len(vals)
                nmi = sum(float(v["NMI"]) for v in vals) / len(vals)
            else:
                ari = ""
                nmi = ""
            w.writerow({"algorithm": "HVGH", "dataset": ds, "cases": len(vals), "mean_ARI": ari, "mean_NMI": nmi, "failed": failed})


def make_error_row(case, args, attempt: int, seconds: float, exc: BaseException, log_file: Path | None = None) -> dict:
    out_dir = args.out_dir
    _, _, raw_file, default_log, _ = _case_paths(case, args, attempt)
    log = log_file or default_log
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "ARI": "",
        "NMI": "",
        "states": "",
        "seconds": round(seconds, 3),
        "attempts": attempt,
        "status": "error",
        "error": repr(exc),
        "prediction_file": "",
        "raw_file": str(raw_file.relative_to(out_dir)),
        "log_file": str(log.relative_to(out_dir)),
        "hvgh_win_size": case.hvgh_win_size,
        "hvgh_epoch": case.hvgh_epoch,
        "hvgh_iteration": case.hvgh_iteration,
    }


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("HVGH strict baseline with retry accumulation")
    print("Repo root  :", args.repo_root)
    print("Output     :", args.out_dir)
    print("Datasets   :", args.datasets)
    print("Max retries:", args.max_retries)
    print("Rerun done :", int(args.rerun_completed))
    print("============================================================")

    runtime = import_runtime(args.repo_root)
    HVGH_cls = None if args.eval_existing_only else import_hvgh_class()

    all_cases = []
    for ds in args.datasets:
        all_cases.extend(load_cases(ds, args.repo_root, runtime, args.max_cases))
    print("Loaded cases:", len(all_cases))

    out_csv = args.out_dir / "case_results.csv"
    history_csv = args.out_dir / "attempt_history.csv"
    row_by_case = read_existing_results(out_csv)

    if args.rerun_completed:
        row_by_case = {k: v for k, v in row_by_case.items() if str(v.get("status", "")).lower() != "ok"}

    total_cases = len(all_cases)
    for idx, case in enumerate(all_cases, start=1):
        k = case_key(case.dataset, case.case_id)
        existing = row_by_case.get(k)
        print(f"[{idx}/{total_cases}] {case.dataset}/{case.case_id}", flush=True)

        if existing is not None and str(existing.get("status", "")).lower() == "ok" and not args.rerun_completed:
            print(f"  skip existing success: ARI={float(existing.get('ARI', 0)):.4f} NMI={float(existing.get('NMI', 0)):.4f}", flush=True)
            continue

        last_error_row = None
        success = None
        max_retries = max(1, int(args.max_retries))

        for attempt in range(1, max_retries + 1):
            print(f"  attempt {attempt}/{max_retries}", flush=True)
            t0 = time.time()
            try:
                r = run_one_case(case, args, runtime, HVGH_cls, attempt=attempt)
                r["seconds"] = round(time.time() - t0, 3)
                r["attempts"] = attempt
                success = r
                hist = {"time": datetime.now().isoformat(timespec="seconds"), **r}
                append_attempt_history(history_csv, hist)
                print(f"    OK ARI={float(r['ARI']):.4f} NMI={float(r['NMI']):.4f} states={r['states']} seconds={r['seconds']}", flush=True)
                break
            except Exception as exc:
                seconds = time.time() - t0
                last_error_row = make_error_row(case, args, attempt, seconds, exc)
                hist = {"time": datetime.now().isoformat(timespec="seconds"), **last_error_row}
                append_attempt_history(history_csv, hist)
                print("    [ERROR]", repr(exc), flush=True)
                err_log = args.out_dir / "logs" / f"{safe_case_id(case.dataset)}_{safe_case_id(case.case_id)}_try{attempt:02d}_traceback.txt"
                err_log.parent.mkdir(parents=True, exist_ok=True)
                err_log.write_text(traceback.format_exc(), encoding="utf-8")

        if success is not None:
            row_by_case[k] = success
        elif last_error_row is not None:
            row_by_case[k] = last_error_row

        final_rows = ordered_rows(row_by_case, all_cases)
        write_case_results(out_csv, final_rows)
        write_summary(args.out_dir, final_rows)

    final_rows = ordered_rows(row_by_case, all_cases)
    loaded_ok = sum(1 for c in all_cases if case_key(c.dataset, c.case_id) in row_by_case and row_by_case[case_key(c.dataset, c.case_id)].get("status") == "ok")
    loaded_failed = total_cases - loaded_ok
    all_ok = sum(1 for r in final_rows if r.get("status") == "ok")
    all_failed = sum(1 for r in final_rows if r.get("status") != "ok")

    status = {
        "ok": loaded_failed == 0,
        "algorithm": "HVGH",
        "repo_root": str(args.repo_root),
        "out_dir": str(args.out_dir),
        "datasets": args.datasets,
        "max_retries": int(args.max_retries),
        "loaded_cases": total_cases,
        "loaded_cases_ok": loaded_ok,
        "loaded_cases_failed": loaded_failed,
        "all_rows_ok": all_ok,
        "all_rows_failed": all_failed,
        "case_results_csv": str(out_csv),
        "attempt_history_csv": str(history_csv),
    }
    (args.out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print("DONE")
    print("Loaded OK    :", loaded_ok)
    print("Loaded FAILED:", loaded_failed)
    print("CSV          :", out_csv)
    print("History      :", history_csv)
    print("============================================================")
    return 0 if loaded_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
