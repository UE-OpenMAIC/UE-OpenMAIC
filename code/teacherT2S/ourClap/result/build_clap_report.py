from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "result" else SCRIPT_DIR
OUTPUT_DIR = ROOT / "result"

METRICS = [
    ("ari", "ARI"),
    ("nmi", "NMI"),
    ("covering", "Covering"),
    ("ami", "AMI"),
]

METRIC_STATUS_KEYS = {
    "ari": ["mean_ari", "mean_ari_this_time"],
    "nmi": ["mean_nmi", "mean_nmi_this_time"],
    "covering": ["mean_covering_score", "mean_covering", "mean_covering_this_time"],
    "ami": ["mean_ami_score", "mean_ami", "mean_ami_this_time"],
}

METRIC_CASE_COLUMNS = {
    "ari": ["ari", "branch_ari"],
    "nmi": ["nmi", "branch_nmi"],
    "covering": ["covering_score", "covering"],
    "ami": ["ami_score", "ami"],
}

FORMAL_RUNS = [
    {
        "dataset": "HAS2",
        "baseline": ROOT / "HAS2" / "results_has2_clap_official_default",
        "pid": ROOT / "HAS2" / "results_has2_clap_pid_meta_9x3_cgain",
    },
    {
        "dataset": "MITBIH2",
        "baseline": ROOT / "MITBIH2" / "results_mitbih2_clap_official_default",
        "pid": ROOT / "MITBIH2" / "results_mitbih2_clap_pid_meta_9x3_cgain",
    },
    {
        "dataset": "SKAB2",
        "baseline": ROOT / "SKAB2" / "results_skab2_clap_official_default",
        "pid": ROOT / "SKAB2" / "results_skab2_clap_pid_meta_9x3_cgain",
    },
    {
        "dataset": "TSSB2",
        "baseline": ROOT / "TSSB2" / "results_tssb2_clap_official_default",
        "pid": ROOT / "TSSB2" / "results_tssb2_clap_pid_meta_9x3_cgain",
    },
    {
        "dataset": "UTSA2",
        "baseline": ROOT / "UTSA2" / "results_utsa2_clap_official_default",
        "pid": ROOT / "UTSA2" / "results_utsa2_clap_pid_meta_9x3_cgain",
    },
    {
        "dataset": "mocap",
        "baseline": ROOT / "mocap" / "results_mocap_clap_plain_default",
        "pid": ROOT / "mocap" / "results_mocap_clap_pid_meta_12x3_cgain",
    },
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_numeric(data: dict, keys: Iterable[str]) -> float:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return math.nan


def path_is_in_old(path: Path) -> bool:
    return any(part.lower() == "old" for part in path.parts)


def iter_formal_runs() -> Iterable[dict[str, object]]:
    for run in FORMAL_RUNS:
        baseline = run["baseline"]
        pid = run["pid"]
        if path_is_in_old(baseline) or path_is_in_old(pid):
            continue
        if not (baseline / "run_status.json").exists():
            continue
        if not (pid / "run_status.json").exists():
            continue
        yield run


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_case_metrics(result_dir: Path) -> dict[str, float]:
    csv_path = result_dir / "all_case_results.csv"
    if not csv_path.exists():
        return {key: math.nan for key, _ in METRICS}

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")]

    metrics: dict[str, float] = {}
    for key, _ in METRICS:
        col = find_column(df, METRIC_CASE_COLUMNS[key])
        metrics[key] = float(pd.to_numeric(df[col], errors="coerce").mean()) if col else math.nan
    return metrics


def read_result_metrics(result_dir: Path) -> dict[str, object]:
    status = load_json(result_dir / "run_status.json")
    case_metrics = read_case_metrics(result_dir)

    row: dict[str, object] = {
        "result_dir": str(result_dir),
        "has_status": bool(status),
        "cases": status.get("cases", status.get("n_cases_loaded", math.nan)),
        "ok": status.get("ok", status.get("n_ok_this_time", math.nan)),
    }
    for key, _ in METRICS:
        value = first_numeric(status, METRIC_STATUS_KEYS[key])
        if math.isnan(value):
            value = case_metrics[key]
        row[key] = value
    return row


def normalize_selected(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).eq(1)
    return series.astype(str).str.lower().isin(["1", "true", "yes", "y"])


def read_branch_summary(dataset_name: str, pid_dir: Path) -> dict[str, object]:
    csv_path = pid_dir / "all_branch_results.csv"
    row: dict[str, object] = {
        "dataset": dataset_name,
        "pid_dir": str(pid_dir),
        "case_count": math.nan,
        "selected_branch_rows": math.nan,
        "unselected_branch_rows": math.nan,
    }
    for key, label in METRICS:
        row[f"selected_{key}"] = math.nan
        row[f"unselected_{key}"] = math.nan
        row[f"selected_minus_unselected_{key}"] = math.nan
        row[f"selected_gt_unselected_cases_{key}"] = math.nan
        row[f"selected_lt_unselected_cases_{key}"] = math.nan
        row[f"selected_eq_unselected_cases_{key}"] = math.nan

    if not csv_path.exists():
        return row

    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if df.empty or "selected_for_meta" not in df.columns:
        return row

    df["_selected"] = normalize_selected(df["selected_for_meta"])
    case_col = "case_id" if "case_id" in df.columns else "dataset"
    row["case_count"] = int(df[case_col].nunique())
    row["selected_branch_rows"] = int(df["_selected"].sum())
    row["unselected_branch_rows"] = int((~df["_selected"]).sum())

    for key, _ in METRICS:
        col = find_column(df, METRIC_CASE_COLUMNS[key])
        if col is None:
            continue
        df["_metric"] = pd.to_numeric(df[col], errors="coerce")

        per_case_rows = []
        for _, group in df.groupby(case_col, sort=False):
            selected_mean = group.loc[group["_selected"], "_metric"].mean()
            unselected_mean = group.loc[~group["_selected"], "_metric"].mean()
            if pd.isna(selected_mean) or pd.isna(unselected_mean):
                continue
            per_case_rows.append((float(selected_mean), float(unselected_mean)))

        if not per_case_rows:
            continue

        per_case = pd.DataFrame(per_case_rows, columns=["selected", "unselected"])
        diff = per_case["selected"] - per_case["unselected"]
        row[f"selected_{key}"] = float(per_case["selected"].mean())
        row[f"unselected_{key}"] = float(per_case["unselected"].mean())
        row[f"selected_minus_unselected_{key}"] = float(diff.mean())
        row[f"selected_gt_unselected_cases_{key}"] = int((diff > 1e-12).sum())
        row[f"selected_lt_unselected_cases_{key}"] = int((diff < -1e-12).sum())
        row[f"selected_eq_unselected_cases_{key}"] = int((diff.abs() <= 1e-12).sum())
    return row


def build_method_summary() -> pd.DataFrame:
    rows = []
    for run in iter_formal_runs():
        baseline = read_result_metrics(run["baseline"])
        pid = read_result_metrics(run["pid"])
        row: dict[str, object] = {
            "dataset": run["dataset"],
            "baseline_cases": baseline["cases"],
            "baseline_ok": baseline["ok"],
            "pid_cases": pid["cases"],
            "pid_ok": pid["ok"],
            "baseline_dir": baseline["result_dir"],
            "pid_dir": pid["result_dir"],
        }
        for key, label in METRICS:
            b = baseline[key]
            p = pid[key]
            row[f"baseline_{key}"] = b
            row[f"pid_{key}"] = p
            row[f"pid_minus_baseline_{key}"] = float(p - b) if pd.notna(b) and pd.notna(p) else math.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    macro = {
        "dataset": "MacroMean",
        "baseline_cases": df["baseline_cases"].sum(numeric_only=True),
        "baseline_ok": df["baseline_ok"].sum(numeric_only=True),
        "pid_cases": df["pid_cases"].sum(numeric_only=True),
        "pid_ok": df["pid_ok"].sum(numeric_only=True),
        "baseline_dir": "",
        "pid_dir": "",
    }
    for key, _ in METRICS:
        macro[f"baseline_{key}"] = pd.to_numeric(df[f"baseline_{key}"], errors="coerce").mean()
        macro[f"pid_{key}"] = pd.to_numeric(df[f"pid_{key}"], errors="coerce").mean()
        macro[f"pid_minus_baseline_{key}"] = (
            macro[f"pid_{key}"] - macro[f"baseline_{key}"]
            if pd.notna(macro[f"baseline_{key}"]) and pd.notna(macro[f"pid_{key}"])
            else math.nan
        )
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def build_branch_summary() -> pd.DataFrame:
    rows = [read_branch_summary(run["dataset"], run["pid"]) for run in iter_formal_runs()]
    df = pd.DataFrame(rows)
    macro = {
        "dataset": "MacroMean",
        "pid_dir": "",
        "case_count": pd.to_numeric(df["case_count"], errors="coerce").sum(),
        "selected_branch_rows": pd.to_numeric(df["selected_branch_rows"], errors="coerce").sum(),
        "unselected_branch_rows": pd.to_numeric(df["unselected_branch_rows"], errors="coerce").sum(),
    }
    for key, _ in METRICS:
        for prefix in ["selected", "unselected", "selected_minus_unselected"]:
            macro[f"{prefix}_{key}"] = pd.to_numeric(df[f"{prefix}_{key}"], errors="coerce").mean()
        for prefix in [
            "selected_gt_unselected_cases",
            "selected_lt_unselected_cases",
            "selected_eq_unselected_cases",
        ]:
            macro[f"{prefix}_{key}"] = pd.to_numeric(df[f"{prefix}_{key}"], errors="coerce").sum()
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def build_master_table(method_df: pd.DataFrame, branch_df: pd.DataFrame) -> pd.DataFrame:
    method_cols = [
        "dataset",
        "baseline_cases",
        "baseline_ok",
        "pid_cases",
        "pid_ok",
    ]
    for key, _ in METRICS:
        method_cols.extend([f"baseline_{key}", f"pid_{key}", f"pid_minus_baseline_{key}"])
    branch_cols = ["dataset", "case_count", "selected_branch_rows", "unselected_branch_rows"]
    for key, _ in METRICS:
        branch_cols.extend([f"selected_{key}", f"unselected_{key}", f"selected_minus_unselected_{key}"])
    return method_df[method_cols].merge(branch_df[branch_cols], on="dataset", how="left")


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def write_markdown(master_df: pd.DataFrame, method_df: pd.DataFrame, branch_df: pd.DataFrame) -> Path:
    md_path = OUTPUT_DIR / "clap_report_summary.md"
    included = ", ".join(method_df.loc[method_df["dataset"] != "MacroMean", "dataset"].astype(str).tolist())
    metric_cols = []
    for key, label in METRICS:
        metric_cols.extend(
            [
                (f"baseline_{key}", f"Base {label}"),
                (f"pid_{key}", f"PID {label}"),
                (f"pid_minus_baseline_{key}", f"Delta {label}"),
            ]
        )
    branch_cols = []
    for key, label in METRICS:
        branch_cols.extend(
            [
                (f"selected_{key}", f"Sel {label}"),
                (f"unselected_{key}", f"Unsel {label}"),
                (f"selected_minus_unselected_{key}", f"Sel-Unsel {label}"),
            ]
        )

    def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
        headers = ["Dataset"] + [label for _, label in columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            values = [str(row["dataset"])]
            values.extend(fmt(row[col]) for col, _ in columns)
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    text = "\n\n".join(
        [
            "# CLaP Baseline and PID-Meta Report",
            f"Formal runs included: {included}. Runs under old/ and smoke-only runs are excluded from the formal table.",
            "## Baseline vs PID-Meta",
            markdown_table(method_df, metric_cols),
            "## PID Selected vs Unselected Branch Means",
            "Branch means are computed per case first, then averaged across cases.",
            markdown_table(branch_df, branch_cols),
            "## Output Files",
            "- clap_report_summary_table.csv",
            "- clap_baseline_pid_summary.csv",
            "- clap_selected_unselected_branch_means.csv",
            "- clap_report_summary.png",
        ]
    )
    md_path.write_text(text + "\n", encoding="utf-8")
    return md_path


def set_axis_clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)


def plot_delta_heatmap(ax: plt.Axes, df: pd.DataFrame, columns: list[str], title: str) -> None:
    plot_df = df[df["dataset"] != "MacroMean"].copy()
    data = plot_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    max_abs = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.01
    max_abs = max(max_abs, 0.01)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(masked, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["dataset"].tolist(), fontsize=9)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = "NA" if np.isnan(value) else f"{value:+.3f}"
            color = "#222222" if np.isnan(value) or abs(value) < max_abs * 0.55 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_macro_bars(
    ax: plt.Axes,
    labels: list[str],
    left: list[float],
    right: list[float],
    left_label: str,
    right_label: str,
    title: str,
) -> None:
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, left, width, label=left_label, color="#5078a8")
    ax.bar(x + width / 2, right, width, label=right_label, color="#d88c4a")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, min(1.0, max([v for v in left + right if pd.notna(v)] + [0.1]) + 0.12))
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
    set_axis_clean(ax)
    for offset, values in [(-width / 2, left), (width / 2, right)]:
        for i, value in enumerate(values):
            if pd.isna(value):
                continue
            ax.text(i + offset, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def make_report_figure(method_df: pd.DataFrame, branch_df: pd.DataFrame) -> Path:
    png_path = OUTPUT_DIR / "clap_report_summary.png"
    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], hspace=0.36, wspace=0.28)

    plot_delta_heatmap(
        fig.add_subplot(gs[0, 0]),
        method_df,
        [f"pid_minus_baseline_{key}" for key, _ in METRICS],
        "PID-Meta minus CLaP Baseline",
    )
    plot_delta_heatmap(
        fig.add_subplot(gs[0, 1]),
        branch_df,
        [f"selected_minus_unselected_{key}" for key, _ in METRICS],
        "Selected branches minus unselected branches",
    )

    macro_method = method_df[method_df["dataset"] == "MacroMean"].iloc[0]
    macro_branch = branch_df[branch_df["dataset"] == "MacroMean"].iloc[0]
    metric_labels = [label for _, label in METRICS]
    plot_macro_bars(
        fig.add_subplot(gs[1, 0]),
        metric_labels,
        [macro_method[f"baseline_{key}"] for key, _ in METRICS],
        [macro_method[f"pid_{key}"] for key, _ in METRICS],
        "Baseline",
        "PID-Meta",
        "Macro metric means",
    )
    plot_macro_bars(
        fig.add_subplot(gs[1, 1]),
        metric_labels,
        [macro_branch[f"selected_{key}"] for key, _ in METRICS],
        [macro_branch[f"unselected_{key}"] for key, _ in METRICS],
        "Selected",
        "Unselected",
        "Macro branch means",
    )

    fig.suptitle("CLaP Baseline + PID-Meta Summary", fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "Runs under old/ and smoke-only outputs are excluded. Branch means are computed per case first, then averaged.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    method_df = build_method_summary()
    branch_df = build_branch_summary()
    master_df = build_master_table(method_df, branch_df)

    outputs = {
        "master": OUTPUT_DIR / "clap_report_summary_table.csv",
        "method": OUTPUT_DIR / "clap_baseline_pid_summary.csv",
        "branch": OUTPUT_DIR / "clap_selected_unselected_branch_means.csv",
    }
    master_df.to_csv(outputs["master"], index=False, encoding="utf-8-sig")
    method_df.to_csv(outputs["method"], index=False, encoding="utf-8-sig")
    branch_df.to_csv(outputs["branch"], index=False, encoding="utf-8-sig")
    md_path = write_markdown(master_df, method_df, branch_df)
    png_path = make_report_figure(method_df, branch_df)

    print("Wrote:")
    for path in [outputs["master"], outputs["method"], outputs["branch"], md_path, png_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
