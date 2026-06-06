                       
r"""
Generate paper-ready PID branch selection table.

Input:
    clap_selected_unselected_branch_means.csv

Output:
    paper_pid_branch_selection_table.csv
    paper_pid_branch_selection_table.md
    paper_pid_branch_selection_table.tex
    paper_pid_branch_selection_table.svg
    paper_pid_branch_selection_table.png
"""

from __future__ import annotations

import csv
import html
import os
import subprocess
import sys
import traceback
from pathlib import Path


METRICS = ["ARI", "NMI", "Covering", "AMI"]


def read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    print("[INFO] CSV columns:")
    print("       " + ", ".join(rows[0].keys()))
    return rows


def to_float(x) -> float:
    if x is None:
        raise ValueError("Found empty value in CSV.")
    return float(str(x).strip())


def get_value(row: dict, candidates: list[str]) -> float:
    lower_map = {str(k).strip().lower(): k for k in row.keys()}

    for name in candidates:
        key = str(name).strip().lower()
        if key in lower_map:
            return to_float(row[lower_map[key]])

    raise ValueError(
        "Cannot find columns:\n"
        + "\n".join(candidates)
        + "\n\nCurrent CSV columns:\n"
        + "\n".join(row.keys())
    )


def is_macro_row(row: dict) -> bool:
    dataset = str(row.get("dataset", row.get("Dataset", ""))).strip().lower()
    return dataset == "macromean"


def is_valid_dataset_row(row: dict) -> bool:
    dataset = str(row.get("dataset", row.get("Dataset", ""))).strip().lower()
    return dataset not in ["", "macromean"]


def row_to_values(row: dict) -> dict:
    return {
        "ARI": {
            "selected": get_value(row, ["selected_ari", "Sel ARI"]),
            "unselected": get_value(row, ["unselected_ari", "Unsel ARI"]),
        },
        "NMI": {
            "selected": get_value(row, ["selected_nmi", "Sel NMI"]),
            "unselected": get_value(row, ["unselected_nmi", "Unsel NMI"]),
        },
        "Covering": {
            "selected": get_value(row, ["selected_covering", "Sel Covering"]),
            "unselected": get_value(row, ["unselected_covering", "Unsel Covering"]),
        },
        "AMI": {
            "selected": get_value(row, ["selected_ami", "Sel AMI"]),
            "unselected": get_value(row, ["unselected_ami", "Unsel AMI"]),
        },
    }


def get_macro_values(rows: list[dict]) -> dict:
    for row in rows:
        if is_macro_row(row):
            print("[INFO] Found MacroMean row. Using it directly.")
            return row_to_values(row)

    print("[INFO] MacroMean row not found. Computing macro mean over dataset rows.")

    dataset_rows = [row for row in rows if is_valid_dataset_row(row)]
    if not dataset_rows:
        raise ValueError("No valid dataset rows found.")

    all_values = [row_to_values(row) for row in dataset_rows]

    macro = {}
    for metric in METRICS:
        macro[metric] = {
            "selected": sum(v[metric]["selected"] for v in all_values) / len(all_values),
            "unselected": sum(v[metric]["unselected"] for v in all_values) / len(all_values),
        }

    return macro


def build_paper_table(values: dict) -> list[list[str]]:
    header = ["Method"] + METRICS

    unselected_row = ["Unselected branches"]
    selected_row = ["PID-selected branches"]
    relative_row = ["Relative gain"]

    for metric in METRICS:
        selected = values[metric]["selected"]
        unselected = values[metric]["unselected"]

        relative_gain = 0.0 if abs(unselected) < 1e-12 else (selected - unselected) / unselected * 100.0

        unselected_row.append(f"{unselected:.4f}")
        selected_row.append(f"{selected:.4f}")
        relative_row.append(f"+{relative_gain:.2f}%")

    return [header, unselected_row, selected_row, relative_row]


def save_csv(table: list[list[str]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(table)


def save_markdown(table: list[list[str]], out_path: Path) -> None:
    header = table[0]
    rows = table[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_latex(table: list[list[str]], out_path: Path) -> None:
    header = table[0]
    rows = table[1:]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Macro-averaged comparison between PID-selected and unselected CLaP branches.}",
        r"\label{tab:pid_branch_selection_clap}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]

    for row in rows:
        method = row[0]
        vals = row[1:]

        if method in ["PID-selected branches", "Relative gain"]:
            method = r"\textbf{" + method + "}"
            vals = [r"\textbf{" + v + "}" for v in vals]

        lines.append(method + " & " + " & ".join(vals) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_svg(table: list[list[str]], out_path: Path) -> None:
    header = table[0]
    rows = table[1:]

    width = 1300
    height = 310
    left = 40
    top = 35
    col_widths = [430, 190, 190, 230, 190]
    row_h = 58
    font_family = "Times New Roman, Arial, sans-serif"

    def x_at(col: int) -> int:
        return left + sum(col_widths[:col])

    def y_at(row: int) -> int:
        return top + row * row_h

    total_w = sum(col_widths)
    total_h = row_h * (len(rows) + 1)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        f'<rect x="{left}" y="{top}" width="{total_w}" height="{total_h}" '
        f'fill="none" stroke="black" stroke-width="2"/>'
    )

    for r in range(1, len(rows) + 1):
        y = y_at(r)
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{left + total_w}" y2="{y}" '
            f'stroke="black" stroke-width="1"/>'
        )

    cur_x = left
    for w in col_widths[:-1]:
        cur_x += w
        svg.append(
            f'<line x1="{cur_x}" y1="{top}" x2="{cur_x}" '
            f'y2="{top + total_h}" stroke="black" stroke-width="1"/>'
        )

    for c, text in enumerate(header):
        x = x_at(c) + col_widths[c] / 2
        y = y_at(0) + 37
        svg.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-family="{font_family}" font-size="28" font-weight="700">'
            f'{html.escape(text)}</text>'
        )

    for r, row in enumerate(rows, start=1):
        is_bold = row[0] in ["PID-selected branches", "Relative gain"]
        weight = "700" if is_bold else "400"

        for c, text in enumerate(row):
            y = y_at(r) + 37

            if c == 0:
                x = x_at(c) + 24
                anchor = "start"
            else:
                x = x_at(c) + col_widths[c] / 2
                anchor = "middle"

            svg.append(
                f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                f'font-family="{font_family}" font-size="27" font-weight="{weight}">'
                f'{html.escape(text)}</text>'
            )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def ps_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def save_png_with_powershell(table: list[list[str]], out_path: Path) -> bool:
    """
    Hard-code every cell draw call.
    This avoids PowerShell array flattening and fixes the blank Method column.
    """
    if os.name != "nt":
        return False

    if out_path.exists():
        out_path.unlink()

    header = table[0]
    rows = table[1:]

    ps_path = out_path.with_suffix(".ps1")

    width = 1980
    height = 430
    left = 45
    top = 35
    row_h = 78
    col_w = [680, 280, 280, 360, 280]

    x_positions = [
        left,
        left + col_w[0],
        left + col_w[0] + col_w[1],
        left + col_w[0] + col_w[1] + col_w[2],
        left + col_w[0] + col_w[1] + col_w[2] + col_w[3],
    ]

    def draw_call(text: str, x: int, y: int, w: int, h: int, bold: bool, align: str) -> str:
        return (
            f"DrawCell {ps_str(text)} {x} {y} {w} {h} "
            f"${str(bold).lower()} {ps_str(align)}"
        )

    draw_lines = []

            
    for c, text in enumerate(header):
        draw_lines.append(draw_call(text, x_positions[c], top, col_w[c], row_h, True, "center"))

          
    for r, row in enumerate(rows):
        y = top + (r + 1) * row_h
        bold = row[0] in ["PID-selected branches", "Relative gain"]

                                     
        draw_lines.append(draw_call(row[0], x_positions[0] + 25, y, col_w[0] - 25, row_h, bold, "left"))

                        
        for c in range(1, 5):
            draw_lines.append(draw_call(row[c], x_positions[c], y, col_w[c], row_h, bold, "center"))

    draw_lines_text = "\n".join(draw_lines)

    script = f"""
Add-Type -AssemblyName System.Drawing

$out = {ps_str(str(out_path))}

$width = {width}
$height = {height}
$left = {left}
$top = {top}
$rowH = {row_h}
$colW = @({",".join(str(x) for x in col_w)})

$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$g.FillRectangle([System.Drawing.Brushes]::White, 0, 0, $width, $height)

$blackBrush = [System.Drawing.Brushes]::Black
$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 2
$thinPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 1

$fontHeader = New-Object System.Drawing.Font("Times New Roman", 24, [System.Drawing.FontStyle]::Bold)
$fontBody = New-Object System.Drawing.Font("Times New Roman", 23, [System.Drawing.FontStyle]::Regular)
$fontBold = New-Object System.Drawing.Font("Times New Roman", 23, [System.Drawing.FontStyle]::Bold)

function DrawCell([string]$text, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$bold, [string]$align) {{
    if ($bold) {{
        $font = $fontBold
    }} else {{
        $font = $fontBody
    }}

    $fmt = New-Object System.Drawing.StringFormat
    if ($align -eq "left") {{
        $fmt.Alignment = [System.Drawing.StringAlignment]::Near
    }} else {{
        $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    }}
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center

    $rect = New-Object System.Drawing.RectangleF($x, $y, $w, $h)
    $g.DrawString($text, $font, $blackBrush, $rect, $fmt)
}}

$totalW = 0
foreach ($w in $colW) {{ $totalW += $w }}
$totalH = $rowH * 4

$g.DrawRectangle($pen, $left, $top, $totalW, $totalH)

for ($r = 1; $r -le 3; $r++) {{
    $y = $top + $r * $rowH
    $g.DrawLine($thinPen, $left, $y, $left + $totalW, $y)
}}

$x = $left
for ($c = 0; $c -lt $colW.Count - 1; $c++) {{
    $x += $colW[$c]
    $g.DrawLine($thinPen, $x, $top, $x, $top + $totalH)
}}

{draw_lines_text}

$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
"""

    ps_path.write_text(script, encoding="utf-8-sig")

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("[WARN] PowerShell PNG generation failed.")
        print(result.stdout)
        print(result.stderr)
        return False

    return out_path.exists()


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    input_csv = work_dir / "clap_selected_unselected_branch_means.csv"

    if not input_csv.exists():
        raise FileNotFoundError(
            "Input CSV not found:\n"
            f"{input_csv}\n\n"
            "Please put clap_selected_unselected_branch_means.csv in the same folder as this script."
        )

    print("[INFO] Reading input CSV:")
    print(f"       {input_csv}")

    rows = read_csv_rows(input_csv)
    values = get_macro_values(rows)
    table = build_paper_table(values)

    out_csv = work_dir / "paper_pid_branch_selection_table.csv"
    out_md = work_dir / "paper_pid_branch_selection_table.md"
    out_tex = work_dir / "paper_pid_branch_selection_table.tex"
    out_svg = work_dir / "paper_pid_branch_selection_table.svg"
    out_png = work_dir / "paper_pid_branch_selection_table.png"

    save_csv(table, out_csv)
    save_markdown(table, out_md)
    save_latex(table, out_tex)
    save_svg(table, out_svg)

    png_ok = save_png_with_powershell(table, out_png)

    print()
    print("[OK] Generated paper-ready files:")
    print(f"  CSV : {out_csv}")
    print(f"  MD  : {out_md}")
    print(f"  TEX : {out_tex}")
    print(f"  SVG : {out_svg}")

    if png_ok:
        print(f"  PNG : {out_png}")
    else:
        print("  PNG : failed, but SVG was generated successfully.")

    print()
    print("[Preview]")
    for row in table:
        print("\t".join(row))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print()
        print("[FATAL] Script failed.")
        traceback.print_exc()
        sys.exit(1)
