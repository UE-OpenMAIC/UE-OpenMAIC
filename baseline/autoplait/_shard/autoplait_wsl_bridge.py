
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


WSL_DISTRO = "Ubuntu-20.04"

WSL_AUTOPLAIT = (
    "/mnt/d/code/teacherT2S/E2Usd-main/"
    "Baselines/AutoPlait/experiments/src/autoplait"
)


def win_to_wsl_path(p: str) -> str:
    p = str(p).strip().strip('"')

    if p.startswith("/"):
        return p

    if len(p) >= 3 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"

    return p.replace("\\", "/")


def cleanup_old_outputs(output_prefix: Path) -> None:
    parent = output_prefix.parent
    prefix_name = output_prefix.name

    patterns = [
        prefix_name,
        prefix_name + "segment.*",
        prefix_name + "model.*",
    ]

    for pat in patterns:
        for p in parent.glob(pat):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


def find_segment_files(output_prefix: Path) -> list[tuple[int, Path]]:
    parent = output_prefix.parent
    prefix_name = output_prefix.name

    files: list[tuple[int, Path]] = []
    for p in parent.glob(prefix_name + "segment.*"):
        tail = p.name.split("segment.", 1)[-1]
        if tail.isdigit():
            files.append((int(tail), p))

    files.sort(key=lambda x: x[0])
    return files


def parse_segment_file_to_pairs(seg_file: Path) -> list[tuple[int, int]]:
    """
    AutoPlait segment.N files are expected to contain segment boundary pairs.

    Supported formats:
      1) plain pair line:
           123 456

      2) tagged range line:
           [id] xxx [123-456](333)

    Return:
      list of (start, end) pairs.
    """
    pairs: list[tuple[int, int]] = []

    pair_re = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s*$")
    range_re = re.compile(r"\[(\d+)-(\d+)\]\((\d+)\)")

    text = seg_file.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = pair_re.match(line)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            pairs.append((a, b))
            continue

        m2 = range_re.search(line)
        if m2:
            a = int(m2.group(1))
            b = int(m2.group(2))
            pairs.append((a, b))
            continue

    return pairs


def merge_segment_files_to_runner_output(output_prefix: Path) -> int:
    """
    Convert AutoPlait outputs:
        output_prefix + "segment.0"
        output_prefix + "segment.1"
        ...

    into the file expected by autoplait_runner.py:
        output_prefix

    Format:
        start end

        start end

    Blank line separates groups/states.
    """
    segment_files = find_segment_files(output_prefix)
    if not segment_files:
        print("[WSL-BRIDGE] No segment.N files found.", flush=True)
        print("[WSL-BRIDGE] Output prefix:", output_prefix, flush=True)
        return 4

    total_pairs = 0
    with output_prefix.open("w", encoding="utf-8", newline="\n") as fout:
        for state_id, seg_file in segment_files:
            pairs = parse_segment_file_to_pairs(seg_file)

            print(
                f"[WSL-BRIDGE] segment.{state_id}: pairs={len(pairs)}, file={seg_file}",
                flush=True,
            )

            if not pairs:
                continue

            for a, b in pairs:
                fout.write(f"{int(a)} {int(b)}\n")


            fout.write("\n")
            total_pairs += len(pairs)

    if total_pairs <= 0:
        print("[WSL-BRIDGE] AutoPlait finished, but no segment pairs were parsed.", flush=True)
        print("[WSL-BRIDGE] Checked segment files:", flush=True)
        for _, p in segment_files:
            print("  ", p, flush=True)
        return 5

    print("[WSL-BRIDGE] Merged segment pairs for runner:", flush=True)
    print("  output file:", output_prefix, flush=True)
    print("  total pairs:", total_pairs, flush=True)

    return 0


def main() -> int:
    if len(sys.argv) < 4:
        print("[ERROR] Expected arguments: d fin fout", flush=True)
        print("[DEBUG] argv =", sys.argv, flush=True)
        return 2

    d = sys.argv[1]
    fin_win = Path(sys.argv[2].strip().strip('"'))
    fout_win = Path(sys.argv[3].strip().strip('"'))

    fin_wsl = win_to_wsl_path(str(fin_win))
    fout_wsl = win_to_wsl_path(str(fout_win))

    if not fin_win.exists():
        print(f"[ERROR] Input sequence file does not exist: {fin_win}", flush=True)
        return 3

    fout_win.parent.mkdir(parents=True, exist_ok=True)
    cleanup_old_outputs(fout_win)

    manifest_win = fin_win.with_suffix(fin_win.suffix + ".autoplait_manifest.txt")
    manifest_wsl = win_to_wsl_path(str(manifest_win))


    manifest_win.write_text(fin_wsl + "\n", encoding="utf-8")

    cmd = [
        "wsl.exe",
        "-d",
        WSL_DISTRO,
        "--",
        WSL_AUTOPLAIT,
        str(d),
        manifest_wsl,
        fout_wsl,
    ]

    print("[WSL-BRIDGE] Input sequence file:", fin_win, flush=True)
    print("[WSL-BRIDGE] Manifest file      :", manifest_win, flush=True)
    print("[WSL-BRIDGE] Output prefix      :", fout_win, flush=True)
    print("[WSL-BRIDGE] Running:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd)
    code = int(proc.returncode)

    if code != 0:
        print(f"[WSL-BRIDGE] AutoPlait returned non-zero exit code: {code}", flush=True)
        return code

    return merge_segment_files_to_runner_output(fout_win)


if __name__ == "__main__":
    raise SystemExit(main())