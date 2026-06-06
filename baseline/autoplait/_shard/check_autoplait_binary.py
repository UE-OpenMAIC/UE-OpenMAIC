from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--bin", type=Path, default=Path(r"D:\code\teacherT2S\E2Usd-main\Baselines\AutoPlait\experiments\src\autoplait"))
args = p.parse_args()

print("AutoPlait binary:", args.bin)
print("Exists:", args.bin.exists())
if args.bin.exists():
    try:
        proc = subprocess.run([str(args.bin)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=10)
        print("Exit code:", proc.returncode)
        print(proc.stdout[:2000])
    except Exception as exc:
        print("Run failed:", repr(exc))
        print("If this is WinError 193, the file is probably a Linux binary. Compile autoplait.exe or run through WSL/MSYS2.")
