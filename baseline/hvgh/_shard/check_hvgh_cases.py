
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def is_number(x: object) -> bool:
    try:
        v = float(str(x).strip())
        return math.isfinite(v)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-csv", type=Path, required=True)
    ap.add_argument("--expected", type=int, required=True)
    args = ap.parse_args()

    path = args.case_csv
    if not path.exists():
        print(f"[CHECK] case_results.csv not found yet: {path}")
        return 10

    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    ok = {}
    bad = {}

    for r in rows:
        cid = str(r.get("case_id") or r.get("case") or "").strip()
        if not cid:
            continue

        err = str(r.get("error") or "").strip()
        ari = r.get("ARI", r.get("ari", ""))
        nmi = r.get("NMI", r.get("nmi", ""))

        if not err and is_number(ari) and is_number(nmi):
            ok[cid] = True
        else:
            bad[cid] = True

    print(f"[CHECK] OK unique cases: {len(ok)}/{args.expected}")
    if ok:
        print("[CHECK] OK cases:", ", ".join(sorted(ok)))
    if bad:
        print("[CHECK] Failed/error rows seen:", ", ".join(sorted(bad)))

    return 0 if len(ok) >= args.expected else 10


if __name__ == "__main__":
    raise SystemExit(main())
