from __future__ import annotations

import sys
from pathlib import Path

SHARD_DIR = Path(__file__).resolve().parents[1] / "_shard"
sys.path.insert(0, str(SHARD_DIR))

from e2usd_launcher import main


if __name__ == "__main__":
    raise SystemExit(main("actrectut", "actrectut_config.txt"))
