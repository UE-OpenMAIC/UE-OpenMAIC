
from __future__ import annotations

"""
Prepare an importable HVGH implementation from the original HVGH.ipynb.

The original implementation is notebook-based and uses two %%cython cells.
This script extracts the implementation cells into:
  _generated_hvgh_impl/hvgh_gp.pyx
  _generated_hvgh_impl/hvgh_logsumexp.pyx
  _generated_hvgh_impl/hvgh_model.py

Run automatically by hvgh_runner.py when needed.
"""

import json
import re
import shutil
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ORIGINAL_NB = THIS_DIR / "original" / "HVGH.ipynb"
GEN_DIR = THIS_DIR / "_generated_hvgh_impl"


def _cell_sources(nb_path: Path) -> list[str]:
    if not nb_path.exists():
        raise FileNotFoundError(
            f"Missing original HVGH notebook: {nb_path}\n"
            "Please copy HVGH.ipynb into baseline\\hvgh\\_shard\\original\\HVGH.ipynb."
        )
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _strip_cython_magic(src: str) -> str:
    lines = src.splitlines()
    if lines and lines[0].strip().startswith("%%cython"):
        lines = lines[1:]
    return "\n".join(lines) + "\n"


def _patch_py_code(src: str) -> str:

    src = src.replace("Adam(lr=0.0001)", "Adam(learning_rate=0.0001)")
    src = src.replace("np.float", "float")
    src = src.replace("np.int", "int")
    return src


def build() -> Path:


    if GEN_DIR.exists():
        shutil.rmtree(GEN_DIR, ignore_errors=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    sources = _cell_sources(ORIGINAL_NB)


    gp_cell = None
    logsum_cell = None
    py_cells = []

    for src in sources:
        if src.lstrip().startswith("%%cython") and "cdef class GP" in src:
            gp_cell = _strip_cython_magic(src)
        elif src.lstrip().startswith("%%cython") and "cpdef logsumexp" in src:
            logsum_cell = _strip_cython_magic(src)
        elif any(marker in src for marker in [
            "class Variational_Auto_Encoder", "class GPMD", "class GPSegmentation",
            "def learn(", "class HVGH"
        ]):

            if "def exp_on_" not in src and "%" not in src.splitlines()[0:1]:
                py_cells.append(src)

    if gp_cell is None:
        raise RuntimeError("Cannot find Cython GP cell in HVGH.ipynb")
    if logsum_cell is None:
        raise RuntimeError("Cannot find Cython logsumexp cell in HVGH.ipynb")

    (GEN_DIR / "hvgh_gp.pyx").write_text(gp_cell, encoding="utf-8")
    (GEN_DIR / "hvgh_logsumexp.pyx").write_text(logsum_cell, encoding="utf-8")

    header = r'''# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

try:
    from .hvgh_gp import GP
    from .hvgh_logsumexp import logsumexp
except Exception:
    from hvgh_gp import GP
    from hvgh_logsumexp import logsumexp

'''
    body = "\n\n".join(_patch_py_code(x) for x in py_cells)

    body = re.split(r"\n\s*data_path\s*=", body)[0]
    (GEN_DIR / "hvgh_model.py").write_text(header + body + "\n", encoding="utf-8")
    (GEN_DIR / "__init__.py").write_text("", encoding="utf-8")
    return GEN_DIR


if __name__ == "__main__":
    print(build())
