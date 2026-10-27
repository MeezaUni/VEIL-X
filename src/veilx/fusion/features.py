"""Pair-level feature table and the registered fusion ladder (docs/phase3_protocol.md §5).

A *pair table* is a dict of 1-D arrays (one entry per pair): ``label``, ``id_a``, ``id_b``,
``s`` (AdaFace cosine), ``s_arc``, ``q`` / ``q_arc`` (min feature norm of the pair), and for each
visibility source ``src`` in {ve, heur, gt} the co-visibility features of
:func:`veilx.vis.covis.pair_features` prefixed ``{src}_`` (e.g. ``ve_cv_spatial``), plus
``gt_engine_covis`` = min of the engine's pixel-level visibility ratios (Phase 2 oracle).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.vis.covis import pair_features

F64 = NDArray[np.float64]
SOURCES = ("ve", "heur", "gt")

# model -> (feature names; the visibility source used for vis/cv features or None)
LADDER: dict[str, tuple[tuple[str, ...], str | None]] = {
    "M0": (("s",), None),
    "M0_arc": (("s_arc",), None),
    "M1": (("s", "q", "s*q"), None),
    "M2": (("s", "vmin", "vmax", "s*vmin"), "ve"),
    "M3": (("s", "cv", "asym", "s*cv"), "ve"),
    "M4": (("s", "q", "s*q", "cv", "asym", "s*cv"), "ve"),
    "M5": (
        ("s", "q", "s*q", "cv", "asym", "s*cv", "s_arc", "q_arc", "s_arc*q_arc", "disagree"),
        "ve",
    ),
    "M4_heur": (("s", "q", "s*q", "cv", "asym", "s*cv"), "heur"),
    "M4_oracle": (("s", "q", "s*q", "cv", "asym", "s*cv"), "gt"),
}


def covis_columns(
    maps_a: NDArray[np.floating], maps_b: NDArray[np.floating], src: str, chunk: int = 8192
) -> dict[str, F64]:
    """Co-visibility features of every pair, computed in chunks (maps: (N, G, G) in [0, 1])."""
    parts: dict[str, list[F64]] = {}
    for i in range(0, len(maps_a), chunk):
        f = pair_features(
            np.asarray(maps_a[i : i + chunk], np.float64),
            np.asarray(maps_b[i : i + chunk], np.float64),
        )
        for k, v in f.items():
            parts.setdefault(f"{src}_{k}", []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def design(table: dict[str, NDArray[Any]], model: str, cv_def: str) -> tuple[F64, tuple[str, ...]]:
    """Design matrix of ``model`` with the registered co-visibility definition ``cv_def``."""
    names, src = LADDER[model]
    get: dict[str, Callable[[], F64]] = {
        "s": lambda: table["s"],
        "s_arc": lambda: table["s_arc"],
        "q": lambda: table["q"],
        "q_arc": lambda: table["q_arc"],
        "disagree": lambda: np.abs(table["s"] - table["s_arc"]),
    }
    if src is not None:
        get |= {
            "vmin": lambda: table[f"{src}_vmin"],
            "vmax": lambda: table[f"{src}_vmax"],
            "cv": lambda: table[f"{src}_{cv_def}"],
            "asym": lambda: table[f"{src}_asym"],
        }
    cols = []
    for n in names:
        if "*" in n:
            a, b = n.split("*")
            cols.append(np.asarray(get[a](), np.float64) * np.asarray(get[b](), np.float64))
        else:
            cols.append(np.asarray(get[n](), np.float64))
    return np.column_stack(cols), names
