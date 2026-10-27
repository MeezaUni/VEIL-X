"""Helpers of scripts/evaluate_lfw.py: official 10-fold protocol and threshold search."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import numpy as np

from .conftest import ROOT


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_lfw", ROOT / "scripts" / "evaluate_lfw.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_lfw"] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _module()


def test_best_accuracy_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    s = rng.normal(1.2 * y, 1.0)
    t = EV.best_accuracy_threshold(s, y)
    acc = ((s >= t) == (y == 1)).mean()
    brute = max(((s >= c) == (y == 1)).mean() for c in np.concatenate([s, [s.max() + 1]]))
    assert acc == brute


def test_official_protocol_uses_other_folds_only() -> None:
    rng = np.random.default_rng(1)
    fold = np.repeat(np.arange(1, 11), 60)
    y = np.tile(np.r_[np.ones(30, int), np.zeros(30, int)], 10)
    s = rng.normal(2.0 * y, 1.0)
    res = EV.official_accuracy(s, y, fold)
    assert len(res["per_fold"]) == 10 and 0.7 < res["mean"] <= 1.0 and res["std"] is not None
    # the threshold for fold k must not depend on fold k's scores
    s2 = s.copy()
    s2[fold == 3] += 100.0
    res2 = EV.official_accuracy(s2, y, fold)
    assert res2["thresholds"][2] == res["thresholds"][2]


def test_perfect_separation_gives_unit_accuracy() -> None:
    fold = np.repeat(np.arange(1, 11), 20)
    y = np.tile(np.r_[np.ones(10, int), np.zeros(10, int)], 10)
    s = np.where(y == 1, 0.9, 0.1)
    assert EV.official_accuracy(s, y, fold)["mean"] == 1.0
