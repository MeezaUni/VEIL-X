"""Helper functions of scripts/evaluate_phase2.py (synthetic inputs, no data access)."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import numpy as np

from veilx.eval.scoring import ScoredPairs

from .conftest import ROOT


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase2", ROOT / "scripts" / "evaluate_phase2.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_phase2"] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _module()


def _sp(scores: list[float], labels: list[int], covis: list[float] | None = None) -> ScoredPairs:
    n = len(scores)
    cv = np.asarray(covis if covis is not None else [1.0] * n, dtype=np.float64)
    s = np.asarray([str(i) for i in range(n)])
    e = np.asarray([""] * n)
    return ScoredPairs(
        pair_id=s,
        score=np.asarray(scores, dtype=np.float64),
        label=np.asarray(labels),
        id_a=s,
        id_b=s,
        vis_a=cv,
        vis_b=np.ones(n),
        fam_a=e,
        fam_b=e,
        lvl_a=e,
        lvl_b=e,
        stratum=np.asarray(["clean_clean"] * n),
        norm_a=np.ones(n),
        norm_b=np.ones(n),
    )


def test_best_accuracy_threshold_is_optimal() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.normal(y * 1.5, 1.0)
    t = EV._best_accuracy_threshold(_sp(s.tolist(), y.tolist()))
    acc = ((s >= t) == (y == 1)).mean()
    brute = max(((s >= c) == (y == 1)).mean() for c in s)
    assert acc == brute


def test_grid_rules_are_ordered_pairs_and_include_extra() -> None:
    rules = EV._grid_rules(np.linspace(-1, 1, 1000), [0.123])
    assert all(r.tau_low <= r.tau_high for r in rules)
    assert any(r.tau_low == 0.123 for r in rules)
    k = len({r.tau_low for r in rules})
    assert len(rules) == k * (k + 1) // 2


def test_strata_and_decisions() -> None:
    st = EV._strata(np.array([1.0, 0.9, 0.5, 0.7]), 0.7)
    assert st.tolist() == ["clean", "vis_high", "vis_low", "vis_high"]
    d = EV._decide([0.2, 0.6], np.array([0.1, 0.4, 0.7]))
    assert d.tolist() == [0, -1, 1]
    assert EV._decide(None, np.array([0.5])).tolist() == [-1]
    sp = _sp([0.1, 0.9, 0.9], [0, 1, 1], covis=[1.0, 0.9, 0.3])
    arm = {"clean": {"rule": [0.5, 0.5]}, "vis_high": {"rule": [0.2, 0.95]}}
    assert EV._decide_stratified(arm, sp, 0.7).tolist() == [0, -1, -1]  # vis_low has no rule
