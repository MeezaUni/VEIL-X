"""Paired identity-jackknife differences and Holm adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.stats.jackknife import jackknife_interval
from veilx.stats.paired import holm, jackknife_ratio_difference, normal_p_value


def _data(seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(300), 4)
    ea = (rng.random(ids.size) < 0.10).astype(float)
    eb = (rng.random(ids.size) < 0.05).astype(float)
    return ids, ea, eb


def test_same_denominator_matches_jackknife_of_per_pair_difference() -> None:
    ids, ea, eb = _data()
    one = np.ones(ids.size)
    d = jackknife_ratio_difference(ea, one, eb, one, ids, ids)
    j = jackknife_interval(ea - eb, one, ids, ids, scale="linear")
    assert d.estimate == pytest.approx(j.estimate)
    assert d.se == pytest.approx(j.se)
    assert d.ci_low < d.estimate < d.ci_high


def test_identical_systems_have_zero_difference() -> None:
    ids, ea, _ = _data(1)
    one = np.ones(ids.size)
    d = jackknife_ratio_difference(ea, one, ea, one, ids, ids)
    assert d.estimate == 0 and d.se == 0 and d.p_value == 1.0


def test_different_denominators() -> None:
    ids, ea, eb = _data(2)
    acc_a = np.ones(ids.size)
    acc_b = (np.arange(ids.size) % 2 == 0).astype(float)
    d = jackknife_ratio_difference(ea * acc_a, acc_a, eb * acc_b, acc_b, ids, ids)
    assert d.estimate == pytest.approx(ea.mean() - (eb * acc_b).sum() / acc_b.sum())
    assert np.isfinite(d.se) and d.se > 0


def test_normal_p_value_and_holm() -> None:
    assert normal_p_value(1.96, 1.0) == pytest.approx(0.05, abs=1e-3)
    assert normal_p_value(0.0, 1.0) == pytest.approx(1.0)
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.5], strict=True))
