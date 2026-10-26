from __future__ import annotations

import math

import numpy as np
import pytest

from veilx.stats.jackknife import _norm_ppf, jackknife_interval, leave_one_identity_out, t_ppf


def test_quantile_functions() -> None:
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.001) == pytest.approx(-3.090232, abs=1e-5)
    assert t_ppf(0.975, 9) == pytest.approx(2.262157, abs=2e-3)
    assert t_ppf(0.975, 149) == pytest.approx(1.976013, abs=1e-4)


def test_leave_one_out_removes_both_sides_of_impostor_pairs() -> None:
    num = np.array([1, 0, 1, 0], float)
    den = np.ones(4)
    a = np.array(["x", "x", "y", "z"])
    b = np.array(["y", "z", "z", "x"])
    theta, loo, g = leave_one_identity_out(num, den, a, b)
    assert theta == 0.5 and g == 3
    # x: drop pairs 0,1,3 -> pair 2 (num 1) remains; y: drop 0,2 -> pairs 1,3 (num 0);
    # z: drop 1,2,3 -> pair 0 (num 1) remains
    assert loo.tolist() == [1.0, 0.0, 1.0]


def test_genuine_pairs_counted_once() -> None:
    num, den = np.array([1, 0, 1], float), np.ones(3)
    ids = np.array(["p", "p", "q"])
    theta, loo, _ = leave_one_identity_out(num, den, ids, ids)
    assert theta == pytest.approx(2 / 3) and loo.tolist() == [1.0, 0.5]


def test_interval_matches_iid_binomial_when_no_clustering() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = (rng.random(n) < 0.1).astype(float)
    ids = np.arange(n)
    res = jackknife_interval(x, np.ones(n), ids, ids, scale="linear")
    se_binom = math.sqrt(x.mean() * (1 - x.mean()) / n)
    assert res.se == pytest.approx(se_binom, rel=0.02)
    lg = jackknife_interval(x, np.ones(n), ids, ids, scale="logit")
    assert 0 < lg.ci_low < lg.estimate < lg.ci_high < 1
