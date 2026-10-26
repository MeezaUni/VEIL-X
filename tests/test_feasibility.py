from __future__ import annotations

import math

import pytest

from veilx.stats.feasibility import (
    IndependentPairCapacity,
    binom_cdf,
    binomial_pvalue_risk_above,
    clopper_pearson_upper,
    min_n_to_certify,
    zero_error_n,
)

pytestmark = pytest.mark.structural


def test_binom_cdf_small_exact() -> None:
    # P[X<=1], n=3, p=0.5 = (1 + 3) / 8
    assert binom_cdf(1, 3, 0.5) == pytest.approx(0.5)
    assert binom_cdf(0, 10, 0.1) == pytest.approx(0.9**10)
    assert binom_cdf(10, 10, 0.3) == 1.0


def test_clopper_pearson_zero_errors_closed_form() -> None:
    # With 0 errors the one-sided bound is 1 - alpha**(1/n).
    for n in (10, 299, 3000):
        assert clopper_pearson_upper(0, n, 0.95) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-6)


def test_rule_of_three_scale() -> None:
    assert zero_error_n(0.01, 0.95) == 299
    assert zero_error_n(0.001, 0.95) == 2995
    assert zero_error_n(0.05, 0.95) == 59


def test_pvalue_is_cdf_under_target() -> None:
    assert binomial_pvalue_risk_above(0, 59, 0.05) == pytest.approx(0.95**59)
    assert binomial_pvalue_risk_above(0, 59, 0.05) < 0.05


def test_min_n_certify_monotone_in_rate() -> None:
    ns = [min_n_to_certify(0.05, r, 0.95) for r in (0.0, 0.02, 0.03, 0.04)]
    assert all(isinstance(n, int) for n in ns)
    assert ns == sorted(ns)  # type: ignore[type-var]
    assert min_n_to_certify(0.05, 0.05, 0.95) is None
    n = ns[1]
    assert n is not None
    assert clopper_pearson_upper(math.floor(0.02 * n), n, 0.95) <= 0.05


def test_independent_pair_capacity() -> None:
    cap = IndependentPairCapacity.from_identities(300, genuine_fraction=0.5)
    # g + 2m <= N with g == m  ->  m = N // 3
    assert cap.max_disjoint_impostor_pairs == 100
    assert cap.max_disjoint_genuine_pairs == 100
    assert cap.max_disjoint_genuine_pairs + 2 * cap.max_disjoint_impostor_pairs <= 300
    with pytest.raises(ValueError):
        IndependentPairCapacity.from_identities(10, genuine_fraction=1.0)
