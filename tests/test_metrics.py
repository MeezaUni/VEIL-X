"""Metric tests against brute-force references, hand calculations and published closed forms."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from veilx.errors import InsufficientSamplesError
from veilx.metrics import (
    ABSTAIN,
    MATCH,
    NON_MATCH,
    augrc,
    aurc,
    brier_score,
    decisions_from_thresholds,
    deployment_operating_point,
    equal_error_rate,
    evaluate_at_threshold,
    expected_calibration_error,
    reliability_diagram,
    reliability_svg,
    risk_coverage_curve,
    roc_auc,
    roc_auc_trapezoid,
    roc_curve,
    summarize_decisions,
    tar_at_far,
    threshold_at_far,
    verification_failures,
)

RNG = np.random.default_rng(0)


def brute_auc(s: np.ndarray, y: np.ndarray) -> float:
    g, i = s[y == 1], s[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a, b in itertools.product(g, i)]))


# ------------------------------------------------------------------ ROC family


@pytest.mark.parametrize("seed", range(5))
def test_auc_three_ways_agree_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    s = np.round(rng.normal(size=60), 1)  # rounding creates ties
    y = rng.integers(0, 2, 60)
    y[:2] = (0, 1)
    assert roc_auc(s, y) == pytest.approx(brute_auc(s, y))
    assert roc_auc_trapezoid(s, y) == pytest.approx(brute_auc(s, y))


def test_auc_extremes_and_degenerate() -> None:
    y = np.array([1, 1, 0, 0])
    assert roc_auc([4, 3, 2, 1], y) == 1.0
    assert roc_auc([1, 2, 3, 4], y) == 0.0
    assert roc_auc([5, 5, 5, 5], y) == 0.5
    assert equal_error_rate([5, 5, 5, 5], y).eer == pytest.approx(0.5)
    with pytest.raises(InsufficientSamplesError):
        roc_auc([1, 2], [1, 1])
    with pytest.raises(ValueError):
        roc_auc([1, np.nan], [1, 0])
    with pytest.raises(ValueError):
        roc_auc([1, 2], [1, 2])


def test_roc_curve_endpoints() -> None:
    far, tar, thr = roc_curve([0.9, 0.8, 0.3, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0, 0])
    assert (far[0], tar[0], thr[0]) == (0.0, 0.0, math.inf)
    assert (far[-1], tar[-1]) == (1.0, 1.0)
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)


def test_eer_hand_example() -> None:
    # genuine 0.9, 0.8, 0.3; impostor 0.7, 0.2, 0.1. At t in (0.3, 0.7]: FAR=1/3, FNMR=1/3.
    res = equal_error_rate([0.9, 0.8, 0.3, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0, 0])
    assert res.eer == pytest.approx(1 / 3)


def test_eer_gaussian_reference() -> None:
    rng = np.random.default_rng(1)
    n = 40000
    s = np.r_[rng.normal(1, 1, n), rng.normal(-1, 1, n)]
    y = np.r_[np.ones(n), np.zeros(n)]
    expected = 0.5 * (1 + math.erf(-1 / math.sqrt(2)))  # Phi(-1)
    assert equal_error_rate(s, y).eer == pytest.approx(expected, abs=0.006)


def test_threshold_at_far_is_tight_and_conservative() -> None:
    rng = np.random.default_rng(2)
    imp = np.round(rng.normal(size=2000), 2)  # ties
    for far in (0.01, 0.05):
        t = threshold_at_far(imp, far)
        realised = float(np.mean(imp >= t))
        assert realised <= far
        lower = imp[imp < t].max()  # next candidate threshold down
        assert float(np.mean(imp >= lower)) > far


def test_operating_point_requires_enough_impostors() -> None:
    rng = np.random.default_rng(3)
    s, y = rng.normal(size=500), np.r_[np.ones(250), np.zeros(250)]
    with pytest.raises(InsufficientSamplesError, match="1000"):
        tar_at_far(s, y, 0.01)  # needs 10/0.01 = 1000 impostors
    op = tar_at_far(s, y, 0.01, min_impostor_factor=2)  # explicit, documented override
    assert op.realised_far <= 0.01


def test_tar_at_far_matches_brute_force() -> None:
    rng = np.random.default_rng(4)
    g, i = rng.normal(2, 1, 3000), rng.normal(0, 1, 3000)
    s, y = np.r_[g, i], np.r_[np.ones(3000), np.zeros(3000)]
    op = tar_at_far(s, y, 0.01, min_impostor_factor=10)
    best = max(float(np.mean(g >= t)) for t in np.unique(s) if np.mean(i >= t) <= 0.01)
    assert op.tar == pytest.approx(best)
    assert op.mode == "benchmark_threshold_on_test"


def test_deployment_mode_uses_calibration_threshold() -> None:
    rng = np.random.default_rng(5)
    cal = np.r_[rng.normal(2, 1, 2000), rng.normal(0, 1, 2000)]
    test = np.r_[rng.normal(2, 1, 2000), rng.normal(0.3, 1, 2000)]  # shifted impostors
    y = np.r_[np.ones(2000), np.zeros(2000)]
    op = deployment_operating_point(cal, y, test, y, 0.01, min_impostor_factor=10)
    assert op.threshold == threshold_at_far(cal[y == 0], 0.01)
    ev = evaluate_at_threshold(test, y, op.threshold)
    assert op.realised_far == ev.far > 0.01  # shift is visible, not hidden
    with pytest.raises(InsufficientSamplesError):
        deployment_operating_point(cal, y, test[:1500], y[:1500], 0.001)


# ------------------------------------------------------------------ selective


def test_decisions_and_summary_hand_example() -> None:
    s = np.array([0.95, 0.9, 0.5, 0.1, 0.05, 0.92, 0.5, 0.2])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    gate = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    d = decisions_from_thresholds(s, 0.3, 0.8, gate=gate)
    assert d.tolist() == [MATCH, MATCH, ABSTAIN, NON_MATCH, NON_MATCH, MATCH, ABSTAIN, ABSTAIN]
    summ = summarize_decisions(d, y)
    assert summ.accepted == 5 and summ.false_matches == 1 and summ.false_non_matches == 1
    assert summ.coverage == pytest.approx(5 / 8)
    assert summ.coverage_genuine == pytest.approx(3 / 4)
    assert summ.coverage_impostor == pytest.approx(2 / 4)
    assert summ.selective_fmr == pytest.approx(1 / 2)
    assert summ.selective_fnmr == pytest.approx(1 / 3)
    assert summ.accepted_error == pytest.approx(2 / 5)
    assert summ.generalized_risk == pytest.approx(2 / 8)
    assert summ.cost_weighted_risk(10) == pytest.approx((10 + 1) / 5)
    assert summ.confusion["impostor"]["abstain"] == 2


def test_empty_denominators_are_none_not_zero() -> None:
    summ = summarize_decisions([ABSTAIN, ABSTAIN, MATCH], [0, 0, 1])
    assert summ.selective_fmr is None and summ.coverage_impostor == 0.0
    all_abstain = summarize_decisions([ABSTAIN, ABSTAIN], [1, 0])
    assert all_abstain.accepted_error is None and all_abstain.cost_weighted_risk(10) is None


def test_invalid_decision_inputs() -> None:
    with pytest.raises(ValueError):
        summarize_decisions([2], [1])
    with pytest.raises(ValueError):
        decisions_from_thresholds([0.1], 0.9, 0.1)
    with pytest.raises(ValueError):
        verification_failures([ABSTAIN, MATCH], [1, 1])


@pytest.mark.parametrize("seed", range(6))
def test_augrc_matches_published_closed_form(seed: int) -> None:
    """Traub et al. (NeurIPS 2024) Eq. 7: AUGRC = (1-AUROC_f)acc(1-acc) + (1-acc)^2/2."""
    rng = np.random.default_rng(seed)
    n = 300
    conf = np.round(rng.normal(size=n), 1)  # ties included
    fail = (rng.random(n) < 1 / (1 + np.exp(2 * conf))).astype(int)
    acc = 1 - fail.mean()
    auroc_f = roc_auc(conf, 1 - fail)  # P(conf_success > conf_failure) + ties/2
    closed = (1 - auroc_f) * acc * (1 - acc) + 0.5 * (1 - acc) ** 2
    assert augrc(conf, fail) == pytest.approx(closed, abs=1e-12)


def test_aurc_equals_mean_selective_risk_without_ties() -> None:
    rng = np.random.default_rng(7)
    conf = rng.normal(size=200)
    fail = rng.integers(0, 2, 200)
    order = np.argsort(-conf)
    sr = np.cumsum(fail[order]) / np.arange(1, 201)
    assert aurc(conf, fail) == pytest.approx(sr.mean())


def test_risk_coverage_curve_properties() -> None:
    conf = np.array([0.9, 0.9, 0.5, 0.1])
    fail = np.array([0, 1, 0, 1])
    rc = risk_coverage_curve(conf, fail)
    assert rc.coverage.tolist() == [0.0, 0.5, 0.75, 1.0]  # tie group accepted together
    assert rc.generalized_risk[-1] == pytest.approx(fail.mean())
    assert rc.selective_risk[1] == pytest.approx(0.5)
    assert np.isnan(rc.selective_risk[0])
    perfect = augrc([3, 2, 1, 0], [0, 0, 1, 1])
    worst = augrc([3, 2, 1, 0], [1, 1, 0, 0])
    assert perfect < worst


# ------------------------------------------------------------------ calibration


def test_ece_and_brier_hand_example() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    # equal-width, 2 bins: bin0 mean p .1 freq .5; bin1 mean p .9 freq 1
    assert expected_calibration_error(p, y, 2, "equal_width") == pytest.approx(
        0.5 * 0.4 + 0.5 * 0.1
    )
    assert brier_score(p, y) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)


def test_ece_near_zero_for_calibrated_probabilities() -> None:
    rng = np.random.default_rng(8)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.01
    assert expected_calibration_error(p, 1 - y) > 0.4


def test_reliability_diagram_and_svg() -> None:
    rng = np.random.default_rng(9)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    diag = reliability_diagram(p, y, 15)
    assert sum(diag.counts) == 300 and len(diag.counts) == 15
    assert diag.ece == pytest.approx(expected_calibration_error(p, y, 15))
    svg = reliability_svg(diag, title="SIMULATED <test>")
    assert svg.startswith("<svg") and svg.count("<circle") == 15 and "&lt;test&gt;" in svg
    with pytest.raises(InsufficientSamplesError):
        reliability_diagram(p[:10], y[:10], 15)
    with pytest.raises(ValueError):
        brier_score([1.2], [1])
