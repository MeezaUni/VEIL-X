"""Learn-then-Test calibrator: validity preconditions, mechanics and simulated error control."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.config.schema import SplitName
from veilx.errors import ProtocolViolationError
from veilx.metrics import ABSTAIN
from veilx.pairs import PairSpec, build_pairs
from veilx.seeding import make_rng
from veilx.selective import (
    EmpiricalCalibrator,
    LTTCalibrator,
    ThresholdRule,
    binomial_pvalue,
    fit_stratified,
    symmetric_rules,
)
from veilx.selective.ltt import order_candidates_on_validation
from veilx.sim.model import ScoreModel, rule_truth, sample_pairs, sim_pair_set
from veilx.stats.feasibility import binom_cdf

from .helpers import toy_images

RULES = symmetric_rules(2.0, [round(0.1 * k, 1) for k in range(26)])


def easy_cal(n: int = 2000, seed: int = 0, **kw):  # type: ignore[no-untyped-def]
    sp = sample_pairs(ScoreModel(), n, np.random.default_rng(seed), **kw)
    return sp, sim_pair_set(sp)


def test_binomial_pvalue_definition() -> None:
    assert binomial_pvalue(0, 0, 0.05) == 1.0  # nothing accepted -> cannot reject
    assert binomial_pvalue(0, 59, 0.05) == pytest.approx(0.95**59)
    assert binomial_pvalue(3, 100, 0.05) == pytest.approx(binom_cdf(3, 100, 0.05))
    with pytest.raises(ValueError):
        binomial_pvalue(0, 10, 0.0)


def test_refuses_dependent_calibration_pairs() -> None:
    sp = sample_pairs(
        ScoreModel(), 600, np.random.default_rng(1), design="clustered", n_identities=30
    )
    cal = LTTCalibrator(RULES, target_fm=0.05, delta=0.05)
    with pytest.raises(ProtocolViolationError, match="dependent"):
        cal.fit(sp.scores, sp.labels, pair_set=sim_pair_set(sp))
    with pytest.raises(ProtocolViolationError, match="independence"):
        cal.fit(sp.scores, sp.labels, pair_set=None)
    res = LTTCalibrator(RULES, target_fm=0.05, delta=0.05, require_valid=False).fit(
        sp.scores, sp.labels, pair_set=sim_pair_set(sp)
    )
    assert res.guarantee_valid is False
    assert "NO statistical guarantee" in res.guarantee_scope


def test_refuses_dependent_real_pair_set() -> None:
    images = toy_images(8, 6, SplitName.CAL)
    ps = build_pairs(images, PairSpec("clustered", 40, 0.5, SplitName.CAL, 0))
    scores = np.linspace(-1, 1, 40)
    labels = np.array([1 if p.label == "genuine" else 0 for p in ps.pairs])
    with pytest.raises(ProtocolViolationError):
        LTTCalibrator(RULES, target_fm=0.05, delta=0.05).fit(scores, labels, pair_set=ps)


def test_mixture_estimand_requires_bernoulli_classes() -> None:
    sp, ps = easy_cal(class_assignment="fixed")
    cal = LTTCalibrator(RULES, target_fm=0.05, delta=0.05, estimand="mixture_conditional")
    with pytest.raises(ProtocolViolationError, match="Bernoulli"):
        cal.fit(sp.scores, sp.labels, pair_set=ps)
    sp2, ps2 = easy_cal(class_assignment="bernoulli")
    assert cal.fit(sp2.scores, sp2.labels, pair_set=ps2).guarantee_valid


def test_feasible_fit_and_scope_statement() -> None:
    sp, ps = easy_cal()
    val, _ = easy_cal(seed=99)
    order = order_candidates_on_validation(
        RULES, val.scores, val.labels, n_cal_expected=2000, target=0.05, delta=0.05
    )
    res = LTTCalibrator(order, target_fm=0.05, delta=0.05).fit(sp.scores, sp.labels, pair_set=ps)
    assert res.feasible and res.rule is not None and res.guarantee_valid
    assert "NOT claimed" in res.guarantee_scope and "held-out" in res.guarantee_scope
    chosen = next(c for c in res.candidates if c.rule == res.rule)
    assert chosen.p_value <= 0.05
    d = res.decide(sp.scores)
    assert set(np.unique(d)) <= {-1, 0, 1}


def test_infeasible_target_abstains_everywhere() -> None:
    sp, ps = easy_cal(n=300)
    res = LTTCalibrator(RULES, target_fm=0.001, delta=0.05).fit(sp.scores, sp.labels, pair_set=ps)
    assert not res.feasible and res.rule is None
    assert (res.decide(sp.scores) == ABSTAIN).all()
    assert res.notes


def test_fixed_sequence_stops_at_first_failure() -> None:
    sp, ps = easy_cal()
    impossible = ThresholdRule(-50.0, 50.0)  # accepts nothing -> p = 1
    res = LTTCalibrator([impossible, *RULES], target_fm=0.05, delta=0.05).fit(
        sp.scores, sp.labels, pair_set=ps
    )
    assert res.rejected == () and res.rule is None
    res_b = LTTCalibrator([impossible, *RULES], target_fm=0.05, delta=0.05, fwer="bonferroni").fit(
        sp.scores, sp.labels, pair_set=ps
    )
    assert all(res_b.candidates[j].p_value <= 0.05 / (len(RULES) + 1) for j in res_b.rejected)


def test_class_conditional_pvalue_is_max_of_two() -> None:
    sp, ps = easy_cal()
    res = LTTCalibrator(RULES[:3], target_fm=0.05, target_fnm=0.08, delta=0.05).fit(
        sp.scores, sp.labels, pair_set=ps
    )
    for c in res.candidates:
        expect = max(
            binomial_pvalue(c.false_matches, c.accepted_impostor, 0.05),
            binomial_pvalue(c.false_non_matches, c.accepted_genuine, 0.08),
        )
        assert c.p_value == pytest.approx(expect)


def test_validation_ordering_prunes_hopeless_rules() -> None:
    val, _ = easy_cal(seed=5)
    order = order_candidates_on_validation(
        RULES, val.scores, val.labels, n_cal_expected=100, target=0.05, delta=0.05
    )
    # with 100 calibration pairs no class can reach 59 accepted pairs
    assert order == []
    order = order_candidates_on_validation(
        RULES, val.scores, val.labels, n_cal_expected=5000, target=0.05, delta=0.05
    )
    assert order and len(order) <= len(RULES)


def test_gate_rule_requires_gate_values() -> None:
    rule = ThresholdRule(1.0, 3.0, c_min=0.5)
    with pytest.raises(ValueError):
        rule.decide([2.0])
    assert rule.decide([4.0, 4.0], [0.4, 0.6]).tolist() == [ABSTAIN, 1]
    with pytest.raises(ValueError):
        ThresholdRule(3.0, 1.0)
    with pytest.raises(ValueError):
        LTTCalibrator([], target_fm=0.05, delta=0.05)
    with pytest.raises(ValueError):
        LTTCalibrator(RULES, target_fm=1.5, delta=0.05)


def test_empirical_calibrator_never_claims_guarantee() -> None:
    sp, _ = easy_cal()
    res = EmpiricalCalibrator(RULES, target_fm=0.05).fit(sp.scores, sp.labels)
    assert res.guarantee_valid is False and res.feasible
    assert "NO statistical guarantee" in res.guarantee_scope


def test_stratified_unseen_stratum_abstains() -> None:
    sp, ps = easy_cal(n=3000)
    strata = np.where(sp.covis >= 0.6, "hi", "lo")
    res = fit_stratified(
        LTTCalibrator, RULES, sp.scores, sp.labels, list(strata), ps, target_fm=0.05, delta=0.05
    )
    assert set(res.per_stratum) == {"hi", "lo"}
    assert all(r.delta == pytest.approx(0.025) for r in res.per_stratum.values())
    d = res.decide([3.0, 3.0], ["hi", "never_seen"])
    assert d[1] == ABSTAIN
    dep = sample_pairs(
        ScoreModel(), 600, np.random.default_rng(2), design="clustered", n_identities=20
    )
    with pytest.raises(ProtocolViolationError):
        fit_stratified(
            LTTCalibrator,
            RULES,
            dep.scores,
            dep.labels,
            ["a"] * 600,
            sim_pair_set(dep),
            target_fm=0.05,
            delta=0.05,
        )


def test_simulated_violation_rate_within_delta() -> None:
    """SIMULATED check: with independent calibration data, P(true risk of chosen rule > target)
    should not exceed delta (allowing Monte Carlo error)."""
    model, target, delta, reps, n = ScoreModel(), 0.05, 0.2, 150, 800
    truth = {r: rule_truth(model, r).class_conditional_risk for r in RULES}
    violations = 0
    for rep in range(reps):
        rng = make_rng(11, "ltt-test", rep)
        val = sample_pairs(model, n, rng)
        order = order_candidates_on_validation(
            RULES, val.scores, val.labels, n_cal_expected=n, target=target, delta=delta
        )
        sp = sample_pairs(model, n, rng)
        res = (
            LTTCalibrator(order, target_fm=target, delta=delta).fit(
                sp.scores, sp.labels, pair_set=sim_pair_set(sp)
            )
            if order
            else None
        )
        if res is not None and res.rule is not None:
            violations += truth[res.rule] > target
    rate = violations / reps
    assert rate <= delta + 3 * np.sqrt(delta * (1 - delta) / reps)
