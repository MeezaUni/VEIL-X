"""Simulation harness: analytic truth vs Monte Carlo, determinism, labelling."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.selective import ThresholdRule
from veilx.sim.experiments import EXPERIMENTS, interpolate
from veilx.sim.model import (
    ScoreModel,
    population_threshold_at_far,
    rule_truth,
    sample_pairs,
    sim_pair_set,
)


def test_analytic_truth_matches_monte_carlo() -> None:
    model = ScoreModel()
    rule = ThresholdRule(1.0, 3.0, c_min=0.3)
    truth = rule_truth(model, rule)
    sp = sample_pairs(model, 400_000, np.random.default_rng(0))
    d = rule.decide(sp.scores, sp.covis)
    g, i = sp.labels == 1, sp.labels == 0
    acc = d != -1
    assert np.mean(acc[g]) == pytest.approx(truth.acc_g, abs=0.004)
    assert np.mean(acc[i]) == pytest.approx(truth.acc_i, abs=0.004)
    assert np.mean((d == 0)[g]) == pytest.approx(truth.err_g, abs=0.003)
    assert np.mean((d == 1)[i]) == pytest.approx(truth.err_i, abs=0.003)


def test_population_threshold_at_far() -> None:
    model = ScoreModel()
    t = population_threshold_at_far(model, 0.01)
    sp = sample_pairs(model, 300_000, np.random.default_rng(1), genuine_fraction=0.0)
    assert np.mean(sp.scores >= t) == pytest.approx(0.01, abs=0.001)


def test_pair_set_independence_flags() -> None:
    ind = sample_pairs(ScoreModel(), 100, np.random.default_rng(2))
    clu = sample_pairs(
        ScoreModel(), 100, np.random.default_rng(2), design="clustered", n_identities=10
    )
    assert sim_pair_set(ind).summary.identity_disjoint
    assert not sim_pair_set(clu).summary.identity_disjoint
    with pytest.raises(ValueError):
        sample_pairs(ScoreModel(), 10, np.random.default_rng(0), design="clustered")


def test_interpolation_endpoints() -> None:
    a, b = ScoreModel(), ScoreModel(genuine_slope=9.0)
    assert interpolate(a, b, 0.0) == a
    assert interpolate(a, b, 1.0) == b


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_experiments_quick_are_labelled_and_deterministic(name: str) -> None:
    r1 = EXPERIMENTS[name](scale="quick")
    r2 = EXPERIMENTS[name](scale="quick")
    assert r1 == r2
    assert r1["evidence_type"] == "synthetic_simulation"
    assert "SIMULATED" in r1["disclaimer"] and "NOT evidence" in r1["disclaimer"]
