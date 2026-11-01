"""Phase 4: temperature, lower-bound shift, Mahalanobis novelty, hybrid fallback strata."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.selective.hybrid import hybrid_score, hybrid_strata, novelty_forced_strata, tertile
from veilx.vis.conservative import (
    MahalanobisNovelty,
    conservative_map,
    fit_temperature,
    global_visible,
    lower_bound_shift,
    pair_max,
    tta_disagreement,
)


def test_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, (400, 28, 28))
    y = rng.random(true_logit.shape) < 1 / (1 + np.exp(-true_logit))
    t = fit_temperature(3.0 * true_logit, y)  # logits 3x overconfident
    assert 2.5 <= t <= 3.5


def test_lower_bound_shift_caps_overestimation_rate() -> None:
    rng = np.random.default_rng(1)
    true = rng.uniform(0.3, 0.9, 2000)
    pred = true + rng.normal(0.08, 0.05, 2000)  # systematically optimistic
    c = lower_bound_shift(pred, true, margin=0.10, max_rate=0.05)
    assert c > 0 and float(((pred - c - true) > 0.10).mean()) <= 0.05
    assert lower_bound_shift(true, true) == 0.0  # unbiased predictor: no shift


def test_conservative_map_never_exceeds_calibrated_probability() -> None:
    lg = np.random.default_rng(2).normal(0, 3, (5, 28, 28))
    base = conservative_map(lg, 1.5, 0.0)
    shifted = conservative_map(lg, 1.5, 0.1)
    assert np.all(shifted <= base) and shifted.min() >= 0.0
    assert np.all(global_visible(shifted) <= global_visible(base))


def test_mahalanobis_flags_shifted_distribution_at_registered_id_fpr() -> None:
    rng = np.random.default_rng(3)
    id_tr = rng.normal(0, 1, (5000, 16))
    nov = MahalanobisNovelty.fit(id_tr)
    id_val_a, id_val_b = rng.normal(0, 1, (4000, 16)), rng.normal(0, 1, (4000, 16))
    thr = nov.set_threshold(pair_max(nov.score(id_val_a), nov.score(id_val_b)), fpr=0.05)
    assert thr > 0
    fresh = pair_max(
        nov.score(rng.normal(0, 1, (4000, 16))), nov.score(rng.normal(0, 1, (4000, 16)))
    )
    assert abs(float(nov.is_novel(fresh).mean()) - 0.05) < 0.02  # ID pair FPR ~ 5 %
    ood = pair_max(
        nov.score(rng.normal(1.5, 1, (2000, 16))), nov.score(rng.normal(0, 1, (2000, 16)))
    )
    assert float(nov.is_novel(ood).mean()) > 0.9
    back = MahalanobisNovelty.from_dict(nov.to_dict())
    assert back.threshold == nov.threshold and np.allclose(
        back.score(id_tr[:5]), nov.score(id_tr[:5])
    )


def test_novelty_requires_threshold_and_enough_samples() -> None:
    nov = MahalanobisNovelty.fit(np.random.default_rng(4).normal(0, 1, (100, 4)))
    with pytest.raises(RuntimeError):
        nov.is_novel(np.zeros(3))
    with pytest.raises(ValueError):
        MahalanobisNovelty.fit(np.zeros((3, 8)))


def test_tta_disagreement_zero_for_symmetric_prediction() -> None:
    lg = np.random.default_rng(5).normal(0, 1, (3, 28, 28))
    assert np.allclose(tta_disagreement(lg, lg), 0.0)
    assert np.all(tta_disagreement(lg, -lg) > 0)


def test_hybrid_routes_unreliable_pairs_to_quality_strata() -> None:
    reliable = np.array([True, True, False, False])
    cv = np.array([0.2, 0.9, 0.2, 0.9])
    q = np.array([25.0, 15.0, 25.0, 15.0])
    st = hybrid_strata(reliable, cv, q, (0.4, 0.7), (18.0, 22.0))
    assert st.tolist() == ["v_low", "v_high", "q_high", "q_low"]
    sc = hybrid_score(reliable, np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.9, 0.8, 0.7, 0.6]))
    assert sc.tolist() == [0.1, 0.2, 0.7, 0.6]


def test_novelty_forced_strata_and_tertiles() -> None:
    assert tertile(np.array([0.1, 0.5, 0.9]), (0.4, 0.7), "v").tolist() == [
        "v_low",
        "v_mid",
        "v_high",
    ]
    st = novelty_forced_strata(np.array([True, False]), np.array([0.95, 0.95]), (0.4, 0.7))
    assert st.tolist() == ["v_low", "v_high"]
