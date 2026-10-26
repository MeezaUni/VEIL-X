from __future__ import annotations

import numpy as np
import pytest

from veilx.stats.bootstrap import (
    bootstrap_ci,
    encode_identities,
    identity_weight_chunks,
    weighted_rate,
)


def test_identity_weights_structure() -> None:
    a = np.array(["x", "x", "y", "x"])
    b = np.array(["x", "y", "z", "z"])  # genuine, impostor, impostor, impostor
    ca, cb, k = encode_identities(a, b)
    assert k == 3
    w = next(identity_weight_chunks(ca, cb, k, 50, np.random.default_rng(0), chunk=50))
    assert w.shape == (50, 4)
    # Reconstruct the identity multiplicities with the same RNG stream:
    w_id = np.random.default_rng(0).multinomial(k, np.full(k, 1 / k), size=50).astype(float)
    assert (w_id.sum(axis=1) == k).all()
    np.testing.assert_array_equal(w[:, 0], w_id[:, ca[0]])  # genuine: multiplicity of x
    for j in (1, 2, 3):  # impostor: product of the two identities' multiplicities
        np.testing.assert_array_equal(w[:, j], w_id[:, ca[j]] * w_id[:, cb[j]])


def test_estimate_is_unweighted_statistic_and_deterministic() -> None:
    rng = np.random.default_rng(1)
    flags = rng.integers(0, 2, 200).astype(bool)
    ids = rng.integers(0, 40, 200)
    stat = weighted_rate(flags, np.ones(200, bool))
    r1 = bootstrap_ci(
        stat,
        n_pairs=200,
        unit="identity",
        rng=np.random.default_rng(5),
        n_boot=300,
        id_a=ids,
        id_b=ids,
    )
    r2 = bootstrap_ci(
        stat,
        n_pairs=200,
        unit="identity",
        rng=np.random.default_rng(5),
        n_boot=300,
        id_a=ids,
        id_b=ids,
    )
    assert r1 == r2
    assert r1.estimate == pytest.approx(flags.mean())
    assert r1.ci_low <= r1.estimate <= r1.ci_high
    assert r1.n_units == 40 and r1.unit == "identity"


def test_identity_ci_wider_than_pair_ci_under_clustering() -> None:
    rng = np.random.default_rng(2)
    ids = np.repeat(np.arange(30), 20)
    p_id = rng.random(30)
    flags = rng.random(600) < p_id[ids]  # strong identity clustering
    stat = weighted_rate(flags, np.ones(600, bool))
    ri = bootstrap_ci(stat, n_pairs=600, unit="identity", rng=rng, n_boot=500, id_a=ids, id_b=ids)
    rp = bootstrap_ci(stat, n_pairs=600, unit="pair", rng=rng, n_boot=500)
    assert (ri.ci_high - ri.ci_low) > 1.5 * (rp.ci_high - rp.ci_low)


def test_undefined_replicates_are_counted() -> None:
    den = np.array([1, 0, 0, 0], bool)
    stat = weighted_rate(np.array([1, 0, 0, 0], bool), den)
    ids = np.array([0, 1, 2, 3])
    r = bootstrap_ci(
        stat,
        n_pairs=4,
        unit="identity",
        rng=np.random.default_rng(3),
        n_boot=200,
        id_a=ids,
        id_b=ids,
    )
    assert r.n_valid < 200  # identity 0 absent in some replicates -> statistic undefined


def test_argument_validation() -> None:
    stat = weighted_rate(np.ones(3, bool), np.ones(3, bool))
    with pytest.raises(ValueError):
        bootstrap_ci(stat, n_pairs=3, unit="identity", rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bootstrap_ci(stat, n_pairs=3, unit="pair", rng=np.random.default_rng(0), level=1.5)
    with pytest.raises(ValueError):
        encode_identities(np.array([1, 2]), np.array([1]))
