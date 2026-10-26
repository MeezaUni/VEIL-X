"""Synthetic score model for statistical simulations. **Not a face-recognition model.**

Generative model (all quantities synthetic):

* co-visibility ``c ~ Beta(covis_a, covis_b)`` per pair (a stand-in for shared visible area);
* genuine score  ``s = mu_g - genuine_slope*(1-c) + u_i + sd(c)*e``,  ``u_i ~ N(0, sigma_u^2)``
  per identity (identity random effect -> correlated pairs when identities are reused);
* impostor score ``s = mu_i + impostor_slope*(1-c) + v_a + v_b + sd(c)*e``, ``v ~ N(0, sigma_v^2)``;
* ``sd(c) = sigma_noise * (1 + noise_slope*(1-c))``, ``e ~ N(0, 1)``.

Because identity effects are independent of ``c``, the population (identity-marginal) score
distribution given ``c`` is normal, so population rates of any threshold rule are computed
**exactly** (up to quadrature over ``c``). Simulations therefore compare estimates against known
truth rather than against another simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from veilx.pairs.builder import PairSet, PairSetSummary
from veilx.selective.ltt import ThresholdRule

FloatArr = NDArray[np.float64]
_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(x: FloatArr) -> FloatArr:
    return np.asarray(
        0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))), dtype=np.float64
    )


@dataclass(frozen=True)
class ScoreModel:
    mu_genuine: float = 5.0
    mu_impostor: float = 0.0
    sigma_noise: float = 1.0
    sigma_identity_genuine: float = 0.5
    sigma_identity_impostor: float = 0.3
    covis_a: float = 4.0
    covis_b: float = 1.5
    genuine_slope: float = 3.0
    impostor_slope: float = 0.0
    noise_slope: float = 0.5

    def with_(self, **kw: float) -> ScoreModel:
        return replace(self, **kw)


@dataclass(frozen=True)
class SimPairs:
    scores: FloatArr
    labels: NDArray[np.int64]
    covis: FloatArr
    id_a: NDArray[np.int64]
    id_b: NDArray[np.int64]
    design: Literal["independent", "clustered"]
    class_assignment: Literal["fixed", "bernoulli"]

    def subset(self, mask: NDArray[np.bool_]) -> SimPairs:
        return SimPairs(
            self.scores[mask],
            self.labels[mask],
            self.covis[mask],
            self.id_a[mask],
            self.id_b[mask],
            self.design,
            self.class_assignment,
        )


def sample_pairs(
    model: ScoreModel,
    n_pairs: int,
    rng: np.random.Generator,
    *,
    genuine_fraction: float = 0.5,
    design: Literal["independent", "clustered"] = "independent",
    n_identities: int | None = None,
    class_assignment: Literal["fixed", "bernoulli"] = "fixed",
    genuine_per_identity: int | None = None,
) -> SimPairs:
    """Draw synthetic pairs. ``independent`` uses fresh identities for every pair (identity-
    disjoint); ``clustered`` reuses a finite identity pool (correlated pairs)."""
    if class_assignment == "fixed":
        n_g = round(n_pairs * genuine_fraction)
    else:
        n_g = int(rng.binomial(n_pairs, genuine_fraction))
    n_i = n_pairs - n_g
    labels = np.r_[np.ones(n_g, np.int64), np.zeros(n_i, np.int64)]
    covis = rng.beta(model.covis_a, model.covis_b, size=n_pairs)
    if design == "independent":
        ids_a = np.arange(n_pairs, dtype=np.int64)
        ids_b = np.r_[ids_a[:n_g], n_pairs + np.arange(n_i, dtype=np.int64)]
        u = rng.normal(0, model.sigma_identity_genuine, n_g)
        v = rng.normal(0, model.sigma_identity_impostor, (n_i, 2)).sum(axis=1)
    else:
        if n_identities is None or n_identities < 2:
            raise ValueError("clustered design needs n_identities >= 2")
        pool_u = rng.normal(0, model.sigma_identity_genuine, n_identities)
        pool_v = rng.normal(0, model.sigma_identity_impostor, n_identities)
        if genuine_per_identity is not None:
            g_ids = np.repeat(np.arange(n_identities), genuine_per_identity)[:n_g]
            if g_ids.size < n_g:
                raise ValueError("genuine_per_identity * n_identities < requested genuine pairs")
        else:
            g_ids = rng.integers(0, n_identities, n_g)
        ia = rng.integers(0, n_identities, n_i)
        ib = (ia + rng.integers(1, n_identities, n_i)) % n_identities  # distinct partner
        ids_a = np.r_[g_ids, ia].astype(np.int64)
        ids_b = np.r_[g_ids, ib].astype(np.int64)
        u = pool_u[g_ids]
        v = pool_v[ia] + pool_v[ib]
    occl = 1.0 - covis
    sd = model.sigma_noise * (1.0 + model.noise_slope * occl)
    mean = np.where(
        labels == 1,
        model.mu_genuine - model.genuine_slope * occl,
        model.mu_impostor + model.impostor_slope * occl,
    )
    effect = np.r_[u, v]
    scores = mean + effect + sd * rng.standard_normal(n_pairs)
    return SimPairs(scores, labels, covis, ids_a, ids_b, design, class_assignment)


def sim_pair_set(pairs: SimPairs) -> PairSet:
    """A PairSet carrying only the independence summary of a simulated pair set (no records)."""
    ids = np.r_[pairs.id_a, pairs.id_b[pairs.id_a != pairs.id_b]]
    _, counts = np.unique(ids, return_counts=True)
    disjoint = bool((counts == 1).all())
    n = int(pairs.labels.size)
    summary = PairSetSummary(
        mode=pairs.design,
        n_pairs=n,
        n_genuine=int(pairs.labels.sum()),
        n_impostor=n - int(pairs.labels.sum()),
        n_identities=int(counts.size),
        n_images=2 * n,
        max_pairs_per_identity=int(counts.max()),
        max_pairs_per_image=1,
        max_pairs_per_source_image=1,
        identity_disjoint=disjoint,
        effective_independent_pairs=n if disjoint else None,
        max_identity_disjoint_subset=n if disjoint else 0,
        strata={},
        note="synthetic simulation pair set",
    )
    return PairSet((), summary, {"spec": {"class_assignment": pairs.class_assignment}})


# --------------------------------------------------------------------------- exact population rates


@dataclass(frozen=True)
class ClassRates:
    accept: float  # P(accepted, c in interval | class)
    error: float  # P(accepted and wrong, c in interval | class)
    mass: float  # P(c in interval)


def class_rates(
    model: ScoreModel,
    rule: ThresholdRule,
    genuine: bool,
    *,
    c_lo: float = 0.0,
    c_hi: float = 1.0,
    grid: int = 4001,
) -> ClassRates:
    edges = np.linspace(0.0, 1.0, grid + 1)
    c = (edges[:-1] + edges[1:]) / 2
    logpdf = (model.covis_a - 1) * np.log(c) + (model.covis_b - 1) * np.log1p(-c)
    w = np.exp(logpdf - logpdf.max())
    w /= w.sum()
    inside = (c >= c_lo) & (c < c_hi) if c_hi < 1.0 else (c >= c_lo)
    gate = c >= rule.c_min if rule.c_min is not None else np.ones_like(c, dtype=bool)
    occl = 1.0 - c
    sd_noise = model.sigma_noise * (1.0 + model.noise_slope * occl)
    if genuine:
        m = model.mu_genuine - model.genuine_slope * occl
        sd = np.sqrt(sd_noise**2 + model.sigma_identity_genuine**2)
    else:
        m = model.mu_impostor + model.impostor_slope * occl
        sd = np.sqrt(sd_noise**2 + 2 * model.sigma_identity_impostor**2)
    p_match = 1.0 - norm_cdf((rule.tau_high - m) / sd)
    p_non = norm_cdf((rule.tau_low - m) / sd) if rule.tau_low < rule.tau_high else 1.0 - p_match
    acc = (p_match + p_non) * gate
    err = (p_non if genuine else p_match) * gate
    return ClassRates(
        float((w * acc)[inside].sum()), float((w * err)[inside].sum()), float(w[inside].sum())
    )


@dataclass(frozen=True)
class RuleTruth:
    acc_g: float
    err_g: float
    acc_i: float
    err_i: float

    @property
    def sfmr(self) -> float:
        return self.err_i / self.acc_i if self.acc_i > 0 else 0.0

    @property
    def sfnmr(self) -> float:
        return self.err_g / self.acc_g if self.acc_g > 0 else 0.0

    @property
    def class_conditional_risk(self) -> float:
        return max(self.sfmr, self.sfnmr)

    def mixture_risk(self, prevalence: float) -> float:
        den = prevalence * self.acc_g + (1 - prevalence) * self.acc_i
        return (prevalence * self.err_g + (1 - prevalence) * self.err_i) / den if den > 0 else 0.0

    def coverage(self, prevalence: float) -> float:
        return prevalence * self.acc_g + (1 - prevalence) * self.acc_i


def rule_truth(
    model: ScoreModel, rule: ThresholdRule, *, c_lo: float = 0.0, c_hi: float = 1.0
) -> RuleTruth:
    g = class_rates(model, rule, True, c_lo=c_lo, c_hi=c_hi)
    i = class_rates(model, rule, False, c_lo=c_lo, c_hi=c_hi)
    return RuleTruth(g.accept, g.error, i.accept, i.error)


def combine(truths: list[RuleTruth]) -> RuleTruth:
    """Combine per-stratum joint rates (each already weighted by stratum mass)."""
    return RuleTruth(
        sum(t.acc_g for t in truths),
        sum(t.err_g for t in truths),
        sum(t.acc_i for t in truths),
        sum(t.err_i for t in truths),
    )


def population_threshold_at_far(model: ScoreModel, far: float) -> float:
    """Threshold t with population impostor P(s >= t) = far (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        r = class_rates(model, ThresholdRule(mid, mid), False)
        if r.error > far:  # error for impostors at tl=th=mid is P(s >= mid)
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
