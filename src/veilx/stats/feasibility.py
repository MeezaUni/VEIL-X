"""Statistical-feasibility helpers (pure Python, exact binomial).

Used to check, *before* any experiment, whether a registered constant (risk target,
confidence level, FAR operating point) can be certified with the calibration data that the
protocol can actually supply. Nothing here touches face data.

Independence caveat: binomial bounds assume independent Bernoulli trials. Face-verification
pairs that share identities are dependent, so the protocol's primary calibration set uses
an *identity-disjoint matching* (each identity in at most one pair). The helpers below
therefore also report how many independent pairs a given number of identities can supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def binom_cdf(k: int, n: int, p: float) -> float:
    """P[X <= k] for X ~ Binomial(n, p), computed in log space for stability."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    """One-sided exact upper confidence bound on a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if k >= n:
        return 1.0
    alpha = 1.0 - confidence
    lo, hi = k / n, 1.0
    for _ in range(200):  # bisection: find p with P[X <= k | p] = alpha
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def binomial_pvalue_risk_above(k: int, n: int, target: float) -> float:
    """Exact p-value for H0: risk > target given k errors in n (LTT's binomial p-value)."""
    return binom_cdf(k, n, target)


def zero_error_n(target: float, confidence: float) -> int:
    """Smallest n such that observing 0 errors certifies risk <= target at ``confidence``."""
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    return math.ceil(math.log(1 - confidence) / math.log1p(-target))


def min_n_to_certify(
    target: float, observed_rate: float, confidence: float, n_max: int = 10**7
) -> int | None:
    """Smallest n such that k = floor(observed_rate * n) errors gives an upper bound <= target.

    Returns ``None`` if ``observed_rate >= target`` (never certifiable) or n exceeds ``n_max``.
    """
    if observed_rate >= target:
        return None
    lo, hi = 1, 1
    while hi <= n_max:
        if clopper_pearson_upper(math.floor(observed_rate * hi), hi, confidence) <= target:
            break
        lo, hi = hi, hi * 2
    else:
        return None
    while lo < hi:  # not strictly monotone in n because of flooring; search is a good guide
        mid = (lo + hi) // 2
        if clopper_pearson_upper(math.floor(observed_rate * mid), mid, confidence) <= target:
            hi = mid
        else:
            lo = mid + 1
    return hi


@dataclass(frozen=True)
class IndependentPairCapacity:
    n_identities: int
    max_disjoint_impostor_pairs: int
    max_disjoint_genuine_pairs: int  # needs >= 2 images per identity

    @staticmethod
    def from_identities(
        n_identities: int, *, genuine_fraction: float = 0.5
    ) -> IndependentPairCapacity:
        """Capacity of an identity-disjoint matching with a given genuine:impostor mix.

        A genuine pair consumes 1 identity, an impostor pair consumes 2. With ``g`` genuine
        and ``m`` impostor pairs: g + 2m <= N and g / (g + m) = genuine_fraction.
        """
        if not 0 < genuine_fraction < 1:
            raise ValueError("genuine_fraction must be in (0, 1)")
        ratio = genuine_fraction / (1 - genuine_fraction)  # g = ratio * m
        m = math.floor(n_identities / (ratio + 2))
        g = math.floor(ratio * m)
        return IndependentPairCapacity(n_identities, m, g)
