"""Paired comparisons for Phase 3: identity-jackknife differences, p-values and Holm.

* :func:`jackknife_ratio_difference` - two ratio statistics computed on the SAME pairs (possibly
  with different denominators, e.g. selective FNMR of two decision rules whose accepted sets
  differ). Each delete-one-identity replicate removes every pair involving the identity from BOTH
  ratios, so the dependence between the two systems is preserved.
* :func:`normal_p_value` - two-sided p-value of an estimate / se ratio. With thousands of
  identities the t_{G-1} and normal distributions coincide to < 1e-3, so the normal tail is used.
* :func:`holm` - Holm step-down adjusted p-values (family-wise error control).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from veilx.stats.jackknife import leave_one_identity_out, t_ppf


@dataclass(frozen=True)
class PairedDifference:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_identities: int
    level: float


def normal_p_value(estimate: float, se: float) -> float:
    if se <= 0:
        return 0.0 if estimate != 0 else 1.0
    z = abs(estimate) / se
    return math.erfc(z / math.sqrt(2.0))


def jackknife_ratio_difference(
    num_a: ArrayLike,
    den_a: ArrayLike,
    num_b: ArrayLike,
    den_b: ArrayLike,
    id_a: ArrayLike,
    id_b: ArrayLike,
    *,
    level: float = 0.95,
) -> PairedDifference:
    """theta = sum(num_a)/sum(den_a) - sum(num_b)/sum(den_b) on the same pairs."""
    ta, loo_a, g = leave_one_identity_out(num_a, den_a, id_a, id_b)
    tb, loo_b, _ = leave_one_identity_out(num_b, den_b, id_a, id_b)
    d = loo_a - loo_b
    d = d[np.isfinite(d)]
    est = ta - tb
    se = math.sqrt((g - 1) / g * float(np.sum((d - d.mean()) ** 2))) if d.size > 1 else float("nan")
    t = t_ppf(1 - (1 - level) / 2, max(g - 1, 1))
    return PairedDifference(est, se, est - t * se, est + t * se, normal_p_value(est, se), g, level)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values in the original order (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adj[i] = running
    return adj
