"""Delete-one-identity jackknife intervals for pair-based rates.

For a ratio statistic theta = sum(num) / sum(den) over pairs, deleting identity i removes every
pair that involves i (as either side, so impostor pairs depend on two identities). The jackknife
variance (G - 1)/G * sum_i (theta_(i) - mean)^2 is the classical estimator for (generalised)
U-statistics such as FMR over identity pairs, which is exactly the two-identity dependence that
makes naive and product-weight bootstraps miscalibrated.

Intervals:
* ``jackknife_t``     theta_hat +/- t_{G-1} * se
* ``jackknife_logit`` same on the logit scale, back-transformed (keeps the interval in (0, 1)
  and allows asymmetry near 0, e.g. FMR at 0.1 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from veilx.stats.bootstrap import encode_identities


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation (|error| < 1.2e-9)."""
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - 0.02425:
        return -_norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile via the Cornish-Fisher expansion (adequate for df >= 5)."""
    z = _norm_ppf(p)
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    return z + g1 / df + g2 / df**2 + g3 / df**3


@dataclass(frozen=True)
class JackknifeResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    method: str
    n_identities: int
    level: float


def leave_one_identity_out(
    numerator: ArrayLike, denominator: ArrayLike, id_a: ArrayLike, id_b: ArrayLike
) -> tuple[float, NDArray[np.float64], int]:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    ca, cb, g = encode_identities(id_a, id_b)
    n_i = np.zeros(g)
    d_i = np.zeros(g)
    np.add.at(n_i, ca, num)
    np.add.at(d_i, ca, den)
    other = ca != cb  # count each pair once per *distinct* identity it involves
    np.add.at(n_i, cb[other], num[other])
    np.add.at(d_i, cb[other], den[other])
    total_n, total_d = num.sum(), den.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (total_n - n_i) / (total_d - d_i)
    return float(total_n / total_d), loo, g


def jackknife_interval(
    numerator: ArrayLike,
    denominator: ArrayLike,
    id_a: ArrayLike,
    id_b: ArrayLike,
    *,
    level: float = 0.95,
    scale: str = "logit",
) -> JackknifeResult:
    theta, loo, g = leave_one_identity_out(numerator, denominator, id_a, id_b)
    loo = loo[np.isfinite(loo)]
    t = t_ppf(1 - (1 - level) / 2, max(g - 1, 1))
    if scale == "logit":
        eps = 0.5 / max(float(np.asarray(denominator, dtype=np.float64).sum()), 1.0)

        def lg(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
            x = np.clip(np.asarray(x, dtype=np.float64), eps, 1 - eps)
            return np.asarray(np.log(x / (1 - x)))

        phi, phi_loo = float(lg(theta)), lg(loo)
        se = math.sqrt((g - 1) / g * float(np.sum((phi_loo - phi_loo.mean()) ** 2)))
        lo, hi = phi - t * se, phi + t * se
        return JackknifeResult(
            theta, se, 1 / (1 + math.exp(-lo)), 1 / (1 + math.exp(-hi)), "jackknife_logit", g, level
        )
    se = math.sqrt((g - 1) / g * float(np.sum((loo - loo.mean()) ** 2)))
    return JackknifeResult(theta, se, theta - t * se, theta + t * se, "jackknife_t", g, level)
