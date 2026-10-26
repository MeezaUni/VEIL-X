"""Identity-cluster bootstrap for pair-based verification metrics.

Resampling unit = **identity** (protocol_design.md §9). One bootstrap replicate draws the test
identities with replacement (multiplicities ``W_i``); a genuine pair of identity ``i`` gets weight
``W_i``, an impostor pair (i, j) gets weight ``W_i * W_j`` (it survives only if both identities
are drawn). Statistics are computed as **weighted** quantities, so no pair arrays are copied.
Method differences use the *same* weights for both methods (paired bootstrap).

A naive pair-level bootstrap is provided only to quantify, in simulation, how badly ignoring
identity clustering understates uncertainty. Interval coverage of either method is an
empirical property that VEIL-X validates by simulation (``veilx.sim``) before relying on it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArr = NDArray[np.float64]
WeightedStat = Callable[[FloatArr], FloatArr]  # (B, P) weights -> (B,) values


def encode_identities(
    id_a: ArrayLike, id_b: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Map arbitrary identity labels to 0..K-1 integer codes."""
    a = np.asarray(id_a)
    b = np.asarray(id_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("id_a and id_b must be 1-D and of equal length")
    uniq, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv = inv.astype(np.int64)
    return inv[: a.size], inv[a.size :], int(uniq.size)


def identity_weight_chunks(
    code_a: NDArray[np.int64],
    code_b: NDArray[np.int64],
    n_identities: int,
    n_boot: int,
    rng: np.random.Generator,
    chunk: int = 256,
) -> Iterator[FloatArr]:
    same = code_a == code_b
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        w_id = rng.multinomial(
            n_identities, np.full(n_identities, 1.0 / n_identities), size=b
        ).astype(np.float64)
        wa, wb = w_id[:, code_a], w_id[:, code_b]
        yield np.where(same[None, :], wa, wa * wb)
        done += b


def pair_weight_chunks(
    n_pairs: int, n_boot: int, rng: np.random.Generator, chunk: int = 256
) -> Iterator[FloatArr]:
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        yield rng.multinomial(n_pairs, np.full(n_pairs, 1.0 / n_pairs), size=b).astype(np.float64)
        done += b


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    level: float
    n_boot: int
    n_valid: int  # replicates where the statistic was defined
    unit: Literal["identity", "pair"]
    n_units: int


def bootstrap_ci(
    stat: WeightedStat,
    *,
    n_pairs: int,
    unit: Literal["identity", "pair"],
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
    id_a: ArrayLike | None = None,
    id_b: ArrayLike | None = None,
    chunk: int = 256,
) -> BootstrapResult:
    """Percentile CI of a weighted statistic. ``stat`` must accept (B, P) weights."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    estimate = float(stat(np.ones((1, n_pairs)))[0])
    if unit == "identity":
        if id_a is None or id_b is None:
            raise ValueError("identity bootstrap needs id_a and id_b")
        ca, cb, k = encode_identities(id_a, id_b)
        if ca.size != n_pairs:
            raise ValueError("identity arrays must have one entry per pair")
        chunks = identity_weight_chunks(ca, cb, k, n_boot, rng, chunk)
        n_units = k
    else:
        chunks = pair_weight_chunks(n_pairs, n_boot, rng, chunk)
        n_units = n_pairs
    values = np.concatenate([stat(w) for w in chunks])
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return BootstrapResult(
            estimate, float("nan"), float("nan"), level, n_boot, 0, unit, n_units
        )
    lo, hi = np.quantile(valid, [(1 - level) / 2, 1 - (1 - level) / 2])
    return BootstrapResult(
        estimate, float(lo), float(hi), level, n_boot, int(valid.size), unit, n_units
    )


def weighted_rate(numerator_mask: ArrayLike, denominator_mask: ArrayLike) -> WeightedStat:
    """Statistic sum(w * num) / sum(w * den); NaN where the weighted denominator is 0."""
    num = np.asarray(numerator_mask, dtype=np.float64)
    den = np.asarray(denominator_mask, dtype=np.float64)

    def f(w: FloatArr) -> FloatArr:
        d = w @ den
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, (w @ num) / np.where(d > 0, d, 1.0), np.nan)

    return f
