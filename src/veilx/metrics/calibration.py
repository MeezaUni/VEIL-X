"""Probability calibration metrics for p(match).

* ECE (expected calibration error) of the *binary* probability p(match):
  ECE = sum_b (n_b / N) * | mean(y in b) - mean(p in b) |.
  Guo et al. (ICML 2017) define ECE for the top-label confidence with equal-width bins; VEIL-X's
  registered variant is the binary p(match) with **equal-mass** bins (15), and equal-width is
  available for comparability. The variant is always reported alongside the value.
* Brier score = mean((p - y)^2).
* Reliability diagram data (per-bin mean prediction, observed frequency, count) and a
  dependency-free SVG rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from veilx.errors import InsufficientSamplesError

BinStrategy = Literal["equal_mass", "equal_width"]


def _check(probs: ArrayLike, labels: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if p.ndim != 1 or p.shape != y.shape or p.size == 0:
        raise ValueError("probs and labels must be non-empty 1-D arrays of equal length")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return p, y


@dataclass(frozen=True)
class ReliabilityDiagram:
    strategy: BinStrategy
    bin_lower: tuple[float, ...]
    bin_upper: tuple[float, ...]
    mean_predicted: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def ece(self) -> float:
        n = sum(self.counts)
        return float(
            sum(
                c / n * abs(o - m)
                for c, o, m in zip(
                    self.counts, self.observed_frequency, self.mean_predicted, strict=True
                )
            )
        )


def reliability_diagram(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 15, strategy: BinStrategy = "equal_mass"
) -> ReliabilityDiagram:
    p, y = _check(probs, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "equal_mass":
        if p.size < n_bins:
            raise InsufficientSamplesError(f"{p.size} samples cannot fill {n_bins} equal-mass bins")
        order = np.argsort(p, kind="mergesort")
        groups = [g for g in np.array_split(order, n_bins) if g.size]
    else:
        idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
        groups = [np.flatnonzero(idx == b) for b in range(n_bins)]
        groups = [g for g in groups if g.size]
    lower, upper, mean_p, obs, cnt = [], [], [], [], []
    for g in groups:
        lower.append(float(p[g].min()))
        upper.append(float(p[g].max()))
        mean_p.append(float(p[g].mean()))
        obs.append(float(y[g].mean()))
        cnt.append(int(g.size))
    return ReliabilityDiagram(
        strategy, tuple(lower), tuple(upper), tuple(mean_p), tuple(obs), tuple(cnt)
    )


def expected_calibration_error(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 15, strategy: BinStrategy = "equal_mass"
) -> float:
    return reliability_diagram(probs, labels, n_bins, strategy).ece


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    p, y = _check(probs, labels)
    return float(np.mean((p - y) ** 2))


def reliability_svg(diagram: ReliabilityDiagram, *, title: str, size: int = 320) -> str:
    """Minimal SVG: diagonal reference, per-bin points sized by count. No external deps."""
    pad, plot = 40, size - 60
    total = max(sum(diagram.counts), 1)

    def xy(px: float, py: float) -> tuple[float, float]:
        return pad + px * plot, pad + (1 - py) * plot

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="sans-serif" font-size="11">',
        f'<text x="{pad}" y="18">{_esc(title)}</text>',
        f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#888"/>',
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#bbb" stroke-dasharray="4"/>'.format(
            *xy(0, 0), *xy(1, 1)
        ),
    ]
    for m, o, c in zip(
        diagram.mean_predicted, diagram.observed_frequency, diagram.counts, strict=True
    ):
        cx, cy = xy(m, o)
        r = 2 + 8 * (c / total) ** 0.5
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#3060c0" fill-opacity="0.7"/>'
        )
    parts += [
        f'<text x="{pad}" y="{size - 8}">mean predicted p(match) [{diagram.strategy}]</text>',
        f'<text x="6" y="{pad - 6}">observed</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
