"""Decision-level (selective) metrics for three-way verification decisions.

Decision codes: ``MATCH = 1``, ``NON_MATCH = 0``, ``ABSTAIN = -1``.
Labels: 1 = genuine, 0 = impostor.

Definitions (docs/metrics.md):

* coverage = P(d != ABSTAIN), also per class (genuine / impostor coverage);
* FM = d=MATCH & y=0; FNM = d=NON_MATCH & y=1;
* selective FMR  = FM  / accepted impostors;  selective FNMR = FNM / accepted genuines;
* accepted-decision error (conditional) = (FM + FNM) / accepted = P(d != y | d != ABSTAIN);
* generalized risk = (FM + FNM) / all pairs = P(d != y, d != ABSTAIN)        [Traub et al. 2024];
* risk-coverage curves sweep a confidence threshold over a *fixed* base decision; AUGRC is
  the area under generalized risk vs coverage [Traub et al. 2024, Eq. 6], AURC the area under
  selective risk vs coverage [ibid., Eq. 4]. Ties in confidence are accepted together; the
  AUGRC integral uses the trapezoid rule between tie groups, which equals the expected value
  under random tie-breaking and matches the closed form
  (1 - AUROC_f) * acc * (1 - acc) + (1 - acc)^2 / 2  [ibid., Eq. 7].

Rates with an empty denominator are returned as ``None`` (never silently as 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MATCH, NON_MATCH, ABSTAIN = 1, 0, -1
FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]


def _arrays(decisions: ArrayLike, labels: ArrayLike) -> tuple[IntArr, IntArr]:
    d = np.asarray(decisions).astype(np.int64)
    y = np.asarray(labels).astype(np.int64)
    if d.ndim != 1 or d.shape != y.shape:
        raise ValueError("decisions and labels must be 1-D and of equal length")
    if not np.isin(d, (MATCH, NON_MATCH, ABSTAIN)).all():
        raise ValueError("decisions must be 1 (match), 0 (non-match) or -1 (abstain)")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if d.size == 0:
        raise ValueError("empty input")
    return d, y


def decisions_from_thresholds(
    scores: ArrayLike,
    tau_low: float,
    tau_high: float,
    *,
    gate: ArrayLike | None = None,
) -> IntArr:
    """MATCH if s >= tau_high, NON_MATCH if s <= tau_low, else ABSTAIN.

    Where ``gate`` is False the decision is forced to ABSTAIN.
    """
    if tau_low > tau_high:
        raise ValueError("tau_low must be <= tau_high")
    s = np.asarray(scores, dtype=np.float64)
    d = np.full(s.shape, ABSTAIN, dtype=np.int64)
    d[s >= tau_high] = MATCH
    d[(s <= tau_low) & (s < tau_high)] = NON_MATCH
    if gate is not None:
        d[~np.asarray(gate, dtype=bool)] = ABSTAIN
    return d


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


@dataclass(frozen=True)
class DecisionSummary:
    n: int
    n_genuine: int
    n_impostor: int
    confusion: dict[str, dict[str, int]]  # label -> decision -> count
    accepted: int
    false_matches: int
    false_non_matches: int
    coverage: float
    coverage_genuine: float | None
    coverage_impostor: float | None
    selective_fmr: float | None
    selective_fnmr: float | None
    accepted_error: float | None
    generalized_risk: float
    decision_distribution: dict[str, float]

    def cost_weighted_risk(self, cost_fm: float, cost_fnm: float = 1.0) -> float | None:
        """(c_FM * FM + c_FNM * FNM) / accepted."""
        if self.accepted == 0:
            return None
        return (cost_fm * self.false_matches + cost_fnm * self.false_non_matches) / self.accepted


def summarize_decisions(decisions: ArrayLike, labels: ArrayLike) -> DecisionSummary:
    d, y = _arrays(decisions, labels)
    names = {MATCH: "match", NON_MATCH: "non_match", ABSTAIN: "abstain"}
    confusion = {
        lab: {names[k]: int(((y == yv) & (d == k)).sum()) for k in names}
        for lab, yv in (("genuine", 1), ("impostor", 0))
    }
    n_g, n_i = int((y == 1).sum()), int((y == 0).sum())
    acc_g = confusion["genuine"]["match"] + confusion["genuine"]["non_match"]
    acc_i = confusion["impostor"]["match"] + confusion["impostor"]["non_match"]
    fm, fnm = confusion["impostor"]["match"], confusion["genuine"]["non_match"]
    accepted = acc_g + acc_i
    return DecisionSummary(
        n=int(d.size),
        n_genuine=n_g,
        n_impostor=n_i,
        confusion=confusion,
        accepted=accepted,
        false_matches=fm,
        false_non_matches=fnm,
        coverage=accepted / d.size,
        coverage_genuine=_ratio(acc_g, n_g),
        coverage_impostor=_ratio(acc_i, n_i),
        selective_fmr=_ratio(fm, acc_i),
        selective_fnmr=_ratio(fnm, acc_g),
        accepted_error=_ratio(fm + fnm, accepted),
        generalized_risk=(fm + fnm) / d.size,
        decision_distribution={v: float((d == k).mean()) for k, v in names.items()},
    )


@dataclass(frozen=True)
class RiskCoverageCurve:
    coverage: FloatArr  # starts at 0
    selective_risk: FloatArr  # nan at coverage 0
    generalized_risk: FloatArr
    thresholds: FloatArr  # confidence threshold per point (+inf at coverage 0)


def risk_coverage_curve(confidence: ArrayLike, failures: ArrayLike) -> RiskCoverageCurve:
    """Accept the most confident samples first; one point per distinct confidence value."""
    c = np.asarray(confidence, dtype=np.float64)
    f = np.asarray(failures).astype(np.int64)
    if c.ndim != 1 or c.shape != f.shape or c.size == 0:
        raise ValueError("confidence and failures must be non-empty 1-D arrays of equal length")
    if not np.all(np.isfinite(c)):
        raise ValueError("confidence must be finite")
    if not np.isin(f, (0, 1)).all():
        raise ValueError("failures must be 0/1")
    order = np.argsort(-c, kind="mergesort")
    cs, fs = c[order], f[order]
    ends = np.r_[np.flatnonzero(np.diff(cs)), cs.size - 1]
    k = ends + 1
    errs = np.cumsum(fs)[ends]
    n = c.size
    cov = np.r_[0.0, k / n]
    sel = np.r_[np.nan, errs / k]
    gen = np.r_[0.0, errs / n]
    thr = np.r_[np.inf, cs[ends]]
    return RiskCoverageCurve(cov, sel, gen, thr)


def augrc(confidence: ArrayLike, failures: ArrayLike) -> float:
    """Area under the generalized risk-coverage curve (trapezoid over tie groups)."""
    rc = risk_coverage_curve(confidence, failures)
    return float(
        np.sum(np.diff(rc.coverage) * (rc.generalized_risk[1:] + rc.generalized_risk[:-1]) / 2)
    )


def aurc(confidence: ArrayLike, failures: ArrayLike) -> float:
    """Area under the selective risk-coverage curve.

    Right-endpoint sum over tie groups: sum_g dC_g * SR(C_g); without ties this equals the mean
    of the selective risk over k = 1..N.
    """
    rc = risk_coverage_curve(confidence, failures)
    return float(np.sum(np.diff(rc.coverage) * rc.selective_risk[1:]))


def verification_failures(base_decisions: ArrayLike, labels: ArrayLike) -> IntArr:
    """Failure indicator of a non-abstaining base decision (1 = wrong)."""
    d, y = _arrays(base_decisions, labels)
    if (d == ABSTAIN).any():
        raise ValueError("base decisions for risk-coverage analysis must not abstain")
    return np.asarray(d != y, dtype=np.int64)
