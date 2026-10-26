"""Risk control for three-way verification decisions (audit: docs/ltt_audit.md).

Estimands (for a fixed decision rule lambda; population quantities):

* ``class_conditional`` (**primary**): sFMR(lambda) = P(d=MATCH | d accepted, y=impostor) and
  sFNMR(lambda) = P(d=NON_MATCH | d accepted, y=genuine). Null H_lambda: sFMR > a_FM OR
  sFNMR > a_FNM. Rejecting it implies, for **any** class prevalence, that the accepted-decision
  error P(d != y | accepted) <= max(a_FM, a_FNM) (it is a convex combination of the two).
* ``mixture_conditional``: E(lambda) = P(d != y | accepted) for the calibration prevalence
  only. Valid only if calibration pairs are i.i.d. draws from the mixture, i.e. class labels
  are Bernoulli (``class_assignment="bernoulli"``), not fixed-count designs.

p-values (Angelopoulos et al., *Learn then Test*, §3.2 uses exactly this for selective
classification): conditional on the accepted count m(lambda), the accepted calibration pairs are
i.i.d. from the accepted-conditional distribution, so the error count is Binomial(m, E(lambda))
and p = P(Bin(m, alpha) <= k) is super-uniform under H_lambda (p = 1 when m = 0). This requires
that acceptance of pair i depends only on pair i and the pre-specified lambda: **no rule
parameter may be estimated on the calibration set itself**. For the class-conditional null
(a union of two nulls) the intersection-union p-value max(p_FM, p_FNM) is valid.

Family-wise error over the pre-specified lambda list: fixed-sequence testing (order fixed before
seeing calibration data; stop at the first non-rejection) or Bonferroni. Any lambda in the
rejected set may then be chosen (here: maximal calibration coverage) — LTT Theorem 1 controls
the supremum risk over the rejected set.

Validity preconditions are checked, not assumed: an identity-disjoint calibration pair set
(``PairSetSummary.identity_disjoint``) is required, and the mixture estimand requires Bernoulli
class assignment. If they fail, ``LTTCalibrator`` refuses (or, with ``require_valid=False``,
labels the result ``guarantee_valid=False``). :class:`EmpiricalCalibrator` never claims a
guarantee. No result is ever valid for test distributions that differ from calibration
(held-out challenge families, real-occlusion sets, other benchmarks).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from veilx.errors import ProtocolViolationError
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH, decisions_from_thresholds
from veilx.pairs.builder import PairSet
from veilx.stats.feasibility import binom_cdf

Estimand = Literal["class_conditional", "mixture_conditional"]
Fwer = Literal["fixed_sequence", "bonferroni"]
IntArr = NDArray[np.int64]

ASSUMPTIONS: tuple[str, ...] = (
    "A1 calibration and test pairs are exchangeable (i.i.d. from one distribution)",
    "A2 calibration pairs are mutually independent (identity-disjoint matching)",
    "A3 decision rules (scores, thresholds grid, gates, strata) are fixed before calibration",
    "A6 the candidate list and its testing order are pre-specified",
)
SCOPE_VALID = (
    "With probability >= 1 - delta over the calibration draw, the selected rule's population "
    "{estimand} is <= target, for test data exchangeable with the calibration data. NOT claimed "
    "for held-out challenge families, real-occlusion benchmarks or any distribution shift."
)
SCOPE_INVALID = (
    "NO statistical guarantee: {reason}. Reported values are empirical calibration-set statistics."
)


@dataclass(frozen=True)
class ThresholdRule:
    """Decision rule: MATCH if s >= tau_high, NON_MATCH if s <= tau_low, else ABSTAIN;
    optional gate: ABSTAIN when the gate covariate (e.g. co-visibility) is < c_min."""

    tau_low: float
    tau_high: float
    c_min: float | None = None

    def __post_init__(self) -> None:
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low must be <= tau_high")

    def decide(self, scores: ArrayLike, gate_values: ArrayLike | None = None) -> IntArr:
        gate = None
        if self.c_min is not None:
            if gate_values is None:
                raise ValueError("rule has c_min but no gate values were supplied")
            gate = np.asarray(gate_values, dtype=np.float64) >= self.c_min
        return decisions_from_thresholds(scores, self.tau_low, self.tau_high, gate=gate)


def symmetric_rules(
    center: float, half_widths: Sequence[float], c_min: float | None = None
) -> list[ThresholdRule]:
    """Rules ordered most-conservative first (largest abstention band first)."""
    widths = sorted({float(w) for w in half_widths}, reverse=True)
    if any(w < 0 for w in widths):
        raise ValueError("half widths must be >= 0")
    return [ThresholdRule(center - w, center + w, c_min) for w in widths]


def order_candidates_on_validation(
    rules: Sequence[ThresholdRule],
    val_scores: ArrayLike,
    val_labels: ArrayLike,
    *,
    n_cal_expected: int,
    target: float,
    delta: float,
    val_gate_values: ArrayLike | None = None,
) -> list[ThresholdRule]:
    """Pre-specify the fixed-sequence order using **validation data only** (never calibration).

    Fixed-sequence testing stops at the first non-rejected hypothesis, so the order matters for
    power but not for validity, *provided it does not depend on the calibration data*. Rules
    are (1) pruned if, scaled to the calibration size, either class would have fewer accepted
    pairs than the zero-error minimum ``ceil(log(delta)/log(1-target))`` (their null could never
    be rejected), then (2) ordered by increasing validation class-conditional risk (ties: lower
    validation coverage first).
    """
    s, y = _validate_inputs(val_scores, val_labels)
    n_min = math.ceil(math.log(delta) / math.log1p(-target))
    frac_g = float((y == 1).mean())
    scale = n_cal_expected / s.size
    keep: list[tuple[float, float, int, ThresholdRule]] = []
    for j, rule in enumerate(rules):
        acc, acc_g, acc_i, fm, fnm = _stats(rule, s, y, val_gate_values)
        if min(acc_g, acc_i) * scale < n_min or not (0 < frac_g < 1):
            continue
        risk = max(fm / acc_i, fnm / acc_g)
        keep.append((risk, acc / s.size, j, rule))
    keep.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in keep]


def binomial_pvalue(errors: int, accepted: int, target: float) -> float:
    """Exact binomial p-value for H0: risk > target, conditional on the accepted count."""
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    if accepted == 0:
        return 1.0
    return binom_cdf(errors, accepted, target)


@dataclass(frozen=True)
class CandidateStats:
    rule: ThresholdRule
    n: int
    accepted: int
    accepted_genuine: int
    accepted_impostor: int
    false_matches: int
    false_non_matches: int
    p_value: float

    @property
    def coverage(self) -> float:
        return self.accepted / self.n if self.n else 0.0

    @property
    def accepted_error(self) -> float | None:
        return (
            (self.false_matches + self.false_non_matches) / self.accepted if self.accepted else None
        )


@dataclass(frozen=True)
class RiskControlResult:
    method: str
    estimand: Estimand
    target_fm: float
    target_fnm: float
    delta: float
    fwer: Fwer | None
    feasible: bool
    rule: ThresholdRule | None
    candidates: tuple[CandidateStats, ...]
    rejected: tuple[int, ...]
    guarantee_valid: bool
    guarantee_scope: str
    assumptions: tuple[str, ...]
    calibration_families: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def decide(self, scores: ArrayLike, gate_values: ArrayLike | None = None) -> IntArr:
        """Apply the selected rule; abstain on everything when no rule was certified/selected."""
        s = np.asarray(scores, dtype=np.float64)
        if self.rule is None:
            return np.full(s.shape, ABSTAIN, dtype=np.int64)
        return self.rule.decide(s, gate_values)


def _stats(
    rule: ThresholdRule, s: NDArray[np.float64], y: IntArr, g: ArrayLike | None
) -> tuple[int, int, int, int, int]:
    d = rule.decide(s, g)
    acc = d != ABSTAIN
    acc_g = int((acc & (y == 1)).sum())
    acc_i = int((acc & (y == 0)).sum())
    fm = int(((d == MATCH) & (y == 0)).sum())
    fnm = int(((d == NON_MATCH) & (y == 1)).sum())
    return int(acc.sum()), acc_g, acc_i, fm, fnm


def _validate_inputs(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray[np.float64], IntArr]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    if s.ndim != 1 or s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty 1-D arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return s, y


class LTTCalibrator:
    """Learn-then-Test with conditional exact-binomial p-values (see module docstring)."""

    def __init__(
        self,
        candidates: Sequence[ThresholdRule],
        *,
        target_fm: float,
        target_fnm: float | None = None,
        delta: float,
        estimand: Estimand = "class_conditional",
        fwer: Fwer = "fixed_sequence",
        require_valid: bool = True,
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate rule is required")
        for t in (target_fm, target_fnm if target_fnm is not None else target_fm, delta):
            if not 0 < t < 1:
                raise ValueError("targets and delta must be in (0, 1)")
        self.candidates = tuple(candidates)
        self.target_fm = target_fm
        self.target_fnm = target_fnm if target_fnm is not None else target_fm
        self.delta = delta
        self.estimand = estimand
        self.fwer = fwer
        self.require_valid = require_valid
        if estimand == "mixture_conditional" and self.target_fnm != self.target_fm:
            raise ValueError("mixture estimand uses a single target")

    def _validity(self, pair_set: PairSet | None, n: int) -> tuple[bool, str]:
        if pair_set is None:
            return False, "calibration pair independence was not established (no PairSet given)"
        if pair_set.summary.n_pairs != n:
            raise ValueError("scores/labels must align one-to-one with pair_set.pairs")
        if (
            not pair_set.summary.identity_disjoint
            or pair_set.summary.max_pairs_per_source_image > 1
        ):
            return False, "calibration pairs share identities/images (dependent observations)"
        if self.estimand == "mixture_conditional":
            spec = pair_set.provenance.get("spec", {})
            if not isinstance(spec, dict) or spec.get("class_assignment") != "bernoulli":
                return False, (
                    "mixture estimand needs Bernoulli class assignment; fixed genuine/impostor "
                    "counts make the pooled error count non-binomial"
                )
        return True, ""

    def fit(
        self,
        scores: ArrayLike,
        labels: ArrayLike,
        *,
        pair_set: PairSet | None,
        gate_values: ArrayLike | None = None,
        calibration_families: tuple[str, ...] = (),
    ) -> RiskControlResult:
        s, y = _validate_inputs(scores, labels)
        valid, reason = self._validity(pair_set, s.size)
        if not valid and self.require_valid:
            raise ProtocolViolationError(f"Learn-then-Test preconditions violated: {reason}")
        stats: list[CandidateStats] = []
        for rule in self.candidates:
            acc, acc_g, acc_i, fm, fnm = _stats(rule, s, y, gate_values)
            if self.estimand == "class_conditional":
                p = max(
                    binomial_pvalue(fm, acc_i, self.target_fm),
                    binomial_pvalue(fnm, acc_g, self.target_fnm),
                )
            else:
                p = binomial_pvalue(fm + fnm, acc, self.target_fm)
            stats.append(CandidateStats(rule, int(s.size), acc, acc_g, acc_i, fm, fnm, p))
        rejected = self._fwer_reject([c.p_value for c in stats])
        chosen: ThresholdRule | None = None
        if rejected:
            best = max(rejected, key=lambda j: (stats[j].coverage, -j))
            chosen = stats[best].rule
        estimand_txt = (
            "selective FMR and selective FNMR (hence accepted-decision error for any prevalence)"
            if self.estimand == "class_conditional"
            else "accepted-decision error at the calibration prevalence"
        )
        scope = (
            SCOPE_VALID.format(estimand=estimand_txt)
            if valid
            else SCOPE_INVALID.format(reason=reason)
        )
        return RiskControlResult(
            method="learn_then_test_binomial_conditional",
            estimand=self.estimand,
            target_fm=self.target_fm,
            target_fnm=self.target_fnm,
            delta=self.delta,
            fwer=self.fwer,
            feasible=chosen is not None,
            rule=chosen,
            candidates=tuple(stats),
            rejected=tuple(rejected),
            guarantee_valid=valid,
            guarantee_scope=scope,
            assumptions=ASSUMPTIONS,
            calibration_families=calibration_families,
            notes=()
            if chosen is not None
            else ("no candidate certified: system abstains on all pairs",),
        )

    def _fwer_reject(self, pvals: Sequence[float]) -> list[int]:
        if self.fwer == "bonferroni":
            return [j for j, p in enumerate(pvals) if p <= self.delta / len(pvals)]
        out = []
        for j, p in enumerate(pvals):
            if p > self.delta:
                break
            out.append(j)
        return out


class EmpiricalCalibrator:
    """Largest-coverage rule whose *empirical* calibration risk meets the target. No guarantee."""

    def __init__(
        self,
        candidates: Sequence[ThresholdRule],
        *,
        target_fm: float,
        target_fnm: float | None = None,
    ) -> None:
        self.candidates = tuple(candidates)
        self.target_fm = target_fm
        self.target_fnm = target_fnm if target_fnm is not None else target_fm

    def fit(
        self, scores: ArrayLike, labels: ArrayLike, *, gate_values: ArrayLike | None = None
    ) -> RiskControlResult:
        s, y = _validate_inputs(scores, labels)
        stats = []
        ok = []
        for j, rule in enumerate(self.candidates):
            acc, acc_g, acc_i, fm, fnm = _stats(rule, s, y, gate_values)
            stats.append(
                CandidateStats(rule, int(s.size), acc, acc_g, acc_i, fm, fnm, float("nan"))
            )
            sfmr = fm / acc_i if acc_i else 0.0
            sfnmr = fnm / acc_g if acc_g else 0.0
            if acc and sfmr <= self.target_fm and sfnmr <= self.target_fnm:
                ok.append(j)
        chosen = stats[max(ok, key=lambda j: (stats[j].coverage, -j))].rule if ok else None
        return RiskControlResult(
            method="empirical_threshold_selection",
            estimand="class_conditional",
            target_fm=self.target_fm,
            target_fnm=self.target_fnm,
            delta=float("nan"),
            fwer=None,
            feasible=chosen is not None,
            rule=chosen,
            candidates=tuple(stats),
            rejected=tuple(ok),
            guarantee_valid=False,
            guarantee_scope=SCOPE_INVALID.format(reason="empirical selection without a test"),
            assumptions=(),
        )


@dataclass(frozen=True)
class StratifiedResult:
    per_stratum: dict[str, RiskControlResult]
    delta_total: float
    guarantee_valid: bool
    guarantee_scope: str

    def decide(
        self, scores: ArrayLike, strata: Sequence[str], gate_values: ArrayLike | None = None
    ) -> IntArr:
        s = np.asarray(scores, dtype=np.float64)
        st = np.asarray(strata)
        g = None if gate_values is None else np.asarray(gate_values, dtype=np.float64)
        out = np.full(s.shape, ABSTAIN, dtype=np.int64)  # unseen strata abstain
        for name, res in self.per_stratum.items():
            m = st == name
            if m.any():
                out[m] = res.decide(s[m], None if g is None else g[m])
        return out


def fit_stratified(
    calibrator_factory: type[LTTCalibrator],
    candidates: Sequence[ThresholdRule],
    scores: ArrayLike,
    labels: ArrayLike,
    strata: Sequence[str],
    pair_set: PairSet | None,
    *,
    target_fm: float,
    target_fnm: float | None = None,
    delta: float,
    estimand: Estimand = "class_conditional",
    fwer: Fwer = "fixed_sequence",
    require_valid: bool = True,
    candidates_by_stratum: Mapping[str, Sequence[ThresholdRule]] | None = None,
) -> StratifiedResult:
    """Per-stratum LTT at delta / S (Bonferroni over strata).

    Strata must be a pre-specified function of each pair alone (e.g. co-visibility bins with
    edges fixed on validation data). With every stratum controlled simultaneously, the overall
    accepted-pair risk (a mixture over strata) is also controlled at the same targets.
    Independence/estimand checks are run once on the whole calibration set.
    """
    s, y = _validate_inputs(scores, labels)
    st = np.asarray(strata)
    if st.shape != s.shape:
        raise ValueError("strata must have one label per pair")
    names = sorted(set(st.tolist()))
    probe = calibrator_factory(
        candidates,
        target_fm=target_fm,
        target_fnm=target_fnm,
        delta=delta,
        estimand=estimand,
        fwer=fwer,
        require_valid=require_valid,
    )
    valid, reason = probe._validity(pair_set, s.size)
    if not valid and require_valid:
        raise ProtocolViolationError(f"Learn-then-Test preconditions violated: {reason}")
    per: dict[str, RiskControlResult] = {}
    for name in names:
        m = st == name
        cands = candidates if candidates_by_stratum is None else candidates_by_stratum.get(name, ())
        if not cands:
            continue  # no pre-specified rule for this stratum: it abstains
        cal = calibrator_factory(
            cands,
            target_fm=target_fm,
            target_fnm=target_fnm,
            delta=delta / len(names),
            estimand=estimand,
            fwer=fwer,
            require_valid=False,
        )
        res = cal.fit(s[m], y[m], pair_set=None)
        per[name] = replace(
            res,
            guarantee_valid=valid,
            guarantee_scope=(
                SCOPE_VALID.format(estimand=f"stratum '{name}' risk")
                if valid
                else SCOPE_INVALID.format(reason=reason)
            ),
        )
    scope = f"Simultaneous control over {len(names)} strata at delta/{len(names)} each; " + (
        SCOPE_VALID.format(estimand="per-stratum and overall accepted risk")
        if valid
        else SCOPE_INVALID.format(reason=reason)
    )
    return StratifiedResult(per, delta, valid, scope)
