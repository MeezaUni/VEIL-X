"""Rate estimates with the protocol's interval methods, and verification/selective summaries.

Interval methods (protocol ``bootstrap.interval_method_by_statistic``, validated by SIM-010):
genuine-pair rates use the delete-one-identity jackknife on the logit scale; impostor-pair rates
use the linear delete-one-identity jackknife. When a rate has zero events the jackknife
collapses, so the exact Clopper-Pearson upper bound is reported as well; it treats pairs as
independent, which is anti-conservative for clustered test sets, and is flagged as such.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from veilx.eval.scoring import ScoredPairs
from veilx.metrics.roc import equal_error_rate, roc_auc
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH
from veilx.stats.feasibility import clopper_pearson_upper
from veilx.stats.jackknife import jackknife_interval

Kind = Literal["genuine", "impostor"]
LEVEL = 0.95


def rate(
    events: NDArray[np.bool_], denom: NDArray[np.bool_], sp: ScoredPairs, kind: Kind
) -> dict[str, Any]:
    """Rate = sum(events & denom) / sum(denom) with an identity-jackknife interval."""
    k = int((events & denom).sum())
    n = int(denom.sum())
    out: dict[str, Any] = {"k": k, "n": n, "value": (k / n) if n else None}
    if n == 0:
        return out | {"lo": None, "hi": None, "method": None}
    if k == 0 or k == n:
        cp = clopper_pearson_upper(k, n, LEVEL) if k == 0 else None
        return out | {
            "lo": 0.0 if k == 0 else None,
            "hi": cp,
            "method": "clopper_pearson_upper_pairs_iid (jackknife degenerate at 0 events)"
            if k == 0
            else "degenerate (all events)",
        }
    m = denom
    jk = jackknife_interval(
        (events & m)[m].astype(np.float64),
        np.ones(n),
        sp.id_a[m],
        sp.id_b[m],
        level=LEVEL,
        scale="logit" if kind == "genuine" else "linear",
    )
    return out | {
        "lo": max(0.0, jk.ci_low),
        "hi": min(1.0, jk.ci_high),
        "method": jk.method,
        "n_identities": jk.n_identities,
    }


def verification_summary(sp: ScoredPairs, thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Threshold-free metrics plus FMR/FNMR at externally fixed thresholds."""
    g, i = sp.label == 1, sp.label == 0
    out: dict[str, Any] = {
        "n_pairs": len(sp),
        "n_genuine": int(g.sum()),
        "n_impostor": int(i.sum()),
        "n_identities": int(np.unique(np.concatenate([sp.id_a, sp.id_b])).size),
    }
    if g.any() and i.any():
        eer = equal_error_rate(sp.score, sp.label)
        out |= {
            "roc_auc": roc_auc(sp.score, sp.label),
            "eer": eer.eer,
            "eer_threshold": eer.threshold,
        }
    out["score_quantiles"] = {
        name: np.quantile(sp.score[m], [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
        .round(5)
        .tolist()
        for name, m in (("genuine", g), ("impostor", i))
        if m.any()
    }
    ops = {}
    for name, t in thresholds.items():
        acc = sp.score >= t
        ops[name] = {
            "threshold": t,
            "fmr": rate(acc, i, sp, "impostor"),
            "fnmr": rate(~acc, g, sp, "genuine"),
        }
    out["operating_points"] = ops
    return out


def selective_summary(
    decisions: NDArray[np.int64], sp: ScoredPairs, target_fm: float, target_fnm: float
) -> dict[str, Any]:
    g, i = sp.label == 1, sp.label == 0
    acc = decisions != ABSTAIN
    sfmr = rate(decisions == MATCH, acc & i, sp, "impostor")
    sfnmr = rate(decisions == NON_MATCH, acc & g, sp, "genuine")
    err = int(((decisions == MATCH) & i).sum() + ((decisions == NON_MATCH) & g).sum())
    n_acc = int(acc.sum())

    def verdict(r: Mapping[str, Any], target: float) -> str:
        if r["value"] is None:
            return "no_accepted_pairs"
        if r["lo"] is not None and r["lo"] > target:
            return "violated_ci_excludes_target"
        if r["value"] > target:
            return "point_estimate_above_target"
        return "within_target"

    return {
        "n_pairs": len(sp),
        "coverage": n_acc / len(sp) if len(sp) else None,
        "coverage_genuine": float(acc[g].mean()) if g.any() else None,
        "coverage_impostor": float(acc[i].mean()) if i.any() else None,
        "selective_fmr": sfmr,
        "selective_fnmr": sfnmr,
        "accepted_error": err / n_acc if n_acc else None,
        "verdict_fm": verdict(sfmr, target_fm),
        "verdict_fnm": verdict(sfnmr, target_fnm),
        "target_fm": target_fm,
        "target_fnm": target_fnm,
    }


FPR_GRID = tuple(float(x) for x in np.logspace(-4, 0, 41))
HIST_EDGES = tuple(float(x) for x in np.linspace(-0.4, 1.0, 71))


def roc_points(sp: ScoredPairs) -> dict[str, list[float]]:
    """TPR at a fixed log-spaced FPR grid (accept >= threshold; empirical, aggregate only)."""
    gen = np.sort(sp.score[sp.label == 1])
    imp = np.sort(sp.score[sp.label == 0])[::-1]
    tpr = []
    for f in FPR_GRID:
        k = int(np.floor(f * imp.size + 1e-12))
        t = -np.inf if k >= imp.size else float(np.nextafter(imp[k], np.inf))
        tpr.append(float((gen >= t).mean()))
    return {"fpr": list(FPR_GRID), "tpr": tpr}


def score_histogram(sp: ScoredPairs) -> dict[str, list[float] | list[int]]:
    edges = np.asarray(HIST_EDGES)
    clip = np.clip(sp.score, edges[0], edges[-1] - 1e-9)
    return {
        "edges": list(HIST_EDGES),
        "genuine": np.histogram(clip[sp.label == 1], edges)[0].tolist(),
        "impostor": np.histogram(clip[sp.label == 0], edges)[0].tolist(),
    }
