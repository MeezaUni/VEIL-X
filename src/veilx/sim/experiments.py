"""Deterministic Monte Carlo experiments on the synthetic score model.

Every function returns a JSON-serialisable dict tagged ``evidence_type = "synthetic_simulation"``.
These results describe the *statistical machinery* (bootstrap, pair independence, Learn-then-Test,
stratification) under a known synthetic score model. **They are not evidence about the
performance of any face-recognition system.**

Scales: ``quick`` (used by tests; few replicates) and ``full`` (used for the committed reports).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from veilx.seeding import make_rng
from veilx.selective.ltt import (
    Estimand,
    LTTCalibrator,
    ThresholdRule,
    fit_stratified,
    order_candidates_on_validation,
    symmetric_rules,
)
from veilx.sim.model import (
    RuleTruth,
    ScoreModel,
    combine,
    population_threshold_at_far,
    rule_truth,
    sample_pairs,
    sim_pair_set,
)
from veilx.stats.bootstrap import bootstrap_ci, weighted_rate
from veilx.stats.feasibility import clopper_pearson_upper

Scale = Literal["quick", "full"]
SIM_VERSION = "1.0.0"
DISCLAIMER = (
    "SIMULATED: synthetic scores from veilx.sim.model.ScoreModel. Characterises statistical "
    "procedures only; NOT evidence of face-recognition performance."
)
BASE = ScoreModel()
CENTER = 2.0  # decision centre for symmetric rules (between degraded genuine and impostor means)
HALF_WIDTHS = [round(0.1 * k, 2) for k in range(26)]  # 0.0 .. 2.5


def _wrap(
    name: str, scale: Scale, seed: int, params: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    return {
        "evidence_type": "synthetic_simulation",
        "disclaimer": DISCLAIMER,
        "experiment": name,
        "sim_version": SIM_VERSION,
        "scale": scale,
        "seed": seed,
        "params": params,
        "results": results,
    }


def _mc_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def _truth_table(model: ScoreModel, rules: list[ThresholdRule]) -> dict[ThresholdRule, RuleTruth]:
    return {r: rule_truth(model, r) for r in rules}


def _truth_fn(
    table: dict[ThresholdRule, RuleTruth],
) -> Callable[[ThresholdRule], tuple[float, float]]:
    """(true class-conditional risk, true coverage at prevalence 0.5) of a rule."""

    def f(rule: ThresholdRule) -> tuple[float, float]:
        return table[rule].class_conditional_risk, table[rule].coverage(0.5)

    return f


# --------------------------------------------------------------------------- E1


def bootstrap_coverage(scale: Scale = "full", seed: int = 101) -> dict[str, Any]:
    """Coverage of 95% CIs for FNMR/FMR at a fixed threshold: identity-cluster vs pair bootstrap."""
    reps, n_boot = (300, 400) if scale == "full" else (25, 120)
    model = BASE.with_(sigma_identity_genuine=0.8, sigma_identity_impostor=0.5)
    n_ids, g_per_id, n_imp = 150, 6, 3000
    t = population_threshold_at_far(model, 0.01)
    rule = ThresholdRule(t, t)
    truth = rule_truth(model, rule)
    true_fnmr = truth.err_g / truth.acc_g
    true_fmr = truth.err_i / truth.acc_i
    units: tuple[Literal["identity", "pair"], ...] = ("identity", "pair")
    hits = {u: {"fnmr": 0, "fmr": 0} for u in units}
    widths: dict[str, dict[str, list[float]]] = {u: {"fnmr": [], "fmr": []} for u in units}
    for rep in range(reps):
        rng = make_rng(seed, "e1", rep)
        sp = sample_pairs(
            model,
            n_ids * g_per_id + n_imp,
            rng,
            genuine_fraction=n_ids * g_per_id / (n_ids * g_per_id + n_imp),
            design="clustered",
            n_identities=n_ids,
            genuine_per_identity=g_per_id,
        )
        g, acc = sp.labels == 1, sp.scores >= t
        stats = {"fnmr": weighted_rate(g & ~acc, g), "fmr": weighted_rate(~g & acc, ~g)}
        truths = {"fnmr": true_fnmr, "fmr": true_fmr}
        for unit in units:
            for name, stat in stats.items():
                res = bootstrap_ci(
                    stat,
                    n_pairs=sp.labels.size,
                    unit=unit,
                    rng=make_rng(seed, "e1b", rep, unit, name),
                    n_boot=n_boot,
                    id_a=sp.id_a,
                    id_b=sp.id_b,
                )
                hits[unit][name] += int(res.ci_low <= truths[name] <= res.ci_high)
                widths[unit][name].append(res.ci_high - res.ci_low)
    results: dict[str, Any] = {
        unit: {
            name: {
                "coverage": hits[unit][name] / reps,
                "mc_se": _mc_se(hits[unit][name] / reps, reps),
                "mean_ci_width": float(np.mean(widths[unit][name])),
            }
            for name in ("fnmr", "fmr")
        }
        for unit in hits
    }
    results["truth"] = {"threshold": t, "fnmr": true_fnmr, "fmr": true_fmr}
    params = {
        "model": asdict(model),
        "n_identities": n_ids,
        "genuine_per_identity": g_per_id,
        "n_impostor_pairs": n_imp,
        "reps": reps,
        "n_boot": n_boot,
        "nominal": 0.95,
    }
    return _wrap("bootstrap_coverage", scale, seed, params, results)


# --------------------------------------------------------------------------- LTT helpers


def _ltt_once(
    cal_model: ScoreModel,
    n_cal: int,
    rng: np.random.Generator,
    rules: list[ThresholdRule],
    *,
    target: float,
    delta: float,
    design: Literal["independent", "clustered"] = "independent",
    n_identities: int | None = None,
    estimand: Estimand = "class_conditional",
    class_assignment: Literal["fixed", "bernoulli"] = "fixed",
) -> ThresholdRule | None:
    # Validation draw (independent of calibration) fixes the candidate order -- assumption A6.
    val = sample_pairs(cal_model, n_cal, rng, class_assignment=class_assignment)
    ordered = order_candidates_on_validation(
        rules, val.scores, val.labels, n_cal_expected=n_cal, target=target, delta=delta
    )
    if not ordered:
        return None
    sp = sample_pairs(
        cal_model,
        n_cal,
        rng,
        design=design,
        n_identities=n_identities,
        class_assignment=class_assignment,
    )
    cal = LTTCalibrator(
        ordered,
        target_fm=target,
        delta=delta,
        estimand=estimand,
        require_valid=False,
    )
    return cal.fit(sp.scores, sp.labels, pair_set=sim_pair_set(sp)).rule


def _risk_summary(
    risks: list[float], covs: list[float], n_infeasible: int, target: float, reps: int
) -> dict[str, Any]:
    viol = [r > target for r in risks]
    rate = sum(viol) / reps  # infeasible runs abstain on everything: never a violation
    return {
        "violation_rate": rate,
        "violation_rate_mc_se": _mc_se(rate, reps),
        "infeasible_rate": n_infeasible / reps,
        "mean_true_risk_selected": float(np.mean(risks)) if risks else None,
        "max_true_risk_selected": float(np.max(risks)) if risks else None,
        "mean_violation_ratio": float(np.mean(np.array(risks) / target)) if risks else None,
        "mean_true_coverage": float(np.mean(covs)) if covs else 0.0,
    }


def _run_ltt_reps(
    reps: int,
    seed: int,
    tag: str,
    fn: Callable[[np.random.Generator], ThresholdRule | None],
    truth_of: Callable[[ThresholdRule], tuple[float, float]],
    target: float,
) -> dict[str, Any]:
    risks, covs, infeasible = [], [], 0
    for rep in range(reps):
        rule = fn(make_rng(seed, tag, rep))
        if rule is None:
            infeasible += 1
            covs.append(0.0)
            continue
        risk, cov = truth_of(rule)
        risks.append(risk)
        covs.append(cov)
    return _risk_summary(risks, covs, infeasible, target, reps)


# --------------------------------------------------------------------------- E2


def independence_vs_correlation(scale: Scale = "full", seed: int = 202) -> dict[str, Any]:
    """LTT violation rate with identity-disjoint vs identity-reusing calibration sets."""
    reps = 400 if scale == "full" else 40
    target, delta, n_cal = 0.05, 0.1, 600
    model = BASE.with_(sigma_identity_genuine=1.0, sigma_identity_impostor=0.6)
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    tf = _truth_fn(_truth_table(model, rules))
    configs: list[tuple[str, Literal["independent", "clustered"], int | None]] = [
        ("independent", "independent", None),
        ("clustered_30_identities", "clustered", 30),
        ("clustered_120_identities", "clustered", 120),
    ]
    out = {}
    for label, design, n_ids in configs:
        out[label] = _run_ltt_reps(
            reps,
            seed,
            label,
            partial(
                _ltt_once,
                model,
                n_cal,
                rules=rules,
                target=target,
                delta=delta,
                design=design,
                n_identities=n_ids,
            ),
            tf,
            target,
        )
    params = {
        "model": asdict(model),
        "target": target,
        "delta": delta,
        "n_cal": n_cal,
        "reps": reps,
        "estimand": "class_conditional",
        "n_rules": len(rules),
    }
    return _wrap("independence_vs_correlation", scale, seed, params, out)


# --------------------------------------------------------------------------- E3


def calibration_sample_size(scale: Scale = "full", seed: int = 303) -> dict[str, Any]:
    reps = 200 if scale == "full" else 20
    sizes = [100, 300, 1000, 3000] if scale == "full" else [100, 300]
    target, delta = 0.05, 0.05
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    truth = _truth_table(BASE, rules)
    oracle = max(
        (truth[r].coverage(0.5) for r in rules if truth[r].class_conditional_risk <= target),
        default=0.0,
    )
    out: dict[str, Any] = {"oracle_max_coverage_at_target": oracle}
    for n in sizes:
        out[f"n_cal={n}"] = _run_ltt_reps(
            reps,
            seed,
            f"n{n}",
            partial(_ltt_once, BASE, n, rules=rules, target=target, delta=delta),
            _truth_fn(truth),
            target,
        )
    params = {"model": asdict(BASE), "target": target, "delta": delta, "reps": reps, "sizes": sizes}
    return _wrap("calibration_sample_size", scale, seed, params, out)


# --------------------------------------------------------------------------- E4

SHIFTED = BASE.with_(covis_a=1.5, covis_b=2.5, genuine_slope=5.0, noise_slope=1.0)


def interpolate(a: ScoreModel, b: ScoreModel, w: float) -> ScoreModel:
    """Parameter-wise linear interpolation (w=0 -> a, w=1 -> b)."""
    da, db = asdict(a), asdict(b)
    return ScoreModel(**{k: (1 - w) * da[k] + w * db[k] for k in da})


def guarantee_under_shift(scale: Scale = "full", seed: int = 404) -> dict[str, Any]:
    """Calibrate on the seen distribution; evaluate exact risk under increasing synthetic shift.

    Severity 0 = calibration distribution; severity 1 = SHIFTED (lower co-visibility, steeper
    degradation). The magnitudes depend entirely on this arbitrary synthetic shift; only the
    qualitative pattern (guarantee holds at 0, fails progressively with shift) is meaningful.
    """
    reps = 200 if scale == "full" else 20
    target, delta, n_cal = 0.05, 0.05, 1000
    severities = [0.0, 0.1, 0.25, 0.5, 1.0]
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    tables = {w: _truth_table(interpolate(BASE, SHIFTED, w), rules) for w in severities}
    out = {}
    for w, table in tables.items():
        out[f"severity={w}"] = _run_ltt_reps(
            reps,
            seed,
            "e4",  # identical calibration draws for every severity
            partial(_ltt_once, BASE, n_cal, rules=rules, target=target, delta=delta),
            _truth_fn(table),
            target,
        )
    params = {
        "cal_model": asdict(BASE),
        "shift_endpoint_model": asdict(SHIFTED),
        "severities": severities,
        "target": target,
        "delta": delta,
        "n_cal": n_cal,
        "reps": reps,
    }
    return _wrap("guarantee_under_shift", scale, seed, params, out)


# --------------------------------------------------------------------------- E5

STRATA_EDGES = (0.0, 0.4, 0.7, 1.0)


def _stratum(c: NDArray[np.float64]) -> NDArray[np.str_]:
    return np.asarray(np.digitize(c, STRATA_EDGES[1:-1]), dtype=np.int64).astype(np.str_)


def stratified_vs_global(scale: Scale = "full", seed: int = 505) -> dict[str, Any]:
    reps = 200 if scale == "full" else 15
    target, delta, n_cal = 0.05, 0.05, 1500
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    scenarios = {
        "covariate_shift_only": BASE.with_(covis_a=1.5, covis_b=2.5),
        "conditional_shift": SHIFTED,
        "no_shift": BASE,
    }
    bins = list(itertools.pairwise(STRATA_EDGES))
    global_tables = {name: _truth_table(m, rules) for name, m in scenarios.items()}
    bin_tables = {
        name: [{r: rule_truth(m, r, c_lo=lo, c_hi=hi) for r in rules} for lo, hi in bins]
        for name, m in scenarios.items()
    }
    acc: dict[str, dict[str, list[tuple[float, float]]]] = {
        s: {"global": [], "stratified": []} for s in scenarios
    }
    n_strata = len(bins)
    for rep in range(reps):
        rng = make_rng(seed, "e5", rep)
        val = sample_pairs(BASE, n_cal, rng)
        sp = sample_pairs(BASE, n_cal, rng)
        ps = sim_pair_set(sp)
        g_order = order_candidates_on_validation(
            rules, val.scores, val.labels, n_cal_expected=n_cal, target=target, delta=delta
        )
        g_rule = (
            LTTCalibrator(g_order, target_fm=target, delta=delta)
            .fit(sp.scores, sp.labels, pair_set=ps)
            .rule
            if g_order
            else None
        )
        v_strata = _stratum(val.covis)
        by_stratum = {}
        for b in range(n_strata):
            m = v_strata == str(b)
            if m.sum() < 20 or len(set(val.labels[m].tolist())) < 2:
                continue
            by_stratum[str(b)] = order_candidates_on_validation(
                rules,
                val.scores[m],
                val.labels[m],
                n_cal_expected=int(m.sum()),
                target=target,
                delta=delta / n_strata,
            )
        strat = fit_stratified(
            LTTCalibrator,
            rules,
            sp.scores,
            sp.labels,
            list(_stratum(sp.covis)),
            ps,
            target_fm=target,
            delta=delta,
            candidates_by_stratum=by_stratum,
        )
        for name in scenarios:
            if g_rule is None:
                acc[name]["global"].append((0.0, 0.0))
            else:
                tr = global_tables[name][g_rule]
                acc[name]["global"].append((tr.class_conditional_risk, tr.coverage(0.5)))
            parts = []
            for b in range(len(bins)):
                res = strat.per_stratum.get(str(b))
                if res is not None and res.rule is not None:
                    parts.append(bin_tables[name][b][res.rule])
            if parts:
                tr = combine(parts)
                acc[name]["stratified"].append((tr.class_conditional_risk, tr.coverage(0.5)))
            else:
                acc[name]["stratified"].append((0.0, 0.0))
    out: dict[str, Any] = {}
    for name in scenarios:
        out[name] = {}
        for arm, vals in acc[name].items():
            risks = np.array([v[0] for v in vals])
            rate = float((risks > target).mean())
            out[name][arm] = {
                "violation_rate": rate,
                "violation_rate_mc_se": _mc_se(rate, reps),
                "mean_true_risk": float(risks.mean()),
                "mean_true_coverage": float(np.mean([v[1] for v in vals])),
            }
    params = {
        "cal_model": asdict(BASE),
        "scenarios": {k: asdict(v) for k, v in scenarios.items()},
        "strata_edges": STRATA_EDGES,
        "target": target,
        "delta": delta,
        "n_cal": n_cal,
        "reps": reps,
    }
    return _wrap("stratified_vs_global", scale, seed, params, out)


# --------------------------------------------------------------------------- E6


def clean_tolerance_power(scale: Scale = "full", seed: int = 606) -> dict[str, Any]:
    """Power to show non-inferiority (margin 0.5 pp TAR@FAR=1%) when the true difference is 0."""
    reps, n_boot = (100, 300) if scale == "full" else (8, 80)
    sizes = [200, 1000, 3333] if scale == "full" else [200]
    margin, g_per_id, imp_per_id = 0.005, 4, 10
    clean = BASE.with_(covis_a=50.0, covis_b=1.0)  # nearly full visibility
    t = population_threshold_at_far(clean, 0.01)
    out: dict[str, Any] = {"threshold": t}
    for n_ids in sizes:
        demonstrated, widths = 0, []
        for rep in range(reps):
            rng = make_rng(seed, "e6", n_ids, rep)
            n_g, n_i = n_ids * g_per_id, n_ids * imp_per_id
            sp = sample_pairs(
                clean.with_(sigma_noise=0.9),
                n_g + n_i,
                rng,
                genuine_fraction=n_g / (n_g + n_i),
                design="clustered",
                n_identities=n_ids,
                genuine_per_identity=g_per_id,
            )
            e_sd = math.sqrt(1.0 - 0.81)  # restores total noise sd to 1.0 for both methods
            s_a = sp.scores + rng.normal(0, e_sd, sp.scores.size)
            s_b = sp.scores + rng.normal(0, e_sd, sp.scores.size)
            g = sp.labels == 1
            diff_num = (g & (s_b >= t)).astype(float) - (g & (s_a >= t)).astype(float)
            den = g.astype(float)

            def stat(
                w: NDArray[np.float64],
                num: NDArray[np.float64] = diff_num,
                d: NDArray[np.float64] = den,
            ) -> NDArray[np.float64]:
                return np.asarray((w @ num) / (w @ d), dtype=np.float64)

            res = bootstrap_ci(
                stat,
                n_pairs=sp.labels.size,
                unit="identity",
                rng=make_rng(seed, "e6b", n_ids, rep),
                n_boot=n_boot,
                id_a=sp.id_a,
                id_b=sp.id_b,
                level=0.90,
            )  # one-sided 95% lower bound
            demonstrated += int(res.ci_low > -margin)
            widths.append(res.ci_high - res.ci_low)
        power = demonstrated / reps
        out[f"n_identities={n_ids}"] = {
            "power_noninferiority": power,
            "power_mc_se": _mc_se(power, reps),
            "mean_90pct_ci_width_tar_diff": float(np.mean(widths)),
            "n_genuine_pairs": n_ids * g_per_id,
        }
    params = {
        "model": asdict(clean),
        "margin": margin,
        "reps": reps,
        "n_boot": n_boot,
        "genuine_per_identity": g_per_id,
        "impostor_per_identity": imp_per_id,
        "true_difference": 0.0,
        "test": "one-sided 95% (lower bound of 90% two-sided CI)",
    }
    return _wrap("clean_tolerance_power", scale, seed, params, out)


# --------------------------------------------------------------------------- E7


def prevalence_sensitivity(scale: Scale = "full", seed: int = 707) -> dict[str, Any]:
    reps = 200 if scale == "full" else 20
    target, delta, n_cal = 0.05, 0.05, 1000
    prevalences = [0.05, 0.2, 0.5, 0.8, 0.95]
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    truth = _truth_table(BASE, rules)
    out: dict[str, Any] = {}
    for estimand in ("mixture_conditional", "class_conditional"):
        chosen = [
            _ltt_once(
                BASE,
                n_cal,
                make_rng(seed, "e7", estimand, rep),
                rules,
                target=target,
                delta=delta,
                estimand=estimand,
                class_assignment="bernoulli",
            )
            for rep in range(reps)
        ]
        out[estimand] = {}
        for pi in prevalences:
            risks = [truth[r].mixture_risk(pi) if r is not None else 0.0 for r in chosen]
            rate = float(np.mean(np.array(risks) > target))
            out[estimand][f"test_prevalence={pi}"] = {
                "violation_rate": rate,
                "violation_rate_mc_se": _mc_se(rate, reps),
                "mean_true_accepted_error": float(np.mean(risks)),
                "mean_true_coverage": float(
                    np.mean([truth[r].coverage(pi) if r is not None else 0.0 for r in chosen])
                ),
            }
    params = {
        "model": asdict(BASE),
        "target": target,
        "delta": delta,
        "n_cal": n_cal,
        "reps": reps,
        "calibration_prevalence": 0.5,
        "class_assignment": "bernoulli",
    }
    return _wrap("prevalence_sensitivity", scale, seed, params, out)


# --------------------------------------------------------------------------- E8


def cost_sensitivity(scale: Scale = "full", seed: int = 808) -> dict[str, Any]:
    """Which certified rule minimises cost-weighted accepted risk as the FM:FNM cost ratio
    varies."""
    n_cal = 20000 if scale == "full" else 4000
    target, delta = 0.05, 0.05
    offsets = [round(0.25 * k, 2) for k in range(13)]  # 0 .. 3
    rules = [ThresholdRule(CENTER - a, CENTER + b) for a in offsets for b in offsets]
    truth = _truth_table(BASE, rules)
    rng = make_rng(seed, "e8")
    val = sample_pairs(BASE, n_cal, rng)
    pruned = order_candidates_on_validation(
        rules, val.scores, val.labels, n_cal_expected=n_cal, target=target, delta=delta / len(rules)
    )
    sp = sample_pairs(BASE, n_cal, rng)
    if not pruned:
        return _wrap("cost_sensitivity", scale, seed, {"n_cal": n_cal}, {"n_certified": 0})
    res = LTTCalibrator(pruned, target_fm=target, delta=delta, fwer="bonferroni").fit(
        sp.scores, sp.labels, pair_set=sim_pair_set(sp)
    )
    certified = [res.candidates[j].rule for j in res.rejected]
    out: dict[str, Any] = {"n_candidates": len(rules), "n_certified": len(certified)}
    if certified:
        floor = 0.8 * max(truth[r].coverage(0.5) for r in certified)
        certified = [r for r in certified if truth[r].coverage(0.5) >= floor]
        out["coverage_floor"] = floor
    for ratio in (1, 10, 100):

        def cost(r: ThresholdRule, k: float = ratio) -> float:
            tr = truth[r]
            accepted = tr.coverage(0.5)
            return (k * 0.5 * tr.err_i + 0.5 * tr.err_g) / accepted if accepted else math.inf

        if not certified:
            out[f"cost_ratio={ratio}"] = None
            continue
        best = min(certified, key=lambda r: (cost(r), -truth[r].coverage(0.5)))
        tr = truth[best]
        out[f"cost_ratio={ratio}"] = {
            "tau_low": best.tau_low,
            "tau_high": best.tau_high,
            "true_coverage": tr.coverage(0.5),
            "true_sfmr": tr.sfmr,
            "true_sfnmr": tr.sfnmr,
            "true_cost_weighted_accepted_risk": cost(best),
        }
    params = {
        "model": asdict(BASE),
        "n_cal": n_cal,
        "target": target,
        "delta": delta,
        "fwer": "bonferroni",
        "selection": "min true cost-weighted accepted risk among LTT-certified rules whose "
        "true coverage >= 80% of the best certified coverage",
        "prevalence": 0.5,
    }
    return _wrap("cost_sensitivity", scale, seed, params, out)


# --------------------------------------------------------------------------- binomial sanity


def binomial_bound_sanity(scale: Scale = "full", seed: int = 909) -> dict[str, Any]:
    """Empirical coverage of the one-sided Clopper-Pearson bound used by the feasibility report."""
    reps = 4000 if scale == "full" else 400
    rng = make_rng(seed, "cp")
    out = {}
    for n, p in ((59, 0.02), (300, 0.03), (1000, 0.045)):
        k = rng.binomial(n, p, reps)
        hit = np.mean([clopper_pearson_upper(int(x), n, 0.95) >= p for x in k])
        out[f"n={n},p={p}"] = {"coverage": float(hit), "nominal_min": 0.95}
    return _wrap("binomial_bound_sanity", scale, seed, {"reps": reps}, out)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {
    "bootstrap_coverage": bootstrap_coverage,
    "independence_vs_correlation": independence_vs_correlation,
    "calibration_sample_size": calibration_sample_size,
    "guarantee_under_shift": guarantee_under_shift,
    "stratified_vs_global": stratified_vs_global,
    "clean_tolerance_power": clean_tolerance_power,
    "prevalence_sensitivity": prevalence_sensitivity,
    "cost_sensitivity": cost_sensitivity,
    "binomial_bound_sanity": binomial_bound_sanity,
}
