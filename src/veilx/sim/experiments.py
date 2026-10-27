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
from veilx.stats.jackknife import jackknife_interval

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


# --------------------------------------------------------------------------- E10 interval methods


def _inflated(res_lo: float, res_hi: float, est: float, g: int) -> tuple[float, float]:
    f = math.sqrt(g / (g - 1))
    return est + f * (res_lo - est), est + f * (res_hi - est)


def interval_method_comparison(scale: Scale = "full", seed: int = 1010) -> dict[str, Any]:
    """Coverage of five 95% interval methods for FNMR (one-identity pairs) and FMR (two-identity
    impostor pairs) under strong identity effects, versus the number of test identities.

    Methods: pair percentile bootstrap, identity percentile bootstrap (product weights for
    impostors), identity bootstrap with sqrt(G/(G-1)) inflation, delete-one-identity jackknife
    (t), and the same jackknife on the logit scale. Also reports the impostor design effect
    (true variance of FMR-hat / naive binomial variance) quantifying two-identity dependence.
    """
    reps, n_boot = (1000, 500) if scale == "full" else (60, 150)
    sizes = [50, 150, 500] if scale == "full" else [50, 150]
    model = BASE.with_(sigma_identity_genuine=0.8, sigma_identity_impostor=0.5)
    g_per_id, imp_per_id = 6, 20
    conditions = [("fnmr", 0.01), ("fmr", 0.01), ("fmr", 0.001)]
    out: dict[str, Any] = {}
    for g_ids in sizes:
        for stat_name, far in conditions:
            if far == 0.001 and g_ids < 500:
                continue  # expected false matches would be < 10: not estimable (protocol rule)
            t = population_threshold_at_far(model, far)
            tr = rule_truth(model, ThresholdRule(t, t))
            truth = tr.err_g / tr.acc_g if stat_name == "fnmr" else tr.err_i / tr.acc_i
            hits: dict[str, int] = dict.fromkeys(
                (
                    "pair_percentile",
                    "identity_percentile",
                    "identity_percentile_inflated",
                    "jackknife_t",
                    "jackknife_logit",
                ),
                0,
            )
            widths: dict[str, list[float]] = {k: [] for k in hits}
            ests: list[float] = []
            var_est: dict[str, list[float]] = {
                k: [] for k in ("jackknife_t", "identity_percentile")
            }
            for rep in range(reps):
                rng = make_rng(seed, "e10", g_ids, stat_name, str(far), rep)
                n_g, n_i = g_ids * g_per_id, g_ids * imp_per_id
                sp = sample_pairs(
                    model,
                    n_g + n_i,
                    rng,
                    genuine_fraction=n_g / (n_g + n_i),
                    design="clustered",
                    n_identities=g_ids,
                    genuine_per_identity=g_per_id,
                )
                g, acc = sp.labels == 1, sp.scores >= t
                num, den = (g & ~acc, g) if stat_name == "fnmr" else (~g & acc, ~g)
                stat = weighted_rate(num, den)
                ivs: dict[str, tuple[float, float]] = {}
                units: tuple[Literal["pair", "identity"], ...] = ("pair", "identity")
                for u in units:
                    unit = u
                    r = bootstrap_ci(
                        stat,
                        n_pairs=sp.labels.size,
                        unit=u,
                        n_boot=n_boot,
                        rng=make_rng(seed, "e10b", g_ids, stat_name, str(far), rep, unit),
                        id_a=sp.id_a,
                        id_b=sp.id_b,
                    )
                    ivs[f"{unit}_percentile"] = (r.ci_low, r.ci_high)
                    if unit == "identity":
                        ivs["identity_percentile_inflated"] = _inflated(
                            r.ci_low, r.ci_high, r.estimate, g_ids
                        )
                        var_est["identity_percentile"].append(
                            ((r.ci_high - r.ci_low) / (2 * 1.96)) ** 2
                        )
                for scale_name, key in (("linear", "jackknife_t"), ("logit", "jackknife_logit")):
                    j = jackknife_interval(num, den, sp.id_a, sp.id_b, scale=scale_name)
                    ivs[key] = (j.ci_low, j.ci_high)
                    if key == "jackknife_t":
                        var_est["jackknife_t"].append(j.se**2)
                        ests.append(j.estimate)
                for k, (lo, hi) in ivs.items():
                    hits[k] += int(lo <= truth <= hi)
                    widths[k].append(hi - lo)
            true_var = float(np.var(ests, ddof=1))
            n_den = (g_ids * g_per_id) if stat_name == "fnmr" else (g_ids * imp_per_id)
            res: dict[str, Any] = {
                "truth": truth,
                "methods": {
                    k: {
                        "coverage": hits[k] / reps,
                        "mc_se": _mc_se(hits[k] / reps, reps),
                        "mean_width": float(np.mean(widths[k])),
                    }
                    for k in hits
                },
                "true_sampling_variance": true_var,
                "naive_binomial_variance": truth * (1 - truth) / n_den,
                "design_effect": true_var / (truth * (1 - truth) / n_den),
                "variance_ratio_estimate_over_true": {
                    k: float(np.mean(v)) / true_var for k, v in var_est.items()
                },
            }
            out[f"G={g_ids},{stat_name}@FAR={far}"] = res
    params = {
        "model": asdict(model),
        "reps": reps,
        "n_boot": n_boot,
        "genuine_per_identity": g_per_id,
        "impostor_pairs_per_identity": imp_per_id,
        "nominal": 0.95,
        "acceptance_rule": "coverage >= 0.94 in every condition (MC se ~0.007 at 1000 reps)",
    }
    return _wrap("interval_method_comparison", scale, seed, params, out)


def _paired_scores(
    rng: np.random.Generator, g_ids: int, g_per_id: int, imp_per_id: int, p: dict[str, float]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Two systems scored on the SAME identity-clustered pairs.

    System A: genuine s = mu_g + u_i + e, impostor s = mu_i + v_i + v_k + e. System B adds a
    shift, a B-specific identity effect and pair noise: s_B = s_A + shift + b_(ids) + eta. The
    two systems are therefore strongly but not perfectly correlated, as two backbones (or a
    backbone with and without a module) scored on one test set would be.
    """
    u = rng.normal(0, p["sig_u"], g_ids)
    v = rng.normal(0, p["sig_v"], g_ids)
    b = rng.normal(0, p["sig_b"], g_ids)
    gid = np.repeat(np.arange(g_ids), g_per_id)
    ia = np.repeat(np.arange(g_ids), imp_per_id)
    ib = (ia + rng.integers(1, g_ids, ia.size)) % g_ids
    sg = p["mu_g"] + u[gid] + rng.normal(0, 1, gid.size)
    si = p["mu_i"] + v[ia] + v[ib] + rng.normal(0, 1, ia.size)
    s_a = np.concatenate([sg, si])
    id_a = np.concatenate([gid, ia])
    id_b = np.concatenate([gid, ib])
    bb = np.where(id_a == id_b, b[id_a], 0.5 * (b[id_a] + b[id_b]))
    s_b = s_a + p["shift"] + bb + rng.normal(0, p["sig_eta"], s_a.size)
    lab = np.concatenate([np.ones(gid.size, np.int64), np.zeros(ia.size, np.int64)])
    return s_a, s_b, lab, id_a, id_b


def paired_difference_coverage(scale: Scale = "full", seed: int = 1111) -> dict[str, Any]:
    """SIM-011: coverage of 95 % intervals for PAIRED differences of error rates between two
    correlated systems on one identity-clustered test set (the clean non-inferiority margin).

    Statistics: delta FNMR (genuine pairs) and delta FMR (impostor pairs) at fixed thresholds.
    Methods: delete-one-identity jackknife on the per-pair difference (t, linear scale) - the
    method proposed for the margin test - and a naive pair-level normal interval (reference).
    Truth: Monte Carlo with 2 M pairs per class.
    """
    reps = 1000 if scale == "full" else 80
    sizes = [500, 2000] if scale == "full" else [300]
    params = {
        "mu_g": 3.0,
        "mu_i": 0.0,
        "sig_u": 0.8,
        "sig_v": 0.5,
        "sig_b": 0.3,
        "sig_eta": 0.4,
        "shift": -0.1,
    }
    g_per_id, imp_per_id = 3, 3
    t_fnmr = params["mu_g"] - 2.0  # FNMR around 3 %
    t_fmr = 2.6  # FMR around 1 %
    big = make_rng(seed, "truth")
    sa, sb, lab, _, _ = _paired_scores(big, 400_000, 5, 5, params)
    g = lab == 1
    truth = {
        "fnmr": float(np.mean(sb[g] < t_fnmr) - np.mean(sa[g] < t_fnmr)),
        "fmr": float(np.mean(sb[~g] >= t_fmr) - np.mean(sa[~g] >= t_fmr)),
        "fnmr_a": float(np.mean(sa[g] < t_fnmr)),
        "fmr_a": float(np.mean(sa[~g] >= t_fmr)),
    }
    out: dict[str, Any] = {"truth": truth}
    for g_ids in sizes:
        hits = {(s, m): 0 for s in ("fnmr", "fmr") for m in ("jackknife_t", "naive_pair")}
        widths: dict[tuple[str, str], list[float]] = {k: [] for k in hits}
        for rep in range(reps):
            rng = make_rng(seed, "rep", g_ids, rep)
            sa, sb, lab, id_a, id_b = _paired_scores(rng, g_ids, g_per_id, imp_per_id, params)
            gm = lab == 1
            for stat, m, ea, eb in (
                ("fnmr", gm, sa < t_fnmr, sb < t_fnmr),
                ("fmr", ~gm, sa >= t_fmr, sb >= t_fmr),
            ):
                d = (eb.astype(np.float64) - ea.astype(np.float64))[m]
                j = jackknife_interval(d, np.ones(d.size), id_a[m], id_b[m], scale="linear")
                se = float(np.std(d, ddof=1) / math.sqrt(d.size))
                for meth, lo, hi in (
                    ("jackknife_t", j.ci_low, j.ci_high),
                    ("naive_pair", d.mean() - 1.96 * se, d.mean() + 1.96 * se),
                ):
                    hits[(stat, meth)] += int(lo <= truth[stat] <= hi)
                    widths[(stat, meth)].append(hi - lo)
        out[f"identities_{g_ids}"] = {
            f"{s}:{m}": {
                "coverage": hits[(s, m)] / reps,
                "mc_se": _mc_se(hits[(s, m)] / reps, reps),
                "mean_width_pp": 100 * float(np.mean(widths[(s, m)])),
            }
            for s, m in hits
        }
    return _wrap(
        "paired_difference_coverage",
        scale,
        seed,
        params
        | {
            "g_per_id": g_per_id,
            "imp_per_id": imp_per_id,
            "reps": reps,
            "t_fnmr": t_fnmr,
            "t_fmr": t_fmr,
        },
        out,
    )


def adaptive_ordering_validity(scale: Scale = "full", seed: int = 1212) -> dict[str, Any]:
    """SIM-012: validity of the procedure actually used in Phase 2 - fit LTT with the
    pre-registered order (v1) and the p-value order (v2) on the SAME calibration set, then report
    (a) v2 alone, (b) v1 if feasible else v2, (c) whichever certifies more coverage. The
    violation rate (true class-conditional risk of the reported rule > target) is compared with
    delta and with the union bound 2 * delta.
    """
    from veilx.selective.ltt import order_candidates_by_validation_pvalue

    reps = 1000 if scale == "full" else 120
    model, target, delta = BASE, 0.05, 0.1
    rules = symmetric_rules(2.0, [round(0.1 * k, 1) for k in range(26)])
    truth = {r: rule_truth(model, r).class_conditional_risk for r in rules}
    out: dict[str, Any] = {}
    for n_cal in (300, 1000):
        viol = dict.fromkeys(("v1", "v2", "v1_else_v2", "max_coverage"), 0)
        feas = dict.fromkeys(viol, 0)
        for rep in range(reps):
            rng = make_rng(seed, "e12", n_cal, rep)
            val, cal = sample_pairs(model, 2000, rng), sample_pairs(model, n_cal, rng)
            o1 = order_candidates_on_validation(
                rules, val.scores, val.labels, n_cal_expected=n_cal, target=target, delta=delta
            )
            o2 = order_candidates_by_validation_pvalue(
                rules,
                val.scores,
                val.labels,
                n_cal_expected=n_cal,
                target_fm=target,
                target_fnm=target,
                delta=delta,
            )
            res = {}
            for name, order in (("v1", o1), ("v2", o2)):
                if order:
                    r = LTTCalibrator(order, target_fm=target, delta=delta).fit(
                        cal.scores, cal.labels, pair_set=sim_pair_set(cal)
                    )
                    res[name] = (
                        r.rule,
                        max((c.coverage for c in r.candidates if c.rule == r.rule), default=0.0),
                    )
                else:
                    res[name] = (None, 0.0)
            picks = {
                "v1": res["v1"][0],
                "v2": res["v2"][0],
                "v1_else_v2": res["v1"][0] if res["v1"][0] is not None else res["v2"][0],
                "max_coverage": max((res["v1"], res["v2"]), key=lambda t: t[1])[0],
            }
            for k, rule in picks.items():
                if rule is not None:
                    feas[k] += 1
                    viol[k] += int(truth[rule] > target)
        out[f"n_cal_{n_cal}"] = {
            k: {
                "violation_rate": viol[k] / reps,
                "mc_se": _mc_se(viol[k] / reps, reps),
                "feasible_rate": feas[k] / reps,
            }
            for k in viol
        }
    return _wrap(
        "adaptive_ordering_validity",
        scale,
        seed,
        {"target": target, "delta": delta, "union_bound": 2 * delta, "reps": reps},
        out,
    )


def _predicted(
    c: NDArray[np.float64], rng: np.random.Generator, sigma: float, bias: float
) -> NDArray[np.float64]:
    return np.clip(c + bias + rng.normal(0.0, sigma, c.size), 0.0, 1.0)


def _empirical_cc_risk(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    strata: NDArray[np.str_],
    rules: dict[str, ThresholdRule | None],
) -> tuple[float, float, float]:
    """(pooled class-conditional risk, coverage, worst per-stratum class-conditional risk) of
    per-stratum rules on a large evaluation sample."""
    d = np.full(scores.shape, -1, dtype=np.int64)  # -1 abstain, 1 match, 0 non-match
    for name, rule in rules.items():
        m = strata == name
        if rule is not None and m.any():
            d[m] = rule.decide(scores[m])
    acc = d != -1
    g, i = labels == 1, labels == 0
    fnm = ((d == 0) & g).sum() / max((acc & g).sum(), 1)
    fm = ((d == 1) & i).sum() / max((acc & i).sum(), 1)
    worst = 0.0
    for name in rules:
        m = strata == name
        a_g, a_i = (acc & g & m).sum(), (acc & i & m).sum()
        if a_g:
            worst = max(worst, ((d == 0) & g & m).sum() / a_g)
        if a_i:
            worst = max(worst, ((d == 1) & i & m).sum() / a_i)
    return float(max(fm, fnm)), float(acc.mean()), float(worst)


def noisy_strata_validity(scale: Scale = "full", seed: int = 1313) -> dict[str, Any]:
    """SIM-013: stratified LTT when strata come from a PREDICTED (noisy) co-visibility.

    Calibration and the ID evaluation use predicted c_hat = clip(c + N(0, 0.1)). The guarantee
    is conditional on the *predicted* stratum, so it should hold at delta when the predictor's
    error distribution is unchanged. Scenarios then change the predictor (noisier; biased up, as
    when an unseen occluder is mistaken for visible face; biased down) or the score model
    (conditional shift) at evaluation time. The violation rate is the fraction of replicates
    whose reported rules have true pooled class-conditional risk above the target, measured on a
    200k-pair evaluation sample per scenario (Monte Carlo truth).
    """
    reps = 300 if scale == "full" else 25
    target, delta, n_cal, n_eval = 0.05, 0.05, 1500, 200_000 if scale == "full" else 40_000
    sigma_id = 0.1
    rules = symmetric_rules(CENTER, HALF_WIDTHS)
    scenarios: dict[str, tuple[ScoreModel, float, float]] = {
        "id_predictor": (BASE, sigma_id, 0.0),
        "oracle_strata_reference": (BASE, 0.0, 0.0),
        "noisier_predictor": (BASE, 0.25, 0.0),
        "overestimating_predictor": (BASE, sigma_id, 0.2),
        "underestimating_predictor": (BASE, sigma_id, -0.2),
        "conditional_shift_id_predictor": (SHIFTED, sigma_id, 0.0),
    }
    evals = {}
    for k, (name, (m, sg, bias)) in enumerate(scenarios.items()):
        ev = sample_pairs(m, n_eval, make_rng(seed, "e13-eval", k))
        evals[name] = (ev, _stratum(_predicted(ev.covis, make_rng(seed, "e13-pred", k), sg, bias)))
    n_strata = len(STRATA_EDGES) - 1
    res: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        s: {"stratified_predicted": []} for s in scenarios
    }
    for rep in range(reps):
        rng = make_rng(seed, "e13", rep)
        for cal_kind in ("predicted", "oracle"):
            sg = sigma_id if cal_kind == "predicted" else 0.0
            val = sample_pairs(BASE, n_cal, rng)
            cal = sample_pairs(BASE, n_cal, rng)
            v_st = _stratum(_predicted(val.covis, rng, sg, 0.0))
            c_st = _stratum(_predicted(cal.covis, rng, sg, 0.0))
            by = {}
            for b in range(n_strata):
                msk = v_st == str(b)
                if msk.sum() < 20 or len(set(val.labels[msk].tolist())) < 2:
                    continue
                by[str(b)] = order_candidates_on_validation(
                    rules,
                    val.scores[msk],
                    val.labels[msk],
                    n_cal_expected=int(msk.sum()),
                    target=target,
                    delta=delta / n_strata,
                )
            fit = fit_stratified(
                LTTCalibrator,
                rules,
                cal.scores,
                cal.labels,
                list(c_st),
                sim_pair_set(cal),
                target_fm=target,
                delta=delta,
                candidates_by_stratum=by,
            )
            chosen = {
                str(b): (r.rule if (r := fit.per_stratum.get(str(b))) is not None else None)
                for b in range(n_strata)
            }
            for name in scenarios:
                if (cal_kind == "oracle") != (name == "oracle_strata_reference"):
                    continue
                ev, st = evals[name]
                res[name]["stratified_predicted"].append(
                    _empirical_cc_risk(ev.scores, ev.labels, st, chosen)
                )
    out: dict[str, Any] = {}
    for name, (m, sg, bias) in scenarios.items():
        vals = res[name]["stratified_predicted"]
        risks = np.array([v[0] for v in vals])
        worst = np.array([v[2] for v in vals])
        rate = float((risks > target).mean())
        srate = float((worst > target).mean())
        out[name] = {
            "predictor_sigma": sg,
            "predictor_bias": bias,
            "score_model": "SHIFTED" if m == SHIFTED else "BASE",
            "violation_rate": rate,
            "violation_rate_mc_se": _mc_se(rate, len(vals)),
            "mean_true_risk": float(risks.mean()),
            "mean_true_coverage": float(np.mean([v[1] for v in vals])),
            "per_stratum_violation_rate": srate,
            "per_stratum_violation_rate_mc_se": _mc_se(srate, len(vals)),
            "mean_worst_stratum_risk": float(worst.mean()),
        }
    params = {
        "target": target,
        "delta": delta,
        "n_cal": n_cal,
        "n_eval": n_eval,
        "reps": reps,
        "strata_edges": STRATA_EDGES,
        "calibration_predictor_sigma": sigma_id,
        "note": "Monte Carlo truth on one evaluation sample per scenario (its MC error ~1e-3).",
    }
    return _wrap("noisy_strata_validity", scale, seed, params, out)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {
    "noisy_strata_validity": noisy_strata_validity,
    "paired_difference_coverage": paired_difference_coverage,
    "adaptive_ordering_validity": adaptive_ordering_validity,
    "interval_method_comparison": interval_method_comparison,
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
