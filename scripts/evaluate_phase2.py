"""Phase 2 evaluation of the frozen backbones on DigiFace (clean + synthetic occlusion).

Two stages, so that nothing is ever tuned on test data:

``develop``  (validation + calibration splits only)
    * clean thresholds: EER and best-accuracy on val; FAR 1 % / 0.1 % on val impostors;
      FAR 1 % on the independent calibration impostors (deployment threshold);
    * per held-out fold: LTT candidate grid (score quantiles of the fold's val plan), candidate
      order fixed on val (``order_candidates_on_validation``), oracle-visibility strata edge =
      median co-visibility of occluded val pairs;
    * LTT fits on the calibration split for three arms x two target settings:
        - ``global``         occluded cal plan (seen families), one rule;
        - ``stratified_vis`` per co-visibility stratum (clean / high / low), delta/3 each;
        - ``clean_cal``      clean calibration pairs only (no occlusion seen);
    * everything is written to a frozen config whose SHA-256 is recorded.

``final --final``  (test split, once)
    * refuses to run without ``--final``; appends a run-log entry (protocol ``test_access``);
    * clean baselines, LOFO degradation (Test-ID vs Test-OOD), realised selective risk of the
      frozen LTT rules, stress (exploratory), C1 exploratory analysis, suspicious-performance
      audit. Aggregated numbers only are written to ``reports/experiments/``.

    .venv\\Scripts\\python.exe scripts\\evaluate_phase2.py develop
    .venv\\Scripts\\python.exe scripts\\evaluate_phase2.py final --final
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.config.schema import SplitName
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import DataLayout, resolve_data_root
from veilx.embeddings.store import read_shard
from veilx.eval.scoring import EmbeddingLookup, ScoredPairs, score_pairs
from veilx.eval.summaries import (
    rate,
    roc_points,
    score_histogram,
    selective_summary,
    verification_summary,
)
from veilx.guards import require_final_for_test
from veilx.metrics.roc import roc_auc, tar_at_far, threshold_at_far
from veilx.models.spec import BACKBONES
from veilx.pairs.builder import PairSet, pairs_digest, summarize
from veilx.provenance import collect_provenance
from veilx.selective.ltt import (
    LTTCalibrator,
    RiskControlResult,
    ThresholdRule,
    fit_stratified,
    order_candidates_by_validation_pvalue,
    order_candidates_on_validation,
)

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
CONF = float(PROTO.constants.confidence_level.value)  # type: ignore[arg-type]
DELTA = round(1.0 - CONF, 10)
ALPHA = float(PROTO.constants.accepted_decision_error_target.value)  # type: ignore[arg-type]
FARS = tuple(float(x) for x in PROTO.constants.far_operating_points.value)  # type: ignore[union-attr]
# Pre-registered target settings (docs/baseline_evaluation.md section 2).
TARGETS: dict[str, tuple[float, float]] = {
    "primary_alpha": (ALPHA, ALPHA),  # sFMR <= 5 %, sFNMR <= 5 %
    "far1_fm": (FARS[0], ALPHA),  # sFMR <= 1 %, sFNMR <= 5 %
}
N_GRID = 61
VIS_BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999999, 1.0 + 1e-9)
STRESS_NOTE = "exploratory: combined physical occlusions on both sides; not part of LOFO"

# ----------------------------------------------------------------- loading


def _layout() -> DataLayout:
    return resolve_data_root()


def _identity_map(layout: DataLayout) -> dict[str, str]:
    return {
        r.image_id: r.identity_id
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
    }


def _pairs(layout: DataLayout, tag: str) -> list[PairRecord]:
    return list(read_jsonl(layout.manifests / "pairs" / f"{tag}.jsonl", PairRecord))


def _pair_set(layout: DataLayout, tag: str, images: list[ImageRecord]) -> PairSet:
    """Re-derive the summary from the manifest and check the recorded pair digest."""
    pairs = _pairs(layout, tag)
    prov = json.loads((layout.manifests / "pairs" / f"{tag}.provenance.json").read_text("utf-8"))
    if pairs_digest(pairs) != prov["pairs_digest"]:
        raise RuntimeError(f"{tag}: pair file does not match its provenance digest")
    mode = str(prov["spec"]["mode"])
    return PairSet(tuple(pairs), summarize(pairs, images, mode), prov)


def _plan(layout: DataLayout, fold: str, role: str, tag: str) -> dict[str, dict[str, Any]]:
    path = layout.manifests / "occlusion" / f"{fold}__{role}__{tag}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {(r := json.loads(line))["pair_id"]: r for line in fh}


class Embeddings:
    """Lazy per-backbone shard access (clean shard kept, plan shards loaded on demand)."""

    def __init__(self, layout: DataLayout, backbone_id: str) -> None:
        base = layout.embeddings / backbone_id
        runs = [p for p in base.iterdir() if p.is_dir() and (p / "clean.npz").exists()]
        if len(runs) != 1:
            raise RuntimeError(f"{backbone_id}: expected exactly one extraction run, found {runs}")
        self.dir = runs[0]
        self.clean = read_shard(self.dir / "clean.npz")
        self.meta = self.clean.meta

    def lookup(self, group: str | None = None) -> EmbeddingLookup:
        shards = [self.clean]
        if group:
            shards.append(read_shard(self.dir / f"{group}.npz"))
        return EmbeddingLookup(shards)


# ----------------------------------------------------------------- helpers


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _best_accuracy_threshold(sp: ScoredPairs) -> float:
    order = np.argsort(sp.score)
    s, y = sp.score[order], sp.label[order]
    # accept >= t: correct = genuines at/above + impostors below
    gen_above = np.cumsum(y[::-1])[::-1]
    imp_below = np.concatenate([[0], np.cumsum(1 - y)[:-1]])
    k = int(np.argmax(gen_above + imp_below))
    return float(s[k])


def _grid_rules(scores: np.ndarray, extra: Sequence[float]) -> list[ThresholdRule]:
    taus = np.unique(
        np.concatenate([np.quantile(scores, np.linspace(0.005, 0.995, N_GRID)), extra])
    )
    return [ThresholdRule(float(lo), float(hi)) for k, lo in enumerate(taus) for hi in taus[k:]]


def _strata(covis: np.ndarray, edge: float) -> np.ndarray:
    return np.where(covis >= 0.999999, "clean", np.where(covis >= edge, "vis_high", "vis_low"))


def _rules_json(rules: Sequence[ThresholdRule]) -> list[list[float]]:
    return [[r.tau_low, r.tau_high] for r in rules]


def _rule(r: ThresholdRule | None) -> list[float] | None:
    return None if r is None else [r.tau_low, r.tau_high]


def _ltt_record(res: RiskControlResult) -> dict[str, Any]:
    chosen = next((c for c in res.candidates if c.rule == res.rule), None)
    return {
        "feasible": res.feasible,
        "rule": _rule(res.rule),
        "guarantee_valid": res.guarantee_valid,
        "n_candidates": len(res.candidates),
        "n_rejected": len(res.rejected),
        "cal_stats": None
        if chosen is None
        else {
            "n": chosen.n,
            "accepted": chosen.accepted,
            "accepted_genuine": chosen.accepted_genuine,
            "accepted_impostor": chosen.accepted_impostor,
            "false_matches": chosen.false_matches,
            "false_non_matches": chosen.false_non_matches,
            "p_value": chosen.p_value,
        },
    }


STRATA = ("clean", "vis_high", "vis_low")


def _fit_ordered(
    ordered: list[ThresholdRule],
    sc: ScoredPairs,
    cal_set: PairSet,
    afm: float,
    afnm: float,
    families: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not ordered:
        return {"feasible": False, "rule": None, "n_ordered": 0, "note": "no candidate survives"}
    res = LTTCalibrator(ordered, target_fm=afm, target_fnm=afnm, delta=DELTA).fit(
        sc.score, sc.label, pair_set=cal_set, calibration_families=families
    )
    return _ltt_record(res) | {"n_ordered": len(ordered)}


# ----------------------------------------------------------------- develop


def develop() -> int:
    layout = _layout()
    images = list(read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord))
    ident = {r.image_id: r.identity_id for r in images}
    cal_set = _pair_set(layout, "cal_independent", images)
    if not cal_set.summary.identity_disjoint:
        raise RuntimeError("calibration set is not identity-disjoint")
    val_pairs = _pairs(layout, "val")
    cal_pairs = list(cal_set.pairs)
    frozen: dict[str, Any] = {
        "protocol_version": PROTO.protocol_version,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "delta": DELTA,
        "targets": TARGETS,
        "n_grid": N_GRID,
        "cal_summary": cal_set.summary.__dict__,
        "backbones": {},
    }
    for bid in BACKBONES:
        emb = Embeddings(layout, bid)
        clean = emb.lookup()
        sv = score_pairs(val_pairs, ident, clean)
        sc = score_pairs(cal_pairs, ident, clean)
        eer_val = verification_summary(sv, {})
        thr = {
            "val_eer": float(eer_val["eer_threshold"]),
            "val_best_accuracy": _best_accuracy_threshold(sv),
            "val_far_1pct": threshold_at_far(sv.score[sv.label == 0], FARS[0]),
            "val_far_0.1pct": threshold_at_far(sv.score[sv.label == 0], FARS[1]),
            "cal_far_1pct": threshold_at_far(sc.score[sc.label == 0], FARS[0]),
        }
        b: dict[str, Any] = {
            "backbone_fingerprint": emb.meta["backbone_fingerprint"],
            "embedding_run": emb.dir.name,
            "thresholds": thr,
            "val_clean": {k: eer_val[k] for k in ("n_pairs", "roc_auc", "eer")},
            "cal_clean_auc": roc_auc(sc.score, sc.label),
            "folds": {},
            "clean_cal_arm": {},
        }
        # clean-calibration arm (fold independent): candidates ordered on clean val
        rules_c = _grid_rules(sv.score, [thr["val_eer"]])
        for tname, (afm, afnm) in TARGETS.items():
            ordered = order_candidates_on_validation(
                rules_c,
                sv.score,
                sv.label,
                n_cal_expected=len(sc),
                target=min(afm, afnm),
                delta=DELTA,
            )
            res = LTTCalibrator(ordered, target_fm=afm, target_fnm=afnm, delta=DELTA).fit(
                sc.score, sc.label, pair_set=cal_set
            )
            b["clean_cal_arm"][tname] = _ltt_record(res) | {"n_ordered": len(ordered)}
            # v2 (post-hoc sensitivity, D-042): order by projected validation p-value
            o2 = order_candidates_by_validation_pvalue(
                rules_c,
                sv.score,
                sv.label,
                n_cal_expected=len(sc),
                target_fm=afm,
                target_fnm=afnm,
                delta=DELTA,
            )
            b["clean_cal_arm"][f"{tname}__v2"] = _fit_ordered(o2, sc, cal_set, afm, afnm)
        for fold in FAMILIES:
            vrows = _plan(layout, fold, "val", "val")
            crows = _plan(layout, fold, "cal", "cal_independent")
            sv_f = score_pairs(val_pairs, ident, emb.lookup(f"{fold}__val__val"), vrows)
            sc_f = score_pairs(cal_pairs, ident, emb.lookup(f"{fold}__cal__cal_independent"), crows)
            occ = sv_f.covis < 0.999999
            edge = float(np.median(sv_f.covis[occ]))
            rules = _grid_rules(sv_f.score, [thr["val_eer"]])
            st_val, st_cal = _strata(sv_f.covis, edge), _strata(sc_f.covis, edge)
            f: dict[str, Any] = {
                "vis_edge": edge,
                "val_occluded_auc": roc_auc(sv_f.score, sv_f.label),
                "cal_occluded_auc": roc_auc(sc_f.score, sc_f.label),
                "cal_strata_counts": {
                    s: int((st_cal == s).sum()) for s in ("clean", "vis_high", "vis_low")
                },
                "arms": {},
            }
            for tname, (afm, afnm) in TARGETS.items():
                ordered = order_candidates_on_validation(
                    rules,
                    sv_f.score,
                    sv_f.label,
                    n_cal_expected=len(sc_f),
                    target=min(afm, afnm),
                    delta=DELTA,
                )
                g = LTTCalibrator(ordered, target_fm=afm, target_fnm=afnm, delta=DELTA).fit(
                    sc_f.score,
                    sc_f.label,
                    pair_set=cal_set,
                    calibration_families=tuple(x for x in FAMILIES if x != fold),
                )
                by_stratum: dict[str, list[ThresholdRule]] = {}
                for s_name in ("clean", "vis_high", "vis_low"):
                    m = st_val == s_name
                    n_exp = int((st_cal == s_name).sum())
                    by_stratum[s_name] = order_candidates_on_validation(
                        rules,
                        sv_f.score[m],
                        sv_f.label[m],
                        n_cal_expected=max(n_exp, 1),
                        target=min(afm, afnm),
                        delta=DELTA / 3,
                    )  # empty -> no pre-specified rule: the stratum abstains
                strat = fit_stratified(
                    LTTCalibrator,
                    rules,
                    sc_f.score,
                    sc_f.label,
                    st_cal.tolist(),
                    cal_set,
                    target_fm=afm,
                    target_fnm=afnm,
                    delta=DELTA,
                    candidates_by_stratum=by_stratum,
                )
                seen = tuple(x for x in FAMILIES if x != fold)
                o2 = order_candidates_by_validation_pvalue(
                    rules,
                    sv_f.score,
                    sv_f.label,
                    n_cal_expected=len(sc_f),
                    target_fm=afm,
                    target_fnm=afnm,
                    delta=DELTA,
                )
                by2 = {
                    s_name: order_candidates_by_validation_pvalue(
                        rules,
                        sv_f.score[st_val == s_name],
                        sv_f.label[st_val == s_name],
                        n_cal_expected=max(int((st_cal == s_name).sum()), 1),
                        target_fm=afm,
                        target_fnm=afnm,
                        delta=DELTA / 3,
                    )
                    for s_name in STRATA
                }
                strat2 = fit_stratified(
                    LTTCalibrator,
                    rules,
                    sc_f.score,
                    sc_f.label,
                    st_cal.tolist(),
                    cal_set,
                    target_fm=afm,
                    target_fnm=afnm,
                    delta=DELTA,
                    candidates_by_stratum=by2,
                )
                f["arms"][tname] = {
                    "global": _ltt_record(g) | {"n_ordered": len(ordered)},
                    "stratified_vis": {
                        s_name: _ltt_record(r) | {"n_ordered": len(by_stratum[s_name])}
                        for s_name, r in strat.per_stratum.items()
                    },
                    # v2 = post-hoc sensitivity arms (D-042); not part of the pre-registration
                    "global_v2": _fit_ordered(o2, sc_f, cal_set, afm, afnm, seen),
                    "stratified_vis_v2": {
                        s_name: _ltt_record(r) | {"n_ordered": len(by2[s_name])}
                        for s_name, r in strat2.per_stratum.items()
                    },
                }
            b["folds"][fold] = f
            print(
                f"{bid} {fold}: edge={edge:.3f} "
                + json.dumps(
                    {
                        t: [
                            a["global"]["feasible"],
                            {s: v["feasible"] for s, v in a["stratified_vis"].items()},
                            a["global_v2"]["feasible"],
                            {s: v["feasible"] for s, v in a["stratified_vis_v2"].items()},
                        ]
                        for t, a in f["arms"].items()
                    }
                ),
                flush=True,
            )
        frozen["backbones"][bid] = b
    out = layout.private_reports / "phase2_frozen_eval_config.json"
    frozen["sha256"] = _sha(frozen)
    out.write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    pub = ROOT / "reports" / "experiments" / "phase2_develop.json"
    pub.parent.mkdir(parents=True, exist_ok=True)
    with pub.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            frozen
            | {
                "evidence_type": "real_measured_model_result",
                "scope": "validation/calibration only (no test access)",
                "provenance": collect_provenance(ROOT, argv=sys.argv),
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"frozen config sha256 {frozen['sha256']}")
    return 0


# ----------------------------------------------------------------- final


def _decide(rule: list[float] | None, scores: np.ndarray) -> np.ndarray:
    if rule is None:
        return np.full(scores.shape, -1, dtype=np.int64)
    return ThresholdRule(rule[0], rule[1]).decide(scores)


def _decide_stratified(arm: dict[str, Any], sp: ScoredPairs, edge: float) -> np.ndarray:
    st = _strata(sp.covis, edge)
    d = np.full(len(sp), -1, dtype=np.int64)
    for s_name, rec in arm.items():
        m = st == s_name
        d[m] = _decide(rec["rule"], sp.score[m])
    return d


def _groups(sp: ScoredPairs) -> Iterator[tuple[str, str, np.ndarray]]:
    for s in np.unique(sp.stratum):
        yield "stratum", str(s), sp.stratum == s
    fam = np.char.add(np.char.add(sp.fam_a, "|"), sp.fam_b)
    for f in np.unique(fam):
        yield "families", str(f), fam == f
    rank = {"": 0, "low": 1, "medium": 2, "high": 3}
    lvl = np.maximum(np.vectorize(rank.get)(sp.lvl_a), np.vectorize(rank.get)(sp.lvl_b))
    for k, name in ((0, "none"), (1, "low"), (2, "medium"), (3, "high")):
        yield "max_level", name, lvl == k
    cv = sp.covis
    for lo, hi in itertools.pairwise(VIS_BINS):
        yield "covis_bin", f"[{lo:.2f},{min(hi, 1.0):.2f})", (cv >= lo) & (cv < hi)


def _fixed_threshold_block(sp: ScoredPairs, t: float) -> dict[str, Any]:
    acc = sp.score >= t
    g, i = sp.label == 1, sp.label == 0
    return {
        "n": len(sp),
        "fmr": rate(acc, i, sp, "impostor"),
        "fnmr": rate(~acc, g, sp, "genuine"),
        "auc": roc_auc(sp.score, sp.label) if g.any() and i.any() else None,
    }


def _c1_exploratory(sp: ScoredPairs, t: float) -> dict[str, Any]:
    """Does oracle co-visibility rank errors beyond the feature-norm quality proxy?
    Fit-free: AUROC of -covis for decision errors, overall and within quality deciles."""
    acc = sp.score >= t
    err = (acc & (sp.label == 0)) | (~acc & (sp.label == 1))
    out: dict[str, Any] = {"n": len(sp), "n_errors": int(err.sum())}
    if err.sum() < 20 or (~err).sum() < 20:
        return out | {"note": "too few errors"}
    out["auroc_error_by_low_covis"] = roc_auc(-sp.covis, err.astype(np.int64))
    out["auroc_error_by_low_quality"] = roc_auc(-sp.min_norm, err.astype(np.int64))
    dec = np.quantile(sp.min_norm, np.linspace(0, 1, 11))
    within = []
    for lo, hi in itertools.pairwise(dec):
        m = (sp.min_norm >= lo) & (sp.min_norm <= hi)
        e = err[m]
        if 5 <= e.sum() < e.size - 5 and np.unique(sp.covis[m]).size > 1:
            within.append((int(m.sum()), roc_auc(-sp.covis[m], e.astype(np.int64))))
    if within:
        w = np.array([n for n, _ in within], dtype=np.float64)
        out["covis_auroc_within_quality_deciles"] = float(
            np.average([a for _, a in within], weights=w)
        )
        out["n_deciles_used"] = len(within)
    return out


def _audit(layout: DataLayout, ident: dict[str, str], res: dict[str, Any]) -> dict[str, Any]:
    split_ids: dict[str, set[str]] = defaultdict(set)
    for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord):
        split_ids[r.split.value].add(r.identity_id)
    pair_ids = {
        tag: {ident[p.image_a] for p in _pairs(layout, tag)}
        | {ident[p.image_b] for p in _pairs(layout, tag)}
        for tag in ("val", "cal_independent", "test_balanced", "test_far")
    }
    checks = {
        "identity_overlap_test_vs_cal": len(
            pair_ids["test_balanced"] & pair_ids["cal_independent"]
        ),
        "identity_overlap_test_vs_val": len(pair_ids["test_balanced"] & pair_ids["val"]),
        "identity_overlap_splits": {
            f"{a}-{b}": len(split_ids[a] & split_ids[b])
            for a in ("train", "val", "cal", "test")
            for b in ("train", "val", "cal", "test")
            if a < b
        },
    }
    for bid, r in res.items():
        c = r["clean"]["test_balanced"]
        checks[bid] = {
            "auc_is_exactly_1": c["roc_auc"] == 1.0,
            "label_shuffle_auc": r["audit_shuffle_auc"],
            "val_minus_test_auc": r["val_auc"] - c["roc_auc"],
            "per_part_auc": r["per_part_auc"],
        }
    return checks


def final(args: argparse.Namespace) -> int:
    """Test evaluation, or (``--rehearse``) the identical code path on the VALIDATION split
    with val plans standing in for test plans: no test access, no run-log entry, private
    output only. Used to catch failures before the single logged test run."""
    rehearse = bool(args.rehearse)
    if not rehearse:
        require_final_for_test(SplitName.TEST, final=args.final)
    pairs_tag, far_tag = ("val", "val") if rehearse else ("test_balanced", "test_far")
    plan_role = {"test_id": "val", "test_ood": "val"} if rehearse else {}
    layout = _layout()
    frozen = json.loads(
        (layout.private_reports / "phase2_frozen_eval_config.json").read_text("utf-8")
    )
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    if _sha(body) != frozen["sha256"]:
        raise RuntimeError("frozen evaluation config was modified after freezing")
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    log_entry = {
        "run_id": run_id,
        "utc": datetime.now(tz=UTC).isoformat(),
        "script": "scripts/evaluate_phase2.py final",
        "frozen_config_sha256": frozen["sha256"],
        "protocol_version": PROTO.protocol_version,
        "git_commit": prov.get("git", {}).get("commit")
        if isinstance(prov.get("git"), dict)
        else None,
        "splits": ["test"],
        "note": args.note,
    }
    if not rehearse:
        run_log = ROOT / PROTO.test_access.run_log
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(log_entry) + "\n")
    ident = _identity_map(layout)
    tb = _pairs(layout, pairs_tag)
    tf = _pairs(layout, far_tag)
    results: dict[str, Any] = {}
    for bid, fb in frozen["backbones"].items():
        emb = Embeddings(layout, bid)
        if emb.meta["backbone_fingerprint"] != fb["backbone_fingerprint"]:
            raise RuntimeError(f"{bid}: embeddings differ from those used in develop")
        clean = emb.lookup()
        thr = fb["thresholds"]
        s_tb = score_pairs(tb, ident, clean)
        s_tf = score_pairs(tf, ident, clean)
        r: dict[str, Any] = {"thresholds": thr, "val_auc": fb["val_clean"]["roc_auc"]}
        r["clean"] = {
            "test_balanced": verification_summary(s_tb, thr),
            "test_far_impostors": {
                name: rate(s_tf.score >= t, s_tf.label == 0, s_tf, "impostor")
                for name, t in thr.items()
            },
            "benchmark_mode_tar_at_far": {
                f"{far:g}": tar_at_far(s_tb.score, s_tb.label, far).__dict__ for far in FARS
            },
            "roc_points": roc_points(s_tb),
            "score_histogram": score_histogram(s_tb),
            "accuracy_at_val_best_threshold": float(
                ((s_tb.score >= thr["val_best_accuracy"]) == (s_tb.label == 1)).mean()
            ),
        }
        # DigiFace parts: identities 0-1999 have 72 images, identities >= 100000 have 5
        part = np.where(s_tb.id_a.astype(np.int64) >= 100_000, "5_img", "72_img")
        r["per_part_auc"] = {
            p: roc_auc(s_tb.score[part == p], s_tb.label[part == p])
            for p in ("5_img", "72_img")
            if np.unique(s_tb.label[part == p]).size == 2
        }
        rng = np.random.default_rng(0)
        r["audit_shuffle_auc"] = roc_auc(s_tb.score, rng.permutation(s_tb.label))
        r["clean_selective"] = {
            key: selective_summary(
                _decide(fb["clean_cal_arm"][key]["rule"], s_tb.score), s_tb, *TARGETS[tname]
            )
            for tname in TARGETS
            for key in (tname, f"{tname}__v2")
        }
        r["folds"] = {}
        for fold, ff in fb["folds"].items():
            fr: dict[str, Any] = {}
            for role in ("test_id", "test_ood"):
                prole = plan_role.get(role, role)
                rows = _plan(layout, fold, prole, pairs_tag)
                sp = score_pairs(tb, ident, emb.lookup(f"{fold}__{prole}__{pairs_tag}"), rows)
                if not np.array_equal(sp.pair_id, s_tb.pair_id):
                    raise RuntimeError("plan/clean pair order mismatch")
                block: dict[str, Any] = {
                    "overall": verification_summary(
                        sp,
                        {
                            "cal_far_1pct": thr["cal_far_1pct"],
                            "val_far_0.1pct": thr["val_far_0.1pct"],
                        },
                    ),
                    "paired_clean_same_pairs": {
                        "auc_clean": roc_auc(s_tb.score, s_tb.label),
                        "mean_score_drop_genuine": float(
                            (s_tb.score - sp.score)[sp.label == 1].mean()
                        ),
                        "mean_score_drop_impostor": float(
                            (s_tb.score - sp.score)[sp.label == 0].mean()
                        ),
                        "decision_flips_at_cal_far_1pct": int(
                            (
                                (s_tb.score >= thr["cal_far_1pct"])
                                != (sp.score >= thr["cal_far_1pct"])
                            ).sum()
                        ),
                    },
                    "roc_points": roc_points(sp),
                    "score_histogram": score_histogram(sp),
                    "subgroups": defaultdict(dict),
                    "selective": {},
                }
                for gname, gval, m in _groups(sp):
                    if m.any():
                        block["subgroups"][gname][gval] = _fixed_threshold_block(
                            sp.subset(m), thr["cal_far_1pct"]
                        )
                for tname in TARGETS:
                    arms = ff["arms"][tname]
                    d_g = _decide(arms["global"]["rule"], sp.score)
                    d_s = _decide_stratified(arms["stratified_vis"], sp, ff["vis_edge"])
                    d_c = _decide(fb["clean_cal_arm"][tname]["rule"], sp.score)
                    d_g2 = _decide(arms["global_v2"]["rule"], sp.score)
                    d_s2 = _decide_stratified(arms["stratified_vis_v2"], sp, ff["vis_edge"])
                    d_c2 = _decide(fb["clean_cal_arm"][f"{tname}__v2"]["rule"], sp.score)
                    st = _strata(sp.covis, ff["vis_edge"])
                    block["selective"][tname] = {
                        "global": selective_summary(d_g, sp, *TARGETS[tname]),
                        "clean_cal": selective_summary(d_c, sp, *TARGETS[tname]),
                        "stratified_vis": selective_summary(d_s, sp, *TARGETS[tname]),
                        "global_v2": selective_summary(d_g2, sp, *TARGETS[tname]),
                        "clean_cal_v2": selective_summary(d_c2, sp, *TARGETS[tname]),
                        "stratified_vis_v2": selective_summary(d_s2, sp, *TARGETS[tname]),
                        "stratified_vis_per_stratum": {
                            s_name: selective_summary(
                                d_s[st == s_name], sp.subset(st == s_name), *TARGETS[tname]
                            )
                            for s_name in ("clean", "vis_high", "vis_low")
                            if (st == s_name).any()
                        },
                    }
                block["c1_exploratory"] = _c1_exploratory(
                    sp.subset(sp.covis < 0.999999), thr["cal_far_1pct"]
                )
                fr[role] = block
            r["folds"][fold] = fr
            print(f"{bid} {fold} done", flush=True)
        stress_path = layout.manifests / "occlusion" / "stress__test_balanced.jsonl.gz"
        if not rehearse and (emb.dir / "stress__test_balanced.npz").exists():
            r["stress"] = _stress(layout, emb, ident, tb, stress_path, thr)
        results[bid] = r
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "DigiFace-1M (synthetic faces) with synthetic occlusions; not real-world",
        "run_id": run_id,
        "frozen_config_sha256": frozen["sha256"],
        "protocol_version": PROTO.protocol_version,
        "targets": TARGETS,
        "delta": DELTA,
        "results": results,
        "audit": _audit(layout, ident, results),
        "provenance": prov,
    }
    if rehearse:
        report["scope"] = "REHEARSAL on the validation split (val plans as stand-ins); not a result"
    pub = (
        layout.private_reports / "phase2_rehearsal.json"
        if rehearse
        else ROOT / "reports" / "experiments" / (args.out or "phase2_final.json")
    )
    with pub.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"run {run_id}: wrote {pub}")
    return 0


def _stress(
    layout: DataLayout,
    emb: Embeddings,
    ident: dict[str, str],
    tb: list[PairRecord],
    path: Path,
    thr: dict[str, float],
) -> dict[str, Any]:
    from veilx.embeddings.jobs import stress_key

    by_id = {p.pair_id: p for p in tb}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    lk = emb.lookup("stress__test_balanced")
    pairs = [by_id[r["pair_id"]] for r in rows]
    ka = [stress_key(p.image_a, r["steps_a"]) for p, r in zip(pairs, rows, strict=True)]
    kb = [stress_key(p.image_b, r["steps_b"]) for p, r in zip(pairs, rows, strict=True)]
    s = np.einsum("ij,ij->i", lk.emb[lk.rows(ka)], lk.emb[lk.rows(kb)]).astype(np.float64)
    clean = score_pairs(pairs, ident, emb.lookup())
    sp = ScoredPairs(**{f: getattr(clean, f) for f in clean.__dataclass_fields__} | {"score": s})
    return {
        "note": STRESS_NOTE,
        "stressed": _fixed_threshold_block(sp, thr["cal_far_1pct"]),
        "clean_same_pairs": _fixed_threshold_block(clean, thr["cal_far_1pct"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["develop", "final"])
    ap.add_argument("--final", action="store_true", help="required for test-split access")
    ap.add_argument("--rehearse", action="store_true", help="run the final code path on val")
    ap.add_argument("--note", default="", help="purpose recorded in the run log")
    ap.add_argument("--out", default="", help="output file name (reproducibility runs)")
    args = ap.parse_args()
    return develop() if args.stage == "develop" else final(args)


if __name__ == "__main__":
    sys.exit(main())
