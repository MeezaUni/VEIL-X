"""Phase 2 real-photo evaluation on LFW (Kaggle v4, D-043). No parameter is fitted on View 2.

Reported separately (never mixed):

A. **Official clean LFW** (View 2, 6,000 pairs, unmodified): 10-fold accuracy with the
   protocol's own fold-internal threshold (best accuracy on the other nine folds), plus ROC-AUC,
   EER and benchmark-mode TAR@FAR=1 % (3,000 impostors cannot support 0.1 %).
   Two preprocessing paths: ``det`` (SCRFD + 5-point alignment; detector-inclusive) and
   ``fixed`` (one transform from View-1 DevTrain; detector-free). In ``det``, pairs with an
   undetected image are scored as forced non-matches (official, all 6,000 pairs) and are also
   reported excluded (with counts).
B. **Transfer of frozen DigiFace rules** (freeze ``0e8a43fb…``): DigiFace thresholds and
   clean-calibrated LTT rules applied unchanged to clean LFW. A1 is violated by design
   (synthetic -> real), so no guarantee is claimed; realised error is measured.
C. **In-domain development thresholds** from LFW View-1 DevTrain (flagged: DevTrain shares
   identities with View 2).
D. **Derived synthetic occlusion** on detector-aligned crops (not official LFW, not physical
   occlusion): per family x level x {one side, both sides}; family F is scored with the frozen
   DigiFace rules of the fold where F was HELD OUT (the real-photo analogue of Test-OOD);
   mask false-match check; C1 (quality vs oracle co-visibility).

    .venv\\Scripts\\python.exe scripts\\evaluate_lfw.py --rehearse      (View-1 DevTest stand-in)
    .venv\\Scripts\\python.exe scripts\\evaluate_lfw.py --final         (View 2, logged once)
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import read_shard
from veilx.eval.scoring import ScoredPairs
from veilx.eval.summaries import (
    rate,
    roc_points,
    score_histogram,
    selective_summary,
    verification_summary,
)
from veilx.metrics.roc import roc_auc, tar_at_far
from veilx.models.spec import BACKBONES
from veilx.provenance import collect_provenance
from veilx.selective.ltt import ThresholdRule

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
LEVELS = ("low", "medium", "high")
FORCED = -1.0  # score for a pair with an undetected image (forced non-match)
TARGETS = {"primary_alpha": (0.05, 0.05), "far1_fm": (0.01, 0.05)}


def _strata(covis: np.ndarray, edge: float) -> np.ndarray:
    return np.where(covis >= 0.999999, "clean", np.where(covis >= edge, "vis_high", "vis_low"))


def _decide(rule: list[float] | None, s: np.ndarray) -> np.ndarray:
    if rule is None:
        return np.full(s.shape, -1, dtype=np.int64)
    return ThresholdRule(rule[0], rule[1]).decide(s)


def _decide_strat(arm: dict[str, Any], sp: ScoredPairs, edge: float) -> np.ndarray:
    st = _strata(sp.covis, edge)
    d = np.full(len(sp), -1, dtype=np.int64)
    for name, rec in arm.items():
        m = st == name
        d[m] = _decide(rec["rule"], sp.score[m])
    return d


def best_accuracy_threshold(score: np.ndarray, label: np.ndarray) -> float:
    """Threshold (accept >= t) maximising accuracy; candidates are midpoints of sorted scores."""
    order = np.argsort(score)
    ss, yy = score[order], label[order]
    cand = np.unique(ss)
    mids = np.concatenate([[cand[0] - 1e-6], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-6]])
    idx = np.searchsorted(ss, mids, side="left")  # number of scores below t
    gcum = np.concatenate([[0], np.cumsum(yy)])
    icum = np.concatenate([[0], np.cumsum(1 - yy)])
    correct = (gcum[-1] - gcum[idx]) + icum[idx]
    return float(mids[int(np.argmax(correct))])


def official_accuracy(score: np.ndarray, label: np.ndarray, fold: np.ndarray) -> dict[str, Any]:
    """LFW View-2 protocol: for each fold, the threshold maximising accuracy on the other nine
    folds is applied to the held-out fold; mean and standard deviation over folds."""
    accs, thrs = [], []
    folds = np.unique(fold)
    for k in folds:
        tr, te = (fold != k, fold == k) if len(folds) > 1 else (fold == k, fold == k)
        t = best_accuracy_threshold(score[tr], label[tr])
        thrs.append(t)
        accs.append(float(((score[te] >= t) == (label[te] == 1)).mean()))
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "per_fold": accs,
        "thresholds": thrs,
    }


class LFWData:
    def __init__(self, layout: Any, pairs_name: str) -> None:
        lfw = layout.manifests / "lfw"
        self.pairs = list(read_jsonl(lfw / f"{pairs_name}.jsonl", PairRecord))
        self.ident = {
            r.image_id: r.identity_id for r in read_jsonl(lfw / "images.jsonl", ImageRecord)
        }
        dets = [json.loads(x) for x in (lfw / "detections.jsonl").open(encoding="utf-8")]
        self.ok = {d["image_id"] for d in dets if d["ok"]}
        self.fold = np.array(
            [
                int(p.pair_id.split("-f")[1][:2])
                if "-f" in p.pair_id
                else int(p.pair_id[-3:]) % 10 + 1
                for p in self.pairs
            ]
        )
        self.label = np.array([1 if p.label == "genuine" else 0 for p in self.pairs], np.int64)
        self.id_a = np.array([self.ident[p.image_a] for p in self.pairs])
        self.id_b = np.array([self.ident[p.image_b] for p in self.pairs])
        self.both_ok = np.array([p.image_a in self.ok and p.image_b in self.ok for p in self.pairs])
        with gzip.open(lfw / "occlusion_plan.jsonl.gz", "rt", encoding="utf-8") as fh:
            want = {p.pair_id for p in self.pairs}
            self.plan = [r for line in fh if (r := json.loads(line))["pair_id"] in want]
        self.vis = json.loads((lfw / "occlusion_visibility.json").read_text("utf-8"))

    def sp(self, score: np.ndarray, mask: np.ndarray | None = None, **kw: Any) -> ScoredPairs:
        n = len(self.pairs)
        one, empty = np.ones(n), np.asarray([""] * n)
        base = {
            "pair_id": np.asarray([p.pair_id for p in self.pairs]),
            "score": score,
            "label": self.label,
            "id_a": self.id_a,
            "id_b": self.id_b,
            "vis_a": one,
            "vis_b": one,
            "fam_a": empty,
            "fam_b": empty,
            "lvl_a": empty,
            "lvl_b": empty,
            "stratum": np.asarray(["clean_clean"] * n),
            "norm_a": one,
            "norm_b": one,
        } | kw
        sp = ScoredPairs(**base)
        return sp if mask is None else sp.subset(mask)


def _lookup(shard: Any) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    return shard.index(), shard.emb, shard.norm.astype(np.float64)


def _clean_scores(
    data: LFWData, shard: Any, path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, emb, nrm = _lookup(shard)
    s = np.full(len(data.pairs), FORCED)
    na, nb = np.ones(len(data.pairs)), np.ones(len(data.pairs))
    for k, p in enumerate(data.pairs):
        if p.image_a in idx and p.image_b in idx:
            s[k] = float(emb[idx[p.image_a]] @ emb[idx[p.image_b]])
            na[k], nb[k] = nrm[idx[p.image_a]], nrm[idx[p.image_b]]
        elif path == "fixed":
            raise RuntimeError("fixed-crop path must cover every image")
    return s, na, nb


def _clean_block(
    data: LFWData,
    s: np.ndarray,
    frozen_b: dict[str, Any],
    dev_thr: dict[str, float],
    mask: np.ndarray | None,
) -> dict[str, Any]:
    sp = data.sp(s, mask)
    fold = data.fold if mask is None else data.fold[mask]
    thr = frozen_b["thresholds"] | {f"devtrain_{k}": v for k, v in dev_thr.items()}
    out = {
        "n_pairs": len(sp),
        "official_10fold_accuracy": official_accuracy(sp.score, sp.label, fold),
        "verification": verification_summary(sp, thr),
        "benchmark_tar_at_far_1pct": tar_at_far(sp.score, sp.label, 0.01).__dict__,
        "roc_points": roc_points(sp),
        "score_histogram": score_histogram(sp),
        "ltt_transfer_clean_cal": {
            key: selective_summary(
                _decide(frozen_b["clean_cal_arm"][key]["rule"], sp.score),
                sp,
                *TARGETS[key.split("__")[0]],
            )
            for key in frozen_b["clean_cal_arm"]
        },
    }
    return out


def _devtrain_thresholds(layout: Any, shard: Any) -> dict[str, float]:
    dev = LFWData(layout, "view1_devtrain_pairs")
    s, _, _ = _clean_scores(dev, shard, "det")
    keep = s > FORCED
    v = verification_summary(dev.sp(s, keep), {})
    return {
        "eer": float(v["eer_threshold"]),
        "best_accuracy": best_accuracy_threshold(s[keep], dev.label[keep]),
    }


def _occluded(
    data: LFWData, clean: Any, occ: Any, frozen_b: dict[str, Any], thr: float
) -> dict[str, Any]:
    ci, ce, cn = _lookup(clean)
    oi, oe, on = _lookup(occ)
    base_scores, _, _ = _clean_scores(data, clean, "det")
    pos = {p.pair_id: k for k, p in enumerate(data.pairs)}
    groups: dict[tuple[str, str, str], list[tuple[int, float, float, float, float, float]]] = (
        defaultdict(list)
    )
    for r in data.plan:
        k = pos[r["pair_id"]]
        p = data.pairs[k]
        sides = []
        for side, img in (("a", p.image_a), ("b", p.image_b)):
            spec = r[side]
            if spec is None:
                sides.append((ce[ci[img]], cn[ci[img]], 1.0))
            else:
                key = occluded_key(img, r["family"], r["level"], spec["seed"])
                sides.append((oe[oi[key]], on[oi[key]], float(data.vis[key]["visibility_ratio"])))
        s = float(sides[0][0] @ sides[1][0])
        groups[(r["family"], r["level"], r["mode"])].append(
            (k, s, sides[0][1], sides[1][1], sides[0][2], sides[1][2])
        )
    out: dict[str, Any] = {}
    for fam in FAMILIES:
        fam_rows: list[Any] = []
        cells: dict[str, Any] = {}
        for lvl in LEVELS:
            for mode in ("one", "both"):
                rows = groups[(fam, lvl, mode)]
                fam_rows += rows
                cells[f"{lvl}:{mode}"] = _occl_cell(data, rows, base_scores, thr)
        sp = _rows_sp(data, fam_rows, fam)
        ff = frozen_b["folds"][fam]
        sel = {}
        for tname in TARGETS:
            arms = ff["arms"][tname]
            sel[tname] = {
                "global": selective_summary(
                    _decide(arms["global"]["rule"], sp.score), sp, *TARGETS[tname]
                ),
                "stratified_vis": selective_summary(
                    _decide_strat(arms["stratified_vis"], sp, ff["vis_edge"]), sp, *TARGETS[tname]
                ),
                "global_v2": selective_summary(
                    _decide(arms["global_v2"]["rule"], sp.score), sp, *TARGETS[tname]
                ),
                "stratified_vis_v2": selective_summary(
                    _decide_strat(arms["stratified_vis_v2"], sp, ff["vis_edge"]),
                    sp,
                    *TARGETS[tname],
                ),
            }
        out[fam] = {
            "cells": cells,
            "pooled_at_digiface_cal_far_1pct": _fixed(sp, thr),
            "pooled_auc": roc_auc(sp.score, sp.label),
            "selective_heldout_fold_rules": sel,
            "c1": _c1(sp, thr),
        }
    return out


def _rows_sp(data: LFWData, rows: list[Any], fam: str) -> ScoredPairs:
    k = np.array([r[0] for r in rows])
    sp = ScoredPairs(
        pair_id=np.asarray([data.pairs[i].pair_id for i in k]),
        score=np.array([r[1] for r in rows]),
        label=data.label[k],
        id_a=data.id_a[k],
        id_b=data.id_b[k],
        vis_a=np.array([r[4] for r in rows]),
        vis_b=np.array([r[5] for r in rows]),
        fam_a=np.asarray([fam if r[4] < 1 else "" for r in rows]),
        fam_b=np.asarray([fam] * len(rows)),
        lvl_a=np.asarray([""] * len(rows)),
        lvl_b=np.asarray([""] * len(rows)),
        stratum=np.asarray(["lfw_derived"] * len(rows)),
        norm_a=np.array([r[2] for r in rows]),
        norm_b=np.array([r[3] for r in rows]),
    )
    return sp


def _fixed(sp: ScoredPairs, t: float) -> dict[str, Any]:
    acc = sp.score >= t
    return {
        "n": len(sp),
        "fmr": rate(acc, sp.label == 0, sp, "impostor"),
        "fnmr": rate(~acc, sp.label == 1, sp, "genuine"),
    }


def _occl_cell(data: LFWData, rows: list[Any], base: np.ndarray, t: float) -> dict[str, Any]:
    sp = _rows_sp(data, rows, "")
    k = np.array([r[0] for r in rows])
    clean = data.sp(base).subset(np.isin(np.arange(len(data.pairs)), k))  # same pairs, clean
    return {
        "n": len(sp),
        "auc": roc_auc(sp.score, sp.label),
        "official_style_10fold_accuracy": official_accuracy(sp.score, sp.label, data.fold[k])[
            "mean"
        ],
        "at_digiface_cal_far_1pct": _fixed(sp, t),
        "clean_same_pairs_at_digiface_cal_far_1pct": _fixed(clean, t),
        "mean_covis": float(sp.covis.mean()),
    }


def _c1(sp: ScoredPairs, t: float) -> dict[str, Any]:
    occ = sp.covis < 0.999999
    sp = sp.subset(occ)
    acc = sp.score >= t
    err = ((acc & (sp.label == 0)) | (~acc & (sp.label == 1))).astype(np.int64)
    out: dict[str, Any] = {"n": len(sp), "n_errors": int(err.sum())}
    if err.sum() < 20 or (1 - err).sum() < 20:
        return out | {"note": "too few errors"}
    out["auroc_error_by_low_covis"] = roc_auc(-sp.covis, err)
    out["auroc_error_by_low_quality"] = roc_auc(-sp.min_norm, err)
    dec = np.quantile(sp.min_norm, np.linspace(0, 1, 11))
    within = []
    for lo, hi in itertools.pairwise(dec):
        m = (sp.min_norm >= lo) & (sp.min_norm <= hi)
        e = err[m]
        if 5 <= e.sum() < e.size - 5 and np.unique(sp.covis[m]).size > 1:
            within.append((int(m.sum()), roc_auc(-sp.covis[m], e)))
    if within:
        out["covis_auroc_within_quality_deciles"] = float(
            np.average([a for _, a in within], weights=[n for n, _ in within])
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--final", action="store_true")
    args = ap.parse_args()
    layout = resolve_data_root()
    frozen = json.loads(
        (layout.private_reports / "phase2_frozen_eval_config.json").read_text("utf-8")
    )
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != frozen["sha256"]:
        raise RuntimeError("frozen DigiFace config was modified")
    pairs_name = "view2_pairs" if args.final else "view1_devtest_pairs"
    data = LFWData(layout, pairs_name)
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    if args.final:
        entry = {
            "run_id": run_id,
            "utc": datetime.now(tz=UTC).isoformat(),
            "script": "scripts/evaluate_lfw.py --final",
            "splits": ["lfw_view2"],
            "frozen_config_sha256": frozen["sha256"],
            "git_commit": (prov.get("git") or {}).get("commit"),
        }
        log = ROOT / PROTO.test_access.run_log
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry) + "\n")
    results: dict[str, Any] = {}
    for bid in BACKBONES:
        d = next(
            p for p in (layout.embeddings / bid).iterdir() if (p / "lfw_det_clean.npz").exists()
        )
        fb = frozen["backbones"][bid]
        sh_det, sh_fix, sh_occ = (
            read_shard(d / f"{n}.npz") for n in ("lfw_det_clean", "lfw_fixed_clean", "lfw_det_occl")
        )
        dev_thr = _devtrain_thresholds(layout, sh_det)
        s_det, _na, _nb = _clean_scores(data, sh_det, "det")
        s_fix, _fa, _fbn = _clean_scores(data, sh_fix, "fixed")
        r: dict[str, Any] = {"devtrain_thresholds": dev_thr}
        r["clean_det_forced_nonmatch"] = _clean_block(data, s_det, fb, dev_thr, None)
        r["clean_det_excluding_failed"] = _clean_block(data, s_det, fb, dev_thr, data.both_ok)
        r["clean_fixed_crop"] = _clean_block(data, s_fix, fb, dev_thr, None)
        r["derived_occlusion_det"] = _occluded(
            data, sh_det, sh_occ, fb, fb["thresholds"]["cal_far_1pct"]
        )
        results[bid] = r
        print(
            bid,
            json.dumps(
                {
                    k: round(v["official_10fold_accuracy"]["mean"], 4)
                    for k, v in r.items()
                    if k.startswith("clean")
                }
            ),
            flush=True,
        )
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "LFW (Kaggle v4) official View 2 clean + derived synthetic occlusion; "
            "REHEARSAL on View-1 DevTest (not a result)"
            if args.rehearse
            else "LFW (Kaggle v4) official View 2 clean + derived synthetic occlusion"
        ),
        "run_id": run_id,
        "pairs": pairs_name,
        "n_pairs": len(data.pairs),
        "pairs_with_undetected_image": int((~data.both_ok).sum()),
        "frozen_digiface_config_sha256": frozen["sha256"],
        "results": results,
        "provenance": prov,
    }
    out = (
        ROOT / "reports" / "experiments" / "lfw_final.json"
        if args.final
        else layout.private_reports / "lfw_rehearsal.json"
    )
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"run {run_id}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
