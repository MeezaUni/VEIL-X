# ruff: noqa: E501  (long result-record literals)
"""Phase 4F: detector-inclusive robustness with an explicit failure taxonomy and policies.

Synthetic occluders (the Phase 2 LFW plan, "one" mode: side b only) are composited into the
ORIGINAL photograph, then SCRFD re-detects and the face is re-aligned. Each composite gets one
category (``veilx.detect.policy``): no_face / low_quality / multiple_faces / success.

- ``min_score`` = 0.6 (registered).
- ``max_residual`` = 99th percentile of the 5-point alignment residual on clean View-1 DevTrain
  detections.

Policies for no_face / low_quality: abstain; forced non-match; fallback. The fallback embeds the
detector-free fixed crop of the composite and scores it with the frozen quality-stratified rule
A3. Decisions use each family's held-out fold (frozen Phase 3 M1 + A3 rules).

``--split dev``      LFW View 1 DevTest, 300 seeded pairs (development). Applies the
                     REGISTERED SELECTION RULE: minimise the worst-family sFNMR subject to every
                     family's sFMR <= 5 %; ties go to the higher mean genuine coverage.
``--split view2``    LFW View 2, the same 300 pairs as the Phase 3 exploratory study.
                     FOLLOW-UP only (logged); reports every policy and marks the dev-selected one.

    .venv-gpu\\Scripts\\python.exe scripts\\phase4_detector.py --split dev|view2 [--workers 4]
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb, pixel_sha256
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.detect.align import align_to_template, alignment_residual, warp
from veilx.detect.policy import POLICIES, apply_policy, categorize
from veilx.embeddings.jobs import occluded_key
from veilx.fusion.features import design
from veilx.fusion.logreg import LogisticModel
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH
from veilx.occlusion.engine import apply_occlusion
from veilx.provenance import collect_provenance
from veilx.seeding import derive_seed

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
LEVELS = ("low", "medium", "high")
DET_MODEL = "insightface_buffalo_l_v0.7/det_10g.onnx"
MIN_SCORE = 0.6
_spec = importlib.util.spec_from_file_location(
    "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
)
assert _spec is not None and _spec.loader is not None
ev3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev3)
_W: dict[str, Any] = {}


def _init(model: str, photo_root: str, crop_root: str, m_fixed: list[list[float]]) -> None:
    from veilx.detect.scrfd import SCRFDDetector

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    _W.update(
        det=SCRFDDetector(Path(model)),
        photo=Path(photo_root),
        crop=Path(crop_root),
        m_fixed=np.asarray(m_fixed),
    )


def _composite(job: dict[str, Any]) -> dict[str, Any]:
    import cv2

    from veilx.detect.scrfd import select_central

    photo = load_rgb(_W["photo"] / job["relpath"])
    crop = load_rgb(_W["crop"] / (job["image_id"].replace(":", "__") + ".png"))
    _, m0 = align_to_template(photo, np.asarray(job["landmarks"], dtype=np.float64))
    res = apply_occlusion(
        crop, job["family"], job["level"], job["seed"], source_pixel_sha256=pixel_sha256(crop)
    )
    h, w = photo.shape[:2]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    occ_back = cv2.warpAffine(res.image, m0, (w, h), flags=flags, borderValue=0)
    mask_back = cv2.warpAffine(
        res.mask.astype(np.uint8) * 255, m0, (w, h), flags=flags, borderValue=0
    )
    inside = cv2.warpAffine(
        np.full((112, 112), 255, np.uint8), m0, (w, h), flags=flags, borderValue=0
    )
    sel = inside >= 128 if job["family"] == "F5_extreme_lighting" else mask_back >= 128
    comp = photo.copy()
    comp[sel] = occ_back[sel]
    dets = _W["det"].detect(comp)
    out: dict[str, Any] = {
        "key": job["key"],
        "n_faces": len(dets),
        "fixed": warp(comp, _W["m_fixed"]),
    }
    k = select_central(dets, comp.shape[:2])
    if k is None:
        return out | {"score": None, "residual": None, "crop": None}
    lm1 = dets[k].landmarks.astype(np.float64)
    new_crop, _ = align_to_template(comp, lm1)
    return out | {
        "score": float(dets[k].score),
        "residual": alignment_residual(lm1),
        "crop": new_crop,
    }


def _summ(d: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    g, i = y == 1, y == 0
    acc = d != ABSTAIN
    return {
        "coverage_genuine": float(acc[g].mean()),
        "coverage_impostor": float(acc[i].mean()),
        "sfnmr": float(((d == NON_MATCH) & g).sum() / max((acc & g).sum(), 1)),
        "sfmr": float(((d == MATCH) & i).sum() / max((acc & i).sum(), 1)),
        "n_genuine": int(g.sum()),
        "n_impostor": int(i.sum()),
    }


def main() -> int:
    from veilx.models.extractor import FrozenExtractor
    from veilx.models.spec import BACKBONES

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=["dev", "view2"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()
    lay = resolve_data_root()
    lfw = lay.manifests / "lfw"
    fz3 = json.loads((lay.private_reports / "phase3_frozen.json").read_text("utf-8"))
    dets = {
        d["image_id"]: d
        for line in (lfw / "detections.jsonl").open(encoding="utf-8")
        if (d := json.loads(line))
    }
    ok = {i for i, d in dets.items() if d["ok"]}
    imgs = {r.image_id: r for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    devtrain = list(read_jsonl(lfw / "view1_devtrain_pairs.jsonl", PairRecord))
    dt_ids = sorted({p.image_a for p in devtrain} | {p.image_b for p in devtrain})
    residuals = np.array(
        [alignment_residual(np.asarray(dets[i]["landmarks"])) for i in dt_ids if dets[i]["ok"]]
    )
    max_res = float(np.quantile(residuals, 0.99))
    if args.split == "dev":
        pool_pairs = [
            p
            for p in read_jsonl(lfw / "view1_devtest_pairs.jsonl", PairRecord)
            if p.image_a in ok and p.image_b in ok
        ]
        rng = np.random.default_rng(derive_seed(PROTO.seeds.global_seed, "p4-detector-dev"))
    else:
        pool_pairs = [
            p
            for p in read_jsonl(lfw / "view2_pairs.jsonl", PairRecord)
            if p.image_a in ok and p.image_b in ok
        ]
        rng = np.random.default_rng(
            derive_seed(PROTO.seeds.global_seed, "p3-detector-inclusive")
        )  # same 300 as Phase 3
    gen = sorted((p for p in pool_pairs if p.label == "genuine"), key=lambda p: p.pair_id)
    imp = sorted((p for p in pool_pairs if p.label != "genuine"), key=lambda p: p.pair_id)
    sample = [gen[i] for i in sorted(rng.choice(len(gen), 150, replace=False))] + [
        imp[i] for i in sorted(rng.choice(len(imp), 150, replace=False))
    ]
    want = {p.pair_id for p in sample}
    with gzip.open(lfw / "occlusion_plan.jsonl.gz", "rt", encoding="utf-8") as fh:
        plan = {
            (r["pair_id"], r["family"], r["level"]): r
            for line in fh
            if (r := json.loads(line))["pair_id"] in want and r["mode"] == "one"
        }
    jobs = []
    for p in sample:
        for fam in FAMILIES:
            for lvl in LEVELS:
                sb = plan[(p.pair_id, fam, lvl)]["b"]
                jobs.append(
                    {
                        "key": occluded_key(p.image_b, fam, lvl, int(sb["seed"])),
                        "pair_id": p.pair_id,
                        "image_id": p.image_b,
                        "relpath": imgs[p.image_b].relpath,
                        "landmarks": dets[p.image_b]["landmarks"],
                        "family": fam,
                        "level": lvl,
                        "seed": int(sb["seed"]),
                        "label": 1 if p.label == "genuine" else 0,
                    }
                )
    if args.pilot:
        jobs = jobs[: args.pilot // 2] + jobs[-(args.pilot // 2) :]
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    if args.split == "view2" and not args.pilot:
        with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "utc": datetime.now(tz=UTC).isoformat(),
                        "script": "scripts/phase4_detector.py --split view2",
                        "splits": ["lfw_view2"],
                        "git_commit": (prov.get("git") or {}).get("commit"),
                        "note": "Phase 4F FOLLOW-UP detector taxonomy/policies; 300 pairs; frozen Phase 3 A3 rules",
                    }
                )
                + "\n"
            )
    m_fixed = json.loads((lfw / "fixed_transform.json").read_text("utf-8"))["matrix"]
    with ProcessPoolExecutor(
        args.workers,
        initializer=_init,
        initargs=(
            str(lay.models / DET_MODEL),
            str(lay.extracted / "lfw_kaggle_v4"),
            str(lay.extracted / "lfw_aligned" / "det"),
            m_fixed,
        ),
    ) as pool:
        results = list(pool.map(_composite, jobs, chunksize=8))
    ada = FrozenExtractor(
        BACKBONES["adaface_ir101_webface4m"],
        lay.models
        / BACKBONES["adaface_ir101_webface4m"].hf_repo.replace("/", "__")
        / BACKBONES["adaface_ir101_webface4m"].revision,
        device="cuda",
        precision="fp16",
    )

    def embed(crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        e, n = [], []
        for i in range(0, len(crops), 256):
            b = ada.embed(np.stack(crops[i : i + 256]))
            e.append(b.embeddings)
            n.append(b.feature_norms)
        return np.concatenate(e), np.concatenate(n)

    has = [r["crop"] is not None for r in results]
    det_e = np.zeros((len(results), 512), np.float32)
    det_n = np.zeros(len(results), np.float32)
    if any(has):
        e, n = embed([r["crop"] for r in results if r["crop"] is not None])
        det_e[np.array(has)], det_n[np.array(has)] = e, n
    fix_e, fix_n = embed([r["fixed"] for r in results])
    clean = ev3.Group(
        lay.root / "features" / "phase3" / "followup", "lfw_clean", ("emb_ada", "norm_ada")
    )
    cats = np.array(
        [
            categorize(
                r["n_faces"], r["score"], r["residual"], min_score=MIN_SCORE, max_residual=max_res
            )
            for r in results
        ]
    )
    out: dict[str, Any] = {
        "split": args.split,
        "n_pairs": len(sample),
        "n_composites": len(jobs),
        "min_score": MIN_SCORE,
        "max_residual_p99_devtrain_clean": max_res,
        "taxonomy": {},
        "policies": {},
    }
    for fam in FAMILIES:
        idx = np.array([k for k, j in enumerate(jobs) if j["family"] == fam])
        out["taxonomy"][fam] = {
            lvl: {
                c: int(
                    ((cats[idx] == c) & (np.array([jobs[k]["level"] for k in idx]) == lvl)).sum()
                )
                for c in ("no_face", "low_quality", "multiple_faces", "success")
            }
            for lvl in LEVELS
        }
        f3 = fz3["folds"][fam]
        m1 = LogisticModel.from_dict(f3["models"]["M1"])
        rec = f3["arms"]["A3_quality|primary|v2"]
        y = np.array([jobs[k]["label"] for k in idx])

        def decisions(
            emb: np.ndarray,
            nrm: np.ndarray,
            idx: np.ndarray = idx,
            m1: LogisticModel = m1,
            rec: dict[str, Any] = rec,
            f3: dict[str, Any] = f3,
        ) -> np.ndarray:
            ea = np.stack(
                [
                    clean.arr["emb_ada"][
                        clean.index[
                            next(p for p in sample if p.pair_id == jobs[k]["pair_id"]).image_a
                        ]
                    ]
                    for k in idx
                ]
            ).astype(np.float32)
            na = np.array(
                [
                    clean.arr["norm_ada"][
                        clean.index[
                            next(p for p in sample if p.pair_id == jobs[k]["pair_id"]).image_a
                        ]
                    ]
                    for k in idx
                ],
                np.float64,
            )
            ea /= np.linalg.norm(ea, axis=1, keepdims=True)
            eb = emb[idx] / np.maximum(np.linalg.norm(emb[idx], axis=1, keepdims=True), 1e-12)
            t = {
                "s": np.einsum("ij,ij->i", ea, eb).astype(np.float64),
                "q": np.minimum(na, nrm[idx].astype(np.float64)),
            }
            p = m1.predict_proba(design(t, "M1", "cv_region")[0])
            return ev3.decide(rec, p, ev3.strata_labels("quality", t, f3["edges"], "cv_region"))

        normal, fallback = decisions(det_e, det_n), decisions(fix_e, fix_n)
        out["policies"][fam] = {
            pol: _summ(apply_policy(pol, cats[idx], normal, fallback), y) for pol in POLICIES
        }
        out["policies"][fam]["fixed_crop_only"] = _summ(fallback, y)
    worst = {pol: max(out["policies"][f][pol]["sfnmr"] for f in FAMILIES) for pol in POLICIES}
    feasible = [
        pol for pol in POLICIES if all(out["policies"][f][pol]["sfmr"] <= 0.05 for f in FAMILIES)
    ]
    cov = {
        pol: float(np.mean([out["policies"][f][pol]["coverage_genuine"] for f in FAMILIES]))
        for pol in POLICIES
    }
    out["selection"] = {
        "rule": "min worst-family sFNMR s.t. all sFMR <= 5 %; ties -> higher mean genuine coverage",
        "worst_family_sfnmr": worst,
        "mean_genuine_coverage": cov,
        "selected": min(feasible, key=lambda p: (round(worst[p], 6), -cov[p]))
        if feasible
        else None,
    }
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "Phase 4F DEVELOPMENT (LFW View 1 DevTest)"
            if args.split == "dev"
            else "Phase 4F FOLLOW-UP (LFW View 2; dev-selected policy marked)"
        )
        + ": synthetic occluders composited into real photographs; frozen Phase 3 A3 rules",
        "run_id": run_id,
        "results": out,
        "provenance": prov,
    }
    if args.split == "view2":
        dev = json.loads(
            (ROOT / "reports" / "experiments" / "phase4_detector_dev.json").read_text("utf-8")
        )
        report["dev_selected_policy"] = dev["results"]["selection"]["selected"]
    name = "pilot" if args.pilot else args.split
    dest = (
        (lay.private_reports / f"phase4_detector_{name}.json")
        if args.pilot
        else (ROOT / "reports" / "experiments" / f"phase4_detector_{args.split}.json")
    )
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps(out["selection"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
