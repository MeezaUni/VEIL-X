"""EXPLORATORY (Phase 3, gate 3G): detector-inclusive robustness on LFW View 2.

In the derived-occlusion study the occluder is drawn on the already-aligned crop, so detection and
alignment are untouched. Here the same occluder (same family, level, seed and recipe as the
"one"-mode LFW plan row) is drawn on the aligned crop of side b and then **inverse-warped into the
original photograph**; SCRFD then re-detects and re-aligns the composite. This measures what the
crop-level study cannot: detection failures, landmark displacement and their effect on scores,
visibility estimates and the frozen Phase 3 decisions.

Design (fixed before running):
- 300 View-2 pairs (150 genuine / 150 impostor; both images detected when clean), seeded sample;
- every family x level (15 cells), side b occluded only;
- the frozen fold-F models and rules for family F (F held out of that fold's estimator/fusion);
- detection failure = no face (or no face near the centre): reported, then handled two ways,
  as abstention and as forced non-match;
- GT visibility of the re-aligned crop = the composite mask warped by the NEW alignment.
The occluder is composited only inside the 112x112 crop footprint. For F5 (extreme lighting) this
means the photometric change covers the crop region only, not the whole photograph.

LFW View 2 is re-used here (exploratory; no fitting). Aggregates only are written publicly.

    .venv-gpu\\Scripts\\python.exe scripts\\lfw_detector_inclusive.py --final [--workers 6]
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import time
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
from veilx.detect.align import align_to_template, transform_points
from veilx.embeddings.jobs import occluded_key
from veilx.occlusion.engine import apply_occlusion
from veilx.provenance import collect_provenance
from veilx.seeding import derive_seed
from veilx.vis.heuristic import cell_means
from veilx.vis.regions import visibility_target

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
LEVELS = ("low", "medium", "high")
N_PER_CLASS = 150
DET_MODEL = "insightface_buffalo_l_v0.7/det_10g.onnx"
_W: dict[str, Any] = {}


def _load_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init(model: str, photo_root: str, crop_root: str, mu: np.ndarray, sd: np.ndarray) -> None:
    from veilx.detect.scrfd import SCRFDDetector

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    _W.update(
        det=SCRFDDetector(Path(model)), photo=Path(photo_root), crop=Path(crop_root), mu=mu, sd=sd
    )


def _composite(job: dict[str, Any]) -> dict[str, Any]:
    """Occlude the aligned crop, inverse-warp into the photo, re-detect, re-align."""
    import cv2

    from veilx.detect.scrfd import select_central

    photo = load_rgb(_W["photo"] / job["relpath"])
    crop = load_rgb(_W["crop"] / (job["image_id"].replace(":", "__") + ".png"))
    lm0 = np.asarray(job["landmarks"], dtype=np.float64)
    _, m0 = align_to_template(photo, lm0)
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
    # photometric family (F5): the whole crop footprint is re-lit; others: the occluder mask
    sel = inside >= 128 if job["family"] == "F5_extreme_lighting" else mask_back >= 128
    comp = photo.copy()
    comp[sel] = occ_back[sel]
    dets = _W["det"].detect(comp)
    k = select_central(dets, comp.shape[:2])
    out: dict[str, Any] = {"key": job["key"], "n_faces": len(dets)}
    if k is None:
        return out | {"ok": False}
    lm1 = dets[k].landmarks.astype(np.float64)
    new_crop, m1 = align_to_template(comp, lm1)
    # landmark displacement measured in the CLEAN crop's coordinate frame (pixels of 112)
    disp = np.linalg.norm(transform_points(m0, lm1) - transform_points(m0, lm0), axis=1)
    gt_mask = (
        cv2.warpAffine(
            (mask_back >= 128).astype(np.uint8) * 255, m1, (112, 112), flags=cv2.INTER_LINEAR
        )
        >= 128
    )
    zz = (cell_means(new_crop[None]) - _W["mu"]) / _W["sd"]
    return out | {
        "ok": True,
        "score": float(dets[k].score),
        "landmark_disp_mean_px": float(disp.mean()),
        "landmark_disp_max_px": float(disp.max()),
        "crop": new_crop,
        "gt": np.rint(255 * visibility_target(gt_mask)).astype(np.uint8),
        "z": np.sqrt((zz**2).sum(-1))[0].astype(np.float16),
        "vis_crop_level": float(res.spec.visibility_ratio),
    }


def main() -> int:
    import torch
    from safetensors.torch import load_file

    from veilx.models.extractor import FrozenExtractor
    from veilx.models.spec import BACKBONES
    from veilx.vis.model import VisNet, to_input

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--final", action="store_true", help="required: uses LFW View 2 (logged)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--pilot", type=int, default=0, help="first N jobs only; nothing logged/written publicly"
    )
    args = ap.parse_args()
    if not (args.final or args.pilot):
        raise SystemExit("use --pilot N or --final")
    ev = _load_eval_module()
    layout = resolve_data_root()
    lfw = layout.manifests / "lfw"
    fz = json.loads((layout.private_reports / "phase3_frozen.json").read_text("utf-8"))
    cv_def = fz["cv_selection"]["selected"]
    dets = {
        d["image_id"]: d
        for line in (lfw / "detections.jsonl").open(encoding="utf-8")
        if (d := json.loads(line))
    }
    ok = {i for i, d in dets.items() if d["ok"]}
    imgs = {r.image_id: r for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    ident = {i: r.identity_id for i, r in imgs.items()}
    view2 = [
        p
        for p in read_jsonl(lfw / "view2_pairs.jsonl", PairRecord)
        if p.image_a in ok and p.image_b in ok
    ]
    rng = np.random.default_rng(derive_seed(PROTO.seeds.global_seed, "p3-detector-inclusive"))
    gen = sorted((p for p in view2 if p.label == "genuine"), key=lambda p: p.pair_id)
    imp = sorted((p for p in view2 if p.label != "genuine"), key=lambda p: p.pair_id)
    sample = [gen[i] for i in sorted(rng.choice(len(gen), N_PER_CLASS, replace=False))] + [
        imp[i] for i in sorted(rng.choice(len(imp), N_PER_CLASS, replace=False))
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
                    }
                )
    if args.pilot:
        jobs = jobs[: args.pilot]
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    if args.final:
        with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "utc": datetime.now(tz=UTC).isoformat(),
                        "script": "scripts/lfw_detector_inclusive.py",
                        "splits": ["lfw_view2"],
                        "frozen_config_sha256": fz["sha256"],
                        "git_commit": (prov.get("git") or {}).get("commit"),
                        "note": "EXPLORATORY detector-inclusive; 300 pairs; frozen Phase 3 rules",
                    }
                )
                + "\n"
            )
    st = np.load(layout.root / "models" / "phase3" / "heuristic_stats.npz")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers,
        initializer=_init,
        initargs=(
            str(layout.models / DET_MODEL),
            str(layout.extracted / "lfw_kaggle_v4"),
            str(layout.extracted / "lfw_aligned" / "det"),
            st["mu"],
            st["sd"],
        ),
    ) as pool:
        results = list(pool.map(_composite, jobs, chunksize=8))
    detect_s = time.perf_counter() - t0
    by_key = {r["key"]: r for r in results}
    okr = [r for r in results if r["ok"]]
    # ---- embeddings + learned maps of re-aligned composites (GPU)
    dev = torch.device("cuda")
    bbs = [
        FrozenExtractor(
            s,
            layout.models / s.hf_repo.replace("/", "__") / s.revision,
            device="cuda",
            precision="fp16",
        )
        for s in BACKBONES.values()
    ]
    crops = np.stack([r["crop"] for r in okr]) if okr else np.zeros((0, 112, 112, 3), np.uint8)
    arr: dict[str, Any] = {"emb_ada": [], "emb_arc": [], "norm_ada": [], "norm_arc": []}
    for i in range(0, len(crops), 256):
        for bb in bbs:
            e = bb.embed(crops[i : i + 256])
            arr[f"emb_{bb.backbone_id[:3]}"].append(e.embeddings.astype(np.float16))
            arr[f"norm_{bb.backbone_id[:3]}"].append(e.feature_norms)
    arr = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in arr.items()}
    fam_of = np.array([by_key[r["key"]]["key"].split("|")[1] for r in okr])
    ve_maps = np.zeros((len(okr), 28, 28), np.uint8)
    for fam in FAMILIES:
        m = fam_of == fam
        if not m.any():
            continue
        net = VisNet().to(dev).eval()
        net.load_state_dict(
            load_file(
                str(layout.root / "models" / "phase3" / "ve" / fam / "best.safetensors"),
                device="cuda",
            )
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(torch.from_numpy(to_input(crops[m])).to(dev)).float())
        ve_maps[m] = np.rint(255 * p.cpu().numpy()).astype(np.uint8)
    occ = object.__new__(ev.Group)
    occ.keys = np.array([r["key"] for r in okr])
    occ.arr = {
        **arr,
        "ve": ve_maps,
        "gt": np.stack([r["gt"] for r in okr]) if okr else np.zeros((0, 28, 28), np.uint8),
        "z": np.stack([r["z"] for r in okr]) if okr else np.zeros((0, 28, 28), np.float16),
        "vis": np.array([r["vis_crop_level"] for r in okr], np.float32),
    }
    occ.index = {str(k): i for i, k in enumerate(occ.keys.tolist())}
    fdir = layout.root / "features" / "phase3" / "followup"
    out: dict[str, Any] = {
        "n_pairs": len(sample),
        "n_jobs": len(jobs),
        "detect_seconds": round(detect_s, 1),
        "families": {},
    }
    for fam in FAMILIES:
        ff = fz["folds"][fam]
        link = ev.LogisticModel.from_dict(ff["heuristic_link"])
        models = {m: ev.LogisticModel.from_dict(d) for m, d in ff["models"].items()}
        clean = ev.Group(
            fdir, "lfw_clean", ("emb_ada", "emb_arc", "norm_ada", "norm_arc", "z", f"ve_{fam[:2]}")
        )
        fam_jobs = [j for j in jobs if j["family"] == fam]
        fails = [j for j in fam_jobs if not by_key[j["key"]]["ok"]]
        pairs_ok, rows = [], {}
        for j in fam_jobs:
            r = by_key[j["key"]]
            if not r["ok"]:
                continue
            p = next(q for q in sample if q.pair_id == j["pair_id"])
            pid = f"{p.pair_id}|{j['level']}"
            pairs_ok.append(p.model_copy(update={"pair_id": pid}))
            rows[pid] = {
                "stratum": "one",
                "a": None,
                "b": {
                    "family": fam,
                    "severity_level": j["level"],
                    "seed": j["seed"],
                    "visibility_ratio": r["vis_crop_level"],
                },
            }
        rec: dict[str, Any] = {
            "n_composites": len(fam_jobs),
            "detection_failures": len(fails),
            "detection_failure_rate_by_level": {
                lv: float(
                    np.mean([not by_key[j["key"]]["ok"] for j in fam_jobs if j["level"] == lv])
                )
                for lv in LEVELS
            },
            "detection_failures_genuine": sum(
                1
                for j in fails
                if next(q for q in sample if q.pair_id == j["pair_id"]).label == "genuine"
            ),
            "landmark_disp_mean_px": float(
                np.mean(
                    [
                        by_key[j["key"]]["landmark_disp_mean_px"]
                        for j in fam_jobs
                        if by_key[j["key"]]["ok"]
                    ]
                )
            )
            if pairs_ok
            else None,
            "landmark_disp_p95_of_max_px": float(
                np.quantile(
                    [
                        by_key[j["key"]]["landmark_disp_max_px"]
                        for j in fam_jobs
                        if by_key[j["key"]]["ok"]
                    ],
                    0.95,
                )
            )
            if pairs_ok
            else None,
        }
        if pairs_ok:
            t = ev.build_table(pairs_ok, rows, clean, occ, fam, link, ident)
            sc = {m: mdl.predict_proba(ev.design(t, m, cv_def)[0]) for m, mdl in models.items()}
            rec["detected_pairs"] = ev.evaluate_occluded(t, sc, ff, cv_def)
            rec["learned_visibility_global_mae_vs_realigned_gt"] = float(
                np.mean(np.abs(t["ve_vmin"] - t["gt_vmin"]))
            )
            # failures handled as abstention vs forced non-match (genuine side; impostor
            # failures can only lower FMR, so they are counted as abstentions in both variants)
            n_fail_g = int(rec["detection_failures_genuine"])
            for key in (
                "A4_predvis|primary|v2",
                "A3_quality|primary|v2",
                "A2_global_M0|primary|v2",
            ):
                arm = ff["arms"][key]
                d = ev.decide(
                    arm, sc[arm["model"]], ev.strata_labels(arm["strata"], t, ff["edges"], cv_def)
                )
                g = t["label"] == 1
                acc_g = int(((d != ev.ABSTAIN) & g).sum())
                fnm = int(((d == ev.NON_MATCH) & g).sum())
                n_g = int(g.sum()) + n_fail_g
                rec.setdefault("failure_handling", {})[key] = {
                    "genuine_total_incl_failures": n_g,
                    "as_abstain": {
                        "coverage_genuine": acc_g / max(n_g, 1),
                        "sfnmr": fnm / max(acc_g, 1),
                    },
                    "as_forced_nonmatch": {
                        "coverage_genuine": (acc_g + n_fail_g) / max(n_g, 1),
                        "sfnmr": (fnm + n_fail_g) / max(acc_g + n_fail_g, 1),
                    },
                }
        out["families"][fam] = rec
        print(fam, {k: rec[k] for k in ("detection_failures", "landmark_disp_mean_px")}, flush=True)
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "EXPLORATORY detector-inclusive study on LFW View 2 (300 pairs; synthetic occluders "
            "composited into real photographs; frozen Phase 3 rules)"
        ),
        "run_id": run_id,
        "frozen_sha256": fz["sha256"],
        "results": out,
        "provenance": prov,
    }
    dest = (
        (layout.private_reports / "phase3_detector_inclusive_pilot.json")
        if args.pilot
        else (ROOT / "reports" / "experiments" / "phase3_lfw_detector_inclusive.json")
    )
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("written", dest.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
