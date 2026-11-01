# ruff: noqa: E501
"""Phase 5: parity of the application pipeline with the offline frozen evaluation (development data).

(a) DigiFace, pre-aligned path. The first N rows of the Phase 3 fold-F1 p3_val plan, clean and
    occluded sides re-rendered from their recipes, are run through ``VerificationPipeline.verify``.
    The fused score and A3 decision are compared with the offline values computed by
    ``scripts/evaluate_phase3.py`` from the Phase 3 feature store (fp16-stored embeddings).
(b) LFW, detector path. On LFW View 1 DevTrain images (development; never View 2) the app's crop
    is compared pixel-for-pixel with the offline aligned crop (``lfw_aligned/det``), and statuses
    with the offline detections.
(c) CPU vs GPU. AdaFace embeddings (fp32 CPU vs fp16 GPU) and A3 decisions on a subset of (a).

Aggregates only -> reports/experiments/phase5_parity.json.

    .venv-gpu\\Scripts\\python.exe scripts\\phase5_parity.py [--n 1000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from veilx.app.config import load_app_config
    from veilx.app.pipeline import VerificationPipeline
    from veilx.data.images import load_rgb
    from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
    from veilx.data.paths import resolve_data_root
    from veilx.embeddings.jobs import occluded_key
    from veilx.fusion.features import design
    from veilx.occlusion.engine import apply_occlusion
    from veilx.provenance import collect_provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-lfw", type=int, default=300)
    ap.add_argument("--n-cpu", type=int, default=100)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
    )
    assert spec is not None and spec.loader is not None
    ev3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev3)
    lay = resolve_data_root()
    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    fold = cfg.decision.default_fold
    pipe = VerificationPipeline(cfg, lay.root, device="cuda")
    rule = pipe.rule(fold)
    out: dict[str, Any] = {"fold": fold}

    # ---------------------------------------------------------- (a) DigiFace pre-aligned
    ctx = ev3.Ctx()
    clean, occ = ctx.clean(fold), ctx.occ(fold, "val")
    rows = ctx.plan(fold, "val")
    pairs = ctx.pairs["p3_val_pairs"][: args.n]
    f3 = (
        ctx.fz3
        if hasattr(ctx, "fz3")
        else json.loads((lay.private_reports / "phase3_frozen.json").read_text("utf-8"))
    )
    link = ev3.LogisticModel.from_dict(f3["folds"][fold]["heuristic_link"])
    t = ev3.build_table(
        pairs, {p.pair_id: rows[p.pair_id] for p in pairs}, clean, occ, fold, link, ctx.ident
    )
    m1 = ev3.LogisticModel.from_dict(f3["folds"][fold]["models"]["M1"])
    p_off = m1.predict_proba(design(t, "M1", "cv_region")[0])
    rec = f3["folds"][fold]["arms"]["A3_quality|primary|v2"]
    d_off = ev3.decide(
        rec, p_off, ev3.strata_labels("quality", t, f3["folds"][fold]["edges"], "cv_region")
    )
    rel = {r.image_id: (r.relpath, r.pixel_sha256 or r.sha256) for r in ctx.images}

    def render(img_id: str, spec_: dict[str, Any] | None) -> np.ndarray:
        rp, psha = rel[img_id]
        im = load_rgb(lay.extracted / rp)
        if spec_ is None:
            return im
        return apply_occlusion(
            im,
            spec_["family"],
            spec_["severity_level"],
            int(spec_["seed"]),
            source_pixel_sha256=psha,
        ).image

    p_app, d_app, s_diff, q_diff, keys_ok = [], [], [], [], 0
    imgs_for_cpu = []
    for k, p in enumerate(pairs):
        r = rows[p.pair_id]
        a, b = render(p.image_a, r["a"]), render(p.image_b, r["b"])
        if k < args.n_cpu:
            imgs_for_cpu.append((a, b))
        res = pipe.verify(a, b, fold=fold, pre_aligned=True, with_diagnostics=False)
        p_app.append(res.decision.fused_score)
        d_app.append(res.decision.code)
        s_diff.append(abs(res.s_adaface - t["s"][k]))  # type: ignore[operator]
        q_diff.append(abs(res.quality - t["q"][k]))  # type: ignore[operator]
        keys_ok += int(
            r["b"] is None
            or occluded_key(
                p.image_b, r["b"]["family"], r["b"]["severity_level"], int(r["b"]["seed"])
            )
            in occ.index
        )
    p_app_a, d_app_a = np.array(p_app), np.array(d_app)
    dis = np.flatnonzero(d_app_a != d_off)
    margin = []
    for i in dis:
        lo, hi = rule.rules[
            ev3.strata_labels(
                "quality", {"q": t["q"][i : i + 1]}, f3["folds"][fold]["edges"], "cv_region"
            )[0]
        ] or (np.nan, np.nan)
        margin.append(float(min(abs(p_off[i] - lo), abs(p_off[i] - hi))))
    out["digiface_pre_aligned_path"] = {
        "n_pairs": len(pairs),
        "n_occluded_pairs": int(t["occluded"].sum()),
        "render_keys_found_offline": keys_ok,
        "max_abs_cosine_diff": float(max(s_diff)),
        "max_abs_quality_diff": float(max(q_diff)),
        "max_abs_fused_score_diff": float(np.abs(p_app_a - p_off).max()),
        "mean_abs_fused_score_diff": float(np.abs(p_app_a - p_off).mean()),
        "decision_agreement": float((d_app_a == d_off).mean()),
        "n_disagreements": len(dis),
        "disagreement_distance_to_threshold_max": max(margin) if margin else None,
        "decision_counts_app": {
            lab: int((d_app_a == c).sum())
            for lab, c in (("MATCH", 1), ("NON-MATCH", 0), ("ABSTAIN", -1))
        },
        "note": "offline embeddings were stored as float16; the app computes them fresh (fp16 autocast on GPU)",
    }
    print("DigiFace", out["digiface_pre_aligned_path"], flush=True)

    # ---------------------------------------------------------- (b) LFW detector path (DevTrain)
    lfw = lay.manifests / "lfw"
    dets = {
        d["image_id"]: d
        for line in (lfw / "detections.jsonl").open(encoding="utf-8")
        if (d := json.loads(line))
    }
    devtrain = list(read_jsonl(lfw / "view1_devtrain_pairs.jsonl", PairRecord))
    ids = sorted({p.image_a for p in devtrain} | {p.image_b for p in devtrain})[: args.n_lfw]
    imgs = {r.image_id: r.relpath for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    stats = {
        "n_images": len(ids),
        "offline_ok": 0,
        "app_aligned": 0,
        "app_multiple_faces_refused": 0,
        "app_fallback": 0,
        "crop_identical": 0,
        "crop_compared": 0,
        "max_abs_pixel_diff": 0,
    }
    for i in ids:
        im = load_rgb(lay.extracted / "lfw_kaggle_v4" / imgs[i])
        prep = pipe.faces.prepare(im)
        off_ok = dets[i]["ok"]
        stats["offline_ok"] += int(off_ok)
        stats["app_aligned"] += int(prep.status == "aligned")
        stats["app_multiple_faces_refused"] += int(prep.status == "multiple_faces")
        stats["app_fallback"] += int(prep.status == "fallback_fixed_crop")
        if (
            prep.status == "multiple_faces" and off_ok
        ):  # explicit selection of the face offline chose
            from veilx.detect.scrfd import select_central

            k_central = select_central(pipe.faces.detector().detect(im), im.shape[:2])
            prep = pipe.faces.prepare(im, select=k_central)
            stats["selected_central_compared"] = stats.get("selected_central_compared", 0)
            if prep.status == "selected_face":
                ref = load_rgb(
                    lay.extracted / "lfw_aligned" / "det" / (i.replace(":", "__") + ".png")
                )
                diff = int(np.abs(ref.astype(int) - prep.crop.astype(int)).max())  # type: ignore[union-attr]
                stats["selected_central_compared"] += 1
                stats["selected_central_identical"] = stats.get(
                    "selected_central_identical", 0
                ) + int(diff == 0)
            continue
        if prep.status == "aligned" and off_ok:
            ref = load_rgb(lay.extracted / "lfw_aligned" / "det" / (i.replace(":", "__") + ".png"))
            diff = int(np.abs(ref.astype(int) - prep.crop.astype(int)).max())  # type: ignore[union-attr]
            stats["crop_compared"] += 1
            stats["crop_identical"] += int(diff == 0)
            stats["max_abs_pixel_diff"] = max(stats["max_abs_pixel_diff"], diff)
    out["lfw_detector_path_devtrain"] = stats | {
        "note": "offline Phase 2 used the central face when several were found; the app refuses unless the user selects one (multiple_faces)"
    }
    print("LFW", out["lfw_detector_path_devtrain"], flush=True)

    # ---------------------------------------------------------- (c) CPU vs GPU
    cpu = VerificationPipeline(cfg, lay.root, device="cpu")
    cos, agree, dp = [], 0, []
    for a, b in imgs_for_cpu:
        rg = pipe.verify(a, b, fold=fold, pre_aligned=True, with_diagnostics=False)
        rc = cpu.verify(a, b, fold=fold, pre_aligned=True, with_diagnostics=False)
        eg = pipe.backbone("adaface_ir101_webface4m").embed(np.stack([a, b])).embeddings
        ec = cpu.backbone("adaface_ir101_webface4m").embed(np.stack([a, b])).embeddings
        cos += [
            float(x) for x in np.einsum("ij,ij->i", eg.astype(np.float64), ec.astype(np.float64))
        ]
        agree += int(rg.decision.code == rc.decision.code)
        dp.append(abs(rg.decision.fused_score - rc.decision.fused_score))
    out["cpu_vs_gpu"] = {
        "n_pairs": len(imgs_for_cpu),
        "min_embedding_cosine_fp16gpu_vs_fp32cpu": min(cos),
        "decision_agreement": agree / max(len(imgs_for_cpu), 1),
        "max_abs_fused_score_diff": max(dp),
    }
    print("CPU/GPU", out["cpu_vs_gpu"], flush=True)
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 5 application parity with offline frozen evaluation (development data: p3_val, LFW View 1 DevTrain)",
        "results": out,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase5_parity.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
