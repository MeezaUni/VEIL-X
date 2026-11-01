# ruff: noqa: E501  (diagnostic report script: long metric-name dict literals)
"""Phase 4A: root-cause analysis of visibility overestimation (DEVELOPMENT data only).

Everything here uses **p3_val** identities: images the visibility estimator (VE) never trained on,
whose labels were only used for early stopping. No p3_test, Phase 2 test or LFW data is read.
Held-out families are rendered fresh, with seeds salted "p4-diag".

Experiments:
E1  reproduction: for each fold F, the fold-F VE on renders of its held-out family F vs its seen
    families. Reports global bias / MAE / IoU by level.
E2  confidence: predicted P(visible) on truly occluded face cells (mean, share > 0.9, > 0.5) and
    on truly visible cells; logit entropy; is the error confident?
E3  spatial: occluded-cell error by face region and by zone. "Prior" zones are the eye band and
    lower face, where the seen physical families put occluders. "Novel" zones are the rest of
    the face.
E4  appearance: F4 error vs patch colour (dark / bright / saturated / skin-like) and texture.
E5  controlled probes: a fixed rectangle on fixed positions with fixed fills (black, white,
    mid-grey, mean skin, saturated R/G/B, uniform noise, another identity's face patch), scored
    by every fold model whose seen families include none of F4. Isolates "position" vs
    "appearance" priors.
E6  preprocessing: VE input statistics (per-channel mean/sd after normalisation) of clean,
    seen-family and F4 renders; an identical pipeline is confirmed by code path.
E7  lighting semantics: on p3_val feature shards of folds where F5 is SEEN (so the renders
    exist with embeddings), genuine-score drop (clean minus occluded cosine, AdaFace) vs
    ground-truth visible fraction, per family. Does "visibility" mean the same for lighting as
    for physical occluders?

    .venv-gpu\\Scripts\\python.exe scripts\\phase4_diagnose.py [--n 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb, pixel_sha256
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.embeddings.store import read_array_shard
from veilx.occlusion.engine import apply_occlusion
from veilx.provenance import collect_provenance
from veilx.seeding import derive_seed
from veilx.vis.metrics import all_scores
from veilx.vis.regions import face_cells, pool, region_cells, visibility_target

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
LEVELS = ("low", "medium", "high")
SEED = 20261101


def _ve(layout: Any, fold: str) -> Any:
    from safetensors.torch import load_file

    from veilx.vis.model import VisNet

    m = VisNet().cuda().eval()
    m.load_state_dict(
        load_file(str(layout.root / "models/phase3/ve" / fold / "best.safetensors"), device="cuda")
    )
    return m


def _predict(model: Any, imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P(visible), logits) as float64 (N, 28, 28); fp32 inference."""
    import torch

    from veilx.vis.model import to_input

    out = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 256):
            out.append(model(torch.from_numpy(to_input(imgs[i : i + 256])).cuda()).float().cpu())
    logit = torch.cat(out).numpy().astype(np.float64)
    return 1 / (1 + np.exp(-logit)), logit


def _render(
    imgs: list[np.ndarray], fam: str, lvl: str, tag: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    out, gts, params = [], [], []
    for i, im in enumerate(imgs):
        res = apply_occlusion(
            im, fam, lvl, derive_seed(SEED, tag, fam, lvl, i), source_pixel_sha256=pixel_sha256(im)
        )  # type: ignore[arg-type]
        out.append(res.image)
        gts.append(visibility_target(res.mask))
        params.append({"mask": res.mask, **res.spec.geometry})
    return np.stack(out), np.stack(gts), params


def e1_e2_e3(layout: Any, imgs: list[np.ndarray]) -> dict[str, Any]:
    fc = face_cells()
    regions = region_cells()
    # prior zones: eye band (rows of both eye regions dilated) + lower face (nose/mouth/lower)
    eye = regions["left_eye"] | regions["right_eye"]
    rows = np.where(eye.any(1))[0]
    band = np.zeros_like(fc)
    band[max(rows.min() - 2, 0) : rows.max() + 3, :] = True
    lower = regions["mouth"] | regions["lower_face"] | regions["nose"]
    prior_zone = (band | lower) & fc
    novel_zone = fc & ~prior_zone
    out: dict[str, Any] = {
        "prior_zone_cells": int(prior_zone.sum()),
        "novel_zone_cells": int(novel_zone.sum()),
    }
    for fold in FAMILIES:
        model = _ve(layout, fold)
        rec: dict[str, Any] = {}
        for fam in FAMILIES:
            for lvl in LEVELS:
                occ, gt, _ = _render(imgs, fam, lvl, "e1")
                p, logit = _predict(model, occ)
                s = all_scores(p, gt)
                occl = (gt < 0.5) & fc
                vis = (gt >= 0.5) & fc
                po, pv = p[occl], p[vis]
                ent = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
                cell = {
                    "iou": s["iou"],
                    "global_mae": s["global_mae"],
                    "global_bias": s["global_bias"],
                    "occluded_cells_mean_p_visible": float(po.mean()) if po.size else None,
                    "occluded_cells_share_p_gt_0.9": float((po > 0.9).mean()) if po.size else None,
                    "occluded_cells_share_p_gt_0.5": float((po > 0.5).mean()) if po.size else None,
                    "visible_cells_mean_p_visible": float(pv.mean()) if pv.size else None,
                    "mean_entropy_occluded_cells": float(ent[occl].mean()) if po.size else None,
                    "occluded_cells_mean_abs_logit": float(np.abs(logit[occl]).mean())
                    if po.size
                    else None,
                }
                for zname, z in (("prior_zone", prior_zone), ("novel_zone", novel_zone)):
                    oz = (gt < 0.5) & z
                    cell[f"{zname}_occluded_share_p_gt_0.5"] = (
                        float((p[oz] > 0.5).mean()) if oz.any() else None
                    )
                    cell[f"{zname}_share_of_occluded_cells"] = float(oz.sum() / max(occl.sum(), 1))
                rec[f"{fam}|{lvl}"] = cell
        out[fold] = rec
        f4 = rec.get(f"{fold}|high", {})
        print(
            fold,
            "held-out high:",
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in f4.items()
                if k
                in (
                    "iou",
                    "global_bias",
                    "occluded_cells_share_p_gt_0.5",
                    "prior_zone_occluded_share_p_gt_0.5",
                    "novel_zone_occluded_share_p_gt_0.5",
                )
            },
            flush=True,
        )
        del model
    return out


def e4_patch_appearance(layout: Any, imgs: list[np.ndarray]) -> dict[str, Any]:
    """F4 renders only, fold-F4 model: miss rate of occluded cells by patch appearance."""
    model = _ve(layout, "F4_random_patch")
    occ, gt, params = _render(imgs, "F4_random_patch", "high", "e4")
    p, _ = _predict(model, occ)
    fc = face_cells()
    groups: dict[str, list[float]] = {}
    for i in range(len(imgs)):
        m = params[i]["mask"]
        px = occ[i][m].astype(np.float64)
        if px.size == 0:
            continue
        mean = px.mean(0)
        sat = float(px.max(1).mean() - px.min(1).mean())
        lum = float(mean.mean())
        sd = float(px.std(0).mean())
        if lum < 60:
            kind = "dark"
        elif lum > 200:
            kind = "bright"
        elif sat > 90:
            kind = "saturated"
        else:
            kind = "mid"
        tex = "textured" if sd > 15 else "flat"
        occl = (gt[i] < 0.5) & fc
        if occl.any():
            miss = float((p[i][occl] > 0.5).mean())
            groups.setdefault(kind, []).append(miss)
            groups.setdefault(tex, []).append(miss)
    return {k: {"n_images": len(v), "mean_miss_rate": float(np.mean(v))} for k, v in groups.items()}


def e5_probes(layout: Any, imgs: list[np.ndarray], donors: list[np.ndarray]) -> dict[str, Any]:
    """Fixed 24x24 rectangle at fixed positions with fixed fills; fold-F4 model (never saw F4)."""
    rng = np.random.default_rng(SEED)
    positions = {
        "forehead": (16, 44),
        "left_cheek": (60, 22),
        "eye_band": (44, 30),
        "mouth": (80, 44),
        "nose": (58, 44),
    }
    skin = np.mean([im[40:90, 30:80].reshape(-1, 3).mean(0) for im in imgs[:200]], 0)
    fills: dict[str, Any] = {
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "mid_grey": np.array([128, 128, 128]),
        "mean_skin": skin,
        "red": np.array([230, 30, 30]),
        "green": np.array([30, 200, 30]),
        "blue": np.array([30, 30, 230]),
        "uniform_noise": None,
        "other_face_patch": None,
    }
    out: dict[str, Any] = {}
    for fold in FAMILIES:
        model = _ve(layout, fold)
        res: dict[str, Any] = {}
        for pname, (y, x) in positions.items():
            for fname, val in fills.items():
                batch = []
                for i, im in enumerate(imgs[:400]):
                    a = im.copy()
                    if fname == "uniform_noise":
                        a[y : y + 24, x : x + 24] = rng.integers(0, 256, (24, 24, 3))
                    elif fname == "other_face_patch":
                        a[y : y + 24, x : x + 24] = donors[i % len(donors)][y : y + 24, x : x + 24]
                    else:
                        a[y : y + 24, x : x + 24] = val
                    batch.append(a)
                p, _ = _predict(model, np.stack(batch))
                m = np.zeros((112, 112), bool)
                m[y : y + 24, x : x + 24] = True
                cells = pool(m) > 0.99
                res[f"{pname}|{fname}"] = float(p[:, cells].mean())
        out[fold] = res
        del model
    return {
        "note": "mean predicted P(visible) inside a fully covered 24x24 probe (truth: 0)",
        "by_fold": out,
    }


def e6_inputs(imgs: list[np.ndarray]) -> dict[str, Any]:
    from veilx.vis.model import to_input

    stats = {}
    for name, fam in (
        ("clean", None),
        ("F1_mask", "F1_mask"),
        ("F2_sunglasses", "F2_sunglasses"),
        ("F4_random_patch", "F4_random_patch"),
    ):
        arr = np.stack(imgs[:500]) if fam is None else _render(imgs[:500], fam, "high", "e6")[0]
        x = to_input(arr)
        stats[name] = {
            "channel_mean": x.mean((0, 2, 3)).round(4).tolist(),
            "channel_sd": x.std((0, 2, 3)).round(4).tolist(),
            "share_saturated_pixels": float(((arr <= 3) | (arr >= 252)).mean()),
        }
    return stats


def e7_lighting(layout: Any) -> dict[str, Any]:
    """p3_val feature shards: genuine cosine drop vs GT visible fraction, per family (seen folds)."""
    feat = layout.root / "features" / "phase3"
    p3 = layout.manifests / "phase3"
    pairs = {p.pair_id: p for p in read_jsonl(p3 / "pairs" / "p3_val_pairs.jsonl", PairRecord)}
    import gzip

    ck, ca, _ = _load_group(feat, "p3_clean", ("emb_ada",))
    cidx = {str(k): i for i, k in enumerate(ck.tolist())}
    out: dict[str, Any] = {}
    # F2-F5 are seen families in fold F1's val plan; F1 comes from fold F2's val plan
    for fold, fams in (
        ("F1_mask", ("F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")),
        ("F2_sunglasses", ("F1_mask",)),
    ):
        ok, oa, _ = _load_group(feat, f"{fold}__val__p3_val_pairs", ("emb_ada", "vis"))
        oidx = {str(k): i for i, k in enumerate(ok.tolist())}
        with gzip.open(
            p3 / "occlusion" / f"{fold}__val__p3_val_pairs.jsonl.gz", "rt", encoding="utf-8"
        ) as fh:
            rows = [json.loads(x) for x in fh]
        from veilx.embeddings.jobs import occluded_key

        acc: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            p = pairs[r["pair_id"]]
            if p.label != "genuine" or r["a"] is not None or r["b"] is None:
                continue
            s = r["b"]
            if s["family"] not in fams:
                continue
            key = occluded_key(p.image_b, s["family"], s["severity_level"], int(s["seed"]))
            ea = ca["emb_ada"][cidx[p.image_a]].astype(np.float32)
            eb_c = ca["emb_ada"][cidx[p.image_b]].astype(np.float32)
            eb_o = oa["emb_ada"][oidx[key]].astype(np.float32)
            ea, eb_c, eb_o = (v / np.linalg.norm(v) for v in (ea, eb_c, eb_o))
            acc.setdefault(s["family"], []).append(
                (float(oa["vis"][oidx[key]]), float(ea @ eb_c - ea @ eb_o))
            )
        for fam, v in acc.items():
            arr = np.array(v)
            bins = np.digitize(arr[:, 0], [0.4, 0.7, 0.9])
            out[fam] = {
                "n_genuine_pairs": len(arr),
                "corr_visibility_vs_score_drop": float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]),
                "mean_score_drop": float(arr[:, 1].mean()),
                "mean_gt_visibility": float(arr[:, 0].mean()),
                "score_drop_by_visibility_bin": {
                    b: float(arr[bins == i, 1].mean())
                    for i, b in enumerate(("<0.4", "0.4-0.7", "0.7-0.9", ">=0.9"))
                    if (bins == i).any()
                },
            }
    return out


def _load_group(
    feat: Path, name: str, cols: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray], None]:
    keys, arr = [], {c: [] for c in cols}  # type: ignore[var-annotated]
    for p in sorted(feat.glob(f"{name}.part*.npz")):
        k, a, _ = read_array_shard(p)
        keys.append(k)
        for c in cols:
            arr[c].append(a[c].copy())
        del a
    return np.concatenate(keys), {c: np.concatenate(v) for c, v in arr.items()}, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    layout = resolve_data_root()
    sub = json.loads((layout.manifests / "phase3" / "subsplits.json").read_text("utf-8"))
    val_ids = sorted(k for k, v in sub.items() if v == "p3_val")
    rng = np.random.default_rng(SEED)
    pick = set(rng.choice(len(val_ids), args.n + 200, replace=False).tolist())
    chosen = [val_ids[i] for i in sorted(pick)]
    recs = {
        r.image_id: r.relpath
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
        if r.image_id in set(chosen)
    }
    imgs = [load_rgb(layout.extracted / recs[i]) for i in chosen[: args.n]]
    donors = [load_rgb(layout.extracted / recs[i]) for i in chosen[args.n :]]
    report: dict[str, Any] = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 4A root-cause diagnostics on DEVELOPMENT data (p3_val identities; fresh renders; no test data)",
        "n_images": len(imgs),
    }
    report["e1_e2_e3"] = e1_e2_e3(layout, imgs)
    report["e4_f4_appearance"] = e4_patch_appearance(layout, imgs)
    print("E4", report["e4_f4_appearance"], flush=True)
    report["e5_probes"] = e5_probes(layout, imgs, donors)
    report["e6_inputs"] = e6_inputs(imgs)
    report["e7_lighting_semantics"] = e7_lighting(layout)
    print("E7", json.dumps(report["e7_lighting_semantics"])[:600], flush=True)
    report["provenance"] = collect_provenance(ROOT, argv=sys.argv)
    with (ROOT / "reports" / "experiments" / "phase4_diagnostics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1)
    print("written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
