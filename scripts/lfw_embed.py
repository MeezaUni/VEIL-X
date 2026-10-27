"""Embed LFW crops (both preprocessing paths) and the derived synthetic-occlusion renders.

Groups (one atomic shard per backbone under ``<data_root>/embeddings/<backbone>/<fp>/``):

* ``lfw_det_clean``    detector-aligned crops of every image with a successful detection;
* ``lfw_fixed_clean``  fixed-transform crops of every image (no detector);
* ``lfw_det_occl``     derived occlusions on detector-aligned crops of the official View-2 pairs
                       (and View-1 DevTest pairs, used only to rehearse the evaluation): for every
                       pair, family F1-F5 and level low/medium/high, side b alone ("one") and both
                       sides ("both"); seeds derived from (global seed, "lfw", pair id, family,
                       level, tag). The plan is written before any LFW score is computed.

Derived occlusions are SYNTHETIC overlays on real photographs - not physical occlusions.

    .venv-gpu\\Scripts\\python.exe scripts\\lfw_embed.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb, pixel_sha256
from veilx.data.manifest import PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import is_complete, write_shard
from veilx.occlusion.engine import GENERATOR_VERSION, apply_occlusion
from veilx.occlusion.families import LEVELS
from veilx.provenance import collect_provenance
from veilx.seeding import derive_seed

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
SEED = PROTO.seeds.global_seed


def crop_name(image_id: str) -> str:
    return image_id.replace(":", "__") + ".png"


def build_plan(pairs: list[PairRecord], ok: set[str]) -> list[dict[str, Any]]:
    """Deterministic plan rows: pair x family x level x {one, both}; only pairs whose two images
    were detected (the derived-occlusion study runs on detector-aligned crops)."""
    rows = []
    for p in sorted(pairs, key=lambda q: q.pair_id):
        if p.image_a not in ok or p.image_b not in ok:
            continue
        for fam in FAMILIES:
            for lvl in LEVELS:
                for mode in ("one", "both"):
                    sb = derive_seed(SEED, "lfw", p.pair_id, fam, lvl, mode, "b")
                    sa = derive_seed(SEED, "lfw", p.pair_id, fam, lvl, mode, "a")
                    rows.append(
                        {
                            "pair_id": p.pair_id,
                            "family": fam,
                            "level": lvl,
                            "mode": mode,
                            "a": None if mode == "one" else {"image_id": p.image_a, "seed": sa},
                            "b": {"image_id": p.image_b, "seed": sb},
                        }
                    )
    return rows


def _embed_all(extractors: list[Any], images: list[np.ndarray], batch: int = 256) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ex in extractors:
        embs, norms = [], []
        for i in range(0, len(images), batch):
            b = ex.embed(np.stack(images[i : i + batch]))
            embs.append(b.embeddings)
            norms.append(b.feature_norms)
        out[ex.backbone_id] = (np.concatenate(embs), np.concatenate(norms))
    return out


def main() -> int:
    import argparse

    from veilx.models.extractor import FrozenExtractor
    from veilx.models.spec import BACKBONES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0, help="first N occluded items, pilot dir only")
    ap.add_argument("--part-size", type=int, default=32_768)
    args = ap.parse_args()
    layout = resolve_data_root()
    lfw = layout.manifests / "lfw"
    dets = [json.loads(line) for line in (lfw / "detections.jsonl").open(encoding="utf-8")]
    ok = {d["image_id"] for d in dets if d["ok"]}
    all_ids = sorted(d["image_id"] for d in dets)
    view2 = list(read_jsonl(lfw / "view2_pairs.jsonl", PairRecord))
    devtest = list(read_jsonl(lfw / "view1_devtest_pairs.jsonl", PairRecord))
    plan = build_plan(view2, ok) + build_plan(devtest, ok)
    plan_path = lfw / "occlusion_plan.jsonl.gz"
    with gzip.open(plan_path, "wt", encoding="utf-8", newline="\n") as fh:
        for r in plan:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    extractors = [
        FrozenExtractor(
            s,
            layout.models / s.hf_repo.replace("/", "__") / s.revision,
            device="cuda",
            precision="fp16",
        )
        for s in BACKBONES.values()
    ]
    sub = "pilot_lfw" if args.pilot else ""
    out_dirs = {
        ex.backbone_id: layout.embeddings / ex.backbone_id / ex.fingerprint / sub
        for ex in extractors
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    aligned = layout.extracted / "lfw_aligned"
    log: dict[str, Any] = {}

    def run(name: str, keys: list[str], render: Any) -> None:
        meta_expect = {"group": name, "n": len(keys)}
        if all(
            is_complete(out_dirs[ex.backbone_id] / f"{name}.npz", meta_expect) for ex in extractors
        ):
            log[name] = {"n": len(keys), "status": "cached"}
            return
        t0 = time.perf_counter()
        emb: dict[str, list[Any]] = {ex.backbone_id: [[], []] for ex in extractors}
        chunk = 8192
        for i in range(0, len(keys), chunk):
            imgs = [render(k) for k in keys[i : i + chunk]]
            for bid, (e, n) in _embed_all(extractors, imgs).items():
                emb[bid][0].append(e)
                emb[bid][1].append(n)
        for ex in extractors:
            write_shard(
                out_dirs[ex.backbone_id] / f"{name}.npz",
                keys,
                np.concatenate(emb[ex.backbone_id][0]),
                np.concatenate(emb[ex.backbone_id][1]),
                {
                    "group": name,
                    "backbone_id": ex.backbone_id,
                    "backbone_fingerprint": ex.fingerprint,
                    "weights_tensor_fingerprint": ex.weights_fingerprint,
                    "occlusion_generator_version": GENERATOR_VERSION,
                    "dataset": "kaggle_jessicali9530_lfw",
                },
            )
        secs = time.perf_counter() - t0
        log[name] = {
            "n": len(keys),
            "seconds": round(secs, 1),
            "images_per_s": round(len(keys) / secs, 1),
        }
        print(name, log[name], flush=True)

    def run_parts(name: str, keys: list[str], render: Any, vis: dict[str, Any], part: int) -> None:
        """Large group: atomic part shards (``<name>.partNNNN.npz``) plus a per-part visibility
        record; completed parts are skipped on restart; nothing larger than one part is held
        in memory."""
        t0 = time.perf_counter()
        computed = 0
        parts = [keys[i : i + part] for i in range(0, len(keys), part)]
        for k, pk in enumerate(parts):
            pname = f"{name}.part{k:04d}"
            vpath = out_dirs[extractors[0].backbone_id] / f"{pname}.vis.json"
            expect = {"group": pname, "n": len(pk)}
            done = all(
                is_complete(out_dirs[ex.backbone_id] / f"{pname}.npz", expect) for ex in extractors
            )
            if done and vpath.exists():
                continue
            pvis: dict[str, Any] = {}
            imgs = []
            for key in pk:
                img, spec = render(key)
                imgs.append(img)
                pvis[key] = spec
            for bid, (e, n) in _embed_all(extractors, imgs).items():
                write_shard(
                    out_dirs[bid] / f"{pname}.npz",
                    pk,
                    e,
                    n,
                    {
                        "group": pname,
                        "parent_group": name,
                        "backbone_id": bid,
                        "backbone_fingerprint": next(
                            x.fingerprint for x in extractors if x.backbone_id == bid
                        ),
                        "occlusion_generator_version": GENERATOR_VERSION,
                        "dataset": "kaggle_jessicali9530_lfw",
                    },
                )
            vpath.write_text(json.dumps(pvis), encoding="utf-8")
            computed += len(pk)
            del imgs
            print(f"  {pname}: {len(pk)} items", flush=True)
        for k in range(len(parts)):
            vis |= json.loads(
                (out_dirs[extractors[0].backbone_id] / f"{name}.part{k:04d}.vis.json").read_text(
                    "utf-8"
                )
            )
        secs = time.perf_counter() - t0
        log[name] = {
            "n": len(keys),
            "n_parts": len(parts),
            "computed_this_session": computed,
            "seconds": round(secs, 1),
            "images_per_s": round(computed / secs, 1) if computed else None,
        }
        print(name, log[name], flush=True)

    det_ids = sorted(ok)
    if not args.pilot:
        run("lfw_det_clean", det_ids, lambda k: load_rgb(aligned / "det" / crop_name(k)))
        run("lfw_fixed_clean", all_ids, lambda k: load_rgb(aligned / "fixed" / crop_name(k)))
    cache: dict[str, tuple[np.ndarray, str]] = {}

    def crop(image_id: str) -> tuple[np.ndarray, str]:
        if image_id not in cache:
            img = load_rgb(aligned / "det" / crop_name(image_id))
            cache[image_id] = (img, pixel_sha256(img))
        return cache[image_id]

    items: dict[str, tuple[str, str, str, int]] = {}
    for r in plan:
        for side in ("a", "b"):
            s = r[side]
            if s is not None:
                items[occluded_key(s["image_id"], r["family"], r["level"], s["seed"])] = (
                    s["image_id"],
                    r["family"],
                    r["level"],
                    s["seed"],
                )
    keys = sorted(items)
    if args.pilot:
        keys = keys[: args.pilot]

    vis: dict[str, dict[str, Any]] = {}

    def render(k: str) -> tuple[np.ndarray, dict[str, Any]]:
        image_id, fam, lvl, seed = items[k]
        img, psha = crop(image_id)
        res = apply_occlusion(img, fam, lvl, seed, source_pixel_sha256=psha)  # type: ignore[arg-type]
        spec = {
            "visibility_ratio": res.spec.visibility_ratio,
            "severity": res.spec.severity,
            "recipe_sha256": res.spec.recipe_sha256,
        }
        return res.image, spec

    run_parts("lfw_det_occl", keys, render, vis, part=args.part_size)
    if args.pilot:
        print("pilot done", json.dumps(log), flush=True)
        return 0
    (lfw / "occlusion_visibility.json").write_text(json.dumps(vis), encoding="utf-8")
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "LFW embedding extraction (throughput only)",
        "n_plan_rows": len(plan),
        "groups": log,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "lfw_extraction.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=2)
    print("done", json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
