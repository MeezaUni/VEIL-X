"""Phase 2 GPU embedding extraction for the frozen backbones (inference only).

Groups (each written as one atomic shard per backbone under
``<data_root>/embeddings/<backbone_id>/<spec_fingerprint>-<precision>/``):

* ``clean``                       every image referenced by the evaluation pair sets
                                  (val, val_independent, cal_independent, test_balanced,
                                  test_far); train images are not needed in Phase 2;
* ``<fold>__<role>__<pairset>``   every occluded side of a leave-one-family-out plan;
* ``stress__test_balanced``       combined occlusions (exploratory).

Images are decoded and occlusions re-rendered in a CPU process pool (bounded queue) and every
backbone embeds each batch in the same pass. A shard is skipped only if its meta matches the
current backbone fingerprint, weight-tensor fingerprint and job hash (cache invalidation).
Nothing is silently dropped: any decode/render/embedding failure aborts the group with an error.

    .venv-gpu\\Scripts\\python.exe scripts\\extract_embeddings.py --pilot 2048
    .venv-gpu\\Scripts\\python.exe scripts\\extract_embeddings.py              (full)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from veilx.config import load_protocol
from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import DataLayout, resolve_data_root
from veilx.embeddings.jobs import Item, clean_items, occluded_items, render, stress_items
from veilx.embeddings.store import is_complete, write_shard
from veilx.models.extractor import FrozenExtractor
from veilx.models.spec import BACKBONES
from veilx.occlusion.engine import GENERATOR_VERSION
from veilx.provenance import collect_provenance

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
EVAL_PAIR_SETS = ("val", "val_independent", "cal_independent", "test_balanced", "test_far")

# ----------------------------------------------------------------- CPU workers

_W: dict[str, Any] = {}


def _init_worker(extracted: str, recs: dict[str, tuple[str, str]]) -> None:
    _W["root"], _W["recs"] = Path(extracted), recs
    _W["cache"] = {}


def _render_batch(items: list[Item]) -> np.ndarray:
    out = np.empty((len(items), 112, 112, 3), dtype=np.uint8)
    for k, it in enumerate(items):
        relpath, psha = _W["recs"][it.image_id]
        img = load_rgb(_W["root"] / relpath)
        if img.shape != (112, 112, 3):
            raise ValueError(f"{it.image_id}: unexpected shape {img.shape}")
        out[k] = render(it, img, psha)
    return out


# ----------------------------------------------------------------- groups


def _pairs(layout: DataLayout, tag: str) -> dict[str, PairRecord]:
    return {
        p.pair_id: p for p in read_jsonl(layout.manifests / "pairs" / f"{tag}.jsonl", PairRecord)
    }


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def build_groups(layout: DataLayout, kinds: set[str]) -> list[tuple[str, list[Item]]]:
    groups: list[tuple[str, list[Item]]] = []
    cache: dict[str, dict[str, PairRecord]] = {}

    def pairs(tag: str) -> dict[str, PairRecord]:
        if tag not in cache:
            cache[tag] = _pairs(layout, tag)
        return cache[tag]

    if "clean" in kinds:
        ids: set[str] = set()
        for tag in EVAL_PAIR_SETS:
            for p in pairs(tag).values():
                ids.update((p.image_a, p.image_b))
        groups.append(("clean", clean_items(ids)))
    occl = layout.manifests / "occlusion"
    if "lofo" in kinds:
        for f in sorted(occl.glob("F*__*__*.jsonl.gz")):
            name = f.name.removesuffix(".jsonl.gz")
            groups.append((name, occluded_items(_rows(f), pairs(name.split("__")[2]))))
    if "stress" in kinds:
        f = occl / "stress__test_balanced.jsonl.gz"
        groups.append(("stress__test_balanced", stress_items(_rows(f), pairs("test_balanced"))))
    return groups


def _job_sha(items: list[Item]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.key.encode())
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------- main loop


def run_group(
    name: str,
    items: list[Item],
    extractors: list[FrozenExtractor],
    out_dirs: dict[str, Path],
    pool: ProcessPoolExecutor,
    batch: int,
    max_inflight: int,
) -> dict[str, Any]:
    job = _job_sha(items)
    need = [
        ex
        for ex in extractors
        if not is_complete(
            out_dirs[ex.backbone_id] / f"{name}.npz",
            {
                "job_sha256": job,
                "backbone_fingerprint": ex.fingerprint,
                "weights_tensor_fingerprint": ex.weights_fingerprint,
            },
        )
    ]
    if not need:
        return {"group": name, "n": len(items), "status": "cached"}
    chunks = [items[i : i + batch] for i in range(0, len(items), batch)]
    emb = {ex.backbone_id: np.empty((len(items), ex.embedding_dim), np.float32) for ex in need}
    nrm = {ex.backbone_id: np.empty(len(items), np.float32) for ex in need}
    gpu_s = {ex.backbone_id: 0.0 for ex in need}
    cuda = any(ex.device.type == "cuda" for ex in need)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    inflight: deque[Future[np.ndarray]] = deque()
    it = iter(chunks)
    pos = 0
    last = t0
    while True:
        while len(inflight) < max_inflight:
            nxt = next(it, None)
            if nxt is None:
                break
            inflight.append(pool.submit(_render_batch, nxt))
        if not inflight:
            break
        imgs = inflight.popleft().result()  # raises on any render/decode failure
        n = len(imgs)
        for ex in need:
            tg = time.perf_counter()
            b = ex.embed(imgs)
            if cuda:
                torch.cuda.synchronize()
            gpu_s[ex.backbone_id] += time.perf_counter() - tg
            emb[ex.backbone_id][pos : pos + n] = b.embeddings
            nrm[ex.backbone_id][pos : pos + n] = b.feature_norms
        pos += n
        if time.perf_counter() - last > 30:
            last = time.perf_counter()
            rate = pos / (last - t0)
            print(f"  {name}: {pos:,}/{len(items):,} ({rate:,.0f} img/s)", flush=True)
    assert pos == len(items)
    secs = time.perf_counter() - t0
    keys = [i.key for i in items]
    for ex in need:
        write_shard(
            out_dirs[ex.backbone_id] / f"{name}.npz",
            keys,
            emb[ex.backbone_id],
            nrm[ex.backbone_id],
            {
                "group": name,
                "job_sha256": job,
                "backbone_id": ex.backbone_id,
                "backbone_fingerprint": ex.fingerprint,
                "weights_tensor_fingerprint": ex.weights_fingerprint,
                "preprocess_fingerprint": ex.spec.preprocess.fingerprint(),
                "precision": ex.precision,
                "occlusion_generator_version": GENERATOR_VERSION,
                "protocol_version": PROTO.protocol_version,
                "created_utc": datetime.now(tz=UTC).isoformat(),
            },
        )
    return {
        "group": name,
        "n": len(items),
        "status": "extracted",
        "backbones": [ex.backbone_id for ex in need],
        "seconds": round(secs, 1),
        "images_per_s_end_to_end": round(len(items) / secs, 1),
        "gpu_seconds": {k: round(v, 1) for k, v in gpu_s.items()},
        "gpu_images_per_s": {k: round(len(items) / v, 1) for k, v in gpu_s.items() if v > 0},
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--kinds", default="clean,lofo,stress")
    ap.add_argument("--groups", default="", help="comma list of group names to restrict to")
    ap.add_argument("--pilot", type=int, default=0, help="first N items of 3 groups, separate dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; pass --device cpu explicitly for the CPU fallback", flush=True)
        return 1
    layout = resolve_data_root()
    recs = {
        r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
    }
    groups = build_groups(layout, set(args.kinds.split(",")))
    if args.groups:
        wanted = set(args.groups.split(","))
        groups = [g for g in groups if g[0] in wanted]
    if args.pilot:
        picked: list[tuple[str, list[Item]]] = []
        for prefix in ("clean", "F1_mask__test_ood", "stress"):
            g = next((g for g in groups if g[0].startswith(prefix)), None)
            if g:
                picked.append((g[0], g[1][: args.pilot]))
        groups = picked
    total = sum(len(g[1]) for g in groups)
    print(f"{len(groups)} groups, {total:,} items", flush=True)
    extractors = [
        FrozenExtractor(
            BACKBONES[b],
            layout.models / BACKBONES[b].hf_repo.replace("/", "__") / BACKBONES[b].revision,
            device=args.device,
            precision=args.precision,
        )
        for b in args.backbones.split(",")
    ]
    sub = "pilot" if args.pilot else ""
    out_dirs = {
        ex.backbone_id: layout.embeddings / ex.backbone_id / ex.fingerprint / sub
        for ex in extractors
    }
    log: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(layout.extracted), recs),
    ) as pool:
        for name, items in groups:
            res = run_group(name, items, extractors, out_dirs, pool, args.batch, 2 * args.workers)
            print(json.dumps(res), flush=True)
            log.append(res)
    summary = {
        "evidence_type": "real_measured_model_result",
        "scope": "embedding extraction run (throughput/latency/memory; no accuracy numbers)",
        "pilot": bool(args.pilot),
        "precision": args.precision,
        "batch": args.batch,
        "device": args.device,
        "cpu_workers": args.workers,
        "backbones": {ex.backbone_id: ex.fingerprint for ex in extractors},
        "n_groups": len(groups),
        "n_items": total,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "groups": log,
        "accelerator": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    private = layout.embeddings / (
        "extraction_log_pilot.json" if args.pilot else "extraction_log.json"
    )
    private.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    public = (
        ROOT
        / "reports"
        / "experiments"
        / ("extraction_pilot.json" if args.pilot else "extraction_run.json")
    )
    public.parent.mkdir(parents=True, exist_ok=True)
    with public.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done in {summary['wall_seconds']}s; wrote {public.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
