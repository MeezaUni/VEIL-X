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
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import DataLayout, resolve_data_root
from veilx.embeddings.jobs import Item, clean_items, occluded_items, render, stress_items
from veilx.embeddings.store import is_complete, read_shard, write_shard
from veilx.models.spec import BACKBONES
from veilx.occlusion.engine import GENERATOR_VERSION
from veilx.provenance import collect_provenance

if TYPE_CHECKING:  # torch is imported lazily so spawned render workers never load it
    from veilx.models.extractor import FrozenExtractor

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


GroupFn = Callable[[], list[Item]]


def build_groups(layout: DataLayout, kinds: set[str]) -> list[tuple[str, GroupFn]]:
    """Group names with LAZY item builders: items are materialised one group at a time."""
    groups: list[tuple[str, GroupFn]] = []

    def clean() -> list[Item]:
        ids: set[str] = set()
        for tag in EVAL_PAIR_SETS:
            for p in _pairs(layout, tag).values():
                ids.update((p.image_a, p.image_b))
        return clean_items(ids)

    def lofo(f: Path, tag: str) -> GroupFn:
        return lambda: occluded_items(_rows(f), _pairs(layout, tag))

    def stress(f: Path) -> GroupFn:
        return lambda: stress_items(_rows(f), _pairs(layout, "test_balanced"))

    if "clean" in kinds:
        groups.append(("clean", clean))
    occl = layout.manifests / "occlusion"
    if "lofo" in kinds:
        # val/cal plans first so the develop stage can start while test groups extract
        order = {"val": 0, "cal": 1, "test_id": 2, "test_ood": 3}
        files = sorted(
            occl.glob("F*__*__*.jsonl.gz"), key=lambda f: (order[f.name.split("__")[1]], f.name)
        )
        for f in files:
            name = f.name.removesuffix(".jsonl.gz")
            groups.append((name, lofo(f, name.split("__")[2])))
    if "stress" in kinds:
        groups.append(("stress__test_balanced", stress(occl / "stress__test_balanced.jsonl.gz")))
    return groups


def _job_sha(items: Sequence[Item]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.key.encode())
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------- main loop


def _meta(ex: FrozenExtractor, name: str, job: str) -> dict[str, Any]:
    return {
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
    }


def _expect(ex: FrozenExtractor, job: str) -> dict[str, Any]:
    return {
        "job_sha256": job,
        "backbone_fingerprint": ex.fingerprint,
        "weights_tensor_fingerprint": ex.weights_fingerprint,
    }


def _part(out: Path, name: str, k: int, suffix: str = ".npz") -> Path:
    return out / f"{name}.part{k:04d}{suffix}"


def run_group(
    name: str,
    items: list[Item],
    extractors: list[FrozenExtractor],
    out_dirs: dict[str, Path],
    pool: ProcessPoolExecutor,
    batch: int,
    max_inflight: int,
    part_size: int,
) -> dict[str, Any]:
    """Embed one group. Progress is checkpointed every ``part_size`` items as atomic part
    shards; a restart skips completed parts. When every part exists they are verified,
    merged into ``<name>.npz`` and removed."""
    import torch

    job = _job_sha(items)
    need = [
        ex
        for ex in extractors
        if not is_complete(out_dirs[ex.backbone_id] / f"{name}.npz", _expect(ex, job))
    ]
    if not need:
        return {"group": name, "n": len(items), "status": "cached"}
    parts = [items[i : i + part_size] for i in range(0, len(items), part_size)]
    gpu_s = {ex.backbone_id: 0.0 for ex in need}
    cuda = any(ex.device.type == "cuda" for ex in need)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    done_items = 0
    computed = 0
    last = t0
    for k, part in enumerate(parts):
        pjob = _job_sha(part)
        todo = [
            ex
            for ex in need
            if not is_complete(_part(out_dirs[ex.backbone_id], name, k), _expect(ex, pjob))
        ]
        if not todo:
            done_items += len(part)
            continue
        emb = {ex.backbone_id: np.empty((len(part), ex.embedding_dim), np.float32) for ex in todo}
        nrm = {ex.backbone_id: np.empty(len(part), np.float32) for ex in todo}
        batches = iter([part[i : i + batch] for i in range(0, len(part), batch)])
        inflight: deque[Future[np.ndarray]] = deque()
        pos = 0
        while True:
            while len(inflight) < max_inflight:
                nxt = next(batches, None)
                if nxt is None:
                    break
                inflight.append(pool.submit(_render_batch, nxt))
            if not inflight:
                break
            imgs = inflight.popleft().result()  # raises on any render/decode failure
            n = len(imgs)
            for ex in todo:
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
                rate = (computed + pos) / (last - t0)
                print(
                    f"  {name}: {done_items + pos:,}/{len(items):,} ({rate:,.0f} img/s)",
                    flush=True,
                )
        assert pos == len(part)
        keys = [i.key for i in part]
        for ex in todo:
            write_shard(
                _part(out_dirs[ex.backbone_id], name, k),
                keys,
                emb[ex.backbone_id],
                nrm[ex.backbone_id],
                _meta(ex, f"{name}.part{k:04d}", pjob),
            )
        done_items += len(part)
        computed += len(part)
    for ex in need:  # merge verified parts, then remove them
        out = out_dirs[ex.backbone_id]
        shards = [read_shard(_part(out, name, k)) for k in range(len(parts))]
        keys = [str(x) for sh in shards for x in sh.keys.tolist()]
        if keys != [i.key for i in items]:
            raise RuntimeError(f"{name}: merged part keys do not match the job")
        write_shard(
            out / f"{name}.npz",
            keys,
            np.concatenate([sh.emb for sh in shards]),
            np.concatenate([sh.norm for sh in shards]),
            _meta(ex, name, job),
        )
        for k in range(len(parts)):
            _part(out, name, k).unlink()
            _part(out, name, k, ".meta.json").unlink()
    secs = time.perf_counter() - t0
    return {
        "group": name,
        "n": len(items),
        "n_computed_this_session": computed,
        "status": "extracted",
        "backbones": [ex.backbone_id for ex in need],
        "seconds": round(secs, 1),
        "images_per_s_end_to_end": round(computed / secs, 1) if computed else None,
        "model_seconds": {k: round(v, 1) for k, v in gpu_s.items()},
        "model_images_per_s": {k: round(computed / v, 1) for k, v in gpu_s.items() if v > 0},
        "peak_gpu_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--part-size", type=int, default=32_768, help="checkpoint granularity")
    ap.add_argument("--kinds", default="clean,lofo,stress")
    ap.add_argument("--groups", default="", help="comma list of group names to restrict to")
    ap.add_argument("--pilot", type=int, default=0, help="first N items of 3 groups, separate dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 4)))
    args = ap.parse_args()
    import torch

    from veilx.models.extractor import FrozenExtractor

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; pass --device cpu explicitly for the CPU fallback", flush=True)
        return 1
    layout = resolve_data_root()
    groups = build_groups(layout, set(args.kinds.split(",")))
    if args.groups:
        wanted = set(args.groups.split(","))
        groups = [g for g in groups if g[0] in wanted]
    if args.pilot:
        picked: list[tuple[str, GroupFn]] = []
        for prefix in ("clean", "F1_mask__test_ood", "stress"):
            g = next((g for g in groups if g[0].startswith(prefix)), None)
            if g:
                fn = g[1]
                picked.append((g[0], lambda fn=fn: fn()[: args.pilot]))  # type: ignore[misc]
        groups = picked
    print(f"{len(groups)} groups", flush=True)
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
    # per-group results accumulate across sessions (a stopped run resumes from part shards)
    glog_path = layout.embeddings / (
        "extraction_groups_pilot.json" if args.pilot else "extraction_groups.json"
    )
    glog: dict[str, Any] = json.loads(glog_path.read_text("utf-8")) if glog_path.exists() else {}
    t0 = time.perf_counter()
    total = 0
    for name, make in groups:
        items = make()
        total += len(items)
        need_ids = {it.image_id for it in items}
        recs = {
            r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
            for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
            if r.image_id in need_ids
        }
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(layout.extracted), recs),
        ) as pool:
            res = run_group(
                name,
                items,
                extractors,
                out_dirs,
                pool,
                args.batch,
                2 * args.workers,
                args.part_size,
            )
        del items, recs
        print(json.dumps(res), flush=True)
        if res["status"] == "extracted" or name not in glog:
            glog[name] = res
            glog_path.write_text(json.dumps(glog, indent=2), encoding="utf-8")
    summary = {
        "evidence_type": "real_measured_model_result",
        "scope": "embedding extraction run (throughput/latency/memory; no accuracy numbers)",
        "pilot": bool(args.pilot),
        "precision": args.precision,
        "batch": args.batch,
        "part_size": args.part_size,
        "device": args.device,
        "cpu_workers": args.workers,
        "backbones": {ex.backbone_id: ex.fingerprint for ex in extractors},
        "n_groups": len(groups),
        "n_items": total,
        "wall_seconds_last_session": round(time.perf_counter() - t0, 1),
        "groups": [glog[n] for n, _ in groups if n in glog],
        "accelerator": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    private = layout.embeddings / (
        f"extraction_log_pilot_{args.device}.json" if args.pilot else "extraction_log.json"
    )
    private.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    public = (
        ROOT
        / "reports"
        / "experiments"
        / (f"extraction_pilot_{args.device}.json" if args.pilot else "extraction_run.json")
    )
    public.parent.mkdir(parents=True, exist_ok=True)
    with public.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done; wrote {public.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
