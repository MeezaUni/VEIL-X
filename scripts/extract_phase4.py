"""Phase 4 feature extraction (docs/phase4_protocol.md). Run in .venv-gpu, memory-monitored.

``--set p3dev``  DEVELOPMENT data only: the Phase 3 sub-split images and LOFO plans for roles
                 train / val / cal (and, with ``--followup``, test_id / test_ood of p3_test,
                 a follow-up set). Each image is re-rendered with the same recipes as Phase 3,
                 and only the NEW Phase 4 quantities are stored. Embeddings, Phase 3 VE maps and
                 GT come from the Phase 3 feature store, joined by key.
                   vec   VE-C logits (float16, 28x28)
                   vecf  VE-C logits of the horizontally flipped input, flipped back (TTA)
                   feat  VE-C bottleneck feature (float16, 256): input to the novelty score
                 Groups: ``p3_clean`` (per-fold columns ``vec_F1``... for all 5 VE-C models)
                 and ``<fold>__<role>__<pairset>`` (that fold's VE-C only).
``--set p6``     Phase 6 replication partition p6_test (``docs/phase6_protocol.md``): identical to
                 ``--set p4`` with the p6 manifests, stored under ``features/phase6/p6``.
``--set p4``     CONFIRMATORY p4_test (never scored before; extraction stores features only,
                 with no decisions or scores). Everything is stored: AdaFace/ArcFace embeddings
                 and norms, the Phase 3 VE map (uint8), VE-C vec / vecf / feat, GT map, visibility
                 ratio. Groups: ``p4_clean`` and ``e1`` (per-fold columns) and
                 ``<fold>__<role>__p4_test_pairs`` for role in test_id / test_ood.

Part shards of 16,384 items; resumable; torch-free render workers.

    .venv-gpu\\Scripts\\python.exe scripts\\extract_phase4.py --set p3dev|p4|p6 [--pilot N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.data.untouched import PARTITIONS
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import is_complete, write_array_shard
from veilx.occlusion.engine import apply_occlusion
from veilx.provenance import collect_provenance
from veilx.vis.regions import visibility_target

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
FAMILIES = tuple(PROTO.family_ids)
PART = 16_384
Items = list[tuple[str, str, str | None, str | None, int]]  # key, image, family, level, seed
_W: dict[str, Any] = {}


def _init(root: str, recs: dict[str, tuple[str, str]]) -> None:
    _W.update(root=Path(root), recs=recs)


def _render(items: Items) -> dict[str, np.ndarray]:
    n = len(items)
    out = {
        "img": np.empty((n, 112, 112, 3), np.uint8),
        "gt": np.empty((n, 28, 28), np.uint8),
        "vis": np.empty(n, np.float32),
    }
    for k, (_key, image_id, fam, lvl, seed) in enumerate(items):
        relpath, psha = _W["recs"][image_id]
        img = load_rgb(_W["root"] / relpath)
        if fam is None:
            out["img"][k], out["gt"][k], out["vis"][k] = img, 255, 1.0
        else:
            res = apply_occlusion(img, fam, lvl, seed, source_pixel_sha256=psha)  # type: ignore[arg-type]
            out["img"][k] = res.image
            out["gt"][k] = np.rint(255 * visibility_target(res.mask)).astype(np.uint8)
            out["vis"][k] = res.spec.visibility_ratio
    return out


def _stream(
    pool: ProcessPoolExecutor, items: Items, bs: int, inflight: int
) -> Iterator[dict[str, np.ndarray]]:
    chunks = iter([items[i : i + bs] for i in range(0, len(items), bs)])
    q: deque[Future[dict[str, np.ndarray]]] = deque()
    while True:
        while len(q) < inflight:
            c = next(chunks, None)
            if c is None:
                break
            q.append(pool.submit(_render, c))
        if not q:
            return
        yield q.popleft().result()


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _side_items(rows: Iterator[dict[str, Any]], pairs: dict[str, tuple[str, str]]) -> Items:
    items: Items = []
    for r in rows:
        for side, img in zip(("a", "b"), pairs[r["pair_id"]], strict=True):
            sp = r[side]
            if sp is not None:
                fam, lvl, seed = sp["family"], sp["severity_level"], int(sp["seed"])
                items.append((occluded_key(img, fam, lvl, seed), img, fam, lvl, seed))
    return items


def main() -> int:
    import torch
    from safetensors.torch import load_file

    from veilx.vis.model import VisNet, to_input

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=["p3dev", *PARTITIONS])
    ap.add_argument("--followup", action="store_true", help="p3dev: also p3_test test_id/test_ood")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    lay = resolve_data_root()
    part = PARTITIONS.get(args.set)  # None for p3dev
    sub = part.directory if part is not None else "phase4"
    out_dir = lay.root / "features" / sub / args.set / ("pilot" if args.pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    def load_vis(path: Path) -> Any:
        m = VisNet().to(dev).eval()
        m.load_state_dict(load_file(str(path), device="cuda"))
        return m

    vec = {f: load_vis(lay.root / "models/phase4/vec" / f / "best.safetensors") for f in FAMILIES}
    ve3 = backbones = None
    if part is not None:
        from veilx.models.extractor import FrozenExtractor
        from veilx.models.spec import BACKBONES

        ve3 = {
            f: load_vis(lay.root / "models/phase3/ve" / f / "best.safetensors") for f in FAMILIES
        }
        backbones = [
            FrozenExtractor(
                s,
                lay.models / s.hf_repo.replace("/", "__") / s.revision,
                device="cuda",
                precision="fp16",
            )
            for s in BACKBONES.values()
        ]

    def vec_outputs(model: Any, imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            x = torch.from_numpy(to_input(imgs)).to(dev)
            lg, ft = model.forward_with_features(x)
            lgf = model(torch.flip(x, dims=[3])).float().flip(dims=[2])
        return (
            lg.float().cpu().numpy().astype(np.float16),
            lgf.cpu().numpy().astype(np.float16),
            ft.float().cpu().numpy().astype(np.float16),
        )

    def ve3_map(model: Any, imgs: np.ndarray) -> np.ndarray:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(torch.from_numpy(to_input(imgs)).to(dev)).float())
        return np.rint(255 * p.cpu().numpy()).astype(np.uint8)

    # ------------------------------------------------------------------ inputs per set
    groups: list[tuple[str, Items, tuple[str, ...]]] = []
    if args.set == "p3dev":
        p3 = lay.manifests / "phase3"
        sub = json.loads((p3 / "subsplits.json").read_text("utf-8"))
        # development roles; --followup extracts ONLY the p3_test follow-up roles, with their
        # clean images in a separate group (never rebuilds p3_clean: resume checks part sizes)
        roles = [
            ("train", "p3_train_pairs"),
            ("val", "p3_val_pairs"),
            ("cal", "p3_cal_independent"),
        ]
        if args.followup:
            roles = [("test_id", "p3_test_pairs"), ("test_ood", "p3_test_pairs")]
        tags = sorted({t for _, t in roles})
        pairs = {
            t: {
                p.pair_id: (p.image_a, p.image_b)
                for p in read_jsonl(p3 / "pairs" / f"{t}.jsonl", PairRecord)
            }
            for t in tags
        }
        used = sorted({i for ps in pairs.values() for ab in ps.values() for i in ab})
        # Mahalanobis fit needs clean p3_train images beyond pair images: add a seeded sample
        tr = sorted(k for k, v in sub.items() if v == "p3_train")
        extra = [
            tr[j] for j in np.random.default_rng(20261101).choice(len(tr), 20_000, replace=False)
        ]
        clean_ids = sorted(set(used)) if args.followup else sorted(set(used) | set(extra))
        need = set(clean_ids)
        recs = {
            r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
            for r in read_jsonl(lay.manifests / "digiface_final.jsonl", ImageRecord)
            if r.image_id in need
        }
        root = lay.extracted
        clean_name = "p3test_clean" if args.followup else "p3_clean"
        groups.append((clean_name, [(i, i, None, None, 0) for i in clean_ids], FAMILIES))
        for role, tag in roles:
            for fold in FAMILIES:
                name = f"{fold}__{role}__{tag}"
                groups.append(
                    (
                        name,
                        _side_items(_rows(p3 / "occlusion" / f"{name}.jsonl.gz"), pairs[tag]),
                        (fold,),
                    )
                )
    else:
        assert part is not None
        p4 = lay.manifests / part.directory
        pairs4 = {
            p.pair_id: (p.image_a, p.image_b)
            for p in read_jsonl(p4 / f"{part.pairs_name}.jsonl", PairRecord)
        }
        used = sorted({i for ab in pairs4.values() for i in ab})
        used_set = set(used)  # built once (membership test per manifest record)
        recs = {
            r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
            for r in read_jsonl(p4 / "images.jsonl", ImageRecord)
            if r.image_id in used_set
        }
        root = lay.extracted / part.directory
        groups.append((part.clean_group, [(i, i, None, None, 0) for i in used], FAMILIES))
        groups.append(
            (
                "e1",
                _side_items(
                    _rows(
                        p4 / "occlusion" / f"E1_hair_strands__test_e1__{part.pairs_name}.jsonl.gz"
                    ),
                    pairs4,
                ),
                FAMILIES,
            )
        )
        for role in ("test_id", "test_ood"):
            for fold in FAMILIES:
                name = f"{fold}__{role}__{part.pairs_name}"
                groups.append(
                    (
                        name,
                        _side_items(_rows(p4 / "occlusion" / f"{name}.jsonl.gz"), pairs4),
                        (fold,),
                    )
                )
    if args.pilot:
        groups = [(g[0], g[1][: args.pilot], g[2]) for g in groups[:2]]
    log: dict[str, Any] = {}
    t_all = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(root), recs)) as pool:
        for name, items, folds in groups:
            t0, computed = time.perf_counter(), 0
            parts = [items[i : i + PART] for i in range(0, len(items), PART)]
            multi = len(folds) > 1
            for k, part in enumerate(parts):
                path = out_dir / f"{name}.part{k:04d}.npz"
                if is_complete(path, {"group": name, "n": len(part)}):
                    continue
                acc: dict[str, list[np.ndarray]] = {}
                for b in _stream(pool, part, 256, 2 * args.workers):
                    for fold in folds:
                        sfx = f"_{fold[:2]}" if multi else ""
                        lg, lgf, ft = vec_outputs(vec[fold], b["img"])
                        acc.setdefault(f"vec{sfx}", []).append(lg)
                        acc.setdefault(f"vecf{sfx}", []).append(lgf)
                        acc.setdefault(f"feat{sfx}", []).append(ft)
                        if ve3 is not None:
                            acc.setdefault(f"ve{sfx}", []).append(ve3_map(ve3[fold], b["img"]))
                    if backbones is not None:
                        for bb in backbones:
                            e = bb.embed(b["img"])
                            acc.setdefault(f"emb_{bb.backbone_id[:3]}", []).append(
                                e.embeddings.astype(np.float16)
                            )
                            acc.setdefault(f"norm_{bb.backbone_id[:3]}", []).append(e.feature_norms)
                        acc.setdefault("gt", []).append(b["gt"])
                        acc.setdefault("vis", []).append(b["vis"])
                arrays = {c: np.concatenate(v) for c, v in acc.items()}
                write_array_shard(
                    path,
                    [it[0] for it in part],
                    arrays,
                    {"group": name, "set": args.set, "part": k, "folds": list(folds)},
                )
                computed += len(part)
                del acc, arrays
            secs = time.perf_counter() - t0
            log[name] = {
                "n": len(items),
                "parts": len(parts),
                "computed": computed,
                "seconds": round(secs, 1),
                "items_per_s": round(computed / secs, 1) if computed else None,
            }
            print(name, log[name], flush=True)
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": f"Phase 4 feature extraction, set {args.set} (throughput only; no scores)",
        "pilot": bool(args.pilot),
        "groups": log,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    if not args.pilot:
        with (ROOT / "reports" / "experiments" / f"{sub}_extraction_{args.set}.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as fh:
            json.dump(report, fh, indent=1)
    print("done", round(time.perf_counter() - t_all, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
