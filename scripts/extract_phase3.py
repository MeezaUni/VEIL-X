"""Phase 3 feature extraction: backbone embeddings + visibility maps for every planned image.

Groups (part shards of 16,384 items under ``<data_root>/features/phase3/``; resumable):

* ``p3_clean``   every image referenced by the four Phase 3 pair sets: AdaFace/ArcFace unit
                 embeddings + feature norms, the learned visibility map of EACH fold model
                 (``ve_F1``..``ve_F5``, uint8 = round(255 * P(visible))), the heuristic anomaly map
                 (``z``, float16);
* ``<fold>__<role>__<pairset>``  every occluded side of that LOFO plan: embeddings/norms, the
                 visibility map of the fold's model (``ve``), the anomaly map, the ORACLE
                 ground-truth map (``gt``, uint8) and ``visibility_ratio``.

The oracle map is stored for reference/upper-bound analyses only; it is never an input to the
estimator. Heuristic statistics (per-cell mean/sd) come from 20,000 clean p3_train images.

    .venv-gpu\\Scripts\\python.exe scripts\\extract_phase3.py [--pilot N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import is_complete, write_array_shard
from veilx.occlusion.engine import apply_occlusion
from veilx.provenance import collect_provenance
from veilx.vis.heuristic import cell_means
from veilx.vis.regions import visibility_target

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P3 = PROTO.phase3
assert P3 is not None
FAMILIES = tuple(PROTO.family_ids)
ROLES = (
    ("train", "p3_train_pairs"),
    ("val", "p3_val_pairs"),
    ("cal", "p3_cal_independent"),
    ("test_id", "p3_test_pairs"),
    ("test_ood", "p3_test_pairs"),
)
PART = 16_384
Items = list[tuple[str, str, str | None, str | None, int]]  # key, image, family, level, seed
_W: dict[str, Any] = {}


def _init(root: str, recs: dict[str, tuple[str, str]], mu: np.ndarray, sd: np.ndarray) -> None:
    _W.update(root=Path(root), recs=recs, mu=mu, sd=sd)


def _render(items: list[tuple[str, str, str | None, str | None, int]]) -> dict[str, np.ndarray]:
    n = len(items)
    out = {
        "img": np.empty((n, 112, 112, 3), np.uint8),
        "gt": np.empty((n, 28, 28), np.uint8),
        "vis": np.empty(n, np.float32),
        "z": np.empty((n, 28, 28), np.float16),
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
    zz = (cell_means(out["img"]) - _W["mu"]) / _W["sd"]
    out["z"] = np.sqrt((zz**2).sum(-1)).astype(np.float16)
    return out


def _stream(
    pool: ProcessPoolExecutor, items: list[Any], bs: int, inflight: int
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


def main() -> int:
    import torch
    from safetensors.torch import load_file

    from veilx.models.extractor import FrozenExtractor
    from veilx.models.spec import BACKBONES
    from veilx.vis.model import VisNet, to_input

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--folds", default=",".join(FAMILIES), help="pilot only: subset of fold models")
    args = ap.parse_args()
    layout = resolve_data_root()
    p3 = layout.manifests / "phase3"
    out_dir = layout.root / "features" / "phase3" / ("pilot" if args.pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = json.loads((p3 / "subsplits.json").read_text("utf-8"))
    pairs = {  # pair_id -> (image_a, image_b); plain tuples keep the parent process small
        tag: {
            p.pair_id: (p.image_a, p.image_b)
            for p in read_jsonl(p3 / "pairs" / f"{tag}.jsonl", PairRecord)
        }
        for _, tag in ROLES
    }
    used = sorted({i for ps in pairs.values() for ab in ps.values() for i in ab})
    hs = layout.root / "models" / "phase3" / "heuristic_stats.npz"
    tr = sorted(k for k, v in sub.items() if v == "p3_train")
    hpick = [tr[j] for j in np.random.default_rng(P3.seed).choice(len(tr), 20_000, replace=False)]
    del sub, tr
    need = set(used) | set(hpick)
    recs = {
        r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
        if r.image_id in need
    }
    del need
    # ---- heuristic statistics from clean p3_train images (training data only)
    if not hs.exists():
        cms = []
        with ThreadPoolExecutor(8) as tp:
            for i in range(0, len(hpick), 1000):  # chunked: float64 cell means stay small
                imgs = np.stack(
                    list(
                        tp.map(
                            lambda j: load_rgb(layout.extracted / recs[j][0]),
                            hpick[i : i + 1000],
                        )
                    )
                )
                cms.append(cell_means(imgs))
        c = np.concatenate(cms)
        hs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(hs, mu=c.mean(0), sd=np.maximum(c.std(0), 1e-3), n=len(hpick))
        del cms, c, imgs
    recs = {i: recs[i] for i in used}  # workers only need pair images
    st = np.load(hs)
    mu, sd = st["mu"], st["sd"]
    dev = torch.device("cuda")
    backbones = [
        FrozenExtractor(
            s,
            layout.models / s.hf_repo.replace("/", "__") / s.revision,
            device="cuda",
            precision="fp16",
        )
        for s in BACKBONES.values()
    ]
    folds_used = tuple(args.folds.split(","))
    if not args.pilot and folds_used != FAMILIES:
        raise SystemExit("a fold subset is allowed only with --pilot")
    ve: dict[str, Any] = {}
    for fam in folds_used:
        m = VisNet().to(dev).eval()
        m.load_state_dict(
            load_file(
                str(layout.root / "models" / "phase3" / "ve" / fam / "best.safetensors"),
                device="cuda",
            )
        )
        ve[fam] = m

    def ve_maps(model: Any, imgs: np.ndarray) -> np.ndarray:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(torch.from_numpy(to_input(imgs)).to(dev)).float())
        return np.rint(255 * p.cpu().numpy()).astype(np.uint8)

    def plan_items(name: str, tag: str) -> Items:
        items: Items = []
        for r in _rows(p3 / "occlusion" / f"{name}.jsonl.gz"):
            for side, img in zip(("a", "b"), pairs[tag][r["pair_id"]], strict=True):
                sp = r[side]
                if sp is not None:
                    fam, lvl, seed = sp["family"], sp["severity_level"], int(sp["seed"])
                    items.append((occluded_key(img, fam, lvl, seed), img, fam, lvl, seed))
        return items

    def groups() -> Iterator[tuple[str, Items, tuple[str, ...]]]:
        """Built lazily, one group at a time (plans hold 1.2 M occluded sides in total)."""
        n = args.pilot or None
        yield "p3_clean", [(i, i, None, None, 0) for i in used][:n], folds_used
        for role, tag in ROLES:
            for fold in FAMILIES:
                name = f"{fold}__{role}__{tag}"
                yield name, plan_items(name, tag)[:n], (fold,)
                if args.pilot:
                    return

    log: dict[str, Any] = {}
    t_all = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(layout.extracted), recs, mu, sd)
    ) as pool:
        for name, items, folds in groups():
            t0, computed = time.perf_counter(), 0
            parts = [items[i : i + PART] for i in range(0, len(items), PART)]
            for k, part in enumerate(parts):
                path = out_dir / f"{name}.part{k:04d}.npz"
                if is_complete(path, {"group": name, "n": len(part)}):
                    continue
                acc: dict[str, list[np.ndarray]] = {}
                for b in _stream(pool, part, 256, 2 * args.workers):
                    for bb in backbones:
                        e = bb.embed(b["img"])
                        acc.setdefault(f"emb_{bb.backbone_id[:3]}", []).append(
                            e.embeddings.astype(np.float16)
                        )
                        acc.setdefault(f"norm_{bb.backbone_id[:3]}", []).append(e.feature_norms)
                    if name == "p3_clean":
                        for fam in folds:
                            acc.setdefault(f"ve_{fam[:2]}", []).append(ve_maps(ve[fam], b["img"]))
                    else:
                        acc.setdefault("ve", []).append(ve_maps(ve[folds[0]], b["img"]))
                        acc.setdefault("gt", []).append(b["gt"])
                        acc.setdefault("vis", []).append(b["vis"])
                    acc.setdefault("z", []).append(b["z"])
                arrays = {k2: np.concatenate(v) for k2, v in acc.items()}
                write_array_shard(
                    path,
                    [it[0] for it in part],
                    arrays,
                    {"group": name, "fold": folds[0] if len(folds) == 1 else "all", "part": k},
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
        "scope": "Phase 3 feature extraction (throughput only; no accuracy numbers)",
        "pilot": bool(args.pilot),
        "groups": log,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    if not args.pilot:
        with (ROOT / "reports" / "experiments" / "phase3_extraction.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as fh:
            json.dump(report, fh, indent=1)
    print("done", round(time.perf_counter() - t_all, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
