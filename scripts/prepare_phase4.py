"""Untouched confirmatory partitions of DigiFace-1M (``veilx.data.untouched``).

``--partition p4`` (default): Phase 4, 72-image parts 2000-3999 and 4000-5999 (unchanged).
``--partition p6``: Phase 6 replication, 72-image parts 6000-7999 and 8000-9999
(``docs/phase6_protocol.md``); dedup also excludes every Phase 4 image. Text below says
"phase4" / "p4_test"; for p6 read "phase6" / "p6_test".


These identities were never downloaded before Phase 4 (owner approval 2026-11-01), so no model,
threshold or rule has ever seen them. They are kept in a SEPARATE private manifest and extraction
directory so that every Phase 1-3 artefact stays bit-identical.

Stages (private outputs under ``<data_root>``):
  extract   official size + member names + CRC-32 (``inspect_zip``), then safe-extract into
            ``extracted/phase4/<part>/``
  scan      DigiFace adapter over ``extracted/phase4`` -> ``manifests/phase4/images.jsonl``
            (every record re-labelled split=test; the partition is test-only)
  dedup     decode every image; pixel SHA-256; exact duplicates within the new set and against
            every existing DigiFace image (``digiface_final.jsonl``) are dropped (test side) and
            counted
  pairs     ``p4_test_pairs``: clustered, balanced, 12 x identities pairs, genuine cap 6
  plans     LOFO plans test_id / test_ood per held-out family (seed 20261101)
  e1plan    evaluation-only E1_hair_strands plan (side b, levels balanced; shared by folds)
  summary   aggregate counts -> ``reports/data/phase4_data_summary.json`` (no ids)

    python scripts/prepare_phase4.py extract scan dedup pairs plans e1plan summary
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_dataset_registry, load_protocol
from veilx.config.schema import SplitName
from veilx.data.archive import inspect_zip, safe_extract
from veilx.data.digiface import DigiFaceAdapter
from veilx.data.images import decode_rgb, pixel_sha256
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl, write_jsonl
from veilx.data.paths import resolve_data_root
from veilx.data.untouched import P4, PARTITIONS, UntouchedPartition
from veilx.occlusion.engine import apply_occlusion
from veilx.occlusion.plan import assert_plan_isolation, plan_conditions
from veilx.pairs import PairSpec, build_pairs
from veilx.seeding import derive_seed

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
REGISTRY = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
FAMILIES = tuple(PROTO.family_ids)
P: UntouchedPartition = P4  # set by --partition (default p4: Phase 4, unchanged)
MEMBER_RE = re.compile(r"\d{1,6}/\d{1,3}\.png")
_W: dict[str, Any] = {}


def _layout() -> Any:
    lay = resolve_data_root()
    (lay.manifests / P.directory / "occlusion").mkdir(parents=True, exist_ok=True)
    (lay.extracted / P.directory).mkdir(parents=True, exist_ok=True)
    return lay


def stage_extract() -> None:
    lay = _layout()
    reports: dict[str, Any] = {}
    for name, size in P.parts:
        part = name.removesuffix(".zip")
        if (lay.extracted / P.directory / part).exists():
            print(f"{part}: already extracted")
            continue
        t0 = time.perf_counter()
        rep = inspect_zip(lay.raw / name, expected_size=size, member_pattern=MEMBER_RE)
        print(f"{name}: members={rep.n_members} crc_ok={rep.crc_ok} rejected={len(rep.rejected)}")
        if rep.rejected or not rep.crc_ok:
            raise SystemExit(f"{name}: integrity check failed")
        safe_extract(lay.raw / name, lay.extracted / P.directory, rep, wrap_dir=part)
        reports[name] = rep.to_dict() | {"seconds": round(time.perf_counter() - t0, 1)}
        print(f"  extracted in {time.perf_counter() - t0:.0f}s", flush=True)
    path = lay.private_reports / f"{P.directory}_archive_reports.json"
    old = json.loads(path.read_text("utf-8")) if path.exists() else {}
    path.write_text(json.dumps(old | reports, indent=1), encoding="utf-8")


def stage_scan() -> None:
    lay = _layout()
    adapter = DigiFaceAdapter(lay.extracted / P.directory, PROTO.splits.bands, REGISTRY)
    records, report = adapter.scan()
    recs = [r.model_copy(update={"split": SplitName.TEST}) for r in records]
    write_jsonl(lay.manifests / P.directory / "scan.jsonl", recs)
    print(
        f"scanned {len(recs):,} images; issues={report.issue_counts()} "
        f"per-identity={report.images_per_identity_histogram}"
    )


def _hash(job: tuple[str, str]) -> tuple[str, str | None, str | None]:
    image_id, rel = job
    try:
        img = decode_rgb((_W["root"] / rel).read_bytes())
    except Exception as exc:
        return image_id, None, f"{type(exc).__name__}"
    if img.shape != (112, 112, 3):
        return image_id, None, f"shape {img.shape}"
    return image_id, pixel_sha256(img), None


def _init(root: str) -> None:
    _W["root"] = Path(root)


def stage_dedup() -> None:
    lay = _layout()
    recs = list(read_jsonl(lay.manifests / P.directory / "scan.jsonl", ImageRecord))
    with ProcessPoolExecutor(
        min(8, max(1, (os.cpu_count() or 2) - 4)),
        initializer=_init,
        initargs=(str(lay.extracted / P.directory),),
    ) as pool:
        res = list(pool.map(_hash, [(r.image_id, r.relpath) for r in recs], chunksize=512))
    failed = [(i, e) for i, _, e in res if e is not None]
    psha = {i: h for i, h, e in res if e is None}
    existing = {
        r.pixel_sha256
        for m in ("digiface_final.jsonl", *P.earlier)
        for r in read_jsonl(lay.manifests / m, ImageRecord)
        if r.pixel_sha256
    }
    seen: set[str] = set()
    keep, dup_existing, dup_within = [], 0, 0
    for r in recs:
        h = psha.get(r.image_id)
        if h is None:
            continue
        if h in existing:
            dup_existing += 1
            continue
        if h in seen:
            dup_within += 1
            continue
        seen.add(h)
        keep.append(r.model_copy(update={"pixel_sha256": h}))
    write_jsonl(lay.manifests / P.directory / "images.jsonl", keep)
    stats = {
        "scanned": len(recs),
        "decode_or_shape_failures": len(failed),
        "exact_duplicates_of_existing_digiface": dup_existing,
        "exact_duplicates_within_new_parts": dup_within,
        "kept": len(keep),
        "identities": len({r.identity_id for r in keep}),
    }
    (lay.private_reports / f"{P.directory}_dedup.json").write_text(
        json.dumps(stats, indent=1), "utf-8"
    )
    print(stats)


def stage_pairs() -> None:
    lay = _layout()
    imgs = list(read_jsonl(lay.manifests / P.directory / "images.jsonl", ImageRecord))
    n_ids = len({r.identity_id for r in imgs})
    spec = PairSpec(
        "clustered",
        12 * n_ids,
        0.5,
        SplitName.TEST,
        P.seed,
        genuine_cap_per_identity=6,
        pair_id_prefix=P.pair_prefix,
    )
    ps = build_pairs(imgs, spec)
    write_jsonl(lay.manifests / P.directory / f"{P.pairs_name}.jsonl", ps.pairs)
    (lay.manifests / P.directory / f"{P.pairs_name}.summary.json").write_text(
        json.dumps(asdict(ps.summary), indent=1, default=str), "utf-8"
    )
    print(
        {k: asdict(ps.summary)[k] for k in ("n_pairs", "n_genuine", "n_impostor", "n_identities")}
    )


def _spec(job: tuple[str, str, str, int]) -> dict[str, Any]:
    image_id, family, level, seed = job
    rel, psha = _W["recs"][image_id]
    if family == "F5_extreme_lighting":
        img = decode_rgb((_W["root"] / rel).read_bytes())
    else:  # physical masks are pixel-independent; the recipe still binds the real source hash
        img = np.zeros((112, 112, 3), dtype=np.uint8)
    return apply_occlusion(img, family, level, seed, source_pixel_sha256=psha).spec.model_dump()  # type: ignore[arg-type]


def _init_plan(root: str, recs: dict[str, tuple[str, str]]) -> None:
    _W["root"], _W["recs"] = Path(root), recs


def stage_plans() -> None:
    lay = _layout()
    imgs = {
        r.image_id: r for r in read_jsonl(lay.manifests / P.directory / "images.jsonl", ImageRecord)
    }
    pairs = list(read_jsonl(lay.manifests / P.directory / f"{P.pairs_name}.jsonl", PairRecord))
    by_id = {p.pair_id: p for p in pairs}
    recs = {i: (r.relpath, r.pixel_sha256 or r.sha256) for i, r in imgs.items()}
    with ProcessPoolExecutor(
        min(8, max(1, (os.cpu_count() or 2) - 4)),
        initializer=_init_plan,
        initargs=(str(lay.extracted / P.directory), recs),
    ) as pool:
        for heldout in FAMILIES:
            for role in ("test_id", "test_ood"):
                plan = plan_conditions(pairs, FAMILIES, heldout, role, global_seed=P.seed)  # type: ignore[arg-type]
                assert_plan_isolation(plan)
                jobs, where = [], []
                for k, c in enumerate(plan):
                    for side, cond in (("a", c.side_a), ("b", c.side_b)):
                        if cond.family is not None:
                            img = (
                                by_id[c.pair_id].image_a
                                if side == "a"
                                else by_id[c.pair_id].image_b
                            )
                            jobs.append((img, cond.family, cond.level, cond.seed))
                            where.append((k, side))
                specs = list(pool.map(_spec, jobs, chunksize=256))
                rows: list[dict[str, Any]] = [
                    {
                        "pair_id": c.pair_id,
                        "fold": c.fold,
                        "role": c.role,
                        "stratum": c.stratum,
                        "a": None,
                        "b": None,
                    }
                    for c in plan
                ]
                for (k, side), sp in zip(where, specs, strict=True):
                    rows[k][side] = sp
                path = (
                    lay.manifests
                    / P.directory
                    / "occlusion"
                    / f"{heldout}__{role}__{P.pairs_name}.jsonl.gz"
                )
                with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
                print(
                    heldout,
                    role,
                    len(rows),
                    "rows",
                    sum(1 for _ in jobs),
                    "occluded sides",
                    flush=True,
                )


def stage_e1plan() -> None:
    """Evaluation-only E1 (hair strands): every pair, side b occluded, level cycling with a
    seeded permutation; one plan shared by all folds (no fold has ever seen E1)."""
    lay = _layout()
    imgs = {
        r.image_id: r for r in read_jsonl(lay.manifests / P.directory / "images.jsonl", ImageRecord)
    }
    pairs = list(read_jsonl(lay.manifests / P.directory / f"{P.pairs_name}.jsonl", PairRecord))
    recs = {i: (r.relpath, r.pixel_sha256 or r.sha256) for i, r in imgs.items()}
    rng = np.random.default_rng(derive_seed(P.seed, "e1-levels"))
    levels = np.array(["low", "medium", "high"])[rng.permutation(len(pairs)) % 3]
    jobs = [
        (p.image_b, "E1_hair_strands", str(lvl), derive_seed(P.seed, "e1", p.pair_id))
        for p, lvl in zip(pairs, levels, strict=True)
    ]
    with ProcessPoolExecutor(
        min(8, max(1, (os.cpu_count() or 2) - 4)),
        initializer=_init_plan,
        initargs=(str(lay.extracted / P.directory), recs),
    ) as pool:
        specs = list(pool.map(_spec, jobs, chunksize=256))
    path = (
        lay.manifests
        / P.directory
        / "occlusion"
        / f"E1_hair_strands__test_e1__{P.pairs_name}.jsonl.gz"
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        for p, sp in zip(pairs, specs, strict=True):
            fh.write(
                json.dumps(
                    {
                        "pair_id": p.pair_id,
                        "fold": "E1_hair_strands",
                        "role": "test_e1",
                        "stratum": "clean_occluded",
                        "a": None,
                        "b": sp,
                    }
                )
                + "\n"
            )
    print("E1 plan", len(pairs), "rows", dict(Counter(levels.tolist())))


def stage_summary() -> None:
    lay = _layout()
    dd = json.loads((lay.private_reports / f"{P.directory}_dedup.json").read_text("utf-8"))
    ps = json.loads(
        (lay.manifests / P.directory / f"{P.pairs_name}.summary.json").read_text("utf-8")
    )
    imgs = list(read_jsonl(lay.manifests / P.directory / "images.jsonl", ImageRecord))
    per_id = Counter(r.identity_id for r in imgs)
    summary = {
        "evidence_type": "real_data_statistic",
        "scope": P.scope,
        "dedup": dd,
        "images_per_identity_histogram": dict(Counter(per_id.values())),
        P.pairs_name: {k: ps[k] for k in ("n_pairs", "n_genuine", "n_impostor", "n_identities")},
        "plan_seed": P.seed,
    }
    out = ROOT / "reports" / "data" / f"{P.directory}_data_summary.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


STAGES = {
    "extract": stage_extract,
    "scan": stage_scan,
    "dedup": stage_dedup,
    "pairs": stage_pairs,
    "plans": stage_plans,
    "e1plan": stage_e1plan,
    "summary": stage_summary,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--partition", choices=list(PARTITIONS), default="p4")
    args = ap.parse_args()
    global P
    P = PARTITIONS[args.partition]
    for s in args.stages:
        STAGES[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
