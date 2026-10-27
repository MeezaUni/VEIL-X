"""Phase 3 data: sub-splits of the never-scored DigiFace train identities, pair sets and LOFO plans.

Follows docs/phase3_protocol.md section 2 exactly (protocol v0.5.0). Outputs (private):
``<data_root>/manifests/phase3/{subsplits.json, pairs/*.jsonl, occlusion/*.jsonl.gz}``; aggregate
summary -> ``reports/data/phase3_data_summary.json``.

    python scripts/prepare_phase3.py          (data root from VEILX_DATA_ROOT)
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.config.schema import SplitName
from veilx.data.images import decode_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl, write_jsonl
from veilx.data.paths import resolve_data_root
from veilx.data.splits import identity_bucket
from veilx.occlusion.engine import apply_occlusion
from veilx.occlusion.plan import Role, assert_plan_isolation, plan_conditions
from veilx.pairs import PairSpec, build_pairs

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P3 = PROTO.phase3
assert P3 is not None
SEED = P3.seed
FAMILIES = tuple(PROTO.family_ids)

_W: dict[str, Any] = {}


def subsplit_of(identity: str) -> str:
    b = identity_bucket(P3.subsplit_salt, identity)  # type: ignore[union-attr]
    for name, (lo, hi) in P3.subsplit_bands.items():  # type: ignore[union-attr]
        if lo <= b < hi:
            return name
    raise ValueError("bands do not cover bucket")


def _init(extracted: str, recs: dict[str, tuple[str, str]]) -> None:
    _W["root"], _W["recs"] = Path(extracted), recs


def _side_spec(args: tuple[str, str, str, int]) -> dict[str, Any]:
    image_id, family, level, seed = args
    relpath, psha = _W["recs"][image_id]
    if family == "F5_extreme_lighting":
        img = decode_rgb((_W["root"] / relpath).read_bytes())
    else:  # physical masks are pixel-independent; the recipe still binds the real source hash
        img = np.zeros((112, 112, 3), dtype=np.uint8)
    return apply_occlusion(img, family, level, seed, source_pixel_sha256=psha).spec.model_dump()  # type: ignore[arg-type]


def main() -> int:
    layout = resolve_data_root()
    out = layout.manifests / "phase3"
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "occlusion").mkdir(parents=True, exist_ok=True)
    final = list(read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord))
    train = [r for r in final if r.split is SplitName.TRAIN]
    by_sub: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in train:
        by_sub[subsplit_of(r.identity_id)].append(r)
    ids = {k: len({r.identity_id for r in v}) for k, v in by_sub.items()}
    (out / "subsplits.json").write_text(
        json.dumps({r.image_id: subsplit_of(r.identity_id) for r in train}), encoding="utf-8"
    )
    print("identities per sub-split", ids, flush=True)
    specs = {
        "p3_train_pairs": (
            "p3_train",
            PairSpec(
                "clustered",
                40_000,
                0.5,
                SplitName.TRAIN,
                SEED,
                genuine_cap_per_identity=1,
                pair_id_prefix="p3tr",
            ),
        ),
        "p3_val_pairs": (
            "p3_val",
            PairSpec(
                "clustered",
                4 * ids["p3_val"],
                0.5,
                SplitName.TRAIN,
                SEED,
                genuine_cap_per_identity=2,
                pair_id_prefix="p3va",
            ),
        ),
        "p3_cal_independent": (
            "p3_cal",
            PairSpec(
                "independent",
                2 * (ids["p3_cal"] // 3),
                0.5,
                SplitName.TRAIN,
                SEED,
                pair_id_prefix="p3ca",
            ),
        ),
        "p3_test_pairs": (
            "p3_test",
            PairSpec(
                "clustered",
                4 * ids["p3_test"],
                0.5,
                SplitName.TRAIN,
                SEED,
                genuine_cap_per_identity=2,
                pair_id_prefix="p3te",
            ),
        ),
    }
    stats: dict[str, Any] = {
        "identities_per_subsplit": ids,
        "images_per_subsplit": {k: len(v) for k, v in by_sub.items()},
    }
    pairsets: dict[str, list[PairRecord]] = {}
    for tag, (sub, spec) in specs.items():
        t0 = time.perf_counter()
        ps = build_pairs(by_sub[sub], spec)
        write_jsonl(out / "pairs" / f"{tag}.jsonl", ps.pairs)
        (out / "pairs" / f"{tag}.provenance.json").write_text(
            json.dumps(ps.provenance, indent=2, default=str), encoding="utf-8"
        )
        pairsets[tag] = list(ps.pairs)
        stats[tag] = asdict(ps.summary) | {"seconds": round(time.perf_counter() - t0, 1)}
        print(
            tag,
            {
                k: stats[tag][k]
                for k in ("n_pairs", "n_genuine", "n_impostor", "n_identities", "identity_disjoint")
            },
            flush=True,
        )
    recs = {r.image_id: (r.relpath, r.pixel_sha256 or r.sha256) for r in train}
    role_sets: list[tuple[Role, str]] = [
        ("train", "p3_train_pairs"),
        ("val", "p3_val_pairs"),
        ("cal", "p3_cal_independent"),
        ("test_id", "p3_test_pairs"),
        ("test_ood", "p3_test_pairs"),
    ]
    stats["plans"] = {}
    with ProcessPoolExecutor(
        max_workers=min(8, max(1, (os.cpu_count() or 2) - 4)),
        initializer=_init,
        initargs=(str(layout.extracted), recs),
    ) as pool:
        for heldout in FAMILIES:
            for role, tag in role_sets:
                pairs = pairsets[tag]
                plan = plan_conditions(pairs, FAMILIES, heldout, role, global_seed=SEED)
                assert_plan_isolation(plan)
                by_id = {p.pair_id: p for p in pairs}
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
                spec_list = list(pool.map(_side_spec, jobs, chunksize=256))
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
                for (k, side), sp in zip(where, spec_list, strict=True):
                    rows[k][side] = sp
                path = out / "occlusion" / f"{heldout}__{role}__{tag}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                stats["plans"][f"{heldout}:{role}:{tag}"] = {
                    "n_pairs": len(plan),
                    "n_occluded_sides": len(jobs),
                    "strata": dict(Counter(c.stratum for c in plan)),
                }
                print(
                    f"{heldout} {role}:{tag} pairs={len(plan):,} occluded={len(jobs):,}", flush=True
                )
    summary = {
        "evidence_type": "real_data_preparation_statistics",
        "note": "Phase 3 sub-splits of never-scored DigiFace train identities; aggregates only.",
        "protocol_version": PROTO.protocol_version,
        "stats": {k: v for k, v in stats.items()},
    }
    pub = ROOT / "reports" / "data" / "phase3_data_summary.json"
    with pub.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("wrote", pub.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
