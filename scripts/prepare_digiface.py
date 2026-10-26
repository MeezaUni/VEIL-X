"""Phase 1b DigiFace preparation pipeline (private data stays under VEILX_DATA_ROOT).

Stages (run in order; each writes its outputs to the private data root):

  extract      verify archive size/SHA-256/member names/CRC-32, then safe-extract
  scan         Phase 1a adapter over the extracted parts -> manifest + anomaly report
  fingerprint  decode every image; pixel SHA-256, flip-aware pHash and NCC thumbnails
  dedup        validate the near-duplicate rule on train (select) / val (confirm), then apply
               it exhaustively; exact + near-duplicate groups; non-test-side removal
  pairs        deterministic pair manifests per split (independent calibration set)
  occlusion    leave-one-family-out condition plans with severity/visibility records
  summary      aggregated, non-identifying statistics -> reports/data/ (committable)

    python scripts/prepare_digiface.py extract scan fingerprint dedup pairs occlusion summary
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_dataset_registry, load_protocol
from veilx.config.schema import SplitName
from veilx.data.archive import inspect_zip, safe_extract
from veilx.data.dedup import candidate_pairs, frontier, review_rule, validate_rule
from veilx.data.digiface import DigiFaceAdapter
from veilx.data.images import decode_rgb
from veilx.data.integrity import (
    FingerprintTable,
    finalize_records,
    find_duplicates,
    fingerprint_records,
    validation_sets,
)
from veilx.data.manifest import (
    ImageRecord,
    PairRecord,
    read_jsonl,
    validate_image_manifest,
    write_jsonl,
)
from veilx.data.paths import DataLayout, resolve_data_root
from veilx.occlusion.engine import apply_occlusion
from veilx.occlusion.plan import (
    Role,
    assert_plan_isolation,
    plan_conditions,
    plan_stress,
)
from veilx.pairs import PairSpec, build_pairs
from veilx.pairs.builder import source_keys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = Path(os.environ.get("VEILX_PUBLIC_REPORTS", ROOT / "reports" / "data"))
APPROVED_PARTS: dict[str, int] = {
    "subjects_0-1999_72_imgs.zip": 2_938_553_163,
    "subjects_100000-133332_5_imgs.zip": 3_399_664_847,
}
MEMBER_RE = re.compile(r"\d{1,6}/\d{1,3}\.png")
GENUINE_CAP = 10  # equal per-identity weight for 72- and 5-image identities (D-026)
FAR_IMPOSTORS = int(os.environ.get("VEILX_FAR_IMPOSTORS", "100000"))
DEDUP_N_TRAIN = int(os.environ.get("VEILX_DEDUP_N_TRAIN", "4000"))
DEDUP_N_VAL = int(os.environ.get("VEILX_DEDUP_N_VAL", "2000"))
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
REGISTRY = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
SEED = PROTO.seeds.global_seed


def _layout() -> DataLayout:
    return resolve_data_root().ensure()


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- extract


def stage_extract() -> None:
    layout = _layout()
    path = layout.private_reports / "archive_reports.json"
    reports = _load(path) if path.exists() else {}
    for name, size in APPROVED_PARTS.items():
        part = name.removesuffix(".zip")
        if (layout.extracted / part).exists():
            print(f"{part}: already extracted")
            continue
        t0 = time.perf_counter()
        rep = inspect_zip(layout.raw / name, expected_size=size, member_pattern=MEMBER_RE)
        print(
            f"{name}: members={rep.n_members} crc_ok={rep.crc_ok} rejected={len(rep.rejected)}",
            flush=True,
        )
        if not rep.rejected:
            safe_extract(layout.raw / name, layout.extracted, rep, wrap_dir=part)
            print(f"  extracted ({time.perf_counter() - t0:.0f}s)", flush=True)
        reports[name] = rep.to_dict()
        _dump(path, reports)


# --------------------------------------------------------------------------- scan


def stage_scan() -> None:
    layout = _layout()
    adapter = DigiFaceAdapter(layout.extracted, PROTO.splits.bands, REGISTRY)
    meta = adapter.write_manifest(layout.manifests / "scan")
    _, report = adapter.scan()
    print(json.dumps({k: meta[k] for k in ("n_records", "manifest_sha256")}))
    print("issues:", report.issue_counts(), "histogram:", report.images_per_identity_histogram)


# --------------------------------------------------------------------------- fingerprint


def _scan_records(layout: DataLayout) -> list[ImageRecord]:
    return list(read_jsonl(layout.manifests / "scan" / "manifest.jsonl", ImageRecord))


def _alpha_summary(table: FingerprintTable) -> dict[str, int]:
    has = table.alpha_min >= 0
    return {
        "images_with_alpha": int(has.sum()),
        "images_not_fully_opaque": int((has & (table.alpha_min < 255)).sum()),
        "min_alpha": int(table.alpha_min[has].min()) if has.any() else -1,
    }


def stage_fingerprint() -> None:
    layout = _layout()
    records = _scan_records(layout)
    t0 = time.perf_counter()
    table = fingerprint_records(records, layout.extracted)
    table.save(layout.manifests / "fingerprints.npz")
    dims = Counter(
        zip(table.width[table.ok].tolist(), table.height[table.ok].tolist(), strict=True)
    )
    print(
        f"fingerprinted {len(records):,} in {time.perf_counter() - t0:.0f}s; "
        f"ok={int(table.ok.sum()):,} "
        f"failed={len(table.errors)} dims={dict(dims)} modes={dict(Counter(table.mode))} "
        f"alpha={_alpha_summary(table)}"
    )


# --------------------------------------------------------------------------- dedup


def stage_dedup() -> None:
    layout = _layout()
    records = _scan_records(layout)
    table = FingerprintTable.load(layout.manifests / "fingerprints.npz")
    assert table.image_ids == [r.image_id for r in records], (
        "fingerprints out of sync with manifest"
    )
    target = float(PROTO.deduplication.near_duplicate.target_duplicate_recall.value)  # type: ignore[arg-type]
    train_sets, train_ops = validation_sets(
        records, table, layout.extracted, SplitName.TRAIN, seed=SEED, n_each=DEDUP_N_TRAIN
    )
    val_sets, val_ops = validation_sets(
        records, table, layout.extracted, SplitName.VAL, seed=SEED, n_each=DEDUP_N_VAL
    )
    validation = validate_rule(train_sets, val_sets, target_recall=target)
    print("validation:", json.dumps(validation.to_dict(), default=str))
    t0 = time.perf_counter()
    findings = find_duplicates(records, table, validation)
    final = finalize_records(records, table, findings)
    validate_image_manifest(final)
    write_jsonl(layout.manifests / "digiface_final.jsonl", final)
    # Review-only nomination (never automatic removal): strict rule at <= 0.1 % train
    # same-identity FPR, confirmed on val; exhaustive search over all decodable images.
    front = frontier(*train_sets)
    rv_train, rv_val = review_rule(train_sets, val_sets, max_same_identity_fpr=0.001)
    review: dict[str, Any] = {
        "train_rule": asdict(rv_train) if rv_train else None,
        "val_eval": asdict(rv_val) if rv_val else None,
    }
    if rv_train is not None:
        ok_idx = np.flatnonzero(table.ok)
        cands = candidate_pairs(
            table.phash[ok_idx],
            table.phash_flip[ok_idx],
            table.thumb[ok_idx],
            table.thumb_flip[ok_idx],
            hamming_max=rv_train.hamming_max,
            ncc_min=rv_train.ncc_min,
        )
        rel: Counter[str] = Counter()
        rows = []
        for i, j, ham, ncc in cands:
            a, b = records[int(ok_idx[i])], records[int(ok_idx[j])]
            kind = (
                "cross_split"
                if a.split is not b.split
                else "cross_identity_same_split"
                if a.identity_id != b.identity_id
                else "same_identity"
            )
            rel[kind] += 1
            rows.append(
                {
                    "a": a.image_id,
                    "b": b.image_id,
                    "hamming": ham,
                    "ncc": round(ncc, 4),
                    "relation": kind,
                }
            )
        _dump(layout.private_reports / "near_duplicate_review_candidates.json", rows)
        review |= {
            "n_candidates": len(cands),
            "by_relation": dict(rel),
            "candidates_file": "<data>/reports/near_duplicate_review_candidates.json",
            "status": "nominated for manual review; NOT removed (rule not validated at the "
            "registered recall target)",
        }
    summary = {
        "validation": validation.to_dict(),
        "frontier_train": [asdict(e) for e in front],
        "review": review,
        "transform_counts": {"train": train_ops, "val": val_ops},
        "distance_quantiles": {
            name: {
                kind: np.quantile(arr[:, k], [0.01, 0.05, 0.5, 0.95, 0.99]).round(4).tolist()
                for k, kind in ((0, "hamming"), (1, "ncc"))
            }
            for name, arr in zip(("pos", "same_identity", "impostor"), train_sets, strict=True)
        },
        "findings": {k: v for k, v in asdict(findings).items() if k != "group_of"},
        "n_scan_records": len(records),
        "n_final_records": len(final),
        "decode_failures": len(table.errors),
        "search_seconds": round(time.perf_counter() - t0, 1),
    }
    _dump(layout.private_reports / "dedup_report.json", summary)
    print(json.dumps(summary["findings"]), "final records:", len(final))


# --------------------------------------------------------------------------- pairs


def _final(layout: DataLayout) -> list[ImageRecord]:
    return list(read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord))


def _genuine_capacity(images: list[ImageRecord], cap: int) -> int:
    keys = source_keys(images)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for r in images:
        groups[r.identity_id][keys[r.image_id]] += 1
    total = 0
    for c in groups.values():
        k = sum(c.values())
        valid = k * (k - 1) // 2 - sum(m * (m - 1) // 2 for m in c.values())
        total += min(cap, valid)
    return total


def stage_pairs() -> None:
    layout = _layout()
    final = _final(layout)
    by_split: dict[SplitName, list[ImageRecord]] = defaultdict(list)
    for r in final:
        by_split[r.split].append(r)
    out_dir = layout.manifests / "pairs"
    stats: dict[str, Any] = {}
    plans: list[tuple[str, SplitName, PairSpec]] = []
    for split, tag in (
        (SplitName.TRAIN, "train"),
        (SplitName.VAL, "val"),
        (SplitName.TEST, "test_balanced"),
    ):
        cap = _genuine_capacity(by_split[split], GENUINE_CAP)
        plans.append(
            (
                tag,
                split,
                PairSpec(
                    "clustered",
                    2 * cap,
                    0.5,
                    split,
                    SEED,
                    genuine_cap_per_identity=GENUINE_CAP,
                    pair_id_prefix=tag,
                ),
            )
        )
    plans.append(
        (
            "test_far",
            SplitName.TEST,
            PairSpec(
                "clustered", FAR_IMPOSTORS, 0.0, SplitName.TEST, SEED, pair_id_prefix="test_far"
            ),
        )
    )
    cal = by_split[SplitName.CAL]
    n_ids = len({r.identity_id for r in cal})
    n_g = n_ids // 3
    plans.append(
        (
            "cal_independent",
            SplitName.CAL,
            PairSpec("independent", 2 * n_g, 0.5, SplitName.CAL, SEED, pair_id_prefix="cal"),
        )
    )
    plans.append(
        (
            "val_independent",
            SplitName.VAL,
            PairSpec(
                "independent",
                2 * (len({r.identity_id for r in by_split[SplitName.VAL]}) // 3),
                0.5,
                SplitName.VAL,
                SEED,
                pair_id_prefix="val_ind",
            ),
        )
    )
    for tag, split, spec in plans:
        t0 = time.perf_counter()
        ps = build_pairs(by_split[split], spec)
        write_jsonl(out_dir / f"{tag}.jsonl", ps.pairs)
        _dump(out_dir / f"{tag}.provenance.json", ps.provenance)
        stats[tag] = asdict(ps.summary) | {
            "split": split.value,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(
            tag,
            json.dumps(
                {
                    k: stats[tag][k]
                    for k in (
                        "n_pairs",
                        "n_genuine",
                        "n_impostor",
                        "n_identities",
                        "identity_disjoint",
                    )
                }
            ),
            flush=True,
        )
    _dump(layout.private_reports / "pair_stats.json", stats)


# --------------------------------------------------------------------------- occlusion

_W: dict[str, Any] = {}


def _init_worker(extracted: str, recs: dict[str, tuple[str, str]], shape: tuple[int, int]) -> None:
    _W["root"], _W["recs"], _W["shape"] = Path(extracted), recs, shape


def _side_spec(args: tuple[str, str, str, int]) -> dict[str, Any]:
    image_id, family, level, seed = args
    relpath, psha = _W["recs"][image_id]
    if family == "F5_extreme_lighting":  # photometric: depends on the real pixels
        img = decode_rgb((_W["root"] / relpath).read_bytes())
    else:  # physical families: mask is pixel-independent; recipe still binds the real source hash
        img = np.zeros((*_W["shape"], 3), dtype=np.uint8)
    spec = apply_occlusion(img, family, level, seed, source_pixel_sha256=psha).spec  # type: ignore[arg-type]
    return spec.model_dump()


def stage_occlusion() -> None:
    layout = _layout()
    final = {r.image_id: r for r in _final(layout)}
    table = FingerprintTable.load(layout.manifests / "fingerprints.npz")
    shape = (int(np.median(table.height[table.ok])), int(np.median(table.width[table.ok])))
    recs = {k: (r.relpath, r.pixel_sha256 or r.sha256) for k, r in final.items()}
    families = list(PROTO.family_ids)
    role_sets: list[tuple[Role, str]] = [
        ("train", "train"),
        ("val", "val"),
        ("cal", "cal_independent"),
        ("test_id", "test_balanced"),
        ("test_ood", "test_balanced"),
        ("test_id", "test_far"),
        ("test_ood", "test_far"),
    ]
    out_dir = layout.manifests / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = defaultdict(dict)
    with ProcessPoolExecutor(
        max_workers=max(1, (os.cpu_count() or 2) - 2),
        initializer=_init_worker,
        initargs=(str(layout.extracted), recs, shape),
    ) as pool:
        for heldout in families:
            for role, pair_tag in role_sets:
                pairs = list(
                    read_jsonl(layout.manifests / "pairs" / f"{pair_tag}.jsonl", PairRecord)
                )
                plan = plan_conditions(pairs, families, heldout, role, global_seed=SEED)
                assert_plan_isolation(plan)
                by_id = {p.pair_id: p for p in pairs}
                jobs, where = [], []
                for k, c in enumerate(plan):
                    for side_name, side in (("a", c.side_a), ("b", c.side_b)):
                        if side.family is not None:
                            img = (
                                by_id[c.pair_id].image_a
                                if side_name == "a"
                                else by_id[c.pair_id].image_b
                            )
                            jobs.append((img, side.family, side.level, side.seed))
                            where.append((k, side_name))
                t0 = time.perf_counter()
                specs = list(pool.map(_side_spec, jobs, chunksize=256))
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
                agg: dict[str, list[float]] = defaultdict(list)
                for (k, side_name), spec in zip(where, specs, strict=True):
                    rows[k][side_name] = spec
                    agg[f"{spec['family']}|{spec['severity_level']}"].append(
                        spec["visibility_ratio"]
                    )
                path = out_dir / f"{heldout}__{role}__{pair_tag}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                stats[heldout][f"{role}:{pair_tag}"] = {
                    "n_pairs": len(plan),
                    "n_occluded_sides": len(jobs),
                    "strata": dict(Counter(c.stratum for c in plan)),
                    "families": dict(Counter(s[1] for s in jobs)),
                    "visibility_by_family_level": {
                        k: {
                            "n": len(v),
                            "mean": round(float(np.mean(v)), 4),
                            "q05": round(float(np.quantile(v, 0.05)), 4),
                            "q50": round(float(np.quantile(v, 0.5)), 4),
                            "q95": round(float(np.quantile(v, 0.95)), 4),
                        }
                        for k, v in sorted(agg.items())
                    },
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                print(
                    f"{heldout} {role}:{pair_tag} pairs={len(plan):,} occluded_sides={len(jobs):,} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
    stress_pairs = list(read_jsonl(layout.manifests / "pairs" / "test_balanced.jsonl", PairRecord))
    stress = plan_stress(stress_pairs, global_seed=SEED)
    with gzip.open(
        out_dir / "stress__test_balanced.jsonl.gz", "wt", encoding="utf-8", newline="\n"
    ) as fh:
        for s in stress:
            fh.write(json.dumps(asdict(s), sort_keys=True) + "\n")
    stats["stress"] = {
        "n_pairs": len(stress),
        "combos": dict(Counter("+".join(f for f, _, _ in s.steps_a) for s in stress)),
    }
    _dump(layout.private_reports / "occlusion_stats.json", stats)


# --------------------------------------------------------------------------- summary


def stage_summary() -> None:
    layout = _layout()
    pr = layout.private_reports
    arch_path = pr / "archive_reports.json"
    archives = (
        _load(arch_path) if arch_path.exists() else {}
    )  # absent if images were not from archives
    scan = _load(layout.manifests / "scan" / "scan_report.json")
    dedup = _load(pr / "dedup_report.json")
    pairs = _load(pr / "pair_stats.json")
    occl = _load(pr / "occlusion_stats.json")
    final = _final(layout)
    table = FingerprintTable.load(layout.manifests / "fingerprints.npz")
    split_ids: dict[str, set[str]] = defaultdict(set)
    split_imgs: Counter[str] = Counter()
    per_part: Counter[str] = Counter()
    for r in final:
        split_ids[r.split.value].add(r.identity_id)
        split_imgs[r.split.value] += 1
        per_part["72_img_part" if int(r.identity_id) < 100_000 else "5_img_part"] += 1
    summary = {
        "evidence_type": "real_data_preparation_statistics",
        "note": (
            "Aggregated counts from the approved DigiFace-1M subset; no images, identities or ids."
        ),
        "archives": {
            k: {
                f: v[f]
                for f in (
                    "size_bytes",
                    "expected_size_bytes",
                    "size_matches",
                    "sha256",
                    "n_members",
                    "n_files",
                    "crc_ok",
                    "rejected",
                    "uncompressed_bytes",
                )
            }
            for k, v in archives.items()
        },
        "scan": {
            k: scan[k]
            for k in (
                "layout",
                "parts",
                "n_identity_dirs",
                "n_identities_in_manifest",
                "n_images_in_manifest",
                "n_images_excluded",
                "images_per_identity_histogram",
                "split_identity_counts",
                "issue_counts",
            )
        },
        "dedup": {
            k: dedup[k]
            for k in (
                "validation",
                "frontier_train",
                "distance_quantiles",
                "findings",
                "n_scan_records",
                "n_final_records",
                "decode_failures",
                "transform_counts",
            )
        }
        | {"review": {k: v for k, v in dedup["review"].items() if k != "candidates_file"}},
        "image_format": {
            "sizes": {
                f"{w}x{h}": c
                for (w, h), c in Counter(
                    zip(table.width.tolist(), table.height.tolist(), strict=True)
                ).items()
            },
            "modes": dict(Counter(table.mode)),
            "alpha": _alpha_summary(table),
        },
        "final_manifest": {
            "images_per_split": dict(sorted(split_imgs.items())),
            "identities_per_split": {k: len(v) for k, v in sorted(split_ids.items())},
            "images_per_part": dict(per_part),
        },
        "pairs": {
            k: {
                f: v[f]
                for f in (
                    "split",
                    "mode",
                    "n_pairs",
                    "n_genuine",
                    "n_impostor",
                    "n_identities",
                    "max_pairs_per_identity",
                    "identity_disjoint",
                    "effective_independent_pairs",
                    "max_identity_disjoint_subset",
                )
            }
            for k, v in pairs.items()
        },
        "occlusion": occl,
    }
    _dump(PUBLIC / "digiface_phase1b_summary.json", summary)
    print("wrote public summary digiface_phase1b_summary.json")


STAGES = {
    "extract": stage_extract,
    "scan": stage_scan,
    "fingerprint": stage_fingerprint,
    "dedup": stage_dedup,
    "pairs": stage_pairs,
    "occlusion": stage_occlusion,
    "summary": stage_summary,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    args = ap.parse_args()
    for s in args.stages:
        t0 = time.perf_counter()
        print(f"=== stage {s}", flush=True)
        STAGES[s]()
        print(f"=== stage {s} done in {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
