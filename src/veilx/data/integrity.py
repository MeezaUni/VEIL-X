"""Real-image integrity pass: full decode, fingerprints, duplicate validation and grouping.

All outputs that list image ids stay in the private data root; only aggregated counts are
published (``docs/data_integrity_report.md``).
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from veilx.config.schema import SplitName
from veilx.data.dedup import (
    DedupValidation,
    Signals,
    candidate_pairs,
    connected_groups,
    pair_distance,
    random_duplicate,
    signals,
)
from veilx.data.images import decode_rgb, pixel_sha256
from veilx.data.manifest import ImageRecord
from veilx.seeding import make_rng


@dataclass
class FingerprintTable:
    image_ids: list[str]
    ok: NDArray[np.bool_]
    width: NDArray[np.int32]
    height: NDArray[np.int32]
    mode: list[str]
    pixel_sha: list[str]
    phash: NDArray[np.uint64]
    phash_flip: NDArray[np.uint64]
    thumb: NDArray[np.float32]
    thumb_flip: NDArray[np.float32]
    errors: dict[str, str] = field(default_factory=dict)
    # minimum alpha per image (-1 = no alpha channel); RGB conversion drops alpha, so this must
    # be 255 for every image with an alpha channel (verified in the integrity report)
    alpha_min: NDArray[np.int16] = field(default_factory=lambda: np.zeros(0, np.int16))

    def save(self, path: Path) -> None:
        np.savez(
            path,
            image_ids=np.array(self.image_ids),
            ok=self.ok,
            width=self.width,
            height=self.height,
            mode=np.array(self.mode),
            pixel_sha=np.array(self.pixel_sha),
            phash=self.phash,
            phash_flip=self.phash_flip,
            thumb=self.thumb,
            thumb_flip=self.thumb_flip,
            errors=np.array([f"{k}\t{v}" for k, v in self.errors.items()]),
            alpha_min=self.alpha_min,
        )

    @classmethod
    def load(cls, path: Path) -> FingerprintTable:
        z = np.load(path, allow_pickle=False)
        errors = dict(e.split("\t", 1) for e in z["errors"].tolist())
        return cls(
            z["image_ids"].tolist(),
            z["ok"],
            z["width"],
            z["height"],
            z["mode"].tolist(),
            z["pixel_sha"].tolist(),
            z["phash"],
            z["phash_flip"],
            z["thumb"],
            z["thumb_flip"],
            errors,
            z["alpha_min"] if "alpha_min" in z.files else np.full(len(z["ok"]), -1, np.int16),
        )

    def signals_of(self, i: int) -> Signals:
        return Signals(
            int(self.phash[i]), int(self.phash_flip[i]), self.thumb[i], self.thumb_flip[i]
        )


def _fingerprint_one(
    path: str,
) -> tuple[bool, int, int, str, str, int, int, bytes, bytes, str, int]:
    from PIL import Image  # local import keeps worker start-up cheap

    try:
        data = Path(path).read_bytes()
        with Image.open(Path(path)) as im:
            mode = im.mode
            alpha_min = int(np.asarray(im.getchannel("A")).min()) if "A" in im.getbands() else -1
        arr = decode_rgb(data)
        s = signals(arr)
        return (
            True,
            arr.shape[1],
            arr.shape[0],
            mode,
            pixel_sha256(arr),
            s.phash,
            s.phash_flip,
            s.thumb.tobytes(),
            s.thumb_flip.tobytes(),
            "",
            alpha_min,
        )
    except Exception as exc:  # corrupt files must be reported, not crash the pass
        return (False, 0, 0, "", "", 0, 0, b"", b"", f"{type(exc).__name__}: {exc}"[:200], -1)


def fingerprint_records(
    records: Sequence[ImageRecord], root: Path, *, workers: int | None = None
) -> FingerprintTable:
    paths = [str(root / r.relpath) for r in records]
    n = len(paths)
    t = FingerprintTable(
        [r.image_id for r in records],
        np.zeros(n, bool),
        np.zeros(n, np.int32),
        np.zeros(n, np.int32),
        [""] * n,
        [""] * n,
        np.zeros(n, np.uint64),
        np.zeros(n, np.uint64),
        np.zeros((n, 256), np.float32),
        np.zeros((n, 256), np.float32),
        alpha_min=np.full(n, -1, np.int16),
    )
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for i, res in enumerate(pool.map(_fingerprint_one, paths, chunksize=512)):
            ok, w, h, mode, psha, ph, phf, th, thf, err, amin = res
            t.alpha_min[i] = amin
            t.ok[i], t.width[i], t.height[i], t.mode[i], t.pixel_sha[i] = ok, w, h, mode, psha
            if ok:
                t.phash[i], t.phash_flip[i] = np.uint64(ph), np.uint64(phf)
                t.thumb[i] = np.frombuffer(th, np.float32)
                t.thumb_flip[i] = np.frombuffer(thf, np.float32)
            else:
                t.errors[records[i].image_id] = err
            if (i + 1) % 50_000 == 0:
                print(f"  fingerprinted {i + 1:,}/{n:,}", flush=True)
    return t


# --------------------------------------------------------------------------- validation sets

Triple = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def validation_sets(
    records: Sequence[ImageRecord],
    table: FingerprintTable,
    root: Path,
    split: SplitName,
    *,
    seed: int,
    n_each: int,
) -> tuple[Triple, dict[str, int]]:
    """(positives, same-identity negatives, impostor negatives) as (hamming, ncc) arrays."""
    rng = make_rng(seed, "dedup-validation", split.value)
    idx = [i for i, r in enumerate(records) if r.split is split and table.ok[i]]
    by_ident: dict[str, list[int]] = {}
    for i in idx:
        by_ident.setdefault(records[i].identity_id, []).append(i)
    multi = [v for v in by_ident.values() if len(v) >= 2]
    pos, same, imp = [], [], []
    ops: Counter[str] = Counter()
    for i in rng.choice(idx, size=min(n_each, len(idx)), replace=False):
        arr = decode_rgb((root / records[int(i)].relpath).read_bytes())
        dup, op = random_duplicate(arr, rng)
        ops.update(op.split("+"))
        pos.append(pair_distance(table.signals_of(int(i)), signals(dup)))
    for _ in range(n_each):
        group = multi[int(rng.integers(len(multi)))]
        a, b = rng.choice(group, 2, replace=False)
        same.append(pair_distance(table.signals_of(int(a)), table.signals_of(int(b))))
    keys = list(by_ident)
    for _ in range(n_each):
        ka, kb = rng.choice(len(keys), 2, replace=False)
        a = by_ident[keys[int(ka)]][int(rng.integers(len(by_ident[keys[int(ka)]])))]
        b = by_ident[keys[int(kb)]][int(rng.integers(len(by_ident[keys[int(kb)]])))]
        imp.append(pair_distance(table.signals_of(a), table.signals_of(b)))
    as_arr = [np.array(x, dtype=np.float64) for x in (pos, same, imp)]
    return (as_arr[0], as_arr[1], as_arr[2]), dict(ops)


# --------------------------------------------------------------------------- grouping


@dataclass
class DuplicateFindings:
    exact_groups: int
    exact_images: int
    near_candidate_pairs: int
    edges_by_relation: dict[str, int]
    groups_spanning_identities: int
    groups_spanning_splits: int
    removed_non_test_images: int
    removed_ambiguous_label_images: int
    group_of: NDArray[np.int64]


def find_duplicates(
    records: Sequence[ImageRecord],
    table: FingerprintTable,
    validation: DedupValidation,
) -> DuplicateFindings:
    n = len(records)
    edges: list[tuple[int, int]] = []
    by_hash: dict[str, list[int]] = {}
    for i in range(n):
        if table.ok[i]:
            by_hash.setdefault(table.pixel_sha[i], []).append(i)
    exact_groups = [g for g in by_hash.values() if len(g) > 1]
    for g in exact_groups:
        edges.extend((g[0], j) for j in g[1:])
    near: list[tuple[int, int, int, float]] = []
    if validation.accepted and validation.train_rule is not None:
        ok_idx = np.flatnonzero(table.ok)
        found = candidate_pairs(
            table.phash[ok_idx],
            table.phash_flip[ok_idx],
            table.thumb[ok_idx],
            table.thumb_flip[ok_idx],
            hamming_max=validation.train_rule.hamming_max,
            ncc_min=validation.train_rule.ncc_min,
        )
        near = [(int(ok_idx[i]), int(ok_idx[j]), h, c) for i, j, h, c in found]
        edges.extend((i, j) for i, j, _, _ in near)
    relation: Counter[str] = Counter()
    for i, j in edges:
        a, b = records[i], records[j]
        if a.split is not b.split:
            relation["cross_split"] += 1
        elif a.identity_id != b.identity_id:
            relation["cross_identity_same_split"] += 1
        else:
            relation["same_identity"] += 1
    group = connected_groups(n, edges)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(int(group[i]), []).append(i)
    span_ids = span_splits = removed_split = removed_label = 0
    drop: set[int] = set()
    for g in members.values():
        if len(g) < 2:
            continue
        ids = {records[i].identity_id for i in g}
        splits = {records[i].split for i in g}
        if len(ids) > 1:
            span_ids += 1
            drop.update(g)  # same picture under two identities: label is ambiguous
            removed_label += len(g)
        elif len(splits) > 1:
            span_splits += 1
            for i in g:
                if records[i].split is not SplitName.TEST:
                    drop.add(i)
                    removed_split += 1
    for i in drop:
        group[i] = -1  # marker: excluded from the final manifest
    return DuplicateFindings(
        exact_groups=len(exact_groups),
        exact_images=sum(len(g) for g in exact_groups),
        near_candidate_pairs=len(near),
        edges_by_relation=dict(relation),
        groups_spanning_identities=span_ids,
        groups_spanning_splits=span_splits,
        removed_non_test_images=removed_split,
        removed_ambiguous_label_images=removed_label,
        group_of=group,
    )


def finalize_records(
    records: Sequence[ImageRecord], table: FingerprintTable, findings: DuplicateFindings
) -> list[ImageRecord]:
    """Attach pixel hashes and root-photo groups; drop failed/removed images."""
    out = []
    for i, r in enumerate(records):
        if not table.ok[i] or findings.group_of[i] < 0:
            continue
        rep = int(findings.group_of[i])
        out.append(
            r.model_copy(
                update={
                    "pixel_sha256": table.pixel_sha[i],
                    "source_group": table.pixel_sha[rep],
                }
            )
        )
    return out
