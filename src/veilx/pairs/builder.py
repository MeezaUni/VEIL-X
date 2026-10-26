"""Reproducible genuine/impostor pair construction (protocol_design.md Â§4).

Two modes:

* ``independent`` â€” an **identity-disjoint matching**: every identity (hence every image, source
  image and occlusion variant) appears in at most one pair. Genuine pairs consume one identity,
  impostor pairs two. Under i.i.d. sampling of identities the resulting pair outcomes are
  independent, which is what binomial risk-control tests require (assumption A2). Requests the
  data cannot satisfy raise :class:`InfeasibleRequestError` â€” they are never silently shrunk.
* ``clustered`` â€” the usual evaluation design: identities and images are reused across pairs
  (genuine pairs capped per identity). Pairs are **not** independent; the summary reports
  ``effective_independent_pairs = None`` and the number of identity clusters, and downstream
  inference must use the identity-cluster bootstrap.

Universal rules: pairs come from a single split; genuine pairs use two *different source images*
with *different content hashes* (two variants of one photo, or a duplicate file, are never a
genuine pair); no duplicate unordered pairs; every pair is re-validated against the manifest.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from veilx.config.schema import SplitName
from veilx.data.manifest import (
    ImageRecord,
    PairRecord,
    expected_stratum,
    validate_image_manifest,
    validate_pair_manifest,
)
from veilx.errors import InfeasibleRequestError, LeakageError, ProtocolViolationError
from veilx.seeding import make_rng

PAIR_GENERATOR_VERSION = "1.0.0"
IdentityKey = tuple[str, str]


@dataclass(frozen=True)
class PairSpec:
    mode: Literal["independent", "clustered"]
    n_pairs: int
    genuine_fraction: float
    split: SplitName
    seed: int
    class_assignment: Literal["fixed", "bernoulli"] = "fixed"
    genuine_cap_per_identity: int = 20
    max_impostor_pairs_per_identity: int | None = None
    protocol: str = "veilx"
    pair_id_prefix: str = "p"

    def __post_init__(self) -> None:
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be >= 1")
        if not 0.0 <= self.genuine_fraction <= 1.0:
            raise ValueError("genuine_fraction must be in [0, 1]")
        if self.genuine_cap_per_identity < 1:
            raise ValueError("genuine_cap_per_identity must be >= 1")
        if (
            self.max_impostor_pairs_per_identity is not None
            and self.max_impostor_pairs_per_identity < 1
        ):
            raise ValueError("max_impostor_pairs_per_identity must be >= 1")


@dataclass(frozen=True)
class PairSetSummary:
    mode: str
    n_pairs: int
    n_genuine: int
    n_impostor: int
    n_identities: int
    n_images: int
    max_pairs_per_identity: int
    max_pairs_per_image: int
    max_pairs_per_source_image: int
    identity_disjoint: bool
    effective_independent_pairs: int | None
    max_identity_disjoint_subset: int
    strata: dict[str, int]
    note: str


@dataclass(frozen=True)
class PairSet:
    pairs: tuple[PairRecord, ...]
    summary: PairSetSummary
    provenance: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def source_keys(images: Sequence[ImageRecord]) -> dict[str, str]:
    """Map image_id -> content hash of its *root* source photo.

    Occlusion variants follow ``source_image_id`` back to the original photo, and byte-identical
    copies share a content hash, so variants and duplicates of one photo get the same key and can
    never form a "genuine" pair or be counted as independent observations.
    """
    by_id = {r.image_id: r for r in images}
    out: dict[str, str] = {}
    for r in images:
        root, seen = r, {r.image_id}
        while root.source_image_id is not None and root.source_image_id in by_id:
            root = by_id[root.source_image_id]
            if root.image_id in seen:
                break  # defensive: cycles are rejected by manifest validation anyway
            seen.add(root.image_id)
        out[r.image_id] = root.sha256
    return out


def _ident(r: ImageRecord) -> IdentityKey:
    return (r.dataset_id, r.identity_id)


def manifest_digest(images: Sequence[ImageRecord]) -> str:
    h = hashlib.sha256()
    for r in sorted(images, key=lambda r: r.image_id):
        h.update(f"{r.image_id}\t{r.sha256}\t{r.split.value}\n".encode())
    return h.hexdigest()


def pairs_digest(pairs: Sequence[PairRecord]) -> str:
    h = hashlib.sha256()
    for p in pairs:
        h.update((p.model_dump_json() + "\n").encode())
    return h.hexdigest()


_ENUMERATE_MAX = 40  # identities with more images are sampled, not enumerated


def _valid_genuine(a: ImageRecord, b: ImageRecord, src: Mapping[str, str]) -> bool:
    return src[a.image_id] != src[b.image_id] and a.sha256 != b.sha256


def _has_genuine(imgs: Sequence[ImageRecord], src: Mapping[str, str]) -> bool:
    return any(_valid_genuine(a, b, src) for a, b in itertools.combinations(imgs, 2))


def _sample_genuine(
    imgs: Sequence[ImageRecord], k: int, rng: np.random.Generator, src: Mapping[str, str]
) -> list[tuple[ImageRecord, ImageRecord]]:
    """Up to ``k`` distinct valid genuine pairs, uniformly without replacement."""
    n = len(imgs)
    if n <= _ENUMERATE_MAX:
        cands = [(a, b) for a, b in itertools.combinations(imgs, 2) if _valid_genuine(a, b, src)]
        take = min(k, len(cands))
        return (
            [cands[j] for j in sorted(rng.choice(len(cands), take, replace=False))] if take else []
        )
    found: dict[tuple[int, int], tuple[ImageRecord, ImageRecord]] = {}
    attempts = 0
    while len(found) < k and attempts < 100 * k + 1000:
        attempts += 1
        i, j = sorted(int(x) for x in rng.choice(n, 2, replace=False))
        if (i, j) not in found and _valid_genuine(imgs[i], imgs[j], src):
            found[(i, j)] = (imgs[i], imgs[j])
    if len(found) < k:  # pathological groups (many duplicates/variants): exact fallback
        cands = [(a, b) for a, b in itertools.combinations(imgs, 2) if _valid_genuine(a, b, src)]
        take = min(k, len(cands))
        return (
            [cands[j] for j in sorted(rng.choice(len(cands), take, replace=False))] if take else []
        )
    return [found[key] for key in sorted(found)]


def _class_counts(spec: PairSpec, rng: np.random.Generator) -> tuple[int, int]:
    if spec.class_assignment == "fixed":
        n_g = round(spec.n_pairs * spec.genuine_fraction)
    else:
        n_g = int(rng.binomial(spec.n_pairs, spec.genuine_fraction))
    return n_g, spec.n_pairs - n_g


def summarize(
    pairs: Sequence[PairRecord], images: Sequence[ImageRecord], mode: str
) -> PairSetSummary:
    by_id = {r.image_id: r for r in images}
    src = source_keys(images)
    ident_use: Counter[IdentityKey] = Counter()
    img_use: Counter[str] = Counter()
    src_use: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    n_g = 0
    for p in pairs:
        a, b = by_id[p.image_a], by_id[p.image_b]
        for ident in {_ident(a), _ident(b)}:
            ident_use[ident] += 1
        img_use[a.image_id] += 1
        img_use[b.image_id] += 1
        src_use[src[a.image_id]] += 1
        src_use[src[b.image_id]] += 1
        strata[p.stratum] += 1
        n_g += p.label == "genuine"
    disjoint = all(c == 1 for c in ident_use.values())
    used: set[IdentityKey] = set()
    greedy = 0
    for p in pairs:
        ids = {_ident(by_id[p.image_a]), _ident(by_id[p.image_b])}
        if not ids & used:
            used |= ids
            greedy += 1
    note = (
        "identity-disjoint: pair outcomes independent under i.i.d. identity sampling"
        if disjoint
        else "pairs share identities/images: NOT independent; use identity-cluster bootstrap"
    )
    return PairSetSummary(
        mode=mode,
        n_pairs=len(pairs),
        n_genuine=n_g,
        n_impostor=len(pairs) - n_g,
        n_identities=len(ident_use),
        n_images=len(img_use),
        max_pairs_per_identity=max(ident_use.values(), default=0),
        max_pairs_per_image=max(img_use.values(), default=0),
        max_pairs_per_source_image=max(src_use.values(), default=0),
        identity_disjoint=disjoint,
        effective_independent_pairs=len(pairs) if disjoint else None,
        max_identity_disjoint_subset=greedy,
        strata=dict(sorted(strata.items())),
        note=note,
    )


def assert_independent(pair_set: PairSet) -> None:
    """Raise unless the pair set may be treated as independent observations."""
    s = pair_set.summary
    if not s.identity_disjoint or s.max_pairs_per_source_image > 1:
        raise ProtocolViolationError(
            f"pair set is not identity-disjoint ({s.n_pairs} pairs over {s.n_identities} "
            f"identities, max {s.max_pairs_per_identity} pairs/identity); "
            "it cannot be used where independent observations are assumed"
        )


# --------------------------------------------------------------------------- builder


def build_pairs(images: Sequence[ImageRecord], spec: PairSpec) -> PairSet:
    validate_image_manifest(images)
    wrong = [r.image_id for r in images if r.split is not spec.split]
    if wrong:
        raise LeakageError(
            f"{len(wrong)} image(s) are not in split {spec.split.value}: {wrong[:3]}"
        )
    rng = make_rng(spec.seed, "pairs", spec.split.value, spec.mode, spec.protocol)
    groups: dict[IdentityKey, list[ImageRecord]] = defaultdict(list)
    for r in sorted(images, key=lambda r: r.image_id):
        groups[_ident(r)].append(r)
    identities = sorted(groups)
    src = source_keys(images)
    n_g, n_i = _class_counts(spec, rng)

    if spec.mode == "independent":
        raw = _independent(groups, identities, n_g, n_i, rng, src)
    else:
        raw = _clustered(groups, identities, n_g, n_i, spec, rng, src)

    order = rng.permutation(len(raw))
    pairs = tuple(
        PairRecord(
            pair_id=f"{spec.pair_id_prefix}-{spec.split.value}-{spec.seed}-{k:07d}",
            image_a=raw[j][0].image_id,
            image_b=raw[j][1].image_id,
            label=raw[j][2],
            stratum=expected_stratum(raw[j][0], raw[j][1]),
            split=spec.split,
            protocol=spec.protocol,
        )
        for k, j in enumerate(order)
    )
    validate_pair_manifest(pairs, images)
    summary = summarize(pairs, images, spec.mode)
    if spec.mode == "independent":
        assert_independent(PairSet(pairs, summary))
    provenance: dict[str, object] = {
        "generator": "veilx.pairs.builder",
        "generator_version": PAIR_GENERATOR_VERSION,
        "spec": {k: (v.value if isinstance(v, SplitName) else v) for k, v in asdict(spec).items()},
        "manifest_digest": manifest_digest(images),
        "pairs_digest": pairs_digest(pairs),
        "summary": asdict(summary),
    }
    return PairSet(pairs, summary, json.loads(json.dumps(provenance)))


Raw = list[tuple[ImageRecord, ImageRecord, Literal["genuine", "impostor"]]]


def _independent(
    groups: dict[IdentityKey, list[ImageRecord]],
    identities: list[IdentityKey],
    n_g: int,
    n_i: int,
    rng: np.random.Generator,
    src: Mapping[str, str],
) -> Raw:
    eligible = [i for i in identities if _has_genuine(groups[i], src)]
    n_total = len(identities)
    if n_g > len(eligible) or n_g + 2 * n_i > n_total:
        raise InfeasibleRequestError(
            f"cannot build {n_g} genuine + {n_i} impostor identity-disjoint pairs: "
            f"{len(eligible)} identities support a genuine pair, {n_total} identities in total "
            f"(need n_genuine <= {len(eligible)} and n_genuine + 2*n_impostor <= {n_total})"
        )
    gen_ids = [eligible[j] for j in rng.permutation(len(eligible))[:n_g]]
    chosen = set(gen_ids)
    rest = [i for i in identities if i not in chosen]
    rest = [rest[j] for j in rng.permutation(len(rest))[: 2 * n_i]]
    out: Raw = []
    for ident in gen_ids:
        ((a, b),) = _sample_genuine(groups[ident], 1, rng, src)
        out.append((a, b, "genuine"))
    for k in range(n_i):
        ga, gb = groups[rest[2 * k]], groups[rest[2 * k + 1]]
        out.append((ga[int(rng.integers(len(ga)))], gb[int(rng.integers(len(gb)))], "impostor"))
    return out


def _clustered(
    groups: dict[IdentityKey, list[ImageRecord]],
    identities: list[IdentityKey],
    n_g: int,
    n_i: int,
    spec: PairSpec,
    rng: np.random.Generator,
    src: Mapping[str, str],
) -> Raw:
    pool: list[tuple[ImageRecord, ImageRecord]] = []
    for ident in identities:
        pool.extend(_sample_genuine(groups[ident], spec.genuine_cap_per_identity, rng, src))
    if n_g > len(pool):
        raise InfeasibleRequestError(
            f"requested {n_g} genuine pairs but only {len(pool)} are available "
            f"(cap {spec.genuine_cap_per_identity}/identity, distinct sources and content)"
        )
    out: Raw = [(*pool[j], "genuine") for j in sorted(rng.choice(len(pool), n_g, replace=False))]

    sizes = np.array([len(groups[i]) for i in identities], dtype=np.int64)
    capacity = (int(sizes.sum()) ** 2 - int((sizes**2).sum())) // 2
    if len(identities) < 2 or n_i > capacity:
        raise InfeasibleRequestError(
            f"requested {n_i} impostor pairs but only {capacity} distinct cross-identity "
            f"image pairs exist over {len(identities)} identities"
        )
    cap = spec.max_impostor_pairs_per_identity
    if cap is not None and 2 * n_i > cap * len(identities):
        raise InfeasibleRequestError(
            f"{n_i} impostor pairs need {2 * n_i} identity slots but the cap allows "
            f"{cap} x {len(identities)}"
        )
    seen: set[tuple[str, str]] = set()
    use: Counter[IdentityKey] = Counter()
    attempts, max_attempts = 0, 50 * n_i + 1000
    while len(seen) < n_i:
        attempts += 1
        if attempts > max_attempts:
            raise InfeasibleRequestError(
                f"could only sample {len(seen)} of {n_i} distinct impostor pairs "
                "within the attempt budget (identity cap or tiny pools)"
            )
        ia, ib = rng.choice(len(identities), 2, replace=False)
        ka, kb = identities[int(ia)], identities[int(ib)]
        if cap is not None and (use[ka] >= cap or use[kb] >= cap):
            continue
        a = groups[ka][int(rng.integers(len(groups[ka])))]
        b = groups[kb][int(rng.integers(len(groups[kb])))]
        key = (min(a.image_id, b.image_id), max(a.image_id, b.image_id))
        if key in seen:
            continue
        seen.add(key)
        use[ka] += 1
        use[kb] += 1
        out.append((a, b, "impostor"))
    return out
