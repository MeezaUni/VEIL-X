"""Identity-level, hash-based split assignment.

``bucket = int(sha256(f"{dataset}:{identity_id}")[:8], 16) % 100``; the bucket falls into a
half-open band ``[lo, hi)`` defined in the research protocol. Assignment is deterministic,
seed-free, independent of iteration order and stable when identities are added.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping

from veilx.config.schema import SplitName


def identity_bucket(dataset_id: str, identity_id: str) -> int:
    digest = hashlib.sha256(f"{dataset_id}:{identity_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(
    dataset_id: str, identity_id: str, bands: Mapping[SplitName, tuple[int, int]]
) -> SplitName:
    bucket = identity_bucket(dataset_id, identity_id)
    for name, (lo, hi) in bands.items():
        if lo <= bucket < hi:
            return name
    raise ValueError(f"bucket {bucket} not covered by bands")  # unreachable with validated bands


def split_counts(
    dataset_id: str, identity_ids: Iterable[str], bands: Mapping[SplitName, tuple[int, int]]
) -> Counter[SplitName]:
    return Counter(assign_split(dataset_id, i, bands) for i in identity_ids)
