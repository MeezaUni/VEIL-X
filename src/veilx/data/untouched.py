"""Registered untouched confirmatory partitions of DigiFace-1M (72-image identities).

Each partition is kept in its own private manifest / extraction / feature directory, so earlier
partitions stay bit-identical. ``P4`` is the Phase 4 partition (``docs/phase4_protocol.md``);
``P6`` is the Phase 6 replication partition (``docs/phase6_protocol.md``), registered before any of
its images was downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UntouchedPartition:
    key: str  # CLI name: "p4" / "p6"
    directory: str  # manifests/<dir>, extracted/<dir>, features/<dir>/<key>
    split: str  # name of the test split in run logs, e.g. "p4_test"
    pair_prefix: str  # pair_id prefix
    seed: int  # pair-building and occlusion-plan seed
    parts: tuple[tuple[str, int], ...]  # (archive name, official Content-Length)
    earlier: tuple[str, ...]  # manifests of earlier partitions checked for duplicates (dedup)
    scope: str

    @property
    def pairs_name(self) -> str:
        return f"{self.split}_pairs"

    @property
    def clean_group(self) -> str:
        return f"{self.key}_clean"


P4 = UntouchedPartition(
    key="p4",
    directory="phase4",
    split="p4_test",
    pair_prefix="p4te",
    seed=20261101,
    parts=(
        ("subjects_2000-3999_72_imgs.zip", 2_936_098_225),
        ("subjects_4000-5999_72_imgs.zip", 2_936_895_527),
    ),
    earlier=(),
    scope="Phase 4 untouched partition: DigiFace-1M 72-image parts 2000-3999, 4000-5999",
)

P6 = UntouchedPartition(
    key="p6",
    directory="phase6",
    split="p6_test",
    pair_prefix="p6te",
    seed=20261102,
    parts=(
        ("subjects_6000-7999_72_imgs.zip", 2_940_940_004),  # HTTP HEAD 2026-11-02
        ("subjects_8000-9999_72_imgs.zip", 2_927_632_784),  # HTTP HEAD 2026-11-02
    ),
    earlier=("phase4/images.jsonl",),
    scope="Phase 6 untouched replication partition: DigiFace-1M 72-image parts 6000-9999",
)

PARTITIONS = {p.key: p for p in (P4, P6)}
