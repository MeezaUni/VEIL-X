"""Dataset and pair manifest schemas.

Manifests are JSON-lines files describing images and pairs *by reference*: they contain
pseudonymous ids, dataset-relative paths and hashes, never pixels or embeddings. They live
under the git-ignored ``data/`` tree.

Integrity rules enforced here (structural, no data needed):

* image ids are unique; relative paths are POSIX, relative and contain no ``..``;
* every identity maps to exactly one split (identity-disjoint partitions);
* an occluded variant inherits the split and identity of its source image;
* a pair references two distinct images of the same split; its label agrees with the
  identities; its stratum agrees with the images' occlusion state.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from veilx.config.schema import APPROVED_FAMILY_IDS, SplitName
from veilx.errors import LeakageError, ManifestError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OcclusionSpec(_Record):
    """Fully reproducible description of a synthetic challenge applied to an image."""

    family: str
    asset_id: str | None = None
    geometry: dict[str, float] = Field(default_factory=dict)
    seed: int = Field(ge=0)
    generator_version: str
    severity_level: Literal["low", "medium", "high"] | None = None
    severity: float | None = Field(default=None, ge=0.0, le=1.0)  # occluded face fraction
    visibility_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    physical: bool | None = None  # False for photometric (lighting) challenges
    recipe_sha256: str | None = None  # hash of (source pixels, family, params, version)

    @field_validator("family")
    @classmethod
    def _approved(cls, v: str) -> str:
        if v not in APPROVED_FAMILY_IDS:
            raise ValueError(f"unknown challenge family {v!r}")
        return v


class ImageRecord(_Record):
    image_id: str
    dataset_id: str
    identity_id: str
    relpath: str
    sha256: str
    split: SplitName
    source_image_id: str | None = None
    occlusion: OcclusionSpec | None = None
    real_occlusion_label: str | None = None  # e.g. ROF "masked"/"sunglasses"/"neutral"
    pixel_sha256: str | None = None  # hash of decoded pixels (exact-duplicate key)
    source_group: str | None = None  # root-photo group after exact/near-duplicate merging

    @field_validator("image_id", "dataset_id", "identity_id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"invalid id {v!r}")
        return v

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not _SHA256.match(v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v

    @field_validator("relpath")
    @classmethod
    def _relative_posix(cls, v: str) -> str:
        p = PurePosixPath(v)
        if "\\" in v or p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", v):
            raise ValueError("relpath must be a dataset-relative POSIX path without '..'")
        return v

    @property
    def is_challenged(self) -> bool:
        return self.occlusion is not None or self.real_occlusion_label not in (None, "neutral")


PairLabel = Literal["genuine", "impostor"]
PairStratum = Literal[
    "clean_clean",
    "clean_occluded",
    "occluded_occluded_same_family",
    "occluded_occluded_cross_family",
]


class PairRecord(_Record):
    pair_id: str
    image_a: str
    image_b: str
    label: PairLabel
    stratum: PairStratum
    split: SplitName
    protocol: str = "veilx"  # or an official benchmark protocol name, reported separately

    @field_validator("pair_id", "image_a", "image_b")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"invalid id {v!r}")
        return v


# --------------------------------------------------------------------------- integrity


def validate_image_manifest(records: Sequence[ImageRecord]) -> None:
    """Raise :class:`ManifestError` / :class:`LeakageError` if structural rules are violated."""
    by_id: dict[str, ImageRecord] = {}
    for rec in records:
        if rec.image_id in by_id:
            raise ManifestError(f"duplicate image_id {rec.image_id!r}")
        by_id[rec.image_id] = rec

    identity_split: dict[tuple[str, str], SplitName] = {}
    for rec in records:
        key = (rec.dataset_id, rec.identity_id)
        prev = identity_split.setdefault(key, rec.split)
        if prev is not rec.split:
            raise LeakageError(
                f"identity {rec.dataset_id}:{rec.identity_id} appears in splits "
                f"{prev.value} and {rec.split.value}"
            )

    for rec in records:
        if rec.source_image_id is None:
            if rec.occlusion is not None:
                raise ManifestError(f"{rec.image_id}: synthetic occlusion requires source_image_id")
            continue
        src = by_id.get(rec.source_image_id)
        if src is None:
            raise ManifestError(f"{rec.image_id}: unknown source_image_id {rec.source_image_id!r}")
        if src.split is not rec.split:
            raise LeakageError(f"{rec.image_id}: variant split differs from its source image")
        if (src.dataset_id, src.identity_id) != (rec.dataset_id, rec.identity_id):
            raise ManifestError(f"{rec.image_id}: variant identity differs from its source")


def _image_family(rec: ImageRecord) -> str | None:
    if rec.occlusion is not None:
        return rec.occlusion.family
    if rec.real_occlusion_label not in (None, "neutral"):
        return f"real:{rec.real_occlusion_label}"
    return None


def expected_stratum(a: ImageRecord, b: ImageRecord) -> PairStratum:
    fa, fb = _image_family(a), _image_family(b)
    if fa is None and fb is None:
        return "clean_clean"
    if fa is None or fb is None:
        return "clean_occluded"
    return "occluded_occluded_same_family" if fa == fb else "occluded_occluded_cross_family"


def validate_pair_manifest(pairs: Sequence[PairRecord], images: Sequence[ImageRecord]) -> None:
    """Check pairs against their image manifest (split purity, labels, strata, duplicates)."""
    by_id = {r.image_id: r for r in images}
    seen_ids: set[str] = set()
    seen_unordered: set[tuple[str, str, str]] = set()
    for p in pairs:
        if p.pair_id in seen_ids:
            raise ManifestError(f"duplicate pair_id {p.pair_id!r}")
        seen_ids.add(p.pair_id)
        if p.image_a == p.image_b:
            raise ManifestError(f"{p.pair_id}: self-pair")
        try:
            a, b = by_id[p.image_a], by_id[p.image_b]
        except KeyError as exc:
            raise ManifestError(f"{p.pair_id}: unknown image {exc.args[0]!r}") from exc
        if not (a.split is b.split is p.split):
            raise LeakageError(f"{p.pair_id}: pair crosses splits")
        same = (a.dataset_id, a.identity_id) == (b.dataset_id, b.identity_id)
        if (p.label == "genuine") != same:
            raise ManifestError(f"{p.pair_id}: label {p.label!r} contradicts identities")
        if p.stratum != expected_stratum(a, b):
            raise ManifestError(f"{p.pair_id}: stratum {p.stratum!r} != {expected_stratum(a, b)!r}")
        key = (p.protocol, *sorted((p.image_a, p.image_b)))
        if key in seen_unordered:
            raise ManifestError(f"{p.pair_id}: duplicate unordered pair within protocol")
        seen_unordered.add(key)  # type: ignore[arg-type]


def assert_family_isolation(
    records: Iterable[ImageRecord], heldout_family: str, *, splits: Iterable[SplitName]
) -> None:
    """Raise :class:`LeakageError` if ``heldout_family`` occurs in any of ``splits``."""
    guarded = set(splits)
    for rec in records:
        if (
            rec.split in guarded
            and rec.occlusion is not None
            and rec.occlusion.family == heldout_family
        ):
            raise LeakageError(
                f"held-out family {heldout_family} present in {rec.split.value} ({rec.image_id})"
            )


# --------------------------------------------------------------------------- JSONL IO

_T = TypeVar("_T", bound=_Record)


def write_jsonl(path: Path, records: Iterable[_Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[_T]) -> Iterator[_T]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line)
            except ValidationError as exc:
                raise ManifestError(f"{path.name}:{lineno}: {exc.errors()[0]['msg']}") from exc
