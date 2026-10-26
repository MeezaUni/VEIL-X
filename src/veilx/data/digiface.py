"""DigiFace-1M dataset adapter (Bae et al., WACV 2023; licence R-UDA, non-commercial research).

Supported on-disk layouts (as documented in the official repository README: per-subject folders
containing sequentially numbered PNGs ``0.png, 1.png, ...``):

* **parts layout**: ``<root>/subjects_<lo>-<hi>_<k>_imgs/<subject_id>/<index>.png``
  (one directory per downloaded zip part), or
* **flat layout**: ``<root>/<subject_id>/<index>.png``.

Expected images per identity follow the official ID ranges: IDs 0-9,999 have 72 images, IDs
100,000-199,998 have 5. Anything else is reported, never silently accepted.

The adapter never stores pixels. It produces :class:`ImageRecord` rows (relative path, SHA-256
of file bytes, identity-hash split) plus a :class:`DigiFaceScanReport` listing every anomaly:
non-numeric or out-of-range identity folders, identities split across parts, empty and
single-image identities, missing/unexpected indices, non-PNG files, corrupt PNGs and exact
duplicate files. Corrupt files and later copies of duplicate content are excluded from the
manifest; everything excluded is listed in the report.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from veilx.config.schema import AccessStatus, DatasetRegistry, SplitName
from veilx.data.manifest import ImageRecord, PairRecord, write_jsonl
from veilx.data.png import check_png_bytes
from veilx.data.splits import assign_split
from veilx.errors import DataAccessError, ManifestError

DATASET_ID = "digiface1m"
ADAPTER_VERSION = "1.0.0"
_PART_RE = re.compile(r"^subjects_(\d+)-(\d+)_(\d+)_imgs$")
_IMG_RE = re.compile(r"^(\d+)\.png$")
_ID_RE = re.compile(r"^\d+$")

IssueKind = Literal[
    "non_numeric_identity_dir",
    "identity_outside_official_ranges",
    "identity_outside_part_range",
    "identity_in_multiple_parts",
    "empty_identity",
    "single_image_identity",
    "missing_index",
    "unexpected_index",
    "unexpected_file",
    "corrupt_image",
    "duplicate_content",
    "duplicate_content_cross_identity",
    "unreadable_file",
]


def expected_images_per_identity(identity: int) -> int | None:
    """Official DigiFace-1M image count for an identity ID (``None`` if outside both ranges)."""
    if 0 <= identity <= 9_999:
        return 72
    if 100_000 <= identity <= 199_998:
        return 5
    return None


@dataclass(frozen=True)
class ScanIssue:
    kind: IssueKind
    identity_id: str | None
    detail: str


@dataclass
class DigiFaceScanReport:
    layout: Literal["parts", "flat", "empty"]
    parts: list[str] = field(default_factory=list)
    n_identity_dirs: int = 0
    n_identities_in_manifest: int = 0
    n_images_in_manifest: int = 0
    n_images_excluded: int = 0
    images_per_identity_histogram: dict[int, int] = field(default_factory=dict)
    split_identity_counts: dict[str, int] = field(default_factory=dict)
    issues: list[ScanIssue] = field(default_factory=list)

    def count(self, kind: IssueKind) -> int:
        return sum(1 for i in self.issues if i.kind == kind)

    def issue_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(i.kind for i in self.issues).items()))

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["issue_counts"] = self.issue_counts()
        return d


class DigiFaceAdapter:
    """Deterministic manifest builder for a local DigiFace-1M copy (implements DatasetAdapter)."""

    dataset_id = DATASET_ID

    def __init__(
        self,
        root: Path,
        bands: Mapping[SplitName, tuple[int, int]],
        registry: DatasetRegistry,
        *,
        verify_png: bool = True,
        io_workers: int = 16,
    ) -> None:
        self.root = root
        self.bands = dict(bands)
        self.registry = registry
        self.verify_png = verify_png
        self.io_workers = io_workers
        self._prefetched: dict[Path, tuple[IssueKind | Literal["ok"], str, str]] = {}
        self._ambiguous: set[str] = set()
        self._cache: tuple[list[ImageRecord], DigiFaceScanReport] | None = None

    # -- DatasetAdapter -------------------------------------------------------------------
    def check_access(self) -> None:
        try:
            entry = self.registry.get(self.dataset_id)
        except KeyError as exc:
            raise DataAccessError(f"{self.dataset_id} is not in the dataset registry") from exc
        if entry.access_status is not AccessStatus.CONFIRMED:
            raise DataAccessError(
                f"{self.dataset_id} access status is {entry.access_status.value}, not confirmed"
            )
        if not self.root.is_dir():
            raise DataAccessError(f"DigiFace root directory not found: {self.root.name}")

    def iter_images(self) -> Iterator[ImageRecord]:
        yield from self.scan()[0]

    def official_pairs(self) -> Sequence[PairRecord]:
        return ()  # DigiFace-1M defines no verification protocol

    # -- scanning -------------------------------------------------------------------------
    def scan(self) -> tuple[list[ImageRecord], DigiFaceScanReport]:
        if self._cache is None:
            self.check_access()
            self._cache = self._scan()
        return self._cache

    def _identity_dirs(
        self, report: DigiFaceScanReport
    ) -> list[tuple[Path, str | None, tuple[int, int] | None]]:
        children = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.name)
        part_dirs = [p for p in children if _PART_RE.match(p.name)]
        out: list[tuple[Path, str | None, tuple[int, int] | None]] = []
        if part_dirs:
            report.layout = "parts"
            for part in part_dirs:
                m = _PART_RE.match(part.name)
                assert m is not None
                report.parts.append(part.name)
                lo, hi = int(m.group(1)), int(m.group(2))
                out.extend(
                    (d, part.name, (lo, hi))
                    for d in sorted(part.iterdir(), key=lambda p: p.name)
                    if d.is_dir()
                )
            for other in children:
                if other not in part_dirs:
                    report.issues.append(
                        ScanIssue("unexpected_file", None, f"non-part directory {other.name}")
                    )
        elif children:
            report.layout = "flat"
            out = [(d, None, None) for d in children]
        else:
            report.layout = "empty"
        return out

    def _scan(self) -> tuple[list[ImageRecord], DigiFaceScanReport]:
        report = DigiFaceScanReport(layout="empty")
        records: list[ImageRecord] = []
        seen_ids: dict[str, str | None] = {}
        content_owner: dict[str, str] = {}
        per_identity: Counter[int] = Counter()
        split_ids: Counter[str] = Counter()
        processed: list[str] = []
        self._ambiguous = set()

        dirs = self._identity_dirs(report)
        report.n_identity_dirs = len(dirs)
        # Sort numerically for deterministic output regardless of part ordering.
        dirs.sort(key=lambda t: (int(t[0].name) if _ID_RE.match(t[0].name) else -1, t[0].name))
        # Prefetch per-file checks with a thread pool (file I/O dominates on real data); results
        # are consumed in the same deterministic order as the serial path.
        self._prefetched = {}
        if self.io_workers > 1:
            files = [
                f
                for d, _, _ in dirs
                if _ID_RE.match(d.name)
                for f in d.iterdir()
                if _IMG_RE.match(f.name)
            ]
            with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
                self._prefetched = dict(zip(files, pool.map(self._read_check, files), strict=True))

        for ident_dir, part, part_range in dirs:
            name = ident_dir.name
            if not _ID_RE.match(name):
                report.issues.append(ScanIssue("non_numeric_identity_dir", None, name))
                continue
            identity_num = int(name)
            identity_id = str(identity_num)
            if identity_id in seen_ids:
                report.issues.append(
                    ScanIssue(
                        "identity_in_multiple_parts",
                        identity_id,
                        f"also in {seen_ids[identity_id]}; second copy ignored",
                    )
                )
                continue
            seen_ids[identity_id] = part
            expected = expected_images_per_identity(identity_num)
            if expected is None:
                report.issues.append(
                    ScanIssue("identity_outside_official_ranges", identity_id, "count unchecked")
                )
            if part_range is not None and not part_range[0] <= identity_num <= part_range[1]:
                report.issues.append(
                    ScanIssue("identity_outside_part_range", identity_id, f"in {part}")
                )

            self._scan_identity(ident_dir, identity_id, expected, report, content_owner, records)
            processed.append(identity_id)

        # Cross-identity duplicates: drop the first copy too (its label is ambiguous), then derive
        # per-identity statistics from the final records.
        if self._ambiguous:
            report.n_images_excluded += sum(r.image_id in self._ambiguous for r in records)
            records = [r for r in records if r.image_id not in self._ambiguous]
        kept_by_id = Counter(r.identity_id for r in records)
        for identity_id in processed:
            kept = kept_by_id.get(identity_id, 0)
            per_identity[kept] += 1
            if kept == 0:
                report.issues.append(ScanIssue("empty_identity", identity_id, "no valid images"))
                continue
            if kept == 1:
                report.issues.append(
                    ScanIssue("single_image_identity", identity_id, "no genuine pairs possible")
                )
            split_ids[assign_split(self.dataset_id, identity_id, self.bands).value] += 1

        report.n_images_in_manifest = len(records)
        report.n_identities_in_manifest = sum(split_ids.values())
        report.images_per_identity_histogram = dict(sorted(per_identity.items()))
        report.split_identity_counts = dict(sorted(split_ids.items()))
        return records, report

    def _read_check(self, path: Path) -> tuple[IssueKind | Literal["ok"], str, str]:
        """(status, detail, sha256) for one file; I/O-bound, safe to run in threads."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            return "unreadable_file", exc.__class__.__name__, ""
        if self.verify_png:
            chk = check_png_bytes(data)
            if not chk.ok:
                return "corrupt_image", chk.error or "invalid PNG", ""
        return "ok", "", hashlib.sha256(data).hexdigest()

    def _scan_identity(
        self,
        ident_dir: Path,
        identity_id: str,
        expected: int | None,
        report: DigiFaceScanReport,
        content_owner: dict[str, str],
        records: list[ImageRecord],
    ) -> int:
        indices: dict[int, Path] = {}
        for f in sorted(ident_dir.iterdir(), key=lambda p: p.name):
            m = _IMG_RE.match(f.name)
            if not f.is_file() or m is None:
                report.issues.append(ScanIssue("unexpected_file", identity_id, f.name))
                continue
            indices[int(m.group(1))] = f
        if expected is not None:
            missing = sorted(set(range(expected)) - set(indices))
            extra = sorted(i for i in indices if i >= expected)
            if missing:
                report.issues.append(
                    ScanIssue("missing_index", identity_id, f"missing {_compact(missing)}")
                )
            if extra:
                report.issues.append(
                    ScanIssue("unexpected_index", identity_id, f"extra {_compact(extra)}")
                )
        split = assign_split(self.dataset_id, identity_id, self.bands)
        kept = 0
        paths = [indices[idx] for idx in sorted(indices)]
        checked = [
            self._prefetched.pop(p) if p in self._prefetched else self._read_check(p) for p in paths
        ]
        for idx, path, (status, detail, digest) in zip(
            sorted(indices), paths, checked, strict=True
        ):
            if status != "ok":
                report.issues.append(ScanIssue(status, identity_id, f"{path.name}: {detail}"))
                report.n_images_excluded += 1
                continue
            image_id = f"{self.dataset_id}:{identity_id}:{idx}"
            if digest in content_owner:
                owner = content_owner[digest]
                if owner.split(":")[1] == identity_id:  # redundant copy within one identity
                    report.issues.append(
                        ScanIssue(
                            "duplicate_content",
                            identity_id,
                            f"{path.name} duplicates {owner}; excluded",
                        )
                    )
                else:  # the same picture under two identities: its label is ambiguous
                    report.issues.append(
                        ScanIssue(
                            "duplicate_content_cross_identity",
                            identity_id,
                            f"{path.name} duplicates {owner}; both excluded",
                        )
                    )
                    self._ambiguous.add(owner)
                report.n_images_excluded += 1
                continue
            content_owner[digest] = image_id
            records.append(
                ImageRecord(
                    image_id=image_id,
                    dataset_id=self.dataset_id,
                    identity_id=identity_id,
                    relpath=path.relative_to(self.root).as_posix(),
                    sha256=digest,
                    split=split,
                )
            )
            kept += 1
        return kept

    # -- persistence ----------------------------------------------------------------------
    def write_manifest(self, out_dir: Path) -> dict[str, object]:
        """Write ``manifest.jsonl``, ``scan_report.json`` and ``manifest_meta.json``.

        ``out_dir`` must be inside a git-ignored location (e.g. ``data/manifests/``).
        """
        records, report = self.scan()
        if not records:
            raise ManifestError("no valid DigiFace images found; nothing to write")
        manifest = out_dir / "manifest.jsonl"
        write_jsonl(manifest, records)
        (out_dir / "scan_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        meta: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "adapter_version": ADAPTER_VERSION,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "n_records": len(records),
            "bands": {k.value: list(v) for k, v in self.bands.items()},
            "hash_definition": "sha256 of file bytes",
        }
        (out_dir / "manifest_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        return meta


def _compact(values: list[int], limit: int = 10) -> str:
    shown = ",".join(map(str, values[:limit]))
    return shown + (f",... (+{len(values) - limit})" if len(values) > limit else "")
