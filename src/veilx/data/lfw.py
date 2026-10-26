"""LFW official verification protocol (View 2 ``pairs.txt``), parsed unmodified.

Status: **blocked** in Phase 2 (no authorised distribution reachable; docs/real_world_benchmark.md).
The parser is ready and tested on synthetic text only.

``pairs.txt``: a header ``<n_folds> <n>`` followed, per fold, by ``n`` matched lines
``name idx1 idx2`` and then ``n`` mismatched lines ``name1 idx1 name2 idx2`` (whitespace
separated). The Kaggle ``pairs.csv`` holds the same 6,000 rows in the same order, without the
header. Fold membership and order are preserved exactly; nothing is filtered or re-balanced.
Images are ``<root>/<name>/<name>_<idx:04d>.jpg``; a missing image is an error.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from veilx.config.schema import SplitName
from veilx.data.manifest import PairRecord
from veilx.errors import ManifestError

PROTOCOL = "lfw_view2_official"
OFFICIAL_FOLDS = 10
OFFICIAL_PER_CLASS = 300


@dataclass(frozen=True)
class LFWPair:
    fold: int  # 1-based, as in the official protocol
    position: int  # 0-based position inside the fold
    genuine: bool
    name_a: str
    idx_a: int
    name_b: str
    idx_b: int


def image_relpath(name: str, idx: int) -> str:
    if idx < 1:
        raise ManifestError(f"invalid LFW image index {idx} for {name}")
    return f"{name}/{name}_{idx:04d}.jpg"


def _rows_to_pairs(rows: Sequence[Sequence[str]], n_folds: int, per_class: int) -> list[LFWPair]:
    if len(rows) != n_folds * 2 * per_class:
        raise ManifestError(f"expected {n_folds * 2 * per_class} pair rows, found {len(rows)}")
    out: list[LFWPair] = []
    for k, row in enumerate(rows):
        fold, pos = divmod(k, 2 * per_class)
        genuine = pos < per_class
        cells = [c.strip() for c in row if c.strip() != ""]
        try:
            if genuine:
                if len(cells) != 3:
                    raise ValueError
                name, i1, i2 = cells
                out.append(LFWPair(fold + 1, pos, True, name, int(i1), name, int(i2)))
            else:
                if len(cells) != 4:
                    raise ValueError
                n1, i1, n2, i2 = cells
                if n1 == n2:
                    raise ValueError
                out.append(LFWPair(fold + 1, pos, False, n1, int(i1), n2, int(i2)))
        except ValueError as exc:
            kind = "matched" if genuine else "mismatched"
            raise ManifestError(f"row {k}: malformed {kind} pair {row!r}") from exc
    return out


def parse_pairs_txt(text: str) -> list[LFWPair]:
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    if not lines or len(lines[0]) != 2:
        raise ManifestError("pairs.txt header must be '<n_folds> <pairs_per_class>'")
    n_folds, per_class = (int(x) for x in lines[0])
    return _rows_to_pairs(lines[1:], n_folds, per_class)


def parse_pairs_csv(
    text: str, n_folds: int = OFFICIAL_FOLDS, per_class: int = OFFICIAL_PER_CLASS
) -> list[LFWPair]:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if rows and not rows[0][1].strip().isdigit():  # header such as name,imagenum1,imagenum2
        rows = rows[1:]
    return _rows_to_pairs(rows, n_folds, per_class)


def image_id(name: str, idx: int) -> str:
    return f"lfw:{name}:{idx:04d}"


def to_pair_records(pairs: Sequence[LFWPair]) -> list[PairRecord]:
    """One record per official pair; ``pair_id`` encodes fold and position."""
    return [
        PairRecord(
            pair_id=f"lfw-f{p.fold:02d}-{p.position:03d}",
            image_a=image_id(p.name_a, p.idx_a),
            image_b=image_id(p.name_b, p.idx_b),
            label="genuine" if p.genuine else "impostor",
            stratum="clean_clean",
            split=SplitName.TEST,
            protocol=PROTOCOL,
        )
        for p in pairs
    ]


def missing_images(root: Path, pairs: Sequence[LFWPair]) -> list[str]:
    need = {(p.name_a, p.idx_a) for p in pairs} | {(p.name_b, p.idx_b) for p in pairs}
    return sorted(
        image_relpath(n, i) for n, i in need if not (root / image_relpath(n, i)).is_file()
    )


def require_images(root: Path, pairs: Sequence[LFWPair]) -> None:
    miss = missing_images(root, pairs)
    if miss:
        raise ManifestError(
            f"{len(miss)} LFW images referenced by the protocol are missing: {miss[:5]}"
        )
