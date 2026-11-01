"""Registered untouched partitions (Phase 4 values must never change; Phase 6 is disjoint)."""

from __future__ import annotations

import re

from veilx.data.untouched import P4, P6, PARTITIONS


def _id_ranges(p: object) -> set[int]:
    ids: set[int] = set()
    for name, _ in p.parts:  # type: ignore[attr-defined]
        m = re.fullmatch(r"subjects_(\d+)-(\d+)_72_imgs\.zip", name)
        assert m is not None
        ids |= set(range(int(m[1]), int(m[2]) + 1))
    return ids


def test_phase4_partition_is_frozen() -> None:
    assert (P4.directory, P4.split, P4.pair_prefix, P4.seed) == (
        "phase4",
        "p4_test",
        "p4te",
        20261101,
    )
    assert P4.pairs_name == "p4_test_pairs" and P4.clean_group == "p4_clean" and P4.earlier == ()
    assert dict(P4.parts) == {
        "subjects_2000-3999_72_imgs.zip": 2_936_098_225,
        "subjects_4000-5999_72_imgs.zip": 2_936_895_527,
    }


def test_phase6_partition_is_disjoint_and_dedups_against_phase4() -> None:
    assert _id_ranges(P4).isdisjoint(_id_ranges(P6))
    assert _id_ranges(P6).isdisjoint(range(0, 2000))  # Phase 1-3 72-image identities
    assert P6.seed != P4.seed and P6.directory != P4.directory
    assert "phase4/images.jsonl" in P6.earlier
    assert set(PARTITIONS) == {"p4", "p6"}
