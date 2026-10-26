"""LFW official-protocol parser on synthetic pair lists (no LFW data is used)."""

from __future__ import annotations

from pathlib import Path

import pytest

from veilx.data.lfw import (
    PROTOCOL,
    image_relpath,
    missing_images,
    parse_pairs_csv,
    parse_pairs_txt,
    require_images,
    to_pair_records,
)
from veilx.errors import ManifestError


def _txt(n_folds: int = 2, per: int = 2) -> str:
    lines = [f"{n_folds}\t{per}"]
    for f in range(n_folds):
        lines += [f"Person_{f}{k}\t1\t2" for k in range(per)]
        lines += [f"Person_{f}{k}\t1\tOther_{f}{k}\t3" for k in range(per)]
    return "\n".join(lines) + "\n"


def test_txt_parse_preserves_folds_and_order() -> None:
    pairs = parse_pairs_txt(_txt())
    assert len(pairs) == 8
    assert [p.fold for p in pairs] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert [p.genuine for p in pairs[:4]] == [True, True, False, False]
    assert pairs[2].name_b == "Other_00" and pairs[2].idx_b == 3
    recs = to_pair_records(pairs)
    assert recs[0].pair_id == "lfw-f01-000" and recs[0].image_a == "lfw:Person_00:0001"
    assert recs[2].label == "impostor" and all(r.protocol == PROTOCOL for r in recs)


def test_csv_matches_txt() -> None:
    body = _txt().splitlines()[1:]
    csv_text = "name,imagenum1,imagenum2,\n" + "\n".join(",".join(r.split("\t")) for r in body)
    assert parse_pairs_csv(csv_text, n_folds=2, per_class=2) == parse_pairs_txt(_txt())


@pytest.mark.parametrize(
    "text",
    [
        "2 2\nA\t1\t2\n",  # wrong row count
        "x\n",  # bad header
        "1 1\nA\t1\nB\t1\tC\t2\n",  # malformed matched row
        "1 1\nA\t1\t2\nB\t1\tB\t2\n",  # mismatched row with the same name
    ],
)
def test_malformed_lists_raise(text: str) -> None:
    with pytest.raises(ManifestError):
        parse_pairs_txt(text)


def test_missing_images_are_reported_not_skipped(tmp_path: Path) -> None:
    pairs = parse_pairs_txt("1 1\nA\t1\t2\nA\t1\tB\t1\n")
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "A_0001.jpg").write_bytes(b"x")
    (tmp_path / "A" / "A_0002.jpg").write_bytes(b"x")
    assert missing_images(tmp_path, pairs) == ["B/B_0001.jpg"]
    with pytest.raises(ManifestError, match="missing"):
        require_images(tmp_path, pairs)
    assert image_relpath("A", 12) == "A/A_0012.jpg"
    with pytest.raises(ManifestError):
        image_relpath("A", 0)
