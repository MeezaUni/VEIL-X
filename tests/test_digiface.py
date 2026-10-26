"""DigiFace adapter on synthetic fixture trees (random-noise PNGs; no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veilx.config import load_dataset_registry, load_protocol
from veilx.config.schema import DatasetRegistry
from veilx.data.digiface import DigiFaceAdapter, expected_images_per_identity
from veilx.data.manifest import ImageRecord, read_jsonl, validate_image_manifest
from veilx.data.splits import assign_split
from veilx.errors import DataAccessError, ManifestError

from .conftest import DATASETS, PROTOCOL
from .helpers import noise_png, write_identity

BANDS = load_protocol(PROTOCOL).splits.bands
REGISTRY = load_dataset_registry(DATASETS)


def adapter(root: Path, registry: DatasetRegistry = REGISTRY) -> DigiFaceAdapter:
    return DigiFaceAdapter(root, BANDS, registry)


def test_expected_counts_follow_official_ranges() -> None:
    assert expected_images_per_identity(0) == 72
    assert expected_images_per_identity(9_999) == 72
    assert expected_images_per_identity(100_000) == 5
    assert expected_images_per_identity(199_998) == 5
    assert expected_images_per_identity(10_000) is None
    assert expected_images_per_identity(199_999) is None


def test_flat_layout_clean(tmp_path: Path) -> None:
    for ident in ("100000", "100001", "100002"):
        write_identity(tmp_path, ident, 5)
    records, report = adapter(tmp_path).scan()
    assert report.layout == "flat"
    assert report.n_identities_in_manifest == 3
    assert report.n_images_in_manifest == 15
    assert report.issues == []
    validate_image_manifest(records)
    for r in records:
        assert r.split is assign_split("digiface1m", r.identity_id, BANDS)
        assert not Path(r.relpath).is_absolute() and "\\" not in r.relpath
    assert [r.image_id for r in records][:2] == ["digiface1m:100000:0", "digiface1m:100000:1"]


def test_parts_layout_and_determinism(tmp_path: Path) -> None:
    part_a = tmp_path / "subjects_100000-133332_5_imgs"
    part_b = tmp_path / "subjects_0-1999_72_imgs"
    write_identity(part_a, "100003", 5)
    write_identity(part_a, "100001", 5)
    write_identity(part_b, "7", 72)
    r1, rep1 = adapter(tmp_path).scan()
    r2, _ = adapter(tmp_path).scan()
    assert r1 == r2
    assert rep1.layout == "parts" and len(rep1.parts) == 2
    assert [r.identity_id for r in r1][::5][:3] == ["7", "7", "7"]  # numeric order: 7 first
    assert rep1.images_per_identity_histogram == {5: 2, 72: 1}


def test_anomalies_reported(tmp_path: Path) -> None:
    write_identity(tmp_path, "100000", 5)
    write_identity(tmp_path, "100001", 3)  # missing indices 3, 4
    write_identity(tmp_path, "100002", 1)  # single image + missing
    (tmp_path / "100003").mkdir()  # empty identity
    write_identity(tmp_path, "100004", 6)  # extra index 5
    write_identity(tmp_path, "100005", 5)
    (tmp_path / "100005" / "2.png").write_bytes(b"corrupt")  # corrupt
    (tmp_path / "100005" / "notes.txt").write_text("x")  # unexpected file
    write_identity(tmp_path, "100006", 5)
    dup = (tmp_path / "100000" / "0.png").read_bytes()
    (tmp_path / "100006" / "4.png").write_bytes(dup)  # duplicate content across identities
    (tmp_path / "abc").mkdir()  # non-numeric identity dir
    write_identity(tmp_path, "50000", 2)  # outside official ranges
    records, report = adapter(tmp_path).scan()
    counts = report.issue_counts()
    assert counts["missing_index"] == 3  # 100001, 100002, 100003 (empty)
    assert counts["unexpected_index"] == 1
    assert counts["single_image_identity"] == 1
    assert counts["empty_identity"] == 1
    assert counts["corrupt_image"] == 1
    assert counts["unexpected_file"] == 1
    assert counts["duplicate_content"] == 1
    assert counts["non_numeric_identity_dir"] == 1
    assert counts["identity_outside_official_ranges"] == 1
    ids = {r.image_id for r in records}
    assert "digiface1m:100005:2" not in ids  # corrupt excluded
    assert "digiface1m:100006:4" not in ids  # duplicate excluded
    assert report.n_images_excluded == 2
    validate_image_manifest(records)


def test_identity_in_two_parts(tmp_path: Path) -> None:
    write_identity(tmp_path / "subjects_100000-133332_5_imgs", "100001", 5)
    write_identity(tmp_path / "subjects_133333-166665_5_imgs", "100001", 5, seed=9)
    _, report = adapter(tmp_path).scan()
    assert report.count("identity_in_multiple_parts") == 1
    assert report.count("identity_outside_part_range") == 0  # first copy is in range


def test_access_checks(tmp_path: Path, registry_dict: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(DataAccessError):
        adapter(tmp_path / "missing").scan()
    blocked = {**registry_dict, "datasets": [dict(e) for e in registry_dict["datasets"]]}
    entry = next(e for e in blocked["datasets"] if e["id"] == "digiface1m")
    entry["access_status"] = "unresolved"
    with pytest.raises(DataAccessError, match="unresolved"):
        adapter(tmp_path, DatasetRegistry.model_validate(blocked)).scan()


def test_write_manifest(tmp_path: Path) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    write_identity(root, "100000", 5)
    meta = adapter(root).write_manifest(out)
    assert meta["n_records"] == 5
    back = list(read_jsonl(out / "manifest.jsonl", ImageRecord))
    assert len(back) == 5
    report = json.loads((out / "scan_report.json").read_text(encoding="utf-8"))
    assert report["n_images_in_manifest"] == 5
    assert str(root) not in (out / "scan_report.json").read_text(encoding="utf-8")


def test_nothing_valid_raises(tmp_path: Path) -> None:
    (tmp_path / "100000").mkdir()
    (tmp_path / "100000" / "0.png").write_bytes(b"bad")
    with pytest.raises(ManifestError):
        adapter(tmp_path).write_manifest(tmp_path / "out")


def test_tampered_manifest_rejected(tmp_path: Path) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    write_identity(root, "100000", 5)
    adapter(root).write_manifest(out)
    lines = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"sha256":"', '"sha256":"zz')
    (out / "manifest.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ManifestError):
        list(read_jsonl(out / "manifest.jsonl", ImageRecord))


def test_noise_png_is_valid() -> None:
    from veilx.data.png import check_png_bytes

    assert check_png_bytes(noise_png(3)).ok
