from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from veilx.config.schema import SplitName
from veilx.data.manifest import (
    ImageRecord,
    OcclusionSpec,
    PairRecord,
    assert_family_isolation,
    expected_stratum,
    read_jsonl,
    validate_image_manifest,
    validate_pair_manifest,
    write_jsonl,
)
from veilx.errors import LeakageError, ManifestError

pytestmark = pytest.mark.structural


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def img(
    image_id: str,
    identity: str,
    split: SplitName = SplitName.TRAIN,
    *,
    source: str | None = None,
    family: str | None = None,
) -> ImageRecord:
    occ = (
        OcclusionSpec(family=family, seed=1, generator_version="0.0.1")
        if family is not None
        else None
    )
    return ImageRecord(
        image_id=image_id,
        dataset_id="toy",
        identity_id=identity,
        relpath=f"{identity}/{image_id}.png",
        sha256=_sha(image_id),
        split=split,
        source_image_id=source,
        occlusion=occ,
    )


@pytest.mark.parametrize(
    "relpath", ["/abs/x.png", "C:/x.png", "a\\b.png", "../x.png", "a/../../x.png"]
)
def test_relpath_must_be_relative_posix(relpath: str) -> None:
    with pytest.raises(ValueError):
        ImageRecord(
            image_id="i1",
            dataset_id="toy",
            identity_id="p1",
            relpath=relpath,
            sha256=_sha("x"),
            split=SplitName.TRAIN,
        )


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValueError):
        OcclusionSpec(family="F9_glitter", seed=0, generator_version="0")


def test_identity_split_leakage_detected() -> None:
    recs = [img("a", "p1", SplitName.TRAIN), img("b", "p1", SplitName.TEST)]
    with pytest.raises(LeakageError):
        validate_image_manifest(recs)


def test_variant_must_inherit_split_and_identity() -> None:
    src = img("a", "p1", SplitName.CAL)
    with pytest.raises(LeakageError):
        validate_image_manifest(
            [src, img("a_v", "p1", SplitName.TEST, source="a", family="F1_mask")]
        )
    with pytest.raises((ManifestError, LeakageError)):
        validate_image_manifest(
            [src, img("a_v", "p2", SplitName.CAL, source="a", family="F1_mask")]
        )
    validate_image_manifest([src, img("a_v", "p1", SplitName.CAL, source="a", family="F1_mask")])


def test_synthetic_occlusion_requires_source() -> None:
    with pytest.raises(ManifestError):
        validate_image_manifest([img("a", "p1", family="F2_sunglasses")])


def test_duplicate_image_ids_rejected() -> None:
    with pytest.raises(ManifestError):
        validate_image_manifest([img("a", "p1"), img("a", "p1")])


def test_pair_rules() -> None:
    images = [
        img("a", "p1"),
        img("b", "p1"),
        img("c", "p2"),
        img("a_m", "p1", source="a", family="F1_mask"),
        img("c_s", "p2", source="c", family="F2_sunglasses"),
        img("t", "p3", SplitName.TEST),
    ]
    validate_image_manifest(images)
    good = [
        PairRecord(
            pair_id="g1",
            image_a="a",
            image_b="b",
            label="genuine",
            stratum="clean_clean",
            split=SplitName.TRAIN,
        ),
        PairRecord(
            pair_id="i1",
            image_a="a_m",
            image_b="c_s",
            label="impostor",
            stratum="occluded_occluded_cross_family",
            split=SplitName.TRAIN,
        ),
    ]
    validate_pair_manifest(good, images)

    def bad(**kw: object) -> list[PairRecord]:
        base = {
            "pair_id": "x",
            "image_a": "a",
            "image_b": "c",
            "label": "impostor",
            "stratum": "clean_clean",
            "split": SplitName.TRAIN,
        }
        base.update(kw)
        return [PairRecord.model_validate(base)]

    with pytest.raises(ManifestError):
        validate_pair_manifest(bad(label="genuine"), images)  # label contradicts identities
    with pytest.raises(ManifestError):
        validate_pair_manifest(bad(stratum="clean_occluded"), images)  # wrong stratum
    with pytest.raises(LeakageError):
        validate_pair_manifest(bad(image_b="t"), images)  # crosses splits
    with pytest.raises(ManifestError):
        validate_pair_manifest(bad(image_b="a"), images)  # self pair
    with pytest.raises(ManifestError):
        validate_pair_manifest(
            [*good, good[0].model_copy(update={"pair_id": "g2", "image_a": "b", "image_b": "a"})],
            images,
        )  # duplicate


def test_expected_stratum() -> None:
    a, b = img("a", "p1"), img("b", "p2")
    m = img("m", "p1", source="a", family="F1_mask")
    m2 = img("m2", "p2", source="b", family="F1_mask")
    s = img("s", "p2", source="b", family="F2_sunglasses")
    assert expected_stratum(a, b) == "clean_clean"
    assert expected_stratum(a, m) == "clean_occluded"
    assert expected_stratum(m, m2) == "occluded_occluded_same_family"
    assert expected_stratum(m, s) == "occluded_occluded_cross_family"


def test_family_isolation_guard() -> None:
    recs = [
        img("a", "p1", SplitName.CAL),
        img("a_m", "p1", SplitName.CAL, source="a", family="F1_mask"),
    ]
    with pytest.raises(LeakageError):
        assert_family_isolation(
            recs, "F1_mask", splits=[SplitName.TRAIN, SplitName.VAL, SplitName.CAL]
        )
    assert_family_isolation(recs, "F2_sunglasses", splits=[SplitName.CAL])


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    recs = [img("a", "p1"), img("a_m", "p1", source="a", family="F3_hand_object")]
    p = tmp_path / "m.jsonl"
    assert write_jsonl(p, recs) == 2
    assert list(read_jsonl(p, ImageRecord)) == recs
    p.write_text('{"image_id": "bad"}\n', encoding="utf-8")
    with pytest.raises(ManifestError):
        list(read_jsonl(p, ImageRecord))
