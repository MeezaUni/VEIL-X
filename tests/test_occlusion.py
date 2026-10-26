"""Occlusion engine and LOFO condition planning (synthetic images only)."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.config.schema import SplitName
from veilx.data.images import pixel_sha256
from veilx.errors import ProtocolViolationError
from veilx.occlusion import (
    LEVELS,
    Landmarks,
    apply_occlusion,
    apply_stress,
    face_region,
    visibility_grid,
)
from veilx.occlusion.plan import STRATA, assert_plan_isolation, plan_conditions, plan_stress
from veilx.pairs import PairSpec, build_pairs

from .helpers import toy_images

FAMILIES = ("F1_mask", "F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")
PHYSICAL = FAMILIES[:4]


def face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (112, 112, 3), dtype=np.uint8)


@pytest.mark.parametrize("family", FAMILIES)
def test_deterministic_and_recorded(family: str) -> None:
    img = face()
    a = apply_occlusion(img, family, "medium", 123)
    b = apply_occlusion(img, family, "medium", 123)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.mask, b.mask) and a.spec == b.spec
    s = a.spec
    assert s.family == family and s.seed == 123 and s.severity_level == "medium"
    assert s.severity is not None and s.visibility_ratio == pytest.approx(1 - s.severity, abs=1e-6)
    assert s.physical is (family != "F5_extreme_lighting")
    assert s.recipe_sha256 and len(s.recipe_sha256) == 64 and s.geometry
    assert apply_occlusion(img, family, "medium", 124).spec.recipe_sha256 != s.recipe_sha256
    assert a.image.dtype == np.uint8 and a.image.shape == img.shape


@pytest.mark.parametrize("family", PHYSICAL)
def test_severity_increases_with_level(family: str) -> None:
    img = face()
    means = [
        np.mean([apply_occlusion(img, family, lvl, s).spec.severity for s in range(40)])
        for lvl in LEVELS
    ]
    assert means[0] < means[1] < means[2]


@pytest.mark.parametrize("family", PHYSICAL)
def test_physical_masks_are_pixel_independent(family: str) -> None:
    """Justifies computing physical-family masks from a stand-in image in the pipeline."""
    a = apply_occlusion(face(1), family, "high", 7)
    b = apply_occlusion(np.zeros((112, 112, 3), np.uint8), family, "high", 7)
    assert np.array_equal(a.mask, b.mask) and a.spec.severity == b.spec.severity


def test_occluded_pixels_change_and_others_do_not() -> None:
    img = face(2)
    for family in PHYSICAL:
        r = apply_occlusion(img, family, "high", 5)
        assert r.mask.any()
        assert np.array_equal(r.image[~r.mask], img[~r.mask])


def test_lighting_is_non_physical_and_masks_destroyed_pixels() -> None:
    img = face(3)
    r = apply_occlusion(img, "F5_extreme_lighting", "high", 11)
    assert r.spec.physical is False
    assert 0.0 <= r.spec.visibility_ratio <= 1.0  # type: ignore[operator]


def test_recipe_binds_source_pixels() -> None:
    img = face(4)
    a = apply_occlusion(img, "F1_mask", "low", 1)
    b = apply_occlusion(img, "F1_mask", "low", 1, source_pixel_sha256=pixel_sha256(img))
    assert a.spec.recipe_sha256 == b.spec.recipe_sha256
    c = apply_occlusion(face(5), "F1_mask", "low", 1)
    assert c.spec.recipe_sha256 != a.spec.recipe_sha256


def test_template_landmarks_and_face_region() -> None:
    lm = Landmarks.template(112, 112)
    assert lm.source == "arcface_template" and 30 < lm.eye_distance < 40
    region = face_region(lm, 112, 112)
    assert 0.3 < region.mean() < 0.8 and region[60, 56]
    grid = visibility_grid(np.zeros((112, 112), bool))
    assert grid.shape == (7, 7) and (grid == 1.0).all()


def test_stress_composition_rules() -> None:
    img = face(6)
    s = apply_stress(img, [("F1_mask", "medium", 1), ("F2_sunglasses", "medium", 2)])
    assert len(s.steps) == 2 and s.combined_visibility_ratio <= min(
        st.visibility_ratio for st in s.steps
    )  # type: ignore[type-var]
    with pytest.raises(ValueError):
        apply_stress(img, [("F1_mask", "low", 1)])
    with pytest.raises(ValueError):
        apply_stress(img, [("F1_mask", "low", 1), ("F5_extreme_lighting", "low", 2)])
    with pytest.raises(ValueError):
        apply_occlusion(img, "F9_glitter", "low", 1)


# --------------------------------------------------------------------------- planning


def pairs(n_ids: int = 60) -> list:  # type: ignore[type-arg]
    ps = build_pairs(
        toy_images(n_ids, 3, SplitName.TEST),
        PairSpec("clustered", 200, 0.5, SplitName.TEST, 0, genuine_cap_per_identity=3),
    )
    return list(ps.pairs)


@pytest.mark.parametrize("heldout", FAMILIES)
def test_plans_respect_lofo_isolation(heldout: str) -> None:
    pr = pairs()
    for role in ("train", "val", "cal", "test_id"):
        plan = plan_conditions(pr, FAMILIES, heldout, role, global_seed=1)  # type: ignore[arg-type]
        assert all(heldout not in c.families() for c in plan)
        assert_plan_isolation(plan)
    ood = plan_conditions(pr, FAMILIES, heldout, "test_ood", global_seed=1)
    assert all(heldout in c.families() for c in ood if c.families())
    assert_plan_isolation(ood)


def test_plan_strata_quotas_and_determinism() -> None:
    pr = pairs()
    a = plan_conditions(pr, FAMILIES, "F1_mask", "train", global_seed=7)
    b = plan_conditions(list(reversed(pr)), FAMILIES, "F1_mask", "train", global_seed=7)
    assert [x.stratum for x in a] != [] and {c.pair_id: c for c in a} == {c.pair_id: c for c in b}
    counts = {s: sum(c.stratum == s for c in a) for s in STRATA}
    assert max(counts.values()) - min(counts.values()) <= 1
    for c in a:
        n_occ = len([x for x in (c.side_a.family, c.side_b.family) if x])
        assert n_occ == {"clean_clean": 0, "clean_occluded": 1}.get(c.stratum, 2)


def test_isolation_violation_is_detected() -> None:
    from dataclasses import replace

    plan = plan_conditions(pairs(), FAMILIES, "F2_sunglasses", "cal", global_seed=3)
    bad = next(c for c in plan if c.side_a.family)
    tampered = replace(bad, side_a=replace(bad.side_a, family="F2_sunglasses"))
    with pytest.raises(AssertionError):
        assert_plan_isolation([tampered])
    with pytest.raises(ValueError):
        plan_conditions(pairs(), FAMILIES, "F9", "train", global_seed=1)  # type: ignore[arg-type]
    from veilx.guards import require_heldout_isolation

    with pytest.raises(ProtocolViolationError):
        require_heldout_isolation("F2_sunglasses", ("F2_sunglasses",))


def test_stress_plan_uses_physical_combinations_only() -> None:
    st = plan_stress(pairs(), global_seed=2)
    for s in st:
        for f, _, _ in (*s.steps_a, *s.steps_b):
            assert f != "F5_extreme_lighting"
        assert len(s.steps_a) == 2 and len(s.steps_b) == 2
