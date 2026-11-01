"""Evaluation-only occluder E1_hair_strands (Phase 4): deterministic, outside the LOFO folds."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.config import APPROVED_FAMILY_IDS, EVAL_ONLY_FAMILY_IDS, load_protocol
from veilx.data.manifest import OcclusionSpec
from veilx.occlusion.engine import apply_occlusion

from .conftest import PROTOCOL


def test_e1_is_deterministic_and_scales_with_level() -> None:
    img = np.full((112, 112, 3), 140, np.uint8)
    sev = []
    for lvl in ("low", "medium", "high"):
        a = apply_occlusion(img, "E1_hair_strands", lvl, 11)
        b = apply_occlusion(img, "E1_hair_strands", lvl, 11)
        assert np.array_equal(a.image, b.image) and a.spec.recipe_sha256 == b.spec.recipe_sha256
        assert a.spec.physical is True
        sev.append(a.spec.severity)
    assert sev[0] < sev[1] < sev[2]


def test_e1_is_not_a_lofo_family() -> None:
    proto = load_protocol(PROTOCOL)
    assert "E1_hair_strands" in EVAL_ONLY_FAMILY_IDS
    assert "E1_hair_strands" not in APPROVED_FAMILY_IDS
    assert "E1_hair_strands" not in proto.family_ids


def test_spec_accepts_eval_only_but_rejects_unknown() -> None:
    OcclusionSpec(family="E1_hair_strands", seed=1, generator_version="x")
    with pytest.raises(ValueError, match="unknown"):
        OcclusionSpec(family="F9_unknown", seed=1, generator_version="x")
