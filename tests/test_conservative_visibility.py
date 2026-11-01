"""Phase 4 conservative visibility: hard negatives, lighting-as-corruption, training options."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from veilx.training.config import load_training_config
from veilx.vis.regions import face_cells
from veilx.vis.samples import (
    HARD_NEGATIVE,
    lighting_information_free,
    make_hard_negative,
    make_sample,
)

from .conftest import ROOT


def _face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (112, 112, 3)).astype(np.uint8)


def test_hard_negative_is_deterministic_and_targets_blob_cells() -> None:
    img, donor = _face(0), _face(1)
    a = make_hard_negative(img, donor, 42)
    b = make_hard_negative(img, donor, 42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2]["family"] == HARD_NEGATIVE
    changed = (a[0] != img).any(axis=2)
    assert changed.any()  # some pixels replaced by donor texture
    assert 0.0 < a[1].mean() < 1.0  # some cells not visible, some visible
    assert a[1][face_cells()].min() < 0.5


def test_hard_negative_texture_comes_from_donor_not_a_flat_fill() -> None:
    img, donor = _face(2), _face(3)
    out, _tgt, _ = make_hard_negative(img, donor, 7)
    changed = (out != img).any(axis=2)
    assert out[changed].std() > 10  # natural (here random) texture, not a uniform colour


def test_lighting_as_corruption_marks_only_information_free_pixels() -> None:
    src = np.full((4, 4, 3), 120, np.uint8)
    lit = src.copy()
    lit[0, 0] = 5  # crushed
    lit[1, 1] = 255  # newly clipped
    lit[2, 2] = 30  # strongly darkened but still informative
    m = lighting_information_free(src, lit)
    assert m[0, 0] and m[1, 1] and not m[2, 2]


def test_make_sample_lighting_flag_changes_only_f5() -> None:
    img = _face(4)
    for fam in ("F1_mask", "F5_extreme_lighting"):
        a = make_sample(img, fam, "high", 3)[1]
        b = make_sample(img, fam, "high", 3, lighting_as_corruption=True)[1]
        if fam == "F1_mask":
            assert np.array_equal(a, b)
        else:
            assert b.mean() >= a.mean()  # fewer cells counted as missing


@pytest.mark.parametrize("path", sorted((ROOT / "configs/experiments/phase4").glob("vec_*.yaml")))
def test_vec_configs_are_registered_values(path: Path) -> None:
    c = load_training_config(path)
    assert c.phase == "phase4" and c.visibility is not None
    assert c.visibility.occluded_weight == 4.0 and c.visibility.hard_negative_fraction == 0.2
    assert c.visibility.lighting_as_corruption and c.max_epochs == 30 and c.seed == 20261101
    assert "test" not in c.train_pairs and "cal" not in c.val_pairs
