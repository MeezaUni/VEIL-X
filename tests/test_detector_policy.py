"""Phase 4F: detector-failure taxonomy and policies; alignment residual."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.detect.align import TEMPLATE, alignment_residual
from veilx.detect.policy import apply_policy, categorize, needs_policy
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH


def test_residual_zero_for_template_and_large_for_scrambled_points() -> None:
    assert alignment_residual(TEMPLATE * 2.0 + 7.0) == pytest.approx(0.0, abs=1e-9)
    scrambled = TEMPLATE[[4, 0, 3, 1, 2]]
    assert alignment_residual(scrambled) > 5.0


def test_categories_are_exhaustive_and_ordered() -> None:
    kw = {"min_score": 0.6, "max_residual": 3.0}
    assert categorize(0, None, None, **kw) == "no_face"
    assert categorize(1, 0.5, 1.0, **kw) == "low_quality"
    assert categorize(1, 0.9, 4.0, **kw) == "low_quality"
    assert categorize(3, 0.9, 1.0, **kw) == "multiple_faces"
    assert categorize(1, 0.9, 1.0, **kw) == "success"
    assert needs_policy("no_face") and needs_policy("low_quality")
    assert not needs_policy("multiple_faces") and not needs_policy("success")


def test_policies_never_drop_failed_pairs() -> None:
    cats = np.array(["success", "no_face", "low_quality", "multiple_faces"])
    dec = np.array([MATCH, MATCH, MATCH, NON_MATCH])
    fb = np.array([NON_MATCH, MATCH, ABSTAIN, MATCH])
    a = apply_policy("abstain", cats, dec)
    n = apply_policy("non_match", cats, dec)
    f = apply_policy("fallback", cats, dec, fb)
    for out in (a, n, f):
        assert len(out) == len(cats)  # nothing removed
        assert out[0] == MATCH and out[3] == NON_MATCH  # unaffected categories untouched
    assert a[1:3].tolist() == [ABSTAIN, ABSTAIN]
    assert n[1:3].tolist() == [NON_MATCH, NON_MATCH]
    assert f[1:3].tolist() == [MATCH, ABSTAIN]
    with pytest.raises(ValueError):
        apply_policy("fallback", cats, dec)
