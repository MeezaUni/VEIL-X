"""Phase 4 registered endpoint logic: safety (P1), benefit (P2), recommendation (A3 default)."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from .conftest import ROOT


@pytest.fixture(scope="module")
def ev4() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase4", ROOT / "scripts" / "evaluate_phase4.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells(
    ev4: Any, *, h4_gain: float, h4_p: float, c4_violates: bool, worse_c4: bool
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    arms = (
        "A3_quality",
        "A4_original",
        "A5_oracle",
        "B0_raw_global",
        "B1_quality_global",
        *ev4.CANDIDATES,
    )
    for f in ev4.FAMILIES:
        for role in ("test_ood", "e1"):
            a = {
                f"{x}|primary|v2": {"verdict_fnm": "within_target", "verdict_fm": "within_target"}
                for x in arms
            }
            if c4_violates and f == "F4_random_patch" and role == "test_ood":
                a["C4_conservative|primary|v2"]["verdict_fnm"] = "violated_ci_excludes_target"
            pv = {}
            for x in (*ev4.CANDIDATES, "A4_original", "A5_oracle", "B1_quality_global"):
                worse = (
                    worse_c4
                    and x == "C4_conservative"
                    and f == "F4_random_patch"
                    and role == "test_ood"
                )
                pv[x] = {
                    "sFNMR": {
                        "estimate": 0.05 if worse else -0.001,
                        "p_value": 1e-6 if worse else 0.5,
                    },
                    "sFMR": {"estimate": 0.0, "p_value": 1.0},
                    "coverage": {
                        "estimate": h4_gain if x == "H4_quality_first_hybrid" else 0.0,
                        "p_value": h4_p if x == "H4_quality_first_hybrid" else 1.0,
                    },
                }
            cells[f"{f}|{role}"] = {"arms": a, "paired_vs_a3": pv}
    return cells


def test_default_recommendation_is_a3_without_significant_gain(ev4: Any) -> None:
    r = ev4.endpoints(_cells(ev4, h4_gain=0.01, h4_p=0.5, c4_violates=False, worse_c4=False))
    assert r["recommendation"] == "A3_quality"
    assert all(r[x]["safe"] for x in ev4.CANDIDATES)
    assert not any(r[x]["better_than_A3"] for x in ev4.CANDIDATES)


def test_unsafe_candidate_cannot_be_recommended_even_with_gain(ev4: Any) -> None:
    r = ev4.endpoints(_cells(ev4, h4_gain=0.03, h4_p=1e-6, c4_violates=True, worse_c4=True))
    assert not r["C4_conservative"]["safe"]
    assert r["C4_conservative"]["significantly_worse_cells"]["sFNMR"] == [
        "F4_random_patch|test_ood"
    ]
    assert r["H4_quality_first_hybrid"]["better_than_A3"]
    assert r["recommendation"] == "H4_quality_first_hybrid"
