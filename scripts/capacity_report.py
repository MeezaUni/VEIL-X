"""Statistical capacity of the calibration design for the current protocol (Phase 2, v0.4.0).

MATHEMATICAL CALCULATION from real pair counts (no model scores): exact binomial /
Clopper-Pearson bounds for the identity-disjoint calibration set, globally and per stratum
under Bonferroni-split delta. Writes ``reports/data/capacity_v<protocol>.json``.

    python scripts/capacity_report.py       (data root from VEILX_DATA_ROOT)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from veilx.config import load_protocol
from veilx.data.paths import resolve_data_root
from veilx.stats.feasibility import clopper_pearson_upper, zero_error_n

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")


def max_errors_certifiable(n: int, target: float, confidence: float) -> int | None:
    """Largest k with Clopper-Pearson upper(k, n) <= target (None if even k = 0 fails)."""
    if n == 0 or clopper_pearson_upper(0, n, confidence) > target:
        return None
    k = 0
    while clopper_pearson_upper(k + 1, n, confidence) <= target:
        k += 1
    return k


def block(n: int, confidence: float, targets: tuple[float, ...]) -> dict[str, Any]:
    return {
        "n": n,
        "best_certifiable_rate_zero_errors": clopper_pearson_upper(0, n, confidence) if n else None,
        "max_errors_still_certifiable": {
            f"{t:g}": max_errors_certifiable(n, t, confidence) for t in targets
        },
    }


def main() -> int:
    layout = resolve_data_root()
    stats = json.loads((layout.private_reports / "pair_stats.json").read_text("utf-8"))
    cal = stats["cal_independent"]
    conf = float(PROTO.constants.confidence_level.value)  # type: ignore[arg-type]
    alpha = float(PROTO.constants.accepted_decision_error_target.value)  # type: ignore[arg-type]
    fars = tuple(float(x) for x in PROTO.constants.far_operating_points.value)  # type: ignore[union-attr]
    targets = (alpha, *fars)
    n_g, n_i = int(cal["n_genuine"]), int(cal["n_impostor"])
    # Pairs are ACCEPTED only partly under selective rules; capacity below is for full coverage
    # (an upper bound). Per-stratum: oracle-visibility strata clean / vis_high / vis_low at
    # delta/3; the occluded calibration plan has 1/4 clean-clean pairs and splits the rest at
    # the validation median co-visibility, i.e. roughly 1/4, 3/8, 3/8 of each class.
    conf3 = 1 - (1 - conf) / 3
    strat = {
        name: {
            "genuine": block(math.floor(n_g * f), conf3, targets),
            "impostor": block(math.floor(n_i * f), conf3, targets),
        }
        for name, f in (("clean", 0.25), ("vis_high", 0.375), ("vis_low", 0.375))
    }
    report = {
        "evidence_type": "mathematical_calculation",
        "protocol_version": PROTO.protocol_version,
        "confidence": conf,
        "targets": list(targets),
        "calibration_identities": int(cal["n_identities"]),
        "identity_disjoint": bool(cal["identity_disjoint"]),
        "global": {
            "genuine": block(n_g, conf, targets),
            "impostor": block(n_i, conf, targets),
        },
        "zero_error_n_needed": {f"{t:g}": zero_error_n(t, conf) for t in targets},
        "zero_error_n_needed_per_stratum_delta_over_3": {
            f"{t:g}": zero_error_n(t, conf3) for t in targets
        },
        "stratified_approx": strat,
        "far_0.1pct_certifiable": clopper_pearson_upper(0, n_i, conf) <= fars[1],
        "note": (
            "Upper bounds at full coverage. A selective rule accepts only part of each class, "
            "so certified rates are worse. FAR 0.1 % remains estimation-only by protocol "
            "(D-034) even when zero-error capacity is reached: one false match would break it."
        ),
    }
    out = ROOT / "reports" / "data" / f"capacity_v{PROTO.protocol_version}.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
