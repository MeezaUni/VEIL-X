"""Benefit of additional DigiFace parts (exact binomial calculations; no data needed).

Compares the approved subset with options that add parts, for: independent calibration
capacity (identity-disjoint matching), best certifiable selective FMR with 0/1/2 false
matches, and test-identity counts relevant to the clean-margin power study (SIM-006).
The calibration share uses the protocol bands (10 % of identities).

    python scripts/parts_benefit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from veilx.config import load_protocol
from veilx.stats.feasibility import IndependentPairCapacity, clopper_pearson_upper

ROOT = Path(__file__).resolve().parents[1]

OPTIONS: list[tuple[str, int, int, float]] = [  # name, 72-img identities, 5-img identities, GiB
    ("approved (72-P1 + 5-P1)", 2_000, 33_333, 6.12),
    ("+ 5-img P2", 2_000, 66_666, 9.28),
    ("+ 5-img P2, P3", 2_000, 99_999, 12.45),
    ("+ 72-img P2-P5", 10_000, 33_333, 19.09),
    ("all 8 parts", 10_000, 99_999, 26.47),
]


def main() -> int:
    proto = load_protocol(ROOT / "configs" / "research_protocol.yaml")
    bands = proto.splits.bands
    share = {k.value: (hi - lo) / 100 for k, (lo, hi) in bands.items()}
    conf = float(proto.constants.confidence_level.value)  # type: ignore[arg-type]
    print(
        "| Option | Archives (GiB) | Cal IDs | Indep. impostor pairs | Certifiable sFMR "
        "(0 / 1 / 2 FM) | Test IDs | Train IDs with 72 imgs |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, n72, n5, gib in OPTIONS:
        n = n72 + n5
        cal = int(n * share["cal"])
        cap = IndependentPairCapacity.from_identities(cal)
        m = cap.max_disjoint_impostor_pairs
        bounds = " / ".join(f"{clopper_pearson_upper(k, m, conf):.2%}" for k in (0, 1, 2))
        print(
            f"| {name} | {gib:.2f} | {cal:,} | {m:,} | {bounds} | {int(n * share['test']):,} | "
            f"{int(n72 * share['train']):,} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
