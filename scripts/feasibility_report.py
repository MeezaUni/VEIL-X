"""Statistical feasibility of the provisional constants (exact binomial calculations).

Prints a Markdown report. Inputs are the registered constants plus dataset sizes from
configs/datasets.yaml. These are *calculations*, not experimental results.

    python scripts/feasibility_report.py [--protocol configs/research_protocol.yaml]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from veilx.config import load_protocol
from veilx.stats.feasibility import (
    IndependentPairCapacity,
    clopper_pearson_upper,
    min_n_to_certify,
    zero_error_n,
)

ROOT = Path(__file__).resolve().parents[1]

# Identity counts per candidate source (DigiFace counts from the official README;
# ROF counts from the ROF README). Calibration band = 10% of identities (protocol bands).
SOURCES: list[tuple[str, int]] = [
    ("DigiFace 72-img part P1 (2,000 IDs)", 2_000),
    ("DigiFace 72-img parts P1-P5 (10,000 IDs)", 10_000),
    ("DigiFace 5-img part P1 (~33,333 IDs)", 33_333),
    ("DigiFace 5-img parts P1-P3 (~100,000 IDs)", 100_000),
]


def _fmt(x: float) -> str:
    return f"{x:.2e}" if x < 1e-2 else f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs" / "research_protocol.yaml")
    args = ap.parse_args()
    proto = load_protocol(args.protocol)
    c = proto.constants
    conf = float(c.confidence_level.value)  # type: ignore[arg-type]
    target = float(c.accepted_decision_error_target.value)  # type: ignore[arg-type]
    fars = c.far_operating_points.value
    assert isinstance(fars, tuple)
    gfrac = proto.pairs.genuine_fraction_constructed_sets
    cal_lo, cal_hi = proto.splits.bands[next(s for s in proto.splits.bands if s.value == "cal")]
    cal_frac = (cal_hi - cal_lo) / 100

    out: list[str] = []
    out.append(f"# Feasibility report (protocol {proto.protocol_id} v{proto.protocol_version})\n")
    out.append(
        f"Confidence level: {conf}. Accepted-decision error target: {target}. "
        f"FAR points: {list(fars)}. Genuine fraction: {gfrac}. Cal band: {cal_frac:.0%}.\n"
    )

    out.append(
        "## A. Independent impostor pairs needed to certify FMR <= FAR (zero false matches)\n"
    )
    out.append("| FAR target | n with 0 errors | n with 1 error | n with 2 errors |")
    out.append("|---|---|---|---|")
    for far in fars:
        n0 = zero_error_n(far, conf)
        n1 = _min_n_k(far, 1, conf)
        n2 = _min_n_k(far, 2, conf)
        out.append(f"| {far} | {n0:,} | {n1:,} | {n2:,} |")

    out.append("\n## B. Independent accepted pairs needed to certify selective error <= target\n")
    out.append("| Observed error rate | Minimum n (accepted cal pairs) |")
    out.append("|---|---|")
    for r in (0.0, 0.01, 0.02, 0.03, 0.04, 0.045):
        n = min_n_to_certify(target, r, conf)
        out.append(f"| {r:.3f} | {'not certifiable' if n is None else f'{n:,}'} |")

    out.append("\n## C. What identity-disjoint calibration sets can certify\n")
    out.append(
        "Cal identities = cal band x source identities (expected value; hashing varies). "
        "Identity-disjoint matching: genuine pair uses 1 identity, impostor pair uses 2.\n"
    )
    out.append(
        "| Source | Cal IDs | Indep. genuine | Indep. impostor | Best certifiable FMR (0 FM) "
        "| Error bound if 2% observed, all pairs accepted |"
    )
    out.append("|---|---|---|---|---|---|")
    for name, n_ids in SOURCES:
        cal_ids = math.floor(n_ids * cal_frac)
        cap = IndependentPairCapacity.from_identities(cal_ids, genuine_fraction=gfrac)
        n_total = cap.max_disjoint_genuine_pairs + cap.max_disjoint_impostor_pairs
        fmr_bound = clopper_pearson_upper(0, cap.max_disjoint_impostor_pairs, conf)
        err_bound = clopper_pearson_upper(math.floor(0.02 * n_total), n_total, conf)
        out.append(
            f"| {name} | {cal_ids:,} | {cap.max_disjoint_genuine_pairs:,} | "
            f"{cap.max_disjoint_impostor_pairs:,} | {_fmt(fmr_bound)} | {_fmt(err_bound)} |"
        )

    out.append("\n## D. Test-side estimation (not certification)\n")
    out.append(
        "Expected false matches at a given FAR = n_impostor x FAR. The protocol requires "
        f">= {proto.pairs.min_impostor_pairs_factor:g}/FAR impostor pairs per reported FAR.\n"
    )
    out.append("| FAR | Required impostor pairs | Expected FM events |")
    out.append("|---|---|---|")
    for far in fars:
        req = math.ceil(proto.pairs.min_impostor_pairs_factor / far)
        out.append(f"| {far} | {req:,} | {req * far:.0f} |")
    out.append(
        "\nROF (test-only): 180 identities, 70 with masked images. Impostor *image* pairs are "
        "plentiful, but identity-cluster bootstrap resamples at most 180 (70 for masked) "
        "units, so CI width at FAR=0.001 must be established by simulation in Phase 1 "
        "before any claim."
    )
    print("\n".join(out))
    return 0


def _min_n_k(target: float, k: int, conf: float) -> int:
    n = k + 1
    while clopper_pearson_upper(k, n, conf) > target:
        n = math.ceil(n * 1.05) + 1
    lo, hi = max(k + 1, int(n / 1.06) - 2), n
    while lo < hi:
        mid = (lo + hi) // 2
        if clopper_pearson_upper(k, mid, conf) <= target:
            hi = mid
        else:
            lo = mid + 1
    return hi


if __name__ == "__main__":
    sys.exit(main())
