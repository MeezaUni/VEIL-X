"""Run the synthetic statistical simulations and write labelled JSON reports.

    python scripts/run_simulations.py                  # full scale -> reports/simulations/
    python scripts/run_simulations.py --scale quick --out <dir>
    python scripts/run_simulations.py --only bootstrap_coverage

Outputs are SIMULATED results about statistical procedures, stored apart from any model
results; they are registered in reports/evidence_registry.yaml as ``synthetic_simulation``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from veilx.sim.experiments import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=("quick", "full"), default="full")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "simulations")
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(EXPERIMENTS):
        t0 = time.perf_counter()
        result = EXPERIMENTS[name](scale=args.scale)
        path = args.out / f"{name}.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"{name:<30} {time.perf_counter() - t0:7.1f}s -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
