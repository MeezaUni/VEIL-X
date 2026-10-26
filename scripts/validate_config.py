"""Validate the research protocol and dataset registry; print a short summary.

    python scripts/validate_config.py [--protocol ...] [--datasets ...]

Exit code 0 = valid, 2 = invalid (message printed, no traceback).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from veilx.config import (
    DatasetRole,
    config_sha256,
    load_dataset_registry,
    load_protocol,
)
from veilx.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs" / "research_protocol.yaml")
    ap.add_argument("--datasets", type=Path, default=ROOT / "configs" / "datasets.yaml")
    args = ap.parse_args()
    try:
        proto = load_protocol(args.protocol)
        reg = load_dataset_registry(args.datasets)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2

    print(f"protocol  {proto.protocol_id} v{proto.protocol_version} status={proto.status.value}")
    print(f"          sha256={config_sha256(args.protocol)}")
    print(f"registry  checked_on={reg.checked_on} sha256={config_sha256(args.datasets)}")
    print(f"families  {', '.join(proto.family_ids)}")
    print("provisional:")
    for name in proto.provisional_constants():
        print(f"  - {name}")
    print("datasets by access status:")
    for d in reg.datasets:
        roles = ",".join(r.value for r in d.roles)
        print(f"  {d.access_status.value:<11} {d.id:<42} roles={roles}")
    for role in (DatasetRole.TRAIN, DatasetRole.CAL, DatasetRole.TEST_REAL):
        ids = [d.id for d in reg.usable_for(role)] or ["(none confirmed)"]
        print(f"usable now for {role.value:<10}: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
