# ruff: noqa: E501
"""Reference ranges used by the app's plain-language explanations (development data only).

The app tells the user where a pair's similarity and quality sit relative to typical
same-person and different-person pairs. This script measures those ranges once on the
development split ``p3_val`` (default fold F1 plan: clean and synthetically occluded pairs) from
the Phase 3 feature store.

The output is ``configs/app/explain_reference.json``: aggregate quantiles only, with no
identities or embeddings.

    .venv-gpu\\Scripts\\python.exe scripts\\make_explain_reference.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from veilx.data.paths import resolve_data_root
    from veilx.provenance import collect_provenance

    spec = importlib.util.spec_from_file_location("ev3", ROOT / "scripts" / "evaluate_phase3.py")
    assert spec is not None and spec.loader is not None
    ev3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev3)
    ctx = ev3.Ctx()
    fz3 = json.loads(
        (resolve_data_root().private_reports / "phase3_frozen.json").read_text("utf-8")
    )
    fold = "F1_mask"
    link = ev3.LogisticModel.from_dict(fz3["folds"][fold]["heuristic_link"])
    pairs = ctx.pairs["p3_val_pairs"]
    rows = ctx.plan(fold, "val")
    t = ev3.build_table(
        pairs,
        {p.pair_id: rows[p.pair_id] for p in pairs},
        ctx.clean(fold),
        ctx.occ(fold, "val"),
        fold,
        link,
        ctx.ident,
    )
    y = np.asarray(t["label"]).astype(int)
    s, q = np.asarray(t["s"], float), np.asarray(t["q"], float)
    qs = (10, 25, 50, 75, 90)

    def pct(x: np.ndarray) -> dict[str, float]:
        return {f"p{k}": round(float(np.percentile(x, k)), 4) for k in qs}

    ref = {
        "scope": "Development data (p3_val, fold F1 plan: clean + synthetic occlusions; DigiFace-1M synthetic faces). Aggregates only. Used for plain-language context in the app, never for decisions.",
        "n_pairs": len(y),
        "similarity_same_person": pct(s[y == 1]),
        "similarity_different_people": pct(s[y == 0]),
        "quality": pct(q),
        "provenance": {
            k: v
            for k, v in collect_provenance(ROOT, argv=sys.argv).items()
            if k in ("git", "timestamp_utc", "argv")
        },
    }
    out = ROOT / "configs" / "app" / "explain_reference.json"
    out.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                k: ref[k]
                for k in (
                    "n_pairs",
                    "similarity_same_person",
                    "similarity_different_people",
                    "quality",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
