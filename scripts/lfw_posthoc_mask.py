"""POST-HOC, EXPLORATORY audit of the mask false-match effect on LFW (not pre-registered).

Motivation: the pre-registered LFW run (79331a84f520) used the DigiFace-fixed threshold, at which
almost no LFW impostor is accepted in any condition, so a mask-specific false-match increase
could not be detected. This script reads the SAME LFW View-2 embeddings and reports, per family
and level, for impostor pairs with BOTH sides occluded versus the same pairs clean:

* the paired mean impostor-score shift (threshold-free; identity-jackknife interval);
* FMR at the in-domain View-1 DevTrain EER threshold (flagged: DevTrain shares identities with
  View 2).

Logged in the test-run log with its purpose; results are labelled post-hoc exploratory.

    .venv\\Scripts\\python.exe scripts\\lfw_posthoc_mask.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.paths import resolve_data_root
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import read_shard
from veilx.eval.summaries import rate
from veilx.models.spec import BACKBONES
from veilx.provenance import collect_provenance
from veilx.stats.jackknife import jackknife_interval

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
_spec = importlib.util.spec_from_file_location("evaluate_lfw", ROOT / "scripts" / "evaluate_lfw.py")
assert _spec and _spec.loader
EV = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_lfw"] = EV
_spec.loader.exec_module(EV)


def main() -> int:
    layout = resolve_data_root()
    prov = collect_provenance(ROOT, argv=sys.argv)
    run_id = uuid.uuid4().hex[:12]
    log = ROOT / PROTO.test_access.run_log
    with log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "utc": datetime.now(tz=UTC).isoformat(),
                    "script": "scripts/lfw_posthoc_mask.py",
                    "splits": ["lfw_view2"],
                    "note": (
                        "post-hoc exploratory mask false-match audit "
                        "on embeddings of run 79331a84f520"
                    ),
                    "git_commit": (prov.get("git") or {}).get("commit"),
                }
            )
            + "\n"
        )
    data = EV.LFWData(layout, "view2_pairs")
    out: dict[str, Any] = {}
    for bid in BACKBONES:
        d = next(
            p for p in (layout.embeddings / bid).iterdir() if (p / "lfw_det_clean.npz").exists()
        )
        clean = read_shard(d / "lfw_det_clean.npz")
        occ = EV.read_parts(d, "lfw_det_occl")
        ci, oi = clean.index(), occ.index()
        t_dev = EV._devtrain_thresholds(layout, clean)["eer"]
        res: dict[str, Any] = {"devtrain_eer_threshold": t_dev}
        rows: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
        pos = {p.pair_id: k for k, p in enumerate(data.pairs)}
        for r in data.plan:
            if r["mode"] != "both":
                continue
            k = pos[r["pair_id"]]
            if data.label[k] != 0:
                continue
            p = data.pairs[k]
            ka = occluded_key(p.image_a, r["family"], r["level"], r["a"]["seed"])
            kb = occluded_key(p.image_b, r["family"], r["level"], r["b"]["seed"])
            s_occ = float(occ.emb[oi[ka]] @ occ.emb[oi[kb]])
            s_cln = float(clean.emb[ci[p.image_a]] @ clean.emb[ci[p.image_b]])
            rows.setdefault((r["family"], r["level"]), []).append((k, s_occ, s_cln))
        for (fam, lvl), rr in sorted(rows.items()):
            k = np.array([x[0] for x in rr])
            so, sc = np.array([x[1] for x in rr]), np.array([x[2] for x in rr])
            diff = so - sc
            jk = jackknife_interval(
                diff, np.ones(diff.size), data.id_a[k], data.id_b[k], scale="linear"
            )
            sp_o = EV._rows_sp(
                data, [(i, v, 1.0, 1.0, 0.0, 0.0) for i, v in zip(k, so, strict=True)], fam
            )
            sp_c = EV._rows_sp(
                data, [(i, v, 1.0, 1.0, 1.0, 1.0) for i, v in zip(k, sc, strict=True)], ""
            )
            res[f"{fam}|{lvl}"] = {
                "n_impostor_pairs": int(k.size),
                "mean_score_shift": jk.estimate,
                "shift_ci": [jk.ci_low, jk.ci_high],
                "fmr_occluded_at_devtrain_eer": rate(
                    sp_o.score >= t_dev, sp_o.label == 0, sp_o, "impostor"
                ),
                "fmr_clean_same_pairs_at_devtrain_eer": rate(
                    sp_c.score >= t_dev, sp_c.label == 0, sp_c, "impostor"
                ),
            }
        out[bid] = res
    report = {
        "evidence_type": "real_measured_model_result",
        "label": "POST-HOC EXPLORATORY (not pre-registered)",
        "scope": "LFW View-2 impostor pairs, both sides with the same synthetic occlusion vs clean",
        "run_id": run_id,
        "results": out,
        "provenance": prov,
    }
    with (ROOT / "reports" / "experiments" / "lfw_posthoc_mask.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"run {run_id} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
