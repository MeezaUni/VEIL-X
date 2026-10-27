"""Gate 3B: visibility-estimator report (learned vs heuristic vs oracle; ID vs held-out family).

Reads Phase 3 feature shards. For fold F, renders of family F (from that fold's test_id/test_ood
plans on p3_test) are HELD-OUT (OOD) for estimator F; renders of the other families are ID.
Clean p3_test images measure false occlusion. The heuristic's link is fitted on fold F's p3_val
renders (seen families). Up to 20,000 renders per cell (seeded subsample).

The oracle is the ground-truth map itself: its map metrics are perfect by construction; only its
global-visibility discretisation error (28x28 grid vs the engine's pixel-level ratio) is
informative. It is a reference, never a deployable method.

    .venv\\Scripts\\python.exe scripts\\evaluate_visibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.paths import resolve_data_root
from veilx.embeddings.store import read_array_shard
from veilx.provenance import collect_provenance
from veilx.vis.heuristic import fit_link_cells
from veilx.vis.metrics import all_scores
from veilx.vis.regions import global_visibility

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P3 = PROTO.phase3
assert P3 is not None
FAMILIES = tuple(PROTO.family_ids)
CAP = 20_000


def load(feat: Path, name: str, cols: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    parts = sorted(feat.glob(f"{name}.part*.npz"))
    keys, arr = [], {c: [] for c in cols}  # type: ignore[var-annotated]
    for p in parts:
        k, a, _ = read_array_shard(p)
        keys.append(k)
        for c in cols:
            arr[c].append(a[c])
    return np.concatenate(keys), {c: np.concatenate(v) for c, v in arr.items()}


def scores_for(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray | None) -> dict[str, Any]:
    return all_scores(pred, gt, reference_global=vis)


def main() -> int:
    layout = resolve_data_root()
    feat = layout.root / "features" / "phase3"
    sub = json.loads((layout.manifests / "phase3" / "subsplits.json").read_text("utf-8"))
    out: dict[str, Any] = {"folds": {}}
    rng = np.random.default_rng(P3.seed)  # type: ignore[union-attr]
    ck, carr = load(feat, "p3_clean", ("z", *(f"ve_{f[:2]}" for f in FAMILIES)))
    test_clean = np.array([sub[str(k)] == "p3_test" for k in ck.tolist()])
    for fold in FAMILIES:
        _, va = load(feat, f"{fold}__val__p3_val_pairs", ("z", "gt"))
        link = fit_link_cells(va["z"], va["gt"], P3.seed)  # type: ignore[union-attr]
        del va
        cells: dict[str, Any] = {}
        pools: dict[str, dict[str, list[np.ndarray]]] = {"ood": {}, "id": {}}
        for role in ("test_id", "test_ood"):
            keys, a = load(feat, f"{fold}__{role}__p3_test_pairs", ("ve", "gt", "z", "vis"))
            fam = np.array([str(k).split("|")[1] for k in keys.tolist()])
            lvl = np.array([str(k).split("|")[2] for k in keys.tolist()])
            for cell, m in (("ood", fam == fold), ("id", fam != fold)):
                for c, v in (
                    ("ve", a["ve"][m]),
                    ("gt", a["gt"][m]),
                    ("z", a["z"][m]),
                    ("vis", a["vis"][m]),
                    ("lvl", lvl[m]),
                    ("fam", fam[m]),
                ):
                    pools[cell].setdefault(c, []).append(v)
            del a
        for cell, pl in pools.items():
            d = {c: np.concatenate(v) for c, v in pl.items()}
            if len(d["ve"]) > CAP:
                pick = rng.choice(len(d["ve"]), CAP, replace=False)
                d = {c: v[pick] for c, v in d.items()}
            gt = d["gt"].astype(np.float64) / 255
            ve = d["ve"].astype(np.float64) / 255
            z = d["z"].astype(np.float64)
            heur = link.predict_proba(z.reshape(-1, 1)).reshape(z.shape)
            rec: dict[str, Any] = {
                "n_renders": len(gt),
                "families": sorted(set(d["fam"].tolist())),
                "learned": scores_for(ve, gt, d["vis"]),
                "heuristic": scores_for(heur, gt, d["vis"]),
                "oracle_discretisation_global_mae": float(
                    np.mean(np.abs(global_visibility(gt) - d["vis"]))
                ),
                "by_level_learned": {
                    lv: scores_for(ve[d["lvl"] == lv], gt[d["lvl"] == lv], d["vis"][d["lvl"] == lv])
                    for lv in ("low", "medium", "high")
                    if (d["lvl"] == lv).any()
                },
            }
            if cell == "id":
                rec["by_family_learned"] = {
                    f: scores_for(ve[d["fam"] == f], gt[d["fam"] == f], d["vis"][d["fam"] == f])
                    for f in sorted(set(d["fam"].tolist()))
                }
            cells[cell] = rec
        cm = test_clean
        vec = carr[f"ve_{fold[:2]}"][cm].astype(np.float64) / 255
        zc = carr["z"][cm].astype(np.float64)
        hc = link.predict_proba(zc.reshape(-1, 1)).reshape(zc.shape)
        ones = np.ones_like(vec)
        cells["clean_test_images"] = {
            "n": int(cm.sum()),
            "learned_global_mae": float(np.mean(np.abs(global_visibility(vec) - 1))),
            "learned_false_occlusion_cell_rate": float((vec < 0.5).mean()),
            "heuristic_global_mae": float(np.mean(np.abs(global_visibility(hc) - 1))),
            "heuristic_false_occlusion_cell_rate": float((hc < 0.5).mean()),
            "learned_brier": float(np.mean((vec - ones) ** 2)),
        }
        out["folds"][fold] = cells
        print(
            fold,
            {
                c: (
                    round(v["learned"]["iou"] or 0, 3),
                    round(v["learned"]["global_mae"], 4),
                    round(v["heuristic"]["iou"] or 0, 3),
                )
                for c, v in cells.items()
                if c in ("id", "ood")
            },
            flush=True,
        )
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "Gate 3B visibility-estimator evaluation on p3_test renders "
            "(DigiFace, synthetic occlusions)"
        ),
        "note": (
            "Map metrics on face cells of the 28x28 grid; definitions in veilx.vis.metrics. "
            "Oracle = reference only."
        ),
        "results": out,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase3_visibility_eval.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
