# ruff: noqa: E501
"""Phase 6 audit: behaviour of the frozen M1 / A3 systems below the calibration support of quality.

Found in Phase 6 by the smoke test: two random-noise images declared pre-aligned were decided
MATCH with fused score 1.000.
- **Cause.** The frozen logistic M1 = f(s, q, s*q) has a large negative main effect of q.
  Far below the calibration data (very low AdaFace feature norm q), the linear logit
  extrapolates to p -> 1, even for impostor-like similarity.
- **Scope.** The q_low stratum has no lower bound, so such pairs fall in q_low. When
  p >= tau_high they are decided MATCH.

This script measures, on **development data only** (no test partition):
1. per fold, the q distribution of the calibration table (`p3_cal_independent`, fold cal plan);
   the registered app support floor is the 1st percentile of that table;
2. per fold, on `p3_val`, A3 decisions and errors by q band (below the floor vs above);
3. the M1 surface p(s, q) on a grid;
4. if torch and the models are available, synthetic non-face probes (noise, constant,
   gradients) through the frozen pipeline (pre-aligned mode).

Nothing frozen is changed. Aggregates only -> reports/experiments/phase6_lowq_audit.json.

    .venv-gpu\\Scripts\\python.exe scripts\\phase6_lowq_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("F1_mask", "F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")
FLOOR_PCT = 1.0  # registered: 1st percentile of calibration q


def main() -> int:
    from veilx.data.paths import resolve_data_root
    from veilx.fusion.features import design
    from veilx.provenance import collect_provenance

    spec = importlib.util.spec_from_file_location("ev3", ROOT / "scripts" / "evaluate_phase3.py")
    assert spec is not None and spec.loader is not None
    ev3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev3)
    ctx = ev3.Ctx()
    f3 = json.loads((resolve_data_root().private_reports / "phase3_frozen.json").read_text("utf-8"))
    out: dict[str, Any] = {"floor_percentile_of_calibration_q": FLOOR_PCT, "folds": {}}
    grid_s = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    grid_q = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0]
    for fold in FOLDS:
        ff = f3["folds"][fold]
        link = ev3.LogisticModel.from_dict(ff["heuristic_link"])
        m1 = ev3.LogisticModel.from_dict(ff["models"]["M1"])
        rec = ff["arms"]["A3_quality|primary|v2"]
        clean = ctx.clean(fold)
        res: dict[str, Any] = {}
        tabs = {}
        for role, pname in (("cal", "p3_cal_independent"), ("val", "p3_val_pairs")):
            pairs = ctx.pairs[pname]
            rows = ctx.plan(fold, role)
            tabs[role] = ev3.build_table(
                pairs,
                {p.pair_id: rows[p.pair_id] for p in pairs},
                clean,
                ctx.occ(fold, role),
                fold,
                link,
                ctx.ident,
            )
        qc = tabs["cal"]["q"]
        floor = float(np.percentile(qc, FLOOR_PCT))
        res["calibration_q"] = {
            "n": len(qc),
            "min": float(qc.min()),
            "p0.5": float(np.percentile(qc, 0.5)),
            "p1": floor,
            "p5": float(np.percentile(qc, 5)),
            "median": float(np.median(qc)),
        }
        res["q_support_floor"] = floor
        t = tabs["val"]
        p = m1.predict_proba(design(t, "M1", "cv_region")[0])
        d = ev3.decide(rec, p, ev3.strata_labels("quality", t, ff["edges"], "cv_region"))
        y = np.asarray(t["label"]).astype(int)  # 1 genuine, 0 impostor
        bands = {"q < floor": t["q"] < floor, "q >= floor": t["q"] >= floor}
        res["val_by_band"] = {}
        for name, m in bands.items():
            g, i = m & (y == 1), m & (y == 0)
            acc_g, acc_i = g & (d != -1), i & (d != -1)
            res["val_by_band"][name] = {
                "n_genuine": int(g.sum()),
                "n_impostor": int(i.sum()),
                "impostor_decided_match": int((i & (d == 1)).sum()),
                "genuine_decided_nonmatch": int((g & (d == 0)).sum()),
                "abstain": int((m & (d == -1)).sum()),
                "sFMR": float((i & (d == 1)).sum() / max(acc_i.sum(), 1)),
                "sFNMR": float((g & (d == 0)).sum() / max(acc_g.sum(), 1)),
            }
        res["m1_surface"] = {
            f"s={sv:+.1f}": [
                float(x)
                for x in m1.predict_proba(
                    design(
                        {"s": np.full(len(grid_q), sv), "q": np.array(grid_q)}, "M1", "cv_region"
                    )[0]
                )
            ]
            for sv in grid_s
        }
        out["folds"][fold] = res
        print(fold, json.dumps({k: res[k] for k in ("q_support_floor", "val_by_band")}), flush=True)
    out["m1_surface_q_grid"] = grid_q
    try:
        out["synthetic_non_face_probes"] = probes(out)
    except ImportError as exc:
        out["synthetic_non_face_probes"] = f"skipped ({exc})"
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 6 audit of the frozen M1/A3 systems below the calibration support of quality (development data p3_cal/p3_val; synthetic non-face probes). Exploratory; nothing frozen changed.",
        "results": out,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase6_lowq_audit.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1)
    return 0


def probes(out: dict[str, Any]) -> list[dict[str, Any]]:
    import torch  # noqa: F401  (GPU environment)

    from veilx.app.config import load_app_config
    from veilx.app.pipeline import VerificationPipeline
    from veilx.data.paths import resolve_data_root

    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    pipe = VerificationPipeline(cfg, resolve_data_root().root)
    rng = np.random.default_rng(20261102)
    x = np.linspace(0, 255, 112).astype(np.uint8)
    cases = [
        (
            f"uniform noise pair {k}",
            rng.integers(0, 255, (112, 112, 3), np.uint8),
            rng.integers(0, 255, (112, 112, 3), np.uint8),
        )
        for k in range(5)
    ]
    cases += [
        (
            "constant grey 128 vs 60",
            np.full((112, 112, 3), 128, np.uint8),
            np.full((112, 112, 3), 60, np.uint8),
        ),
        (
            "black vs white",
            np.zeros((112, 112, 3), np.uint8),
            np.full((112, 112, 3), 255, np.uint8),
        ),
        (
            "horizontal vs vertical gradient",
            np.repeat(np.repeat(x[None, :, None], 112, 0), 3, 2),
            np.repeat(np.repeat(x[:, None, None], 112, 1), 3, 2),
        ),
    ]
    res = []
    for name, a, b in cases:
        r = pipe.verify(a, b, pre_aligned=True, with_diagnostics=True, apply_support_floor=False)
        dg = r.diagnostics or {}
        res.append(
            {
                "case": name,
                "s_adaface": r.s_adaface,
                "q": r.quality,
                "fused_score": r.decision.fused_score,
                "frozen_a3_label": r.decision.label,
                "below_support_floor_F1": bool(
                    r.quality is not None and r.quality < out["folds"]["F1_mask"]["q_support_floor"]
                ),
                "novelty_flagged_both": bool(all(dg.get("novel", [False, False]))),
            }
        )
        print(res[-1], flush=True)
    return res


if __name__ == "__main__":
    sys.exit(main())
