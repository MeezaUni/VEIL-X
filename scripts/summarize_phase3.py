# ruff: noqa: E501, RUF001  (markdown table templates: long rows and typographic minus signs are intended)
"""Render the Phase 3 result JSONs as markdown tables (reproducible input to docs/phase3_results.md).

Reads only committed aggregate reports (no private data):
``reports/experiments/phase3_{final,visibility_eval,develop,followup,lfw_detector_inclusive}.json``.

    .venv\\Scripts\\python.exe scripts\\summarize_phase3.py > reports/experiments/phase3_tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
FOLDS = ("F1_mask", "F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")
SHORT = {f: f.split("_", 1)[1].replace("_", " ") for f in FOLDS}
ARMS = (
    "A1_clean",
    "A2_global_M0",
    "A2_global_M4",
    "A2_global_M0arc",
    "A3_quality",
    "A4_predvis",
    "A5_oracle",
    "A6_quality_x_predvis",
)


def _load(name: str) -> dict[str, Any] | None:
    p = EXP / f"{name}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def pct(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}"


def rate(r: dict[str, Any] | None, verdict: str) -> str:
    """Rate in % with a marker: ✗ CI excludes target (violation), † point estimate above."""
    if r is None or r.get("value") is None:
        return "—"
    mark = " ✗" if verdict.startswith("violated") else (" †" if verdict.startswith("point") else "")
    return f"{100 * r['value']:.2f}{mark}"


def p_fmt(p: float | None) -> str:
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def arm_cell(a: dict[str, Any]) -> str:
    return (
        f"{pct(a['coverage'])} / {rate(a['selective_fnmr'], a['verdict_fnm'])} / "
        f"{rate(a['selective_fmr'], a['verdict_fm'])}"
    )


def section_visibility(v: dict[str, Any]) -> None:
    print("## Visibility estimator accuracy (gate 3B; p3_test renders)\n")
    print(
        "| Fold | Cell | Learned IoU | Learned global MAE | Learned bias | Learned ECE | "
        "Heuristic IoU | Heuristic global MAE |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for f in FOLDS:
        for cell, name in (("id", "seen families"), ("ood", "held-out family")):
            c = v["folds"][f][cell]
            lrn, heu = c["learned"], c["heuristic"]
            bias = lrn.get("global_bias")
            print(
                f"| {SHORT[f]} | {name} | {lrn['iou']:.3f} | {lrn['global_mae']:.3f} | "
                f"{'—' if bias is None else f'{bias:+.3f}'} | {lrn['ece']:.3f} | "
                f"{heu['iou']:.3f} | {heu['global_mae']:.3f} |"
            )
    print("\nClean p3_test images (learned): global MAE / false-occlusion cell rate")
    for f in FOLDS:
        c = v["folds"][f]["clean_test_images"]
        print(
            f"- {SHORT[f]}: {c['learned_global_mae']:.4f} / {c['learned_false_occlusion_cell_rate']:.4f} "
            f"(heuristic {c['heuristic_global_mae']:.3f} / {c['heuristic_false_occlusion_cell_rate']:.3f})"
        )
    print()


def section_arms(r: dict[str, Any], role: str, target: str, ordering: str) -> None:
    print(f"### {role}: coverage / sFNMR / sFMR, % (target {target}, ordering {ordering})\n")
    print("| Arm | " + " | ".join(SHORT[f] for f in FOLDS) + " |")
    print("|---|" + "---|" * len(FOLDS))
    for arm in ARMS:
        cells = [arm_cell(r["folds"][f][role]["arms"][f"{arm}|{target}|{ordering}"]) for f in FOLDS]
        print(f"| {arm} | " + " | ".join(cells) + " |")
    print()


def count_viol(r: dict[str, Any], role: str, key: str) -> tuple[int, int]:
    ci = pt = 0
    for f in FOLDS:
        a = r["folds"][f][role]["arms"][key]
        for v in (a["verdict_fnm"], a["verdict_fm"]):
            ci += v.startswith("violated")
            pt += v.startswith("point")
    return ci, pt


def section_final(d: dict[str, Any]) -> None:
    r = d["results"]
    print(
        f"## Confirmatory run `{d['run_id']}` (frozen `{d['frozen_sha256'][:12]}`; "
        f"co-visibility `{d['cv_definition']}`)\n"
    )
    print(
        "Cells: coverage / sFNMR / sFMR in %. ✗ = 95 % CI excludes the target (violation); "
        "† = point estimate above target.\n"
    )
    for role in ("test_ood", "test_id"):
        section_arms(r, role, "primary", "v2")
    print("### C3 (primary): A4 − A3 on Test-OOD, paired identity jackknife\n")
    print("| Fold | ΔsFNMR (pp) | 95 % CI | p | Holm p | Δcoverage (pp) | ΔsFMR (pp) |")
    print("|---|---|---|---|---|---|---|")
    for f in FOLDS:
        b = r["folds"][f]["test_ood"]
        c = b["c3_A4_minus_A3"]
        s = c.get("sFNMR", {})
        print(
            f"| {SHORT[f]} | {100 * s['estimate']:+.2f} | [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] | "
            f"{p_fmt(s['p_value'])} | {p_fmt(b.get('c3_holm_p_sFNMR'))} | "
            f"{100 * c['coverage']['estimate']:+.2f} | {100 * c['sFMR']['estimate']:+.2f} |"
        )
    print(
        "\n### C1 (primary): log-loss(M1) − log-loss(M4) on occluded pairs (positive = co-visibility helps)\n"
    )
    print("| Fold | Test-ID Δ | Holm p | Test-OOD Δ | Holm p |")
    print("|---|---|---|---|---|")
    for f in FOLDS:
        i, o = r["folds"][f]["test_id"], r["folds"][f]["test_ood"]
        print(
            f"| {SHORT[f]} | {i['c1_logloss_M1_minus_M4']['estimate']:+.4f} | {p_fmt(i.get('c1_holm_p'))} | "
            f"{o['c1_logloss_M1_minus_M4']['estimate']:+.4f} | {p_fmt(o.get('c1_holm_p'))} |"
        )
    print("\n### H4: clean FNMR at frozen clean-cal FAR-1 % thresholds\n")
    print(
        "| Fold | FNMR M0 | FNMR M4 | Δ (pp) | 95 % CI | FMR M0 | FMR M4 | Non-inferior (0.5 pp) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for f in FOLDS:
        h = r["folds"][f]["h4_clean"]
        dd = h["delta_fnmr_M4_minus_M0"]
        print(
            f"| {SHORT[f]} | {pct(h['fnmr_M0'], 2)} | {pct(h['fnmr_M4'], 2)} | {100 * dd['estimate']:+.2f} | "
            f"[{100 * dd['ci_low']:+.2f}, {100 * dd['ci_high']:+.2f}] | {pct(h['fmr_M0'], 2)} | "
            f"{pct(h['fmr_M4'], 2)} | {'yes' if h['non_inferior_at_0.5pp'] else 'no'} |"
        )
    print("\n### Verdicts\n")
    print("```json\n" + json.dumps(r["verdicts"], indent=1) + "\n```\n")
    print("### Violation counts on Test-OOD (5 folds × 2 error types)\n")
    print("| Arm | primary v2 ✗/† | primary v1 ✗/† | far1 v2 ✗/† | far1 v1 ✗/† |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        cells = []
        for t, o in (("primary", "v2"), ("primary", "v1"), ("far1", "v2"), ("far1", "v1")):
            ci, pt = count_viol(r, "test_ood", f"{arm}|{t}|{o}")
            cells.append(f"{ci}/{pt}")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    print("\n### Mean Test-OOD coverage (%) by ordering, primary target\n")
    print("| Arm | v2 | v1 |")
    print("|---|---|---|")
    for arm in ARMS:
        vals = []
        for o in ("v2", "v1"):
            cov = [
                r["folds"][f]["test_ood"]["arms"][f"{arm}|primary|{o}"]["coverage"] for f in FOLDS
            ]
            vals.append(pct(sum(cov) / len(cov)))
        print(f"| {arm} | {vals[0]} | {vals[1]} |")
    for target in ("far1",):
        print()
        section_arms(r, "test_ood", target, "v2")
    print("### Fusion ladder on occluded pairs: AUC / log-loss / AUGRC (×100)\n")
    models = ("M0", "M0_arc", "M1", "M2", "M3", "M4", "M5", "M4_heur", "M4_oracle")
    for role in ("test_ood", "test_id"):
        print(f"**{role}**\n")
        print("| Model | " + " | ".join(SHORT[f] for f in FOLDS) + " |")
        print("|---|" + "---|" * len(FOLDS))
        for m in models:
            cells = []
            for f in FOLDS:
                x = r["folds"][f][role]["models"][m]
                ag = x.get("augrc_occluded")
                cells.append(
                    f"{x['auc_occluded']:.4f} / {x['log_loss']:.4f} / {'—' if ag is None else f'{100 * ag:.3f}'}"
                )
            print(f"| {m} | " + " | ".join(cells) + " |")
        print()
    print("### A4 abstention reasons on Test-OOD (non-exclusive counts; not causal)\n")
    print(
        "| Fold | abstained | uncertifiable stratum | low co-visibility | low quality | model conflict | in uncertainty band |"
    )
    print("|---|---|---|---|---|---|---|")
    for f in FOLDS:
        a = r["folds"][f]["test_ood"]["arms"]["A4_predvis|primary|v2"][
            "abstention_reasons_nonexclusive"
        ]
        print(
            f"| {SHORT[f]} | {a['n_abstained']} | {a['uncertifiable_stratum']} | {a['insufficient_covisibility_vlow']} | "
            f"{a['poor_quality_qlow']} | {a['conflicting_models']} | {a['uncertain_score_in_band']} |"
        )
    print("\n### A4 vs A3 by severity on Test-OOD (coverage / sFNMR, %)\n")
    print("| Fold | Arm | clean | low | medium | high |")
    print("|---|---|---|---|---|---|")
    for f in FOLDS:
        for arm in ("A4_predvis", "A3_quality"):
            bl = r["folds"][f]["test_ood"]["arms"][f"{arm}|primary|v2"]["by_level"]
            cells = []
            for lv in ("0", "1", "2", "3"):
                x = bl.get(lv)
                cells.append(
                    "—"
                    if x is None
                    else f"{pct(x['coverage'])} / {pct((x['selective_fnmr'] or {}).get('value'), 2)}"
                )
            print(f"| {SHORT[f]} | {arm} | " + " | ".join(cells) + " |")
    print()


def section_followup(d: dict[str, Any]) -> None:
    r = d["results"]
    print(f"## Follow-up run `{d['run_id']}` (frozen rules; no fitting)\n")
    for name, key in (
        ("Phase 2 DigiFace test (held-out family plans)", "phase2_digiface_test"),
        ("LFW View 2 derived occlusions (all modes)", "lfw_derived"),
    ):
        if key not in r:
            continue
        print(f"### {name}: coverage / sFNMR / sFMR, %\n")
        print("| Arm | " + " | ".join(SHORT[f] for f in FOLDS) + " |")
        print("|---|" + "---|" * len(FOLDS))
        sub = "test_ood" if key == "phase2_digiface_test" else "all"
        blocks = {f: r[key][f][sub] for f in FOLDS}
        for arm in ARMS:
            print(
                f"| {arm} | "
                + " | ".join(arm_cell(blocks[f]["arms"][f"{arm}|primary|v2"]) for f in FOLDS)
                + " |"
            )
        print("\n| Fold | C3 ΔsFNMR A4−A3 (pp) [CI] | C1 ΔLL M1−M4 [CI] | H4 FNMR M0 → M4 (%) |")
        print("|---|---|---|---|")
        for f in FOLDS:
            b = blocks[f]
            s = b["c3_A4_minus_A3"].get("sFNMR")
            c1 = b["c1_logloss_M1_minus_M4"]
            h = r[key][f]["h4_clean"]
            s_txt = (
                "—"
                if s is None
                else f"{100 * s['estimate']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}]"
            )
            print(
                f"| {SHORT[f]} | {s_txt} | {c1['estimate']:+.4f} [{c1['ci'][0]:+.4f}, {c1['ci'][1]:+.4f}] | "
                f"{pct(h['fnmr_M0'], 2)} → {pct(h['fnmr_M4'], 2)} |"
            )
        print()


def section_detector(d: dict[str, Any]) -> None:
    r = d["results"]
    print(f"## Exploratory detector-inclusive LFW study `{d['run_id']}` ({r['n_pairs']} pairs)\n")
    print(
        "| Family | Composites | Detection failures | Failure rate low/med/high (%) | "
        "Landmark shift mean (px) | A4 cov / sFNMR / sFMR | A3 cov / sFNMR / sFMR |"
    )
    print("|---|---|---|---|---|---|---|")
    for f in FOLDS:
        x = r["families"][f]
        fr = x["detection_failure_rate_by_level"]
        dp = x.get("detected_pairs")
        a4 = arm_cell(dp["arms"]["A4_predvis|primary|v2"]) if dp else "—"
        a3 = arm_cell(dp["arms"]["A3_quality|primary|v2"]) if dp else "—"
        lm = x["landmark_disp_mean_px"]
        print(
            f"| {SHORT[f]} | {x['n_composites']} | {x['detection_failures']} | "
            f"{pct(fr['low'])}/{pct(fr['medium'])}/{pct(fr['high'])} | {'—' if lm is None else f'{lm:.2f}'} | {a4} | {a3} |"
        )
    print()


def main() -> int:
    v = _load("phase3_visibility_eval")
    if v:
        section_visibility(v["results"])
    d = _load("phase3_final")
    if d:
        section_final(d)
    fu = _load("phase3_followup")
    if fu:
        section_followup(fu)
    di = _load("phase3_lfw_detector_inclusive")
    if di:
        section_detector(di)
    return 0


if __name__ == "__main__":
    sys.exit(main())
