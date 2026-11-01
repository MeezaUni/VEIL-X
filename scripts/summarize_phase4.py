# ruff: noqa: E501, RUF001  (markdown table templates)
"""Render Phase 4 result JSONs as markdown tables (input to docs/phase4_results.md).

Reads committed aggregate reports only: phase4_final.json, phase4_develop.json,
phase4_followup_p3test.json, phase4_detector_{dev,view2}.json, phase4_cost.json,
phase4_diagnostics.json.

    .venv\\Scripts\\python.exe scripts\\summarize_phase4.py > reports/experiments/phase4_tables.md
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
    "B0_raw_global",
    "B1_quality_global",
    "A3_quality",
    "A4_original",
    "C4_conservative",
    "N4_novelty_aware",
    "H4_quality_first_hybrid",
    "A5_oracle",
)


def load(name: str) -> dict[str, Any] | None:
    p = EXP / f"{name}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def rate(r: dict[str, Any] | None, verdict: str) -> str:
    if not r or r.get("value") is None:
        return "—"
    mark = " ✗" if verdict.startswith("violated") else (" †" if verdict.startswith("point") else "")
    return f"{100 * r['value']:.2f}{mark}"


def cell(a: dict[str, Any]) -> str:
    return f"{100 * a['coverage']:.1f} / {rate(a['selective_fnmr'], a['verdict_fnm'])} / {rate(a['selective_fmr'], a['verdict_fm'])}"


def pfmt(p: float | None) -> str:
    return "—" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def arms_table(cells: dict[str, Any], role: str, key: str) -> None:
    print(f"### {role}: coverage / sFNMR / sFMR, % ({key})\n")
    print("| Arm | " + " | ".join(SHORT[f] for f in FOLDS) + " |")
    print("|---|" + "---|" * len(FOLDS))
    for arm in ARMS:
        row = [
            cell(cells[f"{f}|{role}"]["arms"][f"{arm}|{key}"]) if f"{f}|{role}" in cells else "—"
            for f in FOLDS
        ]
        print(f"| {arm} | " + " | ".join(row) + " |")
    print()


def final_section(d: dict[str, Any]) -> None:
    c = d["cells"]
    print(f"## Confirmatory run `{d['run_id']}` (frozen `{d['frozen_sha256'][:12]}`)\n")
    print("✗ = 95 % identity-jackknife CI excludes the target; † = point estimate above target.\n")
    for role in ("test_ood", "e1", "test_id"):
        arms_table(c, role, "primary|v2")
    arms_table(c, "test_ood", "far1|v2")
    arms_table(c, "e1", "far1|v2")
    e = d["endpoints"]
    print("### Endpoints (P1 safety, P2 benefit; 10 primary cells)\n")
    print(
        "| Arm | violations (primary v2) | safe | sig. worse sFNMR cells | sig. worse sFMR cells | mean coverage gain vs A3 (pp) | cells sig. gain | better than A3 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for x in (
        "A3_quality",
        "A4_original",
        "A5_oracle",
        "B0_raw_global",
        "B1_quality_global",
        "C4_conservative",
        "N4_novelty_aware",
        "H4_quality_first_hybrid",
    ):
        v = e["violations_primary_v2"][x]
        if x in e:
            r = e[x]
            print(
                f"| {x} | {v} | {r['safe']} | {', '.join(r['significantly_worse_cells']['sFNMR']) or '—'} | {', '.join(r['significantly_worse_cells']['sFMR']) or '—'} | {100 * r['mean_coverage_gain']:+.2f} | {r['cells_significant_coverage_gain']} | {r['better_than_A3']} |"
            )
        else:
            print(f"| {x} | {v} | — | — | — | — | — | — |")
    print(f"\n**Registered recommendation: `{e['recommendation']}`**\n")
    print("### Paired differences vs A3 (primary v2): ΔsFNMR / ΔsFMR / Δcoverage, pp\n")
    print("| Cell | C4 | N4 | H4 | A4 original | A5 oracle |")
    print("|---|---|---|---|---|---|")
    for k in sorted(c):
        if k.endswith("|test_id"):
            continue
        pv = c[k]["paired_vs_a3"]

        def tri(x: str, pv: dict[str, Any] = pv) -> str:
            g = pv[x]
            return " / ".join(
                f"{100 * g[m]['estimate']:+.2f}" if m in g else "—"
                for m in ("sFNMR", "sFMR", "coverage")
            )

        print(
            f"| {k} | {tri('C4_conservative')} | {tri('N4_novelty_aware')} | {tri('H4_quality_first_hybrid')} | {tri('A4_original')} | {tri('A5_oracle')} |"
        )
    print("\n### Worst primary cell per arm (primary v2)\n")
    print("| Arm | worst sFNMR (cell) | worst sFMR (cell) | min coverage |")
    print("|---|---|---|---|")
    prim = [k for k in c if not k.endswith("|test_id")]
    for arm in ARMS:
        fn = max(
            prim,
            key=lambda k, arm=arm: (
                (c[k]["arms"][f"{arm}|primary|v2"]["selective_fnmr"] or {}).get("value") or 0
            ),
        )
        fm = max(
            prim,
            key=lambda k, arm=arm: (
                (c[k]["arms"][f"{arm}|primary|v2"]["selective_fmr"] or {}).get("value") or 0
            ),
        )
        cov = min(c[k]["arms"][f"{arm}|primary|v2"]["coverage"] for k in prim)
        print(
            f"| {arm} | {100 * (c[fn]['arms'][f'{arm}|primary|v2']['selective_fnmr'] or {}).get('value', 0):.2f} ({fn}) | {100 * (c[fm]['arms'][f'{arm}|primary|v2']['selective_fmr'] or {}).get('value', 0):.2f} ({fm}) | {100 * cov:.1f} |"
        )
    print("\n### Novelty detector\n")
    print(
        "| Fold | ID pair flag rate (Test-ID) | held-out pair flag | E1 pair flag | AUROC held-out vs seen renders | AUROC E1 vs seen | AUROC held-out vs clean | AUROC E1 vs clean | clean pair flag |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    nv = d["novelty"]
    for f in FOLDS:
        n = nv[f]
        print(
            f"| {SHORT[f]} | {100 * c[f'{f}|test_id']['novel_pair_rate']:.1f} % | {100 * n['test_ood']['pair_flag_rate']:.1f} % | {100 * n['e1']['pair_flag_rate']:.1f} % | "
            f"{n['test_ood']['auroc_vs_seen_family_renders']:.3f} | {n['e1']['auroc_vs_seen_family_renders']:.3f} | {n['test_ood']['auroc_vs_clean_images']:.3f} | {n['e1']['auroc_vs_clean_images']:.3f} | {100 * d['clean_margin'][f]['clean_pair_novelty_flag_rate']:.1f} % |"
        )
    print(
        "\n### Visibility overestimation on p4 renders (signed global bias / share overestimated by >0.10)\n"
    )
    print(
        "| Fold | Cell | Family | Phase 3 VE bias | VE-C bias | Phase 3 VE share >0.10 | VE-C share >0.10 |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, fams in sorted(d["visibility_bias"].items()):
        for fam, b in fams.items():
            print(
                f"| {k.split('|')[0][:2]} | {k.split('|')[1]} | {fam} | {b['bias_ve_phase3']:+.3f} | {b['bias_vec']:+.3f} | {100 * b['share_over_0.10_ve_phase3']:.1f} % | {100 * b['share_over_0.10_vec']:.1f} % |"
            )
    print("\n### Clean margin (FNMR at frozen clean-cal FAR-1 % thresholds vs M0)\n")
    print(
        "| Fold | M0 FNMR | "
        + " | ".join(f"{m} FNMR (Δ CI hi, ok)" for m in ("M1", "M4c", "M5c", "MLP", "H4"))
        + " |"
    )
    print("|---|---|" + "---|" * 5)
    for f in FOLDS:
        cm = d["clean_margin"][f]
        row = [
            f"{100 * cm[m]['fnmr']:.2f} ({100 * cm[m]['delta_fnmr_vs_M0']['ci_high']:+.2f}, {'yes' if cm[m]['non_inferior_at_margin'] else 'no'})"
            for m in ("M1", "M4c", "M5c", "MLP", "H4")
        ]
        print(f"| {SHORT[f]} | {100 * cm['fnmr_M0']:.2f} | " + " | ".join(row) + " |")
    print(
        "\n### Fusion and co-visibility ablations on occluded pairs (Δ log-loss, positive = second model better)\n"
    )
    print(
        "| Cell | M1 − M4c [CI] | M5_noCV − M5c [CI] | M5c − MLP [CI] | AUC M1 / M4c / M5c / MLP |"
    )
    print("|---|---|---|---|---|")
    for k in sorted(c):
        ab, md = c[k]["ablations"], c[k]["models"]

        def ci(x: dict[str, Any]) -> str:
            return f"{x['estimate']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]"

        aucs = " / ".join(
            f"{md[m].get('auc_occluded', md[m]['auc']):.4f}" for m in ("M1", "M4c", "M5c", "MLP")
        )
        print(
            f"| {k} | {ci(ab['M1_minus_M4c'])} | {ci(ab['M5_noCV_minus_M5c'])} | {ci(ab['M5c_minus_MLP'])} | {aucs} |"
        )
    print()


def detector_section(dev: dict[str, Any] | None, v2: dict[str, Any] | None) -> None:
    for name, d in (("DEVELOPMENT (LFW View 1 DevTest)", dev), ("FOLLOW-UP (LFW View 2)", v2)):
        if not d:
            continue
        r = d["results"]
        print(f"## Detector-inclusive, {name}: run `{d['run_id']}`\n")
        print(
            f"max residual (p99 clean DevTrain) = {r['max_residual_p99_devtrain_clean']:.2f}; selection: `{r['selection']['selected']}`"
            + (
                f"; dev-selected policy: `{d.get('dev_selected_policy')}`"
                if d.get("dev_selected_policy")
                else ""
            )
            + "\n"
        )
        print(
            "| Family | no face (low/med/high) | low quality (l/m/h) | multiple faces (l/m/h) | "
            + " | ".join(f"{p}: cov / sFNMR / sFMR" for p in ("abstain", "non_match", "fallback"))
            + " |"
        )
        print("|---|---|---|---|---|---|---|")
        for f in FOLDS:
            t = r["taxonomy"][f]

            def lv(cat: str, t: dict[str, Any] = t) -> str:
                return "/".join(str(t[x][cat]) for x in ("low", "medium", "high"))

            pol = r["policies"][f]
            row = [
                f"{100 * pol[p]['coverage_genuine']:.1f} / {100 * pol[p]['sfnmr']:.2f} / {100 * pol[p]['sfmr']:.2f}"
                for p in ("abstain", "non_match", "fallback")
            ]
            print(
                f"| {SHORT[f]} | {lv('no_face')} | {lv('low_quality')} | {lv('multiple_faces')} | "
                + " | ".join(row)
                + " |"
            )
        print()


def main() -> int:
    d = load("phase4_final")
    if d:
        final_section(d)
    fu = load("phase4_followup_p3test")
    if fu:
        print(f"## Follow-up on p3_test `{fu['run_id']}` (descriptive)\n")
        arms_table(fu["cells"], "test_ood", "primary|v2")
        print(
            f"Endpoints (Test-OOD only, descriptive): recommendation `{fu['endpoints_descriptive_test_ood_only']['recommendation']}`; violations {fu['endpoints_descriptive_test_ood_only']['violations_primary_v2']}\n"
        )
    detector_section(load("phase4_detector_dev"), load("phase4_detector_view2"))
    cost = load("phase4_cost")
    if cost:
        print("## Computational cost (ms per image, batch 256)\n")
        print("| Component | ms / image | peak GPU MiB |")
        print("|---|---|---|")
        for k, v in cost["results"].items():
            print(f"| {k} | {v['ms_per_image']:.3f} | {v.get('peak_gpu_mib', 0):.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
