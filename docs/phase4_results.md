# Phase 4 Results: Robustness Recovery and Conservative Visibility

Protocol: `docs/phase4_protocol.md` (v0.6.0, frozen before any Phase 4 training; D-048).
Implementation corrections made before any p4_test access: D-049. Development freeze:
`bba12900…`. Tables: `reports/experiments/phase4_tables.md` (`scripts/summarize_phase4.py`).

All faces in the confirmatory set are synthetic (DigiFace), and all occluders are synthetic,
including on LFW, which has real faces. **Nothing here is evidence about physical occlusions.**

## 0. Summary

*Pending the single confirmatory p4_test run.*

## 1. Root cause of visibility overestimation (4A; development data only)

Source: `reports/experiments/phase4_diagnostics.json`, from p3_val identities with fresh renders.

| Finding | Evidence |
|---|---|
| Reproduced | Fold-F4 estimator, random patch, high severity: IoU 0.12, global bias +0.54; 88 % of truly occluded face cells predicted visible |
| Errors are confident | On those cells: mean P(visible) 0.85; 67 % above 0.9; mean \|logit\| 5.4. A probability or entropy threshold cannot flag them |
| Not a location prior | Miss rate 88.3 % in the zones where seen occluders sit vs 88.7 % elsewhere |
| Not colour or texture | Miss rate 0.87–0.96 across dark / bright / saturated / mid / flat / textured patches |
| Learned occluder classes, not "non-face" | 24×24 probes: the F4-fold model calls *any* uniform fill visible (0.90–1.00), while the four models that saw patches during training flag the same fills (0.01–0.3) |
| Face-like occluders are undetectable | Every model calls another identity's face patch visible (0.35–1.00) |
| Preprocessing is fine | Identical code path; F4 inputs only mildly more saturated (4.0 % vs 2.1 % of pixels) |
| Lighting is a corruption | F5 "visibility" weakly predicts recognition damage: genuine-score drop correlation −0.31 and mean drop 0.039 at mean labelled visibility 0.52, vs −0.43 to −0.70 and 0.137 (mask, 0.50) for physical occluders |

Phase 4 design follows directly from these findings, so F4/F5 are **design-informed** and a new
evaluation-only occluder (`E1_hair_strands`) is the clean unseen test.

## 2. Data and access audit

- **p4_test.** DigiFace 72-image parts 2000–3999 and 4000–5999, never downloaded before Phase 4
  (owner approval).
  - 288,000 images and 4,000 identities. Integrity checks: 0 decode failures, 0 exact duplicates
    within the new parts and 0 against every existing DigiFace image.
  - 48,000 pairs (24,000 genuine / 24,000 impostor).
  - LOFO test_id / test_ood plans per fold, plus an E1 plan. Seed 20261101.
  - Used **once** for decisions.
- **Re-used, follow-up only:** p3_test (Phase 3 confirmatory); LFW View 2.
- **Development:** p3_train / val / cal and LFW View 1 DevTest.
- **Access log:** `results/test_runs.jsonl`.

## 3. Methods actually trained (4B–4E)

- **VE-C** (5 folds; `reports/experiments/phase4_vec_training_<fold>.json`):
  - asymmetric BCE (occluded cells weighted 4×);
  - 20 % natural-texture blob hard negatives;
  - lighting targets mark only information-free pixels;
  - otherwise the Phase 3 budget.
  - F1 and F4 stopped early (epochs 16 and 10). F2, F3 and F5 reached the 30-epoch cap.
- **Calibration** (D-049):
  - the temperature fit hit the lower grid edge (T = 0.5) in every fold: VE-C is *under*-confident
    on seen physical families;
  - the exact clipped lower-bound shift is c = 0 in every fold: VE-C needs no extra shift on
    seen families;
  - sanity check (protocol §7) passed: on p3_val seen physical families, 0.0 % of renders are
    overestimated by > 0.10, vs 0.0–0.13 % for the Phase 3 VE;
  - a first develop pass that included lighting renders in this calibration was stopped and
    corrected before any test access (D-049).
- **Novelty.** Mahalanobis distance on the VE-C bottleneck; pair threshold at the 95th
  percentile of p3_val in-distribution pairs (9.5–11.1).
- **Fusion.** The MLP improved p3_val log-loss over logistic M5c by 3.1 % (F1), 6.1 %, 8.5 %,
  8.4 % and 5.4 %, so the registered rule makes MLP the deployable fused score in F2–F5.
- **Arms.** B1, C4, N4 and H4 were calibrated on p3_cal; every stratum was certified for the
  primary target with v2 ordering.

## 4–9. Confirmatory results

*Pending.*

## 10. Detector-inclusive robustness (4F)

Taxonomy:
- `no_face`;
- `low_quality` (detector score < 0.6 or 5-point residual > 7.34 px, the p99 of clean View-1
  DevTrain);
- `multiple_faces`;
- `success`.

Occluders were composited into the original photographs (300 pairs × 5 families × 3 levels,
side b). Decisions use the frozen Phase 3 A3 rule of each family's held-out fold.

| | Development: LFW View 1 DevTest (`1c792ba98f5f`) | Follow-up: LFW View 2 (`d57db3ee8923`, logged) |
|---|---|---|
| Random patch, high severity: no face / low quality (of 300) | 62 / 35 | 52 / 43 |
| Lighting, high: no face / low quality | 24 / 14 | 23 / 7 |
| Multiple faces (all composites) | 13.4 % (LFW photos often contain several people) | 15.3 % |
| Worst-family sFNMR: abstain / non-match / **fallback** | 4.21 / **16.13** / 4.09 % | 2.91 / **14.65** / 3.16 % |
| Mean genuine coverage: abstain / non-match / fallback | 93.9 / 98.2 / 97.2 % | 94.4 / 98.7 / 97.5 % |
| sFMR (all policies, all families) | 0.00 % | 0.00 % |

- **Registered selection on DevTest: fallback** (detector-free fixed crop + A3). Its worst-family
  sFNMR is lowest among feasible policies.
- **Forced non-match is unsafe:** it pushes random-patch sFNMR to 15–16 %.
- On View 2 (follow-up) the dev-selected fallback stays within target in every family.
- Failed detections are counted in every rate; none is dropped.

## 11. Physical-occlusion readiness (4H)

`docs/physical_occlusion_readiness.md` §6. **IJB-C** is the best candidate: NIST access request,
real occlusions with annotated locations, standard verification protocols. RMFRD (masks) has no
stated licence. MFR2 is too small. CelebA has only mild occluders, labelled by attribute. ROF
still requires separate approval (pickles; licence conflict).

## 12. Tests, CI and commits

*Pending (final quality gate).*
