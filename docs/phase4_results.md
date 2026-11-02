# Phase 4 Results: Robustness Recovery and Conservative Visibility

Protocol: `docs/phase4_protocol.md` (v0.6.0, frozen before any Phase 4 training; D-048).
Implementation corrections made before any p4_test access: D-049. Development freeze:
`bba12900…`. Tables: `reports/experiments/phase4_tables.md` (`scripts/summarize_phase4.py`).

All faces in the confirmatory set are synthetic (DigiFace), and all occluders are synthetic,
including on LFW, which has real faces. **Nothing here is evidence about physical occlusions.**

## 0. Summary

**Registered outcome: recommend A3 (quality-stratified calibration).** No Phase 4 candidate
reliably improves on it. The negative visibility result stands, and is now explained.

| Question | Result |
|---|---|
| **Q1 safety** (candidate no worse than A3 on 10 primary cells: Test-OOD ×5 + never-seen E1 ×5) | **All three candidates fail.** C4 conservative: 10 violations vs A3's 6, and significantly worse sFNMR in 8 cells. N4 novelty-aware: 9 violations, 9 cells worse. H4 quality-first hybrid: 9 violations, 6 cells worse, and coverage collapses on E1 (mean 49.6 %) |
| **Q2 benefit** | Not reached (no candidate is safe). C4 adds +3.6 pp coverage, but only by accepting riskier pairs |
| **Root cause of overestimation** | The Phase 3 estimator learned *occluder classes*, not "non-face". Its errors are confident; face-like occluders are undetectable; lighting labels overstate information loss |
| **Conservative VE-C** | It **fixes the overestimation**. Held-out random patch: share overestimated by > 0.10 drops from 94.6 % to 21.9 %, bias from +0.34 to +0.07. On the never-seen hair occluder it errs conservative (negative bias) in 4/5 folds. On the calibration-matched p3_test follow-up, C4 removes the Phase 3 random-patch violation (3.64 % vs 8.52 % ✗). Better visibility still does **not** beat quality: even the ground-truth oracle is worse than A3 on every E1 cell |
| **Novelty detection** | E1 AUROC 0.91–0.98 vs seen-family renders and ≈ 1.0 vs clean. Realised ID pair FPR 4.6–5.2 % (registered 5 %) |
| **Fusion** | The MLP beats logistic M5c in log-loss in 9/10 cells. Co-visibility **harms** log-loss on all E1 cells and on 3/5 Test-OOD cells, so it is dropped from the deployable configuration |
| **Clean margin** | Every fused score meets the 0.5 pp margin (it is better than raw similarity) on p4 and on p3_test |
| **Detector** | Failures concentrate in random patch (high severity: up to 32 % no face / low quality). A detector-free fixed-crop **fallback** is the registered policy; forced non-match is unsafe (15–16 % sFNMR) |
| **Unexpected: population shift** | p4_test (72-image identities only) has ~4× harder genuine pairs: clean FNMR(M0) 4.81 % vs 1.12 % on p3_test. **Every arm, including A3 and the oracle, exceeds the 5 % sFNMR target even on seen families** (Test-ID 5.4–7.8 %). The p3_cal calibration does not transfer to this sub-population. That is a validity finding about the guarantees, not about occlusion |

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

## 4. Confirmatory run `903f6a5c7de3` (p4_test, used once; frozen `bba12900…`)

Full tables: `reports/experiments/phase4_tables.md`. Cells are mean over 5 folds, primary target,
v2 ordering. sFNMR / sFMR in %.

| Arm | Test-OOD coverage | Test-OOD sFNMR (mean / worst) | E1 coverage | E1 sFNMR (mean / worst) | Test-ID sFNMR (mean / worst) | worst sFMR (primary cells) | violations (20 cell × error) |
|---|---|---|---|---|---|---|---|
| **A3 quality (reference)** | 91.9 | **6.14 / 7.55** | 91.3 | **5.12 / 5.76** | 5.96 / 6.34 | 2.59 | **6** |
| C4 conservative | 93.5 | 7.64 / 10.53 | 96.9 | 8.77 / 9.62 | 6.56 / 6.95 | 2.57 | 10 |
| N4 novelty-aware | 93.2 | 7.31 / 9.70 | 92.4 | 6.92 / 7.84 | 6.45 / 7.32 | 2.57 | 9 |
| H4 quality-first hybrid | 82.1 | 6.75 / 7.48 | **49.6** | 7.44 / 9.33 | 6.40 / 6.83 | 2.58 | 9 |
| A4 original (Phase 3) | 95.9 | 9.25 / **17.73** | 98.5 | 9.51 / 12.14 | 6.54 / 6.86 | 3.30 | 10 |
| A5 oracle (not deployable) | 92.7 | 6.60 / 9.10 | 98.0 | 9.12 / 10.32 | 6.66 / 7.76 | 2.64 | 9 |
| B0 raw global | — | worst 18.96 | — | worst 8.93 | — | 4.83 | 7 |
| B1 quality global | — | worst 12.78 | — | worst 8.70 | — | 5.51 | 10 |

**P1 (paired identity jackknife vs A3, Holm over 10 cells).**
- C4 is significantly worse in sFNMR in 8 cells and in sFMR in 4 E1 cells.
- N4 is worse in sFNMR in 9 cells.
- H4 is worse in sFNMR in 6 cells and in sFMR in 3.
- **None is safe**, so P2 does not apply.

**Recommendation rule → A3.**

**Mechanism.** The visibility arms accept more pairs (C4 +3.6 pp, N4 +1.2 pp) at higher error.
The hybrid's fallback to quality strata was calibrated jointly on p3_cal, where few pairs are
novel (≈ 5 %). Its unreliable-pair strata got very strict rules, so on E1 (21–88 % of pairs
flagged novel) coverage collapses, yet sFNMR is still worse than A3. Fallback is not safe by
construction, as the protocol anticipated.

## 5. Conservative visibility and novelty

**VE-C vs the Phase 3 VE on p4 renders** (signed global bias; share of renders overestimated by
> 0.10). F4 and F5 are design-informed; F5 is measured against the old darkening-based GT, so
it is not comparable by design.

| Held-out family (fold) | Phase 3 VE bias / share | VE-C bias / share |
|---|---|---|
| random patch (F4) | +0.342 / 94.6 % | **+0.065 / 21.9 %** |
| hand/object (F3) | +0.012 / 4.2 % | −0.006 / 0.5 % |
| E1 hair, never seen (F4 model) | +0.111 / 45.0 % | **−0.051 / 7.4 %** |
| E1 hair (F5 model) | +0.046 / 20.3 % | +0.013 / 10.8 % |
| E1 hair (F3 model) | −0.030 / 10.2 % | −0.056 / 11.7 % |
| E1 hair (F1 / F2 models) | −0.051 / 0.0 %; −0.078 / 0.0 % | −0.033 / 4.3 %; −0.051 / 2.1 % |

**Novelty** (Mahalanobis on VE-C bottleneck; image-level AUROC).

| Comparison | AUROC |
|---|---|
| E1 vs seen-family renders | 0.91–0.98 |
| E1 vs clean | ≈ 1.0 |
| Held-out family vs seen renders | 0.62–0.85 |
| Lighting vs seen renders | 0.34, i.e. **not** novel, consistent with lighting-as-corruption |

Realised pair flag rates:
- Test-ID 4.6–5.2 % (registered 5 %);
- E1 21–88 %;
- held-out random patch 46.5 %;
- clean pairs ≤ 0.1 %.

## 6. A3 vs the new methods at comparable coverage

The new arms sit at similar or higher coverage than A3 (C4/N4: +1 to +4 pp), with higher sFNMR.
At equal accepted-pair counts they are therefore riskier, not more efficient. The hybrid's
lower-risk cells come at much lower coverage (E1 49.6 % vs 91.3 %). The underlying fused scores
rank errors similarly (occluded-pair AUC M1 / M4c / M5c / MLP within 0.003 in most cells; see
the tables). The difference lies in where the strata place the thresholds, not in
discrimination.

## 7. Worst-family behaviour and false matches

- **Worst primary-cell sFNMR:** A3 7.55 % (mask OOD) vs C4 10.53 %, N4 9.70 % (random patch),
  H4 9.33 % (hand/object E1), A4 17.73 % (random patch).
- **sFMR** stays ≤ 2.6 % for A3 and all candidates. False matches are not the failure mode on
  DigiFace (B1 reaches 5.51 % on mask OOD).

## 8. Clean performance and the population shift

- **Clean margin.** At their frozen clean-cal FAR-1 % thresholds, FNMR on p4 clean pairs:
  - M0 (raw similarity) 4.81 %;
  - M1 3.37–3.41 %, M4c 3.46–3.73 %, M5c 3.14–3.47 %, MLP 2.95–3.15 % (upper CI ≤ −0.93 pp).
  - Every model meets the 0.5 pp margin by being *better*. On p3_test: M0 1.12 %, MLP
    0.54–0.61 %.
- **Shift.** p4_test is drawn only from 72-image identities, rendered with varied accessories,
  pose and expression, with up to 6 genuine pairs each. p3_cal is drawn mostly from 5-image
  identities.
  - Clean FNMR(M0) is 4.3× higher on p4 than on p3_test, and every arm violates sFNMR even on
    Test-ID.
  - The LTT certificates hold for the calibration distribution only. **This partition
    demonstrates that they do not transfer across DigiFace sub-populations.**
  - Comparisons between arms remain valid: they are paired on identical pairs and identities.

## 9. Follow-up on p3_test (`db421cf1744d`; same population as calibration; descriptive)

Held-out random patch, primary v2:

| Arm | coverage / sFNMR |
|---|---|
| A4 original | 98.5 % / **8.52 % ✗** |
| **C4** | 91.3 % / 3.64 % |
| **N4** | 90.1 % / 3.14 % |
| A3 | 88.6 % / 2.12 % |
| H4 | 52.2 % / 2.16 % |

- No Test-OOD violations for A3, C4, N4, H4 or A5.
- Even so, C4 and N4 are significantly worse than A3 in sFNMR on 3 cells (hand/object, random
  patch, lighting) for about +1 pp coverage.
- **Conclusion:** conservative visibility repairs the Phase 3 failure but does not beat quality
  stratification.

## 9a. Fusion and co-visibility (Q3)

Δ log-loss on occluded p4 pairs, paired identity jackknife:

| Comparison | Result |
|---|---|
| MLP vs logistic M5c | **MLP better in 9/10 primary cells** (sunglasses OOD not significant), e.g. +0.0108 random-patch OOD, +0.0277 lighting OOD |
| Co-visibility, M1 − M4c | Positive only for mask / sunglasses Test-OOD (+0.009 to +0.010). **Negative in all 5 E1 cells** (−0.009 to −0.039) and in hand / random-patch / lighting OOD |
| Co-visibility, M5_noCV − M5c | Same pattern. **Removing co-visibility improves log-loss in 8/10 cells** |

Decision: co-visibility is removed from the recommended deployable configuration and kept as a
documented research ablation.

## 9b. Computational cost (`reports/experiments/phase4_cost.json`; RTX 4060 Laptop, batch 256)

| Component | ms / image |
|---|---|
| AdaFace IR-101 | 2.01 |
| ArcFace IR-101 | 1.85 |
| Visibility estimator (VE or VE-C) | 0.22 (0.44 with flip TTA) |
| Mahalanobis novelty (CPU) | 0.07 |

Visibility adds about 11 % to a single-backbone pipeline; dual-backbone fusion doubles the
backbone cost.

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

## 12. Statistical validity and limitations

- **One confirmatory look.** p4_test was used once (`903f6a5c7de3`).
- **Follow-up sets** (p3_test `db421cf1744d`, LFW View 2 `d57db3ee8923`) are labelled and
  descriptive.
- **Development diagnostics.** Their use of held-out-family renders on p3_val informed the
  design, so **F4/F5 are design-informed**. E1 is the only never-seen occluder, and on E1 every
  candidate is worse than A3.
- **Intervals.** Delete-one-identity jackknife (paired for differences; SIM-011 coverage
  0.94–0.95). Holm over 10 cells per candidate and error type.
- **No OOD guarantee.**
  - LTT validity is limited to the calibration distribution.
  - The p4 population shift shows this concretely: **all** arms, A3 included, exceed the target
    on seen families.
  - Any deployment must calibrate on data drawn from the deployment population.
- **Implementation corrections before test access** (D-049): lighting excluded from VE-C
  calibration; exact clipped shift. The VE-C temperature hit its grid edge (0.5) in all folds.
  F2/F3/F5 VE-C reached the 30-epoch cap.
- **Synthetic occluders and DigiFace faces only.** LFW adds real faces with synthetic overlays.
  Physical occlusions remain untested.
- **Single detector (SCRFD) and 300 pairs** in the detector study (exploratory scale).

## 13. Recommended deployable architecture (research recommendation; not a product)

> **Erratum (Phase 6, D-052).** Items 2–3 below mix a *validated* component with an
> *unvalidated* one:
> - **Validated:** the A3 decision system evaluated in Phases 3–4 uses **M1 only**, built from
>   the AdaFace cosine and the AdaFace feature norm.
> - **Unvalidated:** ArcFace and the "dual-backbone + quality" fused score were compared only in
>   log-loss and AUC. They were **never calibrated or evaluated as a three-way decision system**.
>
> The validated configuration is therefore: SCRFD (+ fallback) → AdaFace → M1 → A3. A
> dual-backbone fused score remains a hypothesis for future, prospectively evaluated work. The
> per-pair cost "~8 ms with both backbones" below is batch throughput, not end-to-end latency
> (see `docs/phase5_results.md` §6).

1. **Detection:** SCRFD. On no face / low-quality alignment, fall back to a **detector-free fixed
   crop**, scored by the same quality-stratified rule. Never force non-match; never silently
   drop.
2. **Recognition:** frozen AdaFace IR-101 (primary) + ArcFace IR-101. Quality = feature norm.
3. **Fused score without co-visibility:** dual-backbone + quality features. The MLP is better
   than logistic fusion in log-loss, but **evaluate it prospectively without co-visibility
   features before adoption** (the tested MLP used M5c inputs).
4. **Decision rule: A3 quality-stratified three-way LTT**, calibrated on identity-disjoint data
   **from the deployment population**.
5. **Visibility estimation (VE-C + novelty) is not used for decisions.** It may be shown as an
   *explanation* overlay: "possible occlusion here / unknown occluder" are useful
   human-readable signals.

## 14. Phase 5 readiness

- **Ready:** a frozen, documented A3 pipeline with the detector fallback, explainability
  signals (VE-C maps, novelty flag, quality) and measured costs (~3.9 ms per image with both
  backbones, so ~8 ms per pair on GPU).
- **Not ready:**
  - any calibrated-risk claim for a population other than the calibration one;
  - physical occlusions (IJB-C access needed);
  - real-photo calibration.
- **A Phase 5 demo must display decisions as research output, with no guarantee claims.**
