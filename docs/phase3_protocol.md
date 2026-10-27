# Phase 3 Prospective Protocol (pre-registration, frozen before any Phase 3 training)

> Registered 2026-10-27 at protocol v0.5.0, before any visibility estimator, fusion model or
> Phase 3 calibration is trained or fitted, and before any Phase 3 data is scored. Changes after
> this point are amendments: logged in `docs/decision_log.md` with the reason and labelled in
> every result they touch.

## 1. Questions and hypotheses (negative results will be reported)

- **C3 (primary).** Does **learned** visibility, used for visibility-stratified, occlusion-aware LTT
  calibration of a visibility-aware fused score, reduce selective risk-control failures on
  **unseen occlusion families**? Comparators: strong quality-only calibration, and the
  ground-truth-visibility **oracle** as an upper bound.
- **C1 (secondary).** Does pairwise co-visibility add information about verification beyond the
  backbone similarity and quality features?
- **H4 (constraint).** Clean-pair performance must not degrade by more than the provisional
  0.5 pp margin.

## 2. Data and evidence status

| Data | Status | Use in Phase 3 | Evidence label |
|---|---|---|---|
| DigiFace **train**-split identities (61,067). **Verified: no image from them was ever scored by any model in Phases 1–2** | Untouched | Phase 3 sub-splits (below) | p3_test = **confirmatory** |
| DigiFace Phase 2 val/cal/test splits | Observed in Phase 2 | Re-analysis with the Phase 3 models | **follow-up** |
| LFW View 2 (Kaggle v4) and its derived occlusions | Observed in Phase 2 | Phase 3 models applied, no training on it | **follow-up** |
| LFW View 1 DevTrain | Development | Only for in-domain thresholds already defined in Phase 2 | development |

**Phase 3 sub-splits.** These are bands of `sha256("veilx-p3:" + identity)[:8] mod 100` over
train-split identities only:

| Sub-split | Band | Purpose |
|---|---|---|
| `p3_train` | [0, 50) | visibility-estimator training images; fusion training pairs |
| `p3_val` | [50, 60) | early stopping, model selection, LTT candidate orders, strata edges |
| `p3_cal` | [60, 75) | identity-disjoint LTT calibration pairs (independent) |
| `p3_test` | [75, 100) | confirmatory evaluation, used **once** |

**Pair sets** (seed 20261027, generator unchanged):
- `p3_train_pairs`: clustered, 20,000 genuine + 20,000 impostor, genuine cap 1 per identity.
- `p3_val_pairs`: clustered, genuine cap 2, balanced.
- `p3_cal_independent`: identity-disjoint, one third of the cal identities per class.
- `p3_test_pairs`: clustered, genuine cap 2, balanced.

**Occlusion plans.** Leave-one-family-out (LOFO), using the same generator as Phase 2 with global
seed 20261027:

| Pair set | Role(s) |
|---|---|
| `p3_train_pairs` | train |
| `p3_val_pairs` | val |
| `p3_cal_independent` | cal |
| `p3_test_pairs` | test_id **and** test_ood |

Held-out-family isolation is enforced by the existing guards.

## 3. Visibility estimator (VE)

- **Input / output.** Input is an aligned 112×112 RGB crop (clean or occluded). Output is a 28×28
  visibility logit map, P(cell visible). The ground-truth mask is never an input.
- **Target.** 1 − occlusion mask, average-pooled to 28×28 (soft targets). For F5 lighting the mask
  marks destroyed pixels.
- **Derived outputs.**
  - Global visible-face fraction: the mean of the predicted map over face-ellipse cells.
  - Region estimates: mean over five template regions (left eye, right eye, nose, mouth, lower
    face/jaw), defined geometrically from the ArcFace template.
  - Reliability: the mean binary entropy of the map.
- **Architecture.** Lightweight encoder–decoder (U-Net-lite, 4 stages 32→256 channels, roughly 2 M
  parameters), trained from scratch with no pretrained weights.
- **Training.**
  - Samples: `p3_train` images, re-rendered on the fly. 25 % clean, 75 % occluded with one
    **seen** family (the held-out family is excluded), uniform level, random seed.
  - Loss: BCE on soft targets, face-ellipse cells weighted 1.0, others 0.25.
  - Optimiser: AdamW (lr 1e-3, wd 1e-4, cosine), batch 128, AMP fp16 (validated against fp32 on
    val).
  - Schedule: 50,000 samples per epoch, at most 30 epochs, **early stopping on val loss**
    (patience 4) over a fixed 10,000-render `p3_val` set (seen families).
  - Seed 20261027 + fold index; resumable checkpoints.
- **Fold models.** Five, one per held-out family. Each is evaluated on its held-out family (OOD)
  and its seen families (ID).
- **Comparators.**
  - (1) Heuristic: per-cell appearance anomaly against the mean clean `p3_train` face, mapped to a
    probability by a 2-parameter logistic fitted on `p3_val`.
  - (2) Learned VE.
  - (3) **Oracle**: the ground-truth mask. It is a reference only and **never deployable**.
- **Metrics** (defined in `veilx.vis.metrics`). Cells count as occluded when target visibility
  < 0.5 and as predicted occluded when P(visible) < 0.5.
  - IoU and Dice of occluded cells, per image (images with occlusion), mean.
  - Brier score and 10-bin ECE of cell probabilities.
  - Global-visibility MAE against the engine's `visibility_ratio`.
  - Per-region MAE.
  - All reported per family × (ID vs held-out) on `p3_test` renders; follow-up on LFW renders.

## 4. Pairwise co-visibility (from predictions only)

**Definitions** (all computed for heuristic, learned and oracle maps):

| Feature | Definition |
|---|---|
| `cv_min` | min(vis_a, vis_b) |
| `cv_prod` | vis_a · vis_b |
| `cv_spatial` | Σ_face min(pa, pb) / Σ_face 1, the expected commonly visible face fraction ("comparable evidence") |
| `cv_region` | mean over regions of min(ra, rb) |
| `asym` | \|vis_a − vis_b\| |

**Selection rule (fixed now).** The primary co-visibility definition is the one with the highest
fit-free AUROC for predicting errors (decision errors of the frozen AdaFace cosine at its Phase 2
`cal_far_1pct` threshold) on **`p3_val`** occluded pairs. The others are reported, never selected
on test.

## 5. Fusion ladder (regularised logistic regression; AdaFace primary)

Features are standardised on `p3_train`. `s` is the cosine; `q` = min(feature norm a, b). Every
model includes an intercept.

| Model | Features |
|---|---|
| M0 | s (AdaFace); M0-arc: s (ArcFace) |
| M1 quality | s, q, s·q |
| M2 visibility | s, vmin, vmax, s·vmin |
| M3 co-visibility | s, cv, asym, s·cv |
| **M4 quality + co-visibility** | s, q, s·q, cv, asym, s·cv |
| M5 full | M4 + ArcFace s, q, s·q + disagreement \|s_ada − s_arc\| |
| M4-oracle | M4 with ground-truth cv (upper bound) |
| M4-heur | M4 with heuristic cv |

- **Training.** Per fold, on `p3_train` pairs under the fold's train plan (seen families + clean).
- **Regularisation.** L2, with C ∈ {0.01, 0.1, 1, 10} chosen by `p3_val` log-loss.
- **Neural fusion.** Considered only if a 2-layer MLP (64 hidden) improves `p3_val` log-loss of M5
  by > 5 % relative. Otherwise it is not used (decision recorded).

## 6. Selective decisions and calibration arms

**Decisions.**
- MATCH if p ≥ τ_high and NON-MATCH if p ≤ τ_low, otherwise ABSTAIN.
- Targets: `primary` (sFMR ≤ 5 %, sFNMR ≤ 5 %) and `far1` (1 % / 5 %); δ = 0.05.
- Candidate grids from `p3_val` score quantiles.
- **Ordering:** v2 (`order_candidates_by_validation_pvalue`) is the development default; v1 is kept
  as a comparison. v2 was introduced after Phase 2 calibration observations (D-042); its
  simulated validity (SIM-012) is **not** a proof under shift.

**Arms.** All calibrated on `p3_cal_independent` under the fold's cal plan unless stated.

| Arm | Score | Strata (edges fixed on `p3_val`) |
|---|---|---|
| A1 clean-only | M0 | — (clean cal pairs) |
| A2 occlusion-aware global | M0 and M4 | — |
| A3 quality-stratified | M1 | q tertiles |
| **A4 predicted-visibility-stratified** | M4 | predicted cv tertiles |
| A5 oracle-visibility-stratified | M4-oracle | ground-truth clean / high / low as in Phase 2 |
| A6 quality × predicted visibility | M4 | 2 × 2 by medians (if capacity allows; infeasible strata abstain) |

**Abstention reasons.** Recorded per pair, descriptive and **not causal**: insufficient predicted
co-visibility (stratum `vis_low`), poor quality (stratum `q_low`), conflicting models (\|s_ada −
s_arc\| above the `p3_val` 95th percentile), uncertain score (inside [τ_low, τ_high]), and
uncertifiable stratum (no rule certified).

## 7. Endpoints (on `p3_test`, used once; Holm within each family)

- **C3 primary (risk).** Per fold, on Test-OOD:
  - realised sFNMR and sFMR of **A4** versus **A3** (v2, primary targets);
  - the paired difference A4 − A3 of sFNMR, with a delete-one-identity jackknife interval;
  - Holm over the 5 folds.
- **C3 primary (cost).** Coverage difference A4 − A3 on Test-OOD, averaged over folds.
- **C3 verdict rule.** "Learned visibility improves risk control" requires all three of:
  - (a) fewer Test-OOD violations (interval above target) for A4 than A3, summed over folds;
  - (b) no fold where A4 is significantly worse than A3 on sFNMR (Holm);
  - (c) mean coverage loss A4 − A3 ≥ −5 pp.

  Otherwise it is reported as not supported, or as mixed.
- **C1 primary.** The paired difference in mean binary log-loss, M1 − M4, on `p3_test` occluded
  pairs.
  - Per fold × {Test-ID, Test-OOD}; 10 tests, delete-one-identity jackknife, Holm.
  - C1 is supported if the difference is > 0 with a Holm-adjusted interval excluding 0 in at least
    6 of 10 cells and in the random-patch OOD cell.
- **H4 clean margin.**
  - On clean `p3_test` pairs, compare the FNMR of M4 and of M0 at their own thresholds fixed on
    clean `p3_cal` pairs for FAR 1 %.
  - The paired ΔFNMR (M4 − M0) has a jackknife interval (validated by SIM-011).
  - Non-inferiority holds if the upper bound is < 0.5 pp.
- **Secondary.**
  - A1, A2, A5 and A6 arms; v1 against v2; `far1` targets.
  - AUGRC and risk–coverage curves; per-family and per-severity rates; abstention-reason mix.
  - M0-arc and M5; ArcFace analogues.

## 8. Follow-up and exploratory analyses (labelled; no training or selection on them)

**Follow-up.**
- The Phase 2 DigiFace test split under the Phase 2 plans.
- LFW View-2 derived occlusions, with each fold's VE and fusion model applied to its held-out family.

**Exploratory.**
- MLP fusion.
- **Detector-inclusive LFW robustness:** occluders composited into the original photographs, then
  SCRFD re-detection and alignment, on a fixed random subsample of 300 pairs × family × level.
- SIM-013: in-distribution validity of stratified LTT with *noisy predicted* strata.

## 9. Training discipline

- Configurations are versioned in `configs/experiments/phase3/*.yaml` with `authorized: true` and
  `VEILX_PHASE2_AUTHORIZED=1`.
- Checkpoints carry SHA-256 hashes. Early stopping and model selection use `p3_val` only.
- No `p3_test`, Phase 2 test or LFW data enters training or selection.
- Memory-monitored pilots precede large runs. Backbones stay frozen.
