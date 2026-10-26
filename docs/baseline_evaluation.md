# Phase 2 Baseline Evaluation: Frozen Backbones on DigiFace (Clean + Synthetic Occlusion)

> Sections 1–4 are the **analysis plan, written and committed before any test-split access**
> (protocol v0.4.0). Section 5 is filled only from the single logged final run
> (`scripts/evaluate_phase2.py final --final`). Every number in section 5 is MEASURED on
> DigiFace-1M (synthetic faces) with synthetic occlusions. **None is a real-world or LFW
> result.**

## 1. Systems and data

- **Backbones** (frozen, inference only; `docs/model_card_backbones.md`):
  AdaFace IR-101 WebFace4M (primary) and ArcFace IR-101 WebFace4M (secondary). Both use RGB
  112×112 input normalised to [−1, 1], with cosine similarity on L2-normalised 512-d
  embeddings, single view and no flip fusion.
- **Precision:** fp16 autocast for all extraction. On 20,000 validation pairs the minimum cosine
  to fp32 is ≥ 0.9998 and the maximum pair-score difference ≤ 0.003 (`reports/models/model_verification.json`).
- **Data:** DigiFace-1M, four approved parts (protocol v0.4.0). Splits come from the identity
  hash. Pair sets:
  - `val` and `test_balanced` are clustered, with a genuine cap of 3 per identity;
  - `cal_independent` is identity-disjoint;
  - `test_far` holds 100,000 clean impostors.
- **Occlusions:** leave-one-family-out (LOFO) plans over F1–F5. Roles are val, cal, test_id
  (seen families) and test_ood (held-out family). Stress combinations are exploratory only.
- **Detector:** none. DigiFace is pre-aligned (detector-free path). LFW is blocked
  (`docs/real_world_benchmark.md`).

## 2. Fixed before test access (develop stage, val + cal only)

| Quantity | Source |
|---|---|
| EER threshold, best-accuracy threshold | clean `val` |
| Thresholds at FAR 1 % and 0.1 % | clean `val` impostors (0.1 % = estimation-only, D-034) |
| **Deployment threshold** at FAR 1 % | clean `cal_independent` impostors |
| LTT candidate grid | all (τ_low ≤ τ_high) pairs over 61 quantiles of the fold's occluded `val` scores, plus the val EER threshold |
| Candidate order / pruning | `order_candidates_on_validation` on the fold's `val` plan (never cal) |
| Oracle-visibility strata | pair co-visibility = min(side visibility) from the synthetic masks; strata are *clean* (both sides unoccluded), *vis_high* (≥ edge) and *vis_low*, with edge = median co-visibility of occluded val pairs in that fold |
| LTT fits | on the fold's `cal` plan over `cal_independent` (identity-disjoint; assumptions A1–A7) |

**Target settings**, pre-registered, with δ = 0.05 (confidence 0.95, frozen):
- `primary_alpha`: sFMR ≤ 0.05 and sFNMR ≤ 0.05 (provisional accepted-error target);
- `far1_fm`: sFMR ≤ 0.01 and sFNMR ≤ 0.05.

**Arms:**
- `global`: one rule fitted on the occluded cal plan (seen families);
- `stratified_vis`: one rule per oracle-visibility stratum, δ/3 each (Bonferroni);
- `clean_cal`: one rule fitted on clean calibration pairs only.

Oracle visibility is ground truth from the synthetic masks. It is an **upper-bound arm**, not a
deployable estimator; a learned visibility estimator is Phase 3 and is not built here. The
frozen configuration's SHA-256 is recorded, and the final stage refuses to run if it has
changed.

## 2a. Amendment before test access (D-042)

The develop stage (freeze `700cd8e9…`) showed that the pre-registered fixed-sequence order often
stopped at its **first** candidate on calibration data. Examples: the ArcFace clean stratum
of fold F2 at the 5 % targets, and 4 of 10 global `far1_fm` fits. The cause is the
ordering: the first rule tested was the lowest-coverage rule that barely met the zero-error
sample-size minimum, so any calibration fluctuation ended the sequence. This is a statistical
**power** failure, not a validity problem.

A second, validation-only ordering was therefore added. Rules are ranked by the binomial
p-value that their validation counts, scaled to the calibration size, would earn, and pruned
at δ.
- In simulation it keeps the violation rate within δ and certifies at least as often
  (`tests/test_ltt.py`).
- Its arms (`global_v2`, `stratified_vis_v2`, `clean_cal_v2`) are reported **in addition to**
  the unchanged pre-registered arms. The rerun freeze `0e8a43fb…` contains the v1 arms
  byte-identical.
- The amendment was triggered by calibration results, not test results. Its risk statements
  therefore use a union bound over the two procedures: **δ ≤ 0.10** for v2, and δ = 0.05
  remains for v1.

The v2 global arm is feasible in every fold, backbone and target. Low-visibility strata at
sFMR ≤ 1 % often stay infeasible; that is a capacity limit (§ CALC-003), not an ordering
artefact.

## 3. Reported in the final stage

1. **Clean** (`test_balanced`, plus `test_far` impostors for FMR):
   - ROC-AUC and EER;
   - FMR and FNMR at every frozen threshold, with identity-jackknife CIs;
   - benchmark-mode TAR@FAR (threshold chosen on test itself, the literature convention,
     labelled as such);
   - accuracy at the val best-accuracy threshold;
   - score quantiles.
2. **LOFO degradation** (per fold, Test-ID vs Test-OOD, same pairs as clean):
   - AUC, FMR and FNMR at the clean deployment threshold;
   - breakdowns by stratum, family combination, maximum severity level and co-visibility bin;
   - paired clean-vs-occluded score drops and decision flips.
3. **Risk control:** realised selective FMR, selective FNMR and coverage of every frozen
   arm/target on Test-ID and Test-OOD, with CIs. Each gets a verdict: within target, point
   estimate above target, or violated (CI lower bound > target). The guarantee is claimed only
   for Test-ID; on Test-OOD the violation is **measured** (H2a).
4. **C1 (exploratory, fit-free):** among occluded pairs, AUROC of low co-visibility and of the
   low feature-norm quality proxy for decision errors at the deployment threshold, and the
   co-visibility AUROC *within* quality deciles. This is not a test of a trained model.
5. **Stress (exploratory):** two physical families on each side, on a 20,000-pair subsample.
6. **Suspicious-performance audit:**
   - identity overlap between splits and pair sets (must be 0);
   - AUC exactly 1;
   - label-shuffle AUC (≈ 0.5 expected);
   - validation–test AUC gap;
   - per-part AUC (72-image vs 5-image identities).

**Intervals:** genuine-pair rates use the logit jackknife and impostor-pair rates the linear
jackknife, both deleting one identity (SIM-010). At zero events, a Clopper–Pearson bound is
reported and flagged as treating pairs as independent. Intervals for paired *differences* are
not validated and are not reported.

## 4. What the plan cannot show

- No real-world robustness. Faces and occlusions are synthetic, and F5 lighting is non-physical.
- No detector-inclusive results (no detector path was run).
- Risk-control guarantees hold only for Test-ID under A1–A7. Test-OOD is an empirical shift
  measurement.
- One realisation of each calibration set: the realised Test-ID risk of a single calibration
  draw can legitimately exceed the target with probability ≤ δ. Five folds × two backbones ×
  two targets × three arms are correlated draws, not independent replications.

## 5. Results (MEASURED; single final run `38d853d8884c`)

The run used freeze `0e8a43fb…` and was logged in `results/test_runs.jsonl` at commit
`f3376ea`; the source is `reports/experiments/phase2_final.json`. All numbers are on
**DigiFace-1M synthetic faces** with **synthetic occlusions** and are not real-world
performance. Intervals are 95 % identity-jackknife intervals (§3).

### 5.1 Clean baselines (`test_balanced`: 61,122 genuine + 61,122 impostor pairs, 20,374 identities)

| | AdaFace IR-101 | ArcFace IR-101 |
|---|---|---|
| ROC-AUC | 0.99856 | 0.99789 |
| EER | 1.28 % | 1.61 % |
| **Deployment** threshold from cal (FAR 1 %): realised FMR, test_balanced | 1.04 % [0.93, 1.16] | 0.98 % [0.86, 1.09] |
| … realised FMR, `test_far` (100,000 impostors) | 1.10 % [1.00, 1.19] | 1.00 % [0.91, 1.09] |
| … FNMR (= 1 − TAR) | 1.42 % [1.31, 1.53] | 2.21 % [2.08, 2.35] |
| Threshold from val at FAR 0.1 % (estimation-only): realised FMR on `test_far` | 0.128 % [0.096, 0.160] | 0.148 % [0.114, 0.182] |
| … FNMR | 4.35 % [4.16, 4.54] | 6.08 % [5.87, 6.30] |
| Benchmark-mode TAR@FAR 1 % / 0.1 % (threshold chosen on test; literature convention) | 98.54 % / 95.41 % | 97.83 % / 93.50 % |
| Accuracy at the val best-accuracy threshold | 98.77 % | 98.42 % |
| AUC, 5-image vs 72-image identities | 0.99869 vs 0.99274 | 0.99801 vs 0.99295 |

The FAR 1 % threshold fixed on calibration data transfers to test within interval.

The FAR 0.1 % threshold fixed on validation realises 0.13–0.15 %. For ArcFace the interval
excludes 0.1 %. This is consistent with D-034: a 0.1 % operating point estimated from about
31k validation impostors is not reliable, so it stays estimation-only.

The 72-image identities (rendered with more accessory and pose variation) are measurably
harder for both backbones.

### 5.2 Occlusion: LOFO degradation at the clean deployment threshold

| Held-out family (fold) | AdaFace AUC Test-ID / Test-OOD | AdaFace FNMR ID / OOD | ArcFace AUC ID / OOD | ArcFace FNMR ID / OOD |
|---|---|---|---|---|
| F1 mask | 0.9849 / 0.9864 | 10.9 % / 10.8 % | 0.9815 / 0.9825 | 13.7 % / 14.5 % |
| F2 sunglasses | 0.9817 / 0.9923 | 12.5 % / 7.6 % | 0.9777 / 0.9902 | 15.8 % / 9.8 % |
| F3 hand/object | 0.9832 / 0.9880 | 12.3 % / 10.1 % | 0.9794 / 0.9852 | 15.5 % / 12.9 % |
| F4 random patch | 0.9911 / **0.9676** | 8.1 % / **21.6 %** | 0.9877 / **0.9630** | 11.1 % / **25.1 %** |
| F5 extreme lighting (non-physical) | 0.9825 / 0.9908 | 13.5 % / 6.8 % | 0.9792 / 0.9870 | 16.5 % / 9.8 % |
| *Clean (same pairs)* | *0.9986* | *1.4 %* | *0.9979* | *2.2 %* |

Each plan has 25 % clean–clean pairs, 25 % clean–occluded, 25 % occluded–occluded same
family and 25 % cross family, with severity uniform over low/medium/high.

- **Degradation is large and mostly non-match driven.** FNMR rises 4–15× at the clean
  threshold. FMR falls in 9 of 10 fold/role cells per backbone (0.5–0.9 %), because occlusion
  lowers impostor scores too; held-out masks are the exception (1.18 % AdaFace, 1.00 % ArcFace).
- **Masks are the exception.** Mask–mask impostor pairs give FMR 2.7 % (AdaFace) and 2.2 %
  (ArcFace) against about 0.9 % clean. Masked faces of different synthetic people look more alike.
- **"Held-out" is not uniformly harder.**
  - Only random patch (up to 65 % face coverage at high severity) is harder as the held-out
    family than the seen-family mix.
  - Held-out sunglasses or lighting is milder than the Test-ID mix, which contains the harder
    families.
- **Severity is monotone.** Held-out random patch, AdaFace FNMR by the pair's maximum level:
  1.3 % (none) → 5.3 % (low) → 17.6 % (medium) → 44.9 % (high). Figure:
  `reports/figures/severity_fnmr_heatmap.svg`.
- **ArcFace degrades slightly more than AdaFace** in every fold. Both share data and
  architecture, so this is a loss-function difference on this synthetic domain.
- **Stress (exploratory; two physical families per side, 20,000 pairs).** FNMR 37.4 %
  (AdaFace) and 43.3 % (ArcFace), against 1.6 % / 2.4 % on the same pairs clean.

### 5.3 Risk control: realised selective error of frozen LTT rules

Verdicts compare the realised test risk with the target. "Violated" means the 95 % interval
lies entirely above the target.

Counts below are cells of 5 folds × 2 backbones; NA marks an infeasible fit, which abstains on all pairs.

| Arm (target sFMR / sFNMR) | Test-ID: violated / within / NA | Test-OOD: violated / within / NA | Coverage range |
|---|---|---|---|
| `global` v1 (5 % / 5 %) | 0 / 10 / 0 | **2** / 8 / 0 (both F4) | 0.62–0.95 |
| `stratified_vis` v1 (5 % / 5 %) | 0 / 10 / 0 | **0** / 10 / 0 | 0.16–0.82 |
| `clean_cal` (5 % / 5 %) | **10** / 0 / 0 | **10** / 0 / 0 | 1.00 |
| `global_v2` (5 % / 5 %) | 0 / 10 / 0 | **2** / 8 / 0 (both F4) | 0.90–1.00 |
| `stratified_vis_v2` (5 % / 5 %) | 0 / 10 / 0 | **2** / 8 / 0 (both F4) | 0.73–0.95 |
| `global` v1 (1 % / 5 %) | 0 / 6 / 4 | 0 / 6 / 4 | 0.37–0.54 |
| `stratified_vis` v1 (1 % / 5 %) | 0 / 10 / 0 | 0 / 9 / 1 | 0.18–0.68 |
| `global_v2` (1 % / 5 %) | 0 / 10 / 0 | **2** / 8 / 0 (both F4) | 0.80–0.94 |
| `stratified_vis_v2` (1 % / 5 %) | 0 / 10 / 0 | **2** / 7 / 1 | 0.24–0.87 |
| `clean_cal` (1 % / 5 %) | **7** / 3 / 0 | **2** violated + 1 point-above / 7 / 0 | 0.79–0.92 |

1. **Test-ID: no violation in any certifiable occlusion-calibrated arm** (0 violations in the 76
   certifiable Test-ID cells across both targets and orderings; 4 cells were infeasible). This is consistent with LTT validity under A1–A7;
   a single realisation cannot *prove* validity.
2. **Test-OOD: risk control fails exactly where the held-out family is hardest.** With F4
   (random patch) held out:
   - global: sFNMR 10.0 % [CI above 5 %] (AdaFace) and 8.8 % (ArcFace);
   - global v2: 13.2 % and 12.2 %;
   - stratified v2: 7.5 % and 6.8 %.

   All other held-out families stay within target. sFMR stays within target everywhere.
   Hypothesis **H2a** (degradation on held-out families) is supported for one of five
   families on synthetic data.
3. **Oracle-visibility stratification** (pre-registered v1) **removed the F4 violation**:
   sFNMR 0.98 % (AdaFace) and 3.50 % (ArcFace). The price was coverage: 0.46 vs 0.94
   (AdaFace) and 0.74 vs 0.88 (ArcFace).
   - AdaFace abstained on the whole low-visibility stratum, where no rule was certifiable.
   - ArcFace accepted 54 % of low-visibility pairs at 7.0 % in-stratum sFNMR.
   - The v2 stratified arm keeps more coverage and reduces, but does not remove, the
     violation.

   This is **H2b-consistent** evidence with *oracle* visibility only; a learned estimator
   remains untested.
4. **Calibrating on clean pairs does not transfer to occlusion, even for seen families.**
   sFNMR is 6–22 % against a 5 % target in 20 of 20 cells at the primary target.
   Occlusion-aware calibration data is necessary.
5. **Clean pairs.** Every clean-calibrated rule stays within target on clean test pairs
   (sFMR ≤ 1.6 %, sFNMR ≤ 3.8 %).

### 5.4 C1 exploratory (fit-free; occluded pairs; errors at the clean deployment threshold)

| | AUROC of low co-visibility (oracle) | AUROC of low feature norm (quality) | Co-visibility AUROC *within* quality deciles |
|---|---|---|---|
| AdaFace, 10 fold/role cells | 0.58–0.75 | 0.70–0.84 | 0.50–0.65 (> 0.5 in 9/10) |
| ArcFace, 10 fold/role cells | 0.57–0.74 | 0.65–0.78 | 0.52–0.64 (> 0.5 in 10/10) |

The backbone's own feature norm ranks errors better than oracle co-visibility. Co-visibility
keeps a weak-to-moderate residual signal *beyond* quality (19 of 20 cells above 0.5). This
is ESTIMATED, exploratory evidence for C1 (H1), not a test of a trained model.

### 5.5 Throughput and latency (MEASURED, RTX 4060 Laptop, fp16, batch 256)

| | AdaFace | ArcFace |
|---|---|---|
| Model forward, full run (2.17 M items) | 477 img/s (2.1 ms/img) | 520 img/s (1.9 ms/img) |
| CPU fp32 (pilot, 128 images) | 16 img/s | 18 img/s |

- End-to-end with both backbones, including PNG decode and occlusion re-rendering on 8 CPU
  workers: 241 img/s, 2.5 h in total.
- Peak GPU memory: 2.5 GiB.
- Detection and alignment latency: not applicable (pre-aligned data).

Figures: `reports/figures/*.svg`.
