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

## 5. Results

*To be filled from `reports/experiments/phase2_final.json` after the single final run.*
