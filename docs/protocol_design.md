# VEIL-X — Protocol Design (Phase 0 lock)

> Machine-readable source of truth: `configs/research_protocol.yaml` (validated by
> `src/veilx/config/schema.py`). This document explains it. **Status: `provisional`
> (protocol v0.2.0; v0.1.0 → v0.2.0 changes are listed in `decision_log.md` D-019…D-025).** Constants marked *provisional* may change only through
> `docs/decision_log.md`. After the first `--final` run, changes go to
> `evaluation_protocol.md` §10 (deviations).
> Labels [V]/[H]/[D]/[U]/[M] as in `research_landscape.md` §0. **No [M] results exist.**

## 1. Approved research questions

| ID | Role | Question |
|---|---|---|
| **C3** | **Primary** | (i) Do Learn-then-Test (LTT) risk-controlled decisions calibrated on seen challenge families keep their target error on **held-out** families and real occlusions? (ii) Does **visibility-stratified** calibration reduce the violation, and at what coverage cost? |
| **C1** | Secondary | Does pair-level **co-visibility** add predictive power for error prediction and abstention beyond standard image-quality measures? |
| C2 | Optional | Regional (visibility-weighted) feature fusion for raw accuracy. Reported only if built. |

Excluded from the first implementation: backbone fine-tuning, large mixture of experts, and
ArcFace+InfoNCE retraining.

## 2. What the guarantee does and does not mean

LTT ([P27]) certifies, with probability ≥ 1 − δ over the calibration draw, that the
controlled risk on **exchangeable** future data is ≤ the target. VEIL-X states its
assumptions explicitly (config `calibration.assumptions`, A1–A6):

- **A1 Exchangeability.** It holds by construction only for **Test-ID** (seen families,
  disjoint identities, same source). It **does not hold** for held-out families, ROF or other
  benchmarks. There, the guarantee is **not claimed**; realised risk vs target is *measured*
  (H2a).
- **A2 Independence.** Calibration pairs are made independent by an **identity-disjoint
  matching** (§4.3).
- **A3 Calibration isolation.** Scores, fusion and visibility models, quality models and the
  λ grid are fixed on train/val **before** the calibration data are opened.
- **A4 Composition.** Error among accepted pairs depends on the genuine:impostor mix. The
  guarantee is for the calibration mix (genuine fraction 0.5). Class-conditional sFMR and
  sFNMR are always reported.
- **A5 Marginal validity.** The guarantee is not conditional on identity, stratum or family,
  except per-stratum guarantees under stratified calibration, each with its own sample-size
  check.
- **A6 Multiple testing.** λ values are tested in a pre-specified fixed sequence (or
  Bonferroni).

**Prediction target used in calibration (verbatim from config):** the loss is 1 if the system
*accepts* a pair (outputs MATCH or NON-MATCH, not ABSTAIN) and the decision disagrees with the
ground-truth same-identity label, else 0. The controlled quantity is the error rate among
accepted pairs. Alignment failures are abstentions.
**Resolved in Phase 1a (`ltt_audit.md`, D-019, protocol v0.2.0).**
- The binomial p-value on the accepted subset is valid for the conditional risk under i.i.d.
  calibration pairs and *acceptance locality* (A7). This is the construction LTT §3.2 uses.
- The **primary** controlled estimand is now the class-conditional pair: selective FMR and
  selective FNMR, tested with an intersection–union max-p. Together they bound the
  accepted-decision error for **any** genuine:impostor prevalence.
- The pooled error at the calibration prevalence is secondary, and valid only with Bernoulli
  class assignment.
- The candidate order is fixed on validation data (D-020).

## 3. Identity grouping and splits

- **Unit = identity.** `bucket = int(sha256("{dataset}:{identity_id}")[:8],16) mod 100`.
  Bands are train [0,60), val [60,70), cal [70,80) and test [80,100). Assignment is
  deterministic, seed-free and order-independent (`veilx.data.splits`).
- Synthetic variants inherit their source image's split and identity
  (`validate_image_manifest`).
- **Benchmark protocols stay test-only.** ROF, MLFW, CPLFW and XQLFW never enter
  train/val/cal (`DatasetEntry.test_only`, schema-enforced). Their official pair lists are
  reported **as-is and separately** from VEIL-X-constructed pair sets.
- **Test lock.** `guards.require_final_for_test` refuses test access without `--final`. Each
  final run is logged to `results/test_runs.jsonl`.

## 4. Pair construction

### 4.1 Constructed (VEIL-X) pair sets, per split × LOFO run × seed
- **Strata:** clean–clean, clean–occluded, occluded–occluded (same family), and
  occluded–occluded (cross family), with equal quotas.
- **Genuine pairs:** distinct images of the same identity, with **different source images**.
  Two synthetic variants of one photo are never a genuine pair, because that is trivially
  easy and leaky. Cap: 20 per identity.
- **Impostor pairs:** identity pairs sampled uniformly without replacement, one image each.
- **Composition:** genuine fraction 0.5 for selective-risk sets. Separate impostor-heavy
  **FAR-estimation sets** contain ≥ 10/FAR impostor pairs per reported FAR. Their composition
  is reported separately and never mixed into selective-risk numbers.
- **Family assignment:** train/val/cal use only `seen_families(F_k)`. Test-OOD uses F_k on at
  least one side of every occluded pair. Test-ID uses seen families.
- Within-benchmark exact and near-duplicate image pairs are flagged and **excluded from
  constructed genuine pairs**. Counts are reported.

### 4.2 Error definitions (reported with every result)
- y = 1 genuine, y = 0 impostor. Decision d ∈ {MATCH, NON-MATCH, ABSTAIN}.
- False match (FM): d = MATCH ∧ y = 0. False non-match (FNM): d = NON-MATCH ∧ y = 1.
- **FAR/FMR** (score-based) = FM / impostors. **TAR** = 1 − FNMR.
- **Selective error** = (FM + FNM) / accepted. **sFMR** = FM / accepted impostors. **sFNMR** =
  FNM / accepted genuines.
- **Coverage:** overall, genuine-only and impostor-only.
- TAR@FAR is reported in two modes: *benchmark* (threshold set on the test ROC, for
  comparability only) and *deployment* (threshold from cal, with realised FAR and TAR
  reported).

### 4.3 Calibration-set construction (primary vs sensitivity)
- **Primary: identity-disjoint matching.** Each cal identity appears in at most one pair: a
  genuine pair uses 1 identity, an impostor pair uses 2. This gives independent Bernoulli
  losses, so binomial tests are valid under A1.
- **Sensitivity:** (a) one pair per identity pair, (b) all pairs. These are reported as
  approximations whose validity is *not* claimed.

## 5. Held-out-family evaluation (leave-one-family-out)

Families: **F1 masks, F2 sunglasses, F3 hands/objects, F4 random patches, F5 extreme
lighting**. F5 is flagged `physical_occlusion: false` and reported in a **separate table**
with separate interpretation.

For each held-out F_k, nothing that fits or selects may see F_k
(`heldout.isolation_scope`, schema-enforced to include all of the following):
feature selection, model fitting, threshold selection, hyperparameter optimisation,
calibration, visibility-estimator training, occluder-asset selection, and dedup-threshold
validation. Runtime guards: `assert_family_isolation` (manifests) and
`require_heldout_isolation` (fitted artefacts carry `trained_on_families`).

Runs: 5 LOFO folds × 2 backbones × 3 seeds. Each run reports Test-ID, Test-OOD (F_k) and
Test-real (ROF etc.). Each real set is labelled *real-unseen* or *real-seen-type* depending on
whether its occlusion type matches F_k (e.g. ROF-masked is real-unseen only in the F1 fold).

**Caveat [H]:** "unseen" means unseen by VEIL-X's fitted components and calibration. The
frozen backbones were pretrained on web data that contains masks, sunglasses and hands.

## 6. Required comparison arms

| Arm | Description | Needed for |
|---|---|---|
| R-Ada | Raw AdaFace IR101 cosine, single cal threshold | All |
| R-Arc | Raw ArcFace cosine, single cal threshold | All |
| Q | Quality-based abstention (feature norm; CR-FIQA if licence-cleared) | C1, C3 |
| QM | Combined abstention: quality + score margin (logistic, fit on train/val) | C1, C3 |
| G | Global LTT calibration of the arm's score | C3 |
| VS | **Visibility-stratified** LTT calibration (co-visibility bins fixed on train/val) | C3-ii |
| OAT | **Occlusion-adaptive threshold** (cf. [P13]): threshold τ(s) monotone in estimated occlusion severity s = 1 − co-visibility, parameters fit on train/val (seen families). Plus a quality-adaptive variant (norm-binned thresholds). | Credible baseline for C3-ii |
| OF | Oracle-family stratification. **Test-ID only**, as an upper-bound diagnostic: a held-out family has no calibration stratum by definition | C3-ii ceiling |

**C1 feature-set comparison** (same model class, same data, out-of-sample):
`quality_only`, `covisibility_only` and `quality_plus_covisibility`, plus `margin_only` and
`quality_plus_margin` as reference. Outcomes:
1. **Error prediction:** AUROC and log-loss for predicting "decision is wrong at the
   deployment threshold".
2. **Abstention:** AUGRC and risk-coverage. Differences use paired identity-cluster
   bootstrap.

## 7. Registered constants (all **provisional**) and feasibility

Exact binomial calculations from `scripts/feasibility_report.py`. These are calculations,
not measurements.

| Constant | Value | Feasibility finding | Proposed disposition |
|---|---|---|---|
| Clean non-inferiority margin | 0.5 pp at TAR@FAR=1% | **Simulated (SIM-006):** power 28% / 68% / 100% with 200 / 1,000 / 3,333 clean test identities (4 genuine pairs each, true difference 0) | Keep provisional; needs about 3,000 clean test identities |
| FM:FNM cost ratio | 10:1 | Reporting/tie-break weight; no sample-size implication | Ready to freeze |
| Confidence level (1 − δ) | 0.95 | Standard; drives the counts below | Ready to freeze |
| Accepted-decision error target | 5% | Certifiable with **≥ 59** independent accepted cal pairs at 0 errors, **≥ 93** at 2% observed, **≥ 286** at 3%, **≥ 1,282** at 4%. Applies **per stratum** under stratified calibration | Feasible |
| FAR operating points | 1%, 0.1% | **Estimation:** feasible (10,000 test impostor pairs for 0.1%). **Certification of sFMR:** 1% needs **≥ 299** independent impostor pairs (0 FM); 0.1% needs **≥ 2,995** (0 FM) or **≥ 4,742** (1 FM). Identity-disjoint matching gives N/3 impostor pairs from N cal identities: DigiFace 72-img P1 → 66; 5-img P1 → 1,111; all three 5-img parts → 3,333 | **1% certifiable** with a 5-img part; **0.1% estimation-only** unless all three 5-img parts are approved |
| Interval method and level | 95%, identity as resampling/deletion unit | **SIM-010 (Phase 1b):** logit jackknife covers genuine-pair rates 94.8–95.7%; linear jackknife covers impostor-pair rates 96.0–97.5% (conservative); the identity percentile bootstrap undercovers FNMR (93.5%) | **Frozen** in v0.3.0 (level 0.95; methods per statistic). Paired-difference intervals not yet validated |
| Near-dup threshold | **unset** | Fixed only after the §8 validation | — |
| Dedup target recall | 0.99 | Measured on synthetic known duplicates (§8) | Provisional |

**ROF (if approved):** 180 identities (70 with masks). CI widths at FAR = 0.1% must be
established by simulation in Phase 1 before ROF supports any claim. Wide intervals are
expected.

## 8. Deduplication procedure (no arbitrary fixed threshold)

1. **Exact:** SHA-256 of decoded pixels (format-independent).
2. **Candidates:** 64-bit perceptual hash (Hamming distance) and frozen-backbone embedding
   cosine.
3. **Validation on approved non-test data only** (DigiFace train/val identities):
   - *Positives* = synthetic known duplicates made by controlled transforms: JPEG q 30–95,
     resize 0.5–1.5×, crop ≤ 10%, colour jitter, noise, horizontal flip.
   - *Hard negatives* = different images of the **same identity**. These must *not* be
     removed. Further negatives = impostor pairs.
   - Fit the decision rule (pHash ≤ h OR cos ≥ t) on train identities to reach the target
     duplicate recall at the lowest same-identity false-positive rate. **Confirm on val
     identities.** Record the result in `deduplication.near_duplicate.validation_record`.
     The schema refuses a threshold without that record and refuses `test` as the
     validation split.
4. **Application:** compare non-test against test images and remove flagged items from the
   **non-test** side only. Test data are never altered. Report counts.
5. **Limitation:** dedup detects duplicate *images*, not the same *person* across datasets.
   Identity overlap with backbone pretraining data is a separate, documented confound.
6. **Phase 1b outcome on DigiFace (D-029):** the procedure was run with pHash + thumbnail NCC.
   Face-embedding similarity is excluded, because two photos of one person are not duplicates.
   The best rule reaching 99% recall failed the pre-set same-identity FPR criterion, so **no
   near-duplicate threshold was adopted**. Exact pixel duplicates: 0. Details:
   `data_integrity_report.md` §3.

## 9. Bootstrap resampling unit

*Phase 1a design:* identity-cluster bootstrap. Resample test identities with replacement. A
genuine pair is kept with the multiplicity of its identity. An impostor pair is kept if both
identities are drawn, with the product of their multiplicities. Method differences (C1, C3-ii)
are computed **within the same resample**.

**Phase 1b revision (SIM-010, D-032):** the percentile bootstrap undercovered FNMR (93.5%) and
over-covered FMR (product weights roughly double the variance). Intervals now use the
**delete-one-identity jackknife**, where deleting an identity removes every pair it appears
in, on either side:
- **logit scale** for genuine-pair rates (FNMR/TAR): coverage 94.8–95.7%;
- **linear scale** for impostor-pair rates (FAR/FMR): coverage 96.0–97.5%, conservative.

Intervals for paired *differences* between methods still need their own coverage study
before use.

## 10. Decision rules (unchanged from `evaluation_protocol.md` §1, re-indexed)

- **H2a (C3-i, primary):** the violation is reported as realised sFMR/α and selective
  error/target, with identity-bootstrap CIs, per held-out family, with F5 separate. There is
  no success threshold: this is a characterisation.
- **H2b (C3-ii, primary):** VS is said to reduce violation only if the paired 95% CI of the
  reduction excludes 0 on ROF (if approved) and on ≥ 3 of the 4 physical families, after
  Holm–Bonferroni, with coverage cost reported. Results are compared against OAT as well as
  G.
- **H1 (C1, secondary):** co-visibility "adds predictive power" only if
  `quality_plus_covisibility` beats `quality_only` (paired CI excludes 0) on the same criteria.
- **H4:** clean-pair non-inferiority within the registered margin.
- A null or negative result is reported as such.
