# Selective Face Verification Under Unseen Occlusions and Population Shift: Evaluating Quality, Learned Visibility, and Abstention

*VEIL-X research report.* Non-commercial academic research. Final version, Phase 6, 2026-11-02.
Repository: `MeezaUni/VEIL-X`. Every number in this paper comes from a committed, provenance-stamped
report in `reports/experiments/`. The claim-to-evidence matrix is `docs/phase6_audit.md` §4.

## Abstract

Face verification systems that can *abstain* promise a bounded error rate on the decisions they
do make. We ask whether such guarantees survive two realistic shifts: **occluders never seen
during calibration**, and **a population different from the calibration population**.

**Setup.** We use frozen AdaFace and ArcFace IR-101 backbones, synthetic DigiFace-1M faces, and
five synthetic occluder families in leave-one-family-out folds. With these we compare three-way
(MATCH / NON-MATCH / ABSTAIN) decision rules calibrated by Learn-then-Test (LTT) at 5 % selective
false-match and false-non-match targets. The comparison spans three pre-registered phases.

**Findings.**
1. *Quality-stratified calibration (A3).* Stratifying on the backbone's feature norm met both
   targets on every held-out occluder family of the calibration-matched population.
2. *Learned visibility fails on unseen occluders.* A learned visibility estimator overestimated
   visibility on unseen occluders (random patch: IoU 0.13, +0.34 visible-fraction bias), and
   visibility-stratified rules violated the target (sFNMR 8.52 %).
3. *Fixing the estimator does not help the decision.* A conservative, novelty-aware redesign
   corrected the overestimation, yet no visibility-based rule, not even ground-truth visibility,
   was safe relative to A3 on an occluder never seen by any design decision.
4. *Population shift.* On an untouched partition of a harder sub-population (72-image DigiFace
   identities), **every rule, A3 included, violated the 5 % false-non-match target even for
   occluders seen in calibration** (A3: 5.36–6.34 %). A pre-registered replication on 4,000 fresh
   identities replicated this: A3's sFNMR interval excluded the target in 4 of 5 seen-family cells (5.15–6.49 %), no visibility-based candidate was safe, and false-match rates stayed within target in all 15 cells.
5. *A low-quality failure mode.* A post-hoc audit found that the frozen fused score extrapolates
   below the calibration support of quality. There, synthetic non-face pairs are decided MATCH,
   and A3's false-match rate is 18–42 % on development data.

**Conclusion.** A3 is the best-supported method we tested. It is not universally safe:
distribution-free guarantees are calibration-population guarantees. We release a local, privacy-
preserving demonstration that exposes these limits instead of hiding them.

## 1. Introduction

**The problem.** A 1:1 face verifier decides whether two images show the same person. Occlusion
by masks, sunglasses, hands, hair or arbitrary objects removes identity evidence. Its main effect
is to depress genuine-pair similarity, so false non-matches rise. In our Phase 2 baseline the
false-non-match rate (FNMR) at a fixed clean threshold rose from 1.4 % to 6.8–21.6 %
(§9.1).

**Selective decisions.** A natural safety mechanism is *selective* decision making: abstain when
evidence is weak, and certify an error bound on the accepted decisions. Distribution-free risk
control, notably Learn-then-Test (Angelopoulos et al., 2021), gives finite-sample guarantees under
one assumption: exchangeability between calibration and deployment data. The two shifts that
matter for occluded faces are exactly the ones that break that assumption:
- occluders absent from calibration;
- a deployment population that differs from the calibration population.

**The intuitive fix.** If the system knew *which parts of the face are visible*, it could stratify
calibration by visibility. Pairs with little shared visible face would get conservative
thresholds, and well-visible pairs efficient ones. VEIL-X tested this idea prospectively, over
three pre-registered phases, against a strong and simple alternative: stratifying by the
backbone's own feature-norm quality.

**This paper reports, plainly:**
- a negative result for learned visibility;
- a mechanism for that negative result;
- a population-shift failure that affects every rule we tested;
- a low-quality-tail failure of the frozen fused score.

## 2. Research questions and contributions

**Research questions:**
- **RQ1 (C3, primary).** Does stratifying LTT calibration by *learned* visibility reduce
  risk-control failures on unseen occlusion families, compared with quality stratification?
- **RQ2 (C1, secondary).** Does pair-level co-visibility add predictive information beyond
  similarity and quality?
- **RQ3 (Phase 4).** Can a conservative and/or novelty-aware visibility estimator, or a
  quality-first hybrid, be used safely relative to quality stratification?
- **RQ4 (Phases 4 and 6).** Do the calibrated guarantees survive a shift of the face population?

**Original contributions** (prior work and pretrained components are listed in §4):
1. **Protocol.** A pre-registered, leave-one-family-out evaluation protocol for selective face
   verification under occlusion. It has identity-disjoint calibration, a single logged access to
   every test partition, identity-clustered (jackknife) inference, and an evaluation-only
   occluder never used in any design decision.
2. **Visibility estimator and conservative variant.** A learned 28×28 visibility estimator and a
   conservative variant (asymmetric loss, temperature, lower-bound shift, Mahalanobis novelty).
   Calibration arms stratified by quality, learned visibility, ground-truth visibility and
   hybrids.
3. **Negative result with a measured mechanism.** Learned visibility fails because it learns
   occluder *classes*, so unseen occluders are confidently called visible. The failure is
   reproduced in simulation (SIM-013).
4. **The population-shift finding** and its pre-registered replication on untouched identities.
5. **The low-quality-tail finding** for the frozen fused score.
6. **A local, privacy-preserving demonstration.** It reproduces the offline frozen system exactly
   (1000/1000 decisions) and displays its limitations.

## 3. Related work

**Deep face recognition.**
- *Losses.* Margin-based losses underlie modern face verification: ArcFace (Deng et al., 2019)
  and AdaFace (Kim et al., 2022). AdaFace uses the feature norm as an image-quality proxy, and
  quality-aware representations such as MagFace (Meng et al., 2021) make the same link explicit.
- *Training data.* Large training sets come from web-crawled celebrity images, such as
  WebFace260M and its WebFace4M subset (Zhu et al., 2021).
- *Synthetic data.* Synthetic data, such as DigiFace-1M (Bae et al., 2023), offers
  privacy-preserving faces with controllable variation.

**Face image quality.** Quality estimation predicts recognition utility. Examples are
unsupervised estimation from embedding robustness (SER-FIQ; Terhörst et al., 2020) and
norm-based quality (Meng et al., 2021). Our quality stratum is the AdaFace feature norm,
following that line.

**Occluded face recognition.**
- *Methods.* Methods learn to discard corrupted features, for example the pairwise differential
  Siamese network of Song et al. (2019).
- *Evaluations.* Real masked-face evaluations (Ngan et al., 2020) document large accuracy losses.
- *Datasets.* Real occluded-face datasets include ROF (Erakın et al., 2021) and IJB-C (Maze et
  al., 2018), the latter with occlusion annotations.

We do not propose a new recognition model. We study *decision-level* risk control on top of
frozen models.

**Selective prediction and risk control.**
- *Abstention.* Optimal reject options go back to Chow (1970). Selective classification for deep
  networks is formalised by Geifman and El-Yaniv (2017).
- *Distribution-free control.* Conformal prediction (Vovk et al., 2005; Angelopoulos and Bates,
  2021), risk-controlling prediction sets (Bates et al., 2021) and Learn-then-Test (Angelopoulos
  et al., 2021) give finite-sample guarantees under exchangeability.
- *Shift.* Under covariate shift, guarantees need reweighting with known likelihood ratios
  (Tibshirani et al., 2019).
- *Evaluation.* Evaluation pitfalls of selective systems are discussed by Traub et al. (2024).

**Dataset shift and out-of-distribution detection.**
- *Shift.* Dataset shift is surveyed by Quiñonero-Candela et al. (2009), and its detection is
  studied empirically by Rabanser et al. (2019).
- *Novelty.* Our novelty signal is the class-free Mahalanobis distance on intermediate features,
  in the spirit of Lee et al. (2018).

## 4. Datasets, pretrained components and ethical considerations

| Component | Role | Origin | Terms |
|---|---|---|---|
| DigiFace-1M (Bae et al., 2023) | All training, calibration and confirmatory data (synthetic faces) | Official host | Research-only (R-UDA) |
| LFW (Huang et al., 2007), Kaggle mirror | Real-photograph follow-up evaluation, synthetic overlays only | Kaggle mirror; used under the project owner's reported academic authorisation (D-043) | No redistribution rights established; nothing redistributed |
| AdaFace IR-101 / ArcFace IR-101 (WebFace4M), CVLface releases | Frozen backbones (not trained by us) | Pinned revisions, SHA-256 verified | Research / non-commercial |
| SCRFD det_10g (Guo et al., 2022), InsightFace `buffalo_l` | Face detection and landmarks (not trained by us) | Pinned, SHA-256 verified | Non-commercial research |

**DigiFace usage:**
- 72-image identities 0–1999 and 5-image identities 100,000–199,998 (Phases 1–3);
- 72-image identities 2000–5999 as the untouched Phase 4 partition (`p4_test`);
- 72-image identities 6000–9999 as the untouched Phase 6 replication partition (`p6_test`).

The Phase 3 calibration set (`p3_cal_independent`) has 6,146 identity-disjoint pairs over 9,219
identities. **9,033 of them are 5-image identities.**

**Synthetic occluders:**
- *Families:* F1 mask, F2 sunglasses, F3 hand/object, F4 random patch, F5 extreme lighting (a
  corruption, not an occluder).
- *Severity:* three levels each.
- *E1:* an evaluation-only hair-strand occluder, defined before any Phase 4 result and never
  rendered in development.
- *Provenance:* every render is reproducible from a stored recipe and seed.

**Ethics and privacy:**
- **Scope.** Two-image verification only: no identification, gallery or surveillance function.
- **Repository contents.** No face image, embedding, identity list, manifest or weight file is in
  the repository. Only aggregates are published.
- **LFW.** LFW identities likely overlap the backbones' training data, which makes LFW results
  optimistic (§14).
- **Demonstration.** The application runs on the loopback interface only, with no telemetry and
  no persistent storage of images or embeddings (§13).

## 5. Experimental protocol

**Splits and pairs:**
- *Identity splits.* Identities are split by a hash band.
- *Phase 3 sub-splits.* Train-split identities are sub-split into `p3_train` [0, 50),
  `p3_val` [50, 60), `p3_cal` [60, 75) and `p3_test` [75, 100).
- *Calibration pairs.* These are identity-disjoint: no identity or image is shared between
  pairs. This is the LTT independence condition.
- *Occlusion plans.* Plans assign occlusions per pair side in a leave-one-family-out (LOFO)
  design:
  - fold *k* holds out family *k* from training and calibration;
  - **Test-ID** cells use the seen families;
  - **Test-OOD** cells use the held-out family;
  - E1 cells use the never-seen hair occluder.

**Decision rule and metrics.**
- *Rule.* A rule maps a fused score p to MATCH if p ≥ τ_high, NON-MATCH if p < τ_low, and
  ABSTAIN otherwise.
- *Metrics:*
  - selective FMR (sFMR): false matches / accepted impostor pairs;
  - selective FNMR (sFNMR): false non-matches / accepted genuine pairs;
  - coverage: the fraction of pairs decided.
- *Targets.* Primary: sFNMR ≤ 5 % and sFMR ≤ 5 %, at confidence 1 − δ, δ = 0.05.

**Calibration.**
- *Method.* LTT with fixed-sequence testing over a pre-ordered candidate list, using exact
  binomial p-values on class-conditional error counts.
- *Candidate ordering.* The ordering (v2) is fixed from validation p-values before calibration.
- *Stratified arms.* These calibrate one rule per stratum, with stratum edges fixed on
  validation data.

**Inference.**
- *Violations.* A cell × error type is a *violation* when its 95 % delete-one-identity jackknife
  interval excludes the target.
- *Paired comparisons.* Arms are compared with a paired identity jackknife (Efron and Tibshirani, 1993)
  and Holm correction (Holm, 1979).
- *Zero-event rates.* These use Clopper–Pearson bounds (Clopper and Pearson, 1934).

**Discipline:**
- every phase's protocol was committed before training or scoring;
- every test partition was accessed through a logged run (`results/test_runs.jsonl`);
- previously inspected sets are labelled follow-up, never confirmatory;
- the access inventory is in `docs/phase6_audit.md` §3.

## 6. Recognition baselines

Both backbones are frozen IR-101 networks trained on WebFace4M: AdaFace (primary) and ArcFace.

**Inputs and precision:**
- *Inputs:* 112×112 RGB crops aligned to the ArcFace template; embeddings are L2-normalised
  512-d vectors.
- *Precision:* fp16 autocast. On 20,000 validation pairs its minimum cosine to fp32 is
  ≥ 0.9998.

**Clean DigiFace (Phase 2, confirmatory run `38d853d8884c`; 61,122 genuine and 61,122 impostor
pairs):**
- *Discrimination:* AdaFace AUC 0.99856 and EER 1.28 %; ArcFace AUC 0.99789 and EER 1.61 %.
- *Calibrated threshold:* the FAR-1 % threshold fixed on calibration data realised FMR 1.04 %
  [0.93, 1.16] with FNMR 1.42 % [1.31, 1.53] (AdaFace).
- *Harder identities:* the 72-image identities were harder for both backbones (AdaFace AUC
  0.99274 vs 0.99869).

**LFW** (real photographs, official View 2 protocol, run `79331a84f520`):
- *Accuracy:* 99.58 % ± 0.26 (AdaFace) with the detector.
- *Threshold transfer:* the DigiFace FAR-1 % threshold gave 0/3,000 false matches, so it is
  conservative on real photographs.

## 7. Visibility estimation and co-visibility

**Visibility estimator (VE).**
- *Architecture:* a U-Net-style network, 3.6 M parameters (in the spirit of Ronneberger et al.,
  2015), with one model per LOFO fold.
- *Output:* a 28×28 map of P(visible) from an aligned crop.
- *Training target:* the renderer's occlusion mask, average-pooled; the mask is never an input.

**Co-visibility.** Pair-level features summarise the face area visible in both images; the
registered definition is region-weighted.

**Conservative estimator (VE-C, Phase 4):**
- an asymmetric loss (4× penalty on calling occluded cells visible);
- natural-texture hard negatives;
- lighting treated as a corruption;
- temperature scaling and a lower-bound shift fitted on development renders of seen families;
- a Mahalanobis novelty score on the bottleneck features, with a 5 % in-distribution false-flag
  threshold.

## 8. Fused scores and selective calibration methods

**Fused scores.** These are logistic models over pair features, fitted on `p3_train` with
regularisation selected on `p3_val`:
- **M0**: AdaFace cosine *s*;
- **M1**: *s*, quality *q* and *s·q*, where *q* = min of the two AdaFace pre-normalisation
  feature norms;
- **M4**: M1 + co-visibility features;
- **M5**: M4 + ArcFace similarity, ArcFace quality and backbone disagreement;
- an MLP: exploratory, never used for decisions.

**Arms** (all LTT-calibrated on `p3_cal_independent` under the fold's calibration plan):

| Arm | Score | Strata |
|---|---|---|
| A1 | M0 | none, clean calibration pairs only |
| A2 / B0 | M0 | none (occlusion-aware global) |
| **A3** | **M1** | **tertiles of q** |
| A4 | M4 | tertiles of predicted co-visibility |
| A5 (oracle) | M4-oracle | ground-truth visibility strata (not deployable) |
| C4, N4, H4 (Phase 4) | VE-C-based scores | conservative visibility; novelty-forced strata; quality-first hybrid |

**Correction of an earlier description (D-052).** The recommended system A3 uses **AdaFace only**
(M1). ArcFace and dual-backbone fusion were compared in log-loss and AUC, but were never
calibrated as a decision system.

## 9. Experimental results

### 9.1 Occlusion degrades verification (Phase 2)

**At the clean threshold (AdaFace):**
- Test-OOD FNMR was 6.8–21.6 % across folds, against 1.4 % clean.
- Held-out random patch was hardest, rising monotonically with severity to 44.9 % at high
  severity.
- Clean-only calibration fails everywhere under occlusion: Phase 3 arm A1 sFNMR was 5.97–19.81 %,
  all violations.

### 9.2 Phase 3: learned visibility vs quality (confirmatory run `bb5d9244bb3c`)

Test-OOD cells, primary target; values are coverage / sFNMR / sFMR in %; ✗ = violation.

| Arm | mask | sunglasses | hand/object | random patch | lighting |
|---|---|---|---|---|---|
| A3 quality | 93.7 / 1.94 / 2.56 | 96.8 / 1.87 / 1.24 | 94.3 / 1.59 / 1.23 | 88.6 / 2.12 / 2.00 | 96.1 / 1.08 / 1.27 |
| A4 learned visibility | 94.8 / 1.75 / 3.14 | 96.8 / 1.28 / 0.89 | 95.4 / 2.36 / 1.22 | 98.5 / **8.52 ✗** / 2.13 | 98.7 / 2.25 / 1.28 |
| A5 oracle | 92.4 / 1.30 / 2.12 | 98.1 / 2.06 / 1.34 | 97.5 / 2.01 / 2.30 | 90.4 / 2.76 / 1.93 | 93.4 / 0.87 / 1.63 |

Figure: `reports/figures/phase3_ood_arms.svg`.

**Paired comparison.** A4 − A3 in sFNMR was −0.19, −0.59, +0.78, **+6.40** and +1.17 pp (Holm
p < 0.001 each).
- A4 is better where the estimator generalises and worse where it does not.
- RQ1 is answered **negatively**.

**RQ2 (co-visibility).** Co-visibility gave small gains on seen families, but was **harmful** on
held-out random patch (Δ log-loss −0.035). RQ2 is not supported.

**Clean margin.** Quality fusion improved clean FNMR (M4 0.68–0.74 % vs M0 1.12 %).

### 9.3 Phase 4: conservative and novelty-aware visibility (confirmatory run `903f6a5c7de3`)

**Estimator fix.** VE-C fixed the overestimation: on held-out random patch, the share of renders
overestimated by > 0.10 fell from 94.6 % to 21.9 %.

**Safety endpoint.** On the 10 primary cells (Test-OOD × 5, E1 × 5), violations were:

| Arm | Violations |
|---|---|
| A3 | 6 |
| C4 conservative | 10 |
| N4 novelty-aware | 9 |
| H4 quality-first hybrid | 9 |
| A4 original | 10 |
| A5 oracle | 9 |

- C4 was significantly worse than A3 in sFNMR in 8 cells, and N4 in 9.
- No candidate was safe, so by the registered rule **A3 was recommended**.
- Even ground-truth visibility was worse than A3 on every E1 cell (A5 sFNMR 7.51–10.32 % vs A3
  4.65–5.76 %).

**Novelty.** The novelty score separated E1 renders from seen-family renders (AUROC 0.91–0.98).

### 9.4 Detector-inclusive behaviour (follow-up / exploratory)

**Setup.** On LFW composites, occluders are inverse-warped into the photograph, then re-detected
and re-aligned.

**Results:**
- The detector lost up to 17–32 % of high-severity random-patch faces.
- Treating those failures as forced non-matches raised sFNMR to 14.7–16.1 %.
- A detector-free fixed-crop fallback, selected on View 1 DevTest, kept every family within
  target on View 2 (worst sFNMR 3.16 %).
- View 2 had been observed before, so this is follow-up evidence.

## 10. Statistical analysis

**Intervals.** All rates use delete-one-identity jackknife intervals, because pairs sharing
identities are dependent.

**Paired tests.** Paired arm comparisons use the jackknife of the per-identity difference, with
Holm correction within each family of tests:
- 5 folds (Phase 3);
- 10 primary cells per error type (Phase 4 and Phase 6).

**LTT validity:**
- *Assumptions.* It requires exchangeable, independent calibration pairs, rules fixed before
  calibration and a pre-specified candidate order. These hold by construction on
  `p3_cal_independent`.
- *What is not guaranteed.* Nothing in the LTT guarantee covers unseen occluders, other
  populations, or sub-regions of a stratum.
- *How empirical rates are read.* Rates on held-out families and populations are therefore
  reported as empirical rates with intervals, never as guarantees.
- *v2 ordering.* The v2 ordering buys about 40 pp of coverage over v1 (Phase 3), at the price of
  rules closer to the target that break more easily under shift.

**Simulation (SIM-013).** Overestimated visibility (a bias of +0.2) broke per-stratum validity
in 20 % of replicates, while an unbiased predictor kept validity. This is the mechanism behind
RQ1.

## 11. Negative findings and failure mechanisms

1. **Learned visibility learns occluder classes, not "non-face".** Development diagnostics on
   `p3_val`:
   - The fold that never saw random patches called 88 % of truly covered face cells visible.
   - Those errors were confident: mean P(visible) 0.85, with 67 % above 0.9.
   - The misses were the same inside and outside the zones where seen occluders sit, so the
     cause is not a location prior.
   - The model called any uniform fill visible, while models trained with patches flagged the
     same fills.
2. **Face-like occluders are undetectable from appearance.** Every model called another
   identity's face patch visible (0.35–1.00).
3. **Correcting the estimator does not rescue the decision.** Even ground-truth visibility
   strata (A5) were worse than quality strata on the never-seen occluder. Quality, the
   backbone's own uncertainty, already captures most of the decision-relevant information, and
   adding visibility adds variance and failure modes.
4. **Co-visibility hurts under shift.** It was harmful in log-loss on all E1 cells and on 3/5
   Test-OOD cells (Phase 4).
5. **The low-quality tail** (Phase 6 audit, development data; exploratory).
   - *Cause.* The frozen M1 logistic has a large negative main effect of q. Below the
     calibration support (q below the 1st percentile of calibration quality, about 1 % of
     pairs), its logit extrapolates towards p = 1 even for impostor-like similarity:
     fold F1, s = 0.0 and q = 5 gives p = 0.93.
   - *Non-face inputs.* Every synthetic non-face probe pair (noise, constant, gradients; q 5–10)
     was decided **MATCH** with p ≈ 1.000.
   - *Real development pairs.* On `p3_val`, A3's sFMR in that tail was 18–42 % per fold,
     against 1.1–1.5 % elsewhere.
   - *Why aggregates miss it.* Aggregate rates stayed within target because the tail is small.
     LTT controls the stratum-level rate, not the rate within the tail of a stratum.
   - *Figures:* `reports/figures/phase6_m1_extrapolation.svg` and `phase6_lowq_tail.svg`.
   - *Mitigation.* The released application abstains below the support floor. The frozen system
     is unchanged and the finding is reported.

## 12. Population-shift analysis

**Phase 4 finding.** The untouched partition `p4_test` (72-image identities only) had
markedly harder genuine pairs than the calibration-matched `p3_test`:
- clean FNMR of raw similarity was 4.81 % vs 1.12 %;
- every arm, including A3 and the oracle, violated the sFNMR target **even on seen families**:

| A3 on `p4_test` Test-ID, seen families (cov / sFNMR / sFMR, %) | |
|---|---|
| mask fold | 94.0 / 5.89 ✗ / 1.27 |
| sunglasses fold | 89.7 / 6.34 ✗ / 1.67 |
| hand/object fold | 90.2 / 6.16 ✗ / 1.59 |
| random patch fold | 93.2 / 5.36 ✗ / 1.63 |
| lighting fold | 89.4 / 6.05 ✗ / 2.00 |

**Explanation (post hoc).** The calibration set is 98 % 5-image identities (9,033 of 9,219).
The 72-image identities were rendered with more accessory and pose variation, and were already
measurably harder in Phase 2 (AUC 0.99274 vs 0.99869).
- The thresholds certified on the calibration mixture are therefore too permissive for the
  harder sub-population. This is exactly the exchangeability failure that LTT cannot cover.
- The false-match side stayed within target (sFMR 1.03–2.59 %): the shift is asymmetric.

### 12.1 Pre-registered replication (Phase 6)

**Design** (registered before download; `docs/phase6_protocol.md`):
- 4,000 never-downloaded 72-image identities (DigiFace parts 6000–9999), the only remaining
  untouched data;
- the frozen Phase 3/4 systems and the unchanged code path, with a new seed;
- one logged run, `1c291322aa27`.

| A3 on `p6_test` (cov / sFNMR / sFMR, %) | Test-ID (seen) | Test-OOD (held-out) | E1 (never seen) |
|---|---|---|---|
| mask fold | 93.8 / 5.90 ✗ / 1.40 | 90.9 / 7.74 ✗ / 2.77 | 93.0 / 5.97 ✗ / 1.20 |
| sunglasses fold | 89.8 / 6.49 ✗ / 1.73 | 95.7 / 5.76 ✗ / 1.41 | 90.9 / 5.77 ✗ / 1.22 |
| hand/object fold | 90.0 / 6.41 ✗ / 1.77 | 92.3 / 5.72 ✗ / 1.20 | 90.5 / 5.37 ✗ / 1.22 |
| random patch fold | 93.0 / 5.15 † / 1.75 | 85.4 / 7.20 ✗ / 2.03 | 90.9 / 4.83 / 1.26 |
| lighting fold | 89.6 / 6.39 ✗ / 2.02 | 94.5 / 4.26 / 1.47 | 90.6 / 5.13 † / 1.42 |

† = point estimate above the target, CI not excluding it.

**Registered endpoints:**
- **R1 replicates.** The sFNMR interval excluded 5 % in 4 of 5 seen-family cells; the
  pre-registered criterion was ≥ 3.
- **R2 replicates.** No candidate was safe (violations: A3 7, C4 10, N4 10, H4 9; C4 / N4 / H4
  significantly worse than A3 in 8 / 9 / 6 cells for sFNMR). The registered rule again recommends
  A3.
- **R3 holds.** A3 sFMR was within target in all 15 cells (1.20–2.77 %).

**Descriptive.**
- Clean FNMR of raw similarity was 4.60 %, against 4.81 % on `p4_test` and 1.12 % on the
  calibration-matched `p3_test`.
- An exploratory tally registered before the run reproduced the low-quality-tail failure of §11
  on untouched data: below the support floor (0.9 % of pairs), A3 sFMR was 22.97 %
  (289 / 1,258 accepted impostors), against 1.50 % above it.

Figure: `reports/figures/phase6_a3_population_shift.svg` (A3 sFNMR with 95 % CIs across the
three populations); `phase6_lowq_tail.svg`; `phase6_m1_extrapolation.svg`.

## 13. Application architecture

The local demonstration, built with Gradio (Abid et al., 2019), implements the frozen system
exactly. The pipeline runs:
1. input validation: format, size, decodability;
2. SCRFD detection with 5-point alignment:
   - multiple faces are refused unless the user selects one;
   - the fixed-crop fallback is used only for declared LFW-style, near-square portraits (D-054);
3. AdaFace embedding (*s*, *q*) and M1;
4. the A3 rule of the selected frozen fold system, plus the low-quality support guard (D-055).

ArcFace similarity, VE-C visibility maps and novelty are computed *after* the decision and shown
as explanations only. Tests confirm that they cannot change it.

**Interface.** The interface shows:
- the active experimental configuration, and what it was calibrated on;
- the fused score, labelled "not a probability of identity";
- the rule's region, and the recorded abstention reason;
- detector status;
- calibration and domain-shift limitations.

Each comparison requires an authorised-use confirmation and a calibration-limitation
acknowledgement.

**Privacy and security:**
- the server listens on loopback only, with a strict content-security policy, so the browser
  makes no third-party requests (verified);
- no telemetry;
- uploads are deleted after reading; embeddings are never written to disk;
- frozen artefacts are SHA-256 verified at load.

**Parity and speed:**
- *Parity with the offline evaluation:* 1000/1000 A3 decisions; LFW crops pixel-identical;
  CPU/GPU decisions identical on 100/100 pairs.
- *Latency* (RTX 4060 Laptop): 0.23 s per photo pair, dominated by CPU face detection; 45 ms per
  pre-aligned pair.

## 14. Limitations

1. **All occluders are synthetic.** No naturally, physically occluded photographs were evaluated.
   Physical-occlusion validation (e.g. IJB-C) is future work, and access has not been requested.
2. **Synthetic faces.** Confirmatory data are synthetic DigiFace faces. LFW adds real faces with
   synthetic overlays. Its identities likely overlap the backbones' training data, so LFW results
   are optimistic.
3. **Calibration-population guarantees only.** No out-of-distribution, population-shift or
   sub-region guarantee exists. The 5 % target was not maintained on the 72-image population.
4. **Five fold-specific systems.** There is no universal calibration. Each system saw four
   occluder families and a mostly 5-image-identity population.
5. **The frozen fused score is unreliable in the low-quality tail.** Section 11 describes the
   failure; the application mitigates it, but the research system is unchanged.
6. **Single backbones and a single detector.** Detector-inclusive results are exploratory
   (300 pairs).
7. **Follow-up re-use.** Several sets were observed more than once. These results are labelled
   follow-up, not confirmatory.
8. **No deployment-readiness claim.** Nothing here supports operational use for identification,
   access control or any decision affecting people.

## 15. Conclusion

**Main result.** Quality-stratified LTT calibration (A3) is the best-supported method we tested
for selective face verification under unseen synthetic occlusion.

**What failed:**
- Learned visibility, an intuitive improvement, failed for a measurable reason: it learns the
  occluders it was shown. Correcting its bias did not make visibility-stratified decisions safer
  than quality stratification.
- More fundamentally, distribution-free guarantees are guarantees *about the calibration
  population*. A shift to a harder sub-population broke the 5 % target for every rule
  (replicated on 4,000 untouched identities).
- The frozen fused score also fails silently on inputs below its quality support.

**What this implies for practice:**
- calibration from the deployment population and preprocessing;
- explicit input-support checks;
- honest abstention.

## 16. References

- Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J. (2019). Gradio: Hassle-free
  sharing and testing of ML models in the wild. arXiv:1906.02569.
- Angelopoulos, A. N., Bates, S. (2021). A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. arXiv:2107.07511.
- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., Lei, L. (2021). Learn then Test:
  Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
- Bae, G., de La Gorce, M., Baltrušaitis, T., Hewitt, C., Chen, D., Valentin, J., Cipolla, R.,
  Shen, J. (2023). DigiFace-1M: 1 million digital face images for face recognition. IEEE/CVF
  Winter Conference on Applications of Computer Vision (WACV).
- Bates, S., Angelopoulos, A., Lei, L., Malik, J., Jordan, M. I. (2021). Distribution-free,
  risk-controlling prediction sets. Journal of the ACM, 68(6).
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE Transactions on
  Information Theory, 16(1), 41–46.
- Clopper, C. J., Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in
  the case of the binomial. Biometrika, 26(4), 404–413.
- Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep
  face recognition. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Erakın, M. E., Demir, U., Ekenel, H. K. (2021). On recognizing occluded faces in the wild.
  International Conference of the Biometrics Special Interest Group (BIOSIG).
- Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. Advances in
  Neural Information Processing Systems (NeurIPS).
- Guo, J., Deng, J., Lattas, A., Zafeiriou, S. (2022). Sample and computation redistribution for
  efficient face detection. International Conference on Learning Representations (ICLR).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal
  of Statistics, 6(2), 65–70.
- Huang, G. B., Ramesh, M., Berg, T., Learned-Miller, E. (2007). Labeled Faces in the Wild: A
  database for studying face recognition in unconstrained environments. Technical Report 07-49,
  University of Massachusetts, Amherst.
- Kim, M., Jain, A. K., Liu, X. (2022). AdaFace: Quality adaptive margin for face recognition.
  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Lee, K., Lee, K., Lee, H., Shin, J. (2018). A simple unified framework for detecting
  out-of-distribution samples and adversarial attacks. Advances in Neural Information Processing
  Systems (NeurIPS).
- Maze, B., Adams, J., Duncan, J. A., Kalka, N., Miller, T., Otto, C., Jain, A. K., Niggel, W. T.,
  Anderson, J., Cheney, J., Grother, P. (2018). IARPA Janus Benchmark – C: Face dataset and
  protocol. International Conference on Biometrics (ICB).
- Meng, Q., Zhao, S., Huang, Z., Zhou, F. (2021). MagFace: A universal representation for face
  recognition and quality assessment. IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR).
- Ngan, M., Grother, P., Hanaoka, K. (2020). Ongoing Face Recognition Vendor Test (FRVT) Part 6A:
  Face recognition accuracy with masks using pre-COVID-19 algorithms. NISTIR 8311, National
  Institute of Standards and Technology.
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., Lawrence, N. D. (Eds.) (2009). Dataset
  Shift in Machine Learning. MIT Press.
- Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). Failing loudly: An empirical study of methods
  for detecting dataset shift. Advances in Neural Information Processing Systems (NeurIPS).
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical
  image segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI).
- Song, L., Gong, D., Li, Z., Liu, C., Liu, W. (2019). Occlusion robust face recognition based on
  mask learning with pairwise differential Siamese network. IEEE/CVF International Conference on
  Computer Vision (ICCV).
- Terhörst, P., Kolf, J. N., Damer, N., Kirchbuchner, F., Kuijper, A. (2020). SER-FIQ:
  Unsupervised estimation of face image quality based on stochastic embedding robustness. IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (CVPR).
- Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A. (2019). Conformal prediction under
  covariate shift. Advances in Neural Information Processing Systems (NeurIPS).
- Traub, J., Bungert, T. J., Lüth, C. T., Baumgartner, M., Maier-Hein, K. H., Maier-Hein, L.,
  Jaeger, P. F. (2024). Overcoming common flaws in the evaluation of selective classification
  systems. Advances in Neural Information Processing Systems (NeurIPS).
- Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
- Zhu, Z., Huang, G., Deng, J., Ye, Y., Huang, J., Chen, X., Zhu, J., Yang, T., Lu, J., Du, D.,
  Zhou, J. (2021). WebFace260M: A benchmark unveiling the power of million-scale deep face
  recognition. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

**Software and model releases** (not papers):
- CVLface (github.com/mk-minchul/CVLface), source of the AdaFace / ArcFace IR-101 WebFace4M
  checkpoints;
- InsightFace (github.com/deepinsight/insightface), source of SCRFD `buffalo_l`.

*Citation check, Phase 6.* Authors, titles, venues and years are listed only where we are
confident of them, and no DOIs are given. §16 lists every work cited in the text, and nothing
else.
