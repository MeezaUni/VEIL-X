# Phase 2 Statistical Audit

> Labels: **MEASURED** (computed from real model scores on DigiFace), **THEORETICAL** /
> **CALCULATED** (exact mathematics on real counts), **SIMULATED** (synthetic score models,
> Phase 1a/1b), **ESTIMATED** (derived from measured quantities under stated assumptions).
> Data: DigiFace-1M (synthetic faces) with synthetic occlusions. **No real-world claim.**

## 1. Units of dependence and interval methods

| Pair set | Dependence | Unit of uncertainty | Method (protocol `bootstrap`) |
|---|---|---|---|
| `cal_independent` | Identity-disjoint: each identity in at most one pair; one root photo per pair | Pair (i.i.d. under identity sampling) | Exact binomial (LTT p-values) |
| `val`, `test_balanced`, `test_far` | Clustered: an identity appears in up to about 10 pairs (genuine cap 3 plus impostor roles) | Identity | Delete-one-identity jackknife: logit scale for genuine-pair rates, linear for impostor-pair rates (SIM-010 coverage 0.948–0.975) |
| Occluded versions of the same pairs | Same clustering; conditions are seeded per pair (A7) | Identity | Same as above |
| Paired clean-vs-occluded differences | Positively correlated within pair | — | **No validated interval**; only point differences are reported |
| Zero-event rates | Jackknife degenerate | Pair (flagged) | Clopper–Pearson upper bound treating pairs as independent: **anti-conservative** for clustered sets, labelled as such |

The jackknife deletes an identity together with **every** pair it appears in (as either side).
For impostor pairs SIM-010 found it conservative, with variance over-estimated 1.6–1.9×.

## 2. Calibration validity (THEORETICAL / CALCULATED)

- LTT validity requires A1 (exchangeability), A2 (independence), A6 (pre-specified λ and order)
  and A7 (acceptance locality); see `docs/ltt_audit.md`.
- A2 holds by construction. A7 holds because rule parameters, strata edges and candidate
  orders come only from `val`.
- A1 holds for **Test-ID** (same seen-family plan generator and the same identity population).
  It is **violated by design** for **Test-OOD**: the held-out family never appears in calibration.
- Stratified calibration uses δ/3 per stratum (Bonferroni), which gives simultaneous control.
- Capacity, at full coverage, is an upper bound (CALC-003, `reports/data/capacity_v0.4.0.json`):

| Target | Global (δ = 0.05, n = 3,409 per class) | Per visibility stratum (δ/3, n ≈ 852–1,278) |
|---|---|---|
| 5 % | ≤ 149 errors certifiable | ≤ 29–47 errors |
| 1 % | ≤ 24 errors | ≤ 2–5 errors |
| 0.1 % | zero errors and ≥ 2,995 accepted | **never certifiable** |

A selective rule accepts only part of each class, so its certifiable bound is worse than the
table. FAR 0.1 % therefore stays **estimation-only** (D-034). The expansion improved the
zero-error bound from 0.25 % to 0.088 % but did not make 0.1 % certifiable in practice.

## 3. Results-dependent audit (single final run `38d853d8884c`, freeze `0e8a43fb…`)

### 3.1 Test-access integrity (MEASURED)
- **One** logged test run (`results/test_runs.jsonl`: run id, frozen-config hash, commit
  `f3376ea`). Every threshold, candidate order, strata edge and LTT rule was frozen on
  val + cal beforehand. The final stage refuses to run if the frozen file's hash changed.
- The one procedural change (D-042) was made **before** test access and only after seeing
  *calibration* feasibility. It is reported as separate v2 arms; the v1 arms are unchanged.
- Rehearsal of the full final code path ran on **validation** data only (private output, no
  run-log entry).

### 3.2 Suspicious-performance audit (MEASURED)

| Check | AdaFace | ArcFace | Verdict |
|---|---|---|---|
| Identity overlap between splits / between test and cal/val pair sets | 0 | 0 | pass |
| Label-shuffle AUC | 0.4999 | 0.5003 | pass (≈ 0.5) |
| Clean AUC exactly 1 | no (0.99856) | no (0.99789) | pass |
| Val − test AUC | 0.00025 | 0.00017 | pass (no test-specific advantage) |
| Per-part AUC (5-img / 72-img identities) | 0.99869 / 0.99274 | 0.99801 / 0.99295 | plausible: the varied 72-img identities are harder |
| Exact duplicate images (pixel hash) | 0 | — | pass (DATA-004) |
| Shuffle / precision / determinism / reference equality of the model | see `docs/model_card_backbones.md` | | pass |

The high clean AUC is plausible rather than suspicious. These are WebFace4M-trained IR-101
backbones on pre-aligned, well-lit synthetic renders, and the 72-image identities with more
variation are clearly harder. Nevertheless, DigiFace clean accuracy **must not** be read as
real-world accuracy (§4).

### 3.3 Is the 5 % accepted-error target testable? (MEASURED / CALCULATED)
Yes, under exchangeability (Test-ID).
- With 3,409 independent pairs per class, the occlusion-calibrated global arm was certifiable
  in 10/10 fold × backbone cells (v1 and v2) at coverage 0.62–1.00.
- Realised Test-ID selective error was within target in every certifiable cell.
- The realised Test-ID sFNMR (0.1–4.0 % across the 76 certifiable cells) sits well below 5 %. The target is therefore
  achievable with substantial coverage on this data. Being achievable is not the same as being
  appropriate: the 5 % value remains a PI decision (provisional).

At the sFMR ≤ 1 % setting:
- the pre-registered ordering certified only 6/10 global fits;
- the p-value ordering (v2) certified 10/10 at coverage 0.80–0.94;
- the oracle-visibility strata at 1 % are often capacity-limited (CALC-003).

### 3.4 Is the 0.5 pp clean non-inferiority margin testable? (ESTIMATED)
On the clean test set, the identity-jackknife 95 % half-width of FNMR at the deployment
threshold is about ±0.11 pp (AdaFace, 1.42 % [1.31, 1.53]) and ±0.14 pp (ArcFace).
- A Phase 3 system scored on the **same** pairs gives a paired difference whose variance is at
  most the sum of the two variances; with positive correlation it is smaller.
- So a 0.5 pp margin is resolvable with this test set: worst-case half-width
  ≈ √2 × 0.14 ≈ 0.2 pp, below 0.5 pp.
- Caveat: intervals for paired differences have **not** been validated by simulation
  (protocol `bootstrap`). Phase 3 must validate them before testing H4.

### 3.5 Calibration validity under shift (MEASURED, synthetic)
- **Exchangeable (Test-ID).** 0 violations in 76 certifiable cells across both targets and
  both orderings. With δ = 0.05 (v1) or the δ ≤ 0.10 union bound (v2), a handful of
  violations would be tolerable; observing none is consistent with validity. Cells share
  calibration sets and test pairs, so they are not independent replications.
- **Held-out family (Test-OOD).** Violations occur only with random patch held out. That
  family is the most severe, covering up to 65 % of the face, and is unlike seen families in
  spatial structure.
  - Global sFNMR there is 8.8–13.2 %, against 5 % (interval excludes the target), in all four
    backbone × ordering cells.
  - Other held-out families stay within target. Their severities lie inside the range spanned
    by the seen families, so the calibration still "covered" them.
- **Clean-only calibration fails under any occlusion** (20/20 cells violate sFNMR at the 5 %
  target, including seen families). Calibration data must contain the shift it is meant to
  control.
- **Oracle-visibility stratification.** v1 removed the F4 violation; v2 reduced it. In both
  cases the mechanism is abstention on, or tighter rules for, low-visibility pairs, at a
  coverage cost.

### 3.6 Stratified capacity (CALCULATED + MEASURED)
- Per-stratum calibration counts: clean ≈ 1,705–1,727 pairs, vis_high ≈ 2,536–2,591,
  vis_low ≈ 2,500–2,565 (both classes together).
- At 5 % every stratum has enough pairs.
- At sFMR 1 % (δ/3), vis_low was **infeasible in most fold × backbone cells** under both
  orderings: high error and few safe rules. Low-visibility pairs therefore abstain, which is
  the honest outcome at this capacity.

### 3.6a Paired-difference intervals for the clean margin: SIM-011 (SIMULATED)

`reports/simulations/paired_difference_coverage.json`: 1,000 replicates; two correlated systems
on the same identity-clustered pairs (shared identity effects plus a system-specific identity
effect and pair noise); true differences from 2 M Monte-Carlo pairs.

| Statistic | Identities | Paired identity jackknife (t) | Naive pair-level normal |
|---|---|---|---|
| ΔFNMR | 500 | **0.942** (MC SE 0.007), width 2.23 pp | 0.931 |
| ΔFNMR | 2,000 | **0.946** (0.007), width 1.11 pp | 0.930 |
| ΔFMR | 500 | 0.996 (conservative), width 1.61 pp | 0.931 |
| ΔFMR | 2,000 | 0.997 (conservative), width 0.80 pp | 0.941 |

- **Verdict.** The paired delete-one-identity jackknife meets the ≥ 0.94 acceptance criterion
  for FNMR differences, though at 500 identities only marginally: the Monte-Carlo interval
  reaches 0.928. It is conservative for FMR differences. The naive pair-level interval
  undercovers and must not be used.
- **Margin testability.** Scaling the width to the real test set (20,374 identities, genuine
  cap 3) gives a ΔFNMR half-width of about 0.18 pp, so a **0.5 pp clean margin is testable**.
  This replaces the ESTIMATED statement of §3.4 with a SIMULATED one.
- **Caveat.** The simulation uses a Gaussian score model, not real-score dependence.

### 3.6b Multiplicity of the ordering amendment: SIM-012 (SIMULATED)

`reports/simulations/adaptive_ordering_validity.json`: δ = 0.10, target 5 %, 1,000 replicates,
calibration sizes 300 / 1,000.

| Procedure | Violation rate, n = 300 / 1,000 | Feasible rate, n = 300 / 1,000 |
|---|---|---|
| v1 (pre-registered order) | 0.003 / 0.012 | 0.792 / 1.000 |
| v2 (p-value order) | 0.000 / 0.000 | 0.867 / 1.000 |
| v1 if feasible, else v2 (the Phase 2 practice) | 0.003 / 0.012 | 0.937 / 1.000 |
| whichever certifies more coverage | 0.003 / 0.012 | 0.937 / 1.000 |

All procedures stay far below δ, and the adaptive choices far below the union bound 2δ. The
union-bound statement used for the v2 arms (δ ≤ 0.10) is therefore **conservative** in this
model. It is kept, because a simulation does not prove the bound for the real score
distribution. The finding supports adopting v2 as the pre-registered default for Phase 3.

### 3.6c Phase C audit: do the Phase 2 findings hold up? (2026-10-27)

| Finding (Phase 2, DigiFace) | Audit | Verdict |
|---|---|---|
| Reproducibility of the final run | Logged rerun `0ca832e3b3b3` with the identical freeze: 0 differing result fields (RES-007) | **Reproduced exactly** |
| Held-out **random patch** breaks risk control | LFW real photographs with derived random patches, scored with the DigiFace rules of the random-patch-held-out fold: global sFNMR 7.46 % (Ada, CI above 5 %) and 5.28 % (Arc, point above); v2 9.96 % / 7.46 % (both CI above) (RES-010) | **Reproduced on real photos** (synthetic overlays) |
| Oracle-visibility stratification restores control at a coverage cost | LFW random patch: v1 stratified 0.35 % / 1.72 % at coverage 0.43 / 0.87 (vs 0.97 / 0.96 global); v2 stratified 4.92 % / 3.25 % at 0.96 / 0.94 | **Reproduced**; v2 strata are within target on LFW (on DigiFace they reduced but did not remove the violation) |
| **Mask**-specific false-match increase | Pre-registered LFW test is uninformative: the DigiFace threshold is conservative on LFW, so FMR is ≈ 0 everywhere. **Post-hoc**, at the in-domain DevTrain threshold, FMR rises 0.27 → 1.68 % (Ada) and 0.57 → 2.93 % (Arc) for both-masked impostors, but also for sunglasses (1.48 / 1.51 %) and random patch (1.28 / 1.31 %) (RES-011) | **Partly reproduced.** The shared-occluder FM increase replicates; its mask *specificity* does not. Post-hoc, so it needs pre-registration |
| Quality ranks errors better than co-visibility; co-visibility adds a residual signal | LFW (exploratory): quality AUROC 0.52–0.92, co-visibility 0.55–0.83, co-visibility within quality deciles 0.46–0.74 | **Reproduced qualitatively**; the residual is family-dependent (strongest for hand/object and random patch) |
| Calibration ordering v2 and its multiplicity statement | SIM-012: the adaptive v1→v2 procedure violates at ≤ 0.012 ≪ δ = 0.10; v2 never violated. On LFW, v2 global violates exactly where v1 does (random patch) | **Union bound conservative.** v2 is not a substitute for stratification under family shift |
| Paired CI for the clean margin | SIM-011: paired identity jackknife covers ΔFNMR at 0.942 / 0.946, conservative for ΔFMR | **Testable** at the real test size (≈ 0.18 pp half-width) |
| Synthetic-calibrated thresholds | LFW: the DigiFace FAR-1 % threshold accepts 0 / 3,000 real impostors (RES-009) | **New finding.** Synthetic calibration is **conservative, not nominal**, on real photos; a deployment threshold needs in-domain calibration |

Explicit non-claims:
- Synthetic overlays on real photographs are **not** equivalent to physical occlusions.
- The v2 arms and the mask audit are **not** pre-registered primary results.
- No guarantee is claimed for any rule transferred from DigiFace to LFW (A1 violated).

### 3.7 Evidence labels used in Phase 2 documents

| Label | Examples |
|---|---|
| MEASURED | All AUC/EER/FMR/FNMR/coverage/selective-risk values; throughput; data counts |
| CALCULATED | Capacity bounds (CALC-003), zero-error sample sizes |
| SIMULATED | Interval-method coverage (SIM-010); p-value-ordering validity/power test (`tests/test_ltt.py`) |
| ESTIMATED | Margin testability (§3.4); C1 within-decile AUROC as evidence for H1 |

## 4. Domain limitations (apply to every Phase 2 number)

1. **Synthetic faces.** DigiFace renders are a different domain from real photographs. Clean
   accuracy of WebFace-trained backbones on DigiFace says nothing about real-image accuracy.
2. **Synthetic occlusions.** F1–F4 are procedural overlays and F5 is a non-physical photometric
   transform. Real occluders (texture, shadows, 3-D geometry, alignment failures) are absent.
3. **No detector.** DigiFace is pre-aligned, so alignment and detection failures, a main
   real-world failure mode under occlusion, are not exercised.
4. **Oracle visibility.** The stratified arm uses ground-truth mask visibility. It is an upper
   bound on what a learned visibility estimator could achieve, not a deployable method.
5. **LFW / real benchmarks blocked** (D-039). No external validity check was possible.
