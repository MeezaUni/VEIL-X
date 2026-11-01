# Phase 3 Results: Learned Visibility, Co-visibility Fusion and Occlusion-Aware Calibration

Protocol: `docs/phase3_protocol.md` (v0.5.0, frozen before training, D-046). Development freeze:
D-047, config SHA-256 `34409779ae8f…`. Every number below comes from a committed aggregate
report; full tables are in `reports/experiments/phase3_tables.md`
(`scripts/summarize_phase3.py`). DigiFace faces and every occluder are **synthetic**; LFW faces
are real photographs with synthetic overlays. Nothing here is evidence about physical
occlusions.

## 0. Summary

| Question | Registered verdict | Result |
|---|---|---|
| **C3** (primary): does learned-visibility stratification (A4) reduce risk-control failures on unseen families vs quality stratification (A3)? | (a) fewer violations, (b) no fold significantly worse, (c) coverage loss ≥ −5 pp | **Not supported (negative).** A4 had 1 violation and A3 had 0. A4 was significantly worse on 3/5 folds and gained only +3.0 pp mean coverage |
| **C1** (secondary): does co-visibility add value beyond similarity + quality? | ≥ 6/10 cells significantly positive **and** random-patch OOD positive | **Not supported.** 7/10 cells positive, but on random-patch OOD co-visibility significantly **hurt** (Δ log-loss −0.035) |
| **H4** (constraint): clean FNMR margin ≤ 0.5 pp | upper CI of ΔFNMR(M4 − M0) < +0.5 pp | **Met in 5/5 folds.** M4 clean FNMR 0.68–0.74 % vs M0 1.12 %; all CIs below 0 |

**Why C3 failed (mechanism, measured).** The learned visibility estimator generalises to held-out
masks, sunglasses and hands (IoU 0.87–0.90). It fails almost completely on held-out **random
patches** (IoU 0.13, global MAE 0.34) and **extreme lighting** (IoU 0.005, MAE 0.49): it calls
unseen occluders "visible". Pairs then land in the high-visibility stratum, whose rule is loose.
On random-patch OOD the sFNMR rose to 8.5 % (target 5 %) while coverage *rose* to 98.5 %. This
is exactly the failure that SIM-013 isolates in simulation (upward-biased predicted strata).
Quality stratification needs no visibility model and stayed valid (2.1 %, 88.6 % coverage). The
oracle (A5) also stayed valid (2.8 %, 90.4 %). The idea of visibility stratification is sound;
the learned visibility estimate is the weak link.

## 1. Runs and evidence labels

| Run | Script | Data | Label |
|---|---|---|---|
| VE training ×5 | `train_visibility.py` | p3_train / p3_val | development |
| develop + freeze `34409779…` | `evaluate_phase3.py develop` | p3_train / val / cal | development |
| rehearsal | `evaluate_phase3.py final --rehearse` | p3_val | code check, not a result |
| **`bb5d9244bb3c`** | `evaluate_phase3.py final --final` | **p3_test (once)** | **confirmatory** |
| gate 3B | `evaluate_visibility.py` | p3_test renders (map accuracy only) | confirmatory (VE) |
| follow-up | `evaluate_phase3.py followup --final` | Phase 2 DigiFace test; LFW View 2 | **follow-up** (§8) |
| detector-inclusive | `lfw_detector_inclusive.py --final` | 300 LFW View 2 pairs | **exploratory** (§9) |
| SIM-013 | `run_simulations.py --only noisy_strata_validity` | synthetic | simulation |

Test accesses are logged in `results/test_runs.jsonl`. Gate 3B (map accuracy only) touched
p3_test renders three times:
1. stopped by our own memory monitor before any metric was written;
2. a complete run with a lower-memory loader;
3. an identical rerun on the same seeded sample to add the signed visibility bias. Every other
   number was unchanged.

The detector-inclusive study also has one logged 12-pair diagnostic (`diag-detinc-1`, visibility
aggregates only).

## 2. Visibility estimator (gate 3B)

**Architecture and training:** `docs/model_card_visibility.md`. VisNet U-Net-lite, 3.6 M
parameters, 28×28 output, five LOFO fold models, fp16 AMP, early stopping on p3_val.
F4 and F5 hit the 30-epoch cap still improving.

**Accuracy on p3_test renders** (≤ 20,000 renders per cell; face cells of the 28×28 grid):

| Fold | Learned IoU, seen / **held-out** | Learned global MAE, seen / **held-out** | Heuristic IoU, seen / held-out |
|---|---|---|---|
| mask | 0.896 / **0.898** | 0.016 / **0.053** | 0.159 / 0.309 |
| sunglasses | 0.915 / **0.865** | 0.011 / **0.012** | 0.277 / 0.000 |
| hand/object | 0.917 / **0.886** | 0.011 / **0.019** | 0.209 / 0.215 |
| random patch | 0.946 / **0.130** | 0.005 / **0.344** | 0.143 / 0.274 |
| extreme lighting | 0.973 / **0.005** | 0.001 / **0.492** | 0.183 / 0.118 |

- **Direction of the held-out errors** (signed global bias, predicted − true):

  | Held-out family | Bias | Bias as a share of MAE |
  |---|---|---|
  | random patch | **+0.344** | 100 % |
  | lighting | **+0.491** | 100 % |
  | mask | +0.053 | — |

  The estimator **overestimates** visibility on unseen occluders. That is the harmful direction
  identified by SIM-013.
- **Clean test images:** learned global MAE ≤ 0.010, false-occlusion cell rate ≤ 0.6 %.
  Heuristic: MAE 0.15–0.22, false-occlusion rate 2.7–5.0 %.
- **Calibration** (seen families): Brier 0.009–0.018, ECE 0.014–0.021.
- **Held-out** random patch / lighting: ECE 0.34 / 0.49 (badly overconfident).
- The **appearance-anomaly heuristic** is far weaker than the learned estimator everywhere it
  matters. Its only edge is on held-out random patch / lighting, where the learned model
  collapses.
- The **oracle** maps carry only grid-discretisation error vs the engine's pixel ratio
  (global MAE ≤ 0.002).

## 3. Co-visibility (gate 3C)

- **Selection rule (registered; p3_val; learned maps):** `cv_region` (error-prediction AUROC
  0.701) > `cv_spatial` 0.695 > `cv_min` 0.676 > `cv_prod` 0.666. All four were computed for
  heuristic, learned and oracle maps; none was selected on test.
- **Value beyond quality (C1):**
  - On seen families, co-visibility adds a small, significant log-loss gain in all 5 folds
    (M1 − M4 = +0.0008 to +0.0090; Holm p ≤ 0.007).
  - On held-out families the gain holds for mask (+0.0051) and sunglasses (+0.0042). It is null
    for hand/object (−0.0004, p 0.30).
  - It is **harmful** for random patch (−0.0346) and lighting (−0.0098): wrong visibility is
    worse than none.

## 4. Fusion (gate 3D)

Occluded-pair AUC, **held-out family** (Test-OOD):

| Model | mask | sunglasses | hand/object | random patch | lighting |
|---|---|---|---|---|---|
| M0 similarity | 0.9833 | 0.9918 | 0.9851 | 0.9605 | 0.9889 |
| M1 + quality | 0.9893 | 0.9969 | 0.9950 | 0.9833 | 0.9975 |
| M4 + quality + co-visibility | 0.9902 | 0.9972 | 0.9952 | 0.9828 | 0.9971 |
| M5 full (+ArcFace, disagreement) | **0.9914** | **0.9976** | **0.9960** | **0.9860** | 0.9973 |
| M4-oracle | 0.9902 | 0.9972 | 0.9953 | 0.9835 | 0.9962 |

- **Feature norm (quality) carries most of the gain** over raw similarity (M0 → M1).
- Visibility without quality (M2, M3) is clearly worse than M1.
- **Dual-backbone fusion (M5)** is best almost everywhere.
- The ground-truth-visibility M4-oracle is **not** better than learned M4 on seen-family data,
  and only marginally better on held-out random patch. Even perfect visibility adds little
  *discrimination* beyond quality. Its value is in *stratification* (§6).
- MLP fusion met the > 5 % val log-loss criterion in 4/5 folds. It is exploratory and was not
  evaluated on test (D-047).

## 5. Three-way decisions and calibration arms (gates 3E–3F)

Test-OOD, primary target (sFNMR ≤ 5 %, sFMR ≤ 5 %), v2 ordering. Cells are coverage / sFNMR /
sFMR in %, with ✗ marking a violation (the 95 % identity-jackknife CI excludes the target):

| Arm | mask | sunglasses | hand/object | random patch | lighting |
|---|---|---|---|---|---|
| A1 clean-only | 100 / 9.47 ✗ / 1.54 | 100 / 6.45 ✗ / 1.06 | 100 / 8.94 ✗ / 1.02 | 100 / 19.81 ✗ / 0.86 | 100 / 5.97 ✗ / 0.99 |
| A2 global (M0) | 94.4 / 2.74 / 4.72 | 92.4 / 1.24 / 2.38 | 95.2 / 2.90 / 3.68 | 95.6 / 11.27 ✗ / 1.67 | 93.1 / 1.69 / 2.66 |
| A3 quality-stratified | 93.7 / 1.94 / 2.56 | 96.8 / 1.87 / 1.24 | 94.3 / 1.59 / 1.23 | 88.6 / 2.12 / 2.00 | 96.1 / 1.08 / 1.27 |
| **A4 predicted-visibility** | 94.8 / 1.75 / 3.14 | 96.8 / 1.28 / 0.89 | 95.4 / 2.36 / 1.22 | 98.5 / **8.52 ✗** / 2.13 | 98.7 / 2.25 / 1.28 |
| A5 GT-oracle | 92.4 / 1.30 / 2.12 | 98.1 / 2.06 / 1.34 | 97.5 / 2.01 / 2.30 | 90.4 / 2.76 / 1.93 | 93.4 / 0.87 / 1.63 |
| A6 quality × visibility | 91.4 / 1.15 / 2.50 | 95.9 / 1.38 / 0.98 | 93.2 / 1.32 / 1.00 | 95.8 / 6.11 ✗ / 1.92 | 97.4 / 1.72 / 0.84 |

- **Clean-only calibration fails everywhere under occlusion** (sFNMR 6–20 %). Occlusion-aware
  calibration is necessary.
- The **global** occlusion-aware arm reproduces the Phase 2 held-out random-patch failure
  (11.3 %).
- **C3 paired comparison** (A4 − A3, Test-OOD sFNMR, Holm over 5 folds):

  | Fold | Δ (pp) | p |
  |---|---|---|
  | mask | −0.19 | < 0.001 |
  | sunglasses | −0.59 | < 0.001 |
  | hand/object | +0.78 | < 0.001 |
  | random patch | **+6.40** | < 0.001 |
  | lighting | +1.17 | < 0.001 |

  A4 is better where the estimator generalises and worse where it does not.
- **Severity** (random patch OOD, A4 sFNMR): clean 1.2 % → low 2.1 % → medium 7.1 % →
  **high 17.5 %** at 96.6 % coverage. A3 at high severity: 4.4 % at 76.5 % coverage.
- **Abstention reasons** for A4 are recorded per pair and are not causal. Every abstention lay in
  a certified rule's uncertainty band (no stratum was uncertifiable), and 66–100 % coincided with
  low predicted co-visibility.
- **Stricter target** (far1: sFMR ≤ 1 %, sFNMR ≤ 5 %): A4 violates on random patch (12.8 %);
  A3 and A5 stay valid, at a large coverage cost on random patch (23 % and 48 %).
- **v1 vs v2 ordering:**

  | Ordering | Mean Test-OOD coverage, primary | Violations (✗) |
  |---|---|---|
  | v2 | 93–97 % | 1 each for A2, A4, A6 |
  | v1 | 49–76 % | 0 except A6 |

  v2 (the registered default, D-042) buys about 40 pp of coverage. Its rules sit closer to the
  target, so they break more easily under unseen shift. **No arm has a formal guarantee
  out-of-distribution**; these are empirical rates.

## 6. Ground-truth vs predicted visibility

| | A5 oracle | A4 learned |
|---|---|---|
| Violations on Test-OOD (both targets, both error types) | 0 | 1 primary + 1 far1 |
| Mean coverage | 94.4 % | 96.9 % |

The difference is entirely the estimator's held-out error: with correct visibility,
stratification works. The oracle is **not deployable** (it uses the engine's own mask). SIM-013
reproduces the same pattern in simulation:
- validity holds with an unbiased predictor;
- the per-stratum guarantee breaks in 20 % of replicates when visibility is overestimated by 0.2.

## 7. Clean margin (H4)

At their own frozen clean-cal FAR-1 % thresholds, M4 vs M0 clean FNMR on p3_test is
0.68–0.74 % vs 1.12 % (Δ −0.38 to −0.44 pp, CI upper ≤ −0.30 pp), with FMR 1.07–1.17 % vs 1.19 %.
The fused model is **better** on clean pairs, so the 0.5 pp margin is met with room to spare
(SIM-011 paired jackknife).

## 8. Follow-up sets (gate 3G)

Run `9b660f16061a` (logged) used the frozen Phase 3 models and rules with no fitting of any kind.
Both sets were observed in Phase 2, so this is **follow-up** evidence. LFW View 2 is being used
for the third time.

**Phase 2 DigiFace test** (held-out-family plans; 122,244 pairs per fold; identities disjoint
from every Phase 3 sub-split). The confirmatory picture replicates almost exactly:

| Arm | random patch: cov / sFNMR / sFMR, % |
|---|---|
| A4 | 98.5 / **8.90 ✗** / 2.07 |
| A3 | 88.4 / 2.18 / 1.88 |
| A5 | 90.2 / 2.77 / 1.85 |

- C3 Δ (A4 − A3) is −0.14 / −0.58 pp on mask / sunglasses and +0.77 / +6.72 / +1.25 pp on
  hand / patch / lighting.
- C1 has the same signs as on p3_test.
- Clean FNMR: M0 1.31 % → M4 0.76–0.87 %.

**LFW View 2 derived occlusions** (real photographs with synthetic overlays; DigiFace-trained
models and DigiFace-calibrated rules applied without adaptation):

| Arm | random patch: cov / sFNMR / sFMR, % |
|---|---|
| A4 | 99.9 / **8.30 ✗** / 0.16 |
| A3 | 94.6 / 2.26 / 0.10 |
| A5 | 95.0 / 2.20 / 0.13 |

- The other four families are well within target for every occlusion-aware arm. The DigiFace
  rules are conservative on LFW, as in Phase 2, with sFMR ≤ 0.2 %.
- **Co-visibility gives no log-loss benefit on LFW**: M1 − M4 < 0 in all 5 families, from
  −0.0006 to −0.048. The synthetic-to-real transfer of the visibility *features* is therefore
  negative, while the stratification failure on random patch transfers unchanged.
- Clean View-2 FNMR at the frozen thresholds is 0.44 % for both M0 and M4. These clean
  thresholds are far from the LFW impostor distribution, so H4 is uninformative there.

## 9. Detector-inclusive LFW (exploratory)

Run `6a3c151e18a5` (logged; 300 View-2 pairs, 150 genuine / 150 impostor, × 5 families × 3
levels). The setup:
- the same occluder as the crop-level study is inverse-warped into the original photograph;
- SCRFD then re-detects the face and it is re-aligned;
- the frozen fold rules are applied, with side b occluded.

| Family | Detection failures, low / med / high | Mean landmark shift (px of 112) | A4 cov / sFNMR / sFMR, % | A3 cov / sFNMR / sFMR, % |
|---|---|---|---|---|
| mask | 0 / 0.3 / 1.0 % | 3.2 | 99.8 / 1.12 / 0.00 | 99.4 / 0.22 / 0.00 |
| sunglasses | 0 / 0 / 0 % | 0.9 | 99.9 / 0.89 / 0.00 | 100 / 0.67 / 0.00 |
| hand/object | 0 / 0 / 3.7 % | 2.0 | 100 / 1.13 / 0.00 | 99.6 / 0.23 / 0.00 |
| random patch | 0 / 1.3 / **17.3 %** | 4.3 | 100 / **8.49 ✗** / 0.00 | 96.3 / 3.53 / 0.00 |
| lighting | 0 / 2.0 / 7.7 % | 1.6 | 100 / 1.38 / 0.00 | 99.5 / 0.46 / 0.00 |

- **Detection is itself a failure mode.** At high severity the detector loses up to 17 % of
  random-patch faces.
- **Handling detection failures:**
  - Treated as abstentions, they cost genuine coverage. For A4 on random patch: 94.2 % coverage,
    sFNMR 8.5 %.
  - Treated as forced non-matches, they raise sFNMR: A4 random patch 13.8 %, hand/object 2.9 %,
    lighting 4.7 %.
  - The policy for detection failures therefore matters as much as calibration.
- **The visibility estimator degrades after re-alignment.** A diagnostic on 12 pairs (logged
  `diag-detinc-1`, aggregates only) checked two things:
  - The ground-truth warp is correct: re-aligned vs crop-level visible fraction differ by
    0.035 (mask) and 0.005 (sunglasses).
  - The learned estimator's global error rises from 0.066 to 0.204 (mask) and from 0.003 to
    0.101 (sunglasses), always as **overestimation**.

  The occluder is resampled twice: into the photograph, then by re-alignment. That softening is
  enough to move the estimator toward "visible". Over the full study, the learned-vs-re-aligned
  GT error is 0.10–0.53 across families.
- The **F5 lighting composite** re-lights only the crop footprint, so its edge is visible in the
  photograph. It is a weaker stand-in for real illumination change.
- Exploratory only: 300 pairs, one detector, synthetic occluders. sFMR stays 0 because the
  DigiFace rules are conservative on LFW.

## 10. Validity and limitations

- **Synthetic everything.** Faces (DigiFace) and occluders (engine) are synthetic. LFW adds real
  faces with synthetic overlays. Physical occluders are untested
  (`docs/physical_occlusion_readiness.md`).
- **Held-out families are few (5)** and differ in kind:
  - three are localised objects, where the estimator transfers;
  - two are a texture-free patch and a global photometric change, where it does not.

  Generalisation to an arbitrary new occluder cannot be inferred from 5 folds.
- **No formal OOD guarantee.** LTT certificates hold for the calibration distribution only. All
  OOD rates here are empirical, with identity-level jackknife intervals.
- **One confirmatory look.** p3_test was used once for decisions (`bb5d9244bb3c`); map-accuracy
  passes (§1) never touch verification scores or rules. Phase 2 test and LFW View 2 are re-used (follow-up), never for fitting.
- **Estimator budget.** The 30-epoch cap was reached by F4/F5 (still improving), and validation
  loss was noisy. A longer or more diverse training budget might change the held-out picture.
  It was fixed in advance and not tuned on held-out results.
- **Multiple arms.** The confirmatory claims are C3 (5 tests, Holm) and C1 (10 tests, Holm).
  Arm-by-arm, ordering and target tables are secondary and descriptive.

## 11. Recommendations for Phase 4

1. **Make the visibility estimator robust to unseen occluder types before using it for
   stratification:**
   - train on a much wider occluder distribution (procedural textures, photometric
     perturbations);
   - add an uncertainty/OOD signal, such as ensemble disagreement or the anomaly heuristic,
     and route uncertain maps to quality-only strata;
   - prefer **conservative (downward-biased) visibility**, because SIM-013 shows
     overestimation is the harmful direction.
2. **Keep quality-stratified calibration (A3) as the deployable default.** It was the only
   deployable arm with no violations on any held-out family, at either target.
3. **Use dual-backbone fusion (M5)**; evaluate MLP fusion prospectively (criterion met in 4/5
   folds).
4. **Physical-occlusion evaluation**, gated on the authorisations in
   `docs/physical_occlusion_readiness.md`.
5. **In-domain calibration on real photographs** (pending decision 7).
