# Phase 4 Prospective Protocol: Robustness Recovery and Conservative Visibility

> Registered 2026-11-01 at protocol v0.6.0. This is before any Phase 4 method is trained, before
> any Phase 4 rule is calibrated, and before any image of the Phase 4 confirmatory partition is
> scored. Changes after this point are amendments: logged in `docs/decision_log.md` and
> labelled in every result they touch. Phase 3 is accepted as a negative primary result
> (D-046/D-047, `docs/phase3_results.md`). **No positive result will be manufactured: if no
> candidate reliably improves on A3, A3 is recommended.**

## 1. Questions

- **Q1 (primary, safety).** Can a conservative and/or novelty-aware visibility method, or a
  quality-first hybrid, be used for selective verification **without** more risk-control
  failures than the quality-stratified reference A3, on unseen occlusion families and on an
  occluder never seen by any design decision?
- **Q2 (primary, benefit).** If Q1 holds, does it **improve** on A3, i.e. give higher coverage
  at the same targets?
- **Q3 (secondary).**
  - Fusion: does neural fusion beat logistic fusion prospectively?
  - Does co-visibility add value beyond quality?
- **Q4 (secondary).** Detector-inclusive behaviour: failure taxonomy and safe handling.

## 2. Root cause (development-data audit, `scripts/phase4_diagnose.py`)

All of these diagnostics are on **p3_val** identities with fresh renders; no test data was used.
Artefact: `reports/experiments/phase4_diagnostics.json`.

1. **Reproduced.** The fold-F4 estimator on random patch at high severity: IoU 0.12; signed
   global bias +0.54; 88 % of truly occluded face cells predicted visible.
2. **The errors are confident.** On those cells: mean P(visible) 0.85, 67 % > 0.9, mean |logit|
   5.4. Probability or entropy thresholds cannot reliably flag them.
3. **Not a location prior.** Miss rates are equal in the zones where seen occluders sit (eye
   band and lower face: 88.3 %) and elsewhere (88.7 %).
4. **Not colour or texture.** Miss rates are 0.87–0.96 for dark, bright, saturated, mid, flat
   and textured patches.
5. **Controlled probes (24×24 fills).**
   - The fold-F4 model calls any uniform fill visible: black 0.96, white 0.90, grey 1.00,
     saturated 1.00.
   - The four models that saw random patches during training flag the same fills as occluded
     (0.01–0.3).
   - **Every** model calls another identity's face patch visible (0.35–1.00). Face-like
     occluders are undetectable from appearance; this is a fundamental limit.
   - Conclusion: the estimator learned the *occluder classes* it was shown, not
     "non-facial content".
6. **Preprocessing.** Identical code path. F4 inputs differ only mildly (4.0 % vs 2.1 %
   saturated pixels), so there is no preprocessing defect.
7. **Lighting is a corruption, not missing visibility.**
   - The F5 label (crushed, clipped, *or darkened to ≤ 35 %*) weakly tracks recognition damage:
     genuine-score drop correlation −0.31, mean drop 0.039 at mean labelled visibility 0.52.
   - Physical occluders: correlation −0.43 to −0.70; mask mean drop 0.137 at 0.50.
   - Phase 4 therefore treats lighting through **quality**, and as "occluded" only where pixels
     are information-free (crushed ≤ 12 or newly clipped ≥ 250).

**Consequence for validity.** Phase 4 design is informed by the F4/F5 failures, so F4 and F5
results in Phase 4 are labelled **design-informed**. A new **evaluation-only** occluder
`E1_hair_strands` was defined at registration (before any Phase 4 result). It is never rendered
in development, never used for training, selection or calibration, and is the clean unseen test.

## 3. Data and access audit

| Data | Prior use | Phase 4 role |
|---|---|---|
| p3_train identities | VE training, fusion training (Phase 3) | development: training |
| p3_val | early stopping, selection, Phase 3 orders/edges; **Phase 4 diagnostics** | development: selection, thresholds |
| p3_cal | Phase 3 LTT calibration | development: calibration |
| p3_test | Phase 3 confirmatory (decisions once, map accuracy ×3) | **follow-up** only |
| Phase 2 DigiFace val/cal/test | Phase 2 + Phase 3 follow-up | **follow-up** only (test) |
| LFW View 2 | Phase 2 final, post-hoc audit, Phase 3 follow-up + detector study + diagnostic | **follow-up** only |
| LFW View 1 DevTest | Phase 2/3 rehearsals | detector-policy development |
| **DigiFace 72-image parts 2000–3999, 4000–5999** (4,000 identities) | **never downloaded before 2026-11-01** | **p4_test, confirmatory, used once** |

**The p4_test partition.**
- Prepared by `scripts/prepare_phase4.py` in a separate private manifest:
  - official size and CRC-32 checks;
  - safe extraction;
  - pixel-hash deduplication against every existing DigiFace image.
- Pairs: clustered, balanced, 12 × identities, genuine cap 6.
- LOFO plans test_id / test_ood per fold, plus an E1 plan (side b occluded at low / medium /
  high). Seed 20261101.
- Caveat: the 72-image identities carry rendered accessories, which partly confounds F2.

## 4. Methods (all trained, selected and calibrated on development data only)

**Frozen backbones:** AdaFace and ArcFace IR-101. No fine-tuning.

### 4.1 Baselines (retained)

| Arm | Description |
|---|---|
| B0 | Raw similarity M0, global occlusion-aware LTT |
| B1 | Quality-only M1, global LTT |
| **A3** | Quality-stratified M1 (q tertiles). **REFERENCE** |
| A4 | Phase 3 learned-visibility strata: **frozen Phase 3 VE, models and rules unchanged** (`34409779…`) |
| A5 | Ground-truth visibility oracle (not deployable) |

### 4.2 Conservative visibility estimator VE-C

One configuration per fold, no hyperparameter search. Everything else matches Phase 3: same
architecture, data, budget (≤ 30 epochs, patience 4), seed and fp16.
- **Asymmetric loss.** BCE with weight **4.0** on cells whose target is occluded (< 0.5).
  Overestimating visibility costs 4× more.
- **Generic hard negatives** (20 % of samples). 1–4 irregular blobs filled with *natural
  texture*: crops of other p3_train images taken at a random offset, at random positions.
  Targets come from the blob mask.
  - This is a separate generator from every LOFO family; it has no polygon/ellipse
    uniform-colour fills.
  - It still makes F4 "less unseen", which is why F4 is labelled design-informed.
- **Lighting as corruption.** F5 training targets mark only information-free pixels as occluded
  (crushed or newly clipped), not merely darkened ones.
- **Lower bound and calibration** (on p3_val, seen families only):
  - temperature T fitted by cell log-loss;
  - then a global shift c chosen as the smallest value such that the predicted visible
    fraction exceeds the true one by > 0.10 on at most 5 % of occluded val renders.
  - The conservative map is `clip(sigmoid(logit/T) − c, 0, 1)`.

### 4.3 Novelty / unknown-occlusion detector (per fold)

- **Score.** Mahalanobis distance of the VE-C bottleneck feature (global-average-pooled
  encoder output, 256-d) to a Gaussian fitted on p3_train renders of the fold's seen families
  plus clean images, with shrinkage covariance. The pair score is the max over the two sides.
- **Threshold.** The 95th percentile of the score on p3_val in-distribution renders (seen
  families + clean). The ID false-positive rate is therefore 5 % **by construction**.
- **Secondary, reported only:** flip-TTA disagreement of the VE-C map.

### 4.4 Candidate arms (all use logistic fusion M4 recomputed with VE-C co-visibility unless stated)

- **C4 conservative.** Like A4, but with the VE-C conservative co-visibility for features and
  strata.
- **N4 novelty-aware.** C4, except that pairs flagged novel are forced into the lowest
  visibility stratum.
- **H4 quality-first hybrid.**
  - Pair *reliable* = not novel, on both sides.
  - Reliable pairs use VE-C visibility strata on the M4 score; unreliable pairs use quality
    strata on the M1 score.
  - The **whole six-stratum rule is calibrated jointly** on p3_cal (LTT per stratum at δ/6,
    Bonferroni), with orders from p3_val.
  - No validity is inherited from A3. The rule is valid on the calibration distribution
    because the strata are a fixed function of each pair. It carries **no OOD guarantee**,
    since routing itself shifts under novelty (SIM-013).

All arms use targets primary (sFNMR ≤ 5 %, sFMR ≤ 5 %) and far1 (sFMR ≤ 1 %, sFNMR ≤ 5 %),
v2 ordering (v1 reported), δ = 0.05.

### 4.5 Fusion (Q3)

- Per fold: logistic M1, M4, M5, M5-noCV (M5 without co-visibility features), and an **MLP**
  (2 × 64, M5 features).
- All fitted on p3_train, early-stopped and selected on p3_val. Seed 20261101.
- Selection rule (registered): use the MLP as the "deployable fused score" if its p3_val
  log-loss beats logistic M5 by > 5 %. Both are reported on p4_test regardless.

## 5. Endpoints (p4_test, single logged run)

Evaluation cells (10):
- **Test-OOD** per fold (held-out family): 5 cells, of which F4/F5 are design-informed.
- **E1** under each fold's frozen system: 5 cells.

Test-ID cells are secondary.

- **P1 safety (primary).** For each candidate X ∈ {C4, N4, H4}, a violation is a cell × error
  type whose 95 % identity-jackknife CI excludes the target. X is **safe** iff:
  - (a) its violations ≤ A3's, across the 10 cells × 2 error types;
  - (b) no cell has X significantly worse than A3 in sFNMR or sFMR (paired identity jackknife,
    Holm over the 10 cells, per error type).
- **P2 benefit (primary, only for safe candidates).** Mean coverage gain over A3 across the
  10 cells, with a paired jackknife CI per cell and Holm over cells. X is **better** iff it is
  safe, the mean gain is > 0, and ≥ 6/10 cells show a significant positive gain.
- **Recommendation rule.**
  - If no candidate is better: **recommend A3**.
  - If several are better: recommend the one with the fewest violations, then the largest gain.
- **H4 clean margin.** Clean FNMR of each candidate's fused score at its frozen clean-cal
  FAR-1 % threshold vs M0: upper CI < +0.5 pp (SIM-011 paired jackknife).
- **Secondary.**
  - Risk–coverage curves and AUGRC.
  - Worst-cell sFNMR / sFMR.
  - Per-family sFMR (false-match behaviour).
  - VE / VE-C signed visibility bias on p4 renders, per family.
  - Novelty AUROC: ID renders vs held-out, E1 and lighting renders; realised ID FPR.
  - Fusion log-loss / AUC and co-visibility ablations (M1 vs M4, M5 vs M5-noCV; Holm over
    10 cells).
  - Computational cost per pair: GPU ms and peak memory for backbones, VE, VE-C + Mahalanobis,
    and TTA.
- **Follow-up (labelled; not confirmatory):** p3_test, Phase 2 DigiFace test, LFW View 2
  derived occlusions.

## 6. Detector-inclusive robustness (Q4)

- **Taxonomy:** no face; multiple faces (central face chosen); low-quality alignment (detector
  score < 0.6 or 5-point similarity residual above the 99th percentile of clean View-1
  DevTrain); success.
- **Policies:** abstain; forced non-match; quality fallback (fixed crop + A3). They are
  compared on genuine coverage and sFNMR, and impostor sFMR.
- Thresholds and policy are chosen on **LFW View 1 DevTest** composites. **View 2 is
  follow-up only.**
- Failed detections are **never** dropped from any reported rate.

## 7. Training discipline and stopping criteria

- Only p3_train renders are used for training. Held-out-family isolation is enforced (the
  existing guard) and E1 is never rendered during development.
- One VE-C configuration and one MLP architecture. No hyperparameter search on any held-out or
  test data.
- Memory-monitored pilots precede every long job.
- **Stopping:** if VE-C fails to beat the Phase 3 VE on p3_val seen-family overestimation (a
  sanity check), report it and continue with the registered comparisons; do not iterate
  designs against held-out families.

## 8. Claims policy

- No out-of-distribution guarantees.
- Hybrid validity holds only on the calibration distribution.
- E1 results are the only "never-seen occluder" evidence; F4/F5 are design-informed.
- Follow-up sets are labelled as such.
- Negative results are reported.
