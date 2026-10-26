# VEIL-X — Evaluation Protocol (pre-registration draft)

> Labels per `research_landscape.md` §0. This protocol is written **before** any experiment.
> Changes after the first `--final` test run must be logged in §10, with a reason, and reported.
>
> **Phase 0 lock (2026-10-26):** the authoritative, versioned protocol is
> `configs/research_protocol.yaml` (v0.1.0, *provisional*), explained in **`protocol_design.md`**.
> Where this document and `protocol_design.md` differ, `protocol_design.md` wins. Main
> changes, each in `decision_log.md`:
> - C3 (H2a/H2b) is **primary** and C1 (H1) is **secondary** (D-001/D-002).
> - FAR points are **1% and 0.1%** (D-011). 0.1% is estimation-only under the recommended
>   data subset.
> - Primary LTT calibration uses an **identity-disjoint matching** (D-009). That replaces
>   "one pair per identity pair" in §7.
> - Required arms: raw AdaFace, raw ArcFace, quality abstention, combined abstention, global
>   calibration, visibility-stratified calibration, and a credible occlusion-adaptive threshold
>   (`protocol_design.md` §6).
> - C1 compares `quality_only`, `covisibility_only` and `quality_plus_covisibility`.
> - Success criteria count the **4 physical families** (+ ROF if approved). F5 lighting is
>   reported separately (D-014).
> - Every result reports error definitions, genuine/impostor composition, FAR thresholds and
>   their source split, coverage (overall/genuine/impostor), and the verbatim calibration
>   prediction target.

## 1. Hypotheses

| ID | Hypothesis [H] | Primary metric | Data |
|---|---|---|---|
| **H1** (C1) | Adding pair-level visibility features (co-visibility + visibility-weighted region similarity) to the abstention score **lowers AUGRC** compared with the best quality-only baseline using the same frozen backbone | ΔAUGRC (proposed − best of B2/B3/B4/B5) | Test-OOD-synth (each held-out family), Test-real (ROF) |
| **H2a** (C3-i) | LTT thresholds calibrated on seen families **violate** the target selective FMR on held-out families | Violation ratio = realised sFMR / α; violation frequency over resamples | Test-OOD-synth, ROF |
| **H2b** (C3-ii) | Visibility-stratified calibration **reduces** violation relative to global calibration, at a reported coverage cost | Δ violation ratio; Δ coverage | Same |
| **H3** (C2, secondary) | Visibility-weighted fusion raises non-selective TAR@FAR on occluded pairs | ΔTAR@FAR=1e-3 and 1e-2 | Test-OOD-synth, ROF |
| **H4** (non-inferiority) | The proposed system does not degrade clean-pair performance beyond a margin Δ_NI fixed **before** testing (proposed: 0.5 pp TAR@FAR=1e-2 [D, to be approved]) | ΔTAR on clean pairs | Test-ID clean subset, LFW |

**Decision rules (fixed in advance) [D]:**
- H1 is **supported** only if the 95% identity-cluster bootstrap CI of ΔAUGRC excludes 0 (in
  favour of the proposed method) on **ROF** *and* on at least 3 of the 5 held-out families, for
  the primary backbone. It is **generalised** only if the result also holds for the secondary
  backbone.
- If H1 holds on synthetic families but not on ROF, the claim is restricted to synthetic
  occlusions and the synthetic-to-real gap is reported as a primary finding.
- Holm–Bonferroni correction is applied across the 5 families + ROF for H1 and H3.
- No numeric performance target is set. The study reports what is measured.

## 2. Splits and conditions
As defined in `dataset_strategy.md` §3–§5. In summary:
- Identities are partitioned into train/val/cal/test by hash. Pairs are built within a split.
- Seen families are used for train/val/cal. Leave-one-family-out (LOFO) over F1–F5 produces 5
  separate model+calibration instances. Each is evaluated on its own held-out family (OOD) and
  on the seen families (ID).
- **Real tests:** ROF (masked, sunglasses, neutral), MLFW, CPLFW, XQLFW. None are used for any
  tuning. ROF's mask family overlaps F1, and its sunglasses overlap F2. In the LOFO run where F1
  is held out, "ROF-masked" is a *real-unseen* test. In other runs it is real-but-seen-family.
  Both readings are reported, labelled accordingly.
- **Seeds:** 3 seeds for synthetic occlusion sampling, segmenter training and head training.
  Results are reported as mean ± s.d. across seeds, with bootstrap CIs within each seed.

## 3. Pair construction
Within each evaluation split:
- **Condition strata:** clean–clean, clean–occluded, occluded–occluded (same family),
  occluded–occluded (different families). Reported separately and pooled.
- **Genuine pairs:** all same-identity pairs, or a uniform sample capped per identity (cap fixed
  per dataset to prevent prolific identities from dominating).
- **Impostor pairs:** uniformly sampled cross-identity pairs, with **at least 10 / FAR_min
  impostor pairs per reported FAR** (e.g. ≥ 10,000 for FAR = 1e-3) where the data allows. When
  the data does not allow it (ROF), the lowest FAR reported is the one supported by the data, and
  the rest are marked "not estimable".
- Standard benchmark protocols (LFW/CALFW/CPLFW/MLFW/XQLFW official pair lists) are reported
  **as-is** for comparability, *in addition to* the larger VEIL-X pair sets.

## 4. Baselines

All baselines use the **same frozen backbone, alignment, and cal split** as the proposed method.

| ID | Baseline | Abstention score |
|---|---|---|
| B0 | ArcFace IResNet cosine, single threshold | none (coverage = 1) |
| B1 | AdaFace IR101 cosine, single threshold | none |
| B2 | B1 + feature-norm rejection | `q_min` |
| B3 | B1 + CR-FIQA rejection (if weights are licence-cleared) | `min(CR-FIQA_A, CR-FIQA_B)` |
| B4 | B1 + score-margin rejection | `|g − τ₀|` |
| B4+ | Logistic combination of B2 + B4 (strongest "no-visibility" competitor) | fused |
| B5 | Occlusion-severity-adaptive threshold (cf. P13). Severity = 1 − c from the same segmenter | threshold shift, no abstention; plus a variant with B4+ abstention |
| B6 (opt) | DeepFace-EMD-style patch re-scoring (P9) | `|EMD score − τ|` |
| B7 (opt) | PFE-style uncertainty head on frozen embeddings (P17) | mutual likelihood / variance |
| **P** | **Proposed**: fused φ (§3.1 of architecture) + LTT decision rule | calibrated p + evidence gate |

## 5. Ablations

| ID | Change | Question answered |
|---|---|---|
| A1 | Visibility = **oracle** GT synthetic mask | Upper bound: is visibility *estimation* the bottleneck? |
| A2 | Visibility = constant 1 (none) / random | Does the gain come from visibility or from extra parameters? |
| A3 | Remove region similarities; keep the co-visibility scalar c | Is local evidence needed, or is sufficiency (c) enough? |
| A4 | Remove quality features (q_min, q_max) | Does visibility subsume quality? |
| A5 | LR vs tiny MLP vs GBDT fusion | Is nonlinearity needed? |
| A6 (opt) | + periocular embedding | Does a separate periocular embedding help on F1? |
| A7 (opt) | + PFE uncertainty | Does embedding variance add information? |
| A8 | Remove the evidence gate (c_min) | Is the explicit gate needed beyond the fused p? |
| A9 | Global vs visibility-stratified vs family-stratified (oracle family label) calibration | H2b: how much stratification helps and where its ceiling is |
| A10 | Backbone swap (AdaFace ↔ ArcFace) | Generality of conclusions |
| A11 | Segmenter trained on all families (incl. FaceOcc) — **labelled non-OOD** | How much is lost by honest family hold-out |
| A12 | Spatial features PCA-reduced (128-d per cell) vs full | Storage/accuracy trade-off |

## 6. Metrics — definitions

Let each pair i have a label yᵢ ∈ {0 (impostor), 1 (genuine)} and a decision dᵢ ∈ {0, 1, ⊥}.
Alignment failures are counted as dᵢ = ⊥ for selective metrics. For non-selective metrics, they
receive score −∞ (forced non-match). The same rule applies to all methods.

**Non-selective (score-based):**
- **TAR@FAR** at FAR ∈ {1e-2, 1e-3} (registered constant; 1e-4 dropped, D-011). Two reporting
  modes:
  - *Benchmark mode:* the threshold is chosen on the test ROC (literature convention, for
    comparability only).
  - *Deployment mode:* the threshold is chosen on **cal**, then applied to test. Report the
    **realised** FAR and TAR. This is the honest number.
- **ROC-AUC**, **EER**. Also the full ROC curve (log-FAR axis) in the reports.

**Selective (decision-based):**
- Coverage `C = P(d ≠ ⊥)`. Also **per-class coverage** `C_gen = P(d≠⊥ | y=1)` and
  `C_imp = P(d≠⊥ | y=0)`. A method cannot "win" by abstaining on one class only, so both are
  always reported.
- Selective FMR `sFMR = P(d=1 | y=0, d≠⊥)`; selective FNMR `sFNMR = P(d=0 | y=1, d≠⊥)`.
- Selective risk `R = P(d ≠ y | d ≠ ⊥)`, plus the cost-weighted variant
  `R_w = (w_FM·#FM + w_FNM·#FNM) / #accepted`, with costs declared up front (default w_FM = 10,
  w_FNM = 1 [D, to be approved]).
- **Risk–coverage curve** traced by sweeping the abstention threshold on the ranking induced by
  the confidence score. **AURC** is reported for comparability, and **AUGRC** (P26) as the
  primary metric, because P26 shows AURC's flaws. Generalised risk = P(d ≠ y ∧ d ≠ ⊥).
- **Selective TAR@FAR at fixed coverage** (80%, 90%, 95%), plus **coverage at target risk**
  (coverage achievable with sFMR ≤ α on test).
- **Guarantee violation (H2):** for target α: realised sFMR/α on test, and the fraction of
  bootstrap resamples where sFMR > α.

**Calibration:** ECE (15 equal-mass bins) and the Brier score of p(match); reliability diagrams
per family.

**Visibility module:** mIoU and boundary F-score of V vs synthetic GT on held-out families;
correlation of c with GT co-visibility. On ROF there is no GT, so the report is qualitative plus
the abstention effect.

**Operational:** decision **latency** per pair (§8), peak VRAM, alignment failure rate per family.

## 7. Statistics

- **Uncertainty intervals:** identity-cluster bootstrap. Resample *identities*, then use all
  pairs among the resampled identities, with ≥ 1,000 resamples. Because FAR/FRR are
  U-statistics over pairs, follow the recentering guidance of Conti & Clémençon (P30) for
  ROC-derived quantities. Implementation is validated by simulation on synthetic scores before
  use [D].
- **Paired comparisons:** method differences are computed **within** each bootstrap resample
  (same identities for both arms), giving CIs on the differences.
- **LTT:** binomial-tail (Hoeffding–Bentkus or exact binomial) p-values with fixed-sequence or
  Bonferroni testing over the threshold grid. Because pairs share identities, the i.i.d.
  assumption is violated. **Superseded (D-009):** the primary LTT variant uses an
  **identity-disjoint matching** (each identity in at most one cal pair). "One pair per
  identity pair" and "all pairs" are sensitivity analyses only, because pairs that share
  identities are still dependent.
- Multiple comparisons: Holm–Bonferroni across families for each hypothesis.

## 8. Latency protocol

- Hardware is recorded (GPU model, driver, power mode, CPU, RAM), with OS power plan fixed and
  charger connected.
- Stages timed separately: detect+align, embed (×2), segment (×2), pair features + head,
  decision. End-to-end uses batch = 1 (interactive use) and batch = 64 (offline throughput).
- 50 warm-up pairs, then 500 timed pairs. Report p50, p95 and p99 in ms. Use
  `torch.cuda.synchronize()` around GPU stages.
- Report on GPU and CPU-only. No latency target is pre-set. Latency is reported, not promised.

## 9. Test-set discipline

1. Before any `--final` run, the experiment config is committed. The git hash is the
   pre-registration timestamp.
2. The runner refuses `split=test` without `--final`. Each final run is appended to
   `results/test_runs.jsonl` (hash, config, timestamp).
3. **No hyperparameter, threshold, family definition or feature may change because of a test
   result.** If a bug is found after a final run, the fix and the rerun are both reported.
4. ROF is evaluated only in final runs.

## 10. Deviations log
_(empty — to be filled if the protocol changes after the first final run)_
