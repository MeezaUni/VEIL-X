# VEIL-X — Implementation Roadmap

> Labels per `research_landscape.md` §0. Effort sizes are rough single-developer estimates
> **[H]**, not commitments. Nothing has been implemented yet.

## 0. Simplest defensible MVP

**MVP = frozen backbones + quality/margin abstention + LTT risk control + leave-one-family-out
evaluation, including ROF.**

It answers C3(i): *how badly do calibrated face-verification error rates break on occlusion types
that calibration never saw?* It needs **no new model training**, only cached embeddings and CPU
statistics. It also builds the harness that every later component is judged by.

Everything else (visibility segmenter, region similarities, stratified calibration) is the
**Core** layer, needed for C1/C3(ii). PFE, periocular embeddings, EMD and MLP fusion are
**Optional**.

## 1. Phases and gates

### Phase 0 — Approvals, protocol lock and scaffolding (status 2026-10-26)
- [x] Research direction approved (C3 primary, C1 secondary): `decision_log.md`.
- [x] Isolated git repository at the project root; `.gitignore` written before any staging;
      `origin` = `MeezaUni/VEIL-X` (remote empty). **No commit yet** (D-018 pending).
- [x] Licence and access audit → `dataset_feasibility.md` + `configs/datasets.yaml`. This
      replaces the planned `data/LICENSES.md`: `data/` is git-ignored, and the registry must be
      versioned.
- [x] Versioned protocol `configs/research_protocol.yaml` (provisional) + `protocol_design.md`.
- [x] Statistical feasibility of the constants (`scripts/feasibility_report.py`).
- [x] Scaffold: `pyproject.toml`, `src/veilx`, tests, Ruff, mypy (strict); Python 3.12 via uv.
      PyTorch/onnxruntime deliberately **not** installed yet.
- [ ] Downloads (DigiFace subset, weights) need a separate go-ahead. CelebA no longer requested
      (optional).
- [ ] ROF ethics/legal confirmation; MLFW/CPLFW/XQLFW licence checks; Kaggle dataset (blocked).
- **Gate G0:** all datasets used in Phases 1–2 have a recorded licence permitting research use.
  **Currently met for DigiFace-1M only.**

### Phase 1 — Data layer and evaluation harness (MVP) · size: M
- [ ] `veilx.data`: manifests, hash splits, pair builder, near-duplicate guard, checksums.
- [ ] `veilx.align`: detection + 5-point alignment, failure flags.
- [ ] `veilx.occlusion`: families F1–F5, deterministic synthesis, GT masks, asset registry.
- [ ] `veilx.eval`: TAR@FAR (both modes), ROC-AUC, EER, risk-coverage, AURC, AUGRC, ECE/Brier,
      identity-cluster bootstrap, latency harness.
- [ ] CI: unit, property, leakage and determinism tests (`proposed_architecture.md` §6).
- **Gate G1:** metric functions match reference implementations on toy data; leakage tests pass;
  the bootstrap is validated on simulated scores with known ground truth.

### Phase 2 — Frozen baselines and risk control (MVP complete) · size: M
- [ ] `veilx.embed`: AdaFace IR101 + ArcFace wrappers exposing (e, ‖e‖, F); golden tests;
      embedding cache.
- [ ] Baselines B0–B4, B4+. Temperature/Platt calibration on val.
- [ ] `veilx.selective`: LTT over (τ_low, τ_high), global calibration, thinned-cal variant.
- [ ] Run on val/cal only. Sanity-check that LTT controls sFMR **in distribution** (on held-out
      *cal* folds). If it doesn't, there is a bug.
- [ ] First `--final` run: baselines on Test-ID, Test-OOD (LOFO) and ROF → **C3(i) result**.
- **Gate G2 (go/no-go for Core):**
  - If baselines already show **no** measurable guarantee violation on held-out families *and*
    on ROF, C3 loses most of its motivation. Revisit the RQ with the user before building Core.
  - If alignment failures dominate abstentions under occlusion, prioritise alignment robustness
    before visibility.

### Phase 3 — Visibility and pair-level evidence (Core) · size: L
- [ ] `veilx.visibility`: segmenter trained per LOFO split on seen-family synthetic overlays;
      mIoU on held-out families; pooling to 7×7 and 5 regions.
- [ ] `veilx.scoring`: pair features φ, LR fusion, calibration.
- [ ] Evidence gate (c_min) inside the LTT grid; visibility-stratified calibration.
- [ ] Baseline B5 (severity-adaptive threshold).
- [ ] Ablations A1–A5, A8–A10 on **val** during development. Test only via `--final`.
- **Gate G3:** oracle visibility (A1) must improve over no visibility (A2) on *val*. If even
  oracle visibility does not help, the learned segmenter cannot help either: stop Core, and
  report C3(i) plus this negative finding.

### Phase 4 — Final evaluation and report · size: M
- [ ] Final runs for the pre-registered configs (both backbones, 3 seeds, 5 LOFO splits, ROF,
      MLFW, CPLFW, XQLFW).
- [ ] Statistical tests per `evaluation_protocol.md` §1 and §7; figures; deviations log.
- [ ] Write-up separating [V] prior work, [H] hypotheses, [D] design and [M] measured results.

### Phase 5 — Local demo · size: S–M
- [ ] Gradio app per `proposed_architecture.md` §7, bound to localhost, no persistence.
- [ ] Scripted scenarios using DigiFace synthetic identities only.
- [ ] Rationale templates are tested so that every abstention reason maps to the triggering
      quantity.

### Optional extensions (only after Phase 4, each with its own ablation)
| Extension | Ablation | Note |
|---|---|---|
| PFE-style uncertainty head (B7/A7) | A7 | Trains on frozen embeddings; cheap |
| Periocular embedding (A6) | A6 | Backbones are not trained on eye crops. Expect weak features [H] |
| DeepFace-EMD baseline (B6) | — | Strong comparator; EMD cost per pair |
| MLP/GBDT fusion (A5) | A5 | Only if LR underfits on val |
| FaceOcc "all-families" segmenter (A11) | A11 | Clearly labelled non-OOD |

**Explicitly not planned:** backbone fine-tuning, MoE, ArcFace+InfoNCE retraining, generative
de-occlusion, 1:N identification, real-time video.

## 2. Risk register

| Risk | Likelihood [H] | Impact | Mitigation |
|---|---|---|---|
| CelebA identity labels not granted | Medium | Training identities are only synthetic (DigiFace) | Proceed with DigiFace; report the domain gap |
| ROF too small for tight CIs at low FAR | High | Weak real-world conclusions | Report the estimable FAR range only; emphasise AUGRC/coverage; do not over-claim |
| Detector fails on heavy occlusion | High | Abstentions driven by alignment, not visibility | Count failures as abstentions for all arms; report failure rate per family; try a landmark model robust to occlusion as an optional variant |
| Frozen 7×7 local features are not locally discriminative | Medium | A3 shows no benefit from region sims | Keep the co-visibility scalar (A3) as the main signal; report it |
| Synthetic occluders don't transfer to real | Medium-High | H1 fails on ROF | Report as a finding; scope claims |
| Pretraining identity overlap with LFW-derived tests | High (likely present) | Inflated absolute numbers | Relative comparisons only; DigiFace check |
| Laptop thermals distort latency | Medium | Noisy latency | Fixed power mode; warm-up; report p95 |
| Disk usage from spatial maps | Medium | Storage pressure | Store F for eval subsets; PCA variant (A12) |
| A closely matching paper surfaces | Medium | Novelty reduced | Lit-search log; shift framing to replication/extension |

## 3. Decisions requiring your approval (blocking Phase 0)

> **Resolution status (2026-10-26):** items 2–5 and 7 are **approved** (D-001…D-006). Item 1
> is resolved as **blocked**: the assigned Kaggle URL returns 404 (D-008). Item 6 constants
> are **registered as provisional** with a feasibility analysis (D-011, D-012). The
> near-duplicate threshold is replaced by a validated procedure (D-013). Item 8 is **done**:
> an isolated repository is rooted at the project folder (D-016). Open items are listed in
> `decision_log.md` → "Pending PI decisions". The original list follows for traceability.

1. **Exact Kaggle dataset URL**, and whether it is required at all. It cannot serve verification
   (no identity labels). The proposed role is optional segmenter source images only.
2. **Primary contribution framing:** C3 + C1 (recommended), with C2 secondary.
3. **Backbones:** AdaFace IR101 (WebFace4M) primary + ArcFace IResNet secondary. Both are
   **non-commercial research only**. Confirm that this project is non-commercial.
4. **Training identities:** request CelebA identity labels, use DigiFace-1M, or both
   (recommended: both).
5. **Occlusion family list F1–F5** and LOFO as the OOD definition. Confirm that extreme
   illumination (F5) is treated as an occlusion family rather than a separate study.
6. **Pre-registered constants:** non-inferiority margin Δ_NI (proposed 0.5 pp TAR@FAR=1e-2), cost
   weights (w_FM = 10, w_FNM = 1), risk targets (α, β, δ) for the demo and the H2 tests, and the
   near-duplicate threshold.
7. **Detector choice:** insightface SCRFD/RetinaFace (non-commercial weights) vs MediaPipe
   (permissive licence [U]; possibly weaker under occlusion).
8. **Repository location:** git resolved to the parent folder of the project (the user's
   Desktop), which contains unrelated personal files. Recommend `git init` inside the project
   folder before any commit, so that personal documents never end up in VEIL-X history.
   *(Done, D-016.)*

## 4a. Phase 1a status (2026-10-26): complete, without data downloads

Done:
- DigiFace adapter (on synthetic fixtures only).
- Pair engine.
- Metric library.
- Identity bootstrap.
- Audited LTT calibrator and empirical calibrator.
- Simulation harness, with 9 labelled simulation reports.
- Evidence registry.

Not done, because it was deliberately deferred: running the adapter on real DigiFace data,
which needs the download go-ahead. See `DEVELOPMENT_LOG.md`.

## 4c. Phase 1b status (2026-10-26): complete, no recognition model

Done:
- Approved DigiFace subset downloaded, verified and safely extracted.
- Real-data scan, fingerprints and exact deduplication. The near-duplicate rule was evaluated
  and rejected by its pre-set criterion.
- Final manifest and identity-disjoint splits.
- Pair sets, including the independent calibration set.
- Occlusion engine (F1–F5 + stress) and LOFO condition plans with severity and visibility
  records.
- Interval-method study (SIM-010) and protocol v0.3.0.
- GPU training infrastructure behind an authorisation guard.

See `data_integrity_report.md` and `DEVELOPMENT_LOG.md`.

## 4d. Proposed Phase 2 (requires separate authorisation)
1. Download the detector and backbone weights (InsightFace, AdaFace IR101): licence-check,
   checksum, record.
2. Frozen-backbone embedding extraction for clean and occluded variants, rendered on the fly
   from the condition plans, with recipe hashes verified. Cache embeddings privately.
3. Raw-backbone baselines (R-Ada, R-Arc) on Test-ID and Test-OOD with the jackknife intervals.
   C3-i (H2a) is then measurable with the LTT calibrator on `cal_independent`.
4. Re-check clean-margin power with real score distributions; validate paired-difference
   intervals (SIM-011).

## 4b. Phase 1b proposal (kept for traceability)
1. After the download go-ahead: run the adapter on the DigiFace subset. Freeze the manifest and
   the split counts.
2. Build pair lists, including identity-disjoint calibration sets. Record their checksums.
3. Occlusion synthesis for F1–F5 (procedural assets only), with GT masks and
   `assert_family_isolation` in the pipeline.
4. Pixel-level hashing and the validated near-duplicate procedure (`protocol_design.md` §8).
5. Two-way-aware bootstrap refinement (SIM-001 caveat), then freeze the bootstrap constant.
6. Decide on D-019, the 0.5 pp margin and FAR 0.1% (estimation-only). Then freeze the constants.

## 4. Original Phase 1a proposal (kept for traceability)

Scope, after the download go-ahead for the DigiFace subset:
1. `DigiFaceAdapter` (implements `DatasetAdapter`): manifest generation with pixel SHA-256 and
   hash splits; manifest validation; split counts reported.
2. Pair builder implementing `protocol_design.md` §4: strata quotas, different-source genuine
   pairs, identity-disjoint calibration matching, FAR-estimation sets; checksummed pair lists.
3. Metric library (TAR@FAR in both modes, ROC-AUC, EER, risk-coverage, AURC, AUGRC, ECE, Brier),
   tested against hand-computed toy cases.
4. Identity-cluster bootstrap + **coverage simulation** on synthetic scores (gate for the 95% CI
   constant), and power simulation for the 0.5 pp clean margin.
5. Re-check LTT's ratio-risk construction (D-010). Implement the binomial LTT calibrator on
   simulated scores only.

Exit gate: all of the above tested on synthetic/simulated data. Then the remaining
provisional constants are frozen (or revised) in `decision_log.md` before any embedding is
computed.
