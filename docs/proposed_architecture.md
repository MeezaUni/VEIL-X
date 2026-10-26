# VEIL-X — Proposed Architecture

> Labels per `research_landscape.md` §0. Everything here is a **design choice [D]** or a
> **hypothesis [H]**. Nothing is measured yet.
>
> **Phase 0 update (2026-10-26):** approved choices are InsightFace detector (D-004), frozen
> AdaFace IR101 + ArcFace (D-003), and DigiFace-1M training data (D-005). Regional fusion
> (spatial-feature region similarities, C2) is **optional/secondary**. Co-visibility (C1) is
> secondary. Risk calibration under held-out families (C3) is primary. Interface contracts
> now exist in code (`src/veilx/interfaces.py`, documented in `interfaces.md`). The package is
> `src/veilx/` (src layout). Only scaffolding is implemented (Phase 0).

## 1. Design principles

1. **Freeze what others trained well.** The recognition backbones are frozen. VEIL-X trains only
   small modules whose *purpose is testable*.
2. **Pair-level decisions with three outcomes:** `MATCH`, `NON-MATCH`, `ABSTAIN`.
3. **Every module earns its place through an ablation.** A module that does not improve its
   pre-declared metric is removed from the final system, not hidden.
4. **Expensive work happens once.** GPU embedding extraction is cached. Everything downstream
   runs on CPU in seconds, so experiments iterate quickly and stay reproducible.
5. **Separate the decision from the explanation.** The demo's rationale is generated from the
   same quantities the decision used, by a fixed template. There is no free-text generation.

## 2. Pipeline

```
image A ─┐                                                          ┌─> p(match) ─┐
         ├─> detect+align (112×112) ─> frozen backbone ─> e (512-d), ‖e‖, F (512×7×7)
image B ─┘           │                                              │             │
                     └─> visibility segmenter ─> V (112×112) ─> pooled v (7×7, 5 regions)
                                                                    │             │
           pair features: cos(eA,eB), co-visibility, region sims,  ─┤             │
           min/max quality, margin to threshold                     │             ▼
                                                     fusion head (LR / tiny MLP)  │
                                                     + calibration (Platt/temp.)  │
                                                                                  ▼
                                    risk-controlled decision rule (LTT on cal split)
                                    → MATCH / NON-MATCH / ABSTAIN + rationale record
```

## 3. Component decisions (answers to brief item 6)

Tier: **MVP** = needed to test C3(i). **Core** = needed for C1 and C3(ii). **Opt** = optional
extension. **Drop** = not built.

| Component | Decision | Testable purpose | Test that decides whether it stays | Tier |
|---|---|---|---|---|
| Detection + 5-point alignment | **InsightFace SCRFD/RetinaFace (approved, D-004)** → ArcFace-standard 112×112 similarity transform. MediaPipe is no longer planned | Standard input for the backbones | Alignment failure rate per occlusion family. **Failures are counted as abstentions in all metrics**, never silently dropped | MVP |
| Frozen backbone: AdaFace IR101 (primary), ArcFace IResNet (secondary) | Use both, frozen, fp16 inference | Strong baseline; checks that conclusions hold across backbones | Baseline B0/B1 metrics; conclusions must hold on both backbones to be stated generally | MVP |
| **Full-face global embedding** | Yes | Primary identity evidence | — (baseline) | MVP |
| **Separate periocular embedding** (backbone on upsampled eye crop) | **Opt.** Not in the core | H: adds evidence when the lower face is occluded | Ablation A6: AUGRC on F1 (lower-face) with vs without it. Prior concern [H]: backbones are not trained on periocular crops | Opt |
| **Spatial feature map F (7×7) + region similarities** | **Optional (C2, secondary)**. Taken from the backbone's last conv block before the FC/flatten | Local evidence restricted to co-visible regions (cf. DeepFace-EMD, P9) | Ablation A3: remove region sims, keep the co-visibility scalar | **Opt** (was Core) |
| **Patch visibility segmenter** | Yes. Small encoder-decoder (e.g. a MobileNet/ResNet18 U-Net) on 112×112 aligned crops, trained on **synthetic overlays of seen families** | Provides V. Pooled to 7×7 cells and 5 semantic regions (L-eye, R-eye, nose, mouth, jaw) via the fixed aligned template | (a) mIoU on held-out-family synthetic masks; (b) ROF qualitative + abstention effect; (c) Ablation A1/A2: oracle vs learned vs no visibility | Core |
| **Co-visibility** c = mean_k(vA_k · vB_k) | Yes | Pair-level evidence sufficiency (C1, secondary). It also defines the strata for visibility-stratified calibration (C3-ii, primary) | Feature sets `quality_only` vs `covisibility_only` vs `quality_plus_covisibility` (`protocol_design.md` §6) | Core |
| **Per-image quality** (AdaFace feature norm; CR-FIQA optional) | Yes, as **baselines and as features** | Ensures VEIL-X is compared against, and can use, existing quality signals | Baselines B2/B3; ablation A4 removes quality from the fused model | MVP |
| **Learned fusion head** | Logistic regression first (interpretable, monotonic-friendly). Tiny MLP / GBDT only if LR is clearly underfit on val | Combines evidence into p(match) | Val log-loss; A5 compares LR vs MLP | Core (LR) / Opt (MLP) |
| **Calibration** | Temperature or Platt scaling on **val** (and isotonic as a variant); ECE/Brier reported on test | Probabilities that mean something in the demo and for thresholds | ECE, Brier on Test-ID / Test-OOD | MVP |
| **Calibrated rejection (risk control)** | LTT (P27) over a grid of (τ_low, τ_high), controlling selective FMR ≤ α and selective FNMR ≤ β with confidence 1−δ on **cal**. Variant: visibility-stratified (Mondrian) calibration | Turns scores into decisions with a stated risk target. Measures how the guarantee degrades under shift | Guarantee-violation rate on Test-OOD (H2) | MVP (global) / Core (stratified) |
| **Embedding uncertainty head (PFE-style, P17)** | **Opt.** Trains on frozen embeddings | H: per-image variance adds to the quality signal | Ablation A7: AUGRC with vs without it | Opt |
| **Occlusion-severity-adaptive threshold** | Implemented **as baseline B5** (cf. P13), not as a contribution | Tests whether simple threshold shifting already explains any gains | Comparison B5 vs proposed | Core (baseline) |
| Patch EMD matching (P9) | **Opt** baseline B6 | Strong frozen-feature local-matching comparator | B6 vs proposed | Opt |
| Mixture of experts, ArcFace+InfoNCE fine-tuning, backbone fine-tuning, generative inpainting/de-occlusion | **Drop** | No testable link to the RQ that justifies the cost; heavy compute; known prior art | — | Drop |

### 3.1 Pair feature vector (Core) [D]
For aligned images A and B with embeddings e, norms n, spatial features F_k (k = 1…49 cells) and
visibilities v_k ∈ [0,1]:

- `g = cos(eA, eB)`, the global similarity
- `w_k = vA_k · vB_k`, the co-visibility weight of cell k; `c = mean_k w_k`
- `s_k = cos(FA_k, FB_k)`; `r = Σ w_k s_k / (Σ w_k + ε)`, the visibility-weighted local similarity
- region co-visibility for the 5 semantic regions (5 values)
- `q_min = min(nA, nB)`, `q_max = max(nA, nB)` (AdaFace norm), optionally CR-FIQA scores
- `m = |g − τ₀|`, where τ₀ is the global threshold at the target FMR, set on **cal**

Fusion: `logit p = θᵀ φ(g, r, c, regions, q_min, q_max, g·c, r·c)`. Interaction terms let low
co-visibility shrink confidence.

### 3.2 Decision rule [D]
- `p ≥ τ_high` → MATCH; `p ≤ τ_low` → NON-MATCH; otherwise → ABSTAIN.
- **Evidence gate** (Core, ablated in A8): if `c < c_min`, or detection/alignment failed, the
  output is ABSTAIN with the reason "insufficient shared evidence". `c_min` is part of the LTT
  parameter grid. It is never hand-tuned on test data.
- (τ_low, τ_high, c_min) come from LTT on the cal split, for user-chosen α (FMR), β (FNMR) and δ.
- **Guarantee caveat:** LTT's guarantee requires cal/test exchangeability. It is therefore
  **expected not to hold** on held-out families. H2 measures by how much. The full assumption
  list (A1–A6), the verbatim prediction target and the identity-disjoint calibration pairing
  are in `protocol_design.md` §2 and §4.3.

### 3.3 Rationale record (for the demo and auditability)
Each decision emits a structured record: `{decision, p, g, r, c, region_covis[5], q_A, q_B,
thresholds, alpha, beta, triggered_rules[], backbone_id, model_hashes}`. Abstention reasons are
typed as `INSUFFICIENT_COVISIBILITY`, `LOW_QUALITY`, `AMBIGUOUS_SCORE` or `ALIGNMENT_FAILED`.

## 4. Module boundaries (Python package `veilx/`)

> Phase 0 status: implemented = `config`, `data.manifest`, `data.splits`, `guards`,
> `provenance`, `logging_utils`, `seeding`, `errors`, `stats.feasibility`, and `interfaces`
> (contracts only). Not implemented = everything that touches images or models. Manifests
> are JSONL in Phase 0, not parquet; parquet stays an option once pyarrow is justified.

| Module | Responsibility | Inputs → outputs | Depends on |
|---|---|---|---|
| `veilx.data` | Manifests, identity-hash splits, pair builder, near-duplicate guard, licence registry | raw dataset dirs → `manifests/*.parquet`, `pairs/*.parquet` (+ SHA-256) | — |
| `veilx.occlusion` | Family registry F1–F5, deterministic synthesis (`family, asset, geom, seed`), GT masks | aligned image → occluded image + GT mask | `data` |
| `veilx.align` | Detection, landmarks, similarity-transform alignment, failure flags | image → 112×112 crop or `AlignFail` | detector weights |
| `veilx.embed` | Backbone wrappers with a uniform interface `embed(batch) → (e, norm, F)`; disk cache keyed by `(image_sha, occlusion_spec, backbone_id)` | crops → cached `.npy`/zarr | `align` |
| `veilx.visibility` | Segmenter train/infer; pooling to 7×7 and 5 regions | crop → V, v | `occlusion` (for labels) |
| `veilx.quality` | Feature norm, optional CR-FIQA, optional PFE head | crop/emb → q | `embed` |
| `veilx.scoring` | Pair features φ, fusion head, calibration | cached tensors → p(match) | `embed`, `visibility`, `quality` |
| `veilx.selective` | LTT / CRC, stratified calibration, decision rule, rationale records | p, c, labels(cal) → thresholds; p → decisions | `scoring` |
| `veilx.eval` | Metrics, identity-cluster bootstrap, latency harness, report generation | decisions + labels → `results/*.json`, figures | all |
| `veilx.demo` | Local Gradio app (1:1 only) | two images → decision + visuals | inference path only |
| `configs/` | YAML experiment configs; one file per pre-registered experiment | — | — |

Interfaces are plain dataclasses and NumPy arrays. Only `embed`, `visibility` and `align` touch
the GPU.

## 5. Compute plan (RTX 4060 laptop, 8 GB VRAM)

| Workload | Device | Expected cost (estimate [H], to be measured) | Bottleneck |
|---|---|---|---|
| Alignment of source images | GPU/CPU | One pass per real image | Detector failures under heavy occlusion |
| Embedding extraction (IR101, fp16, batch ~128–256) | GPU | Dominant GPU job; scales with images × families × seeds | **Combinatorics of occlusion variants.** Cap variants per image; cache aggressively |
| Spatial-map storage | Disk | 512×7×7 fp16 ≈ 50 KB/image → ≈ 5 GB per 100k variants | **Disk.** Store F only for eval subsets, or PCA-reduce to 128-d per cell (ablated) |
| Segmenter training (112×112, small U-Net) | GPU | Small model; fits easily in 8 GB | Synthetic data generation throughput on CPU (use dataloader workers) |
| Fusion / calibration / LTT / bootstrap | CPU | Seconds to minutes | Bootstrap with 1,000+ resamples × many configs (parallelise with joblib) |
| Demo inference (per pair) | GPU or CPU | Detect + 2× embed + 2× segment + head | Detector latency on CPU; measured, not assumed |

Thermal throttling on laptops affects latency numbers. Fix the power mode, warm up, and report
the GPU clock state.

## 6. Test strategy (software)

| Level | What | Examples |
|---|---|---|
| Unit | Pure functions | Metric functions vs `sklearn.metrics` on known vectors; TAR@FAR interpolation; EER; AUGRC on hand-computed toy curves; LTT p-values vs a reference binomial implementation |
| Property | Invariants | Risk-coverage coverage is monotone in the threshold; co-visibility ∈ [0,1] and symmetric in A↔B; pair-feature symmetry |
| Data integrity | **Leakage guards (run in CI)** | No identity in more than one split; occlusion variants share their source's split; no near-duplicates across splits; test files unread by training entry points (import-path / file-access check) |
| Determinism | Reproducibility | Same `(seed, spec)` → byte-identical occluded image and mask; pair lists match their checksums |
| Golden | Model wrappers | Embedding of a fixed licence-cleared synthetic image matches a stored vector within tolerance (catches preprocessing drift: BGR/RGB, normalisation) |
| Integration | End-to-end smoke | 20 procedurally generated "faces" or DigiFace samples → full pipeline → report, in < 2 min on CPU |
| Experiment | Config validation | Every experiment config declares split, families held out, backbone and seeds. The runner refuses to evaluate on `test` without the `--final` flag, and logs that run |

## 7. Local demo design

**Purpose:** show *why* the system decided, and show it **declining** when evidence is
inadequate. Do not show off accuracy.

- **Form:** Gradio app bound to `127.0.0.1` only. No network calls, no database, no persistence.
  Uploaded images and embeddings are held in memory and discarded on reset.
- **Flow:** the user uploads **exactly two images**. A consent checkbox is required: "I have the
  right to use these images; this is a research demo." Optional toggles add a synthetic occluder
  (family, position) to either image so the user can watch the decision change.
- **Displays:**
  1. The aligned crops with landmarks, and an alignment-failure message if relevant.
  2. Visibility maps overlaid on each image, plus the **co-visibility map** (cells visible in
     both images).
  3. A 7×7 local-similarity heatmap, masked by co-visibility.
  4. Numbers: global cosine, visibility-weighted similarity, co-visibility %, quality
     percentiles (relative to the cal set), and the calibrated p(match).
  5. **Decision badge:** MATCH / NON-MATCH / **ABSTAIN**, with typed reasons from the rationale
     record. Example (wording template, not a result): *"Abstained: only 24% of the face area is
     visible in both images (minimum 40% at the selected risk level); periocular region visible in
     both, mouth/jaw occluded in image B."*
  6. Operating point: the α/β risk targets, and the statement "targets are calibrated on seen
     occlusion types; on unseen types the error rate may exceed them."
- **Scripted scenarios** (using licence-cleared synthetic DigiFace identities, not real people):
  clean genuine pair → MATCH; same pair with mask + sunglasses split across images → ABSTAIN
  (low co-visibility); impostor pair → NON-MATCH; extreme shadow → behaviour shown honestly,
  whatever it is.
- **Not in the demo:** 1:N search, galleries, webcam streaming, batch processing of folders,
  saving results, demographic outputs.

## 8. Privacy, licensing and misuse constraints (architecture-level)

- **Only 1:1 verification of two user-supplied images.** No API for 1:N search, no gallery or
  enrolment database, no embedding export endpoint. [D]
- Model weights and trained heads are labelled **non-commercial research only**, inherited from
  the backbones and data (`dataset_strategy.md` §2.1).
- No face data or embeddings in git (CI check on file types and sizes).
- The rationale record contains no image data and no identity names.
- Report subgroup error rates only if a licence-clear, consented source of attributes is
  available. Do not infer demographics. FairCal-style analysis (P29) is out of scope unless such
  data exists.
