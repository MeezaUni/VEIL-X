# VEIL-X — Module Interfaces (Phase 0 contracts)

> Source: `src/veilx/interfaces.py` (typing `Protocol`s, `runtime_checkable`). These are
> contracts only. **No implementation exists yet.** Every implementation must be
> deterministic in eval mode and record the families it was fitted on.

| Contract | Key members | Inputs → outputs | Invariants enforced or tested |
|---|---|---|---|
| `DatasetAdapter` | `dataset_id`, `check_access()`, `iter_images()`, `official_pairs()` | local, licence-approved dataset → `ImageRecord`s (relative paths + pixel SHA-256), official `PairRecord`s | `check_access` raises `DataAccessError` unless the registry entry is `confirmed` (or its conditions are recorded as met); no pixels held |
| `FaceAligner` | `detector_id`, `align(image)` | RGB image → `AlignmentResult(ok, crop 112×112, landmarks 5×2, score, failure_reason)` | Failures are returned, never raised or dropped; policy = ABSTAIN (selective) / forced non-match (score metrics) |
| `EmbeddingExtractor` | `backbone_id`, `embedding_dim`, `embed(crops)` | uint8 crops (N,112,112,3) → `EmbeddingBatch(embeddings L2-normed (N,D), feature_norms (N,), spatial (N,C,H,W)?)` | Frozen weights; golden-vector test planned (Phase 2) |
| `VisibilityEstimator` | `estimator_id`, `trained_on_families`, `estimate(crops)` | crops → `VisibilityBatch(cell (N,K), region (N,R), region_names, trained_on_families)` | `require_heldout_isolation(heldout, trained_on_families)` |
| `QualityEstimator` | `quality_id`, `quality(crops, emb)` | → (N,) quality scores | Feature norm is the default; CR-FIQA only if licence-cleared |
| `PairFeatureGenerator` | `feature_set`, `feature_names`, `generate(pairs, emb, vis?, q?)` | → `PairFeatures(names, values (P,F), pair_ids)` | Named feature sets: `quality_only`, `covisibility_only`, `quality_plus_covisibility`, `margin_only`, `quality_plus_margin` |
| `RiskCalibrator` | `calibrator_id`, `fit(scores, labels, strata, calibration_families)`, `decide(scores, strata)` | cal data → `CalibrationResult(params, target, δ, n, n_accepted, families, assumptions, guarantee_scope, feasible)`; then decisions + abstain reasons | Fit only on the `cal` split; families recorded; `feasible=False` ⇒ abstain on all |
| `Evaluator` | `evaluate(method_id, decisions, scores, labels, identity_pairs, condition)` | → `EvaluationResult(metrics with CIs, composition, coverage)` | Identity pairs carried for the cluster bootstrap; composition always reported |
| `ResultReporter` | `report(results, provenance)` | → relative path / run id | Results always paired with a provenance record |

## Cross-cutting infrastructure (implemented in Phase 0)

| Module | Purpose |
|---|---|
| `veilx.config` | Pydantic schema + loader for the protocol and dataset registry; rule violations are config errors |
| `veilx.data.manifest` | `ImageRecord`, `OcclusionSpec`, `PairRecord`; integrity checks: identity–split purity, variant inheritance, pair label/stratum/split consistency, family isolation; JSONL IO |
| `veilx.data.splits` | Identity-hash split assignment |
| `veilx.guards` | `require_final_for_test`, `require_heldout_isolation` |
| `veilx.provenance` | Git commit/dirty, versions, config hashes, seed, argv. No hostname, username or absolute paths |
| `veilx.logging_utils` | JSON-lines structured logging |
| `veilx.seeding` | Global seeding (Python, NumPy, torch if present) and stable derived seeds |
| `veilx.errors` | `VeilxError` hierarchy: `ConfigError`, `ManifestError`, `LeakageError`, `ProtocolViolationError`, `DataAccessError` |
| `veilx.stats.feasibility` | Exact binomial CDF, Clopper–Pearson bounds, zero-error n, independent-pair capacity |
