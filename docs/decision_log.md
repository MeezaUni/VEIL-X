# VEIL-X — Decision Log

Status values: **approved** (by PI), **provisional** (registered, may change via this log),
**pending** (awaiting PI), **design** (engineering choice, revisable).

| ID | Date | Decision | Status | Source / rationale |
|---|---|---|---|---|
| D-001 | 2026-10-26 | Primary study = **C3** (risk-control degradation under held-out families; visibility-stratified calibration) | approved | PI brief, Phase 0 |
| D-002 | 2026-10-26 | Secondary study = **C1** (co-visibility vs quality for error prediction/abstention); regional fusion (C2) optional | approved | PI brief |
| D-003 | 2026-10-26 | Frozen AdaFace IR101 (primary), frozen ArcFace (secondary); no fine-tuning, no large MoE | approved | PI brief |
| D-004 | 2026-10-26 | Detector: InsightFace; detection failures evaluated explicitly (ABSTAIN / forced non-match) | approved | PI brief |
| D-005 | 2026-10-26 | Training data: DigiFace-1M subset; CelebA optional, not a dependency | approved | PI brief; licence confirmed as R-UDA (research, non-commercial) |
| D-006 | 2026-10-26 | Challenge families: F1 masks, F2 sunglasses, F3 hands/objects, F4 random patches, F5 extreme lighting (non-physical, interpreted separately). Pose (CPLFW) is a benchmark, not a family. Replaces the earlier lower-face/upper-face naming | approved | PI brief |
| D-007 | 2026-10-26 | ROF used only if ethics/legal/licence/access permit → registry status **conditional** | approved (conditional) | PI brief; repo MIT but images web-collected |
| D-008 | 2026-10-26 | Assigned Kaggle dataset `trainingdatapro/human-face-images` → **blocked** (HTTP 404, licence and labels unknown) | design | `docs/dataset_feasibility.md` §1 |
| D-009 | 2026-10-26 | Primary calibration pairing = **identity-disjoint matching**. It replaces "one pair per identity pair" from the earlier draft, which shares identities and is not independent | design | Validity of binomial LTT p-values (A2) |
| D-010 | 2026-10-26 | Controlled risk = error among accepted pairs (selective error); sFMR/sFNMR secondary; the LTT ratio-risk construction is re-checked before Phase 2 [U] | provisional | `protocol_design.md` §2 |
| D-011 | 2026-10-26 | FAR operating points 1% and 0.1% (replace the earlier 1e-2/1e-3/1e-4). **0.1% = estimation-only** under the recommended DigiFace subset (certification needs ≥ 2,995 independent impostor pairs) | provisional | `feasibility_report.py` §A/§C |
| D-012 | 2026-10-26 | Constants: 0.5 pp clean margin, 10:1 cost, 95% confidence, 5% accepted-error target, 95% identity-cluster bootstrap CIs | provisional | PI brief; feasibility in `protocol_design.md` §7 |
| D-013 | 2026-10-26 | Near-duplicate threshold **not fixed**; set only via validated procedure on non-test data (schema-enforced) | design | PI constraint 7; `protocol_design.md` §8 |
| D-014 | 2026-10-26 | Decision rules re-indexed: H2b/H1 success requires ≥ 3 of the **4 physical** families + ROF; F5 is reported separately | provisional | Lighting is not a physical occlusion |
| D-015 | 2026-10-26 | Oracle-family stratification is a Test-ID-only diagnostic (no stratum exists for a held-out family) | design | Logical constraint |
| D-016 | 2026-10-26 | Isolated git repository rooted at the project folder (nested `git init -b main`); Desktop repository untouched; `.gitignore` written before any staging | design | Security pre-flight |
| D-017 | 2026-10-26 | Phase 0 dependencies limited to numpy, pydantic, PyYAML (+ pytest, ruff, mypy); Python 3.12 via uv | design | Minimal dependencies |
| D-018 | 2026-10-26 | No commit or push in Phase 0, despite the standing GitHub workflow rule. The phase brief says "make no commits or pushes"; the more specific instruction wins until the PI confirms | **resolved**: the PI authorised the first push in the Phase 1a brief; commit `3ed5cb1` | Conflicting instructions resolved |
| D-019 | 2026-10-26 | Primary controlled estimand = **class-conditional** sFMR and sFNMR (intersection–union binomial test). The pooled accepted-decision error is secondary and needs Bernoulli class assignment. Resolves D-010 | provisional | `ltt_audit.md` §2–3; SIM-007 |
| D-020 | 2026-10-26 | LTT candidate order and pruning are fixed on **validation data only** (`order_candidates_on_validation`). Calibration data are only counted | design | Fixed-sequence power collapse found in simulation; `ltt_audit.md` §4 |
| D-021 | 2026-10-26 | "Same source" = same root-photo content hash. Occlusion variants and byte-identical copies can never form a genuine pair or count as independent | design | Bug found by `test_variants_and_duplicates_never_form_genuine_pairs` |
| D-022 | 2026-10-26 | Simulation results are stored in `reports/simulations/` and typed in `reports/evidence_registry.yaml`, separate from model results (`results/`, ignored) | design | Research-integrity requirement |
| D-023 | 2026-10-26 | Manifest `sha256` = hash of file bytes (integrity). The pixel-level hash for dedup is computed later (Phase 1b) | design | Avoids an image-decoding dependency in Phase 1a |
| D-024 | 2026-10-26 | Operating points refuse to run with fewer than 10/FAR impostors unless the caller passes an explicit, documented override | design | "No silent operating points" requirement |
| D-025 | 2026-10-26 | Protocol v0.2.0 (provisional): adds `candidate_ordering: validation_only` and assumption A7 (acceptance locality); controlled risk changed per D-019 | provisional | Audit outcome |

## Pending PI decisions (updated after Phase 1a)
1. Download go-ahead: DigiFace 72-img P1 + 5-img P1 (~5.9 GiB). Optionally all three 5-img
   parts (+6.3 GiB) to certify FMR at 0.1%.
2. Accept FAR 0.1% as estimation-only (no certified sFMR at 0.1%) under the recommended
   subset?
3. ROF: ethics/legal confirmation (web-collected real faces; pickle loading in an isolated
   process).
4. Kaggle dataset: provide its licence text/metadata if visible to you, or confirm exclusion.
5. Freeze the constants marked "ready to freeze" in `protocol_design.md` §7 (cost ratio,
   confidence level)?
6. Approve D-019, so the primary estimand becomes class-conditional sFMR/sFNMR, and the 0.5 pp
   clean margin given SIM-006, which needs about 3,000 clean test identities.
