# Research Protocol Change Log

Every revision of `configs/research_protocol.yaml` is listed here with its reason. Decision IDs
refer to `docs/decision_log.md`. Constants change status only through an entry here.

## v0.3.0 — 2026-10-26 (Phase 1b: real data, occlusion engine, interval method)

Status: **provisional** (protocol-level). Individual constants frozen where justified (see below).

| Area | Change | Reason |
|---|---|---|
| `data_preparation` (new) | Verified DigiFace-1M archive sizes, SHA-256 fingerprints, image size, alignment finding, final image/identity counts per split, pair-generation caps | Phase 1b download and integrity pass (D-027, D-028) |
| `pairs.genuine_pairs_per_identity_cap` | 20 → **10** | Equal per-identity weight for 72-image and 5-image identities (5 images give at most 10 genuine pairs); keeps the clustered test population matched to the identity-uniform calibration population (D-026) |
| `deduplication.near_duplicate` | Method `phash_and_embedding_cosine` → `phash_hamming_and_thumbnail_ncc`; Hamming and NCC thresholds set **only from the validation record** | Face-embedding similarity is not a duplicate signal (PI instruction); rule selected on train, confirmed on val (D-029) |
| `occlusion` (new) | Generator version, template landmarks, severity/visibility definitions, per-level parameters, stress combinations | Occlusion engine (D-030, D-031) |
| `calibration_plan` (new) | Identity-disjoint calibration capacity from the real split | Real counts replace planning estimates |
| `bootstrap.interval_method*` | Interval method selected per statistic from the SIM-010 coverage study | Phase 1a FNMR coverage 93.3 % (D-032) |
| `constants` | `confidence_level` and `cost_ratio_false_match_to_false_non_match` **frozen** (PI-approved); `far_operating_points` **frozen** with FAR 0.1 % estimation-only; `accepted_decision_error_target`, `clean_noninferiority_margin_pp` remain **provisional** | PI approvals of 2026-10-26; power validation for the clean margin pending (SIM-006) |

## v0.2.0 — 2026-10-26 (Phase 1a)
- Primary controlled risk → class-conditional sFMR/sFNMR (D-019); `candidate_ordering:
  validation_only` (D-020); assumption A7 acceptance locality (D-025).

## v0.1.0 — 2026-10-26 (Phase 0)
- Initial provisional protocol.
