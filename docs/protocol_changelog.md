# Research Protocol Change Log

Every revision of `configs/research_protocol.yaml` is listed here with its reason. Decision IDs
refer to `docs/decision_log.md`. Constants change status only through an entry here.

## v0.5.0 — 2026-10-27 (Phase 3 prospective protocol)

| Area | Change | Reason |
|---|---|---|
| `phase3` (new) | Sub-splits of the never-scored DigiFace train identities (p3_train/val/cal/test = 50/10/15/25 %, salt `veilx-p3`), seed 20261027, v2 ordering default with v1 comparison, confirmatory set p3_test, follow-up sets, Holm families, visibility-estimator grid/epochs/patience | Pre-registration before any Phase 3 training (D-046); a genuinely untouched confirmatory test set |

The full design is in `docs/phase3_protocol.md`. All Phase 2 constants are unchanged.

## v0.4.0 — 2026-10-27 (Phase 2: dataset expansion, bounded evaluation design)

Status: **provisional** (protocol-level). No constant changed status. No test result existed
when this version was written.

| Area | Change | Reason |
|---|---|---|
| `data_preparation.parts` | + `subjects_133333-166665_5_imgs`, `subjects_166666-199998_5_imgs` (official host, Content-Length match, CRC-32 OK, SHA-256 recorded) | Phase 2 expansion approved by the PI; triples calibration capacity (D-038) |
| `data_preparation` counts | Recomputed from the rebuilt manifest (see the section values) | New parts |
| Split assignment | **Unchanged rule** (identity-hash bands), so every Phase 1b identity keeps its split | Preserves the existing assignments |
| `genuine_pairs_per_identity_cap` | 10 → per-set mapping `{train: 10, val: 3, test_balanced: 3}` | Bounds the occluded-render budget (≈ 2.6 M renders per backbone instead of ≈ 13 M); weighting stays identity-uniform inside each set (D-026, D-038) |
| `occlusion_plan_sets` (new) | LOFO plans for `val:val`, `cal:cal_independent`, `test_id:test_balanced`, `test_ood:test_balanced`; train-role plans deferred to Phase 3; `test_far` clean-only | Phase 2 trains nothing; `test_balanced` now holds enough impostors for FMR at 1 % under occlusion |
| `stress_max_pairs` (new) | 20,000-pair seeded subsample of `test_balanced` | Combined occlusions are exploratory only |
| Near-duplicate review | Exhaustive review search not re-run (`VEILX_NEAR_DUP_REVIEW=0`); the rule stays unvalidated and is not applied; exact pixel dedup still runs | Phase 1b showed the nominations are base-rate noise (D-031) |
| `data_preparation.summary_report` | `reports/data/digiface_v0.4.0_summary.json` (the Phase 1b summary is kept) | Versioned outputs |

Phase 1b manifests, pairs and occlusion plans are archived in the private data root
(`manifests/archive_v0.3.0/`), not overwritten.

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
