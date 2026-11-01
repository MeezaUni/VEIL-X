# Phase 6 Final Research and Implementation Audit

> Status: in progress (Phase 6, D-052, D-053). Sections 1–2 and 4–8 are completed after the
> pre-registration commit. §3 was written before any Phase 6 data existed.

## 3. Evaluation-set access inventory (from `results/test_runs.jsonl`; private log, 14 entries)

| Set | Accesses (run id → purpose) | Status for Phase 6 |
|---|---|---|
| **Phase 2 DigiFace test** (5-image identities) | `38d853d8884c` confirmatory (Phase 2); `0ca832e3b3b3` reproducibility audit (identical); `9b660f16061a` Phase 3 follow-up | **Inspected (3×)**: follow-up use only |
| **LFW View 2** (Kaggle mirror, D-043) | `79331a84f520` Phase 2 final; `e97048b9c65f` post-hoc mask audit; `9b660f16061a` follow-up; `6a3c151e18a5` exploratory detector-inclusive; `diag-detinc-1` GT-warp diagnostic; `d57db3ee8923` Phase 4F detector follow-up | **Inspected (6×)**: exploratory / follow-up only |
| **p3_test** (DigiFace 72-image parts 0–1999, Phase 3 split) | `bb5d9244bb3c` Phase 3 confirmatory; `21f5b1e2e71d`, `ff1b090ccb76`, `e879da65ea2b` visibility-map accuracy (no verification scores); `db421cf1744d` Phase 4 follow-up | **Inspected (5×)**: follow-up only |
| **p4_test** (DigiFace 72-image parts 2000–5999) | `903f6a5c7de3` Phase 4 confirmatory (single access) | **Inspected (1×)**: its confirmatory use is spent |
| **LFW View 1 DevTrain / DevTest** | development (detector thresholds, fallback selection, Phase 5 parity) | Development data, never a test set |
| **DigiFace 72-image parts 6000–9999** | never downloaded before 2026-11-02 | **The only untouched, identity-disjoint data.** Used once for the pre-registered replication (`docs/phase6_protocol.md`) |

**Conclusions:**
- No previously inspected set is presented as untouched.
- Re-splitting or reshuffling any inspected set would not create a new confirmatory result, and
  is not done.
