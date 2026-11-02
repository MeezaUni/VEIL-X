# Phase 6 Pre-registered Replication Protocol: Frozen Systems on a New Untouched Partition

> **Registered 2026-11-02.** This was before any image of the Phase 6 partition was downloaded,
> scanned or scored (D-053; owner authorisation 2026-11-02). It is committed and pushed before
> the download starts. Any later change is an amendment, labelled in every result it touches.
> **No model is trained, re-fitted, re-calibrated or re-selected.** The result is reported
> whatever it shows.

## 1. Why a replication, and what it can and cannot show

Every previously registered test partition has been accessed at least once:

| Set | Accesses |
|---|---|
| Phase 2 DigiFace test | 3 |
| p3_test | 5 |
| p4_test | 1 |
| LFW View 2 | 6 |

See `docs/phase6_audit.md` §3. None of them can serve as an untouched confirmation. The only
remaining untouched, identity-disjoint data in VEIL-X's authorised source is **DigiFace-1M
72-image parts 6000–7999 and 8000–9999** (4,000 identities, never downloaded).

**What the replication can show:**
- whether the Phase 4 findings hold on fresh identities from the *same kind* of population
  (72-image DigiFace identities). These findings are the population-shift failure of A3's
  sFNMR target, and "no visibility-based candidate is safe relative to A3";
- whether they hold with fresh pairs and fresh occlusion draws.

**What it cannot show:**
- behaviour on a different population type, on real photographs, or on physical occluders;
- that the 5 % target holds anywhere other than the calibration distribution.

## 2. Partition (registered in `src/veilx/data/untouched.py`, `P6`)

- **Source:** official DigiFace-1M host, `subjects_6000-7999_72_imgs.zip` (2,940,940,004 bytes)
  and `subjects_8000-9999_72_imgs.zip` (2,927,632,784 bytes). Sizes from HTTP HEAD on
  2026-11-02. The licence is R-UDA, non-commercial research.
- **Preparation:** `scripts/prepare_phase4.py --partition p6`, the Phase 4 procedure unchanged:
  - **integrity:** official size, member-name pattern, and CRC-32 of every member, then safe
    extraction;
  - **dedup:** pixel SHA-256 dedup against **every** earlier DigiFace image (Phases 1–3 and the
    Phase 4 partition);
  - **pairs:** `p6_test_pairs` are clustered and balanced, 12 × n_identities, with a genuine
    cap of 6;
  - **plans:** LOFO occlusion plans Test-ID / Test-OOD per held-out family, plus the
    evaluation-only E1 hair-strands plan.
- **Seed:** **20261102**, registered here (Phase 4 used 20261101).
- **Storage:** private `manifests/phase6`, `extracted/phase6`, `features/phase6/p6`. Nothing
  from these directories enters git, except the aggregate summary
  `reports/data/phase6_data_summary.json`.

## 3. Frozen systems (unchanged)

| System | Freeze | Content |
|---|---|---|
| Phase 3 freeze | `34409779ae8f…` | Per fold: M1 logistic fused score; A3 quality-stratified rules; A4 (Phase 3 learned visibility); A5 oracle; B0/B1 global rules |
| Phase 4 freeze | `bba129002cf8…` | VE-C temperature and shift, Mahalanobis novelty, and the candidate rules C4 (conservative), N4 (novelty-aware), H4 (quality-first hybrid) |

Backbones are frozen AdaFace and ArcFace IR-101 (WebFace4M). Extraction uses
`scripts/extract_phase4.py --set p6`, the Phase 4 code path. Evaluation uses
`scripts/evaluate_phase4.py final --final --partition p6`, the Phase 4 confirmatory code path.
The run is logged in `results/test_runs.jsonl` and is the **single** access.

## 4. Questions and decision rules (registered before any p6 data exists)

Targets: primary (sFNMR ≤ 5 %, sFMR ≤ 5 %), δ = 0.05, ordering v2. Uncertainty is the 95 %
identity-jackknife CI, as in Phase 4.

- **R1 (primary): replication of the population-shift failure.** In Phase 4, A3's sFNMR CI
  excluded the 5 % target in 5/5 Test-ID (seen-family) cells.
  - **R1 replicates** if A3's sFNMR CI excludes the target in **≥ 3 of the 5 Test-ID cells**.
  - Otherwise R1 does **not** replicate. That is reported as such, and the Phase 4 finding is
    then described as not robust across draws.
- **R2 (primary): replication of the A3 recommendation.** The Phase 4 P1 safety and P2 benefit
  endpoints are recomputed unchanged on the 10 primary cells (Test-OOD × 5, E1 × 5), by
  `endpoints()`.
  - **R2 replicates** if the registered recommendation rule again yields **A3**: no candidate
    in {C4, N4, H4} is both safe and better.
  - If a candidate is safe and better on p6, that is reported as a **failure to replicate**.
    It does **not** change the frozen recommendation. Overturning Phase 4 on one replication
    would itself be post hoc.
- **R3 (secondary): false-match side.** A3's sFMR is within target (point estimate ≤ 5 %, no
  violation) in all 15 cells. This is reported per cell.
- **Descriptive:**
  - coverage;
  - violations of every arm (A3, A4, A5 oracle, B0, B1, C4, N4, H4);
  - clean margin (the clean FNMR of raw similarity, as in Phase 4);
  - VE / VE-C visibility bias;
  - novelty AUROC.

  The Phase 4 vs Phase 6 comparison is descriptive only; no pooled test is planned.

## 5. Procedure and stopping

1. Commit and push this protocol, the `P6` registration and the partition-parameterised scripts.
   The Phase 4 path is verified to be bit-identical: rebuilt p4 pairs and plans match the stored
   ones.
2. Download (`scripts/fetch_digiface.py`), verify, prepare and summarise. Aggregates only are
   inspected; no scores.
3. **Pilot.** Run a memory-monitored extraction pilot (`--pilot`), then the full extraction.
4. Run **one** logged evaluation run.
5. **Failures.** If the run crashes, it may be resumed or re-run with an identical command, and
   each attempt is logged. Any code fix is documented as an amendment, and must not depend on
   p6 results.
6. **No tuning.** No threshold, rule, model or partition choice may depend on p6 data.

## 6. Claims policy

- A replicated result means "replicated on a second untouched draw of 72-image DigiFace
  identities with synthetic occluders". It does not mean "validated for real photographs or
  physical occlusions".
- No formal out-of-distribution or population-shift guarantee is claimed, whatever R1 shows.

## 7. Amendment 1 (2026-11-02; registered before any p6 extraction or score)

**Why.** A Phase 6 development audit found a failure mode of the frozen M1 score below the
calibration support of quality (`scripts/phase6_lowq_audit.py`;
`reports/experiments/phase6_lowq_audit.json`). Development data only:
- **Synthetic non-face probes.** Pairs of random-noise, constant or gradient images get q of
  5–10 and are decided **MATCH** with p ≈ 1.000.
- **p3_val.** The ~1 % of pairs with q below the 1st percentile of calibration quality show A3
  sFMR of 18–42 %, against 1.1–1.5 % above it.

**What is added.** One **exploratory, descriptive** quantity is added to the single p6 run. Per
cell, A3's counts are tallied below and at-or-above the registered floors:
- genuine and impostor pairs;
- false matches and false non-matches;
- accepted pairs and abstentions.

The floors are fixed from development data in `configs/app/veilx_app.yaml`
(`decision.q_support_floor`: F1 10.82, F2 10.87, F3 10.86, F4 12.45, F5 11.01).

**What does not change.** The registered endpoints R1–R3, every frozen system and every decision
are unchanged. The tally is not a decision rule. It does not enter R1–R3, and it is labelled
exploratory wherever reported.
