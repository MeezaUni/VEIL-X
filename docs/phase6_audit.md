# Phase 6 Final Research and Implementation Audit

Phase 6 (final phase): D-052 to D-056; protocol `docs/phase6_protocol.md`; results
`docs/phase6_results.md`.

## 1. Repository, provenance and frozen-artefact verification (2026-11-02)

| Check | Result |
|---|---|
| Git | Branch `main`. Working tree clean at each commit. Remote HEAD equals local HEAD after each push. Phase 5 HEAD `65db8e2` CI success. The Phase 1b backup stash (`stash@{0}`) is preserved, untouched |
| Backbone weights | AdaFace IR-101 and ArcFace IR-101 safetensors match the SHA-256 values pinned in `veilx.models.spec`. Checked at every load (`load_backbone`, `verify_hash=True`) |
| SCRFD `det_10g.onnx` | Matches `5838f7fe…` (app config). Checked at load |
| Phase 3 VE weights (5 folds) | Match the hashes recorded in `reports/experiments/phase3_ve_training_<fold>.json` |
| Phase 4 VE-C weights (5 folds) | Match the hashes in `phase4_vec_training_<fold>.json`. **Gap found:** the app checked only the Phase 4 freeze, not the weights. Hashes are now pinned in `configs/app/veilx_app.yaml` and checked at load (D-054) |
| Frozen decision systems | Phase 3 freeze content SHA-256 `34409779…`; Phase 4 freeze `bba12900…`. Both verified (file hash = recorded hash = recomputed content hash) |
| Dataset provenance | DigiFace from the official host, with official sizes and SHA-256 fingerprints recorded at download. LFW from the Kaggle mirror under D-043, zip SHA-256 recorded. Phase 4 and 6 parts: CRC-32 of every member, 0 decode failures, 0 duplicates against all earlier DigiFace images |
| Partitions | Identity-hash splits (Phase 2) and Phase 3 sub-split bands are deterministic. p4_test and p6_test are separate downloads (identity ranges 2000–5999 and 6000–9999), disjoint from the earlier 0–1999 and the 5-image identities. Checked by `tests/test_untouched.py` and by pixel-hash dedup |
| Access logs | `results/test_runs.jsonl`: 14 entries before Phase 6, each with run id, script, split, freeze hash and git commit (§3) |
| Statistical procedures | Identity-jackknife 95 % CIs. Paired identity-jackknife comparisons with Holm correction. Learn-then-Test fixed-sequence with exact binomial p-values, δ = 0.05. v2 candidate ordering from validation p-values (D-042). Exact Clopper–Pearson bounds for zero-event rates |

## 2. Exact frozen inference architecture (6B; code-traced)

```
image → [SCRFD det_10g, CPU] → 5-point similarity alignment to the ArcFace 112×112 template
        (app: labelled fixed-crop fallback only with declared LFW framing, D-054; multiple faces refused)
      → AdaFace IR-101 (fp16 on GPU): embedding e, pre-normalisation feature norm n
pair  → s = cos(e_A, e_B);  q = min(n_A, n_B)          ← ADAFACE ONLY
      → M1: p = σ(β0 + β1·z(s) + β2·z(q) + β3·z(s·q))  (frozen per fold, Phase 3)
      → stratum = tertile of q (edges from p3_val)
      → A3 rule: MATCH if p ≥ τ_high, NON-MATCH if p < τ_low, else ABSTAIN (frozen LTT thresholds)
      → app-only: ABSTAIN if q < 1st percentile of calibration q (D-055)
ArcFace IR-101: cosine shown as a diagnostic only.
Phase 4 VE-C visibility maps and Mahalanobis novelty: computed after the decision, explanatory only.
```

- **Sources:**
  - `scripts/evaluate_phase3.py` (`ARMS["A3_quality"] = ("M1", "quality")`; `build_table`: `s`
    from `emb_ada`, `q` from `norm_ada`);
  - `veilx.fusion.features.LADDER["M1"] = ("s", "q", "s*q")`;
  - `veilx.app.pipeline.verify` (`decide(rule, s_ada, q)`).
- **Resolution of the discrepancy.** ArcFace enters only M5 / M5c / M5_noCV / MLP. These were
  compared in log-loss and AUC, and **never calibrated or evaluated as three-way decision
  systems**.
  - Phase 3 recommendation 3 and Phase 4 §13 ("dual-backbone + quality") described an
    unvalidated configuration as recommended.
  - This is a **documentation overstatement, not an implementation defect.** The app implements
    exactly the evaluated system: 1000/1000 parity.
  - Errata are added in place (D-052).

## 3. Evaluation-set access inventory (from `results/test_runs.jsonl`; private log, 15 entries)

| Set | Accesses (run id → purpose) | Status for Phase 6 |
|---|---|---|
| **Phase 2 DigiFace test** (5-image identities) | `38d853d8884c` confirmatory (Phase 2); `0ca832e3b3b3` reproducibility audit (identical); `9b660f16061a` Phase 3 follow-up | **Inspected (3×)**: follow-up use only |
| **LFW View 2** (Kaggle mirror, D-043) | `79331a84f520` Phase 2 final; `e97048b9c65f` post-hoc mask audit; `9b660f16061a` follow-up; `6a3c151e18a5` exploratory detector-inclusive; `diag-detinc-1` GT-warp diagnostic; `d57db3ee8923` Phase 4F detector follow-up | **Inspected (6×)**: exploratory / follow-up only |
| **p3_test** (DigiFace 72-image parts 0–1999, Phase 3 split) | `bb5d9244bb3c` Phase 3 confirmatory; `21f5b1e2e71d`, `ff1b090ccb76`, `e879da65ea2b` visibility-map accuracy (no verification scores); `db421cf1744d` Phase 4 follow-up | **Inspected (5×)**: follow-up only |
| **p4_test** (DigiFace 72-image parts 2000–5999) | `903f6a5c7de3` Phase 4 confirmatory (single access) | **Inspected (1×)**: its confirmatory use is spent |
| **LFW View 1 DevTrain / DevTest** | development (detector thresholds, fallback selection, Phase 5 parity) | Development data, never a test set |
| **p6_test** (DigiFace 72-image parts 6000–9999) | never downloaded before 2026-11-02; `1c291322aa27` Phase 6 replication (single access) | **Inspected (1×)**. No untouched identity-disjoint DigiFace 72-image data remains |

**Conclusions:**
- No previously inspected set is presented as untouched.
- Re-splitting or reshuffling any inspected set would not create a new confirmatory result, and
  is not done.

## 4. Claim-to-evidence matrix

Status labels:
- **CONF**: confirmatory (pre-registered, single logged access);
- **FU**: follow-up on a previously observed set;
- **EXPL**: exploratory or post hoc;
- **DEV**: development data;
- **SIM**: simulation.

CIs are 95 % identity-jackknife unless stated.

| # | Claim (as it may be stated) | Experiment / run | Data, partition | Metric and value | Uncertainty | Status | Limitations |
|---|---|---|---|---|---|---|---|
| 1 | Frozen backbones are strong on clean synthetic faces | Phase 2 `38d853d8884c` (reproduced `0ca832e3b3b3`) | DigiFace test_balanced, 61,122 + 61,122 pairs | AdaFace AUC 0.99856, EER 1.28 %; FNMR at cal FAR-1 % 1.42 % | [1.31, 1.53] | CONF | Synthetic faces; 72-image identities harder (AUC 0.99274) |
| 2 | Synthetic occlusion severely degrades verification | Phase 2 `38d853d8884c` | DigiFace LOFO Test-ID / Test-OOD | FNMR at clean threshold 6.8–21.6 % vs 1.4 % clean (AdaFace); held-out random patch 21.6 % | Jackknife CIs in `phase2_final.json` | CONF | Synthetic occluders only |
| 3 | Clean-only calibration fails under occlusion | Phase 3 `bb5d9244bb3c` | p3_test Test-OOD | A1 sFNMR 5.97–19.81 %, all ✗ | CI excludes 5 % | CONF | Calibration-matched population |
| 4 | Quality-stratified A3 kept the 5 % targets on all five held-out families **on the calibration-matched population** | Phase 3 `bb5d9244bb3c` | p3_test Test-OOD | sFNMR 1.08–2.12 %, sFMR 1.23–2.56 %, coverage 88.6–96.8 % | All within target | CONF | Does **not** transfer: see 13 |
| 5 | Learned-visibility stratification (A4) does **not** beat A3 (C3 negative) | Phase 3 `bb5d9244bb3c` | p3_test Test-OOD | A4 random patch sFNMR 8.52 % ✗; paired A4 − A3 +6.40 pp | Holm p < 0.001; worse in 3/5 folds | CONF | — |
| 6 | The learned estimator overestimates visibility on unseen occluders | Gate 3B (`21f5b1e2e71d`, `ff1b090ccb76`, `e879da65ea2b`) | p3_test renders | Held-out random patch IoU 0.13, bias +0.344; lighting IoU 0.005, bias +0.491 | ≤ 20k renders per cell | CONF (map accuracy) | Synthetic occluders |
| 7 | Co-visibility adds no reliable value beyond quality (C1 negative) | Phase 3 `bb5d9244bb3c` | p3_test | 7/10 cells positive, but random patch Δ log-loss −0.035 (harmful) | Holm | CONF | Small seen-family gains |
| 8 | Quality fusion does not cost clean accuracy (H4) | Phase 3 `bb5d9244bb3c` | p3_test clean | M4 clean FNMR 0.68–0.74 % vs M0 1.12 % | Upper CI < 0 | CONF | — |
| 9 | The estimator learned occluder classes; its errors are confident; face-like occluders are undetectable | `phase4_diagnose.py` | p3_val (development) | 88 % of covered cells missed; mean P(visible) 0.85; face patch called visible 0.35–1.00 | Descriptive | DEV / EXPL | Probe set designed after the Phase 3 failure |
| 10 | Conservative VE-C fixes the overestimation | Phase 4 `903f6a5c7de3` | p4_test renders | Random patch share overestimated > 0.10: 94.6 % → 21.9 %; bias +0.34 → +0.07 | Descriptive per cell | CONF (secondary) | F4/F5 design-informed |
| 11 | No visibility-based candidate is safe relative to A3, so recommend A3 | Phase 4 `903f6a5c7de3` | p4_test, 10 primary cells | Violations: A3 6, C4 10, N4 9, H4 9; C4 significantly worse in 8 cells (sFNMR) | Paired jackknife, Holm | CONF | Same population caveat as 13 |
| 12 | Even ground-truth visibility does not beat A3 on a never-seen occluder | Phase 4 `903f6a5c7de3` | p4_test E1 cells | A5 oracle sFNMR 7.51–10.32 % ✗ vs A3 4.65–5.76 % | CI | CONF (descriptive arm) | Oracle not deployable |
| 13 | **Population shift:** p3_cal calibration does not transfer to 72-image identities; every arm violates sFNMR even on seen families | Phase 4 `903f6a5c7de3`; replication: Phase 6 (§5) | p4_test Test-ID | A3 sFNMR 5.36–6.34 % ✗ in 5/5 cells; clean FNMR(M0) 4.81 % vs 1.12 % | CI excludes 5 % | CONF (unexpected in Phase 4; **replicated** pre-registered in Phase 6, claim 24) | Cause (calibration ~98 % 5-image identities: 9,033 of 9,219) is a **post-hoc explanation** (Phase 6 audit) |
| 14 | A3's false-match side stayed within target everywhere in Phase 4 | Phase 4 `903f6a5c7de3` | p4_test, 15 cells | sFMR 1.03–2.59 % | Within target | CONF | Not in the low-quality tail (22) |
| 15 | Novelty flags never-seen occluders | Phase 4 `903f6a5c7de3` | p4_test E1 | AUROC 0.91–0.98 vs seen-family renders; ID FPR 4.6–5.2 % | Descriptive | CONF (secondary) | Misses constant / gradient non-faces (22) |
| 16 | MLP fusion is better than logistic in log-loss; removing co-visibility helps | Phase 4 `903f6a5c7de3` | p4_test | MLP better in 9/10 cells; noCV better in 8/10 | Holm | CONF (secondary) | **Never evaluated as a decision system**; not used |
| 17 | Detector failures need a policy; fallback beats forced non-match | `d57db3ee8923` (View 2) after DevTest selection | LFW View 2 composites, 300 pairs | Fallback worst sFNMR 3.16 %; forced non-match 14.7–16.1 % | Small n | FU / EXPL | View 2 seen before; 300 pairs; synthetic overlays |
| 18 | DigiFace-calibrated rules are conservative on real photographs | `79331a84f520`, `9b660f16061a` | LFW View 2 | FMR 0/3,000 at cal FAR-1 % (≤ 0.10 %, Clopper–Pearson); A3 sFMR ≤ 0.2 % | CP bound | CONF (Phase 2) / FU | Identity overlap with WebFace4M likely; not "valid", just conservative |
| 19 | Clean LFW accuracy 99.58 % (AdaFace) | `79331a84f520` | LFW View 2, official 10-fold | 99.58 % ± 0.26 | Fold SD | CONF | Likely optimistic (training overlap) |
| 20 | The local app reproduces the offline frozen evaluation | Phase 5 parity | p3_val, LFW View 1 DevTrain | 1000/1000 A3 decisions; 244/244 + 48/48 crops pixel-identical; CPU/GPU 100/100 | Exact | DEV | Development data |
| 21 | Overestimated visibility breaks per-stratum validity | SIM-013 | Simulation | Violation in 20 % of replicates at +0.2 bias | Monte Carlo | SIM | Model of the mechanism, not data |
| 22 | **A3 is unreliable below the calibration support of quality** | `phase6_lowq_audit.py` | p3_val + synthetic probes | sFMR 18–42 % for q < 1st percentile of calibration q (~1 % of pairs) vs 1.1–1.5 % above; every non-face probe → MATCH, p ≈ 1 | Small n per fold (≈120 impostors) | DEV / EXPL (Phase 6) | Post hoc; reproduced on p6 (claim 25) |
| 23 | GPU face detection is not adopted | `phase6_gpu_detector.py` | LFW DevTrain | CUDA provider unavailable (needs CUDA 13 runtime); CPU path unchanged | — | DEV | Environment-specific |
| 24 | **The population-shift failure and the A3 recommendation replicate** on 4,000 untouched identities | Phase 6 `1c291322aa27` | p6_test | R1: A3 sFNMR CI excludes 5 % in 4/5 Test-ID cells (5.15–6.49 %). R2: no candidate safe (violations A3 7, C4 10, N4 10, H4 9). R3: sFMR within target 15/15 | CI | CONF (pre-registered replication) | Same population type as p4; synthetic occluders |
| 25 | The low-quality tail reproduces on untouched data | Phase 6 `1c291322aa27`, Amendment 1 | p6_test | below floor (0.9 % of pairs): A3 sFMR 22.97 % (289/1,258) vs 1.50 % | Counts | EXPL (registered before the run) | Descriptive tally |

## 5. Independent final evaluation (6E)

**Why a replication was possible.**
- Every previously registered test set had been inspected (§3).
- One authorised, never-downloaded, identity-disjoint source remained: DigiFace 72-image parts
  6000–9999.
- The owner authorised its download for one pre-registered replication (D-053).
- The protocol was pushed before the download (`ed1bf1e`); Amendment 1 was pushed before any
  score (`168b59b`).

**Results** (run `1c291322aa27`, at clean commit `a05246b`):
- **R1 replicates:** 4/5 Test-ID cells have a CI excluding 5 %.
- **R2 replicates:** A3 again recommended.
- **R3 holds:** sFMR 15/15 within target.

What was *not* done:
- no reshuffling or re-splitting of any inspected set;
- no tuning on p6.

Detailed tables are in `docs/phase6_results.md` §1.

## 6. Claims corrected in Phase 6

| Location | Previous wording | Correction |
|---|---|---|
| Phase 4 results §13; Phase 3 recommendation 3; roadmap §4h | "Dual-backbone + quality" fused score as part of the recommended / deployable architecture | The validated system is AdaFace-only M1 + A3. Dual-backbone fusion is unvalidated as a decision system (D-052) |
| Phase 4 §14 | "~8 ms per pair" | Batch throughput, not app latency (already corrected in Phase 5) |
| Phase 4 §13 / Phase 5 | Fixed-crop fallback as the registered detector policy, also applied in the app | Valid only for LFW-like framing. The app now refuses it otherwise; this also fixes a multi-face defect (D-054) |
| Phase 3 §0 ("quality stratification … stayed valid") | Valid on Test-OOD of the calibration-matched population | Scoped: it did **not** stay valid under population shift (claim 13). It is also unreliable in the low-quality tail (claim 22) |
| Any "guarantee" wording | — | LTT guarantees hold only for exchangeable data from the calibration distribution. No OOD, population-shift or per-sub-region guarantee |
| App docs (Phase 5) | Implicit that every pair below τ_low / above τ_high is decided | App-only abstention below the support floor (D-055) |

## 7. Privacy and security findings (Phase 6)

| Finding | Severity | Resolution |
|---|---|---|
| A low-quality face *selected* in a multi-face image received a whole-image fixed crop | Defect (app) | Refused (D-054); tested |
| The fixed crop was applied to any no-face image, including arbitrary photographs | Safety | Opt-in with declared LFW framing and a near-square image; otherwise refused (D-054) |
| Non-face inputs were decided MATCH with p = 1.000 (frozen score extrapolation) | Serious (method) | Reported (D-055); app support guard; frozen system unchanged |
| VE-C weights were not hash-checked by the app | Integrity gap | Pinned and checked (D-054) |
| Gradio converted non-RGB uploads into copies, so originals stayed until cache expiry (15–25 min). Uploads abandoned by a Gradio-level decode failure also stayed | Privacy (retention) | Original uploads passed untouched (`image_mode=None`); cache expiry ≤ 3 min (D-056); tested and verified live |
| Unexpected exceptions could surface as Gradio errors | Robustness | Safe messages without traceback or paths; OOM reported (D-054). Gradio-level decode failures show a generic "Error" |
| Gradio's CDN script request | Privacy | Blocked by the CSP (re-verified) |
| Kaggle token | Credential | Untouched, never printed or committed |
| Biometric data in git | — | None: privacy scan PASS on every commit; screenshots kept outside the repository |

## 8. Final readiness assessment

| Dimension | Assessment |
|---|---|
| **Scientific record** | Complete for the stated scope:<br>- pre-registered phases, logged accesses, negative results reported;<br>- the population-shift finding replicated;<br>- a new failure mode reported, not hidden.<br>Physical-occlusion validation is outstanding |
| **Reproducibility** | Documented end to end, with a data-free smoke test. Frozen artefacts are hash-verified. Full reproduction needs the private data root and licences |
| **Application** | A sound local research demonstration: parity-verified, local-only, abstains honestly, displays its limitations. **Not deployment-ready, not certified** |
| **Claims** | Every major claim maps to evidence (§4). Overstatements are corrected (§6) |

