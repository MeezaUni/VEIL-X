# Phase 6 Results: Final Audit, Replication, Hardening and Delivery

Protocol: `docs/phase6_protocol.md`, pre-registered at `ed1bf1e` before any download. Amendment 1
was registered at `168b59b` before any p6 score. Decisions: D-052 to D-056. Audit and claim
matrix: `docs/phase6_audit.md`.

**No model was trained, re-fitted, re-calibrated or re-selected in Phase 6.**

## 0. Summary

| Item | Result |
|---|---|
| **6B: frozen pipeline** | The decision uses **AdaFace only**: M1 = f(cosine, min feature norm, product), with quality-tertile A3 rules. ArcFace was never part of a calibrated decision system. Earlier "dual-backbone" recommendations are corrected (D-052). This is a documentation issue, not an implementation defect |
| **6E: replication** (run `1c291322aa27`, untouched DigiFace 72-image identities 6000–9999) | **R1 replicates.** A3's sFNMR CI excludes 5 % in **4/5** seen-family cells (5.15–6.49 %).<br>**R2 replicates.** No candidate is safe; recommendation **A3** (violations: A3 7, C4 10, N4 10, H4 9).<br>**R3 holds.** A3 sFMR is within target in **15/15** cells |
| **New methodological finding (D-055)** | The frozen M1 score extrapolates below the calibration support of quality:<br>- synthetic non-face pairs are decided MATCH with p ≈ 1;<br>- the ~1 % low-quality tail has A3 sFMR 18–42 % on development data, and **22.97 % on the untouched p6 data** (exploratory), against 1.50 % above the floor |
| **6C / 6D: app safety** | Active-configuration card and calibration acknowledgement; fallback only for declared LFW-style near-square portraits (else refused; also fixes a multi-face defect); support guard; VE-C weights hash-pinned; safe errors; uploads passed through untouched and cache expiry ≤ 3 min (D-054, D-056) |
| **6H: GPU detection** | Not adopted. The onnxruntime CUDA provider needs the CUDA 13 runtime, which is absent, so detection stays on the CPU, unchanged |
| **6F: physical occlusion** | Not evaluated: future work. An IJB-C access checklist is prepared |

## 1. Replication on the untouched p6_test

**Data:**
- DigiFace-1M 72-image parts 6000–7999 and 8000–9999 (official sizes; CRC-32 verified);
- 288,000 images, 4,000 identities × 72;
- 0 decode failures and 0 duplicates against every earlier DigiFace image, Phase 4 included;
- 48,000 pairs (24,000 / 24,000), seed 20261102.

**Run.** Frozen Phase 3 freeze `34409779…` and Phase 4 freeze `bba12900…`, unchanged code path,
one logged access at clean commit `a05246b`. Tables: `reports/experiments/phase6_tables.md`.

### 1.1 A3 by cell (coverage / sFNMR / sFMR, %)

✗ = 95 % identity-jackknife CI excludes the 5 % target; † = point estimate above the target.

| Fold system | Test-ID (seen families) | Test-OOD (held-out family) | E1 (never-seen hair) |
|---|---|---|---|
| F1_mask | 93.8 / 5.90 ✗ / 1.40 | 90.9 / 7.74 ✗ / 2.77 | 93.0 / 5.97 ✗ / 1.20 |
| F2_sunglasses | 89.8 / 6.49 ✗ / 1.73 | 95.7 / 5.76 ✗ / 1.41 | 90.9 / 5.77 ✗ / 1.22 |
| F3_hand_object | 90.0 / 6.41 ✗ / 1.77 | 92.3 / 5.72 ✗ / 1.20 | 90.5 / 5.37 ✗ / 1.22 |
| F4_random_patch | 93.0 / 5.15 † / 1.75 | 85.4 / 7.20 ✗ / 2.03 | 90.9 / 4.83 / 1.26 |
| F5_extreme_lighting | 89.6 / 6.39 ✗ / 2.02 | 94.5 / 4.26 / 1.47 | 90.6 / 5.13 † / 1.42 |

Phase 4 (p4_test) for comparison: A3 Test-ID sFNMR was 5.89 / 6.34 / 6.16 / 5.36 / 6.05 %, all ✗.
The replication is within about 0.3 pp in every fold. Figure:
`reports/figures/phase6_a3_population_shift.svg`.

### 1.2 Registered endpoints

- **R1 (population-shift replication): REPLICATES.**
  - 4 of 5 Test-ID cells have a CI excluding 5 %; the registered criterion was ≥ 3.
  - The random-patch fold is 5.15 % with CI [4.84, 5.48], so its point estimate is above the
    target but the CI is not.
- **R2 (A3 recommendation): REPLICATES.**
  - Violations on the 10 primary cells:

    | Arm | Violations |
    |---|---|
    | A3 | 7 |
    | A4 | 10 |
    | A5 oracle | 9 |
    | B0 | 8 |
    | B1 | 10 |
    | C4 | 10 |
    | N4 | 10 |
    | H4 | 9 |

  - Candidates significantly worse than A3 (Holm), in cells:

    | Candidate | sFNMR | sFMR | Safe |
    |---|---|---|---|
    | C4 | 8 | 3 | no |
    | N4 | 9 | 0 | no |
    | H4 | 6 | 1 | no |

  - Mean coverage gains, which are irrelevant because no candidate is safe: C4 +3.7 pp, N4
    +1.2 pp, H4 −25.6 pp.
  - The registered recommendation rule again yields **A3**.
- **R3 (false-match side): HOLDS.** A3 sFMR is 1.20–2.77 % across all 15 cells, all within
  target.
- **Descriptive.** Clean FNMR of raw similarity is 4.60 % (p4_test 4.81 %; p3_test 1.12 %), so
  the harder genuine pairs of the 72-image population reproduce on new identities. The full
  descriptive results are in `phase6_tables.md`:
  - novelty AUROC;
  - visibility bias;
  - clean margins;
  - fusion ablations.

### 1.3 Amendment 1: the low-quality tail on untouched data (exploratory)

Tallied below the registered floors, per cell.

| Pooled over 15 cells | Pairs | A3 sFMR | A3 sFNMR |
|---|---|---|---|
| q below floor (0.9 %) | 6,631 | **22.97 %** (289 / 1,258 accepted impostors) | 19.81 % |
| q at or above floor | 713,369 | 1.50 % | 5.80 % |

- 0.9 % of pairs account for 5.5 % of all A3 false matches.
- The development-data finding (D-055) reproduces on data never used for any decision about it.
- This is exploratory evidence, labelled as such.

**What this means.**
- Stratum-level LTT control did not translate into control within the extreme low-quality tail.
- The aggregate rates hide a region where the frozen score is unreliable.

## 2. Frozen inference architecture and calibration systems

- **6B: architecture.** See `docs/phase6_audit.md` §2 and D-052.
- **6C: calibration systems.** See `docs/calibration_systems.md`. It lists, for all five fold
  systems:
  - what each was calibrated on: 6,146 pairs over 9,219 synthetic identities, 9,033 of them
    5-image identities;
  - which family was held out;
  - why none is universal;
  - what the `F1_mask` default means.

## 3. Detector fallback assessment (6D)

| Input condition | Research evaluation (Phases 2–5) | Application (Phase 6) |
|---|---|---|
| Corrupt or unsupported file | not applicable | Rejected with a reason code (when it reaches the app). If Gradio's own decoder fails first, a generic "Error" is shown and the upload expires within ≤ 3 min (D-056) |
| One face, good detection and alignment | aligned crop | aligned crop (pixel-identical to offline) |
| Several faces | central face chosen (LFW) | **refused** unless the user selects one |
| Selected face of low quality in a multi-face image | — | **refused** (previously: a whole-image fixed crop, a defect) |
| One face, low score or residual, on an LFW-like portrait | fixed-crop fallback (Phase 4F registered) | fallback **only if** LFW-style framing is declared and the image is near-square (aspect 0.8–1.25); otherwise refused |
| No face, LFW-like portrait | fixed-crop fallback | as above |
| No face, arbitrary photograph | n/a | **refused** (`no_reliable_face`) |
| Non-face content declared pre-aligned | n/a | **support guard**: abstains below the calibration-quality floor |

**Reproducibility.** The research behaviour is reproduced by `allow_fallback=True` and
`apply_support_floor=False`, which the parity scripts pass explicitly. Tests cover every row.

**Live check.**
- A wide noise photo in detector mode was refused with "no reliable face … fallback not
  enabled".
- A noise pair in pre-aligned mode abstained, with the note "frozen rule alone would have output
  MATCH (fused score 1.000)".

## 4. Application hardening (6H)

| Check | Result |
|---|---|
| Frozen inference parity | Phase 5: 1000/1000 decisions, LFW crops pixel-identical, CPU/GPU 100/100. Pipeline code is unchanged except the opt-in guards; parity mode reproduces offline exactly (tested) |
| Artefact integrity | All 10 artefacts SHA-256 verified at load (backbones, SCRFD, 5 VE-C weights, 2 freezes); smoke test level 2 |
| Multiple-face refusal; fallback restriction; support guard | Unit tests; live browser checks |
| Invalid images | Typed rejection (empty, too large, unsupported, truncated, corrupt, too small, blank). Gradio-level decode failures are safe but generic (§3) |
| Calibration configuration display | Active-system card with calibration description and measured results; updates on fold change; second acknowledgement required (tested) |
| Diagnostics isolation | Unchanged (tested: corrupted diagnostics cannot change the decision) |
| Privacy | Loopback-only binding confirmed (`127.0.0.1:7860`). The only external request, Gradio's CDN script, is blocked by the CSP: 3/3 page loads in this session |
| Temp files | Uploads are deleted after reading. Gradio now passes original uploads untouched (`image_mode=None`), so no converted copies are left behind. Leftovers from failed events expire within ≤ 3 min (`delete_cache=(60, 120)`; previously 15–25 min). Stale session directories are removed at the next start |
| Safe errors | Unexpected exceptions produce "internal processing error (Type)"; out-of-memory produces "resources exhausted". No traceback or path appears in the page (tested). Tracebacks appear only in the local server console |
| Resource exhaustion | Queue max 8; file ≤ 20 MB; image side ≤ 4096 px; OOM message |
| Repeatability | Identical output on rerun (smoke test level 2, GPU) |
| CPU / GPU behaviour | Phase 5 consistency; `--device cpu` supported |
| GPU face detection | Not adopted: the CUDA provider is unavailable (CUDA 13 runtime missing); the session silently fell back to the CPU. Adoption would require the CUDA 13 runtime and a new parity study (`reports/experiments/phase6_gpu_detector.json`) |

## 5. Physical-occlusion benchmark status (6F)

- **Not evaluated; no dataset requested or downloaded.**
- The actionable IJB-C checklist is in `docs/physical_occlusion_readiness.md` §7. It covers
  eligibility, institutional approval, the request content, file safety, a training-overlap
  check and pre-registration.
- Alternatives are RMFRD, MFR2 and ROF, each with its own caveats.

## 6. Reproducibility package (6G)

- **Guide:** `docs/reproducibility.md`. It covers:
  - environments, CUDA and versions;
  - data acquisition and licences;
  - integrity checks;
  - commands per phase;
  - seeds;
  - expected outputs;
  - hardware and memory;
  - nondeterminism;
  - troubleshooting;
  - runtimes.
- **Smoke test:** `scripts/smoke_test.py`. It needs no biometric data, and both levels pass.
- **Documentation link check:** `scripts/check_docs.py`.

## 7. Deliverables

- **Paper:** `docs/paper/veilx_paper.md`, with figures in `reports/figures/phase6_*.svg` and
  earlier figures.
- **Presentation:** `docs/presentation/slides.md`.
- **Demo script and viva Q&A:** `docs/presentation/demo_and_viva.md`.

## 8. Limitations that remain

- Synthetic occluders only; no physical-occlusion evidence.
- Synthetic confirmatory faces.
- Five fold-specific calibrations; none universal.
- Guarantees are calibration-population guarantees. The 5 % target failed on 72-image
  identities **twice**, on independent identities.
- The frozen fused score is unreliable in the low-quality tail. The app guards against it; the
  research system is unchanged.
- One backbone pair and one detector.
- Follow-up sets have been re-used. No set inspected earlier is presented as untouched.
- **Not deployment-ready**, and no certified-risk claim is made.
