# Phase 5 Results: Local Research Application and End-to-End Validation

> **Scope.** This phase is a local, research-only demonstration of the frozen VEIL-X system.
> - **No model was trained, re-fitted or re-selected:** Phase 4 model selection is closed
>   (D-050), and the app configuration is recorded in D-051.
> - **No claim** is made that the 5 % target holds under population shift. Phase 4 showed that
>   it does not.
> - All numbers below are measured on **development data only**: DigiFace `p3_val` and LFW View 1
>   DevTrain. No test partition was used.

## 1. Architecture

```
upload ─▶ inputs.load_image_bytes ─▶ face.FacePreparer ─▶ backbones (frozen) ─▶ decision.decide ─▶ report/ui
          (validation, EXIF,        (SCRFD + 5-pt        AdaFace IR-101 → s, q   (M1 fused score,     (HTML, rule bar,
           RGB; typed rejection)     alignment; labelled  ArcFace IR-101 →        A3 strata, frozen    abstention reasons,
                                     fixed-crop fallback; diagnostic only         thresholds)          limitations)
                                     multiple faces →                                    │
                                     refused unless chosen)                              ▼
                                                                         diagnostics (Phase 4 VE-C maps,
                                                                         novelty) — computed AFTER the
                                                                         decision, never an input to it
```

Each layer is its own module in `src/veilx/app/`, as the master prompt required:

| Module | Responsibility | Needs a model stack? |
|---|---|---|
| `config` | Frozen YAML config; loads the frozen Phase 3 decision after a SHA-256 check (`FrozenRule`) | no |
| `inputs` | File validation with typed rejection codes | no |
| `face` | SCRFD detection, alignment, fallback and multiple-face policy (detector SHA-checked, lazy) | onnxruntime, cv2 |
| `decision` | M1 fused score, A3 stratum, three-way decision and its reasons | no |
| `pipeline` | Orchestration, lazy backbone and diagnostics loading, per-stage timings | torch |
| `diagnostics` | Phase 4 VE-C visibility maps and Mahalanobis novelty (Phase 4 freeze SHA-checked) | torch |
| `report` | Wording, disclosures, rule bar, overlays | no |
| `ui` | Gradio Blocks layout; `run_comparison` handles each request | gradio |
| `server` | FastAPI wrapper: loopback-only check, CSP and no-store headers | fastapi |
| `tempfiles` | Per-session temp directory with owner PID; stale-directory cleanup | (psutil) |

`decision.decide` uses `veilx.selective.decisions_from_thresholds`, the function used by the offline
Phase 3/4 evaluation, so there is no second implementation of the rule.

## 2. Launch

```powershell
uv pip install --python .venv-gpu\Scripts\python.exe -e ".[gpu,app]"   # torch from the CUDA wheel index
$env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"
.venv-gpu\Scripts\python.exe scripts\run_app.py                   # http://127.0.0.1:7860
.venv-gpu\Scripts\python.exe scripts\run_app.py --device cpu --port 7861 --no-examples
```

The app refuses to start if any frozen file's SHA-256 differs from `configs/app/veilx_app.yaml`.
See `docs/app_user_guide.md`.

## 3. Interface

**Tab "Compare two images":**
- **Inputs:** two uploads; an authorised-use confirmation required for every comparison.
- **Options:**
  - frozen A3 system (fold);
  - pre-aligned mode;
  - explicit face index for multi-face images;
  - diagnostics on or off.
- **Outputs:**
  - **result:** MATCH / NON-MATCH / ABSTAIN in neutral colours (no green, no "safe" or
    "certified" state), with the fused score labelled "not a probability of identity";
  - **decision-rule region:** a bar of the stratum's frozen thresholds, with the pair's score
    marked;
  - **abstention explanation:** the rule condition that applied (described as "not a causal
    explanation");
  - **detector and alignment status** per image, with detector scores and residuals, annotated
    detections with face indices, and the crop actually used (fallback labelled);
  - **similarity and quality table:** AdaFace cosine, feature norms, pair quality, stratum, fused
    score and thresholds (decision inputs); ArcFace cosine and AdaFace–ArcFace disagreement
    (diagnostic only);
  - **visibility maps** (blue visible / orange not), predicted visible fractions per region, and
    novelty scores against the frozen threshold. All are marked *explanatory only — not used in
    the decision*;
  - per-stage timings and the device.

**Other tabs:**
- **Calibration & limitations:** the disclosures of §8.
- **Why visibility is not decisive:** a table of the measured Phase 3/4 findings.
- **Model & experiment information:** checkpoints, SHA-256 values and the frozen rules of all
  five systems.
- **Synthetic examples** (optional): four DigiFace pairs generated at launch, in the private temp
  directory, from the local data root. Nothing is bundled.

## 4. Exact frozen configuration (`configs/app/veilx_app.yaml`)

| Component | Frozen value |
|---|---|
| Decision system | Phase 3 freeze, content SHA-256 `34409779ae8f69f3…a782c3`; arm `A3_quality`, target primary (sFNMR ≤ 5 %, sFMR ≤ 5 %, δ 0.05), ordering v2; fused score M1 = logistic(s, q, s·q) |
| Default system | `F1_mask`: the first registered fold, chosen independently of results (D-051). The other four are selectable |
| Diagnostics | Phase 4 freeze `bba129002cf8fa06…cff84cc`: VE-C fold weights, temperature, lower-bound shift, Mahalanobis novelty threshold |
| Detector | SCRFD `det_10g.onnx` (SHA-256 `5838f7fe…`), det_thresh 0.5, NMS 0.4; usable if score ≥ 0.6 and alignment residual ≤ 7.3354 (Phase 4F) |
| Fallback | Phase 2 fixed similarity transform (LFW View 1 DevTrain mean landmarks) after resizing to 250 × 250; labelled `FALLBACK` |
| Backbones | CVLface AdaFace IR-101 WebFace4M (decision); ArcFace IR-101 WebFace4M (diagnostic); fp16 autocast on CUDA, fp32 on CPU |

Frozen A3 rules. The fused-score thresholds are (τ_low, τ_high), and the abstention band is
[τ_low, τ_high). q_mid and q_high have τ_low = τ_high, so **only the q_low stratum can abstain**.

| System | q tertile edges | q_low | q_mid | q_high |
|---|---|---|---|---|
| F1_mask (default) | 19.895 / 21.966 | 0.154, 0.833 | 0.452 | 0.673 |
| F2_sunglasses | 19.069 / 21.564 | 0.081, 0.885 | 0.448 | 0.448 |
| F3_hand_object | 19.370 / 21.685 | 0.070, 0.879 | 0.452 | 0.634 |
| F4_random_patch | 19.824 / 21.866 | 0.072, 0.854 | 0.426 | 0.426 |
| F5_extreme_lighting | 19.126 / 21.533 | 0.060, 0.862 | 0.421 | 0.595 |

**Not used:**
- the exploratory MLP fused score;
- co-visibility or any visibility or novelty quantity in the decision;
- any re-calibration.

## 5. Parity with the offline frozen evaluation (`reports/experiments/phase5_parity.json`)

**(a) DigiFace, pre-aligned path.** 1,000 p3_val pairs of the F1 plan. 762 of them are occluded,
re-rendered from their recipes; all 1,000 render keys were found in the offline store.

| Quantity | Result |
|---|---|
| A3 decision agreement with the offline `evaluate_phase3` | **1000 / 1000** |
| max \|Δ fused score\| | 0.0025 (mean 7.4e-05) |
| max \|Δ cosine\|, max \|Δ quality\| | 0.0004, 0.0082 |
| App decision counts (MATCH / NON-MATCH / ABSTAIN) | 473 / 490 / 37 |

The offline embeddings were stored as float16 and the app computes them fresh, which explains the
tiny differences.

**(b) LFW View 1 DevTrain, detector path.** 300 images; offline detection succeeded for 299.

| Outcome | Images |
|---|---|
| Single usable face, aligned | 244; crop pixel-identical to the offline aligned crop in 244 / 244 (max pixel difference 0) |
| Several faces: refused by the app | 49. With the offline (central) face selected explicitly, 48 / 48 crops are pixel-identical |
| Fixed-crop fallback | 7 |

The refusals are intended. Phase 2/4 chose the central face automatically; the app never chooses
a person (D-051).

**(c) CPU (fp32) vs GPU (fp16), 100 pairs:**
- minimum AdaFace embedding cosine 0.99991;
- A3 decisions agree on **100 / 100**;
- max |Δ fused score| 0.0092.

## 6. Latency and memory (`reports/experiments/phase5_benchmark.json`)

**Conditions:**
- one pair per request, as in interactive use;
- NVIDIA GeForce RTX 4060 Laptop GPU (8 GB);
- SCRFD on onnxruntime **CPU**;
- warm figures are medians, with p95 in parentheses;
- LFW detector path: 40 pairs on GPU, 10 on CPU. Totals cover completed comparisons only; pairs
  refused for multiple faces stop after detection.

| Stage (warm) | GPU: LFW photo (detector) | GPU: pre-aligned crop | CPU: LFW photo | CPU: pre-aligned |
|---|---|---|---|---|
| Detection + alignment | 174 (182) ms | 0.01 ms | 186 (191) ms | 0.01 ms |
| AdaFace embedding (2 images) | 24 (50) ms | 19 (25) ms | 165 (182) ms | 149 (162) ms |
| ArcFace embedding (2 images) | 21 (52) ms | 18 (20) ms | 139 (155) ms | 136 (145) ms |
| Fused score + A3 decision | 0.07 ms | 0.07 ms | 0.08 ms | 0.08 ms |
| Diagnostics (VE-C, novelty) | 6 (9) ms | 5 (6) ms | 18 (20) ms | 17 (18) ms |
| **Total** | 226 (285) ms | 43 (49) ms | 509 (524) ms | 303 (319) ms |

| Cold start | GPU | CPU |
|---|---|---|
| Load SCRFD | 141 ms | 68 ms |
| Load AdaFace | 1215 ms | 977 ms |
| Load ArcFace | 1002 ms | 1025 ms |
| Load frozen rule + VE-C | 363 ms | 288 ms |
| First comparison (LFW photo) | 670 ms | 497 ms |

**Memory:**
- peak GPU memory allocated: **550 MiB**;
- peak process RSS: 2.20 GiB, both devices loaded in one process. The memory monitor recorded
  a minimum of 10.1 GB system RAM available.

**Reading the numbers:**
- **GPU:** face detection on the CPU dominates. Both backbones together take about 40–45 ms.
- **CPU-only:** works at about 0.3–0.5 s per comparison.
- **Earlier throughput figure:** the "~2 ms per image" figure from Phase 2 is **batch-256 backbone
  throughput, not app latency**.
- **Lazy loading:** models load lazily, and the launcher warms them up before serving.

## 7. Privacy and security checks performed

| Check | Result |
|---|---|
| Bind address | uvicorn on `127.0.0.1`. `server.local_only_headers` raises for any non-loopback host, and the config schema allows only loopback, `share: false` and `analytics: false` (tested) |
| Third-party requests | Browser session (Playwright) with network log: the only external request, Gradio's CDN `iframeResizer` script on cdnjs, was **blocked by the CSP** (`default-src 'self'`). All other requests went to 127.0.0.1 |
| Headers | `Content-Security-Policy`, `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`. FastAPI docs, the Gradio API page, SSR, monitoring, PWA and MCP are disabled |
| Uploads | Read, then deleted from the session temp directory, also when a comparison is refused (tested; checked live: after a comparison of synthetic example 4 the session directory held only the example sources, owner.pid and Gradio's displayed output images) |
| Output images | Gradio writes the images it displays (detections, crops, overlays) into the same private session directory to serve them to the browser. They are deleted by Gradio's cache cleanup (files older than 15 min, checked every 10 min) and with the whole directory at exit. A directory left by a crash is removed at the next start |
| Embeddings and results | Kept in memory for one request only. No database, no gallery, no 1:N search, no logging (access log off) |
| Repository | `scan_sensitive.py`: no images, arrays, manifests, weights, secrets or absolute user paths. Parity and benchmark JSONs hold aggregates only |

**Live session (2026-11-02):**
- Example 4 (an F1 mask occluder on B) gave **ABSTAIN**: fused score 0.542 inside the q_low band
  [0.154, 0.833); predicted visible fraction B 0.39 (mouth 0.00); novelty typical.
- A comparison without the authorised-use confirmation was refused, and its inputs were deleted.
- A directory left by a stopped process was removed at the next start.
- **Bug found and fixed during this check:** Gradio passed the example files' own paths through,
  so the per-request discard deleted an example's source. Examples are now loaded as disposable
  copies (regression test added).

**Screenshots** were taken with every face image blurred (CSS blur and greyscale):
- the comparison tab with the ABSTAIN result;
- the limitations tab.

They are **not committed**: the repository's privacy scan refuses all image files by design, and
DigiFace images are not redistributed. Running `scripts/run_app.py` with a data root reproduces
the synthetic demo.

## 8. Limitations (shown in the app)

- **Population shift:** the 5 % selective-error target was **not** maintained on the Phase 4
  population (A3 5.4–6.3 % on seen families, up to 7.6 % held-out; run `903f6a5c7de3`).
- **Why A3:** it is recommended because it did best among the alternatives, not because it is
  universally safe.
- **Calibration scope:** calibration is population- and preprocessing-specific. Real photographs,
  cameras, demographics and alignment pipelines are other populations. LFW parity in §5 shows
  that the *pipeline* matches; it does not show that the *calibration* transfers.
- **Occlusions:** physical occlusions are unvalidated (all occluders synthetic).
- **No OOD guarantee:** there is no formal out-of-distribution guarantee.
- **Not deployment evidence:** LFW and DigiFace results do not establish deployment readiness.
- **Fallback assumption:** the fixed-crop fallback assumes LFW-like framing. On other framings it
  produces a crop that is not a face, which is why it is labelled.
- **Diagnostics are imperfect:** visibility diagnostics miss face-like occluders and were trained
  on synthetic occluders. "Typical" novelty does not mean "unoccluded".

## 9. Tests

- **`tests/test_app_core.py`** (23 tests; no model stack needed):
  - config integrity and local-only rules (parametrised rejections);
  - frozen-SHA enforcement;
  - input validation (empty, too large, unsupported, truncated, GIF, small, blank);
  - all three decision states;
  - parity of `decide` with the offline `evaluate_phase3.decide` on 500 random (s, q);
  - face statuses (aligned, multiple faces refused, selection, fallback, pre-aligned);
  - diagnostics cannot change the decision;
  - multiple faces → ABSTAIN with reason; fallback flagged;
  - wording (no probability claim, no "certified");
  - CSP and loopback; launcher has no `share=True` or `0.0.0.0`;
  - temp-directory and upload cleanup;
  - authorised-use gate;
  - example files survive a comparison (disposable copies).
- **`tests/test_app_models.py`** (real frozen models; skipped without the private data root):
  - hash-verified loading;
  - real-pair diagnostics isolation;
  - CPU/GPU consistency.
- **Full suites:**

  | Environment | Result |
  |---|---|
  | CPU (`.venv`) | 397 passed, 3 skipped (real-model tests need torch and the data root) |
  | GPU (`.venv-gpu`, models present) | 410 passed |
  | Quality gate | ruff, format, mypy strict, pytest, config validation, privacy scan, torch tests: PASS |

## 10. Commits

- `b53c5d5`: application code and tests.
- the following commit: parity/benchmark results (measured at clean commit `b53c5d5`), the example-copy fix and
  documentation.

## 11. Recommendations for Phase 6 (not started)

1. **Deployment-population calibration.** Re-run LTT calibration of A3 on pairs from the intended
   population and preprocessing. This is the only route to a valid guarantee there (D-050).
2. **Physical occlusions.** Seek access to a licensed physical-occlusion benchmark (e.g. IJB-C
   occlusion annotations) under its terms, and evaluate the frozen system without retuning.
3. **Detector.** SCRFD on the CPU (about 190 ms per pair) dominates GPU latency. A GPU execution
   provider could reduce this. Its effect on crop parity must be measured before adopting it.
4. **Multi-face UX.** Keep the refusal policy; consider showing face thumbnails for selection.
5. **Benchmarking and paper.** Freeze a reporting protocol before any new confirmatory analysis,
   and report the population-shift result prominently.
