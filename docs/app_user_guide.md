# VEIL-X Local Research Application: User Guide (Phases 5–6)

> **Research demonstration only — not an identity-verification product.** The application runs the
> frozen VEIL-X research pipeline on two images on *your* computer. Its output is the response of a
> research decision rule, not proof that two images show the same person. **Do not use it for
> identification, surveillance, access control or any decision affecting people.** Use only images you
> are authorised to process.

## 1. Requirements and launch

- **Hardware/software:**
  - Windows or Linux;
  - Python 3.12 with the GPU environment (`torch` 2.x CUDA build, `safetensors`, `onnxruntime`,
    `opencv-python-headless`, `gradio` 5.x);
  - an NVIDIA GPU is recommended (tested: RTX 4060 Laptop, 8 GB); CPU works but is slower (§5).
- **Private data root** (never in the repository) containing:
  - `models/` (CVLface AdaFace / ArcFace IR-101 safetensors, SCRFD `det_10g.onnx`, VE-C fold
    weights);
  - `reports/phase3_frozen.json` and `reports/phase4_frozen.json`.

  The app verifies each file's SHA-256 against `configs/app/veilx_app.yaml` and refuses to start
  if any differs.
- **Install** (once, into the GPU environment):
  `uv pip install --python .venv-gpu/Scripts/python.exe -e ".[gpu,app]"` (install torch from the CUDA wheel index first, e.g. `--index-url https://download.pytorch.org/whl/cu128`).
- **Launch:**

  ```powershell
  $env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"
  .venv-gpu\Scripts\python.exe scripts\run_app.py            # http://127.0.0.1:7860
  .venv-gpu\Scripts\python.exe scripts\run_app.py --device cpu --port 7861
  ```

  Open the printed address in a browser on the same computer, and stop the app with Ctrl+C.

## 2. What happens to your images (privacy)

| Aspect | Behaviour |
|---|---|
| Network | Binds to `127.0.0.1` only. No Gradio share link, no analytics, Hugging Face hub offline. A Content-Security-Policy confines the page to the local origin: the browser makes no third-party request (Gradio's default CDN script is blocked) |
| Uploads | Gradio stores each upload in a **private per-session temp directory** (`%TEMP%\veilx_app_*`). The app reads the bytes, validates and decodes them in memory, and **deletes the upload file immediately** |
| Displayed images | To show detections, crops and visibility overlays, Gradio writes these images into the **same private session directory**. Files older than 2 minutes are removed every minute (Phase 6, D-056; previously 15 minutes), and the whole directory is removed at exit. This also covers an upload abandoned because Gradio itself could not decode it |
| Examples | Synthetic examples are generated into the session directory at launch. Each load hands the comparison a disposable copy |
| Session end | The session directory is removed on normal exit. A directory left by a killed process is removed at the next start (owner-PID check) |
| Embeddings | Computed in memory for the current comparison only; **never written to disk**, never compared against any stored gallery. There is no identity lookup or search function |
| Logging | No image, embedding or result logging (server access log disabled) |
| Browser | `Cache-Control: no-store`. Images shown in the page are held by your browser tab while it is open |
| Limitations | Gradio keeps the uploaded preview in the browser tab until you remove it or close the page. If the machine crashes, temp files may remain until the next start |

## 3. Using the comparison tab

1. **Upload Image A and Image B** (JPEG, PNG, BMP or WEBP; ≥ 64 px; ≤ 4096 px; ≤ 20 MB). Invalid,
   corrupt, blank or unsupported files are **rejected** with a reason code; no decision is made.
2. **Read the active configuration card** above the options. It names the frozen A3 system in
   use, what it was calibrated on (synthetic DigiFace faces, mostly 5-image identities, four seen
   synthetic occluder families), which occluder family it never saw, and its measured results,
   including the Phase 4 failure of the 5 % target (`docs/calibration_systems.md`).
3. **Options** (optional):
   - **Frozen A3 system.** VEIL-X has five frozen A3 rules, one per Phase 3 cross-validation fold.
     The default is `F1_mask`, the first registered fold, chosen independently of any result
     (D-051). None of them is a deployment calibration.
   - **Pre-aligned.** Tick this only for 112×112 crops already aligned to the ArcFace template
     (e.g. the synthetic DigiFace examples generated locally from your data root). Detection is then skipped.
   - **Face index.** When an image contains **several faces, the app refuses to choose one**.
     Read the indices on the detector panel and enter the face to use.
   - **LFW-style framing (fallback).** Tick this **only** if both images are loosely cropped,
     centred, single-face portraits (like LFW). Only then, and only for near-square images, does
     the app apply the registered fixed-crop fallback when no reliable face is detected.
     Otherwise such an image is **refused** ("no reliable face"). A fixed crop of an arbitrary
     photograph is not a face (D-054).
   - **Diagnostics.** Visibility maps and novelty. They are explanations only and never change the
     decision.
4. Tick **both confirmations**, which are required for every comparison:
   - that you are authorised to process the images;
   - that you understand the calibration limitations.

   Without them the uploads are discarded and nothing is computed. Then press **Compare locally
   with the frozen rule**.

## 4. Reading the result

- **MATCH / NON-MATCH / ABSTAIN** is the output of the frozen *A3 quality-stratified* rule.
  - The colours are deliberately neutral; there is no "safe" or "certified" state.
  - **ABSTAIN** means the rule declines to decide. The recorded reason says which condition
    applied: the score fell in a stratum's abstention band, or a face could not be prepared.
    Reasons describe the rule; they are not causal explanations of the images.
- **Support guard (Phase 6, D-055).** If the pair's quality is below the 1st percentile of the
  active system's *calibration* quality, the app **abstains**. The status note then says what
  the frozen rule alone would have output. Below that support the frozen fused score
  extrapolates: two noise images reach "MATCH, p = 1.000" under the frozen rule. Non-face and
  severely degraded images therefore abstain.
- **Fused score.** A logistic function of the AdaFace cosine similarity and the pair quality, fitted
  on synthetic DigiFace pairs. **It is not a probability that the images show the same person and
  not a confidence guarantee.**
- **Quality.** The smaller of the two AdaFace pre-normalisation feature norms. It selects the
  stratum:
  - low quality: wide abstention band;
  - medium / high quality: single threshold, so the rule never abstains there.
- **Decision-rule region.** Shows where the score falls relative to the stratum's frozen thresholds.
- **Detector status**, per image:
  - aligned;
  - selected face;
  - **multiple faces: refused**;
  - **FALLBACK fixed crop**: no reliable detection, and you declared LFW-style framing. A
    detector-free crop that assumes LFW-like centred framing. It is *not* a detected face and is
    always labelled;
  - **no reliable face: refused**: no detection, or a low-quality one, without an applicable
    fallback;
  - pre-aligned.
- **ArcFace similarity and the AdaFace–ArcFace disagreement** are diagnostics only.
- **Predicted visibility and novelty** (Phase 4 conservative estimator VE-C) are **explanatory
  only**:
  - blue = predicted visible, orange = predicted not visible;
  - "UNUSUAL" = the image lies outside the estimator's training distribution;
  - the estimator cannot see occluders that look like faces, and it was trained on synthetic
    occluders only.

## 5. Speed (measured; `reports/experiments/phase5_benchmark.json`)

See `docs/phase5_results.md` §5. One comparison at a time:
- **GPU:** most of the latency is CPU face detection (SCRFD on onnxruntime CPU).
- **CPU-only:** the two backbones dominate (about 0.15 s each), so a comparison takes about 0.3–0.5 s.
- **Cold start:** the first comparison after launch also pays the model-loading cost.

## 6. Limitations you must read

The **Calibration & limitations** tab is part of the result. In short:
- The 5 % selective-error target was **not** maintained on the new Phase 4 population. A3 was
  chosen because it did best among the alternatives, not because it is universally safe.
- Calibration is population- and preprocessing-specific.
- There is no formal out-of-distribution guarantee.
- Physical occlusions are unvalidated.
- LFW and DigiFace results do not establish deployment readiness.
