---
title: "VEIL-X Demonstration Guide"
subtitle: 3–5 minute live demo for the university evaluation
---

# 1. What you will show

| # | Step | Expected result (verified 2026-11-02 with the frozen system) |
|---|---|---|
| 1 | Launch VEIL-X | Browser page at `http://127.0.0.1:7860` |
| 2 | Synthetic example 2 (same synthetic person) | **MATCH**, fused score 1.000, high-quality group |
| 3 | Synthetic example 1 (two different synthetic people) | **NON-MATCH**, fused score 0.001 |
| 4 | Synthetic example 4 (same person, mask on image B) | **ABSTAIN**, score 0.542 inside the low-quality band [0.154, 0.833) |
| 5 | Diagnostics table and visibility heat maps of step 4 | Quality q = 13.00, set by the dark image A (low group); the mask area is shown in orange on image B |
| 6 | Two random-noise images (prepared beforehand) | **ABSTAIN**: "below the calibration support"; the note says the unguarded rule would have said MATCH |
| 7 | "Calibration & limitations" tab | Read the first two bullets |

**About the images:**
- All face images are **synthetic** DigiFace faces, generated locally at launch from the data
  folder. They are not real people.
- If you add a photograph of a real person, it must be one you are authorised to use, for
  example a consenting classmate.
- Thresholds are fixed. Never change them to force a result. If a case behaves differently,
  explain the actual result shown on screen.

# 2. Before the demo (10 minutes earlier)

1. Plug the laptop into power and close heavy GPU programs.
2. **Easiest:** double-click `run_veilx.bat` in the project folder. It sets everything up on
   the first run and opens the browser; see the README *Quick start*. **Or** open PowerShell in
   the project folder and launch:

```
$env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"
.venv-gpu\Scripts\python.exe scripts\run_app.py
```

   - Wait for the line `VEIL-X research demo: http://127.0.0.1:7860  (local only; Ctrl+C to stop)`. The first start loads
     the models in about 3–10 s.
   - Open that address in the browser.
3. Prepare the noise images, if they are missing:

```
.venv\Scripts\python.exe -c "import numpy as np; from PIL import Image; r=np.random.default_rng(1); [Image.fromarray(r.integers(0,255,(112,112,3),np.uint8)).save(f'noise_{k}.png') for k in 'ab']"
```

4. Do one warm-up comparison (example 2). The first comparison is slower, about 0.7 s, because
   the GPU kernels are loading.

# 3. Script (what to do and say)

**Opening (30 s).** "VEIL-X compares two face images on this laptop, with no internet. Instead of
always saying yes or no, it can also say ABSTAIN when the evidence is too weak, and it explains
why."

**Step 2: MATCH (45 s).**
- Actions:
  - Synthetic examples tab → *example 2* → *Load into the comparison tab*.
  - In Options, tick *pre-aligned*.
  - Tick both confirmations.
  - Click **Compare**.
- Say: "MATCH. The fused score of 1.000 is a calibrated model output, not a 100 % probability of
  identity. The bar shows where the score falls against this quality group's threshold."

**Step 3: NON-MATCH (30 s).** Load *example 1* and compare. "Different synthetic people:
NON-MATCH, with a score of 0.001."

**Step 4: ABSTAIN (60 s).** Load *example 4*, a mask on image B, and compare.
- "ABSTAIN. Image A is a dark, rotated synthetic photo and image B is masked. The weaker image
  sets the pair quality, so this pair falls in the low-quality group. Its score of 0.542 lies in
  the abstention band, so the system declines to decide."
- Scroll to the table: "Quality is the smaller feature norm: image A has 13.00 and B has 18.48,
  so q = 13.00. That is low."
- Scroll to the heat maps: "Orange marks the covered area. This visibility map is only an
  explanation. Our tests showed that learned visibility fails on unseen occluders, so the
  decision uses quality, not visibility."

**Step 6: support guard (45 s).** Upload the two noise images (*pre-aligned*, confirmations) and
compare. "These are not faces. The research rule alone would have answered MATCH with a score of
1.0, because the formula extrapolates when quality is far below anything it was calibrated on.
The application therefore abstains below the calibrated quality range."

**Close (30 s).** Open the *Calibration & limitations* tab. "The thresholds were calibrated on
synthetic data, and the 5 % error target did not hold on a harder population. This is a
demonstration, not an identity-verification product."

# 4. Precomputed versus live

- **Live.** Every comparison runs the full pipeline: detection or pre-aligned input, AdaFace,
  ArcFace, the decision, and diagnostics.
- **Fixed in advance.** The model weights, the calibrated thresholds and the quality-group edges
  were computed earlier and are fixed. They are loaded and hash-checked at start.
- **Synthetic examples** are generated at launch from the local data folder.

# 5. Expected times (RTX 4060 Laptop GPU)

| What | Time |
|---|---|
| Start (model loading) | about 3 s of loading; about 10–20 s in total with Python start-up |
| First comparison | about 0.7 s |
| Warm comparison, pre-aligned examples | about 0.05 s |
| Warm comparison, two photographs | about 0.23 s |
| CPU-only (`--device cpu`) | 0.3–0.5 s per comparison |

# 6. Backup plan

| Problem | Action |
|---|---|
| GPU unavailable or driver error | Restart with `--device cpu`. Everything still works, slightly slower |
| No internet | Not needed; VEIL-X is fully offline |
| Port 7860 busy | `--port 7861`, then open `http://127.0.0.1:7861` |
| Examples tab says "No local synthetic examples" | `VEILX_DATA_ROOT` is not set: stop, set it, relaunch |
| App will not start | Show the screenshots in the report (Section 13) and the presentation slides, and explain them |
| A result differs from this guide | Explain the actual result; never change thresholds |

# 7. Troubleshooting

| Message | Meaning |
|---|---|
| "Please confirm both statements" | Tick both confirmation boxes; the uploads were discarded, so upload again |
| "Several faces detected: REFUSED" | Enter the face index shown on the detection image |
| "No reliable face … REFUSED" | No usable face. The fallback is allowed only for LFW-style portraits (Options) |
| "Input rejected (…)" | The file is corrupt, too small, blank or of an unsupported type |
| "Error" (generic) | The interface (Gradio) could not decode the uploaded file. Try another file; the upload is cleared within about 3 minutes |
| "below the calibration support" | Quality is too low for a reliable decision (support guard) |

# 8. Longer technical walkthrough (optional, 10 minutes)

1. Show the *Model & experiment information* tab: file fingerprints and frozen thresholds of all
   five calibration configurations.
2. Change the frozen system in Options (e.g. `F4_random_patch`). The configuration card updates
   with that system's calibration and measured results.
3. Show the *Why visibility is not decisive* tab, with the measured findings.
4. Show the detector path with a consenting volunteer's photograph:
   - the detection box and face index;
   - the aligned crop;
   - the latency line (detection ≈ 0.17 s on CPU).
5. In a terminal, run the data-free smoke test, `.venv\Scripts\python.exe scripts\smoke_test.py`,
   and the quality gate, `.venv\Scripts\python.exe scripts\quality_gate.py`.
