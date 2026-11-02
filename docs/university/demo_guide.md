---
title: "VEIL-X Demonstration Guide"
subtitle: 3–5 minute live demo for the university evaluation
---

# 1. What you will show

| # | Step | Expected result (verified 2026-11-02 with the frozen system) |
|---|---|---|
| 1 | Launch VEIL-X | Browser page at `http://127.0.0.1:7860` |
| 2 | Example button **Same person** | **MATCH**, score 1.000, high-quality group |
| 3 | Example button **Different people** | **NON-MATCH**, score 0.001 |
| 4 | Example button **Same person, masked** | **ABSTAIN**, score 0.542 inside the low-quality uncertain zone (0.154–0.833) |
| 5 | *Why this result?* steps and the visibility heat maps of step 4 | Quality q = 13.00, set by the dark image A (low group); the mask area is shown in orange on image B |
| 6 | Example button **Not a face (noise)** | **ABSTAIN** from the support guard; the explanation says the unguarded rule would have answered MATCH |
| 7 | *Limitations* tab | Read the first two bullets |

**About the images:**
- All example faces are **synthetic** DigiFace faces, generated locally at launch from the data
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
3. Do one warm-up comparison (**Same person**). The first comparison is slower, about 0.7 s,
   because the GPU kernels are loading.

# 3. Script (what to do and say)

**Opening (30 s).** "VEIL-X compares two face images on this laptop, with no internet. Instead of
always saying yes or no, it can also say ABSTAIN when the evidence is too weak, and it explains
exactly why."

**Step 2: MATCH (45 s).**
- Actions:
  - Click the **Same person** example button. The *already aligned* option switches on by
    itself, because examples are aligned 112×112 crops.
  - Tick both confirmations. **Compare faces** becomes active.
  - Click **Compare faces**.
- Say: "MATCH. The four cards give the similarity, the image quality, the score and the rule
  that applied. The score of 1.000 is a calibrated model output, not a 100 % probability of
  identity. The bar shows where the score falls against this quality group's cut-off, and
  *Why this result?* lists every check with its exact numbers."

**Step 3: NON-MATCH (30 s).** Click **Different people**, then **Compare faces**. "Different
synthetic people: similarity 0.11, typical of different people, so NON-MATCH with a score of
0.001."

**Step 4: ABSTAIN (60 s).** Click **Same person, masked**, then **Compare faces**.
- "ABSTAIN. Image A is a dark, rotated synthetic photo and image B is masked. The weaker image
  sets the pair quality, so this pair falls in the low-quality group. Its score of 0.542 lies in
  that group's uncertain zone, so the system declines to decide."
- Point at step 4 of *Why this result?*: "Quality is the smaller feature norm: image A has 13.00
  and B has 18.48, so q = 13.00. That is low."
- Point at *What would change the result?*: "It tells us exactly what a decision would need: a
  score of at least 0.833 for MATCH, or at most 0.154 for NON-MATCH."
- Open *Where might the faces be covered?*: "Orange marks the covered area. This visibility map
  is only an explanation. Our tests showed that learned visibility fails on unseen occluders, so
  the decision uses quality, not visibility."

**Step 6: support guard (45 s).** Click **Not a face (noise)**, then **Compare faces**. "These
are not faces. The research rule alone would have answered MATCH with a score of 1.0, because the
formula extrapolates when quality is far below anything it was calibrated on. The application
therefore abstains below the calibrated quality range, and says so."

**Close (30 s).** Open the *Limitations* tab. "The thresholds were calibrated on synthetic data,
and the 5 % error target did not hold on a harder population. This is a demonstration, not an
identity-verification product."

# 4. Precomputed versus live

- **Live.** Every comparison runs the full pipeline: detection or pre-aligned input, AdaFace,
  ArcFace, the decision, the explanation and diagnostics.
- **Fixed in advance.** The model weights, the calibrated thresholds and the quality-group edges
  were computed earlier and are fixed. They are loaded and hash-checked at start. The typical
  same-person and different-person ranges quoted in the explanations come from development data
  (`configs/app/explain_reference.json`) and never affect a decision.
- **Synthetic examples** are generated at launch from the local data folder and held in memory.

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
| Only the **Not a face (noise)** example button is shown | The data folder was not found: stop, set `VEILX_DATA_ROOT`, relaunch |
| App will not start | Show the screenshots in the report (Section 13) and the presentation slides, and explain them |
| A result differs from this guide | Explain the actual result; never change thresholds |

# 7. Troubleshooting

| Message | Meaning |
|---|---|
| **Compare faces** is greyed out | Tick both confirmation boxes |
| "Please add image A" | Upload a photo or click an example button |
| "2 faces found. VEIL-X never chooses a person for you" | Advanced options opens by itself: enter the face number shown on the detection image |
| "no usable face" | No reliable face. For a loosely cropped single-face portrait, tick the fixed-crop fallback (LFW-style framing) in Advanced options |
| "Image A not accepted: …" | The file is corrupt, too large, blank or of an unsupported type. Use a JPEG, PNG, BMP or WebP photo |
| "image quality too low" | The support guard: quality is below the calibrated range |

# 8. Longer technical walkthrough (optional, 10 minutes)

1. Open the *How it works* tab: the six steps, then *Exact thresholds of every calibration
   system* and *Models and experiment details* (file fingerprints).
2. In Advanced options, change the calibration system (e.g. `F4_random_patch`). *About this
   calibration system* shows that system's calibration and measured results, and the result
   card names the active system.
3. Show the *Limitations* tab, including the measured findings on why visibility is not
   decisive.
4. Show the detector path with a consenting volunteer's photograph:
   - the numbered detection boxes under *Faces used*;
   - the aligned crop;
   - the timings under *Technical details* (detection ≈ 0.17 s on CPU).
5. In a terminal, run the data-free smoke test, `.venv\Scripts\python.exe scripts\smoke_test.py`,
   and the quality gate, `.venv\Scripts\python.exe scripts\quality_gate.py`.
