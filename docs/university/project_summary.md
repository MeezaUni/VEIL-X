---
title: "VEIL-X: Project Summary"
subtitle: An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention
---

**Student:** [Student Name] · **Register No.:** [Register Number] · **Course:** [Course Code / Name]
· **Department / College:** [Department], [College] · **Guide:** [Guide Name] · **Year:** [Academic Year]

# What it is

VEIL-X is a local AI application that compares two face images. It returns **MATCH**,
**NON-MATCH** or **ABSTAIN**, and explains the result step by step: similarity and quality,
the exact threshold comparison that decided, what would change it, and visibility heat maps. It runs entirely
on the user's computer.

# How it works

1. **Face preparation:** SCRFD detection and alignment; several faces are refused unless one is
   chosen, and unsuitable photos are refused.
2. **Scoring:** the pretrained AdaFace IR-101 embedding and its feature norm (quality), cosine
   similarity, and a logistic score combining similarity and quality.
3. **Decision:** calibrated, quality-dependent thresholds give MATCH / NON-MATCH / ABSTAIN; a
   support guard abstains below the calibrated quality range.
4. **Explanation:** a plain-language "Why this result?" panel; visibility heat maps and a novelty flag are explanatory only.

# Technologies

Python 3.12, PyTorch (CUDA), AdaFace / ArcFace IR-101, SCRFD (ONNX Runtime), OpenCV, Gradio +
FastAPI (localhost only), pytest / Ruff / mypy, GitHub Actions.

# Verified results

| Result | Value |
|---|---|
| Parity with the research pipeline | 1,000 / 1,000 identical decisions; CPU vs GPU 100 / 100 |
| Speed (RTX 4060 Laptop, warm, median) | 226 ms per photo pair; 43 ms per pre-aligned pair; CPU-only 0.3–0.5 s |
| Automated tests | 400+ tests passing (CPU and GPU), plus lint, strict typing and a privacy scan |
| Decision quality on data like the calibration data | Selective errors 1.1–2.6 % at 89–97 % coverage, for occluder types unseen in calibration |

Parity is not accuracy: these results show the app reproduces the research system exactly.

# Honest limitations

- Tested on synthetic faces and synthetic occlusions; real physical occlusions not evaluated.
- The 5 % error target failed on a harder population of identities (5.2–6.5 %), confirmed in
  two independent tests. Calibration is population-dependent.
- Very low-quality inputs produced false matches in the research rule (22.97 % below the
  calibrated quality range). The app now abstains there; this is a safeguard, not a proven fix.
- A score of 1.000 is not a 100 % probability of identity.
- Academic demonstration only: not certified, not for real identity decisions.


**Launch:** set `VEILX_DATA_ROOT`, run `.venv-gpu\Scripts\python.exe scripts\run_app.py`, and open `http://127.0.0.1:7860` (details in the demo guide).
