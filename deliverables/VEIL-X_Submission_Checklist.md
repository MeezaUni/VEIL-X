# VEIL-X University Project: Submission Checklist

**Project:** VEIL-X: An AI-Based Face Verification System with Quality Assessment and Intelligent
Abstention.

## 1. Deliverables

Files are in `deliverables/`, built by `scripts/build_university_deliverables.py`. Checksums are
in `MANIFEST.md`.

| File | Status |
|---|---|
| `VEIL-X_University_Project_Report.docx` | Built: editable, 16 pages with title page, table of contents, 12 figures and 19 sections |
| `VEIL-X_University_Project_Report.pdf` | Built from the DOCX by Microsoft Word |
| `VEIL-X_University_Presentation.pptx` | Built: 15 slides, speaker notes on every slide |
| `VEIL-X_University_Presentation.pdf` | Built from the PPTX by Microsoft PowerPoint |
| `VEIL-X_Demo_Guide.pdf` | Built: 3–5 minute demo, script, backup plan, troubleshooting |
| `VEIL-X_Viva_Questions.pdf` | Built: one-page revision sheet and 35 questions with answers |
| `VEIL-X_Project_Summary.pdf` | Built: one page |

**Sources.** The Markdown sources are in `docs/university/`. To rebuild:

```
.venv-docs\Scripts\python.exe scripts\build_university_deliverables.py "403 CPU + 417 GPU tests passing"
```

## 2. You must fill in before submitting

| Placeholder | Where |
|---|---|
| `[Student Name]`, `[Register Number]` | Report title page, slide 1, project summary, DOCX author property |
| `[Course Code and Name]`, `[Department]`, `[College / University Name]` | Report title page, slide 1, project summary |
| `[Guide / Supervisor Name]`, `[Academic Year]` | Report title page, slide 1, project summary |

**How to fill them in:**
- Edit the DOCX and PPTX directly in Word or PowerPoint, then re-export the PDFs (*File → Save
  as PDF*).
- Alternatively, replace the placeholders in `docs/university/*.md` and in the slide data of
  `scripts/build_university_deliverables.py`, then rebuild.
- In the report, right-click the table of contents and choose *Update Field* if the page numbers
  change.

## 3. Check with your guide

- **Synthetic face images.** The screenshots show synthetic DigiFace-1M faces (computer-generated,
  no real people), used under the dataset's research licence. Confirm that showing them in a
  university report is acceptable. If not, blur the face regions before submitting.
- **Report format.** Confirm the required report format: font, margins, certificate page,
  declaration, acknowledgement. Add any pages your university requires; the report has none of
  these institution-specific pages.

## 4. Before the demo

- [ ] `VEILX_DATA_ROOT` points to the private data folder (models + frozen configurations).
- [ ] `.venv-gpu\Scripts\python.exe scripts\run_app.py` starts, and the page opens at
      `http://127.0.0.1:7860`.
- [ ] Warm-up comparison done (example 2).
- [ ] Noise images `noise_a.png` / `noise_b.png` prepared (command in the demo guide).
- [ ] Backup: the slides and report screenshots open offline.

## 5. What was verified (2026-11-02)

- **Application.** It launched and served on 127.0.0.1 only, and its Content-Security-Policy is
  active. MATCH (example 2), NON-MATCH (example 1) and ABSTAIN (example 4) were all reproduced
  with the frozen thresholds unchanged. Diagnostics and visibility maps work, and the support
  guard abstains on noise images. Screenshots were taken from this run.
- **Tests.**
  - Quality gate PASS: lint, format, strict typing, config, privacy scan, doc links, smoke test.
  - CPU suite: 403 passed, 4 skipped.
  - GPU suite: 417 passed.
  - Smoke test: 8/8.
- **Numbers.** Every figure in the report and slides comes from committed result files:
  - parity: `reports/experiments/phase5_parity.json`;
  - speed: `reports/experiments/phase5_benchmark.json`;
  - accuracy and population shift: `phase2_final.json`, `phase3_final.json`,
    `phase4_final.json`, `phase6_final.json`;
  - low-quality tail: `phase6_lowq_audit.json`, `phase6_final.json`.
- **Privacy.** No real-person images, datasets, embeddings, credentials or model weights are in
  the deliverables or in git. The binary deliverables stay local: the public repository refuses
  documents and images by design.

## 6. Not included, by design

- Raw datasets, private manifests, model weights, embeddings, LFW images.
- A ZIP of the private data folder.
- Any public release. The GitHub repository is already public; only source files are pushed.
