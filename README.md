# VEIL-X

Research code for **risk-controlled two-image (1:1) face verification under occlusion**.
Non-commercial academic research only. **Not an identity-verification product.**

> **Status: Phase 6 (final phase) complete. Waiting for owner review.**
> - Phase 6 (`docs/phase6_results.md`, audit `docs/phase6_audit.md`, paper
>   `docs/paper/veilx_paper.md`): a pre-registered replication on 4,000 untouched identities
>   **reproduced the population-shift failure** and again recommended A3. A new finding: the
>   frozen fused score is unreliable below the calibration support of quality (the app abstains
>   there; D-055). Reproduction guide: `docs/reproducibility.md`.
> - Frozen AdaFace IR-101 and ArcFace IR-101 (WebFace4M); no recognition training.
> - Phase 3 (`docs/phase3_results.md`): learned-visibility calibration did **not** beat
>   quality-stratified calibration on unseen occluders.
> - Phase 4 (`docs/phase4_results.md`): conservative visibility fixed the overestimation, but no
>   visibility-based rule beat **A3 quality-stratified calibration**, which is recommended (D-050).
> - On an untouched DigiFace population, **every rule exceeded the 5 % selective-error target**
>   (population shift). The guarantees are calibration-population-specific.
> - Phase 5 (`docs/phase5_results.md`): a local Gradio demonstration of the frozen pipeline with
>   explanatory diagnostics (`docs/app_user_guide.md`).
>
> All occluders are **synthetic**; physical occlusions are unvalidated. There is no formal
> out-of-distribution risk guarantee. LFW and DigiFace results do not establish deployment
> readiness.

## Research questions

- **C3 (primary):** do risk-controlled decisions (Learn-then-Test) keep their target error on
  **held-out occlusion families**, and does visibility-aware calibration help? **Answer so far:
  no.** Quality stratification is more reliable, and guarantees do not transfer under population
  shift.
- **C1 (secondary):** does pair-level co-visibility add information beyond quality? Only for some
  seen families; it is harmful on never-seen occluders, so it is excluded from the recommended
  system.

## The local research application (Phase 5)

```powershell
uv pip install --python .venv-gpu\Scripts\python.exe -e ".[gpu,app]"      # torch: CUDA wheel index, see pyproject
$env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"                              # private models + frozen configs
.venv-gpu\Scripts\python.exe scripts\run_app.py                             # http://127.0.0.1:7860 (local only)
```

**Pipeline:**
- SCRFD detection + 5-point alignment, with the registered fixed-crop **fallback**. It is always
  labelled, and multiple faces are refused unless the user selects one.
- Frozen AdaFace (decision) and ArcFace (diagnostic).
- Fused score M1 = logistic(cosine, min feature norm, product).
- **A3 quality-stratified MATCH / NON-MATCH / ABSTAIN** rule (frozen Phase 3 system; D-051).
  Five fold-specific systems exist; none is a deployment calibration (`docs/calibration_systems.md`).
- App-only safety (D-054, D-055): the fixed-crop fallback only for declared LFW-style portraits;
  abstention below the calibration support of quality; two required confirmations.

Visibility maps and novelty flags (Phase 4 VE-C) are shown as **explanations only** and never
enter the decision.

**Privacy:**
- loopback only; no share link, no analytics, no third-party requests (strict CSP);
- uploads deleted after reading; no embedding storage; no identity search.

The app verifies SHA-256 of every frozen file. Parity with the offline evaluation and measured
latency are in `docs/phase5_results.md`.

## Repository layout

```
configs/   research_protocol.yaml (v0.6.0), datasets.yaml, experiments/, app/veilx_app.yaml
docs/      protocols (phase3/phase4), results (phase3/phase4/phase5), audit, model cards,
           app_user_guide, decision log, development log
reports/   experiments/*.json (MEASURED, with provenance), simulations/ (SIMULATED),
           data/ (aggregated statistics), figures/, evidence_registry.yaml
scripts/   data preparation, extraction, training, evaluation (phase2-4), app (run_app,
           phase5_parity, phase5_benchmark), quality_gate, scan_sensitive
src/veilx/ data, occlusion, detect, models, embeddings, vis, fusion, selective, stats,
           metrics, sim, training, evidence, app (config, inputs, face, decision, pipeline,
           diagnostics, report, server, tempfiles, ui)
tests/     unit, property, integrity and pipeline tests (synthetic fixtures; model tests skip
           without the private data root)
```

## Setup and checks (Windows PowerShell, uv)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML
.venv\Scripts\python.exe scripts\quality_gate.py          # ruff, format, mypy, pytest, config, privacy scan
.venv-gpu\Scripts\python.exe -m pytest tests\test_app_models.py tests\test_models_torch.py   # GPU / model tests
```

Run `scripts\scan_sensitive.py` before every commit. It refuses images, arrays, manifests,
archives, weights, documents, secrets, large files and absolute user paths.

**Reproducibility.**
- Every measured number traces to a script, a frozen configuration (SHA-256), a seed and, for test
  data, a logged run in `results/test_runs.jsonl`. See `reports/experiments/registry.json` and the
  per-phase result documents.
- Data preparation: `scripts/fetch_digiface.py`, `prepare_digiface.py`, `prepare_phase3.py`,
  `prepare_phase4.py`, `prepare_lfw.py`.
- Evaluation: `scripts/evaluate_phase3.py`, `evaluate_phase4.py`.

## Privacy and licensing

- **No biometric data in git:** no images, embeddings, real-data manifests, weights or
  credentials.
- Scope is **1:1 verification of two images only**. There is no 1:N search, gallery, scraping
  or surveillance functionality.
- Pretrained weights (InsightFace, AdaFace/ArcFace on WebFace) and datasets (DigiFace-1M under
  R-UDA; LFW under user-reported academic authorisation) are **non-commercial research only**.

## Key documents

- **Results:** `docs/phase3_results.md`, `docs/phase4_results.md`, `docs/phase5_results.md`,
  `docs/phase6_results.md`.
- **Final audit, paper and presentation:** `docs/phase6_audit.md`, `docs/paper/veilx_paper.md`,
  `docs/presentation/slides.md`, `docs/presentation/demo_and_viva.md`.
- **Reproducing:** `docs/reproducibility.md` (`scripts/smoke_test.py` needs no data).
- **Protocols:** `docs/phase3_protocol.md`, `docs/phase4_protocol.md`, `docs/phase6_protocol.md`.
- **Using the app:** `docs/app_user_guide.md`.
- **Model cards:** `docs/model_card_backbones.md`, `docs/model_card_visibility.md`.
- **Audit and records:** `docs/statistical_audit.md`, `docs/decision_log.md`,
  `docs/DEVELOPMENT_LOG.md`, `docs/implementation_roadmap.md`.
