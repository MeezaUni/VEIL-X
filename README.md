# VEIL-X

Research code for **risk-controlled two-image (1:1) face verification under occlusion**.
Non-commercial academic research only.

> **Status: Phase 1a complete (evaluation and statistical foundation).** No face-recognition
> model, no dataset and no model weights are in this repository. **No real measured results
> exist.** Numbers in `docs/` and `reports/` fall into four categories, each tracked in
> `reports/evidence_registry.yaml`:
> - cited prior work;
> - registered *provisional* constants;
> - exact calculations;
> - **SIMULATED** results on synthetic scores, which are never evidence of face-recognition
>   performance.

## Research questions (approved)

- **Primary (C3):** Do risk-controlled decisions (Learn-then-Test on calibration data) keep
  their target error rate on **held-out challenge families**? Does **visibility-stratified**
  calibration reduce violations? The guarantees assume exchangeable calibration/test data and
  are **not** claimed under held-out shifts. Measuring the gap is the point.
- **Secondary (C1):** Does pair-level **co-visibility** predict errors beyond image-quality
  measures? It is compared with quality-only, co-visibility-only and quality + co-visibility.

Frozen pretrained backbones: AdaFace IR101 (primary) and ArcFace (secondary). Challenge
families: masks, sunglasses, hands/objects, random patches, extreme lighting. Lighting is
interpreted separately.

## What exists (Phase 1a)

| Area | Module | Notes |
|---|---|---|
| Dataset adapter | `veilx.data.digiface` | DigiFace-1M layout, integrity and anomaly report. Tested on synthetic noise fixtures only |
| Pairs | `veilx.pairs` | Identity-disjoint (independent) or clustered pairs. Explicit infeasibility; independence accounting |
| Metrics | `veilx.metrics` | ROC-AUC, EER, TAR@FAR (benchmark and deployment), selective metrics, AUGRC/AURC, ECE/Brier/reliability. See `docs/metrics.md` |
| Uncertainty | `veilx.stats.bootstrap` | Identity-cluster bootstrap |
| Risk control | `veilx.selective` | Audited Learn-then-Test (`docs/ltt_audit.md`), empirical-only calibrator, stratified LTT |
| Simulations | `veilx.sim`, `reports/simulations/` | 9 deterministic synthetic experiments (`docs/simulation_findings.md`) |
| Integrity | `veilx.evidence`, `reports/evidence_registry.yaml` | Keeps published, calculated, simulated, measured and hypothesised results separate |

## Repository layout

```
configs/   research_protocol.yaml (versioned protocol v0.2.0, provisional), datasets.yaml
docs/      research, protocol, audit, metric and decision documents; development log
reports/   simulations/*.json (SIMULATED), evidence_registry.yaml
scripts/   quality_gate, validate_config, feasibility_report, run_simulations, scan_sensitive
src/veilx/ config, data (manifest, splits, png, digiface), pairs, metrics, stats, selective,
           sim, evidence, interfaces, provenance, logging, seeding, guards, errors
tests/     unit, property, integrity and simulation tests (no real data, no models)
```

## Setup and checks (Windows PowerShell, uv)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML
.venv\Scripts\python.exe scripts\quality_gate.py          # ruff, format, mypy, pytest, config, privacy scan
.venv\Scripts\python.exe scripts\run_simulations.py       # regenerate reports/simulations (about 2 min)
.venv\Scripts\python.exe scripts\feasibility_report.py    # exact sample-size calculations
```

Run `scripts\scan_sensitive.py` before every commit. It refuses images, arrays, manifests
(`.jsonl`), archives, weights, documents, secrets, large files and absolute user paths.

## Privacy and licensing

- **No biometric data in git:** no images, embeddings, real-data manifests, weights or
  credentials.
- Scope is **1:1 verification of two images only**. There is no 1:N search, gallery,
  scraping or surveillance functionality.
- Pretrained weights (InsightFace, AdaFace/WebFace) and datasets (e.g. DigiFace-1M under
  R-UDA) are **non-commercial research only**. See `docs/dataset_feasibility.md`.

## Key documents

`docs/protocol_design.md`, `docs/ltt_audit.md`, `docs/metrics.md`,
`docs/simulation_findings.md`, `docs/evidence_registry.md`, `docs/dataset_feasibility.md`,
`docs/interfaces.md`, `docs/decision_log.md`, `docs/DEVELOPMENT_LOG.md`, plus the Phase 0
research documents (`research_landscape`, `novelty_audit`, `dataset_strategy`,
`proposed_architecture`, `evaluation_protocol`, `implementation_roadmap`).
