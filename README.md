# VEIL-X

Research scaffold for **risk-controlled two-image (1:1) face verification under occlusion**.
Non-commercial academic research only.

> **Status: Phase 0 (scaffolding).** No face-recognition model, no dataset and no model
> weights are in this repository. **No empirical results exist yet.** Every number in
> `docs/` is either cited prior work, a registered *provisional* constant, or an exact
> statistical calculation from `scripts/feasibility_report.py`.

## Research questions (approved)

- **Primary (C3):** Do risk-controlled decisions (Learn-then-Test on calibration data) keep
  their target error rate on **held-out challenge families**? Does **visibility-stratified**
  calibration reduce violations? LTT guarantees assume exchangeable calibration/test data,
  so on unseen families they are **not** expected to hold. Measuring the gap is the point.
- **Secondary (C1):** Does pair-level **co-visibility** predict errors beyond standard
  image-quality measures? It is tested against quality-only, co-visibility-only and
  quality + co-visibility alternatives.

Frozen pretrained backbones: AdaFace IR101 (primary) and ArcFace (secondary). No backbone
fine-tuning or mixture of experts. Challenge families: masks, sunglasses, hands/objects,
random patches, and extreme lighting. Lighting is interpreted separately because it is not a
physical occlusion.

## Repository layout

```
configs/   research_protocol.yaml (versioned protocol + registered constants)
           datasets.yaml          (dataset registry: licence, access, roles)
docs/      research & design documents, decision log, development log
scripts/   validate_config.py, feasibility_report.py, scan_sensitive.py
src/veilx/ config schema, manifests & splits, interfaces, provenance, logging, seeding,
           guards, feasibility statistics
tests/     structural tests (no data, no models)
```

## Setup (Windows, PowerShell; uv)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy
.venv\Scripts\python.exe scripts\validate_config.py
.venv\Scripts\python.exe scripts\feasibility_report.py
```

Before any commit, run `scripts\scan_sensitive.py`. It refuses staged images, archives,
weights, documents, secrets, large files and absolute user paths.

## Privacy and licensing

- **No biometric data in git:** no images, embeddings, manifests of real data, weights or
  credentials. `.gitignore` and `scripts/scan_sensitive.py` enforce this.
- Scope is **1:1 verification of two images only**. There is no 1:N search, gallery,
  scraping or surveillance functionality, and none will be added.
- Pretrained weights (InsightFace, AdaFace/WebFace) and datasets (e.g. DigiFace-1M under
  R-UDA) are **non-commercial research only**. See `docs/dataset_feasibility.md`.

## Key documents

`docs/research_landscape.md`, `docs/novelty_audit.md`, `docs/dataset_strategy.md`,
`docs/dataset_feasibility.md`, `docs/protocol_design.md`, `docs/interfaces.md`,
`docs/proposed_architecture.md`, `docs/evaluation_protocol.md`,
`docs/implementation_roadmap.md`, `docs/decision_log.md`, `docs/DEVELOPMENT_LOG.md`.
