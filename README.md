# VEIL-X

Research code for **risk-controlled two-image (1:1) face verification under occlusion**.
Non-commercial academic research only.

> **Status: Phase 2 complete, including the real-photo evaluation (LFW).** AdaFace IR-101 and
> ArcFace IR-101 (WebFace4M) were evaluated frozen, with no training.
> - **DigiFace-1M** (synthetic faces) with synthetic occlusions: `docs/baseline_evaluation.md`.
> - **LFW** (Kaggle v4, used under user-reported academic authorisation): official View-2 protocol
>   plus derived synthetic occlusions on real photographs: `docs/real_world_benchmark.md` §7.
> - Audit: `docs/statistical_audit.md`.
>
> Derived occlusions are synthetic overlays, **not physical occlusions**. Clean LFW accuracy is
> likely optimistic because of probable identity overlap with the backbones' training data.
>
> Dataset images, manifests, embeddings and weights are never in this repository: they live in
> a private data root (`VEILX_DATA_ROOT`). Every number in `docs/` and `reports/` is typed in
> `reports/evidence_registry.yaml`:
> - cited work;
> - exact calculations;
> - **SIMULATED** results;
> - real-data statistics;
> - **MEASURED** model results, each with run provenance.

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

## What exists (Phases 1a + 1b)

| Area | Module | Notes |
|---|---|---|
| Dataset adapter | `veilx.data.digiface` | DigiFace-1M layout, integrity and anomaly report. Run on the approved real subset: 310,665 images, 0 anomalies |
| Real-data pipeline | `scripts/prepare_digiface.py`, `veilx.data.{download,archive,images,dedup,integrity}` | Verified download, safe extraction, fingerprints, validated dedup, pair sets, occlusion plans (`docs/data_integrity_report.md`) |
| Occlusion engine | `veilx.occlusion` | Masks, sunglasses, hands/objects, random patches, extreme lighting (non-physical); stress combinations; leave-one-family-out plans with isolation checks |
| Pairs | `veilx.pairs` | Identity-disjoint (independent) or clustered pairs. Explicit infeasibility; independence accounting |
| Metrics | `veilx.metrics` | ROC-AUC, EER, TAR@FAR (benchmark and deployment), selective metrics, AUGRC/AURC, ECE/Brier/reliability. See `docs/metrics.md` |
| Uncertainty | `veilx.stats.jackknife`, `veilx.stats.bootstrap` | Delete-one-identity jackknife (selected by SIM-010); identity bootstrap kept for comparison |
| Risk control | `veilx.selective` | Audited Learn-then-Test (`docs/ltt_audit.md`), empirical-only calibrator, stratified LTT |
| Simulations | `veilx.sim`, `reports/simulations/` | 10 deterministic synthetic experiments (`docs/simulation_findings.md`) |
| Training infrastructure | `veilx.training`, `configs/experiments/` | Run configs, checkpoints, mixed precision; **training not authorised** (double guard) |
| Integrity | `veilx.evidence`, `reports/evidence_registry.yaml` | Keeps published, calculated, simulated, measured and hypothesised results separate |

## Repository layout

```
configs/   research_protocol.yaml (protocol v0.3.0), datasets.yaml, experiments/ (templates)
docs/      research, protocol (+ changelog), audit, metrics, data-integrity and decision docs
reports/   simulations/*.json (SIMULATED), data/*.json (aggregated real-data statistics),
           evidence_registry.yaml
scripts/   quality_gate, validate_config, feasibility_report, run_simulations, scan_sensitive,
           fetch_digiface, prepare_digiface, parts_benefit
src/veilx/ config, data (manifest, splits, png, images, archive, download, paths, digiface,
           dedup, integrity), pairs, occlusion, metrics, stats, selective, sim, training,
           evidence, interfaces, provenance, logging, seeding, guards, errors
tests/     unit, property, integrity, simulation and pipeline-integration tests (synthetic
           fixtures only; no real images, no models)
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

### Real-data preparation (approved DigiFace subset; private data root)

```powershell
$env:VEILX_DATA_ROOT = "<folder outside any git repository>"
.venv\Scripts\python.exe scripts\fetch_digiface.py         # approved parts only, ~5.9 GiB
.venv\Scripts\python.exe scripts\prepare_digiface.py extract scan fingerprint dedup pairs occlusion summary
```

Only aggregated, non-identifying statistics are written inside the repository
(`reports/data/`). Everything listing images, identities or pairs stays in the data root.

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
