# Reproducing VEIL-X (Phase 6, 6G)

This guide reproduces the VEIL-X pipeline on an appropriately equipped Windows machine. Use it
with the per-phase protocols and result documents. **No biometric data, weights, embeddings or
credentials are in the repository.** You obtain the data and weights yourself, under their
licences, into a private data root.

## 0. Quick check without any data (5 minutes)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML
.venv\Scripts\python.exe scripts\smoke_test.py --level 1     # imports, configs, decision rule, validation, server policy, LTT determinism
.venv\Scripts\python.exe scripts\quality_gate.py              # ruff, format, mypy (strict), pytest, config validation, privacy scan
```

**Expected result:**
- the smoke test prints `[PASS]` for six checks;
- the quality gate ends `QUALITY GATE: PASS`;
- the CPU suite reports no failures. Model tests are skipped without torch and the data root.

## 1. Hardware used and minimum requirements

| | Used for all published runs | Practical minimum |
|---|---|---|
| OS | Windows 11 Pro (build 26300) | Windows 10/11 or Linux |
| CPU | AMD Ryzen 9 7940HS (8 cores) | 4+ cores |
| RAM | 31.2 GB | 16 GB. Evaluation peaks at 3–6 GB per process; extraction runs under the memory monitor |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB, driver 610.47 | Any CUDA GPU with ≥ 6 GB. CPU-only works for the app and the smoke test, but extraction is impractical |
| Disk | ~120 GB used under the data root | 150 GB free. DigiFace archives ≈ 23 GB + 11.7 GB (Phase 4/6 parts); extracted images, feature stores and renders take the rest |

## 2. Environments

| Environment | Purpose | Create |
|---|---|---|
| `.venv` (CPU) | Tests, evaluation of stored features, configs, docs | `uv venv --python 3.12 .venv` then `uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML` |
| `.venv-gpu` | Model weights, extraction, training, the app | `uv venv --python 3.12 .venv-gpu`, then `uv pip install --python .venv-gpu\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128`, then `uv pip install --python .venv-gpu\Scripts\python.exe -e ".[gpu,app]" pytest` |

**Versions used for every published result:**

| Component | Version |
|---|---|
| Python | 3.12.14 |
| torch | 2.11.0 + CUDA 12.8, cuDNN 9.19 |
| onnxruntime | 1.31 (SCRFD on the **CPU** provider; see §8) |
| opencv-python-headless | 5.0 |
| gradio | 5.50 |

Every result JSON records its own package versions in `provenance.packages`.

## 3. Private data root

Set `VEILX_DATA_ROOT`, for example to `%USERPROFILE%\veilx_data`. The layout is created by the
scripts: `raw/`, `extracted/`, `manifests/`, `features/`, `embeddings/`, `models/` and
`reports/` (private reports). None of it is ever committed.

### 3.1 Data acquisition (licences are your responsibility)

| Data | Source | Licence / authorisation | Command |
|---|---|---|---|
| DigiFace-1M (synthetic faces): 72-image parts 0–1999, 2000–5999 (Phase 4) and 6000–9999 (Phase 6); 5-image parts 100000–199998 | Official Microsoft host linked from github.com/microsoft/DigiFace1M | Research-only (R-UDA). Non-commercial | `python scripts/fetch_digiface.py` downloads only the parts listed in `APPROVED_PARTS`, checks official sizes and records SHA-256 fingerprints |
| LFW (real photographs) | Kaggle `jessicali9530/lfw-dataset` v4 (a mirror; see `docs/real_world_benchmark.md`) | No redistribution rights established. Used under user-reported academic authorisation (D-043). Obtain your own authorisation | Manual Kaggle download into `raw/lfw_kaggle_v4`; SHA-256 of the zip recorded in `configs/datasets.yaml` |
| Backbone weights | Hugging Face `minchul/cvlface_adaface_ir101_webface4m`, `minchul/cvlface_arcface_ir101_webface4m` (pinned revisions) | Research / non-commercial terms of the releases | `.venv-gpu\Scripts\python.exe scripts\fetch_models.py` |
| SCRFD detector | InsightFace `buffalo_l` v0.7, `det_10g.onnx` | Non-commercial research | See `docs/model_card_backbones.md`; SHA-256 pinned in `configs/app/veilx_app.yaml` |

### 3.2 Integrity checks

- **Archives:** official Content-Length, member-name pattern and CRC-32 of every member
  (`inspect_zip`), then safe extraction (no absolute or `..` paths). The checks run inside the
  `prepare_*` scripts.
- **Images:** decode and shape checks, then pixel-SHA-256 dedup within and across partitions
  (Phase 4 and 6: 0 failures, 0 duplicates).
- **Weights:**
  - SHA-256 pinned for both backbones (`veilx.models.spec`), checked at every load;
  - SCRFD and VE-C pinned in the app config;
  - VE / VE-C hashes recorded in `reports/experiments/phase*_ve*_training_<fold>.json`.
- **Frozen decision systems:** content SHA-256 of the Phase 3 (`34409779…`) and Phase 4
  (`bba12900…`) freezes, checked by every evaluation script and by the app.
- **One-shot check:** `scripts\smoke_test.py` level 2 verifies all 10 artefacts.

## 4. Commands, phase by phase

Each phase's protocol says which partitions are development and which are test. **Test commands
append an entry to `results/test_runs.jsonl` and must not be rerun casually.** Every run in the
publications is listed there (`docs/phase6_audit.md` §3).

| Phase | Steps (in order) | Output |
|---|---|---|
| 1b–2 DigiFace | `prepare_digiface.py extract scan fingerprint dedup pairs occlusion summary` → `verify_models.py` → `extract_embeddings.py --pilot 2048`, then the full run → `evaluate_phase2.py develop` → `evaluate_phase2.py final --final` (test, logged) | `reports/experiments/phase2_*.json`, `docs/baseline_evaluation.md` |
| 2 LFW | `prepare_lfw.py extract scan summary` → `lfw_detect_align.py` → `lfw_embed.py` → `evaluate_lfw.py --rehearse` → `evaluate_lfw.py --final` (View 2, logged) | `reports/experiments/lfw_*.json` |
| 3 | `prepare_phase3.py` → `train_visibility.py --config configs/experiments/phase3/ve_<fold>.yaml` (×5) → `extract_phase3.py --set p3` (also `p2test`, `lfw`) → `evaluate_phase3.py develop` (writes the private freeze) → `evaluate_phase3.py final --rehearse` → `evaluate_phase3.py final --final` (logged) → `evaluate_visibility.py` → `summarize_phase3.py` | `reports/experiments/phase3_*.json`, `phase3_tables.md` |
| 4 | `phase4_diagnose.py` → `prepare_phase4.py extract scan dedup pairs plans e1plan summary` → `train_visibility.py --config configs/experiments/phase4/vec_<fold>.yaml` (×5) → `extract_phase4.py --set p3dev`, then `--set p4` → `evaluate_phase4.py develop` → `final --rehearse` → `final --final` (logged) → `followup --final` → `phase4_detector.py --split dev`, then `view2` → `summarize_phase4.py` | `reports/experiments/phase4_*.json`, `phase4_tables.md` |
| 5 | `phase5_parity.py --n 1000` → `phase5_benchmark.py --n 40 --n-cpu 10` → `run_app.py` | `reports/experiments/phase5_*.json` |
| 6 | `smoke_test.py` → `phase6_lowq_audit.py` → `phase6_gpu_detector.py` → `fetch_digiface.py` → `prepare_phase4.py --partition p6 extract scan dedup pairs plans e1plan summary` → `extract_phase4.py --set p6 --pilot 512`, then the full run → `evaluate_phase4.py final --final --partition p6` (logged once) | `reports/experiments/phase6_*.json` |
| Simulations | `run_simulations.py` (`--scale quick` for a fast check) | `reports/simulations/` |

**Running long jobs.** Wrap every long job in the memory monitor:

```powershell
$g = ".venv-gpu\Scripts\python.exe"
& $g scripts\run_with_memory_monitor.py --log <private log>.json --min-free-gb 2.5 -- $g scripts\extract_phase4.py --set p6
```

The monitor aborts the job if free system RAM falls below the floor. Extraction is resumable
(atomic part shards), so rerunning the same command continues where it stopped.

## 5. Seeds

| Use | Seed | Where |
|---|---|---|
| Global / experiment | 20261026; experiment seeds 0, 1, 2 | `configs/research_protocol.yaml` |
| Phase 2 pairs and occlusion plans | 20261027 | protocol `pairs.seed` |
| Phase 3 sub-splits, pairs, plans | 20261027 (sub-split bands from SHA-256 of identity) | `docs/phase3_protocol.md` |
| Phase 4 p4_test pairs and plans; VE-C / MLP training | 20261101 | `veilx.data.untouched.P4`, phase4 configs |
| Phase 6 p6_test pairs and plans | 20261102 | `veilx.data.untouched.P6` |

Every occlusion sample's seed is derived from the global seed, fold, role, pair and side
(`derive_seed`). Each recipe is stored, so any render can be regenerated exactly.

## 6. Expected outputs

- **Committed reports.** Every published number traces to a committed JSON in
  `reports/experiments/`, with provenance: git commit, dirty flag, packages and argv. Rerunning
  a *development* script should reproduce it within the tolerances of §7.
- **Logged test runs.** Test runs are not meant to be rerun. Their reports are the record, and
  the Phase 2 rerun `0ca832e3b3b3` showed exact reproduction.
- **App parity reference:** `reports/experiments/phase5_parity.json`.
  - 1000/1000 A3 decisions match the offline evaluation;
  - LFW crops are pixel-identical;
  - CPU and GPU decisions agree.

## 7. Known nondeterminism

| Source | Effect | Mitigation / size |
|---|---|---|
| fp16 autocast on GPU vs fp32 on CPU | Embedding cosine ≥ 0.9999. Fused-score differences ≤ 0.01; decisions identical on 100/100 pairs | Store the precision with the features. All published runs used fp16 GPU extraction |
| Re-extraction vs stored float16 embeddings | Fused-score differences ≤ 0.0025 (1000 pairs) | — |
| cuDNN algorithm choice | Bit-level differences across GPUs / drivers | `cudnn.benchmark = False` in the extractor; tolerances as above |
| VE / VE-C training (GPU, AMP) | Retraining gives different weights. Exact reproduction of the trained estimators is **not** expected | Use the frozen weights (hash-pinned). Retraining is a new experiment |
| Thread scheduling in multi-process renders | None: outputs are keyed by item and written atomically | — |

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: psutil` from the memory monitor | Monitor run with `.venv` | Run the monitor with `.venv-gpu` (it has psutil) |
| onnxruntime warning `cublasLt64_13.dll ... missing` | onnxruntime-gpu 1.31 needs CUDA 13 / cuDNN 9; torch ships CUDA 12.8 | Harmless: SCRFD falls back to the CPU provider, the path used for every result (`reports/experiments/phase6_gpu_detector.json`) |
| Job "killed" with no traceback | Free RAM fell below the monitor floor (or another tool reaped it) | Rerun the same command (resumable). Reduce `--workers` |
| `frozen config modified` / SHA mismatch | A private freeze or weight file differs from the registered hash | Restore the original file. Never edit freezes |
| App refuses a photo ("no reliable face") | No face, a low-quality detection, or no LFW-framing declaration | Use a clearer frontal photo, or tick the LFW-style framing option only for loosely cropped centred portraits (D-054) |
| App ABSTAINS with "below the calibration support" | Pair quality is below the 1st percentile of calibration quality (D-055) | Expected for non-face or severely degraded images. The frozen score is unreliable there |
| Console `UnicodeEncodeError` | Windows code page | Set `PYTHONIOENCODING=utf-8` |

## 9. Runtime estimates (RTX 4060 Laptop, measured)

| Step | Time |
|---|---|
| Phase 2 / 3 extraction (Phase 3 store) | 1.8 h (6,425 s) |
| LFW detection + alignment (13,233 images, CPU) | 33 min |
| VE training per fold (Phase 3) | 10–32 min |
| VE-C training per fold (Phase 4) | 11–43 min |
| Phase 4 p4 extraction | 77 min (4,642 s) |
| DigiFace part download (2.9 GB) | 6 min each (at ~7.5 MiB/s) |
| Phase 6 preparation (extract + scan + dedup + pairs + plans) | about 1 h |
| Evaluation runs | minutes (CPU-bound) |
| App | ~2.7 s model load; 0.23 s per photo pair, 45 ms per pre-aligned pair (GPU) |
