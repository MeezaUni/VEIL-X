# VEIL-X — Development Log

Newest first. Every entry records what was done, the commands run, **actual** results,
known issues and next steps. No experimental results exist until an entry says so
explicitly, with its provenance record.

---

## 2026-10-27 — Phase 2 (milestone 2): extraction, develop freeze, single test run, results

### Extraction
- The first full run was **stopped by Claude Code under host memory pressure** (idle session):
  it was 101k items into the first group, with no shard saved.
- Fixes, then a restart on the user's instruction:
  - lazy per-group work lists and per-group worker pools;
  - render workers no longer import torch (550 → 107 MB each);
  - 32,768-item atomic part checkpoints, merged and verified;
  - val/cal groups first.
- Result: 22 groups × 2 backbones, 2,170,034 items.
  - 241 img/s end-to-end, 2.5 h; model 477 / 520 img/s; peak GPU 2.5 GiB.
  - 9.2 GB of private shards; no part files left.

### Develop stage and amendment (no test access)
- v1 freeze `700cd8e9…`. 22 of 80 fold-level LTT fits were infeasible, all at step one
  (fixed-sequence power failure).
- Added the p-value ordering (D-042, simulated validity and power tests) as post-hoc v2 arms.
- Re-freeze `0e8a43fb…` contains the v1 arms byte-identical.
- Rehearsal of the final code path on **val**; figures checked visually.
- Committed and pushed (f3376ea) **before** the test run.

### Final run (single, logged): `38d853d8884c`
- Clean, AdaFace / ArcFace: AUC 0.99856 / 0.99789. The calibration-fixed FAR 1 % threshold
  transfers (FMR 1.10 % / 1.00 % on 100k impostors).
- The val-fixed FAR 0.1 % realises 0.128 % / 0.148 % (estimation-only).
- Occlusion: FNMR up 4–15× at the clean threshold; held-out random patch is the hardest;
  mask–mask impostors raise FMR.
- LTT: 0 violations in 76 certifiable Test-ID cells. Test-OOD violations only with random patch
  held out. Oracle-visibility stratification removed that violation at a coverage cost.
  Clean-only calibration fails under any occlusion.
- C1 exploratory: co-visibility adds a weak residual signal beyond the feature norm.
- Audit: no identity overlap; label-shuffle AUC 0.50; val–test AUC gap ≤ 0.00025.

### Documentation and QA
- `baseline_evaluation.md` §5, `statistical_audit.md`, figures (`reports/figures/*.svg`),
  experiment registry (`reports/experiments/registry.json`), evidence RES-001…006,
  roadmap §4e/§4f (Phase 3 recommendation), README status.

---

## 2026-10-27 — Phase 2 (milestone 1): data expansion, frozen backbones, LFW status, evaluation plan

No test-split access has happened at this milestone. Nothing was trained or fine-tuned.

### Gate 0 (preflight)
- Repository root, remote and branch confirmed; `main` clean at b0f389a, equal to the remote.
- Stash `stash@{0}` (Teleport auto-stash) preserved: not applied, not dropped.
- Private data root outside the repository (`VEILX_DATA_ROOT`); about 282 GiB free.
- RTX 4060 Laptop (8 GiB, sm 8.9); `.venv-gpu` torch 2.11.0+cu128, CUDA OK.

### DigiFace expansion (protocol v0.4.0)
- Downloaded 5-image P2 and P3 from the official host:
  - sizes equal Content-Length; CRC-32 OK; 166,665 members each;
  - SHA-256 recorded (`1d99a951…`, `a70f53ae…`).
- `scripts/prepare_digiface.py extract scan fingerprint dedup pairs occlusion summary`:
  - scan 580 s, fingerprint 222 s, dedup 44 s, pairs 76 s; occlusion about 20 min;
  - 643,995 images / 101,999 identities; 0 anomalies, 0 decode failures, 0 exact duplicates;
  - the near-duplicate rule was rejected again.
- All 310,665 Phase 1b images keep their split and pixel hash (checked against the archive).
  The Phase 1b manifests are archived, not overwritten.
- Pair caps and plan sets changed (D-038).
  - Calibration: 3,409 + 3,409 identity-disjoint pairs.
  - Capacity: best certifiable sFMR 0.088 %; 0.1 % stays estimation-only (CALC-003).

### Models (D-037, D-040)
- `scripts/fetch_models.py`: CVLface AdaFace IR-101 and ArcFace IR-101 (WebFace4M).
  - Pinned HF commits; safetensors only; SHA-256 equal to the Hub LFS record.
  - Weights stay in `<data_root>/models`.
- `veilx.models`: strict loader, inference-only re-implementations, numpy preprocessing,
  fp32/fp16/bf16 extractor.
- `scripts/verify_models.py` on 20,000 val pairs:
  - output equals the authors' own modules (max diff 0.0);
  - CUDA output deterministic;
  - RGB ahead of BGR for both backbones (config agrees);
  - fp16 vs fp32 max pair-score diff ≤ 0.003; fp16 chosen.
- Found a gitignore trap again: the `models/` and `embeddings/` rules hid the new source
  packages. Allow-list entries were added after the weight rules; weights stay ignored.

### LFW (D-039): blocked, documented
- The reference notebook resolves to Kaggle `jessicali9530/lfw-dataset` v4 (full deep-funneled
  LFW).
- Its licence field points to a description with no licence. The official host is NXDOMAIN at
  the authoritative nameserver.
- Not downloaded. The parser for the official View-2 pairs is ready and tested on a synthetic
  fixture.
- The exact unblock options are in `docs/real_world_benchmark.md` §3. RMFRD, DFW, CPLFW,
  CALFW, MLFW and XQLFW were assessed but not downloaded.

### Extraction and evaluation code (ready; not yet run on test)
- `veilx.embeddings` (atomic, hash-verified shards with a completion marker; resumable; cache
  invalidation by fingerprint) and `scripts/extract_embeddings.py` (bounded render queue;
  re-rendered occlusions must match the plan's recipe hash).
- `veilx.eval` and `scripts/evaluate_phase2.py`: develop stage (val + cal only, frozen with
  SHA-256), then one logged `final --final` stage. Analysis plan in
  `docs/baseline_evaluation.md` (D-041).

### Quality
- 305 tests pass in the CPU environment; the torch tests also pass in `.venv-gpu`.
- Quality gate PASS: ruff, format, mypy strict, pytest, config, privacy scan, torch tests.

---

## 2026-10-26 — Phase 1b: real data, occlusion engine, experiment readiness (no recognition model)

### Session events
- The session moved machines mid-phase. The working tree was **auto-stashed** ("Teleport
  auto-stash") and reset to HEAD. All 13 uncommitted Phase 1b files were recovered with
  `git stash apply`; the stash is kept as a backup until the commit is verified.
- GitHub DNS resolution failed temporarily, then recovered before any push was needed.

### Data acquisition and preparation (real data; private root outside both git repositories)
- **Download:** the approved parts only (72-img P1, 5-img P1), via a resumable segmented
  HTTPS range downloader. The host throttles single connections to about 0.3 MB/s; 32
  parallel segments reached about 3.4 MiB/s.
- **Integrity:** sizes match the official Content-Length; SHA-256 recorded; CRC-32 passes
  for all 310,665 members; 0 members rejected by the safe extractor.
- **Scan** (Phase 1a adapter, now with parallel I/O):
  - 2,000 × 72 + 33,333 × 5 images;
  - **0 anomalies**.
- **Decode:** all 310,665 images decode; all 112×112 **RGBA** (format detail); alpha = 255
  everywhere, verified on every image.
- **Exact pixel duplicates:** 0.
- **Near-duplicate rule:** rejected by its pre-registered criterion (D-029). The base-rate
  analysis is in `docs/data_integrity_report.md`.
- **Splits:**
  - train 21,057 identities (182,871 images);
  - val 3,552 (31,495);
  - cal 3,570 (31,317);
  - test 7,154 (64,982).
- **Pairs:**
  - train 421,140; val 71,040; test_balanced 143,080; test_far 100,000;
  - cal_independent 2,380 (identity-disjoint); val_independent 2,368.
  - **Re-running the pair stage reproduced all six digests exactly.**
- **Occlusion plans:** 6,129,500 occluded samples (5 folds × 7 role sets) + 143,080 stress
  pairs. Isolation asserted; strata balanced; visibility monotone in level.
- **Disk:** 13.45 GiB in total (5.90 archives, 5.87 images, about 1.6 manifests/plans,
  0.1 reports).

### Statistics
- **SIM-010:** the identity percentile bootstrap undercovers FNMR (93.5%). The logit
  jackknife (94.8–95.7%) and linear jackknife for impostor rates (96.0–97.5%, conservative)
  were selected (D-032). The impostor design effect is about 2.
- **LTT assumptions vs the real pair generator** (`ltt_audit.md` addendum):
  - `cal_independent` is identity-disjoint and uses one root photo per pair, so the
    independence check passes.
  - Equal per-identity genuine weighting (cap 10) keeps the clustered test pairs on the
    calibration population (D-026).
  - Clustered train/val/test pairs are correlated evaluation evidence; they never enter
    calibration.

### Engineering
- **New modules:** `data.{download,archive,paths,images,dedup,integrity}`, `occlusion`,
  `stats.jackknife`, `training`.
- **New scripts:** `prepare_digiface.py`, `fetch_digiface.py`, `parts_benefit.py`,
  `gpu_smoke_check.py`.
- **Protocol v0.3.0** with changelog. Evidence types: added `real_data_statistic`
  (DATA-001…003), SIM-010 and CALC-002.
- **GPU infrastructure:** validated in a separate `.venv-gpu` (torch 2.11.0+cu128).
  - RTX 4060 Laptop, 8 GiB, capability 8.9, bf16 supported.
  - fp16/bf16 autocast and checkpoint round-trip OK.
  - **No model, no training.**

### Bugs found and fixed (all covered by tests)
1. **Zip special-file check.** Members storing only permission bits were classed as "special
   files", so valid archives could be rejected. Fixed by checking the file-type field only.
2. **Data-root probe.** It used a `.png` name, which is ignored everywhere, so any in-repo
   folder looked git-ignored. Fixed with an extension-less probe.
3. **Plan order dependence.** Condition plans depended on input order (contradicting the
   docstring). Fixed by sorting by `pair_id`.
4. **Duplicate-label policy.** The adapter kept the first copy of a cross-identity
   duplicate (ambiguous label) and hid the conflict from the dedup stage. Both copies are now
   excluded (D-033).
5. **Ignore rule (again).** The unanchored `data/` ignore rule hid `reports/data/`. The guard
   test caught it, and an explicit exception was added.
6. **Performance.** Serial adapter I/O ran at about 125 files/s (antivirus on-access cost);
   prefetch threads were added. Occlusion grids are now cached; outputs were verified
   **byte-identical** before and after, and F3 is 4× faster.
7. **Jackknife test arithmetic** was wrong in the test itself, not the code. Fixed.

### Quality gate (actual, final run before commit)
| Step | Result |
|---|---|
| ruff check / format | PASS / PASS |
| mypy strict (46 source files) | PASS |
| pytest | **281 passed**, 0 failed (≈ 62 s), including the end-to-end pipeline integration test on a synthetic fixture |
| validate_config | PASS (v0.3.0; provisional: clean margin, accepted-error target, dedup threshold) |
| scan_sensitive | PASS, 0 findings |
| Real-data reproducibility | 6/6 pair digests identical on re-run |

### Known limitations
- Template landmarks ignore per-image pose. Accessories in DigiFace renders partly confound F2.
- The near-duplicate signal is unusable on aligned synthetic renders, so only exact dedup is
  applied.
- Paired-difference intervals are not validated. The clean margin and the accepted-error
  target remain provisional.
- Additional DigiFace parts are recommended (CALC-002) but not downloaded.

---

## 2026-10-26 — Phase 1a: evaluation and statistical foundation (no models, no downloads)

### Git
- **First publication:**
  - Commit `3ed5cb13c50da886990323ed4c4dbb8ea1680b54` was pushed to `origin/main` after
    pre-flight checks: project-only git root, empty remote, explicit staging of 46 reviewed
    files, content grep for secrets and paths, privacy scan, quality gate.
  - The remote SHA matched local HEAD.
  - GitHub Actions `ci` on that commit: **success** (30 s).
- **Phase 1a commit:** the commit that contains this entry (`git log -1 -- docs/DEVELOPMENT_LOG.md`).
  Its SHA and CI result are reported in the session summary.

### Milestones
- **Learn-then-Test audit** (`docs/ltt_audit.md`):
  - The exact binomial on the accepted subsample is valid for the conditional risk under
    i.i.d. calibration pairs and acceptance locality. This is the construction of LTT §3.2.
  - The primary estimand changed to class-conditional sFMR/sFNMR (prevalence-robust;
    D-019).
  - Validity preconditions are enforced in code.
- **DigiFace adapter** (`veilx.data.digiface`, `veilx.data.png`): layouts, official per-ID
  counts, anomaly reporting, corrupt-PNG detection, duplicate exclusion, deterministic
  manifest. **Tested only on synthetic random-noise fixtures. No DigiFace data downloaded.**
- **Pair engine** (`veilx.pairs`): independent (identity-disjoint) and clustered modes;
  explicit infeasibility; pair counts kept separate from effective independent counts;
  provenance digests.
- **Metrics** (`veilx.metrics`) and the identity-cluster bootstrap (`veilx.stats.bootstrap`).
- **Risk control** (`veilx.selective`): `LTTCalibrator`, `EmpiricalCalibrator`,
  `fit_stratified`, validation-only candidate ordering.
- **Simulations** (`veilx.sim`, `reports/simulations/`): 9 experiments, registered as
  SIM-001…009. Summary in `docs/simulation_findings.md`. All are **SIMULATED**.
- **Evidence registry** (`veilx.evidence`, `reports/evidence_registry.yaml`,
  `docs/evidence_registry.md`).
- **Protocol v0.2.0 (provisional):** class-conditional estimand, `candidate_ordering:
  validation_only`, assumption A7.

### Bugs and pitfalls found during this phase, all fixed and covered by tests
1. **Fixed-sequence power collapse.** Ordering candidates widest-first made LTT infeasible in
   every quick simulation run, because the first hypotheses accept almost nothing. Fixed by
   validation-only pruning and ordering (D-020).
2. **Trivial genuine pairs.** A byte-identical copy and an occlusion variant of the same
   photo were treated as different sources. "Source" is now the root-photo content hash
   (D-021), and the same keys feed the independence check.
3. **Unrealistic synthetic model.** The first synthetic model had FNMR about 68% at FAR 1%, so
   LTT was infeasible everywhere. It was re-parameterised to a more separable regime. The
   shift experiment became a severity sweep, because a single extreme shift exaggerated
   magnitudes.
4. **Typing.** Strict-mypy issues in the simulation code: default-argument lambdas,
   `Literal`s and `Any` returns. They were replaced with typed partials and factories.
   **Re-running all simulations afterwards produced byte-identical reports.**
5. Fixture identity collisions in two new tests (test bugs, not engine bugs).

### Commands run
```
.venv\Scripts\python.exe scripts\run_simulations.py --scale quick --out <scratch>   # smoke runs
.venv\Scripts\python.exe scripts\run_simulations.py --scale full                    # ~2 min
.venv\Scripts\python.exe scripts\quality_gate.py
```

### Test results (actual, final run before commit)
| Step | Result |
|---|---|
| ruff check / format | PASS / PASS |
| mypy (strict, 30 source files) | PASS |
| pytest | **209 passed**, 0 failed, 0 skipped (≈15 s) |
| validate_config | PASS (protocol v0.2.0, provisional) |
| scan_sensitive | PASS, 0 findings |
| Full simulations re-run after refactor | 9/9 reports byte-identical |

### Known limitations
- The adapter has not seen real DigiFace files. Their image format and resolution are
  unverified.
- Identity-bootstrap FNMR coverage is 93.3% in simulation (slightly below nominal), and FMR
  coverage is conservative. A refined scheme is planned for Phase 1b.
- `order_candidates_on_validation` needs a validation set from the same distribution as
  calibration.
- The cost-sensitivity experiment selects among certified rules using oracle (true synthetic)
  costs.
- No occlusion synthesis, embeddings, quality or visibility estimators yet.

### Next steps
`implementation_roadmap.md` §4b (Phase 1b), after the PI decisions in `decision_log.md`.

---

## 2026-10-26 — Phase 0: protocol lock and scaffolding (no models, no data)

### Milestones
- **Security pre-flight.** Git originally resolved to the parent Desktop repository (0
  commits, no remote). Git output was scoped to the project path, so no unrelated file names
  were listed. `.gitignore` was written **before** `git init -b main` inside the project folder.
  The parent `.git` entry count was unchanged (8 → 8); the parent now sees the project as one
  untracked directory. `origin` = `https://github.com/MeezaUni/VEIL-X.git`. `git ls-remote`
  succeeded with no refs, so the remote is **empty** and there is no existing history to
  preserve.
- **Licence/access audit:**
  - The assigned Kaggle URL returns HTTP 404 (controls 200) → blocked.
  - DigiFace-1M confirmed under R-UDA, with part sizes from HTTP HEAD (~23.2 GiB total).
  - ROF repo MIT, but images are web-collected and shipped as pickles → conditional.
  - CPLFW/XQLFW pages carry no licence text; the MLFW page is unreadable → unresolved.
  - Details: `docs/dataset_feasibility.md`.
- **Protocol lock (provisional v0.1.0):** `configs/research_protocol.yaml` +
  `docs/protocol_design.md`. Assumptions A1–A6, verbatim prediction target, identity-disjoint
  calibration matching, LOFO isolation scope, required arms, the C1 feature-set comparison, and
  the dedup procedure without a fixed threshold.
- **Feasibility** (`scripts/feasibility_report.py`, exact binomial):
  - Certifying FMR ≤ 1% needs ≥ 299 independent impostor pairs; ≤ 0.1% needs ≥ 2,995.
  - A 5% selective error target is certifiable from ≥ 59 / 93 / 286 accepted pairs at
    0% / 2% / 3% observed error.
  - FAR 0.1% is estimation-only under the recommended DigiFace subset.
- **Scaffold:**
  - `pyproject.toml` (hatchling; numpy, pydantic, PyYAML; dev: pytest, ruff, mypy).
  - `src/veilx/` with config schema/loader, manifests, splits, interfaces, provenance, logging,
    seeding, guards, errors and feasibility stats.
  - `scripts/`: validate_config, feasibility_report, scan_sensitive, quality_gate.
  - CI workflow `.github/workflows/ci.yml`. It runs only once pushed.
- **Docs:**
  - Targeted updates to the six Phase 0 documents.
  - New: `protocol_design.md`, `dataset_feasibility.md`, `interfaces.md`, `decision_log.md`,
    and this log.

### Environment (actual)
- Windows 11, PowerShell. Host Python 3.14.6 is **outside the supported range** (>=3.11,<3.13),
  so the project uses **CPython 3.12.14 via uv 0.12.21** in `.venv/` (git-ignored).
- GPU detected by `nvidia-smi`: NVIDIA GeForce RTX 4060 Laptop GPU, driver 610.47, 8188 MiB.
  It is not used in Phase 0.
- Installed: numpy 2.5.4, pydantic 2.14.1, PyYAML 6.0.3, pytest 9.1.1, ruff 0.17.2, mypy 2.4.0.

### Commands run (from the project root)
```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest ruff mypy types-PyYAML
.venv\Scripts\python.exe scripts\validate_config.py      # exit 0
.venv\Scripts\python.exe scripts\feasibility_report.py   # exit 0
.venv\Scripts\python.exe scripts\quality_gate.py         # exit 0
```

### Test results (actual, final run)
| Step | Result |
|---|---|
| `ruff check .` | PASS |
| `ruff format --check .` | PASS |
| `mypy` (strict, 15 source files) | PASS: no issues |
| `pytest -q` | **112 passed**, 0 failed |
| `validate_config.py` | PASS (protocol status provisional; 7 provisional items listed) |
| `scan_sensitive.py` | PASS: 44 candidate files, 0 findings |

### Issues found and fixed during this session
1. **Over-broad ignore rule.** The unanchored `data/` pattern in `.gitignore` also ignored the
   `src/veilx/data/` package. Ruff respects `.gitignore`, so it silently skipped those files,
   and they would never have been committed (CI would have failed on import). Fixed with an
   explicit `!src/veilx/data/` exception. Regression tests were added: no
   `.py/.md/.yaml/.toml` file under src/tests/scripts/configs/docs may be ignored. Two lint
   issues in the previously skipped module were then fixed.
2. Ruff line-length/style findings and an unused `noqa` were fixed via `ruff format` and small
   edits. mypy missing stubs for optional `torch` were handled via a targeted override.

### Known limitations
- No dataset adapters, pair builder, metrics, bootstrap or LTT calibrator yet (Phase 1a).
- `min_n_to_certify` uses a search over n. Because of flooring, the bound is not strictly
  monotone in n, so values are guides; the zero-error formula is exact.
- Interface protocols are structurally checked only (`runtime_checkable` checks method
  presence, not signatures).
- The CI workflow has not run yet (nothing pushed).

### Git state (at end of Phase 0)
- Repository root: the project folder. Branch `main`. No commits at that time, per the Phase 0
  instruction (D-018). This was resolved in Phase 1a.

### Next steps
See `implementation_roadmap.md` §4 (Phase 1a) and `decision_log.md` → Pending PI decisions.
