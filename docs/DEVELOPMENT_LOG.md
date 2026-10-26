# VEIL-X — Development Log

Newest first. Every entry records what was done, the commands run, **actual** results,
known issues and next steps. No experimental results exist until an entry says so
explicitly, with its provenance record.

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
