# VEIL-X — Development Log

Newest first. Every entry records what was done, the commands run, **actual** results,
known issues and next steps. No experimental results exist until an entry says so
explicitly, with its provenance record.

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

### Git state
- Repository root: the project folder. Branch `main`. **No commits, nothing staged, nothing
  pushed** (Phase 0 instruction; D-018 pending PI confirmation).

### Next steps
See `implementation_roadmap.md` §4 (Phase 1a) and `decision_log.md` → Pending PI decisions.
