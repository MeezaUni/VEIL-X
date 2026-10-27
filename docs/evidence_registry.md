# Evidence Registry (research integrity)

Machine-readable registry: `reports/evidence_registry.yaml`, validated by `src/veilx/evidence.py`.
`tests/test_evidence.py` enforces the rules below.

| Type | Meaning | Rule |
|---|---|---|
| `published_result` | Reported in a cited paper | Must cite a URL or DOI |
| `mathematical_calculation` | Exact calculation (e.g. binomial bounds) | Reproducible from a script |
| `synthetic_simulation` | Monte Carlo on synthetic scores | Artefact in `reports/simulations/`; statement says "simulated"; JSON carries `evidence_type` and the SIMULATED disclaimer. **Never evidence of face-recognition performance** |
| `real_measured_model_result` | Measured with a real model on real data | Requires provenance: git commit, dataset ids, model ids, split, protocol version, final-run log entry. **Phase 2: RES-001…011**: DigiFace (run 38d853d8884c, reproduced by 0ca832e3b3b3) and LFW (run 79331a84f520; RES-011 is post-hoc exploratory) |
| `real_data_statistic` | Measured facts about the prepared dataset (counts, integrity, splits, pair sets) | Artefact in `reports/data/`; statement may not contain performance terms (TAR, FAR, EER, AUC, accuracy…). **Never model performance** |
| `unverified_hypothesis` | Not yet tested | — |

Simulation outputs (`reports/simulations/`) are kept apart from future model results
(`results/`, git-ignored). Regenerate them with `python scripts/run_simulations.py`.
Each run is deterministic given its seed.
