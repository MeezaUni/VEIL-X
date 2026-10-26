# Evidence Registry (research integrity)

Machine-readable registry: `reports/evidence_registry.yaml`, validated by `src/veilx/evidence.py`.
`tests/test_evidence.py` enforces the rules below.

| Type | Meaning | Rule |
|---|---|---|
| `published_result` | Reported in a cited paper | Must cite a URL or DOI |
| `mathematical_calculation` | Exact calculation (e.g. binomial bounds) | Reproducible from a script |
| `synthetic_simulation` | Monte Carlo on synthetic scores | Artefact in `reports/simulations/`; statement says "simulated"; JSON carries `evidence_type` and the SIMULATED disclaimer. **Never evidence of face-recognition performance** |
| `real_measured_model_result` | Measured with a real model on real data | Requires provenance: git commit, dataset ids, model ids, split, protocol version, final-run log entry. **None exist** |
| `unverified_hypothesis` | Not yet tested | — |

Simulation outputs (`reports/simulations/`) are kept apart from future model results
(`results/`, git-ignored). Regenerate them with `python scripts/run_simulations.py`.
Each run is deterministic given its seed.
