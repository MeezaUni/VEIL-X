# Statistical Simulation Findings (Phase 1a)

> **SIMULATED.** Every number below comes from a synthetic score model
> (`veilx.sim.model.ScoreModel`). It characterises the *statistical procedures* VEIL-X will use.
> **None of it is evidence about face-recognition performance**, and magnitudes depend on
> arbitrary synthetic parameters. Reports: `reports/simulations/*.json`. Registry: SIM-001…009.
> Reproduce with `python scripts/run_simulations.py`. The runs are deterministic: re-running
> after a code refactor produced byte-identical reports.

| ID | Question | Finding (simulated) | Design consequence |
|---|---|---|---|
| SIM-001 | Does the identity-cluster bootstrap cover at 95%? | FNMR 93.3% (MC s.e. 1.4%), FMR 99.3%. A pair-level bootstrap gave 83–84% | Use the identity bootstrap. Never use the pair bootstrap. FNMR intervals are slightly short and FMR intervals conservative; a refined two-way scheme is a Phase 1b item |
| SIM-002 | Does reusing identities in calibration break LTT? | Violations: 2.5% (independent), 3.75% (120 identities), 8.25% (30 identities) at δ = 10% | Keep the identity-disjoint matching mandatory. Dependence increases risk even when it stays below δ |
| SIM-003 | How many calibration pairs are needed? | 100: always infeasible. 300: infeasible in 53.5% of runs. 1,000: 90.4% coverage. 3,000: 94.3% coverage (oracle 96.8%). Violations ≤ 1% | Plan for ≥ 1,000 *independent* calibration pairs per calibrated unit (per stratum if stratified) |
| SIM-004 | What happens under shift? | Violations: 0.5% (none), 16% (severity 0.1), 95.5% (0.25), 100% (0.5 and 1.0) | Guarantees must not be claimed on held-out families (H2a is plausible). Measure the realised violation |
| SIM-005 | Does stratified calibration help? | Covariate-only shift: stratified 1% vs global 100% violations, but 17% vs 87% coverage. Conditional shift: both fail (83% vs 100%) | Visibility stratification can help only if the shift acts *through* visibility. Its coverage cost must be reported (H2b) |
| SIM-006 | Is the 0.5 pp clean margin testable? | Power 28% / 68% / 100% with 200 / 1,000 / 3,333 test identities (4 genuine pairs each) | The 0.5 pp margin needs roughly ≥ 3,000 clean test identities. With the DigiFace 5-img part it is feasible; with ROF alone it is not |
| SIM-007 | Is the pooled estimand prevalence-sensitive? | Mixture estimand: 9% violations at test prevalence 0.95 (δ = 5%). Class-conditional: ≤ 0.5% at every prevalence | Primary estimand = class-conditional sFMR/sFNMR (D-019) |
| SIM-008 | Does the cost ratio change the chosen rule? | 1:1 picks (0.75, 3.25). 10:1 and 100:1 pick (1.25, 3.75), with sFMR 0.55% → 0.14% at equal coverage | The 10:1 ratio is a meaningful tie-breaker among certified rules |
| SIM-010 | Which interval method covers genuine (one-identity) and impostor (two-identity) pair rates? | 1,000 reps. FNMR: identity percentile bootstrap **0.935** (G=50, 150), logit jackknife **0.948–0.957**, linear jackknife 0.944–0.956. FMR: linear jackknife **0.960–0.975**, logit jackknife 0.986–0.996, product-weight bootstrap 0.98–0.99, pair bootstrap 0.81–0.90. Impostor design effect ≈ 2 (1.16 at FAR 0.1%) | Logit jackknife for genuine rates, linear jackknife for impostor rates (conservative: its variance is over-estimated 1.6–1.9×). The Phase 1a identity percentile bootstrap is retired for FNMR. Paired-difference intervals still need validation |
| SIM-009 | Is the Clopper–Pearson bound conservative? | ≥ 96.5% coverage in all tested settings | The feasibility calculations are trustworthy |
