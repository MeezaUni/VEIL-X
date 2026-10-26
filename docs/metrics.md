# VEIL-X Metric Definitions

Implementation: `src/veilx/metrics/`. Tests: `tests/test_metrics.py`. These are checked
against brute-force references, hand calculations, a Gaussian reference, and the published
AUGRC closed form.

**Conventions:**
- Label y = 1 means genuine, y = 0 means impostor.
- A higher score means more likely genuine.
- A pair is **accepted as a match iff score ≥ t**.
- Decisions are MATCH = 1, NON-MATCH = 0, ABSTAIN = −1.

## Score-based (non-selective)

| Metric | Definition | Notes / source |
|---|---|---|
| FAR = FMR | #(impostor score ≥ t) / #impostors | Standard biometric definition |
| TAR = 1 − FNMR | #(genuine score ≥ t) / #genuines | Standard |
| ROC | (FAR, TAR) at every distinct threshold, from (0, 0) at t = +∞ | — |
| ROC-AUC | P(s_gen > s_imp) + ½ P(tie) (Mann–Whitney, mid-ranks) | Equals the trapezoidal ROC area. Tested three ways, including brute force |
| EER | FAR = FNMR crossing, linearly interpolated between adjacent ROC points | Gaussian test reproduces Φ(−1) |
| Threshold at FAR f | Smallest t with empirical FAR ≤ f. With ties, it never exceeds f | **Refuses** (`InsufficientSamplesError`) when #impostors < 10/f (protocol `min_impostor_pairs_factor`) |
| TAR@FAR, *benchmark mode* | Threshold set on the evaluated set itself | Literature convention; comparability only |
| TAR@FAR, *deployment mode* | Threshold from **calibration** impostors, applied to test; reports *realised* FAR and TAR | Both sides need ≥ 10/f impostors |

## Decision-based (selective)

| Metric | Definition | Source |
|---|---|---|
| Coverage | accepted / all; also per class (genuine, impostor) | — |
| Selective FMR | FM / accepted impostors | — |
| Selective FNMR | FNM / accepted genuines | — |
| Accepted-decision error | (FM + FNM) / accepted = P(d ≠ y \| accepted) | Selective risk (Traub et al. 2024, Eq. 2) |
| Generalized risk | (FM + FNM) / all = P(d ≠ y, accepted) | Traub et al. 2024, Eq. 3 |
| Cost-weighted risk | (c_FM·FM + c_FNM·FNM) / accepted | Registered cost ratio 10:1 (provisional) |
| Risk–coverage curve | Sweep a confidence threshold over a *fixed* non-abstaining base decision; tied confidences are accepted together | — |
| AUGRC | ∫ generalized risk d(coverage), trapezoid between tie groups | Traub et al. 2024, Eq. 6. Equals the closed form (1−AUROC_f)·acc·(1−acc) + ½(1−acc)² (Eq. 7) **exactly**. This is tested, and derived below |
| AURC | Σ_g ΔC_g · SR(C_g) (right endpoints) | Traub et al. 2024, Eq. 4. Without ties it equals the mean selective risk over k = 1..N |
| Confusion matrix / decision distribution | label × {match, non-match, abstain} | — |

Empty denominators return `None`, never 0.

### AUGRC trapezoid = closed form (derivation)
Sort by confidence, with N samples, F failures and S = N − F successes. Let F_k be the number
of failures among the top k.
- The trapezoid area is (1/N²)·(Σ_k F_k − F/2).
- Σ_k F_k = Σ_failures (N − pos + 1) = F + F(F−1)/2 + #{failure ranked above success}.
- #{failure ranked above success} = (1 − AUROC_f)·F·S.

Hence AUGRC = ½(F/N)² + (1 − AUROC_f)(S/N)(F/N). With ties, AUROC_f counts ½ per tie,
matching the linear interpolation within tie groups.

## Calibration of p(match)

| Metric | Definition | Notes |
|---|---|---|
| ECE | Σ_b (n_b/N)·\|mean(y)_b − mean(p)_b\| | **Binary p(match)** variant with **15 equal-mass** bins (registered). Guo et al. 2017 use top-label confidence with equal-width bins; equal-width is available for comparability |
| Brier | mean((p − y)²) | — |
| Reliability diagram | Per-bin mean prediction, observed frequency and count; dependency-free SVG | — |

## Uncertainty

Identity-cluster bootstrap (`veilx.stats.bootstrap`):
- Identities are resampled with replacement.
- A genuine pair's weight is W_i; an impostor pair's weight is W_i·W_j.
- Method differences use the same weights (paired).
- A pair-level bootstrap exists only for comparison.

In simulation (SIM-001), the identity bootstrap covered FNMR in 93.3% and FMR in 99.3% of
runs at nominal 95%, against 83–84% for the pair bootstrap. FMR intervals are therefore
conservative and FNMR intervals slightly anti-conservative in that model. A recentred or
two-way-aware variant (cf. Conti & Clémençon 2024) is a Phase 1b item.
