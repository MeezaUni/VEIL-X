## Confirmatory run `903f6a5c7de3` (frozen `bba129002cf8`)

✗ = 95 % identity-jackknife CI excludes the target; † = point estimate above target.

### test_ood: coverage / sFNMR / sFMR, % (primary|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 93.0 / 8.89 ✗ / 4.83 | 91.2 / 3.91 / 2.29 | 94.3 / 6.99 ✗ / 3.60 | 94.6 / 18.96 ✗ / 1.63 | 91.7 / 4.02 / 2.75 |
| B1_quality_global | 100.0 / 10.49 ✗ / 5.51 ✗ | 100.0 / 7.52 ✗ / 1.87 | 95.9 / 6.75 ✗ / 1.38 | 100.0 / 12.78 ✗ / 4.72 | 96.8 / 4.41 / 1.71 |
| A3_quality | 90.9 / 7.55 ✗ / 2.59 | 95.7 / 5.72 ✗ / 1.27 | 92.4 / 5.84 ✗ / 1.24 | 86.0 / 7.26 ✗ / 2.04 | 94.7 / 4.33 / 1.47 |
| A4_original | 93.0 / 8.07 ✗ / 3.30 | 95.6 / 5.35 ✗ / 0.92 | 94.3 / 7.93 ✗ / 1.23 | 98.2 / 17.73 ✗ / 2.12 | 98.4 / 7.19 ✗ / 1.50 |
| C4_conservative | 90.8 / 7.19 ✗ / 2.57 | 95.9 / 5.92 ✗ / 1.17 | 92.7 / 7.21 ✗ / 1.07 | 89.5 / 10.53 ✗ / 1.88 | 98.6 / 7.32 ✗ / 1.49 |
| N4_novelty_aware | 90.8 / 7.19 ✗ / 2.57 | 96.8 / 5.99 ✗ / 1.15 | 92.6 / 6.74 ✗ / 1.04 | 87.7 / 9.70 ✗ / 1.65 | 97.9 / 6.92 ✗ / 1.44 |
| H4_quality_first_hybrid | 89.9 / 6.96 ✗ / 2.58 | 91.1 / 5.84 ✗ / 1.14 | 83.0 / 6.88 ✗ / 1.10 | 51.6 / 7.48 ✗ / 1.15 | 94.9 / 6.57 ✗ / 1.49 |
| A5_oracle | 89.0 / 6.62 ✗ / 2.21 | 97.9 / 6.87 ✗ / 1.33 | 96.7 / 6.80 ✗ / 2.19 | 88.3 / 9.10 ✗ / 2.00 | 91.8 / 3.64 / 1.74 |

### e1: coverage / sFNMR / sFMR, % (primary|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 94.1 / 6.27 ✗ / 2.63 | 90.5 / 5.44 ✗ / 1.65 | 94.4 / 6.32 ✗ / 2.76 | 95.4 / 8.93 ✗ / 1.53 | 91.0 / 5.32 † / 2.00 |
| B1_quality_global | 100.0 / 8.10 ✗ / 2.51 | 100.0 / 8.70 ✗ / 2.26 | 95.7 / 6.30 ✗ / 1.36 | 100.0 / 8.22 ✗ / 2.45 | 96.2 / 6.03 ✗ / 1.59 |
| A3_quality | 93.2 / 5.76 ✗ / 1.11 | 91.1 / 5.46 ✗ / 1.08 | 90.6 / 5.09 † / 1.03 | 91.1 / 4.65 / 1.15 | 90.7 / 4.65 / 1.26 |
| A4_original | 98.0 / 8.41 ✗ / 1.69 | 95.9 / 7.58 ✗ / 1.08 | 99.3 / 9.97 ✗ / 1.56 | 100.0 / 12.14 ✗ / 1.23 | 99.1 / 9.46 ✗ / 1.80 |
| C4_conservative | 96.5 / 7.96 ✗ / 1.42 | 97.5 / 8.93 ✗ / 1.52 | 95.1 / 8.55 ✗ / 1.35 | 97.3 / 9.62 ✗ / 1.27 | 98.3 / 8.78 ✗ / 1.84 |
| N4_novelty_aware | 94.9 / 7.26 ✗ / 1.20 | 95.7 / 7.84 ✗ / 1.27 | 91.8 / 6.98 ✗ / 0.93 | 91.6 / 7.28 ✗ / 0.67 | 88.0 / 5.26 † / 0.43 |
| H4_quality_first_hybrid | 76.2 / 7.20 ✗ / 1.44 | 71.1 / 8.43 ✗ / 1.52 | 56.7 / 9.33 ✗ / 1.36 | 31.8 / 7.81 ✗ / 1.32 | 12.1 / 4.44 / 1.65 |
| A5_oracle | 95.3 / 7.51 ✗ / 1.19 | 100.0 / 10.32 ✗ / 1.60 | 100.0 / 7.81 ✗ / 2.64 | 97.4 / 10.14 ✗ / 1.18 | 97.2 / 9.83 ✗ / 1.25 |

### test_id: coverage / sFNMR / sFMR, % (primary|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 94.3 / 7.22 ✗ / 3.35 | 89.7 / 8.40 ✗ / 2.36 | 94.0 / 8.81 ✗ / 3.82 | 95.7 / 7.82 ✗ / 2.34 | 90.4 / 8.26 ✗ / 2.84 |
| B1_quality_global | 100.0 / 7.83 ✗ / 3.02 | 100.0 / 9.96 ✗ / 3.64 | 95.2 / 7.53 ✗ / 2.22 | 100.0 / 8.14 ✗ / 2.84 | 95.4 / 7.71 ✗ / 2.83 |
| A3_quality | 94.0 / 5.89 ✗ / 1.27 | 89.7 / 6.34 ✗ / 1.67 | 90.2 / 6.16 ✗ / 1.59 | 93.2 / 5.36 ✗ / 1.63 | 89.4 / 6.05 ✗ / 2.00 |
| A4_original | 94.8 / 6.06 ✗ / 1.61 | 92.0 / 6.86 ✗ / 1.69 | 90.0 / 6.72 ✗ / 1.58 | 94.9 / 6.81 ✗ / 1.52 | 90.6 / 6.23 ✗ / 2.09 |
| C4_conservative | 94.6 / 6.05 ✗ / 1.55 | 90.5 / 6.95 ✗ / 1.59 | 89.3 / 6.39 ✗ / 1.39 | 94.6 / 6.68 ✗ / 1.41 | 91.6 / 6.72 ✗ / 2.04 |
| N4_novelty_aware | 94.5 / 6.01 ✗ / 1.53 | 92.8 / 7.32 ✗ / 1.92 | 90.1 / 6.14 ✗ / 1.56 | 94.4 / 6.59 ✗ / 1.40 | 90.4 / 6.17 ✗ / 2.04 |
| H4_quality_first_hybrid | 90.6 / 5.73 ✗ / 1.49 | 86.6 / 6.83 ✗ / 1.63 | 87.4 / 6.70 ✗ / 1.55 | 90.6 / 6.36 ✗ / 1.44 | 87.8 / 6.38 ✗ / 1.98 |
| A5_oracle | 94.4 / 5.91 ✗ / 1.54 | 91.6 / 7.76 ✗ / 1.51 | 91.5 / 6.16 ✗ / 1.97 | 94.5 / 6.66 ✗ / 1.43 | 91.1 / 6.79 ✗ / 1.67 |

### test_ood: coverage / sFNMR / sFMR, % (far1|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 83.1 / 9.05 ✗ / 0.96 | 83.8 / 3.74 / 0.36 | 82.3 / 6.22 ✗ / 0.35 | 86.2 / 22.74 ✗ / 0.25 | 84.8 / 3.94 / 0.50 |
| B1_quality_global | 88.9 / 12.68 ✗ / 0.68 | 90.8 / 6.55 ✗ / 0.15 | 87.6 / 8.06 ✗ / 0.16 | 88.4 / 15.79 ✗ / 0.68 | 89.8 / 5.07 † / 0.15 |
| A3_quality | 75.3 / 8.25 ✗ / 0.29 | 79.2 / 8.52 ✗ / 0.12 | 67.8 / 7.91 ✗ / 0.17 | 23.0 / 7.68 ✗ / 0.25 | 72.3 / 6.23 ✗ / 0.02 |
| A4_original | 79.9 / 9.80 ✗ / 0.48 | 84.7 / 7.20 ✗ / 0.10 | 77.8 / 10.95 ✗ / 0.28 | 86.5 / 24.61 ✗ / 0.39 | 94.0 / 13.10 ✗ / 0.15 |
| C4_conservative | 83.6 / 8.92 ✗ / 0.52 | 87.4 / 7.86 ✗ / 0.12 | 73.0 / 9.70 ✗ / 0.25 | 53.2 / 13.55 ✗ / 0.20 | 94.0 / 13.10 ✗ / 0.15 |
| N4_novelty_aware | 83.1 / 9.02 ✗ / 0.48 | 84.9 / 6.49 ✗ / 0.10 | 67.0 / 9.49 ✗ / 0.26 | 41.5 / 11.64 ✗ / 0.22 | 93.2 / 12.47 ✗ / 0.14 |
| H4_quality_first_hybrid | 38.1 / 7.43 ✗ / 0.23 | 80.0 / 6.37 ✗ / 0.10 | 67.0 / 9.44 ✗ / 0.26 | 41.5 / 11.62 ✗ / 0.23 | 88.7 / 12.11 ✗ / 0.14 |
| A5_oracle | 76.1 / 8.41 ✗ / 0.23 | 90.4 / 8.51 ✗ / 0.17 | 88.8 / 10.24 ✗ / 0.20 | 47.5 / 8.92 ✗ / 0.16 | 82.8 / 5.27 † / 0.12 |

### e1: coverage / sFNMR / sFMR, % (far1|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 84.6 / 6.17 ✗ / 0.38 | 81.0 / 5.41 ✗ / 0.26 | 81.1 / 5.45 ✗ / 0.26 | 86.7 / 10.77 ✗ / 0.18 | 82.0 / 5.24 † / 0.31 |
| B1_quality_global | 89.2 / 10.01 ✗ / 0.14 | 87.8 / 7.94 ✗ / 0.15 | 85.7 / 7.87 ✗ / 0.10 | 90.5 / 9.87 ✗ / 0.19 | 86.4 / 7.45 ✗ / 0.12 |
| A3_quality | 77.5 / 6.86 ✗ / 0.08 | 59.0 / 8.36 ✗ / 0.07 | 55.8 / 8.08 ✗ / 0.09 | 20.7 / 9.58 ✗ / 0.13 | 54.2 / 6.95 ✗ / 0.01 |
| A4_original | 88.5 / 11.13 ✗ / 0.18 | 82.6 / 11.88 ✗ / 0.03 | 90.6 / 15.35 ✗ / 0.21 | 93.7 / 17.01 ✗ / 0.16 | 91.7 / 18.13 ✗ / 0.12 |
| C4_conservative | 88.8 / 10.06 ✗ / 0.17 | 86.1 / 12.78 ✗ / 0.08 | 77.4 / 13.43 ✗ / 0.23 | 78.0 / 13.43 ✗ / 0.12 | 89.9 / 16.73 ✗ / 0.11 |
| N4_novelty_aware | 86.2 / 9.46 ✗ / 0.10 | 79.5 / 9.25 ✗ / 0.01 | 47.5 / 15.05 ✗ / 0.24 | 30.3 / 12.84 ✗ / 0.13 | 72.4 / 8.23 ✗ / 0.03 |
| H4_quality_first_hybrid | 55.2 / 9.34 ✗ / 0.11 | 59.3 / 10.11 ✗ / 0.02 | 47.5 / 14.96 ✗ / 0.24 | 30.3 / 12.81 ✗ / 0.13 | 11.8 / 11.71 ✗ / 0.21 |
| A5_oracle | 84.3 / 8.66 ✗ / 0.11 | 90.1 / 12.63 ✗ / 0.11 | 92.8 / 12.36 ✗ / 0.22 | 78.0 / 11.68 ✗ / 0.11 | 83.8 / 12.94 ✗ / 0.03 |

### Endpoints (P1 safety, P2 benefit; 10 primary cells)

| Arm | violations (primary v2) | safe | sig. worse sFNMR cells | sig. worse sFMR cells | mean coverage gain vs A3 (pp) | cells sig. gain | better than A3 |
|---|---|---|---|---|---|---|---|
| A3_quality | 6 | — | — | — | — | — | — |
| A4_original | 10 | — | — | — | — | — | — |
| A5_oracle | 9 | — | — | — | — | — | — |
| B0_raw_global | 7 | — | — | — | — | — | — |
| B1_quality_global | 10 | — | — | — | — | — | — |
| C4_conservative | 10 | False | F1_mask|e1, F2_sunglasses|e1, F3_hand_object|test_ood, F3_hand_object|e1, F4_random_patch|test_ood, F4_random_patch|e1, F5_extreme_lighting|test_ood, F5_extreme_lighting|e1 | F1_mask|e1, F2_sunglasses|e1, F3_hand_object|e1, F5_extreme_lighting|e1 | +3.60 | 8 | False |
| N4_novelty_aware | 9 | False | F1_mask|e1, F2_sunglasses|test_ood, F2_sunglasses|e1, F3_hand_object|test_ood, F3_hand_object|e1, F4_random_patch|test_ood, F4_random_patch|e1, F5_extreme_lighting|test_ood, F5_extreme_lighting|e1 | — | +1.15 | 7 | False |
| H4_quality_first_hybrid | 9 | False | F1_mask|e1, F2_sunglasses|e1, F3_hand_object|test_ood, F3_hand_object|e1, F4_random_patch|e1, F5_extreme_lighting|test_ood | F1_mask|e1, F2_sunglasses|e1, F3_hand_object|e1 | -25.78 | 0 | False |

**Registered recommendation: `A3_quality`**

### Paired differences vs A3 (primary v2): ΔsFNMR / ΔsFMR / Δcoverage, pp

| Cell | C4 | N4 | H4 | A4 original | A5 oracle |
|---|---|---|---|---|---|
| F1_mask|e1 | +2.20 / +0.30 / +3.28 | +1.50 / +0.08 / +1.68 | +1.45 / +0.33 / -16.98 | +2.65 / +0.58 / +4.84 | +1.76 / +0.08 / +2.12 |
| F1_mask|test_ood | -0.36 / -0.02 / -0.09 | -0.36 / -0.02 / -0.09 | -0.60 / -0.01 / -0.94 | +0.51 / +0.71 / +2.15 | -0.94 / -0.38 / -1.83 |
| F2_sunglasses|e1 | +3.47 / +0.44 / +6.43 | +2.38 / +0.19 / +4.61 | +2.97 / +0.43 / -20.01 | +2.12 / -0.00 / +4.84 | +4.85 / +0.52 / +8.93 |
| F2_sunglasses|test_ood | +0.20 / -0.10 / +0.20 | +0.27 / -0.12 / +1.09 | +0.12 / -0.13 / -4.60 | -0.37 / -0.36 / -0.12 | +1.14 / +0.06 / +2.14 |
| F3_hand_object|e1 | +3.46 / +0.32 / +4.50 | +1.89 / -0.10 / +1.24 | +4.24 / +0.33 / -33.86 | +4.88 / +0.53 / +8.72 | +2.72 / +1.61 / +9.41 |
| F3_hand_object|test_ood | +1.38 / -0.17 / +0.35 | +0.90 / -0.19 / +0.26 | +1.04 / -0.14 / -9.41 | +2.09 / -0.00 / +1.90 | +0.96 / +0.96 / +4.37 |
| F4_random_patch|e1 | +4.97 / +0.12 / +6.24 | +2.62 / -0.49 / +0.49 | +3.15 / +0.17 / -59.26 | +7.49 / +0.08 / +8.93 | +5.49 / +0.02 / +6.36 |
| F4_random_patch|test_ood | +3.27 / -0.16 / +3.55 | +2.45 / -0.39 / +1.75 | +0.23 / -0.89 / -34.36 | +10.47 / +0.08 / +12.23 | +1.84 / -0.04 / +2.32 |
| F5_extreme_lighting|e1 | +4.13 / +0.59 / +7.61 | +0.61 / -0.82 / -2.65 | -0.21 / +0.39 / -78.56 | +4.81 / +0.54 / +8.39 | +5.18 / -0.01 / +6.55 |
| F5_extreme_lighting|test_ood | +2.99 / +0.02 / +3.95 | +2.59 / -0.04 / +3.16 | +2.25 / +0.02 / +0.19 | +2.87 / +0.03 / +3.67 | -0.69 / +0.27 / -2.93 |

### Worst primary cell per arm (primary v2)

| Arm | worst sFNMR (cell) | worst sFMR (cell) | min coverage |
|---|---|---|---|
| B0_raw_global | 18.96 (F4_random_patch|test_ood) | 4.83 (F1_mask|test_ood) | 90.5 |
| B1_quality_global | 12.78 (F4_random_patch|test_ood) | 5.51 (F1_mask|test_ood) | 95.7 |
| A3_quality | 7.55 (F1_mask|test_ood) | 2.59 (F1_mask|test_ood) | 86.0 |
| A4_original | 17.73 (F4_random_patch|test_ood) | 3.30 (F1_mask|test_ood) | 93.0 |
| C4_conservative | 10.53 (F4_random_patch|test_ood) | 2.57 (F1_mask|test_ood) | 89.5 |
| N4_novelty_aware | 9.70 (F4_random_patch|test_ood) | 2.57 (F1_mask|test_ood) | 87.7 |
| H4_quality_first_hybrid | 9.33 (F3_hand_object|e1) | 2.58 (F1_mask|test_ood) | 12.1 |
| A5_oracle | 10.32 (F2_sunglasses|e1) | 2.64 (F3_hand_object|e1) | 88.3 |

### Novelty detector

| Fold | ID pair flag rate (Test-ID) | held-out pair flag | E1 pair flag | AUROC held-out vs seen renders | AUROC E1 vs seen | AUROC held-out vs clean | AUROC E1 vs clean | clean pair flag |
|---|---|---|---|---|---|---|---|---|
| mask | 4.8 % | 1.1 % | 20.9 % | 0.619 | 0.913 | 0.991 | 1.000 | 0.0 % |
| sunglasses | 4.7 % | 5.1 % | 26.8 % | 0.766 | 0.917 | 0.995 | 0.999 | 0.1 % |
| hand object | 5.1 % | 13.2 % | 41.3 % | 0.683 | 0.920 | 0.985 | 0.999 | 0.0 % |
| random patch | 4.6 % | 46.5 % | 68.2 % | 0.851 | 0.965 | 0.989 | 1.000 | 0.0 % |
| extreme lighting | 5.2 % | 4.1 % | 87.8 % | 0.335 | 0.979 | 0.815 | 1.000 | 0.1 % |

### Visibility overestimation on p4 renders (signed global bias / share overestimated by >0.10)

| Fold | Cell | Family | Phase 3 VE bias | VE-C bias | Phase 3 VE share >0.10 | VE-C share >0.10 |
|---|---|---|---|---|---|---|
| F1 | e1 | E1_hair_strands | -0.051 | -0.033 | 0.0 % | 4.3 % |
| F1 | test_id | F2_sunglasses | -0.000 | +0.000 | 0.0 % | 0.0 % |
| F1 | test_id | F3_hand_object | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F1 | test_id | F4_random_patch | +0.001 | -0.001 | 0.1 % | 0.0 % |
| F1 | test_id | F5_extreme_lighting | +0.033 | +0.230 | 12.3 % | 43.3 % |
| F1 | test_ood | F1_mask | +0.051 | +0.020 | 14.8 % | 6.6 % |
| F1 | test_ood | F2_sunglasses | -0.000 | +0.000 | 0.0 % | 0.0 % |
| F1 | test_ood | F3_hand_object | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F1 | test_ood | F4_random_patch | +0.002 | -0.001 | 0.1 % | 0.0 % |
| F1 | test_ood | F5_extreme_lighting | +0.033 | +0.237 | 12.7 % | 43.8 % |
| F2 | e1 | E1_hair_strands | -0.078 | -0.051 | 0.0 % | 2.1 % |
| F2 | test_id | F1_mask | -0.001 | -0.001 | 0.0 % | 0.0 % |
| F2 | test_id | F3_hand_object | -0.002 | -0.002 | 0.0 % | 0.0 % |
| F2 | test_id | F4_random_patch | -0.001 | -0.002 | 0.2 % | 0.0 % |
| F2 | test_id | F5_extreme_lighting | +0.002 | +0.234 | 5.5 % | 43.1 % |
| F2 | test_ood | F1_mask | -0.001 | -0.001 | 0.0 % | 0.0 % |
| F2 | test_ood | F2_sunglasses | +0.003 | +0.003 | 0.6 % | 0.1 % |
| F2 | test_ood | F3_hand_object | -0.002 | -0.002 | 0.0 % | 0.0 % |
| F2 | test_ood | F4_random_patch | -0.001 | -0.002 | 0.3 % | 0.0 % |
| F2 | test_ood | F5_extreme_lighting | +0.001 | +0.223 | 5.9 % | 41.8 % |
| F3 | e1 | E1_hair_strands | -0.030 | -0.056 | 10.2 % | 11.7 % |
| F3 | test_id | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F3 | test_id | F2_sunglasses | -0.001 | -0.000 | 0.0 % | 0.0 % |
| F3 | test_id | F4_random_patch | +0.002 | -0.002 | 0.2 % | 0.0 % |
| F3 | test_id | F5_extreme_lighting | +0.022 | +0.234 | 8.8 % | 43.5 % |
| F3 | test_ood | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F3 | test_ood | F2_sunglasses | -0.001 | -0.001 | 0.0 % | 0.0 % |
| F3 | test_ood | F3_hand_object | +0.012 | -0.006 | 4.2 % | 0.5 % |
| F3 | test_ood | F4_random_patch | +0.002 | -0.002 | 0.1 % | 0.0 % |
| F3 | test_ood | F5_extreme_lighting | +0.022 | +0.233 | 8.9 % | 44.3 % |
| F4 | e1 | E1_hair_strands | +0.111 | -0.051 | 45.0 % | 7.4 % |
| F4 | test_id | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F4 | test_id | F2_sunglasses | -0.000 | -0.000 | 0.0 % | 0.0 % |
| F4 | test_id | F3_hand_object | -0.000 | -0.003 | 0.0 % | 0.0 % |
| F4 | test_id | F5_extreme_lighting | -0.000 | +0.224 | 2.9 % | 43.0 % |
| F4 | test_ood | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F4 | test_ood | F2_sunglasses | -0.000 | -0.000 | 0.0 % | 0.0 % |
| F4 | test_ood | F3_hand_object | -0.000 | -0.003 | 0.0 % | 0.0 % |
| F4 | test_ood | F4_random_patch | +0.342 | +0.065 | 94.6 % | 21.9 % |
| F4 | test_ood | F5_extreme_lighting | -0.001 | +0.219 | 2.2 % | 42.1 % |
| F5 | e1 | E1_hair_strands | +0.046 | +0.013 | 20.3 % | 10.8 % |
| F5 | test_id | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_id | F2_sunglasses | -0.000 | -0.000 | 0.0 % | 0.0 % |
| F5 | test_id | F3_hand_object | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_id | F4_random_patch | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_ood | F1_mask | -0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_ood | F2_sunglasses | -0.000 | -0.000 | 0.0 % | 0.0 % |
| F5 | test_ood | F3_hand_object | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_ood | F4_random_patch | +0.000 | -0.001 | 0.0 % | 0.0 % |
| F5 | test_ood | F5_extreme_lighting | +0.486 | +0.488 | 71.5 % | 71.4 % |

### Clean margin (FNMR at frozen clean-cal FAR-1 % thresholds vs M0)

| Fold | M0 FNMR | M1 FNMR (Δ CI hi, ok) | M4c FNMR (Δ CI hi, ok) | M5c FNMR (Δ CI hi, ok) | MLP FNMR (Δ CI hi, ok) | H4 FNMR (Δ CI hi, ok) |
|---|---|---|---|---|---|---|
| mask | 4.81 | 3.38 (-1.23, yes) | 3.52 (-1.10, yes) | 3.37 (-1.24, yes) | 2.95 (-1.64, yes) | 3.52 (-1.10, yes) |
| sunglasses | 4.81 | 3.37 (-1.24, yes) | 3.60 (-1.04, yes) | 3.14 (-1.47, yes) | 3.08 (-1.51, yes) | 3.60 (-1.04, yes) |
| hand object | 4.81 | 3.38 (-1.24, yes) | 3.60 (-1.04, yes) | 3.32 (-1.30, yes) | 3.07 (-1.52, yes) | 3.60 (-1.04, yes) |
| random patch | 4.81 | 3.41 (-1.20, yes) | 3.46 (-1.17, yes) | 3.23 (-1.38, yes) | 3.10 (-1.49, yes) | 3.46 (-1.17, yes) |
| extreme lighting | 4.81 | 3.37 (-1.25, yes) | 3.73 (-0.93, yes) | 3.47 (-1.17, yes) | 3.15 (-1.44, yes) | 3.71 (-0.95, yes) |

### Fusion and co-visibility ablations on occluded pairs (Δ log-loss, positive = second model better)

| Cell | M1 − M4c [CI] | M5_noCV − M5c [CI] | M5c − MLP [CI] | AUC M1 / M4c / M5c / MLP |
|---|---|---|---|---|
| F1_mask|e1 | -0.0093 [-0.0101, -0.0084] | -0.0086 [-0.0094, -0.0077] | +0.0037 [+0.0027, +0.0046] | 0.9894 / 0.9894 / 0.9911 / 0.9914 |
| F1_mask|test_id | +0.0027 [+0.0018, +0.0037] | +0.0027 [+0.0017, +0.0036] | +0.0060 [+0.0046, +0.0075] | 0.9858 / 0.9859 / 0.9876 / 0.9884 |
| F1_mask|test_ood | +0.0099 [+0.0088, +0.0110] | +0.0094 [+0.0083, +0.0105] | +0.0137 [+0.0119, +0.0155] | 0.9643 / 0.9659 / 0.9682 / 0.9692 |
| F2_sunglasses|e1 | -0.0116 [-0.0127, -0.0104] | -0.0099 [-0.0110, -0.0088] | +0.0070 [+0.0059, +0.0082] | 0.9894 / 0.9892 / 0.9909 / 0.9912 |
| F2_sunglasses|test_id | +0.0043 [+0.0029, +0.0057] | +0.0039 [+0.0025, +0.0053] | +0.0070 [+0.0053, +0.0086] | 0.9759 / 0.9763 / 0.9785 / 0.9801 |
| F2_sunglasses|test_ood | +0.0091 [+0.0078, +0.0104] | +0.0093 [+0.0080, +0.0105] | +0.0009 [-0.0002, +0.0020] | 0.9895 / 0.9905 / 0.9917 / 0.9923 |
| F3_hand_object|e1 | -0.0219 [-0.0237, -0.0202] | -0.0188 [-0.0205, -0.0171] | +0.0080 [+0.0066, +0.0094] | 0.9894 / 0.9888 / 0.9905 / 0.9913 |
| F3_hand_object|test_id | +0.0063 [+0.0046, +0.0080] | +0.0059 [+0.0042, +0.0076] | +0.0076 [+0.0057, +0.0095] | 0.9777 / 0.9785 / 0.9808 / 0.9825 |
| F3_hand_object|test_ood | -0.0094 [-0.0112, -0.0076] | -0.0089 [-0.0107, -0.0071] | +0.0033 [+0.0018, +0.0048] | 0.9845 / 0.9845 / 0.9865 / 0.9874 |
| F4_random_patch|e1 | -0.0269 [-0.0286, -0.0252] | -0.0226 [-0.0243, -0.0210] | +0.0115 [+0.0100, +0.0131] | 0.9893 / 0.9889 / 0.9906 / 0.9912 |
| F4_random_patch|test_id | +0.0062 [+0.0043, +0.0080] | +0.0052 [+0.0034, +0.0070] | +0.0055 [+0.0034, +0.0075] | 0.9832 / 0.9840 / 0.9857 / 0.9870 |
| F4_random_patch|test_ood | -0.0052 [-0.0074, -0.0030] | -0.0067 [-0.0088, -0.0046] | +0.0108 [+0.0083, +0.0134] | 0.9606 / 0.9602 / 0.9657 / 0.9668 |
| F5_extreme_lighting|e1 | -0.0391 [-0.0417, -0.0364] | -0.0348 [-0.0375, -0.0322] | +0.0141 [+0.0121, +0.0161] | 0.9894 / 0.9886 / 0.9901 / 0.9914 |
| F5_extreme_lighting|test_id | +0.0095 [+0.0074, +0.0117] | +0.0098 [+0.0076, +0.0120] | +0.0052 [+0.0032, +0.0072] | 0.9742 / 0.9756 / 0.9785 / 0.9798 |
| F5_extreme_lighting|test_ood | -0.0475 [-0.0517, -0.0434] | -0.0509 [-0.0552, -0.0466] | +0.0277 [+0.0246, +0.0307] | 0.9904 / 0.9896 / 0.9902 / 0.9916 |

## Follow-up on p3_test `db421cf1744d` (descriptive)

### test_ood: coverage / sFNMR / sFMR, % (primary|v2)

| Arm | mask | sunglasses | hand object | random patch | extreme lighting |
|---|---|---|---|---|---|
| B0_raw_global | 94.4 / 2.74 / 4.72 | 92.4 / 1.24 / 2.38 | 95.2 / 2.90 / 3.68 | 95.6 / 11.27 ✗ / 1.67 | 93.1 / 1.69 / 2.66 |
| B1_quality_global | 100.0 / 3.14 / 5.39 ✗ | 100.0 / 2.79 / 1.87 | 97.3 / 2.07 / 1.45 | 100.0 / 5.65 ✗ / 4.65 | 98.1 / 1.07 / 1.57 |
| A3_quality | 93.7 / 1.94 / 2.56 | 96.8 / 1.87 / 1.24 | 94.3 / 1.59 / 1.23 | 88.6 / 2.12 / 2.00 | 96.1 / 1.08 / 1.27 |
| A4_original | 94.8 / 1.75 / 3.14 | 96.8 / 1.28 / 0.89 | 95.4 / 2.36 / 1.22 | 98.5 / 8.52 ✗ / 2.13 | 98.7 / 2.25 / 1.28 |
| C4_conservative | 93.4 / 1.47 / 2.51 | 96.8 / 1.50 / 1.29 | 94.4 / 2.05 / 1.12 | 91.3 / 3.64 / 1.84 | 99.0 / 2.32 / 1.26 |
| N4_novelty_aware | 93.4 / 1.47 / 2.51 | 97.5 / 1.54 / 1.31 | 94.5 / 1.81 / 1.12 | 90.1 / 3.14 / 1.60 | 98.3 / 2.07 / 1.21 |
| H4_quality_first_hybrid | 92.6 / 1.40 / 2.53 | 92.0 / 1.49 / 1.30 | 84.2 / 1.89 / 1.16 | 52.2 / 2.16 / 1.03 | 95.0 / 1.81 / 1.27 |
| A5_oracle | 92.4 / 1.30 / 2.12 | 98.1 / 2.06 / 1.34 | 97.5 / 2.01 / 2.30 | 90.4 / 2.76 / 1.93 | 93.4 / 0.87 / 1.63 |

Endpoints (Test-OOD only, descriptive): recommendation `A3_quality`; violations {'A3_quality': 0, 'A4_original': 1, 'A5_oracle': 0, 'B0_raw_global': 1, 'B1_quality_global': 2, 'C4_conservative': 0, 'N4_novelty_aware': 0, 'H4_quality_first_hybrid': 0}

## Detector-inclusive, DEVELOPMENT (LFW View 1 DevTest): run `1c792ba98f5f`

max residual (p99 clean DevTrain) = 7.34; selection: `fallback`

| Family | no face (low/med/high) | low quality (l/m/h) | multiple faces (l/m/h) | abstain: cov / sFNMR / sFMR | non_match: cov / sFNMR / sFMR | fallback: cov / sFNMR / sFMR |
|---|---|---|---|---|---|---|
| mask | 0/0/0 | 1/4/5 | 42/39/40 | 96.9 / 1.15 / 0.00 | 98.0 / 2.27 / 0.00 | 98.0 / 1.13 / 0.00 |
| sunglasses | 0/0/0 | 2/3/4 | 42/42/43 | 97.8 / 0.91 / 0.00 | 99.1 / 2.24 / 0.00 | 99.1 / 0.90 / 0.00 |
| hand object | 0/0/6 | 2/3/9 | 43/43/40 | 95.3 / 1.17 / 0.00 | 98.0 / 3.85 / 0.00 | 97.6 / 1.14 / 0.00 |
| random patch | 0/3/62 | 2/10/35 | 43/41/27 | 84.4 / 4.21 / 0.00 | 96.4 / 16.13 / 0.00 | 92.4 / 4.09 / 0.00 |
| extreme lighting | 0/6/24 | 2/3/14 | 43/43/31 | 94.9 / 2.11 / 0.00 | 99.3 / 6.49 / 0.00 | 98.7 / 2.03 / 0.00 |

## Detector-inclusive, FOLLOW-UP (LFW View 2): run `d57db3ee8923`

max residual (p99 clean DevTrain) = 7.34; selection: `abstain`; dev-selected policy: `fallback`

| Family | no face (low/med/high) | low quality (l/m/h) | multiple faces (l/m/h) | abstain: cov / sFNMR / sFMR | non_match: cov / sFNMR / sFMR | fallback: cov / sFNMR / sFMR |
|---|---|---|---|---|---|---|
| mask | 0/1/3 | 0/2/3 | 49/49/49 | 98.0 / 0.23 / 0.00 | 99.1 / 1.35 / 0.00 | 98.9 / 0.22 / 0.00 |
| sunglasses | 0/0/0 | 3/3/3 | 47/48/48 | 98.7 / 0.68 / 0.00 | 100.0 / 2.00 / 0.00 | 100.0 / 0.67 / 0.00 |
| hand object | 0/0/11 | 1/2/8 | 50/48/45 | 96.2 / 0.00 / 0.00 | 99.1 / 2.91 / 0.00 | 98.7 / 0.23 / 0.00 |
| random patch | 0/4/52 | 0/12/43 | 50/44/31 | 84.0 / 2.91 / 0.00 | 95.6 / 14.65 / 0.00 | 91.3 / 3.16 / 0.00 |
| extreme lighting | 0/6/23 | 3/6/7 | 47/42/43 | 95.1 / 0.47 / 0.00 | 99.8 / 5.12 / 0.00 | 98.7 / 0.68 / 0.00 |

