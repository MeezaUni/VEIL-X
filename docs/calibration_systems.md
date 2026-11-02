# The Five Frozen A3 Calibration Systems (Phase 6, 6C)

VEIL-X has **no universal calibration system**. Phase 3 froze five quality-stratified A3 systems
(freeze `34409779ae8f…`), one per leave-one-family-out (LOFO) fold. They exist to *test
generalisation to an unseen occluder*, not to be deployed. The local app uses one of them at a
time, and shows which one is active.

## 1. What every system shares

| Element | Value |
|---|---|
| **Score** | M1 = logistic(s, q, s·q):<br>- s = AdaFace IR-101 cosine similarity;<br>- q = min of the two AdaFace pre-normalisation feature norms.<br>ArcFace is not an input (D-052) |
| **Strata** | Tertiles of q, with edges fixed on `p3_val` |
| **Rule** | Per stratum, MATCH if p ≥ τ_high, NON-MATCH if p < τ_low, ABSTAIN otherwise. In q_mid and q_high, τ_low = τ_high (no abstention band) |
| **Calibration method** | Learn-then-Test with a fixed-sequence family:<br>- target sFNMR ≤ 5 % and sFMR ≤ 5 %, δ = 0.05;<br>- candidate order v2 from `p3_val` p-values (D-042) |
| **Calibration data** | `p3_cal_independent`: **6,146 identity-disjoint pairs (3,073 genuine / 3,073 impostor)** over **9,219 synthetic DigiFace-1M identities**. Of these, **9,033 are 5-image identities and only 186 are 72-image identities**. Train-split identities, hash band [60, 75) (`docs/phase3_protocol.md` §2) |
| **Occlusion in calibration** | The fold's cal plan: clean pairs plus synthetic occlusions from the **four seen families**. The fold's own family is never rendered for calibration (enforced by the plan-isolation guard) |
| **Faces and occluders** | All synthetic |

## 2. The five systems

Cells are A3 coverage / sFNMR / sFMR in %:
- ✗ means the 95 % identity-jackknife CI excludes the 5 % target;
- † means the point estimate is above the target.

| System | Held-out family | q edges | q_low band [τ_low, τ_high) | q_mid τ | q_high τ | Phase 3 confirmatory, held-out family (`bb5d9244bb3c`) | Phase 4, seen families (`903f6a5c7de3`) | Phase 4, held-out family | Phase 4, never-seen E1 |
|---|---|---|---|---|---|---|---|---|---|
| **F1_mask** (app default) | mask | 19.895 / 21.966 | [0.154, 0.833) | 0.452 | 0.673 | 93.7 / 1.94 / 2.56 | 94.0 / 5.89 ✗ / 1.27 | 90.9 / 7.55 ✗ / 2.59 | 93.2 / 5.76 ✗ / 1.11 |
| F2_sunglasses | sunglasses | 19.069 / 21.564 | [0.081, 0.885) | 0.448 | 0.448 | 96.8 / 1.87 / 1.24 | 89.7 / 6.34 ✗ / 1.67 | 95.7 / 5.72 ✗ / 1.27 | 91.1 / 5.46 ✗ / 1.08 |
| F3_hand_object | hand / object | 19.370 / 21.685 | [0.070, 0.879) | 0.452 | 0.634 | 94.3 / 1.59 / 1.23 | 90.2 / 6.16 ✗ / 1.59 | 92.4 / 5.84 ✗ / 1.24 | 90.6 / 5.09 † / 1.03 |
| F4_random_patch | random patch | 19.824 / 21.866 | [0.072, 0.854) | 0.426 | 0.426 | 88.6 / 2.12 / 2.00 | 93.2 / 5.36 ✗ / 1.63 | 86.0 / 7.26 ✗ / 2.04 | 91.1 / 4.65 / 1.15 |
| F5_extreme_lighting | extreme lighting | 19.126 / 21.533 | [0.060, 0.862) | 0.421 | 0.595 | 96.1 / 1.08 / 1.27 | 89.4 / 6.05 ✗ / 2.00 | 94.7 / 4.33 / 1.47 | 90.7 / 4.65 / 1.26 |

**Phase 6 pre-registered replication** (`1c291322aa27`; 4,000 untouched 72-image identities),
A3 on seen families (Test-ID):

| System | Coverage / sFNMR / sFMR, % |
|---|---|
| F1_mask | 93.8 / 5.90 ✗ / 1.40 |
| F2_sunglasses | 89.8 / 6.49 ✗ / 1.73 |
| F3_hand_object | 90.0 / 6.41 ✗ / 1.77 |
| F4_random_patch | 93.0 / 5.15 † / 1.75 |
| F5_extreme_lighting | 89.6 / 6.39 ✗ / 2.02 |

The population-shift failure replicated (`docs/phase6_results.md`).

## 3. What the calibration population represents, and why no system is universal

1. **The calibration population is synthetic and mostly 5-image identities.**
   - About 98 % of the calibration identities are DigiFace 5-image identities.
   - The Phase 4 partition (72-image identities only) had about 4× harder genuine pairs: clean
     FNMR of raw similarity 4.81 % vs 1.12 %.
   - **On that population every A3 system violated the sFNMR target even on seen families**,
     and 4/5 did so again on 4,000 new identities (Phase 6). The guarantee is specific to the
     calibration population.
2. **Each system saw only four occluder families.** A system is tested on its fifth family as
   an unseen occluder. A deployment would meet occluders that none of the five families
   represent, such as real scarves, hair, glasses or physical objects.
3. **No system saw real photographs, real cameras or other demographics.** On LFW (real
   photographs with synthetic overlays), DigiFace-calibrated rules were conservative
   (sFMR ≤ 0.2 %). That is a different behaviour from the one they were calibrated for, not
   evidence of validity.
4. **Preprocessing is part of the calibration.** The systems were calibrated on DigiFace crops
   that are pre-aligned to the ArcFace template. Detector-based alignment and the fixed-crop
   fallback produce other input distributions.
5. **Selecting among the five on test results would be post hoc.** The app default **F1_mask**
   is the first registered fold, chosen independently of any result (D-051). It is *not* the
   best fold, and it is *not* a deployment calibration.

## 4. What would be needed for a deployment calibration (not done)

- **Protocol first:** an independently specified development protocol.
- **Data:** identity-disjoint calibration pairs from the target population and capture
  conditions, preprocessed with the deployed pipeline.
- **Evaluation:** a pre-registered evaluation on untouched data from the same population.
- **Decision:** a separate, explicit decision.

VEIL-X does **not** create an all-family or deployment system (Phase 6 stop conditions).
