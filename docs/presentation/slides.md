# VEIL-X: Final Presentation (slide script)

About 20 minutes plus a 5-minute live demo. Each slide gives the **on-screen content** and
*speaker notes*. Every number is in `docs/phase6_audit.md` §4 (claim matrix) or
`docs/paper/veilx_paper.md`. Use only the figures listed in the paper. Show **no** face image
other than synthetic DigiFace examples.

---

### 1. Title
**Selective Face Verification Under Unseen Occlusions and Population Shift**
Quality, learned visibility and abstention. VEIL-X, non-commercial academic research.

*Notes.* Research on decision-level risk control, not a product. The models are frozen; we did
not train recognition models.

### 2. Problem and motivation
- **The task:** 1:1 verification (two images, same person?) with occlusion: masks, glasses,
  hands, hair, objects.
- **The effect:** occlusion removes evidence, so genuine scores drop and false non-matches rise.
- **The goal:** a system that **abstains** when evidence is weak, and keeps errors on accepted
  decisions ≤ 5 %.

*Notes.* At a fixed clean threshold, FNMR rose from 1.4 % to 6.8–21.6 % under synthetic occlusion
(Phase 2).

### 3. Why conventional verification fails under occlusion
- A clean threshold is calibrated on clean pairs. With occlusion, genuine scores shift down, so
  misses go up (up to 44.9 % at high random-patch severity).
- Clean-only calibration violates a 5 % miss target on every held-out family (6–20 %).
- Calibration must see occlusion, but it can never see *every* occluder.

### 4. Hypotheses
- **H-C3:** stratifying calibration by **learned visibility** controls risk better than
  stratifying by quality.
- **H-C1:** co-visibility adds information beyond similarity and quality.
- **Phase 4:** a **conservative / novelty-aware** visibility estimate is safe relative to quality.
- **Population (Phases 4 and 6):** do guarantees survive a different face population?

*Notes.* All pre-registered before data access. One logged access per test partition.

### 5. Baselines: AdaFace and ArcFace (frozen)
- IR-101 backbones trained on WebFace4M (pretrained, not ours).
- **Clean DigiFace:** AdaFace AUC 0.9986, EER 1.28 %; ArcFace AUC 0.9979.
- **LFW official protocol:** 99.58 % (likely optimistic because of training-data overlap).
- **Decisions:** AdaFace is the decision backbone (score M1 from its similarity and feature-norm
  quality). ArcFace is a diagnostic only.

### 6. Selective decisions and Learn-then-Test
- **Rule:** MATCH if p ≥ τ_high, NON-MATCH if p < τ_low, else ABSTAIN.
- **LTT:** certifies sFMR ≤ 5 % and sFNMR ≤ 5 % with 95 % confidence, but only if calibration
  and deployment pairs are **exchangeable**.
- **Stratification:** a separate rule per stratum (quality tertiles, or visibility strata).

### 7. Learned visibility architecture
- U-Net-style, 3.6 M parameters; 28×28 visibility map; one model per left-out occluder family.
- Pair co-visibility features feed the fused score and the strata.
- **Phase 4 VE-C:**
  - asymmetric loss;
  - natural-texture negatives;
  - temperature and lower-bound shift;
  - Mahalanobis novelty.

### 8. Phase 3 result: negative
| Held-out random patch | Coverage | sFNMR |
|---|---|---|
| Quality strata (A3) | 88.6 % | 2.12 % ✓ |
| Learned visibility (A4) | 98.5 % | **8.52 % ✗** |
| Ground-truth visibility (oracle) | 90.4 % | 2.76 % ✓ |

- **Why:** the estimator called unseen occluders "visible" (IoU 0.13, bias +0.34). Pairs moved
  into a loose stratum.
- **Mechanism:** it learned the occluder *classes* it was shown, not "non-face". Errors are
  confident, and face-like occluders are undetectable.

### 9. Phase 4: conservative visibility experiments
- VE-C **fixed** the overestimation (94.6 % → 21.9 % of renders overestimated by > 0.10).
- But on 10 primary cells (held-out families plus a never-seen hair occluder):
  - violations: A3 6, C4 10, N4 9, H4 9;
  - C4 was significantly worse than A3 in 8 cells.
- Even **ground-truth visibility** was worse than A3 on the never-seen occluder.
- **Registered outcome:** recommend **A3**.

### 10. The population-shift discovery
- p4_test held 72-image DigiFace identities only (untouched).
- The calibration set was 98 % 5-image identities.
- Genuine pairs were about 4× harder: clean FNMR 4.81 % vs 1.12 %.
- **Every rule, A3 included, violated the 5 % miss target even on seen occluders** (A3
  5.36–6.34 %).
- The false-match side held (≤ 2.6 %).
- **Phase 6 pre-registered replication** on 4,000 untouched identities:
  - replicated: 4/5 seen-family cells violate (5.15–6.49 %);
  - A3 is again the only safe choice (violations A3 7 vs C4 10, N4 10, H4 9);
  - sFMR within target in 15/15 cells.
- Figure: `reports/figures/phase6_a3_population_shift.svg`.

*Notes.* This is the central message: distribution-free guarantees are guarantees about the
calibration population.

### 11. A failure the aggregates hid (Phase 6 audit)
- Below the calibration support of quality (about 1 % of pairs), the frozen score extrapolates.
- Two random-noise images give **MATCH, p = 1.000**.
- Development data: A3 false-match rate 18–42 % in that tail; on the untouched p6 data, 22.97 %
  vs 1.50 % above the floor (exploratory).
- Figures: `phase6_m1_extrapolation.svg`, `phase6_lowq_tail.svg`.
- **Response:** reported, frozen system unchanged; the app abstains below the floor.

### 12. Final selection: A3, and what it is not
- A3 (quality-stratified M1, AdaFace) is the **best-supported tested method**.
- It is **not** universally safe, **not** certified, and **not** deployment-ready.
- A valid deployment needs calibration on deployment-population data with the deployed
  preprocessing.

### 13. Working local application
- Runs on loopback only; strict CSP; no telemetry; uploads deleted after reading; no stored
  embeddings.
- Parity with the offline frozen system: 1000/1000 decisions; LFW crops pixel-identical.
- It shows:
  - the active fold system and its calibration;
  - the abstention reason;
  - detector status;
  - visibility and novelty, explanatory only;
  - its limitations.
- Safety:
  - multiple faces are refused;
  - the fixed-crop fallback is used only for declared LFW-style portraits;
  - it abstains below the support floor.

### 14. Limitations
- Synthetic occluders only; no physical-occlusion evaluation (IJB-C access is future work).
- Synthetic confirmatory faces. LFW identities overlap backbone training data.
- Five fold-specific calibrations; no universal one.
- No out-of-distribution or population-shift guarantee.
- One backbone pair and one detector.

### 15. Future research
1. Calibration on the deployment population, with pre-registered evaluation.
2. Physical occlusions: IJB-C occlusion-annotated 1:1, frozen systems.
3. Monotone or support-aware fused scores, so they cannot extrapolate below quality support.
4. Shift-aware risk control: detecting population shift before trusting a certificate.

### 16. Summary
- Quality beats learned visibility for risk control under unseen occlusion.
- Guarantees are population-specific; we measured where they break.
- The demonstration abstains honestly.
