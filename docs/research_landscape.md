# VEIL-X — Research Landscape

> Status: research & architecture phase. **No experiments have been run. No result in this
> repository is measured yet.**
> Literature verified via web search on 2026-10-26.
> **Phase 0 update (2026-10-26):** approved direction = C3 primary, C1 secondary (see
> `decision_log.md`). The bibliography is unchanged. Verification status changes are in §5.

## 0. Evidence labels used in all VEIL-X documents

| Label | Meaning |
|---|---|
| **[V]** | Verified prior research. Title, authors, year and an official URL or DOI were confirmed from a primary or index source (arXiv, CVF Open Access, PMLR, NeurIPS/ICLR proceedings, MDPI, IEEE, or the dataset owner's page). Claims attributed to these papers are paraphrased from their abstracts or official pages only. |
| **[V-partial]** | The paper exists and the title/URL is confirmed, but at least one bibliographic field (full author list, journal year, DOI) was **not** confirmed. The missing field is named. |
| **[U]** | Unverified. Recalled or inferred, but not confirmed during this phase. Must be checked before it is cited in any write-up. |
| **[H]** | A VEIL-X hypothesis. Untested. |
| **[D]** | A VEIL-X design choice: a decision, not a finding. |
| **[M]** | A measured VEIL-X result. **None exist yet.** |

The papers' quantitative results are deliberately **not** reproduced here, except where a number
appears in the abstract and matters for motivation. They must be re-read from the PDFs before
they are quoted.

---

## 1. Verified bibliography

### 1.1 Margin-based recognition backbones (the frozen feature extractors)

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P1 | J. Deng, J. Guo, N. Xue, S. Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition* | CVPR 2019 | DOI [10.1109/CVPR.2019.00482](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) | [V] |
| P2 | M. Kim, A. K. Jain, X. Liu. *AdaFace: Quality Adaptive Margin for Face Recognition* | CVPR 2022 | [arXiv:2204.00964](https://arxiv.org/abs/2204.00964), [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html) | [V] |
| P3 | Q. Meng, S. Zhao, Z. Huang, F. Zhou. *MagFace: A Universal Representation for Face Recognition and Quality Assessment* | CVPR 2021 | [arXiv:2103.06627](https://arxiv.org/abs/2103.06627) | [V] |
| P4 | J. Dan, Y. Liu, H. Xie, J. Deng, H. Xie, X. Xie, B. Sun. *TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective* | ICCV 2023 | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Dan_TransFace_Calibrating_Transformer_Training_for_Face_Recognition_from_a_Data-Centric_ICCV_2023_paper.html), [arXiv:2308.10133](https://arxiv.org/abs/2308.10133) | [V] |
| P5 | M. Kim, Y. Su, F. Liu, A. Jain, X. Liu. *KeyPoint Relative Position Encoding for Face Recognition* | CVPR 2024 | [arXiv:2403.14852](https://arxiv.org/abs/2403.14852), [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html) | [V] |

### 1.2 Occluded and masked face recognition

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P6 | L. Song, D. Gong, Z. Li, C. Liu, W. Liu. *Occlusion Robust Face Recognition Based on Mask Learning With Pairwise Differential Siamese Network* (PDSN) | ICCV 2019 | [arXiv:1908.06290](https://arxiv.org/abs/1908.06290), [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Song_Occlusion_Robust_Face_Recognition_Based_on_Mask_Learning_With_Pairwise_ICCV_2019_paper.html) | [V] |
| P7 | H. Qiu, D. Gong, Z. Li, W. Liu, D. Tao. *End2End Occluded Face Recognition by Masking Corrupted Features* (FROM) | IEEE TPAMI (arXiv 2021) | [arXiv:2108.09468](https://arxiv.org/abs/2108.09468), code [haibo-qiu/FROM](https://github.com/haibo-qiu/FROM) | [V-partial]: TPAMI volume, year and DOI not confirmed |
| P8 | Xu et al. *On Improving the Generalization of Face Recognition in the Presence of Occlusions* | CVPR Workshops 2020 | [arXiv:2006.06787](https://arxiv.org/abs/2006.06787), [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w48/Xu_On_Improving_the_Generalization_of_Face_Recognition_in_the_Presence_CVPRW_2020_paper.html) | [V-partial]: full author list not confirmed |
| P9 | H. Phan, A. Nguyen. *DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover's Distance Improves Out-Of-Distribution Face Identification* | CVPR 2022 | [arXiv:2112.04016](https://arxiv.org/abs/2112.04016) | [V] |
| P10 | P. C. Neto, F. Boutros, J. R. Pinto, N. Damer, A. F. Sequeira, J. S. Cardoso, et al. *OCFR 2022: Competition on Occluded Face Recognition From Synthetically Generated Structure-Aware Occlusions* | IJCB 2022 | [arXiv:2208.02760](https://arxiv.org/abs/2208.02760) | [V] |
| P11 | S. Ge, W. Guo, C. Li, J. Zhang, Y. Li, D. Zeng. *Masked Face Recognition with Generative-to-Discriminative Representations* | ICML 2024, PMLR 235:15242–15254 | [PMLR](https://proceedings.mlr.press/v235/ge24a.html), [arXiv:2405.16761](https://arxiv.org/abs/2405.16761) | [V] |
| P12 | K. Zhu, G. Ma. *Occlusion-Aware Face Recognition via Adaptive Local Feature Fusion and Identity-Guided Contrastive Learning* | Sensors 26(13), 2026 | DOI [10.3390/s26133977](https://www.mdpi.com/1424-8220/26/13/3977) | [V] |
| P13 | D. A. Abdullah. *PLGSA-Transformer: Periocular Landmark-Guided Attention with Occlusion-Adaptive Cosine Thresholding for Cross-Modal Masked and Unmasked Face Recognition* | arXiv preprint, Jul 2026 | [arXiv:2607.03581](https://arxiv.org/abs/2607.03581) | [V] (preprint, not peer reviewed) |

### 1.3 Occlusion and visibility estimation (non-recognition)

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P14 | X. Yin, L. Chen. *FaceOcc: A Diverse, High-quality Face Occlusion Dataset for Human Face Extraction* | arXiv 2022 | [arXiv:2201.08425](https://arxiv.org/abs/2201.08425) | [V] |
| P15 | Chiang et al. *ORFormer: Occlusion-Robust Transformer for Accurate Facial Landmark Detection* | WACV 2025 | [arXiv:2412.13174](https://arxiv.org/abs/2412.13174) | [V-partial]: full author list not confirmed |
| P16 | X. Xiang, Z. Li, S. Dhakad, T. Bancroft, J. Zhang, W. Li. *OccFace: Unified Occlusion-Aware Facial Landmark Detection with Per-Point Visibility* | arXiv preprint, Feb 2026 | [arXiv:2602.10728](https://arxiv.org/abs/2602.10728) | [V] (preprint) |

### 1.4 Face image quality and embedding uncertainty

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P17 | Y. Shi, A. K. Jain. *Probabilistic Face Embeddings* (PFE) | ICCV 2019 | [arXiv:1904.09658](https://arxiv.org/abs/1904.09658) | [V] |
| P18 | J. Chang, Z. Lan, C. Cheng, Y. Wei. *Data Uncertainty Learning in Face Recognition* (DUL) | CVPR 2020 | [arXiv:2003.11339](https://arxiv.org/abs/2003.11339) | [V] |
| P19 | P. Terhörst, J. N. Kolf, N. Damer, F. Kirchbuchner, A. Kuijper. *SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness* | CVPR 2020 | [arXiv:2003.09373](https://arxiv.org/abs/2003.09373) | [V] |
| P20 | F. Boutros, M. Fang, M. Klemt, B. Fu, N. Damer. *CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability* | CVPR 2023 | [arXiv:2112.06592](https://arxiv.org/abs/2112.06592), [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.html) | [V] |
| P21 | R. Kail, K. Fedyanin, N. Muravev, A. Zaytsev, M. Panov. *ScaleFace: Uncertainty-aware Deep Metric Learning* | arXiv 2022 | [arXiv:2209.01880](https://arxiv.org/abs/2209.01880) | [V] (venue not confirmed) |
| P22 | K. Ahn, S. Lee, S. Han, C. Y. Low, M. Cha. *Uncertainty-Aware Face Embedding With Contrastive Learning for Open-Set Evaluation* (UCFace) | IEEE TIFS, 2024 | [IBS repository record](https://pr.ibs.re.kr/handle/8788114/15465) | [V-partial]: DOI not confirmed |
| P23 | L. Erlygin, A. Zaytsev. *Holistic Uncertainty Estimation For Open-Set Recognition* (v1 title: *Gallery-Aware Uncertainty Estimation For Open-Set Face Recognition*) | arXiv 2024, revised 2025 | [arXiv:2408.14229](https://arxiv.org/abs/2408.14229) | [V] (preprint) |

### 1.5 Calibration, selective classification, risk control, evaluation statistics

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P24 | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks* | ICML 2017, PMLR 70 | [PMLR](https://proceedings.mlr.press/v70/guo17a.html), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | [V-partial]: title, venue and first author confirmed; co-author list given from memory, re-check on PMLR |
| P25 | Y. Geifman, R. El-Yaniv. *Selective Classification for Deep Neural Networks* | NeurIPS 2017 | [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | [V] |
| P26 | J. Traub, T. J. Bungert, C. T. Lüth, M. Baumgartner, K. H. Maier-Hein, L. Maier-Hein, P. F. Jäger. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems* (AUGRC) | NeurIPS 2024 | [arXiv:2407.01032](https://arxiv.org/abs/2407.01032) | [V] |
| P27 | A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control* | arXiv 2021 (v4 2022) | [arXiv:2110.01052](https://arxiv.org/abs/2110.01052) | [V] (journal version not confirmed) |
| P28 | Angelopoulos et al. *Conformal Risk Control* | arXiv 2022 (v4 2025) | [arXiv:2208.02814](https://arxiv.org/abs/2208.02814) | [V-partial]: author list not confirmed |
| P29 | T. Salvador, S. Cairns, V. Voleti, N. Marshall, A. Oberman. *FairCal: Fairness Calibration for Face Verification* | ICLR 2022 | [arXiv:2106.03761](https://arxiv.org/abs/2106.03761) | [V] |
| P30 | J.-R. Conti, S. Clémençon. *Assessing Uncertainty in Similarity Scoring: Performance & Fairness in Face Recognition* | ICLR 2024 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8d2c36836fb0e7d78fe68762ff8b5f1e-Abstract-Conference.html), [arXiv:2211.07245](https://arxiv.org/abs/2211.07245) | [V] |

### 1.6 Benchmarks and datasets (papers)

| # | Paper | Venue / year | Identifier | Status |
|---|---|---|---|---|
| P31 | M. E. Erakın, U. Demir, H. K. Ekenel. *On Recognizing Occluded Faces in the Wild* (ROF dataset) | BIOSIG 2021 | [arXiv:2109.03672](https://arxiv.org/abs/2109.03672), [repo](https://github.com/ekremerakin/RealWorldOccludedFaces) | [V] |
| P32 | C. Wang, H. Fang, Y. Zhong, W. Deng. *MLFW: A Database for Face Recognition on Masked Faces* | arXiv 2021 | [arXiv:2109.05804](https://arxiv.org/abs/2109.05804) | [V] (peer-reviewed venue not confirmed) |
| P33 | M. Knoche, S. Hörmann, G. Rigoll. *Cross-Quality LFW: A Database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments* (XQLFW) | IEEE FG 2021 | [arXiv:2108.10290](https://arxiv.org/abs/2108.10290), [project](https://martlgap.github.io/xqlfw) | [V] |
| P34 | Bae et al. *DigiFace-1M: 1 Million Digital Face Images for Face Recognition* | WACV 2023 | [arXiv:2210.02579](https://arxiv.org/abs/2210.02579), [repo](https://github.com/microsoft/DigiFace1M) | [V-partial]: full author list not confirmed |

**Coverage check:** there are 34 entries. 13 were first published in 2023–2026: P4, P5, P11, P12,
P13, P15, P16, P20, P22, P23, P26, P30 and P34. This meets the brief's requirement of at least 10
papers with a 2023–2026 emphasis.

### 1.7 References searched for but not verified, and therefore not cited

- A face-verification paper that applies distribution-free risk control (LTT or conformal) and
  evaluates it under **held-out occlusion types**. **None was found.** That absence is the main
  driver of the novelty audit, but a missing search hit is not evidence of novelty (see
  `novelty_audit.md` §4).
- MaskTheFace (synthetic mask tool), CASIA-WebFace's current distribution status, the OFIQ quality
  library, and ISO/IEC 29794-5. All of these are [U] and must be checked before use.

---

## 2. Thematic synthesis (what the verified work establishes)

### 2.1 Pretrained margin-loss embeddings are the strong baseline [V]
ArcFace (P1) and AdaFace (P2) define the standard angular-margin embedding. AdaFace explicitly
uses the **feature norm as an image-quality proxy** during training. MagFace (P3) makes embedding
magnitude a quality indicator by design. **Implication [D]:** any VEIL-X abstention mechanism
must beat *feature-norm-based rejection from the same backbone*. Otherwise the "uncertainty"
component adds nothing. That comparison is mandatory baseline B2 in `evaluation_protocol.md`.

### 2.2 Occlusion handling: feature masking and regional reliability are established [V]
- PDSN (P6) learns feature-discarding masks from occluded/clean pair differences.
- FROM (P7) learns masks end to end to suppress corrupted features. The authors report an
  evaluation protocol that trains on some occluder groups and tests on unseen groups (per the
  search summary of the paper; re-read before citing details).
- Zhu & Ma (P12) estimate per-region reliability with an MLP over three face sub-regions and
  fuse local features. Their abstract states the open problem that salient occluders can receive
  high responses without carrying identity information.
- PLGSA-Transformer (P13, preprint, 858 images) jointly predicts occlusion severity and **raises
  the matching threshold in proportion to the predicted severity**. This is the closest prior
  art to "visibility-conditioned decision rules" that we found.

**Implication:** "regional reliability modelling" and "occlusion-adaptive threshold" are **not
novel**. VEIL-X may use them, but must present them as adopted techniques.

### 2.3 Patch-level matching helps out-of-distribution queries without retraining [V]
DeepFace-EMD (P9) re-ranks with an Earth Mover's Distance over the spatial features of a frozen
ArcFace and reports gains on masked, cropped and rotated queries in *identification*.
**Implication [H]:** frozen spatial feature maps carry usable local identity evidence. This
supports VEIL-X's plan to compute co-visibility-weighted local similarities without fine-tuning
the backbone.

### 2.4 Quality and uncertainty estimation is mature for images, less so for pairs under shift [V]
PFE (P17) trains an uncertainty head on a **frozen** deterministic embedding. DUL (P18) learns the
mean and variance jointly. SER-FIQ (P19) and CR-FIQA (P20) estimate sample quality. ScaleFace
(P21), UCFace (P22) and HolUE (P23) target uncertainty in metric learning or open-set settings.
**Gap observed [H]:** most of these methods score an *image* (or image vs gallery). VEIL-X's
decisions are about *pairs*, and the two images in a pair can be occluded in **different**
regions. If, say, one face is masked and the other wears sunglasses, the region visible in
*both* images (the co-visible region) can be small even when each image has moderate quality.
That is a pair-level quantity that per-image quality cannot express.

### 2.5 Selective prediction and risk control are well-founded in general ML [V]
Geifman & El-Yaniv (P25) build selective classifiers with a risk guarantee. Traub et al. (P26)
show that common selective-classification metrics (e.g. AURC) are flawed and propose AUGRC. LTT
(P27) and Conformal Risk Control (P28) provide finite-sample risk control under exchangeability.
Temperature scaling (P24) is the standard post-hoc calibration. FairCal (P29) calibrates face
verification scores per cluster. Conti & Clémençon (P30) show that a naive bootstrap of ROC
quantities from similarity scores is invalid (they are U-statistics) and requires recentering.
**Implication [D]:** VEIL-X reports AUGRC, and uses identity-aware resampling for its
uncertainty intervals.

### 2.6 Benchmarks: real occlusion data is small; synthetic occlusion is optimistic [V]
ROF (P31) reports that performance drops much more on **real** occlusions than on synthetically
occluded faces. MLFW (P32) reports a 5–16% accuracy decline for SOTA models on masked CALFW
(abstract). OCFR-2022 (P10) uses synthetic, structure-aware occlusions. XQLFW (P33) stresses
cross-quality. **Implication [D]:** a model calibrated on synthetic occlusions must be tested on
real occlusions (ROF) before any robustness claim.

---

## 3. Comparison matrix

Legend: FT = fine-tunes backbone; Frozen = uses pretrained embeddings without retraining.
"Abstain?" asks whether the method outputs an explicit reject decision.

| Method | Problem | Backbone use | Training data (per paper) | Architectural contribution | Evaluation protocol | Abstain? | Weakness relative to VEIL-X RQ |
|---|---|---|---|---|---|---|---|
| ArcFace (P1) | General FR | Trains backbone | Large web face sets | Additive angular margin | Ten FR benchmarks (abstract) | No | No occlusion or uncertainty modelling |
| AdaFace (P2) | Low-quality FR | Trains backbone | MS1MV2 / WebFace4M | Norm-adaptive margin | IJB-B/C, IJB-S, TinyFace | No (norm usable as quality) | Quality is per-image; no visibility |
| MagFace (P3) | FR + quality | Trains backbone | Per paper (not re-read) | Magnitude = quality | FR, quality assessment, clustering (abstract) | No | Per-image; not pair-level co-visibility |
| PDSN (P6) | Occluded FR | FT + mask dictionary | Synthetic occlusion pairs | Feature-discarding masks per grid cell | Synthetic + real occlusion sets | No | Needs training with occluded pairs; no calibrated reject |
| FROM (P7) | Occluded FR | End-to-end | Synthetic occlusion | Learned feature masks | LFW, MegaFace, RMF2, AR; unseen-occluder split reported | No | No uncertainty or abstention; backbone retraining |
| Zhu & Ma (P12) | Occluded FR | Lightweight custom | Per paper (not re-read) | 3-region reliability MLP + fusion + contrastive loss | Per paper (not re-read) | No | Region reliability without decision calibration |
| PLGSA (P13) | Masked vs unmasked FR | EfficientNetB3 + MHSA | 858 images (MDMFR, Kaggle CelebA-HQ masked, self-collected) | Periocular landmark attention; occlusion-severity-adaptive threshold | Pair accuracy, ROC-AUC | Threshold shift only, no abstention | Tiny data; masks only; no OOD occluders; no risk control |
| DeepFace-EMD (P9) | OOD identification | Frozen ArcFace | No retraining | Patch-wise EMD re-ranking | Masked, cropped, rotated, adversarial queries | No | Identification, not calibrated verification; cost of EMD |
| G2D (P11) | Masked FR | Generative + discriminative | Per paper | Generative-to-discriminative representation | Masked FR benchmarks | No | Masks only; heavy training |
| KP-RPE (P5) | Alignment-robust ViT FR | Trains ViT | WebFace-scale | Keypoint-relative positional encoding | Low-quality and misaligned FR | No | Not occlusion-explicit; heavy training |
| TransFace (P4) | ViT FR training | Trains ViT | Large-scale | Patch-level augmentation (DPAP), entropy hard-sample mining (EHSM) | Standard FR | No | Not occlusion- or uncertainty-explicit |
| PFE (P17) | Uncertainty | **Frozen** + uncertainty head | Per paper (not re-read) | Gaussian embedding over a frozen deterministic embedding | Per paper (not re-read) | No explicit | Per-image variance; behaviour under occlusion shift untested in our search |
| DUL (P18) | Uncertainty | Trains | Per paper (not re-read) | Joint mean/variance learning | Per paper (not re-read) | No | Retraining needed |
| SER-FIQ (P19) | Quality | Any model with dropout | None (unsupervised) | Stochastic embedding robustness | Error-vs-reject curves | Reject by quality | Needs dropout at inference; per-image |
| CR-FIQA (P20) | Quality | Trains quality branch | Per paper (not re-read) | Learns relative classifiability | Error-vs-reject curves on 8 benchmarks | Reject by quality | Per-image; not pair-level |
| ScaleFace / HolUE (P21, P23) | Uncertainty, open-set | Trains or Bayesian post-hoc | IJB-C etc. | Learned scale; gallery + embedding uncertainty | Reject-based evaluation | Implicitly | Not occlusion-specific; IJB-C access restricted |
| UCFace (P22) | Uncertainty, open-set | Trains | Per paper | Quality-weighted supervised contrastive | Open-set | No | Not re-read; same per-image limitation |
| FairCal (P29) | Calibration | Frozen | — | Cluster-conditional calibration | FMR gaps across groups | No | Fairness-focused; not occlusion shift |
| LTT / CRC (P27, P28) | Risk control | Model-agnostic | — | Finite-sample risk guarantee | General ML tasks | Yes (via thresholds) | Guarantee assumes exchangeability, which unseen occlusion breaks |
| AUGRC (P26) | Selective-classification eval | — | — | Metric | 6 datasets, 13 confidence scores | — | Classification framing; must be adapted to verification pairs |

---

## 4. Open questions remaining after the review

All of these are [H] or gaps. None is a claim.

1. **Pair-level evidence sufficiency.** Does a pair-level co-visibility signal predict
   verification errors *beyond* per-image quality (feature norm, CR-FIQA) and the score margin?
   We found no verified study of this under held-out occlusion families.
2. **Risk-control validity under occlusion shift.** LTT/CRC guarantees hold under
   exchangeability. How badly do false-match-rate guarantees calibrated on seen occlusion types
   fail on unseen types? Does conditioning the calibration on visibility strata (Mondrian-style)
   reduce the violation?
3. **Synthetic-to-real transfer of visibility estimation.** ROF indicates that synthetic
   occlusion overstates robustness. Does a segmenter trained only on synthetic occluders give
   useful visibility on real masks and sunglasses?
4. **Frozen local features.** Are the backbone's spatial features locally identity-discriminative
   enough for visibility-weighted region similarity? DeepFace-EMD suggests yes for
   identification; this is untested for calibrated verification.
5. **Illumination as occlusion.** Can extreme shadow or saturation be treated as "low visibility"
   by the same mechanism, or does it need a separate quality signal?
6. **Pretraining overlap.** Public pretrained backbones were trained on web-celebrity data that
   may overlap with the identities in LFW-derived test sets. This contaminates absolute numbers
   for every method that uses those backbones. How much does it bias *relative* comparisons?

---

## 5. Source verification status after Phase 0 (2026-10-26)

| Item | Status change | Evidence |
|---|---|---|
| P34 DigiFace-1M | Licence now confirmed as **R-UDA v1.0** (non-commercial research). Part structure and sizes confirmed. Full author list still [V-partial] | LICENSE file, README, HTTP HEAD |
| P31 ROF | Repo licence confirmed MIT. Collection method (Google Image Search, VGGFace2-like process) confirmed from README. Image copyright is **not** covered by the MIT licence | README / LICENSE |
| P33 XQLFW | Dataset composition confirmed (3,743 IDs, 7,263 images, 6,000 pairs). Licence not stated on project page | Project page |
| CPLFW (not in the bibliography) | Protocol confirmed (3,000 positive pairs + same-gender/race negatives, 10 folds). No licence text found | Project page over HTTP |
| P32 MLFW | Project page not machine-readable. Licence still [U] | — |
| P27 LTT | Applicability of the binomial p-value to the *ratio* risk "error among accepted" is **to be re-checked** against the paper's selective-classification section before Phase 2 [U] | `protocol_design.md` §2 |
| P30 Conti & Clémençon | Its implication (naive bootstrap of ROC quantities can be invalid) is now a hard requirement: bootstrap coverage is validated by simulation before use | `protocol_design.md` §9 |

Still open before any write-up: the full author lists of P8, P15, P28 and P34, the DOIs of P7
and P22, the co-authors of P24, and the novelty-check literature search
(`novelty_audit.md` §4).
