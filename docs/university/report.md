---
title: "VEIL-X: An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"
subtitle: University Course Project Report
---

<!-- TITLEPAGE -->

# Abstract

**What VEIL-X is.** VEIL-X is a local AI application that compares two face images and reports
whether they are likely to show the same person.

**How it works.** It combines:
- face detection and alignment;
- pretrained deep-learning face embeddings (AdaFace IR-101);
- cosine similarity;
- an image-quality signal taken from the embedding network itself;
- calibrated decision thresholds.

**Three outcomes.** Unlike a basic binary face matcher, VEIL-X can return three outcomes:
- MATCH;
- NON-MATCH;
- **ABSTAIN**, when the evidence is not strong enough for a reliable decision.

**Explanations.** For every comparison the application explains itself. A step-by-step "Why
this result?" panel states, in plain language and with the exact numbers, how similar the faces
are compared with typical same-person and different-person pairs, how good the images are, the
exact threshold comparison that decided, and what would change the result. The application also
shows predicted-visibility heat maps that highlight where a face may be covered. The heat maps
are explanatory only and never change the decision.

**Engineering results:**
- The application reproduces the offline research pipeline exactly (1,000 of 1,000 decisions
  identical on a development set).
- CPU and GPU decisions agree (100 of 100 pairs).
- A warm comparison of two photographs takes about 0.23 s on an RTX 4060 Laptop GPU.
- An automated suite of over 400 tests passes.

**Privacy.** Everything runs on the user's own computer with no network services.

**Limitations.** The project also documents its limits honestly:
- the calibration was built on synthetic faces and synthetic occlusions;
- its 5 % error target did not hold for a harder population of identities;
- very low-quality inputs required an extra application-level safeguard.

VEIL-X is an academic demonstration, not a production identity-verification system.

**Keywords:** face verification, deep learning, face embeddings, image quality, abstention,
selective prediction, explainability, Gradio.

# 1. Introduction

**Face verification.** Face verification answers a simple question: *do these two images show
the same person?* It is used for phone unlocking, airport e-gates and photo organisation. Modern
systems use deep convolutional neural networks (CNNs) that turn a face image into a numeric
*embedding*, a vector of 512 numbers. Images of the same person produce embeddings that point in
a similar direction.

**The problem with yes/no answers.** Most systems always answer "same" or "different", even when
the image is blurred, badly lit or partly covered by a mask, sunglasses, a hand or another
object. In those cases a forced answer can simply be wrong.

**The VEIL-X approach.** VEIL-X allows the system to say *"I am not confident enough to decide"*.
It also explains why.

**What this project contributes:**
- it builds the system as a working local application;
- it tests it carefully;
- it documents where it works and where it does not.

# 2. Problem Statement

Face verification accuracy drops sharply when faces are occluded or image quality is low. In this
project's own baseline measurements:
- with synthetic occluders, the false non-match rate of a strong pretrained model rose from about
  1.4 % on clean images to between 6.8 % and 21.6 %;
- for the hardest occluder type at high severity it reached 44.9 %.

A system that must always answer will make many such errors silently. The problem addressed is:

> Build a local face verification application that (1) uses strong pretrained models,
> (2) measures image quality, (3) abstains when evidence is insufficient instead of forcing a
> decision, (4) explains its outputs, and (5) protects privacy by running entirely on the user's
> computer, while honestly measuring and reporting how reliable it is.

# 3. Objectives

1. Implement a two-image face verification pipeline using pretrained deep-learning models.
2. Detect and align faces automatically, and refuse ambiguous inputs, such as several faces with
   no selection.
3. Measure image quality and use it in the decision.
4. Produce three outcomes, MATCH, NON-MATCH and ABSTAIN, using thresholds calibrated on held-out
   data.
5. Visualise predicted face visibility, to help users understand occlusions.
6. Build an easy-to-use local web interface with clear explanations.
7. Keep processing local and private: no cloud services, no stored images.
8. Test the system thoroughly, measure its speed, and document its limitations.

# 4. Existing Systems

| Existing approach | Limitation |
|---|---|
| Basic face matchers (one similarity threshold) | Always answer yes or no; no notion of confidence; errors on poor images are silent |
| Commercial / cloud face APIs | Images leave the user's device; behaviour is not transparent; not suitable for a course project's privacy goals |
| Occlusion-robust recognition models (e.g. mask-trained networks) | Require retraining large networks; still give forced answers; often specialised to one occluder type |
| Image-quality filters used separately | Quality is checked but not integrated into the decision threshold |

The pretrained models used here, AdaFace (Kim et al., 2022) and ArcFace (Deng et al., 2019), are
state-of-the-art face recognisers. VEIL-X does not invent a new recognition network. Its work is
in the decision layer, the abstention mechanism, the explanations and the engineering of a
complete, tested application.

# 5. Proposed System

The proposed system has six parts:
- **Face preparation.** Each image is checked for validity. One face is then detected and aligned
  to a standard 112×112 crop with the SCRFD detector (Guo et al., 2022).
  - If an image contains several faces, the user must choose one.
  - If no reliable face is found, the image is refused. A fixed-crop fallback is applied only
    when the user declares a loosely cropped portrait.
- **Embedding.** AdaFace IR-101 produces an embedding and a *feature norm* for each face. The
  feature norm acts as a quality score: blurred or covered faces give smaller norms.
- **Fused score.** A small logistic-regression model (called M1) combines three features into one
  score between 0 and 1:
  - the cosine similarity *s*;
  - the pair quality *q* (the smaller of the two norms);
  - their product *s·q*.
- **Quality-stratified decision (A3).** Pairs are divided into low, medium and high quality.
  - Each quality group has its own calibrated thresholds.
  - In the low-quality group there is an *abstention band*: scores inside it give ABSTAIN.
- **Support guard.** Inputs whose quality is below the range seen during calibration are refused
  with ABSTAIN. This is explained in Section 14.
- **Explanations.** The interface shows:
  - a step-by-step "Why this result?" panel, generated from the same numbers the rule used,
    including the exact threshold comparison and what would change the result;
  - similarity and quality numbers;
  - the threshold region;
  - the abstention reason;
  - predicted-visibility heat maps and an "unusual input" (novelty) indicator from a separate
    visibility network. These are explanations only.

**Advantages over the existing approaches:**
- it says "I don't know" when appropriate;
- it shows its reasoning;
- it runs offline;
- every part is tested.

# 6. System Architecture

![Figure 1. VEIL-X system architecture. Solid boxes form the decision path; dashed boxes are explanatory diagnostics that never change the decision.](assets/architecture.png)

The software is organised into separate modules (Python package `veilx.app`):

| Module | Responsibility |
|---|---|
| `inputs` | File validation (format, size, corruption) |
| `face` | Detection, alignment, multiple-face refusal, restricted fallback |
| `pipeline` | Runs the models, computes similarity and quality, measures timings |
| `decision` | Fused score and the three-way A3 decision with reasons |
| `diagnostics` | Visibility maps and novelty scores (explanatory only) |
| `report` | Texts, threshold bar, heat-map overlays |
| `ui` / `server` | Gradio interface and the local web server with security headers |
| `tempfiles` | Private temporary folder and cleanup |

![Figure 2. User workflow for one comparison.](assets/workflow.png)

# 7. Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Deep learning | PyTorch 2.11 (CUDA 12.8), fp16 inference on GPU |
| Face recognition models | AdaFace IR-101 and ArcFace IR-101, pretrained on WebFace4M (CVLface releases) |
| Face detection | SCRFD-10G (InsightFace `buffalo_l`), ONNX Runtime |
| Visibility network | A small U-Net-style network (3.6 M parameters), trained in this project |
| Image processing | OpenCV, Pillow, NumPy |
| Web interface | Gradio 5 mounted in FastAPI, served by Uvicorn on 127.0.0.1 |
| Testing and quality | pytest, Ruff, strict mypy, a custom privacy scanner, GitHub Actions CI |
| Hardware used | AMD Ryzen 9 7940HS, 31 GB RAM, NVIDIA RTX 4060 Laptop GPU (8 GB), Windows 11 |

# 8. Methodology

1. **Use strong pretrained models without retraining them.**
   - AdaFace and ArcFace are kept frozen, so their behaviour is known and reproducible.
   - Retraining a face recogniser is outside the scope and resources of a course project.
2. **Create controlled test data.**
   - The confirmatory data uses the DigiFace-1M synthetic faces (Bae et al., 2023), which avoids
     using real people's photographs.
   - Occlusions are generated synthetically and reproducibly from stored seeds and recipes:
     masks, sunglasses, hands/objects, random patches and extreme lighting, each at three
     severities.
3. **Leave one occluder out.**
   - Each decision system was calibrated without one occluder type.
   - It was then tested on that unseen type, to measure how well it handles occlusions it never
     saw.
4. **Calibrate the thresholds.**
   - Thresholds were chosen on separate calibration identities with a statistical procedure
     (Learn-then-Test; Angelopoulos et al., 2021).
   - The research target was at most 5 % errors among accepted decisions, for each error type.
5. **Test once.** Final test sets were used once and logged, to avoid tuning on test data.
6. **Build the application around the frozen decision system,** then verify that it gives exactly
   the same decisions as the offline research code.

**Key measures:**
- *False match*: two different people judged the same.
- *False non-match*: the same person judged different.
- *Selective error rates*: errors counted only among the pairs the system decided.
- *Coverage*: the fraction of pairs decided, that is, not abstained.

# 9. Dataset Overview

| Dataset | Use in this project | Notes |
|---|---|---|
| DigiFace-1M (synthetic) | Training the visibility network; calibration; testing | Computer-generated faces; no real people; research licence. Identities kept separate between training, calibration and testing |
| Synthetic occlusions | All occlusion experiments | Five families + one test-only hair occluder; three severities; reproducible |
| LFW (real photographs) | Follow-up tests with synthetic overlays; detector development | Used under the project owner's academic authorisation; not redistributed |

No real-person image is stored in the project repository. The demonstration examples are
synthetic DigiFace faces generated locally from the data folder.

# 10. Implementation

**Face preparation.**
- SCRFD finds faces and five facial landmarks.
- A similarity transform aligns the face to the standard 112×112 template used by the
  recognition models.
- Detection quality is checked with the detector score and the landmark fit error.

**Embedding and quality.**
- The aligned face is passed through AdaFace IR-101.
- The embedding is normalised; its length before normalisation (the feature norm) is kept as the
  quality value.

**Decision.**
- The fused score is p = logistic(β₀ + β₁·s + β₂·q + β₃·s·q), with frozen coefficients.
- The pair's quality *q* selects the low, medium or high group. That group's thresholds decide:
  - MATCH if p ≥ τ_high;
  - NON-MATCH if p < τ_low;
  - otherwise ABSTAIN.
- For the default system:
  - low quality: [0.154, 0.833);
  - medium quality: a single threshold of 0.452;
  - high quality: a single threshold of 0.673.

**Visibility network.**
- A U-Net-style network predicts a 28×28 map of which face areas are visible.
- A conservative version and an "unusual input" (Mahalanobis novelty) score come from later
  development.
- They are shown as heat maps and notes, after the decision.

**Application.**
- Gradio provides the interface; FastAPI adds security headers.
- The server listens only on `127.0.0.1`.
- A Content-Security-Policy blocks third-party requests.
- Uploaded files are deleted right after reading; temporary display files expire within about
  three minutes.
- Every model file is checked against a recorded SHA-256 fingerprint at load time.

# 11. Modules and Features

| Feature | Description |
|---|---|
| Two-image upload | JPEG, PNG, BMP, WEBP up to 20 MB; corrupt or unsupported files are rejected with a reason |
| Automatic face detection and alignment | With multiple-face refusal and optional face selection |
| Three-way decision | MATCH / NON-MATCH / ABSTAIN with the threshold region shown |
| Similarity and quality diagnostics | AdaFace cosine, feature norms, pair quality, fused score, thresholds; ArcFace similarity for comparison |
| Abstention explanation | The exact rule that caused ABSTAIN |
| Visibility heat maps | Blue = predicted visible, orange = predicted covered; explanation only |
| Novelty indicator | Flags inputs unlike the visibility network's training data |
| Active configuration card | States which calibration is in use, what it was calibrated on, and its measured results |
| Low-quality support guard | Abstains when quality is below the calibrated range |
| Safety confirmations | The user confirms authorisation and understanding of the limitations before each comparison |
| Information tabs | Calibration and limitations; why visibility is not used for decisions; model information; synthetic examples |

# 12. Testing and Results

## 12.1 Functional and engineering tests

| Test | Result |
|---|---|
| Automated tests, CPU environment | 415 passed, 4 skipped (model tests that need the GPU environment) |
| Automated tests, GPU environment (models present) | 430 passed |
| Quality gate | PASS: Ruff lint and format, strict mypy, configuration validation, privacy/secret scan, documentation links, smoke test |
| Offline parity (same decisions as the research code) | **1,000 / 1,000** synthetic development pairs identical; largest score difference 0.0025 |
| Detector parity on real photographs (LFW development images) | 244 / 244 single-face crops pixel-identical to the offline crops |
| CPU vs GPU | 100 / 100 decisions identical; minimum embedding cosine 0.99991 |
| Local-only and privacy | Server bound to 127.0.0.1; the only external request (a Gradio CDN script) is blocked; uploads deleted; abandoned uploads removed within 3 min (measured: 149 s) |
| Safety cases | Multiple faces refused; non-face photos refused; noise images abstain via the support guard |

**Parity is not accuracy.** "1,000 / 1,000" means the application reproduces the offline
research system exactly. It does **not** mean the system recognises faces with 100 % accuracy.
Recognition accuracy is reported separately below.

## 12.2 Speed and memory

All figures below come from an NVIDIA RTX 4060 Laptop GPU and an AMD Ryzen 9 7940HS. Each request
is one pair, after warm-up, reported as median (95th percentile).

| Path | GPU | CPU only |
|---|---|---|
| Two photographs (detection on CPU + both models) | **226 ms** (285 ms) | 509 ms |
| Two pre-aligned crops | 43 ms (49 ms) | 303 ms |
| Model loading at start | about 2.7 s | about 2.3 s |
| Peak GPU memory | 550 MiB | — |

Face detection runs on the CPU (about 174 ms) and is the slowest step in the GPU path.

## 12.3 Recognition and decision results (from the research records)

| Measurement | Result |
|---|---|
| Clean synthetic faces, AdaFace | AUC 0.99856, equal error rate 1.28 % |
| Real photographs (LFW, official protocol) | 99.58 % accuracy. Likely optimistic: the model's training data may include LFW identities |
| Unseen occluder, calibration-matched population | Selective false non-match 1.08–2.12 %, false match 1.23–2.56 %, at 88.6–96.8 % coverage: within the 5 % research target |
| Harder population (synthetic identities with 72 images each) | Selective false non-match 5.36–6.34 % on familiar occluders: **above** the 5 % target |
| Independent repeat on 4,000 new identities | 5.15–6.49 % (4 of 5 configurations above target); false-match rate stayed within target (1.20–2.77 %) |

![Figure 3. Selective false non-match rate of the decision rule by configuration and population, with 95 % confidence intervals. The rule met the 5 % target on the population it was calibrated for, but not on a harder population, in two independent tests.](assets/fig_population_shift.png)

# 13. Screenshots of the Application

The examples are synthetic DigiFace faces generated locally. They are not real people.

![Figure 4. Main screen: two uploads, one-click example buttons, and the result area that guides the user through the three steps.](assets/shot_home.png)

![Figure 5. MATCH result (synthetic same-identity pair): the result card, the similarity, quality, score and rule cards, and the score bar against the high-quality cut-off.](assets/shot_match_result.png)

![Figure 6. NON-MATCH result (synthetic different-identity pair).](assets/shot_nonmatch_result.png)

![Figure 7. ABSTAIN result (same identity, one face covered by a synthetic mask): the score falls inside the low-quality uncertain zone.](assets/shot_abstain_result.png)

![Figure 8. "Why this result?" for the ABSTAIN example: each step with its exact numbers, the exact comparison that decided, and what would change the result.](assets/shot_abstain_why.png)

![Figure 9. Predicted-visibility heat maps (blue visible, orange covered). The masked lower face is highlighted. These maps are explanatory only.](assets/shot_abstain_vis.png)

![Figure 10. The same ABSTAIN example enlarged, rendered with the application's own visualisation function. Image A is a dark, rotated synthetic render, which sets the low pair quality (13.00). On image B the synthetic mask is predicted as covered (orange); predicted visible fraction 0.39.](assets/vis_example4.png)

![Figure 11. Low-quality support guard: two random-noise images are refused (ABSTAIN), and the note records that the unguarded research rule would have said MATCH.](assets/shot_guard_result.png)

# 14. Why Abstention and Quality Matter: Key Findings

**Learned visibility was not reliable enough to drive decisions.**
- The visibility network generalised to some unseen occluder types: IoU 0.87–0.90 when masks,
  sunglasses or hands were left out of its training.
- It failed on unseen random patches (IoU 0.13), which it wrongly called "visible".
- A decision rule based on visibility then made too many errors (8.52 %, above the 5 % target).
- The quality-based rule stayed within target (2.12 %).
- Therefore VEIL-X uses **quality** for decisions, and shows visibility only as an explanation.

**Combining visibility across the two faces (co-visibility) did not help reliably.** It is not
used in the final decision.

**Calibration depends on the population.**
- The thresholds were calibrated mostly on one kind of synthetic identity.
- On a harder group of identities the error rate rose above the 5 % target, and an independent
  repeat confirmed this.
- Thresholds must therefore be re-calibrated for any new population before trusting the target.

**Very low-quality inputs can produce false matches.**
- In final testing, two random-noise images received a score of 1.000 (MATCH) from the research
  rule. The fused formula extrapolates beyond the quality range it was fitted on.
- Among the roughly 1 % of test pairs below the calibrated quality range, the false-match rate
  was 22.97 %, compared with 1.50 % above it.
- The application therefore **abstains** below that range (the *support guard*).
- This guard is an application safeguard. It has not been proven to fix every case, and the
  original research results are reported unchanged.
- A score of 1.000 is a model output, **not** a 100 % probability that two images show the same
  person.

![Figure 12. False-match rate of the research rule below and above the calibrated quality range (development data).](assets/fig_lowq.png)

# 15. Advantages

- **Honest outputs.** ABSTAIN instead of a forced guess on weak evidence.
- **Explainable.** Similarity, quality, the threshold region, the abstention reason and
  visibility heat maps.
- **Strong pretrained models.** State-of-the-art embeddings without costly retraining.
- **Private.** Everything runs locally; uploads are deleted; there is no cloud.
- **Fast enough for interactive use.** About 0.2 s per photo pair on a laptop GPU, and under
  0.6 s on CPU.
- **Well tested and reproducible.** Over 400 automated tests, exact parity with the research
  code, fingerprinted model files, and documented reproduction steps.
- **Safe defaults.** Multiple faces are refused, unsuitable photos are not silently cropped, and
  very low-quality inputs abstain.

# 16. Challenges and Limitations

- **Synthetic evaluation.** Main tests used synthetic faces and synthetic occlusions. Real-world
  occlusions, such as real masks or scarves in natural photographs, were **not** evaluated.
- **Population-dependent calibration.** The 5 % target was met on the calibration-like
  population, but not on a harder population, as confirmed in two independent tests.
- **Low-quality inputs.** The research rule produces false matches below its quality range. The
  application guard reduces this risk but is not a proven universal fix.
- **Visibility maps** can miss occluders that look like faces, and were trained on synthetic
  occluders only.
- **No single calibration.** Five calibration configurations exist (one per left-out occluder
  type); none is a universal or production calibration.
- **Real-photo results.** These may be optimistic, because public-figure photos may overlap with
  the models' training data.
- **No certification.** The system is not certified and is **not** suitable for identity
  verification, access control or any decision affecting people.

# 17. Future Enhancements

1. Re-calibrate the thresholds on images from the intended user population and camera.
2. Evaluate on real, naturally occluded photographs, for example the IJB-C benchmark, after
   obtaining access.
3. Use a fused score that cannot extrapolate below its quality range, so that the guard is no
   longer needed.
4. Run face detection on the GPU to reduce latency. This requires a compatible CUDA runtime and a
   new parity test.
5. Improve the interface for multi-face images, for example with clickable face thumbnails.

# 18. Conclusion

**Result.** VEIL-X is a complete, working, local face verification application. It combines
pretrained deep-learning embeddings, image-quality assessment and calibrated three-way decisions.
It explains every output step by step, with the exact numbers and threshold that decided, and with visibility heat maps, runs privately on a laptop,
reproduces its research pipeline exactly, and is backed by an extensive automated test suite.

**Main lesson.** Saying "I don't know" is valuable, but a confidence threshold is only as reliable
as the data it was calibrated on. The project therefore reports where its targets held, where
they did not, and which safeguards were added. The result is a credible engineering
demonstration with clearly stated limits.

# 19. References

1. Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., Lei, L. (2021). Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
2. Bae, G., de La Gorce, M., Baltrušaitis, T., Hewitt, C., Chen, D., Valentin, J., Cipolla, R., Shen, J. (2023). DigiFace-1M: 1 million digital face images for face recognition. WACV 2023.
3. Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. CVPR 2019.
4. Guo, J., Deng, J., Lattas, A., Zafeiriou, S. (2022). Sample and computation redistribution for efficient face detection. ICLR 2022.
5. Huang, G. B., Ramesh, M., Berg, T., Learned-Miller, E. (2007). Labeled Faces in the Wild. Technical Report 07-49, University of Massachusetts, Amherst.
6. Kim, M., Jain, A. K., Liu, X. (2022). AdaFace: Quality adaptive margin for face recognition. CVPR 2022.
7. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI 2015.
8. Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.
9. Zhu, Z., Huang, G., Deng, J., et al. (2021). WebFace260M: A benchmark unveiling the power of million-scale deep face recognition. CVPR 2021.
10. Ngan, M., Grother, P., Hanaoka, K. (2020). Face recognition accuracy with masks using pre-COVID-19 algorithms. NISTIR 8311.

The complete research record, including the detailed research paper, all experiments and the
claim-to-evidence audit, is in the project repository: `docs/paper/veilx_paper.md`,
`docs/phase6_audit.md` and `docs/phase6_results.md`.
