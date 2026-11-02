---
title: "VEIL-X: An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"
subtitle: University Course Project Report
---

<!-- TITLEPAGE -->

<!-- FRONTMATTER -->

# Declaration

I hereby declare that the project report entitled **"VEIL-X: An AI-Based Face Verification System
with Quality Assessment and Intelligent Abstention"**, submitted in partial fulfilment of the
requirements of [Course Code and Name] at [College / University Name], is a record of original
work carried out by me under the guidance of [Guide / Supervisor Name].

The pretrained face recognition and face detection models used in this project are the work of
their respective authors and are acknowledged in the text and in the references. All other
design, implementation, experiments, measurements and documentation described in this report were
carried out as part of this project. This report has not been submitted, in whole or in part, for
any other degree, diploma or course.

Place: [Place]

Date: [Date]

[Student Name]

[Register Number]

# Acknowledgements

I would like to express my sincere gratitude to my guide, [Guide / Supervisor Name], for the
guidance, encouragement and constructive feedback provided throughout this project.

I thank the Head of the [Department] and the faculty members of [College / University Name] for
providing the facilities and the academic environment that made this work possible.

I also acknowledge the authors of the open research resources on which this project builds: the
DigiFace-1M synthetic face dataset, the AdaFace and ArcFace face recognition models released
through CVLface, the SCRFD face detector released through InsightFace, and the open-source
libraries PyTorch, ONNX Runtime, OpenCV, Gradio and FastAPI.

Finally, I thank my family and friends for their constant support.

[Student Name]

# Abstract

Face verification decides whether two face images show the same person. Modern systems compare
deep-learning embeddings with a similarity threshold and always answer "same" or "different",
even when an image is blurred, dark or partly covered. In those situations a forced answer is
often wrong, and the user is not told that the evidence was weak.

This project presents **VEIL-X**, a complete local application for face verification with
quality assessment and intelligent abstention. VEIL-X detects and aligns each face with the
SCRFD detector, computes a 512-dimensional embedding with the pretrained AdaFace IR-101 network,
and measures image quality from the length (feature norm) of that embedding. A small calibrated
logistic model combines similarity and quality into a decision score. A quality-stratified
three-way rule, calibrated with the Learn-then-Test statistical procedure, then returns
**MATCH**, **NON-MATCH** or **ABSTAIN**: in the low-quality group the rule declines to decide when
the score falls in an uncertain zone. An application-level support guard abstains whenever the
image quality lies below anything seen during calibration.

Every result is explained. A step-by-step "Why this result?" panel, generated from the same
numbers that the rule used, shows which faces were found, how similar they are compared with
typical same-person and different-person pairs, the quality group, the exact threshold
comparison that decided, and what would change the result. Predicted-visibility heat maps and a
second model's opinion are displayed as explanations only and never change the decision.

The application runs entirely on the user's computer and stores no images or embeddings. It
reproduces the offline research pipeline exactly (1,000 of 1,000 decisions), compares two
photographs in about 0.23 s on a laptop GPU, and passes 430 automated tests and an end-to-end
browser test of every control. The evaluation also reports its limits: the calibration used
synthetic faces and occlusions, and its 5 % error target held on the calibration-like
population but not on a harder population of identities.

**Keywords:** face verification, deep learning, face embeddings, image quality assessment,
selective prediction, abstention, explainable AI, Learn-then-Test calibration, privacy.

<!-- TOC -->

<!-- LOF -->

<!-- LOT -->

# List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| AUC | Area under the ROC curve |
| CI | Confidence interval |
| CNN | Convolutional neural network |
| CPU / GPU | Central / graphics processing unit |
| CSP | Content-Security-Policy (web security header) |
| EER | Equal error rate |
| FMR / FNMR | False match rate / false non-match rate |
| sFMR / sFNMR | Selective FMR / FNMR: error rates among the decided pairs only |
| IR-101 | 101-layer improved residual network (backbone of AdaFace and ArcFace) |
| IoU | Intersection over union |
| LFW | Labeled Faces in the Wild (real-photograph benchmark) |
| LOFO | Leave-one-family-out (evaluation design) |
| LTT | Learn-then-Test (statistical calibration procedure) |
| M1 | The fused logistic decision score used by VEIL-X |
| A3 | The quality-stratified three-way decision rule used by VEIL-X |
| SCRFD | Sample and Computation Redistribution for Face Detection |
| SHA-256 | Secure Hash Algorithm (256-bit), used as a file fingerprint |
| UI | User interface |
| VE-C | Conservative visibility estimator (explanatory diagnostics) |

<!-- MAINMATTER -->

# 1. Introduction

## 1.1 Background

Face verification is the one-to-one comparison of two face images to decide whether they show the
same person. It is used every day to unlock phones, at airport e-gates and in photo-organisation
software. Since the mid-2010s the field has been dominated by deep convolutional neural networks
(CNNs) that map an aligned face image to a fixed-length vector of numbers called an *embedding*.
Networks trained with margin-based losses such as ArcFace (Deng et al., 2019) and AdaFace (Kim et
al., 2022) produce embeddings in which images of the same person point in a similar direction,
while images of different people point in different directions. Verification then reduces to
measuring the cosine similarity between two embeddings and comparing it with a threshold.

On clean, frontal, well-lit photographs these systems are extremely accurate. In this project's
own measurements, the pretrained AdaFace IR-101 model reaches an equal error rate of 1.28 % on
clean synthetic faces and 99.58 % accuracy on the real-photograph benchmark LFW.

## 1.2 Motivation

Real images are rarely clean. Faces are partly covered by masks, sunglasses, hands, hair or other
objects; images are blurred, dark or taken at an angle. When identity evidence is removed, the
similarity of two images of the same person drops, and a fixed threshold starts to reject genuine
pairs. In this project's baseline experiments, adding synthetic occluders raised the false
non-match rate of a strong pretrained model from about 1.4 % on clean images to between 6.8 % and
21.6 %; for the hardest occluder type at high severity it reached 44.9 %.

A conventional verifier must still answer "same" or "different" for every pair, so these errors
are silent: the user receives a confident-looking answer that is no more reliable than a guess.
A more responsible design lets the system say "I am not confident enough to decide" when the
evidence is weak, and explains why. This idea, known as *selective prediction* or *abstention*,
is the central theme of this project.

## 1.3 Problem Statement

The problem addressed by this project is stated as follows.

> Build a local face verification application that (1) uses strong pretrained models, (2)
> measures image quality and uses it in the decision, (3) abstains when the evidence is
> insufficient instead of forcing a decision, (4) explains every output in plain language, and
> (5) protects privacy by running entirely on the user's computer, while honestly measuring and
> reporting how reliable it is.

## 1.4 Objectives

The specific objectives of the project are:

1. To implement a two-image face verification pipeline using pretrained deep-learning models for
   face detection, alignment and recognition.
2. To detect and align faces automatically, and to refuse ambiguous inputs such as images with
   several faces when the user has not chosen one.
3. To measure image quality and use it to select the decision thresholds.
4. To produce three outcomes, MATCH, NON-MATCH and ABSTAIN, using thresholds calibrated on
   held-out data with a statistical error target.
5. To explain every decision with the exact numbers and threshold that produced it, and to show
   where a face may be covered.
6. To build an easy-to-use local web interface.
7. To keep all processing local and private: no cloud services and no stored images.
8. To test the system thoroughly, measure its speed and document its limitations.

## 1.5 Scope of the Project

The project covers one-to-one verification of two images supplied by the user. It does not
perform identification (searching a person in a gallery), does not store any face database and is
not intended for surveillance. Recognition networks are used as frozen, pretrained components;
the work of the project lies in the decision layer, the calibration and abstention mechanism, the
explanation engine, the privacy-preserving application and its evaluation. The system is an
academic demonstration and not a certified identity-verification product.

## 1.6 Organisation of the Report

Chapter 2 reviews related work and existing systems. Chapter 3 lists the requirements and the
feasibility study. Chapter 4 presents the system design: architecture, modules, use cases and the
sequence of one comparison. Chapter 5 describes the algorithms in detail, from face detection to
the calibrated decision rule and the explanation engine. Chapter 6 describes the datasets.
Chapter 7 explains the implementation and the user interface. Chapter 8 covers testing, and
Chapter 9 presents and discusses the results. Chapter 10 summarises advantages and limitations,
and Chapter 11 concludes with future work. The appendices contain an installation and user guide
and the frozen configuration values.

## 1.7 Summary of Contributions

1. A complete local face verification application that returns MATCH, NON-MATCH or ABSTAIN using
   frozen state-of-the-art models and a quality-stratified, statistically calibrated decision
   rule.
2. An explanation engine that turns every decision into a plain-language, step-by-step account
   with the exact numbers and threshold comparison, and states what would change the result.
3. A privacy-preserving design: a loopback-only server, no stored images or embeddings, immediate
   deletion of uploads and fingerprint-verified model files.
4. An honest evaluation of when the calibrated error targets hold: a negative result for learned
   visibility, a population-shift failure confirmed by a pre-registered replication, and a
   low-quality failure mode together with an application safeguard.
5. A tested and reproducible engineering artefact: 430 automated tests, exact parity with the
   research code, end-to-end browser tests of every control and a one-click installer.

# 2. Literature Survey

## 2.1 Deep Face Recognition

Modern face recognition learns an embedding space with margin-based classification losses.
ArcFace (Deng et al., 2019) adds an additive angular margin to the softmax loss, which makes the
classes more compact and separable on the hypersphere. AdaFace (Kim et al., 2022) adapts the
margin to image quality, using the norm of the feature vector before normalisation as a proxy
for quality: low-quality images receive a different margin so that the network does not overfit
to unrecognisable samples. Both models are commonly trained on large web-crawled datasets such as
WebFace260M and its WebFace4M subset (Zhu et al., 2021). VEIL-X uses the IR-101 versions of both
networks trained on WebFace4M, as released by the CVLface project.

Face detection and alignment precede recognition. SCRFD (Guo et al., 2022) is an efficient
detector that also predicts five facial landmarks (eye centres, nose tip and mouth corners); a
similarity transform then maps the face to a standard 112×112 template.

## 2.2 Face Image Quality Assessment

Face image quality assessment estimates how useful an image is for recognition, rather than how
visually pleasing it is. SER-FIQ (Terhörst et al., 2020) estimates quality from the robustness of
the embedding under dropout, and MagFace (Meng et al., 2021) shows that the magnitude of the
feature vector can be trained to reflect quality. AdaFace exploits the same link. VEIL-X follows
this line and uses the AdaFace feature norm directly as its quality signal, with no extra model.

## 2.3 Occluded Face Recognition

Occlusion is a long-standing challenge. Some methods learn to discard features corrupted by
occlusion, for example the pairwise differential Siamese network of Song et al. (2019). Large
evaluations by NIST of algorithms on masked faces (Ngan et al., 2020) documented substantial
accuracy losses. Real occluded-face datasets such as ROF (Erakın et al., 2021) and IJB-C (Maze et
al., 2018) exist, but most methods still produce a forced answer for every pair, and many are
specialised to one occluder type and require retraining a large network.

## 2.4 Selective Prediction and Risk Control

The idea of rejecting uncertain inputs goes back to Chow (1970), who described the optimal
trade-off between error and rejection. Geifman and El-Yaniv (2017) formalised selective
classification for deep networks. Distribution-free methods such as conformal prediction (Vovk et
al., 2005; Angelopoulos and Bates, 2021), risk-controlling prediction sets (Bates et al., 2021)
and Learn-then-Test (Angelopoulos et al., 2021) provide finite-sample statistical guarantees on
error rates. Their key assumption is that calibration data and future data are exchangeable,
that is, drawn from the same distribution. Under a distribution shift the guarantees need extra
information, such as known likelihood ratios (Tibshirani et al., 2019). Traub et al. (2024)
discuss common pitfalls in evaluating selective systems.

## 2.5 Dataset Shift and Novelty Detection

Dataset shift (Quiñonero-Candela et al., 2009) occurs when deployment data differ from training
or calibration data; Rabanser et al. (2019) studied how to detect it. A simple and effective
novelty signal is the Mahalanobis distance of intermediate network features from the training
distribution (Lee et al., 2018). VEIL-X uses such a score only as an "unusual input" flag in its
explanations.

## 2.6 Existing Systems and Their Limitations

@tab:existing summarises the main types of existing systems and the limitations that motivated
VEIL-X.

Table: Existing approaches and their limitations {#tab:existing}
| Existing approach | Limitation |
|---|---|
| Basic face matchers with one similarity threshold | Always answer yes or no; no notion of confidence; errors on poor images are silent |
| Commercial and cloud face-recognition APIs | Images leave the user's device; behaviour is not transparent; unsuitable for privacy-sensitive use |
| Occlusion-robust recognition models (for example, mask-trained networks) | Require retraining large networks; still give forced answers; often specialised to one occluder type |
| Separate image-quality filters | Quality is checked, but it is not integrated into the decision thresholds |
| Explainability add-ons (saliency maps) | Show where a network looked, but not why a particular accept/reject decision was made |

## 2.7 Summary and Research Gap

The literature provides strong recognition networks, quality signals and statistical tools for
abstention, but these are rarely combined into one transparent, end-user system. VEIL-X combines
them: frozen state-of-the-art models, a quality-aware calibrated three-way decision, a plain
explanation of every result, and a private local application, together with an honest evaluation
of when the calibration holds and when it does not. @tab:compare compares VEIL-X with
the two most common kinds of system.

Table: Comparison of VEIL-X with common approaches {#tab:compare}
| Capability | Single-threshold matcher | Cloud face API | VEIL-X |
|---|---|---|---|
| Abstains when the evidence is weak | No | Varies by provider | Yes, with calibrated thresholds |
| Uses image quality in the decision | No | Not usually disclosed | Yes, through quality groups |
| Explains each decision | No | No | Yes, step by step with exact numbers |
| Images stay on the user's computer | Yes | No | Yes |
| Error target set statistically | No | Not usually disclosed | Yes (5 %, Learn-then-Test) |
| Limitations documented and measured | Rarely | Partly | Yes |

# 3. System Requirements and Analysis

## 3.1 Functional Requirements

Table: Functional requirements {#tab:fr}
| ID | Requirement |
|---|---|
| FR-1 | The user can upload two images (JPEG, PNG, BMP or WebP) or load a built-in example pair |
| FR-2 | Each upload is validated immediately; corrupt, blank, oversized or unsupported files are rejected with a clear message |
| FR-3 | The system detects and aligns one face per image; if an image contains several faces, it refuses to choose and asks the user to select one |
| FR-4 | The system computes similarity and image quality and returns MATCH, NON-MATCH or ABSTAIN using the calibrated rule |
| FR-5 | The system abstains when image quality is below the calibrated range (support guard) |
| FR-6 | Every result is explained step by step with the exact numbers and the exact threshold comparison |
| FR-7 | The system shows the detected faces, the aligned crops and optional visibility heat maps |
| FR-8 | The user must confirm authorisation and acknowledge the calibration limitations before each comparison |
| FR-9 | The user can compare again with other options without re-uploading, and can clear all inputs |
| FR-10 | The interface provides information pages: how it works, limitations, privacy |

## 3.2 Non-Functional Requirements

- **Privacy.** All processing is local. Uploaded files are deleted right after reading; no image,
  embedding or result is written to disk or sent over the network.
- **Security.** The server listens only on the loopback address 127.0.0.1 and sends security
  headers, including a strict Content-Security-Policy, so the page makes no third-party requests.
- **Integrity and reproducibility.** Every model and configuration file is verified against a
  recorded SHA-256 fingerprint when it is loaded; decisions must be identical to those of the
  offline research code.
- **Performance.** A comparison of two photographs should complete in well under one second on a
  laptop GPU and within about half a second on a CPU.
- **Usability.** A first-time user should be able to obtain and understand a result without
  reading documentation; messages must be in plain language.
- **Reliability.** Unexpected errors must produce a safe message without stack traces or file
  paths, and invalid inputs must never produce a decision.
- **Maintainability.** The code is modular, type-checked and covered by automated tests.

## 3.3 Hardware Requirements

Table: Hardware requirements {#tab:hw}
| Component | Minimum | Used in this project |
|---|---|---|
| Processor | 64-bit quad-core CPU | AMD Ryzen 9 7940HS |
| Memory | 8 GB RAM | 31 GB RAM |
| Graphics | Optional; an NVIDIA GPU speeds up recognition | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Storage | About 5 GB for the environment and models | Solid-state drive |
| Operating system | Windows 10/11 or Linux | Windows 11 |

## 3.4 Software Requirements

Table: Software requirements {#tab:sw}
| Category | Software |
|---|---|
| Language and environment | Python 3.12, created with the uv package manager |
| Deep learning | PyTorch 2.11 with CUDA 12.8 (GPU) or the CPU build; safetensors |
| Face detection runtime | ONNX Runtime (CPU) running SCRFD det_10g |
| Image processing | OpenCV (headless), Pillow, NumPy |
| Web application | Gradio 5.50 mounted in FastAPI, served by Uvicorn |
| Configuration and validation | PyYAML, Pydantic |
| Testing and quality | pytest, Playwright (browser testing), Ruff, mypy, GitHub Actions |
| Documentation | python-docx, python-pptx, matplotlib, PyMuPDF |
| Browser | Any modern browser (Chrome, Edge, Firefox) |

## 3.5 Feasibility Study

**Technical feasibility.** All required components are available as mature open-source software
and pretrained models, and the full pipeline runs in real time on a consumer laptop. No model
training of the recognition network is required.

**Economic feasibility.** The project uses free and open-source software and freely available
research models and datasets. No cloud services or paid APIs are needed, so the operating cost
is zero beyond an ordinary computer.

**Operational feasibility.** A one-click Windows launcher installs and starts the application, and
the interface guides the user through three steps (choose images, confirm, compare). No technical
knowledge is needed to obtain and interpret a result.

**Legal and ethical feasibility.** Face images are biometric data. The project therefore uses
synthetic faces for its evaluation and examples, keeps all processing on the user's computer,
requires the user to confirm that they are authorised to process the images, and states clearly
that the output is not a certified identity decision.

## 3.6 Assumptions and Constraints

- Each image contains one face of interest; if it contains several, the user selects one.
- The faces are reasonably visible; extremely low-quality inputs are refused by design.
- The pretrained models and the calibration are used as they are. A different population or
  camera would require re-calibration before the error target can be trusted.
- The application is used by one person at a time on a local computer; it is not designed as a
  multi-user server.
- Only images that the user is authorised to process are used.

# 4. System Design

## 4.1 Overview of the Proposed System

VEIL-X is organised as a decision path and an explanation path. The decision path validates the
two images, detects and aligns the faces, computes AdaFace embeddings, derives the similarity and
the pair quality, computes the fused decision score, and applies the calibrated quality-stratified
rule together with the support guard. The explanation path runs after the decision and can never
change it: it builds the step-by-step explanation and computes optional diagnostics (ArcFace
similarity, visibility maps and a novelty score). A Gradio web interface served only on the local
machine presents the result.

## 4.2 System Architecture

@fig:arch shows the architecture. Solid boxes form the decision path; dashed boxes are
explanatory diagnostics that never change the decision. Every frozen file (model weights,
detector, calibrated thresholds) is checked against its SHA-256 fingerprint at load time.

![VEIL-X system architecture.](assets/architecture.png){#fig:arch h=13}

## 4.3 Module Design

The application is implemented as the Python package `veilx.app`. @tab:modules lists its modules
and their responsibilities.

Table: Modules of the application package veilx.app {#tab:modules}
| Module | Responsibility |
|---|---|
| `config` | Loads and validates the application configuration; loads the frozen decision system after checking its SHA-256 |
| `inputs` | Validates uploaded bytes: format, size, image dimensions, corruption; applies EXIF orientation and converts to RGB |
| `face` | SCRFD detection, five-point alignment, multiple-face refusal, face selection, restricted fixed-crop fallback |
| `pipeline` | Runs detection, both recognition networks, similarity, quality, the decision, the support guard and timings |
| `decision` | The fused score M1 and the three-way A3 decision with recorded reasons |
| `explain` | Builds the plain-language explanation, the result card, metric cards and score bar |
| `diagnostics` | Visibility maps and novelty scores (explanatory only) |
| `report` | User-facing texts, heat-map overlays and detection annotations |
| `ui` | The Gradio interface: layout, event handlers, session state |
| `server` | Mounts the interface in FastAPI with security headers; refuses non-loopback addresses |
| `tempfiles` | Private per-session temporary directory, owner-PID marker and cleanup of stale sessions |

## 4.4 Use Case Diagram

@fig:usecase shows the interactions of the user with the system. All use cases are performed by a
single actor, the user, through the browser on the same computer.

![Use case diagram of the VEIL-X application.](assets/use_case.png){#fig:usecase h=12}

## 4.5 Sequence of One Comparison

@fig:sequence shows the messages exchanged during one comparison. When an image is uploaded, the
server validates it, reads its bytes into the browser session's memory and deletes the temporary
file at once. When the user presses *Compare faces*, the pipeline computes the embeddings, the
decision rule returns the outcome, the diagnostics are computed after the decision, and the
explanation engine converts the numbers into the step-by-step explanation shown to the user.

![Sequence diagram of one comparison.](assets/sequence.png){#fig:sequence h=14}

## 4.6 User Workflow

@fig:workflow summarises the workflow from the user's point of view: six steps from choosing the
images to reading the explanation.

![User workflow for one comparison.](assets/workflow.png){#fig:workflow}

## 4.7 Key Design Decisions

- **Frozen pretrained recognition models.** Retraining a face recogniser needs millions of images
  and large compute budgets. Using frozen AdaFace and ArcFace models keeps their behaviour known,
  reproducible and verifiable by fingerprint.
- **Quality, not visibility, drives the decision.** A learned visibility network was developed
  and evaluated as a decision signal, but it failed on occluder types it had never seen
  (Section 9.8). The feature-norm quality signal proved more reliable, so visibility is shown only
  as an explanation.
- **Three-way decisions with calibrated thresholds.** Thresholds were chosen with a statistical
  procedure with an explicit error target, instead of being tuned by hand.
- **Explanations derived from the decision itself.** The explanation is generated from exactly the
  numbers used by the rule, so it can never contradict the decision.
- **Local and private by design.** The application binds to the loopback interface, deletes
  uploads immediately, keeps images only in session memory and writes no biometric data to disk.

# 5. Methodology and Algorithms

## 5.1 Face Detection and Alignment

Each validated image is passed to the SCRFD det_10g detector (InsightFace `buffalo_l`), run with
ONNX Runtime. The detector returns face boxes, confidence scores and five landmarks per face
(detection threshold 0.5, non-maximum suppression 0.4). A detection is accepted only if its
confidence is at least 0.6 and the landmarks fit the alignment template with a residual of at
most 7.34 pixels (the 99th percentile on clean development photographs). The face is then
warped by a five-point similarity transform to the standard 112×112 template used by the
recognition networks.

Three special cases are handled explicitly:

- **Several faces.** If more than one face is detected and the user has not selected one, the
  image is refused. VEIL-X never chooses a person on the user's behalf. The detection image shows
  each face with its number, and the user can enter the number in the advanced options.
- **No reliable face.** If no face passes the checks, the image is refused with a reason. Only if
  the user declares that both images are loosely cropped, centred, single-face portraits (as in
  the LFW benchmark) and the image is nearly square (aspect ratio 0.8 to 1.25) does the
  application apply a registered fixed-crop fallback: the image is resized to 250×250 and warped
  with a fixed transform derived from average landmark positions. Such a crop is always labelled
  as a fallback.
- **Pre-aligned crops.** Images that are already aligned 112×112 face crops (such as the
  synthetic examples) skip detection; the interface switches this option on automatically when
  all loaded images are 112×112.

## 5.2 Face Embedding and Quality Estimation

The aligned crop is normalised and passed through the frozen AdaFace IR-101 network, which
outputs a 512-dimensional feature vector **f**. On the GPU the network runs with fp16
automatic mixed precision; on 20,000 validation pairs the embeddings agree with fp32 inference
with a cosine similarity of at least 0.9998. The vector is L2-normalised to give the embedding
**e** = **f** / ‖**f**‖, and its length before normalisation, ‖**f**‖, is kept as the *feature
norm*. AdaFace was trained so that this norm reflects image quality: blurred, dark, small or
covered faces produce smaller norms.

## 5.3 Similarity and Pair Quality

The similarity of two faces is the cosine of the angle between their embeddings:

$$ s = e_A · e_B ,   −1 ≤ s ≤ 1 $$

A comparison is only as reliable as its weaker image, so the pair quality is the smaller of the
two feature norms:

$$ q = min( ‖f_A‖ , ‖f_B‖ ) $$

## 5.4 Fused Decision Score (M1)

Similarity and quality interact: a moderate similarity is strong evidence when both images are of
high quality, but weak evidence when one of them is poor. A logistic-regression model, called
M1, therefore combines the similarity, the quality and their product into one decision score:

$$ p = σ( β₀ + β₁·s′ + β₂·q′ + β₃·(s·q)′ ),   with σ(z) = 1 / (1 + exp(−z)) $$

where the prime denotes standardisation with the mean and standard deviation of the training
features. The coefficients were fitted on development pairs and then frozen. @tab:m1 gives the
values for the default system.

Table: Frozen M1 coefficients of the default system F1_mask (standardised features) {#tab:m1}
| Term | Coefficient | Feature mean | Feature standard deviation |
|---|---|---|---|
| Intercept β₀ | 1.1259 | – | – |
| Similarity s (β₁) | 4.0634 | 0.303 | 0.235 |
| Quality q (β₂) | −2.4766 | 20.420 | 2.967 |
| Product s·q (β₃) | 5.0348 | 6.400 | 5.284 |

The score lies between 0 and 1, but it is **not** the probability that two images show the same
person. It is a decision score whose thresholds were calibrated as described next.

## 5.5 Threshold Calibration with Learn-then-Test

The thresholds of the decision rule were chosen with Learn-then-Test (LTT; Angelopoulos et al.,
2021), a distribution-free statistical procedure. A rule maps the score p to MATCH if p ≥ τ_high,
to NON-MATCH if p ≤ τ_low, and to ABSTAIN otherwise. The research target was that, among the pairs
the rule decides, at most 5 % of the same-person pairs are wrongly rejected (selective false
non-match rate, sFNMR) and at most 5 % of the different-person pairs are wrongly accepted
(selective false match rate, sFMR), at a confidence level of 95 % (δ = 0.05).

The procedure works as follows:

1. A list of candidate threshold pairs (τ_low, τ_high) is fixed in advance and ordered from the
   most efficient (deciding most pairs) to the most conservative, using validation data only.
2. For each candidate, the number of errors on an independent calibration set is counted, and an
   exact binomial p-value tests the hypothesis that its true error rate exceeds 5 %.
3. Candidates are tested in the fixed order (fixed-sequence testing). The rule selected is the
   last candidate in the sequence for which the hypothesis is rejected for both error types.

The calibration set contains 6,146 pairs in which no identity or image is shared between pairs,
which satisfies the independence assumption of LTT. The guarantee holds for data that are
exchangeable with the calibration data; it does not extend automatically to other populations or
to occluders not represented in calibration. Chapter 9 measures what happens in those cases.

## 5.6 Quality-Stratified Three-Way Decision Rule (A3)

A single pair of thresholds would have to be conservative for every quality level. The A3 rule
instead divides pairs into three quality groups using the tertiles of q on validation data, and
calibrates separate thresholds for each group. For the default system the groups are low
(q < 19.90), medium (19.90 ≤ q < 21.97) and high (q ≥ 21.97). The decision is:

$$ MATCH if p ≥ τ_high;   NON-MATCH if p ≤ τ_low;   ABSTAIN if τ_low < p < τ_high $$

In the medium and high groups the calibrated τ_low and τ_high coincide, so these groups never
abstain: a single cut-off decides. In the low-quality group the two thresholds are far apart,
creating an uncertain zone in which the system abstains. @fig:rule draws the rule of the default
system: each bar is the score axis of one quality group, and black lines mark the results of three
built-in examples; and @tab:rules lists the
thresholds of all five frozen systems.

![The A3 decision rule of the default system F1_mask.](assets/decision_rule.png){#fig:rule}

Table: Frozen A3 thresholds of the five calibration systems {#tab:rules}
| System | Quality edges | Low-quality band (τ_low / τ_high) | Medium cut-off | High cut-off | Support floor |
|---|---|---|---|---|---|
| F1_mask (default) | 19.90 / 21.97 | 0.154 / 0.833 | 0.452 | 0.673 | 10.82 |
| F2_sunglasses | 19.07 / 21.56 | 0.081 / 0.885 | 0.448 | 0.448 | 10.87 |
| F3_hand_object | 19.37 / 21.69 | 0.070 / 0.879 | 0.452 | 0.634 | 10.86 |
| F4_random_patch | 19.82 / 21.87 | 0.072 / 0.854 | 0.426 | 0.426 | 12.45 |
| F5_extreme_lighting | 19.13 / 21.53 | 0.060 / 0.862 | 0.421 | 0.595 | 11.01 |

Five systems exist because the research evaluation used a leave-one-family-out design: each
system was calibrated without one occluder family, so that its behaviour on that unseen family
could be measured. The application uses F1_mask by default (the first registered system) and lets
the user choose another one in the advanced options.

## 5.7 Low-Quality Support Guard

A final audit of the research system found that the frozen M1 model extrapolates badly when the
pair quality is far below the range seen during calibration. Because the quality coefficient is
negative and the interaction term changes sign, the logit grows towards p = 1 at very low quality
even for dissimilar faces: for the default system, s = 0.0 with q = 5 gives p = 0.93, and pairs of
random-noise images receive p ≈ 1.000. The application therefore adds a *support guard*: if q is
below the 1st percentile of the calibration quality (10.82 for the default system; see
@tab:rules), the result is ABSTAIN regardless of the score. The explanation states what the
unguarded rule would have answered. The frozen research system itself is unchanged.

## 5.8 Explanation Generation

The explanation engine (module `explain`) converts the verification result into a numbered list
of steps. Each step has a status (passed, warning, stopped, information or not reached) and one
or more plain-language sentences containing the exact values:

1. **Images checked**: both files decoded and passed validation.
2. **Faces found and aligned**: for each image, the number of faces, the detector confidence,
   or the reason for refusal and how to fix it.
3. **How similar are the two faces?**: the similarity s, placed against typical values.
4. **How good are the images?**: both feature norms, the weaker image, the quality group and,
   if applicable, the support-guard warning.
5. **Combined score**: the score p, with the reminder that it is not a probability.
6. **Which rule decided?**: the rule of the quality group and the exact comparison, for example
   "p = 0.542 lies between 0.154 and 0.833 → ABSTAIN", with the distance from the cut-off.
7. **Second opinion**: the ArcFace similarity and whether the two models broadly agree
   (difference below 0.15).
8. **Where might the faces be covered?**: the predicted visible fraction and the face regions
   that may be covered.

The explanation ends with **What would change the result?**: the score that a MATCH or a
NON-MATCH would need, a warning when the score is close to a cut-off, or what to fix in the
images. The distance from a single cut-off is described as clear-cut (at least 0.25), fairly
clear (at least 0.08) or close to the cut-off.

The typical values used in step 3 come from 24,732 development pairs and are listed in
@tab:refranges. A similarity is described as typical of the same person when it is at least the
25th percentile of same-person pairs (0.41), typical of different people when it is at most the
75th percentile of different-person pairs (0.15), and ambiguous in between. These ranges give
context only and never affect a decision.

Table: Reference ranges used in explanations (24,732 development pairs) {#tab:refranges}
| Quantity | 10th pct. | 25th pct. | Median | 75th pct. | 90th pct. |
|---|---|---|---|---|---|
| Similarity, same person | 0.298 | 0.409 | 0.518 | 0.615 | 0.688 |
| Similarity, different people | 0.003 | 0.048 | 0.098 | 0.151 | 0.202 |
| Pair quality q | 16.22 | 19.05 | 21.03 | 22.45 | 23.54 |

Because every sentence is generated from the same values that the rule used, the explanation
cannot contradict the decision. An automated test checks this property over a grid of scores in
all three quality groups.

## 5.9 Visibility Estimation and Novelty Detection

A separate visibility network, trained in this project, predicts a 28×28 map of the probability
that each part of the aligned face is visible. It is a U-Net-style network (Ronneberger et al.,
2015) with 3.6 million parameters. The conservative version used by the application (VE-C) was
trained with an asymmetric loss that penalises calling a covered region visible four times as
much as the opposite error, followed by temperature scaling and a lower-bound shift. A novelty
score, the Mahalanobis distance of the network's bottleneck features from the training
distribution, flags inputs that look unusual; its threshold allows 5 % false flags on normal
inputs. The map is shown as a blue (visible) and orange (covered) overlay, and the regions eyes,
nose, mouth and lower face are reported as possibly covered when less than half of the region is
predicted visible. All of this is computed after the decision and is explanatory only.

## 5.10 Evaluation Metrics

- **False match rate (FMR):** the fraction of different-person pairs accepted as the same person.
- **False non-match rate (FNMR):** the fraction of same-person pairs rejected.
- **Selective FMR and FNMR (sFMR, sFNMR):** the same rates computed only over the pairs that the
  system decided, that is, excluding abstentions.
- **Coverage:** the fraction of pairs decided (not abstained).
- **AUC and EER:** the area under the ROC curve and the equal error rate, which summarise how well
  a score separates same-person from different-person pairs.
- **Violation:** a test cell in which the 95 % confidence interval of an error rate lies entirely
  above the 5 % target. Intervals use the delete-one-identity jackknife (Efron and Tibshirani,
  1993), because pairs that share identities are not independent.

# 6. Dataset Description

## 6.1 DigiFace-1M Synthetic Faces

DigiFace-1M (Bae et al., 2023) is a dataset of computer-generated face images rendered from 3D
models with controlled variation in pose, expression, lighting and accessories. Because the faces
are synthetic, no real person's photograph is needed for training, calibration or testing, which
avoids the privacy problems of web-crawled face data. The dataset contains identities with 72
images each and identities with 5 images each. @tab:digiface shows how the identity ranges were
used.

Table: Use of DigiFace-1M identities in the project {#tab:digiface}
| Identities | Images per identity | Use |
|---|---|---|
| 0–1999 and 100,000–199,998 | 72 and 5 | Training of the visibility network, fusion-model fitting, validation, calibration and the first confirmatory test |
| 2000–5999 | 72 | Untouched population-shift test set (`p4_test`) |
| 6000–9999 | 72 | Untouched pre-registered replication set (`p6_test`), 4,000 identities |

The calibration set `p3_cal_independent` contains 6,146 identity-disjoint pairs (3,073 same-person
pairs) over 9,219 identities, of which 9,033 are 5-image identities. This detail becomes important
in Section 9.9.

## 6.2 Synthetic Occlusions

Occlusions were rendered synthetically and reproducibly from stored recipes and random seeds, at
three severity levels each. @tab:occluders lists the families.

Table: Synthetic occluder families {#tab:occluders}
| Family | Description |
|---|---|
| F1 mask | Medical-style masks covering the nose, mouth and chin |
| F2 sunglasses | Sunglasses covering the eye region |
| F3 hand / object | Hands and objects partly covering the face |
| F4 random patch | Random textured patches at random positions |
| F5 extreme lighting | Strong shadows and over-exposure (a corruption rather than an occluder) |
| E1 hair (evaluation only) | Hair strands across the face, defined in advance and never used in any design decision |

## 6.3 Labeled Faces in the Wild (LFW)

LFW (Huang et al., 2007) is a benchmark of real photographs of public figures. It was used for a
recognition-accuracy check under the official protocol, for developing and testing the face
detector path, and for follow-up tests with synthetic overlays. It was used under the project
owner's academic authorisation and is not redistributed. Because LFW identities probably overlap
the training data of the pretrained models, results on LFW are likely optimistic.

## 6.4 Data Splits and Leakage Prevention

Identities were assigned to splits by a hash of the identity number, so that no identity appears
in more than one split. Within the development identities, sub-splits were used for training
(`p3_train`), validation (`p3_val`), calibration (`p3_cal`) and testing (`p3_test`). Every test
partition was accessed through a logged run, and sets that had been inspected before are reported
as follow-up rather than confirmatory results. Decisions about the method were fixed in written
protocols before the corresponding test data were scored.

## 6.5 Ethical Considerations

The project performs two-image verification only: there is no identification, no face gallery
and no surveillance function. No face image, embedding, identity list or model weight file is
stored in the source-code repository; only aggregate results are published. The built-in examples
are synthetic faces generated locally from the data folder. Users must confirm that they are
authorised to process the images before every comparison.

# 7. Implementation

## 7.1 Development Environment

The project is written in Python 3.12. Separate virtual environments, created with the uv package
manager, are used for the CPU-only library and tests, for the GPU application (PyTorch with CUDA
12.8, ONNX Runtime, Gradio) and for building the documentation. The source code is managed with
Git and hosted on GitHub, where a continuous-integration workflow runs the lint, type-check and
test suite on every push.

## 7.2 Project Structure

Table: Structure of the project repository {#tab:structure}
| Folder | Contents |
|---|---|
| `src/veilx/app` | The application: configuration, inputs, face preparation, pipeline, decision, explanation, diagnostics, interface, server |
| `src/veilx/` (other packages) | Research library: data handling, occlusion rendering, models, metrics, calibration, statistics, simulation |
| `configs/` | Application and protocol configuration (YAML), explanation reference ranges (JSON) |
| `scripts/` | Command-line tools: run the application, download models, evaluate, build the runtime pack and the documents |
| `tests/` | 43 automated test modules |
| `docs/` | Protocols, results, decision log, user guide, research paper and university documents |
| `reports/` | Committed aggregate results and figures with provenance records |
| `run_veilx.bat` | One-click Windows launcher |

## 7.3 Core Modules

**Input validation** (`inputs`). Each upload is checked for an accepted format (JPEG, PNG, BMP,
WebP), a maximum size of 20 MB, a minimum side of 64 pixels and a maximum side of 4,096 pixels.
The file is decoded in memory with a structural integrity check; corrupt or truncated files are
rejected with a reason code. EXIF orientation is applied and the image is converted to RGB.

**Face preparation** (`face`). Implements the detection, quality checks, alignment, face
selection, multiple-face refusal and restricted fallback described in Section 5.1. The detector
is loaded lazily and its weight file is verified by SHA-256 before use.

**Pipeline** (`pipeline`). Prepares both faces, runs AdaFace and ArcFace in one batch, computes
s and q, applies the decision rule and the support guard, records the timings of each stage and,
optionally, computes the diagnostics after the decision. If a face cannot be prepared, the result
is ABSTAIN with the recorded reason and no embedding is computed.

**Decision** (`decision`). A pure NumPy implementation of the M1 score and the A3 rule. It is
tested to give exactly the same decisions as the offline evaluation code.

**Explanation** (`explain`). Implements the explanation engine of Section 5.8, and renders the
result card, the four metric cards, the score bar and the explanation steps as HTML. All text is
escaped before rendering.

**Interface and server** (`ui`, `server`). Build the web interface (Section 7.4) and mount it in
FastAPI with security headers (Section 7.5).

## 7.4 User Interface

The interface has four tabs: *Verify faces*, *How it works*, *Limitations* and *Privacy & about*.
The main tab (@fig:home) guides the user through three numbered steps. The left column holds
the two image boxes, one-click example buttons, the advanced options, the two confirmations and
the *Compare faces* and *Clear* buttons; the right column shows the result.

![Main screen of the application.](assets/shot_home.png){#fig:home}

Uploads are validated as soon as they arrive. A green line confirms a valid image; a red line
explains why a file was rejected (@fig:uploaderr). The image bytes are kept in the server-side
memory of the browser session, so the same pair can be compared again with different options
without uploading it again; *Clear* forgets them.

![An invalid upload is rejected immediately with a clear message.](assets/shot_upload_error.png){#fig:uploaderr}

The advanced options (@fig:advanced) allow the user to declare pre-aligned crops (switched on
automatically for 112×112 images), to select the face to use when an image contains several
faces, to allow the fixed-crop fallback for portrait-style photographs, to switch the visibility
diagnostics on or off, and to choose one of the five calibration systems. The *Compare faces*
button is enabled only when both confirmations are ticked, and the server checks the
confirmations again.

![The advanced options.](assets/shot_advanced.png){#fig:advanced h=11}

The *How it works* tab (@fig:how) explains the method in six illustrated steps and defines the
three possible answers. It also contains the exact thresholds of every calibration system and the
model details. The *Limitations* tab states the calibration and domain-shift limitations, and the
*Privacy & about* tab explains what happens to the images.

![The six explanation cards of the How it works tab.](assets/shot_how.png){#fig:how}

The interface uses system fonts only (no external font requests), supports light and dark themes,
and adapts to phone-width screens without horizontal scrolling.

## 7.5 Security and Privacy Implementation

Table: Privacy and security measures {#tab:privacy}
| Aspect | Implementation |
|---|---|
| Network exposure | The server binds to 127.0.0.1 only and refuses any non-loopback address; no public share link; analytics disabled |
| Browser requests | A Content-Security-Policy restricts the page to the local origin, so the browser makes no third-party request; `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` |
| Uploads | Read into memory, validated, and the temporary file is deleted immediately; the bytes stay in session memory for at most one hour and are removed by Clear |
| Displayed images | Written by Gradio to a private per-session cache folder; files older than two minutes are removed every minute |
| Embeddings and results | Computed in memory only; never written to disk, logged or transmitted |
| Session folder | Created per run with an owner-process marker; removed on exit, and folders of crashed sessions are removed at the next start |
| Model integrity | Every weight and configuration file is verified against its recorded SHA-256 fingerprint before use |
| Errors | Users see a safe message without stack traces or file paths |

## 7.6 One-Click Launcher

For users without programming experience, the project provides `run_veilx.bat`. On the first run
it unpacks the runtime pack (frozen configuration, detector, visibility weights and example
images), installs Python 3.12 and all libraries into a private folder with the uv package manager
(choosing the CUDA or CPU build of PyTorch automatically), downloads the two recognition models
(about 520 MB) and verifies their fingerprints, and finally starts the application and opens the
browser. Later runs start within seconds. Appendix A gives the full instructions.

# 8. Testing

## 8.1 Testing Strategy

Testing was carried out at five levels:

1. **Unit tests** of individual functions: input validation, alignment, metrics, calibration,
   decision rule, explanation text and configuration checks.
2. **Integration tests** of the full pipeline with fake detectors and models, which check, for
   example, that diagnostics can never change a decision.
3. **Parity tests** which compare the application with the offline research code on real data.
4. **End-to-end interface tests** in a real browser with Playwright, exercising every button and
   flow.
5. **Static checks** through a quality gate: linting, formatting, strict type checking,
   configuration validation, a privacy scanner that refuses face images or secrets in the
   repository, documentation link checks and a data-free smoke test.

## 8.2 Automated Test Suite

The automated suite contains 43 test modules. In the CPU-only environment 415 tests pass and 4
are skipped because they need the GPU environment; in the GPU environment, with the models
present, all 430 tests pass. @tab:testgroups groups the tests by area.

Table: Automated tests by area {#tab:testgroups}
| Area | Examples of what is tested | Tests |
|---|---|---|
| Application | Input validation, face statuses, fallback restrictions, decision states, consent gate, safe errors, upload deletion, interface construction | 32 |
| Explanations | The stated comparison matches the decision for every outcome; support guard; refusals; HTML escaping; reference ranges | 10 |
| Decision and calibration | LTT calibration, metrics, jackknife and paired statistics, novelty and hybrid rules | 59 |
| Data and integrity | Manifests, pairs, splits, archives, image files, de-duplication, untouched-data guards | 78 |
| Models and features | Pre-processing, model loading, detection and alignment, detector policy, pipeline integration, embedding cache, visibility models | 53 |
| Configuration and privacy | Configuration validation, privacy and secret scanner, imports, infrastructure | 98 |
| Research evaluation | Occlusion rendering, simulations, evaluation helpers and endpoints, LFW protocol, bootstrap, evidence records | 100 |
| **Total** | | **430** |

## 8.3 Parity and Consistency Tests

Table: Parity and consistency results {#tab:parity}
| Test | Result |
|---|---|
| Same decisions as the offline research code | 1,000 / 1,000 development pairs identical; largest score difference 0.0025 |
| Detector path on real photographs | 244 / 244 single-face crops pixel-identical to the offline crops |
| CPU versus GPU | 100 / 100 decisions identical; minimum embedding cosine 0.99991 |
| fp16 versus fp32 embeddings | Minimum cosine at least 0.9998 on 20,000 validation pairs |
| Explanation versus decision | The stated comparison and label match the frozen rule over a grid of scores in all quality groups |

Parity is not accuracy: "1,000 / 1,000" means that the application reproduces the research system
exactly. Recognition accuracy is reported separately in Chapter 9.

## 8.4 End-to-End Interface Testing

The running application was tested in a real browser with Playwright. @tab:uitests lists the test
cases and their results.

Table: End-to-end interface test cases {#tab:uitests}
| ID | Test case | Expected result | Result |
|---|---|---|---|
| TC-01 | Open the page | Compare button disabled until both confirmations are ticked | Pass |
| TC-02 | Example "Same person" | MATCH, score 1.000 | Pass |
| TC-03 | Example "Different people" | NON-MATCH, score 0.001 | Pass |
| TC-04 | Example "Different people (2)" | NON-MATCH, score 0.006 | Pass |
| TC-05 | Example "Same person, masked" | ABSTAIN, score 0.542 in the uncertain zone | Pass |
| TC-06 | Example "Not a face (noise)" | ABSTAIN by the support guard; unguarded answer stated | Pass |
| TC-07 | Press Compare again on the same pair | Same result without re-uploading | Pass |
| TC-08 | Upload two real photographs | Faces detected and aligned; temporary files deleted from disk | Pass |
| TC-09 | Image with two faces | Refused; advanced options open; face numbers shown | Pass |
| TC-10 | Select face 0, face 1, then a non-existent face 5 | MATCH, NON-MATCH, refusal naming the valid numbers | Pass |
| TC-11 | Image without a face, fallback off and on | Refused with advice; with the fallback, a labelled fixed crop | Pass |
| TC-12 | Corrupt PNG file | Rejected with a clear message; box reset | Pass |
| TC-13 | Text file renamed as an image | Rejected by the browser before upload | Pass |
| TC-14 | Remove one image, then compare | Message asking for the missing image | Pass |
| TC-15 | Clear button | Images, messages and result removed | Pass |
| TC-16 | Change calibration system; diagnostics off and on | Result names the system; heat maps hidden and shown | Pass |
| TC-17 | All tabs and collapsible sections | Content shown correctly | Pass |
| TC-18 | Dark theme and 390-pixel phone width | Readable; no horizontal scrolling | Pass |

## 8.5 Defects Found and Fixed

Testing found several defects, all of which were fixed and covered by tests where possible
(@tab:bugs).

Table: Defects found during testing and their fixes {#tab:bugs}
| Defect | Cause | Fix |
|---|---|---|
| Two random-noise images were accepted as MATCH with score 1.000 | The fused score extrapolates below the calibrated quality range | Support guard: ABSTAIN below the 1st percentile of calibration quality |
| Uploads could remain on disk after a failed request | Interface cache retention | Cache files expire within minutes; uploads deleted on arrival |
| Pressing Compare a second time reported missing images | Uploads were deleted when first read | Image bytes kept in session memory; Clear forgets them |
| Example buttons failed after about two minutes | The cache cleanup emptied the whole session folder, including the examples and the owner marker | Separate cache sub-folder; examples held in memory |
| A corrupt upload showed only a generic "Error" | The interface library decoded the file before the application's validator | Upload passed directly to the application's validator |
| The page scrolled sideways on phones | The tab bar did not wrap | Responsive style rule |
| The fallback could apply to a face selected in a multi-face image | Missing condition | The fallback never applies to multi-face images |

## 8.6 Quality Gate and Continuous Integration

A single command runs the quality gate: Ruff lint and format checks, strict mypy type checking,
the full test suite, configuration validation, the privacy and secret scanner, documentation link
checks and a smoke test that needs no data. The same checks run on GitHub Actions for every push.
The final version passes all checks.

# 9. Results and Discussion

## 9.1 Application Results on the Built-in Examples

@tab:examples lists the results of the five built-in examples, measured in the running
application on the GPU with the default system F1_mask (s: AdaFace similarity, q: pair quality,
p: decision score, ArcFace: the second model's similarity, shown for comparison only).

Table: Results of the built-in examples (default system F1_mask) {#tab:examples}
| Example | s | q | Group | p | Result | ArcFace |
|---|---|---|---|---|---|---|
| Same person | 0.655 | 23.47 | High | 1.000 | MATCH (p ≥ 0.673) | 0.606 |
| Different people | 0.108 | 21.48 | Medium | 0.001 | NON-MATCH (p < 0.452) | 0.013 |
| Different people (2) | 0.172 | 22.29 | High | 0.006 | NON-MATCH (p < 0.673) | 0.196 |
| Same person, masked | 0.141 | 13.00 | Low | 0.542 | ABSTAIN (0.154 < p < 0.833) | 0.090 |
| Not a face (noise) | 0.780 | 6.78 | Below range | 1.000 | ABSTAIN (support guard) | 0.714 |

@fig:match, @fig:nonmatch and @fig:abstain show the result panels. Each panel shows the answer and
a one-line summary, four cards with the similarity, quality, score and the rule applied, and a bar
that places the score on the rule of its quality group.

![MATCH result for the "Same person" example.](assets/shot_match_result.png){#fig:match}

![NON-MATCH result for the "Different people" example.](assets/shot_nonmatch_result.png){#fig:nonmatch}

![ABSTAIN result for the "Same person, masked" example.](assets/shot_abstain_result.png){#fig:abstain}

The masked example is a genuine same-person pair, yet its similarity of 0.141 is typical of
different people. A conventional matcher with one threshold would have answered NON-MATCH, which
would be wrong. Because image A is dark and rotated (q = 13.00), the pair falls in the low-quality
group, and VEIL-X abstains instead of making that error.

## 9.2 Explanation of a Decision

@fig:why1 and @fig:why2 show the complete explanation of the masked example. Steps 1 to 4 show
that both images were valid, the similarity is low, and image A has low quality. Steps 5 and 6 give
the score and the exact rule comparison. The second opinion of ArcFace (0.090) agrees, and the
visibility step reports that about 39 % of face B is predicted visible, with the nose, mouth and
lower face possibly covered. The final box states what would change the result.

![Explanation of the masked example, steps 1 to 4.](assets/shot_abstain_why1.png){#fig:why1}

![Explanation of the masked example, steps 5 to 8.](assets/shot_abstain_why2.png){#fig:why2}

## 9.3 Visibility Diagnostics

@fig:vis shows the predicted-visibility maps of the masked example as displayed in the
application (blue visible, orange covered), and @fig:visbig shows the same example enlarged.
Image A is a dark, rotated synthetic render, which sets the low pair quality (13.00). On image B
the synthetic mask is correctly highlighted as covered, with a predicted visible fraction of
0.39. The maps are computed after the decision and are explanations only.

![Visibility maps of the masked example in the application.](assets/shot_abstain_vis.png){#fig:vis}

![The masked example enlarged, with its visibility maps.](assets/vis_example4.png){#fig:visbig}

## 9.4 Safety Behaviour

**Support guard.** For two random-noise images the research rule alone would have answered MATCH
with a score of 1.000. The application abstains and explains why (@fig:guard and @fig:guardrule).

![Support guard: two random-noise images are refused.](assets/shot_guard_result.png){#fig:guard}

![The explanation of the support guard states the unguarded answer.](assets/shot_guard_rule.png){#fig:guardrule}

**Multiple faces.** When an image contains two faces, VEIL-X refuses to choose (@fig:multi),
opens the advanced options and shows the numbered detections (@fig:multifaces). After the user
selected face 0, the comparison completed normally.

![An image with two faces is refused until the user selects one.](assets/shot_multi_result.png){#fig:multi}

![Numbered face detections shown to the user (synthetic faces).](assets/shot_multi_faces.png){#fig:multifaces}

## 9.5 Performance

All timings were measured on an NVIDIA RTX 4060 Laptop GPU and an AMD Ryzen 9 7940HS CPU, with
one request at a time after warm-up, and are reported as median (95th percentile).

Table: Speed and memory of the application {#tab:speed}
| Path | GPU | CPU only |
|---|---|---|
| Two photographs (detection on CPU + both models) | 226 ms (285 ms) | 509 ms |
| Two pre-aligned crops | 43 ms (49 ms) | 303 ms |
| Model loading at start | about 2.7 s | about 2.3 s |
| Peak GPU memory | 550 MiB | – |

Face detection runs on the CPU (about 174 ms) and is the slowest stage in the GPU path. On the CPU
alone, the two recognition networks dominate (about 0.15 s each).

## 9.6 Recognition Accuracy

On clean synthetic faces (61,122 same-person and 61,122 different-person pairs), AdaFace reached
an AUC of 0.99856 and an equal error rate of 1.28 %; ArcFace reached 0.99789 and 1.61 %. A
threshold fixed for a 1 % false match rate on calibration data gave a realised false match rate
of 1.04 % and a false non-match rate of 1.42 % on the test set. On real photographs (LFW, official
protocol) AdaFace with the detector reached 99.58 % ± 0.26 accuracy; this figure is probably
optimistic because LFW identities may appear in the model's training data.

## 9.7 Decision Reliability under Unseen Occlusion

The first confirmatory evaluation tested each calibration system on the occluder family it had
never seen. @tab:phase3 compares the A3 quality rule with a rule stratified by learned visibility
(A4) and an oracle using the true visibility (A5). All A4 and A5 false match rates stayed between
0.89 % and 3.14 %.

Table: Held-out occluder results on the calibration-matched population (%) {#tab:phase3}
| Held-out family | A3 coverage | A3 sFNMR | A3 sFMR | A4 sFNMR | A5 sFNMR |
|---|---|---|---|---|---|
| Mask | 93.7 | 1.94 | 2.56 | 1.75 | 1.30 |
| Sunglasses | 96.8 | 1.87 | 1.24 | 1.28 | 2.06 |
| Hand / object | 94.3 | 1.59 | 1.23 | 2.36 | 2.01 |
| Random patch | 88.6 | 2.12 | 2.00 | **8.52** | 2.76 |
| Lighting | 96.1 | 1.08 | 1.27 | 2.25 | 0.87 |

The A3 rule met both 5 % targets on every held-out family, deciding 88.6–96.8 % of pairs. The
learned-visibility rule violated the target on the unseen random patches (sFNMR 8.52 %). For
comparison, a threshold calibrated on clean images only failed everywhere under occlusion (sFNMR
5.97–19.81 %).

## 9.8 Learned Visibility versus Quality

The visibility network generalised to some unseen occluders (IoU 0.87–0.90 for masks, sunglasses
and hands) but not to random patches (IoU 0.13), which it confidently called visible: it had
learned the *classes* of occluders it was shown, not the general idea of a covered face. A
conservative redesign (VE-C) reduced the share of badly overestimated renders on random patches
from 94.6 % to 21.9 %, and its novelty score separated a never-seen hair occluder from known
occluders well (AUROC 0.91–0.98). Even so, no visibility-based rule was safer than A3 on the ten
primary test cells (@tab:phase4); even true visibility (A5) was worse than A3 on every cell with
the never-seen hair occluder. VEIL-X therefore decides with quality and shows visibility only as an
explanation.

Table: Number of target violations on the 10 primary cells (lower is better) {#tab:phase4}
| Rule | Violations |
|---|---|
| A3 quality (used by VEIL-X) | 6 |
| A4 learned visibility | 10 |
| C4 conservative visibility | 10 |
| N4 novelty-aware visibility | 9 |
| H4 quality-first hybrid | 9 |
| A5 oracle visibility | 9 |

## 9.9 Effect of a Population Shift

The calibration set consisted almost entirely (9,033 of 9,219 identities) of 5-image synthetic
identities. On an untouched test set of 72-image identities, which were rendered with more pose
and accessory variation, same-person pairs were markedly harder (clean false non-match rate 4.81 %
against 1.12 %). Every rule, A3 included, then exceeded the 5 % sFNMR target even on familiar
occluders (@tab:shift). A pre-registered replication on 4,000 further untouched identities
confirmed the finding: the A3 interval excluded the target in 4 of 5 familiar-occluder cells,
while the false match rate stayed within target in all 15 cells (1.20–2.77 %). @fig:shift shows
the results with 95 % confidence intervals: the rule met the 5 % target on the population it was
calibrated for, but not on the harder population, in both independent tests.

Table: A3 on harder identities, familiar occluders (coverage / sFNMR / sFMR, %) {#tab:shift}
| System | First test (2,000 identities) | Replication (4,000 identities) |
|---|---|---|
| F1 mask | 94.0 / 5.89 / 1.27 | 93.8 / 5.90 / 1.40 |
| F2 sunglasses | 89.7 / 6.34 / 1.67 | 89.8 / 6.49 / 1.73 |
| F3 hand / object | 90.2 / 6.16 / 1.59 | 90.0 / 6.41 / 1.77 |
| F4 random patch | 93.2 / 5.36 / 1.63 | 93.0 / 5.15 / 1.75 |
| F5 lighting | 89.4 / 6.05 / 2.00 | 89.6 / 6.39 / 2.02 |

![Selective false non-match rate of A3 by system and population (95 % intervals).](assets/fig_population_shift.png){#fig:shift}

The lesson is that a statistical guarantee is a guarantee about the population used for
calibration. Before trusting the error target for a new camera or population, the thresholds must
be re-calibrated on data from that population.

## 9.10 The Low-Quality Tail

@fig:m1 shows how the frozen M1 score behaves as quality decreases: below the calibrated range
the score rises towards 1 even for dissimilar faces. On development data, the A3 false match rate
among pairs below the support floor was 18–42 % per system, against 1.1–1.5 % elsewhere, and on
the untouched replication set it was 22.97 % (289 of 1,258 accepted different-person pairs)
against 1.50 % above the floor (@fig:lowq). These pairs are only about 1 % of the data, so the
aggregate rates hid the problem. This finding motivated the support guard of Section 5.7.

![Extrapolation of the frozen M1 score below the calibrated quality range.](assets/fig_m1.png){#fig:m1}

![False match rate below and above the calibrated quality range.](assets/fig_lowq.png){#fig:lowq}

## 9.11 Discussion

The results support three conclusions. First, quality-aware abstention works: the A3 rule kept
both error rates within the 5 % target on unseen occluders of the calibration-like population
while deciding about nine in ten pairs, and in the application it turns a probable error on a
dark, masked pair into an honest "not sure". Second, deciding with learned visibility is not
reliable, because a learned estimator only knows the occluders it has seen; quality is more
robust, so VEIL-X decides with quality. Third, calibration is population-specific: the error
target must be re-checked for a new population, and the score must not be trusted below its
quality range, which the support guard enforces.

# 10. Advantages and Limitations

## 10.1 Advantages

- **Honest outputs.** VEIL-X answers ABSTAIN instead of guessing when the evidence is weak.
- **Explainable.** Every decision comes with the exact numbers, the rule and the threshold
  comparison that produced it, and with what would change it.
- **Strong pretrained models.** State-of-the-art embeddings without costly retraining.
- **Private.** Everything runs locally; uploads are deleted; there is no cloud and no database.
- **Fast enough for interactive use.** About 0.2 s per photograph pair on a laptop GPU and about
  0.5 s on a CPU.
- **Safe defaults.** Multiple faces are refused, unsuitable photographs are not silently cropped,
  and very low-quality inputs abstain.
- **Easy to install and use.** A one-click launcher and a three-step interface with examples.
- **Well tested and reproducible.** 430 automated tests, exact parity with the research code,
  fingerprinted model files, an end-to-end browser test of every control and documented
  reproduction steps.

## 10.2 Limitations

- **Synthetic evaluation.** The main evaluation used synthetic faces and synthetic occlusions;
  real-world occlusions such as real masks and scarves in natural photographs were not evaluated.
- **Population-dependent calibration.** The 5 % target was met on the calibration-like population
  but not on a harder population, as confirmed by two independent tests.
- **Low-quality inputs.** The research score is unreliable below its quality range; the support
  guard reduces this risk but is not a proven universal fix.
- **Visibility maps** can miss occluders that look like skin or faces, and were trained on
  synthetic occluders only.
- **No single calibration.** Five calibration systems exist, one per left-out occluder family;
  none is a universal or production calibration.
- **Real-photograph results** may be optimistic, because public-figure photographs may overlap
  with the training data of the models.
- **No certification.** The system is not certified and is not suitable for identity
  verification, access control, surveillance or any decision affecting people.

# 11. Conclusion and Future Work

## 11.1 Conclusion

This project designed, implemented and evaluated VEIL-X, a complete local face verification
application with quality assessment and intelligent abstention. It combines frozen
state-of-the-art face detection and recognition models with a feature-norm quality signal, a
fused decision score, and a quality-stratified three-way rule calibrated with a statistical error
target. Every result is explained step by step with the exact numbers and threshold that produced
it. The application runs privately on a laptop, reproduces the research pipeline exactly, responds
in a fraction of a second and is backed by an extensive automated and end-to-end test suite.

The evaluation showed where the approach works and where it does not. Quality-based abstention
kept errors within target on unseen occluders of the calibration population, while a learned
visibility signal did not. A shift to a harder population broke the error target for every rule,
and the fused score extrapolated badly at very low quality, which led to the support guard. The
main lesson is that saying "I don't know" is valuable, but a confidence threshold is only as
reliable as the data it was calibrated on. By reporting these limits openly and building
safeguards for them, VEIL-X is a credible engineering demonstration with clearly stated limits.

## 11.2 Future Enhancements

1. Re-calibrate the thresholds on images from the intended user population and camera.
2. Evaluate on real, naturally occluded photographs, for example the IJB-C benchmark, after
   obtaining access.
3. Use a fused score that cannot extrapolate below its quality range, so that the support guard
   is no longer needed.
4. Run face detection on the GPU to reduce latency; this requires a compatible CUDA runtime and a
   new parity test.
5. Improve multi-face handling with clickable face thumbnails instead of face numbers.
6. Package the application as a stand-alone installer for Windows and Linux.

# References

1. Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.
2. Angelopoulos, A. N., Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
3. Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., Lei, L. (2021). Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.
4. Bae, G., de La Gorce, M., Baltrušaitis, T., Hewitt, C., Chen, D., Valentin, J., Cipolla, R., Shen, J. (2023). DigiFace-1M: 1 million digital face images for face recognition. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
5. Bates, S., Angelopoulos, A., Lei, L., Malik, J., Jordan, M. I. (2021). Distribution-free, risk-controlling prediction sets. Journal of the ACM, 68(6).
6. Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1), 41–46.
7. Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
8. Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
9. Erakın, M. E., Demir, U., Ekenel, H. K. (2021). On recognizing occluded faces in the wild. International Conference of the Biometrics Special Interest Group (BIOSIG).
10. Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. Advances in Neural Information Processing Systems (NeurIPS).
11. Guo, J., Deng, J., Lattas, A., Zafeiriou, S. (2022). Sample and computation redistribution for efficient face detection. International Conference on Learning Representations (ICLR).
12. Huang, G. B., Ramesh, M., Berg, T., Learned-Miller, E. (2007). Labeled Faces in the Wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst.
13. Kim, M., Jain, A. K., Liu, X. (2022). AdaFace: Quality adaptive margin for face recognition. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
14. Lee, K., Lee, K., Lee, H., Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in Neural Information Processing Systems (NeurIPS).
15. Maze, B., Adams, J., Duncan, J. A., Kalka, N., Miller, T., Otto, C., Jain, A. K., Niggel, W. T., Anderson, J., Cheney, J., Grother, P. (2018). IARPA Janus Benchmark – C: Face dataset and protocol. International Conference on Biometrics (ICB).
16. Meng, Q., Zhao, S., Huang, Z., Zhou, F. (2021). MagFace: A universal representation for face recognition and quality assessment. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
17. Ngan, M., Grother, P., Hanaoka, K. (2020). Ongoing Face Recognition Vendor Test (FRVT) Part 6A: Face recognition accuracy with masks using pre-COVID-19 algorithms. NISTIR 8311, National Institute of Standards and Technology.
18. Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., Lawrence, N. D. (Eds.) (2009). Dataset Shift in Machine Learning. MIT Press.
19. Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. Advances in Neural Information Processing Systems (NeurIPS).
20. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI).
21. Song, L., Gong, D., Li, Z., Liu, C., Liu, W. (2019). Occlusion robust face recognition based on mask learning with pairwise differential Siamese network. IEEE/CVF International Conference on Computer Vision (ICCV).
22. Terhörst, P., Kolf, J. N., Damer, N., Kirchbuchner, F., Kuijper, A. (2020). SER-FIQ: Unsupervised estimation of face image quality based on stochastic embedding robustness. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
23. Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A. (2019). Conformal prediction under covariate shift. Advances in Neural Information Processing Systems (NeurIPS).
24. Traub, J., Bungert, T. J., Lüth, C. T., Baumgartner, M., Maier-Hein, K. H., Maier-Hein, L., Jaeger, P. F. (2024). Overcoming common flaws in the evaluation of selective classification systems. Advances in Neural Information Processing Systems (NeurIPS).
25. Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
26. Zhu, Z., Huang, G., Deng, J., Ye, Y., Huang, J., Chen, X., Zhu, J., Yang, T., Lu, J., Du, D., Zhou, J. (2021). WebFace260M: A benchmark unveiling the power of million-scale deep face recognition. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

**Software and model releases:** CVLface (github.com/mk-minchul/CVLface), the source of the
AdaFace and ArcFace IR-101 WebFace4M checkpoints; InsightFace (github.com/deepinsight/insightface),
the source of the SCRFD `buffalo_l` detector.

The complete research record, including the detailed research paper, all experiments and the
claim-to-evidence audit, is in the project repository: `docs/paper/veilx_paper.md`,
`docs/phase6_audit.md` and `docs/phase6_results.md`.

# Appendix A: Installation and User Guide

## A.1 One-Click Installation on Windows

1. Download the project as a ZIP file from GitHub and extract it, for example to the desktop.
2. Place the file `VEIL-X_runtime_pack.zip` (frozen configuration, detector, visibility weights and
   example images) next to `run_veilx.bat`.
3. Double-click `run_veilx.bat`. The first run takes 5–15 minutes: it unpacks the runtime pack,
   installs Python 3.12 and all libraries in the private folder `.venv-app`, downloads the two
   recognition models (about 520 MB) and verifies their fingerprints.
4. The browser opens at `http://127.0.0.1:7860`. Keep the black console window open while using
   the application. Later runs start within seconds.

## A.2 Manual Start

```
$env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"
.venv-gpu\Scripts\python.exe scripts\run_app.py
.venv-gpu\Scripts\python.exe scripts\run_app.py --device cpu --port 7861
```

## A.3 Using the Application

1. Upload two face photographs into **Image A** and **Image B**, or click an example button.
2. Optionally open **Advanced options** to select a face, allow the portrait fallback, switch the
   heat maps or choose a calibration system.
3. Tick both confirmations, then press **Compare faces**.
4. Read the result card and the **Why this result?** steps. Open *Faces used*, *Where might the
   faces be covered?* or *Technical details* for more information.
5. Press **Clear** to remove the images and the result.

## A.4 Troubleshooting

Table: Troubleshooting {#tab:trouble}
| Problem | Solution |
|---|---|
| "The VEIL-X runtime pack was not found" | Put `VEIL-X_runtime_pack.zip` next to `run_veilx.bat` |
| Port 7860 already in use | Run `run_veilx.bat --port 7861` and open `http://127.0.0.1:7861` |
| No NVIDIA GPU, or GPU errors | Run `run_veilx.bat --device cpu` |
| Compare faces is greyed out | Tick both confirmation boxes |
| "2 faces found" | Enter the face number shown on the detection image in Advanced options |
| "no usable face" | Use a clear, front-facing photograph, or allow the portrait fallback for loosely cropped portraits |
| "Image A not accepted" | The file is corrupt, too large or not a JPEG, PNG, BMP or WebP image |
| Setup interrupted | Delete the `.venv-app` folder and run `run_veilx.bat` again |

# Appendix B: Frozen Configuration

Table: Face detection and input settings {#tab:config}
| Setting | Value |
|---|---|
| Detector | SCRFD det_10g (InsightFace `buffalo_l` v0.7), ONNX Runtime on the CPU |
| Detection / NMS thresholds | 0.5 / 0.4 |
| Minimum accepted detection score | 0.6 |
| Maximum landmark residual | 7.34 pixels |
| Fallback | Near-square images only (aspect ratio 0.8–1.25), resized to 250×250, fixed transform |
| Accepted formats | JPEG, PNG, BMP, WebP |
| File size limit | 20,000,000 bytes |
| Image side limits | Minimum 64 pixels, maximum 4,096 pixels |
| Recognition models | AdaFace IR-101 (decision) and ArcFace IR-101 (second opinion), WebFace4M |
| Precision | fp16 automatic mixed precision on the GPU; fp32 on the CPU |
| Decision rule | A3 quality-stratified rule, primary target (sFNMR ≤ 5 %, sFMR ≤ 5 %, δ = 0.05) |
| Default calibration system | F1_mask |
| Server | 127.0.0.1 only; no share link; analytics off |

Table: M1 coefficients of all five calibration systems (standardised features) {#tab:m1all}
| System | Intercept | Similarity | Quality | Product s·q |
|---|---|---|---|---|
| F1_mask | 1.1259 | 4.0634 | −2.4766 | 5.0348 |
| F2_sunglasses | 1.0191 | 3.3456 | −2.3820 | 4.8983 |
| F3_hand_object | 1.1509 | 3.5891 | −2.3694 | 5.0678 |
| F4_random_patch | 1.1274 | 5.3346 | −1.9674 | 3.5773 |
| F5_extreme_lighting | 1.0988 | 3.3114 | −2.3064 | 5.0272 |
