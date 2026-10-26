# VEIL-X — Dataset Strategy and Licensing Audit

> Labels per `research_landscape.md` §0. **This is not legal advice.** Every licence below must be
> re-read from the owner's page at download time, and the date and version recorded in
> `data/LICENSES.md`.

> **Phase 0 update (2026-10-26):** the up-to-date status of every dataset, including the exact
> assigned Kaggle dataset, is in **`dataset_feasibility.md`** and `configs/datasets.yaml`.
> Approved: DigiFace-1M subset for training (licence confirmed R-UDA); CelebA optional; ROF
> *conditional*; MLFW/CPLFW/XQLFW subject to licence checks. Sections below are amended
> where decisions changed.

## 1. The assigned Kaggle "human-face-images" dataset

**Resolved 2026-10-26:** the assigned dataset is
`https://www.kaggle.com/datasets/trainingdatapro/human-face-images`. It returns **HTTP 404**
(control Kaggle pages return 200). It is therefore **blocked** until its owner page, licence
and identity labelling can be confirmed. Details: `dataset_feasibility.md` §1. The
candidate table below is the superseded pre-assignment search, kept for traceability.

**Superseded:** the exact dataset could not be identified. Several Kaggle datasets match the
name. Kaggle pages did not render for automated fetching, so licence fields could only be read
from search snippets or third-party mirrors.

| Candidate | What search results say | Identity labels? | Licence (as reported) | Status |
|---|---|---|---|---|
| [kaustubhdhote/human-faces-dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset) | ~9.6k images: 5,000 real + 4,630 AI-generated | **No** (real vs AI folders only) | CC BY-SA 4.0 (third-party mirror, hyper.ai) | [U]: licence and provenance of the "real" images unverified |
| [jayeshsonawane/all-images](https://www.kaggle.com/datasets/jayeshsonawane/all-images) ("Human Face Images Dataset") | ~42.5k face images | Not stated | Not retrieved | [U] |
| Others (gender, emotion sets) | — | No | Various | Not relevant |

**Suitability assessment [D]:**
- Face **verification** needs **identity labels** to form genuine and impostor pairs. Without
  identity labels, the dataset **cannot** be used for verification training, calibration or test.
  Pairing two augmentations of the same image is *not* identity verification, and it would create
  trivially easy, leaky "genuine" pairs.
- Impostor-only use is also unsafe: without labels, two images of the same person can silently
  become an "impostor" pair.
- **Possible legitimate uses**, only if the licence and provenance check out:
  1. Unlabelled source faces for training the **visibility segmenter**. The synthetic occlusion
     overlay provides the mask label, so no identity label is needed.
  2. Demo stress images, but only images the licence permits to display.
- **CC BY-SA 4.0 note [U]:** ShareAlike may bind derived *datasets* (e.g. occluded copies) that
  are redistributed. VEIL-X does not plan to redistribute data, which avoids that issue.
- AI-generated faces in the dataset are a **domain confound** for a segmenter. If used, keep real
  and AI-generated images in separately tracked subsets.

**Decision needed from you:** the exact URL of the assigned dataset, and confirmation of its
licence page. Until then it is excluded from every protocol.

---

## 2. Identity-labelled datasets and occlusion benchmarks

| Dataset | Content | IDs / images | Occlusion | Access & licence (as found) | Proposed VEIL-X role | Risk |
|---|---|---|---|---|---|---|
| **LFW** | Web news photos | 5,749 / 13,233 [U: counts from memory] | Little. The official disclaimer says strong occlusion and extreme pose are not a major part of LFW ([source](https://complexity.cecs.ucf.edu/lfw-labeled-faces-in-the-wild)) | Public download; research use; the site warns against using it to vet commercial systems | Clean reference; **source images for synthetic occlusion test sets** | Celebrity identities likely overlap with backbone pretraining data [H] |
| **CALFW / CPLFW** | LFW identities, cross-age / cross-pose pairs | LFW-scale | Pose self-occlusion (CPLFW) | Research use from Deng's lab [U: licence page not retrieved] | Pose stress test (CPLFW) | Same overlap risk |
| **MLFW** (P32) | Masked CALFW | 6,000 pairs [U] | Synthetic masks with many templates | [whdeng.cn/mlfw](http://whdeng.cn/mlfw), research use [U] | Mask-family test | Synthetic masks |
| **XQLFW** (P33) | LFW with cross-quality pairs | LFW-scale | Blur / low resolution | [Project page](https://martlgap.github.io/xqlfw) [U: licence] | Quality-shift control test | — |
| **ROF** (P31) | **Real** masks and sunglasses | 180 IDs / 5,559 images (3,195 neutral, 1,686 sunglasses, 678 masked) | **Real**, upper and lower face | Repo licence MIT ([repo](https://github.com/ekremerakin/RealWorldOccludedFaces)); images are web-scraped, so **copyright of the images is not granted by the MIT licence [H]** | **Primary real-occlusion final test**, now **conditional** on ethics/legal sign-off (D-007). Images ship as pickles: load only with a restricted unpickler in an isolated process | Small. 70 IDs have both occlusion types. Wide CIs are expected |
| **CelebA** | Celebrities, 40 attributes | 10,177 / 202,599 | Natural accessories | Non-commercial research only; no redistribution; **identity labels "released upon request"** ([official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) | **Candidate training/val/cal identities** for the heads and the segmenter, if identity labels are obtained officially | Must request IDs. Overlap with backbone pretraining likely |
| **DigiFace-1M** (P34) | Rendered synthetic faces | 110K IDs / 1.22M (10K×72 + 100K×5) | Rendered accessories | **Non-commercial research only**: R-UDA v1.0, confirmed from the LICENSE file 2026-10-26 ([repo](https://github.com/microsoft/DigiFace1M)) | **Approved training/val/cal source (D-005)**; overlap-free identities for a secondary test | Synthetic-to-real domain gap |
| **FaceOcc** (P14) | CelebA-HQ + internet images, occlusion masks | ~ [U] | Real occluders (hands, masks, glasses, mic…) with pixel masks | [U: licence not retrieved]; CelebA-HQ terms likely inherited | **Secondary** segmenter training. **Excluded from the OOD protocol**, because it contains all occluder families (see §4) | Leaks held-out families |
| **CFP-FP / AgeDB** | Pose / age | — | Pose | Request-based [U] | Optional extra tests | Access delays |
| **IJB-B/C, IJB-S** | NIST templates | — | — | **Restricted. Assume unavailable** (per brief) | **Not used** | — |
| **MS1M / MS1MV2, VGGFace2** | Large web scrapes | — | — | **Withdrawn** by their creators; still circulates unofficially (reporting: [Vice](https://www.vice.com/en/article/microsoft-deleted-a-facial-recognition-database-but-its-not-dead/), [exposing.ai](https://exposing.ai/about/news/)) | **Not used for any VEIL-X training** | Ethical and legal |
| **WebFace260M / 42M** | Large web scrape | 2M IDs / 42M (cleaned) | — | Application-based, research use [U] | **Not needed** (backbones are frozen) | — |

### 2.1 Pretrained weights (also licensed artefacts)

| Model | Training data | Licence (as found) | Role |
|---|---|---|---|
| AdaFace IR101 (WebFace4M), e.g. [HF: minchul/cvlface_adaface_ir101_webface4m](https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m) | WebFace4M | Code MIT; **WebFace-trained models: no commercial use** (per repo/model card) | **Primary frozen backbone [D]** |
| ArcFace IResNet (insightface `arcface_torch` or `buffalo_l` w600k_r50) | MS1MV3 / WebFace600K [U: exact sets per checkpoint] | insightface code MIT; **all pretrained models non-commercial research only** | Secondary backbone (robustness of conclusions) |
| CR-FIQA weights | Per paper [U] | [U] | Quality baseline B3 |
| Face detector / aligner (SCRFD/RetinaFace via insightface, or MediaPipe) | — | insightface models non-commercial; MediaPipe Apache-2.0 [U: re-check] | Detection + 5-point alignment |

**Consequence [D]:** every VEIL-X artefact (code may be MIT, but the trained heads and the demo)
inherits **non-commercial research-only** constraints from its weights and data. The README and
the demo must say so.

---

## 3. Recommended data plan [D]

| Role | Source | Notes |
|---|---|---|
| Visibility segmenter: train / val | Aligned faces from **training-split identities only** (CelebA train IDs or DigiFace train IDs; optionally the Kaggle set if cleared) + **synthetic overlays from seen occluder families only** | Masks come free from the synthesis |
| Fusion/abstention head: train | CelebA (train IDs) and/or DigiFace (train IDs), with seen-family synthetic occlusions | Small model, so modest data is fine |
| Model selection (val) | Disjoint val IDs, seen families | Early stopping, hyperparameters |
| Calibration (cal) | Disjoint cal IDs, seen families | LTT thresholds, temperature/Platt. **Never touched by training** |
| **Test-ID** | Disjoint test IDs, seen families | In-distribution reference |
| **Test-OOD-synth** | Disjoint test IDs, **held-out families** (leave-one-family-out) | Main OOD test |
| **Test-real** | **ROF** (masks, sunglasses); MLFW masks; CPLFW pose; XQLFW quality | Final real-world check. ROF is never used for tuning |

If CelebA identity labels cannot be obtained: use DigiFace-1M for training, val and cal, and
evaluate only on the LFW-derived sets and ROF. Report the synthetic-to-real domain gap as a
limitation.

**Approved plan (2026-10-26, D-005):** DigiFace-1M is the training, val and cal source.
CelebA is optional, not a dependency. Recommended subset: 72-img part P1 + 5-img part P1
(~5.9 GiB, pending download go-ahead). The 5-img part supplies the identity count needed for
identity-disjoint calibration (`dataset_feasibility.md` §3). The Kaggle dataset has **no
role** (blocked).

---

## 4. Occlusion taxonomy and synthesis

Families are defined **before** any experiment. Every synthetic occlusion is parameterised by
(family, asset id, geometry, seed) so that it is exactly reproducible.

**Approved families (D-006, 2026-10-26)** replace the earlier F1–F6 draft. IDs match
`configs/research_protocol.yaml`:

| Family | Examples | Synthesis | Real counterpart | Interpretation |
|---|---|---|---|---|
| `F1_mask` | Surgical/cloth face masks | Landmark-anchored polygons + textures | ROF masked, MLFW | Physical occlusion |
| `F2_sunglasses` | Sunglasses | Landmark-anchored shapes | ROF sunglasses | Physical occlusion |
| `F3_hand_object` | Hands, phones, cups, microphones | Asset cut-outs at random positions | (no identity-labelled real set found) | Physical occlusion |
| `F4_random_patch` | Rectangles, blobs | Procedural | — | Physical occlusion |
| `F5_extreme_lighting` | Hard cast shadows, saturation, underexposure | Procedural illumination masks | (none verified) | **Non-physical**: reported and interpreted separately |

Pose (CPLFW) is a separate benchmark, **not** a challenge family. Scarves/eye masks are no
longer named families. If used at all, they must be assigned to exactly one family before
any experiment.

**Asset licensing:** occluder textures and cut-outs must come from CC0 sources or be generated
procedurally. Each asset's source URL and licence is logged. [D]

**Held-out protocol:** leave-one-family-out over F1–F5. The schema enforces the full isolation
scope (`protocol_design.md` §5). For each held-out family F_k:
- The segmenter, the head and the calibration never see F_k. That includes its assets *and* its
  geometry generator.
- FaceOcc is excluded from this protocol, because its real masks span F1–F3. It may be used only
  in a separately labelled "all-families segmenter" condition, which makes **no OOD claim**.
- **Caveat [H]:** the frozen backbone was pretrained on web data that certainly contains masks,
  sunglasses and hands. "Unseen" means unseen by VEIL-X's trained components and calibration,
  **not** by the backbone. Every write-up must state this.

---

## 5. Leakage-resistant identity-disjoint splits [D]

1. **Unit of splitting = identity.** Assignment uses
   `split = H(dataset_name + ":" + identity_id) mod 100` into fixed bands: train 0–59, val 60–69,
   cal 70–79, test 80–99. It is deterministic, seed-free and stable when data is added.
2. **Cross-dataset identity collisions.** LFW-derived sets (LFW, CALFW, CPLFW, MLFW, XQLFW) share
   identities. They are all **test-only**, so they never touch train/val/cal. CelebA and LFW may
   share celebrities. Name-match CelebA identity names where available [U: CelebA IDs are
   anonymised integers]. Otherwise run the near-duplicate check (step 3) and drop any CelebA
   identity whose images match an LFW test image.
3. **Near-duplicate guard.** Perceptual hash + backbone cosine. **Superseded (D-013):** no
   arbitrary fixed threshold such as "0.9". The threshold is set only by the validated
   procedure in `protocol_design.md` §8 (known-duplicate positives and same-identity hard
   negatives from non-test data). The schema refuses a threshold without a validation record.
   Flagged items are removed from the *non-test* side only.
4. **Pairs are built within a split only.** Pair lists are generated once, given a seed and
   frozen as versioned files with SHA-256 checksums. Genuine pairs must come from different
   source images. Calibration uses an identity-disjoint matching (D-009,
   `protocol_design.md` §4).
5. **Occlusion variants inherit their source image's split.** Synthetic variants of one image
   never cross splits.
6. **Test lock.** Test pair lists and their labels live in a separate directory. A CI check fails
   if any training/tuning script reads it. The test is run **once per pre-registered
   configuration**. Each run is logged with the git hash in `results/test_runs.jsonl`.
7. **Backbone pretraining overlap** cannot be removed for public celebrity data. Mitigations:
   relative comparisons only (same backbone for all arms); DigiFace-1M test identities as an
   overlap-free check; an explicit limitation statement.

---

## 6. Data governance and biometric privacy

- Face images and embeddings are **biometric personal data** in many jurisdictions (e.g. GDPR
  special-category data; India's Digital Personal Data Protection Act 2023 governs personal data
  [U: obligations specific to biometric data not verified]; Illinois BIPA [U]). Get advice before
  any use beyond personal academic research.
- Store datasets outside the git repository. `.gitignore` covers `data/`, `cache/` and
  `results/raw/`. **No face images and no embeddings are committed.** Small test fixtures are
  procedurally generated non-face images, or licence-cleared synthetic faces.
- Embedding caches are deletable with one command and contain no names.
- Do not re-host or redistribute any dataset. Cite every source.
- **Out of scope by design:** 1:N identification, gallery building, web/social-media scraping,
  identification of people in the wild, video surveillance, demographic inference.
