# VEIL-X — Dataset and Licensing Feasibility Report

> Checked 2026-10-26. Machine-readable registry: `configs/datasets.yaml` (schema-validated;
> only `confirmed` entries are returned by `DatasetRegistry.usable_for`).
> **Nothing has been downloaded.** Sizes come from HTTP `HEAD` requests or repository
> metadata. **Not legal advice.** Each licence is re-read at download time.

## 0. Phase 1b status (2026-10-26)

- **DigiFace-1M approved subset downloaded and verified:** 72-img P1 + 5-img P1, 5.90 GiB of
  archives. 310,665 images and 35,333 identities. Sizes match, CRC passes, every image
  decodes, and there are no anomalies or exact duplicates. Full report:
  `data_integrity_report.md`.
- Additional parts were **not downloaded**. Their benefit is quantified for approval in
  `decision_log.md` (pending item 1, CALC-002).
- ROF, MLFW, CPLFW, XQLFW and LFW: **not downloaded** (not authorised or unresolved).
  Kaggle: still HTTP 404.

## 1. The assigned Kaggle dataset: `trainingdatapro/human-face-images`

| Check | Finding | Evidence |
|---|---|---|
| Page reachable? | **No.** `https://www.kaggle.com/datasets/trainingdatapro/human-face-images` returned **HTTP 404** | `curl`, 2026-10-26. Control pages returned 200: `trainingdatapro/keypoints`, `kaustubhdhote/human-faces-dataset`, and the `trainingdatapro` profile |
| Public API | HTTP 403 (requires Kaggle credentials; none were used or requested) | `curl` on `/api/v1/datasets/view/...` |
| Archived copy | Not retrievable from this environment (Wayback rate-limited / blocked) | — |
| Licence | **Unknown.** Search snippets show *other* TrainingDataPro Kaggle datasets under CC BY-NC-ND 4.0; this does **not** establish this dataset's terms | Web search 2026-10-26 |
| Identity labels | **Unknown** | — |
| Images per identity | **Unknown** | — |
| Download size | **Unknown** | — |
| Provider context | TrainingDataPro publishes small free samples of commercial ("data market") datasets (provider pages on Kaggle/HF) [V-partial: general provider description, not this item] | HF org page |

**Verdict: BLOCKED** (`access_status: blocked`, `roles: [none]`).
- It cannot be assessed while the owner page is unreachable.
- Even if restored, it **cannot support face verification** unless it carries reliable identity
  labels with ≥ 2 images per identity. A sample-style selfie set without labels fails this.
- If CC BY-NC-ND 4.0 applies, the **NoDerivatives** term prohibits sharing adapted material,
  e.g. occluded variants. Private research use of adaptations would need checking against the
  actual licence text.
- **Possible causes of the 404 [H]:** the dataset was removed, renamed or made private. If you
  can see it while logged in, export its licence text and metadata (not the images) into
  `docs/` and we will re-assess.

## 2. Feasibility matrix

| Dataset | Access status | Licence (source) | Identity labels | IDs / images | Images per ID | Download | Role | Contamination / validity risks |
|---|---|---|---|---|---|---|---|---|
| **DigiFace-1M** | **confirmed** | **R-UDA v1.0**: non-commercial research; redistribution only with attribution + R-UDA binding ([LICENSE](https://raw.githubusercontent.com/microsoft/DigiFace1M/main/LICENSE)) | Yes (folder per subject) | 110K / 1.22M | 72 (IDs 0–9,999; 5 parts × 2,000) or 5 (IDs 100,000–199,998; 3 parts × ~33,333) | 72-img parts **2.73–2.74 GiB** each; 5-img parts **3.16–3.17 GiB** each; **~23.2 GiB** total (HTTP HEAD) | train / val / cal / test-synth / visibility training | Synthetic domain gap; README warns renders may resemble real people; whether any chosen backbone saw DigiFace is **unverified** |
| **ROF** | **conditional** | Repo MIT (© M. E. Erakin 2021). Images collected via Google Image Search: **third-party copyright not granted by MIT** ([repo](https://github.com/ekremerakin/RealWorldOccludedFaces)) | Yes | 180 / 5,559 | All 180 neutral; 70 masked + sunglasses; 110 sunglasses only (3,195 neutral / 1,686 sunglasses / 678 masked) | Repo ~4.8 MB (GitHub API); images inside **pickle** files | test-real only | VGGFace2-like collection, so likely overlap with backbone pretraining identities; small, wide CIs; **pickle = code-execution risk** |
| **MLFW** | unresolved | Not retrievable (project page rendered empty) | Yes | 6,000-pair protocol reported [U] | Derived from CALFW | Unknown | test-real only | LFW identities; synthetic masks |
| **CPLFW** | unresolved | **No licence text** on project page | Yes | 3,000 positive + negative pairs (same gender/race), 10 folds, `pairs_CPLFW.txt` (project page) | — | Unknown | test-real only (pose stress, not a challenge family) | LFW identities |
| **XQLFW** | unresolved | **No explicit licence** on project page; check `github.com/martlgap/xqlfw` | Yes | 3,743 IDs / 7,263 images; 6,000 pairs (3,000/3,000) (project page) | — | Unknown | test-real only (quality control, not occlusion) | LFW-derived; synthetic degradation |
| LFW | unresolved | Official host unreachable from this environment; disclaimer known via mirror | Yes | [U] | — | — | none (base of MLFW/CPLFW/XQLFW) | Celebrity identities in pretraining data |
| CelebA | conditional (optional) | Non-commercial research; no redistribution; **identity labels on request** ([official](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) | On request | 10,177 / 202,599 | — | — | none (not a dependency) | Celebrity identities in pretraining data |
| FaceOcc | unresolved | Not retrieved | No | — | — | — | none | Would leak held-out families into visibility training |
| MS1M / VGGFace2 | blocked | Withdrawn by creators | — | — | — | — | none | Ethical/legal |
| IJB-B/C/S | blocked | Restricted (brief) | — | — | — | — | none | — |

### 2.1 Pretrained weights (also licensed artefacts; **not downloaded**)
| Artefact | Terms (as found) | Status |
|---|---|---|
| AdaFace IR101 WebFace4M (`minchul/cvlface_adaface_ir101_webface4m`) | Code MIT; WebFace-trained weights: no commercial use | Approved as primary; download needs a go-ahead |
| ArcFace (InsightFace `buffalo_l` w600k_r50 or `arcface_torch`) | InsightFace pretrained models: non-commercial research only | Approved as secondary; exact checkpoint to be chosen at download |
| InsightFace detector (SCRFD/RetinaFace) | Non-commercial research only | Approved |

## 3. Statistical capacity per source (from `scripts/feasibility_report.py`)

Cal band = 10% of identities. With identity-disjoint matching at a 50% genuine mix, a
calibration set of N identities supplies about N/3 genuine and N/3 impostor *independent*
pairs.

| Source subset | Cal IDs | Indep. impostor pairs | Best certifiable FMR (0 FM, 95%) | Selective-error bound if 2% observed |
|---|---|---|---|---|
| 72-img part P1 (2,000 IDs; 2.74 GiB) | 200 | 66 | 4.44% | 4.69% |
| 72-img P1–P5 (10,000 IDs; 13.7 GiB) | 1,000 | 333 | 0.90% | 3.09% |
| 5-img part P1 (~33,333 IDs; 3.17 GiB) | 3,333 | 1,111 | 0.27% | 2.54% |
| 5-img P1–P3 (~100,000 IDs; 9.5 GiB) | 10,000 | 3,333 | 0.090% | 2.30% |

**Recommended "manageable subset" (pending your go-ahead; ~5.9 GiB):**
- **72-img part P1** (2,000 IDs): rich within-identity variation for genuine pairs, the fusion
  head and visibility training.
- **5-img part P1** (~33,333 IDs): calibration and test capacity. It certifies the 5% selective
  error target and **FMR at 1%**. **FMR at 0.1% cannot be certified** with it. FAR = 0.1% is
  then *estimated* (TAR@FAR, realised FAR), not certified. Adding the other two 5-img parts
  (+6.3 GiB) would allow 0.1% certification only if zero false matches are observed.

## 4. Unresolved items and owners

| Item | Needed for | Action |
|---|---|---|
| Kaggle assigned dataset licence and labels | Brief requirement | You: check while logged in; share licence text/metadata only |
| ROF ethics/legal sign-off | Real-occlusion test (C3 on real data) | You or supervisor: confirm academic use of web-collected images is acceptable |
| MLFW / CPLFW / XQLFW / LFW licence text | Additional real tests | Manual browser check; record in `configs/datasets.yaml` |
| Model-weight checkpoints | Phase 2 | Separate go-ahead to download (~hundreds of MB [U]) |
| DigiFace download | Phase 1 | Separate go-ahead (~5.9 GiB recommended subset) |
