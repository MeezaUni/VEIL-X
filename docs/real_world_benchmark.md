# Real-World Benchmark Status (Phase 2)

> Labels: **MEASURED** (observed by a command in this project), **REPORTED** (stated by a
> third party, not verified here).
>
> **Update 2026-10-27 (D-043).** LFW was acquired after the project owner reported
> their professor's academic approval. §0 records the authorisation, provenance and
> verification; results are in §7. §§1–6 below are the original blocked-status record, kept
> for traceability.

## 0. Acquisition under user-reported academic authorisation (D-043)

| Item | Record |
|---|---|
| Authorisation | The project owner reports that their college professor approved using the Kaggle LFW dataset for this academic project. It is recorded as **user-reported academic authorisation**: it is **not** a licence from the LFW authors or the Kaggle uploader, and the uploader's redistribution rights remain unestablished (§2 facts unchanged) |
| Remaining restrictions found | No specific prohibition was found. The LFW creators state the dataset is for research, not for vetting commercial systems; the Kaggle licence field ("Other") contains no terms. Use: non-commercial academic evaluation; images, identity names, manifests and embeddings stay private; only aggregate metrics are published |
| Source | `kaggle datasets download jessicali9530/lfw-dataset` (Kaggle CLI 2.2.5, authenticated with the user's own token, which is not stored in the repository), dataset **version 4** (last updated 2018-05-17) |
| Archive | `lfw-dataset.zip`, 117,895,655 B, SHA-256 `fff0f15b1f848f0d834f8df58ca2e3273bda21e37bd9a895acf66de60521ba10` |
| Integrity (MEASURED) | 13,243 members, CRC-32 OK, 0 rejected (name pattern, traversal, type, size, ratio). Uncompressed 117,149,434 B, equal to Kaggle's recorded `totalBytes` |
| Content (MEASURED) | **13,233** JPEGs, all 250×250 RGB, decode OK; **5,749** identities; 0 exact pixel duplicates; deep-funneled variant |
| Official View 2 (MEASURED) | `pairs.csv`: **6,000** pairs = 10 folds × (300 matched + 300 mismatched); 4,281 identities, 7,701 images; every referenced image present. Used **unmodified** |
| View 1 (MEASURED) | DevTrain 2,200 pairs (1,100 matched), DevTest 1,000 (500 matched). **1,689 DevTrain identities also appear in View 2** (LFW's views share images), so anything set on DevTrain is flagged |
| Private outputs | `<data_root>/raw/lfw_kaggle_v4/`, `extracted/lfw_kaggle_v4/`, `manifests/lfw/` (images, pairs, detections, plan), embeddings |
| Public outputs | `reports/data/lfw_kaggle_v4_summary.json` (aggregates only), `reports/experiments/lfw_*.json` |

**Training-data overlap with LFW (investigated; not verifiable locally).** Both backbones are
trained on WebFace4M, a subset of WebFace260M.
- WebFace260M's celebrity name list is partly borrowed from MS1M (Zhu et al., CVPR 2021).
- Its authors report removing subjects whose feature centres are too similar to those of LFW-family
  benchmarks. Residual overlap therefore cannot be excluded and is not measurable here: the
  WebFace identity list is not available locally.
- For the related MS1MV2 training set, Wu et al. (2024, arXiv:2405.09403) found **2,009 LFW
  identities (46.9 %)** and **384 LFW images (5 %)** with a (near-)duplicate in the training set.
  They also showed that identity overlap inflates LFW-family accuracy.

**Consequence:** clean LFW accuracy of these backbones is likely **optimistic** and near
saturation. It says little about generalisation to unseen people. Relative comparisons
(clean vs derived occlusion, detector vs fixed crop, rule transfer) are less affected than
absolute accuracy, but are not immune.

## 1. Summary (original blocked-status record, 2026-10-26)

| Item | Status |
|---|---|
| Kaggle reference ([notebook `vividharawat/vggface-on-lfw`](https://www.kaggle.com/code/vividharawat/vggface-on-lfw/data)) | **Resolved**: its only data source is `jessicali9530/lfw-dataset` (MEASURED via Kaggle kernel metadata `datasetDataSources`) |
| That dataset | Full LFW, **deep-funneled** variant, version 4, 13,233 images / 5,749 people, official `pairs.csv` + DevTrain/DevTest lists (REPORTED by the Kaggle description; file list MEASURED via metadata API). It is **not** the small 7-class Kaggle LFW |
| Redistribution rights of the mirror | **Not established** → **not downloaded** |
| Original authorised distribution (`vis-www.cs.umass.edu/lfw`) | **Unreachable**: the authoritative UMass nameserver returns NXDOMAIN (MEASURED) |
| Kaggle credentials | Not configured on this machine (no `kaggle.json`, no CLI) (MEASURED) |
| LFW results in Phase 2 | **None.** All Phase 2 model results are DigiFace-only (synthetic faces) |

## 2. Evidence

**Notebook → dataset.** `GET https://www.kaggle.com/api/v1/kernels/pull?userName=vividharawat&kernelSlug=vggface-on-lfw`
(public metadata, no authentication) returns `datasetDataSources: ["jessicali9530/lfw-dataset"]`,
with no competition, kernel or model sources.

**Dataset metadata** (`/api/v1/datasets/view/jessicali9530/lfw-dataset`):

| Field | Value |
|---|---|
| Title | Labelled Faces in the Wild (LFW) Dataset |
| Owner | an individual uploader (not the LFW authors) |
| Version | 4 |
| Last updated | 2018-05-17 |
| Total size | 117,149,434 bytes |
| Licence field | "Other (specified in description)" |
| Files | `lfw-deepfunneled.zip`, `pairs.csv`, `people.csv`, `matchpairsDevTrain/DevTest.csv`, `mismatchpairsDevTrain/DevTest.csv`, `peopleDevTrain/DevTest.csv`, `lfw_allnames.csv`, `lfw_readme.csv` |

The description contains **no licence text**. It contains only acknowledgements to Huang et al.
(2007) and Huang, Mattar, Lee and Learned-Miller (NIPS 2012, deep funneling), and a pointer to
the original site for terms and errata.

**Original host.**
- `curl` against `http(s)://vis-www.cs.umass.edu/lfw/` fails with exit code 6 (cannot resolve).
- Google DNS-over-HTTPS returns `Status: 3` (NXDOMAIN), with the SOA from `cs.umass.edu` (serial 2026102601).

So the name has been removed at the source; this is not a local DNS fault. The LFW author's page
(`people.cs.umass.edu/~elm/`, HTTP 200) still links only to that host. No relocated official
distribution was found.

**Original terms.** No formal licence was located. The creators' homepage statement (quoted in
secondary sources) says LFW was published to help research advance face verification, not to
vet commercial algorithms. Nothing grants third parties the right to redistribute the images.
The Kaggle copy is therefore a re-upload whose redistribution rights cannot be established, and
whose content cannot be checked against an official checksum while the official host is offline.

**Decision (D-039).** Following the Phase 2 brief ("If the mirror's rights or provenance cannot be
established, do not download it… No unauthorized mirrors"), LFW is **blocked**. The
scikit-learn/figshare mirror is excluded for the same reason. Model evaluation continues on
DigiFace only, and no LFW numbers are reported or estimated.

## 3. Exact actions that would unblock LFW (user decision required)

Either of the following is sufficient. Both require an explicit PI decision recorded in
`docs/decision_log.md`.

1. **Use the Kaggle mirror under a documented PI risk acceptance.**
   - The PI records that the Kaggle copy (`jessicali9530/lfw-dataset`, version 4) is acceptable
     for non-commercial academic evaluation despite unestablished redistribution rights.
   - Install the official CLI in the project environment: `uv pip install --python .venv kaggle`.
   - Create an API token at kaggle.com → Settings → API → "Create New Token", and save it as
     `%USERPROFILE%\.kaggle\kaggle.json`. Never put it in the repository; `kaggle.json` is git-ignored.
   - VEIL-X would then download to the private data root only, verify 13,233 images / 5,749
     identity folders, check `pairs.csv` (10 folds × 300 + 300 = 6,000 pairs), and check the
     file list against the published LFW counts.
2. **Provide the original distribution.** If the PI holds an original LFW archive (for example
   from a course or a lab copy), or obtains one directly from the authors, place it under
   `VEILX_DATA_ROOT\raw\lfw\` with any checksum the authors supply.

## 4. LFW adapter (ready, untested on real data)

`veilx.data.lfw` parses the **official pairs protocol unmodified**:
- `pairs.txt`, which has a header line with the fold count and pair count per fold, then
  `name idx1 idx2` (matched) or `name1 idx1 name2 idx2` (mismatched) lines;
- the Kaggle `pairs.csv` rendering of the same lists.

Fold membership and pair order are preserved, and image paths are resolved to
`<root>/<name>/<name>_<idx:04d>.jpg`. Missing images raise an error; they are never skipped.
It is tested only on a synthetic fixture.

Planned LFW preprocessing (not executed): the deep-funneled images are 250×250 crops, not
112×112 ArcFace-template crops. Matching the backbones' training alignment therefore needs a
5-point detector (InsightFace SCRFD `det_10g`, non-commercial research licence).
- Detector-inclusive results would be reported **separately** from detector-free DigiFace results.
- Detection failures, multi-face images and landmark failures are counted, never dropped silently.
- The detector weights were **not** downloaded, because there is no data to apply them to.

## 5. Other real-world sets: assessed, not downloaded

| Dataset | Access (checked 2026-10-27) | Licence / terms found | Derived from LFW? | Phase 2 status |
|---|---|---|---|---|
| **RMFRD** (Real-World Masked Face, Wuhan Univ.) | GitHub README (HTTP 200); data via Google Drive / Baidu | No licence file or terms. Images "crawled from the website"; no consent statement | No (web-crawled) | Not usable: no licence, no consent provenance |
| **DFW** (Disguised Faces in the Wild, IIIT-Delhi) | Former resource pages return HTTP 404 | Historically licence-agreement based (REPORTED); not retrievable now | No | Not accessible |
| **CPLFW** / **CALFW** (BUPT, W. Deng) | Project pages HTTP 200 over http; https refused; licence text not retrievable | Research use (REPORTED); no licence text | Yes (LFW identities) | Unresolved; also requires LFW images |
| **MLFW** (masked CALFW) | Project page HTTP 200; content previously rendered empty | Not found | Yes | Unresolved |
| **XQLFW** (TUM) | Project page HTTP 200; GitHub `martlgap/xqlfw` is **MIT** (code) | No licence for the images; derived from LFW | Yes | Unresolved; image rights inherit LFW's |
| **ROF** | Not assessed further; requires separate approval (brief) | — | No | Not downloaded |

## 6. Consequence for Phase 2 claims

*(Original 2026-10-26 text, superseded by §7.)* All Phase 2 measurements came from DigiFace-1M,
a **synthetic** dataset, with **synthetic** occlusions.

## 7. LFW results (MEASURED; single logged run `79331a84f520`, commit `46e55b8`)

Source: `reports/experiments/lfw_final.json`. Design: D-044, fixed before any LFW score. Thresholds and LTT
rules come from the frozen DigiFace configuration `0e8a43fb…`; nothing was fitted on View 2. The full code
path was rehearsed on View-1 DevTest first; the rehearsal caught one bug (TAR@FAR with too few impostors),
fixed before the logged run. Intervals: identity-jackknife 95 %; Clopper–Pearson upper bounds where a rate
has zero events (pairs treated as independent, flagged).

### 7.1 Detection and alignment (SCRFD det_10g, CPU)

- **Detection.** 13,195 / 13,233 images (99.71 %) have a face. The 38 failures touch 25 View-2 images and
  42 of 6,000 pairs.
- **Multiple faces.** 17.5 % of images contain several faces; the centre-nearest face differed from the
  top-scoring one in 157 cases.
- **Isolated CPU latency:** p50 / p95 / p99 = 92 / 102 / 111 ms per image.
- **Alignment check.** The aligned mean face matches the ArcFace template (visual check, private).
- Figure: `reports/figures/lfw_detection_outcomes.svg`.

### 7.2 Clean LFW, official protocol (View 2, 10-fold, fold-internal thresholds)

| Preprocessing | AdaFace IR-101 accuracy | ArcFace IR-101 accuracy | AUC (Ada / Arc) | EER (Ada / Arc) | TAR@FAR 1 % (benchmark mode) |
|---|---|---|---|---|---|
| Detector, all 6,000 pairs (42 failures = forced non-match) | **99.58 % ± 0.26** | **99.57 % ± 0.24** | 0.9942 / 0.9946 | 0.77 % / 0.77 % | 99.23 % / 99.23 % |
| Detector, 42 failed pairs excluded (n = 5,958) | 99.83 % ± 0.21 | 99.82 % ± 0.20 | 0.9991 / 0.9996 | 0.30 % / 0.30 % | 99.73 % / 99.73 % |
| Fixed crop from View-1 DevTrain landmarks, no detector | 99.28 % ± 0.25 | 99.27 % ± 0.26 | 0.9976 / 0.9967 | 0.93 % / 1.07 % | 99.07 % / 98.93 % |

- **Detection failures.** They cost about 0.25 pp of official accuracy when counted as errors.
- **Detector-free alignment.** It costs a further 0.3 pp.
- **Fixed-crop geometry.** It comes from development landmarks, so the fixed-crop path still uses the
  detector indirectly, once.
- **Interpretation.** The figures are likely **optimistic** given probable identity overlap with
  WebFace4M (§0).

### 7.3 Transfer of DigiFace-fixed operating points to real photos (no LFW fitting)

At the threshold fixed on DigiFace calibration for FAR 1 % (cal_far_1pct):

| | AdaFace | ArcFace |
|---|---|---|
| FMR on 3,000 LFW impostors | **0 / 3,000** (≤ 0.10 %, Clopper–Pearson) | **0 / 3,000** (≤ 0.10 %) |
| FNMR (all pairs; failures = non-match) | 0.93 % [0.61, 1.43] | 1.00 % [0.66, 1.52] |
| FNMR (failed pairs excluded) | 0.44 % [0.24, 0.80] | 0.50 % [0.28, 0.91] |

- **The synthetic-calibrated threshold does not transfer nominally.** It is **conservative** on real
  photographs: realised FMR is far below the 1 % it was set for. DigiFace impostors (similar renders
  of synthetic people) are harder than LFW's real impostors.
- **In-domain comparison.** The View-1 DevTrain EER threshold gives FMR 0.27 % (Ada) / 0.57 % (Arc) with
  FNMR 0.80 % / 0.77 %. DevTrain shares identities with View 2, which is flagged.
- **Clean-calibrated LTT rules** (DigiFace, all four target/ordering settings) stay within target on
  clean LFW, with coverage ≥ 0.99. Their guarantee is not claimed (A1 violated); this is an empirical
  transfer check only.

### 7.4 Derived synthetic occlusion on LFW (not official LFW; not physical occlusion)

Every one of the 5,958 detected View-2 pairs × 5 families × 3 levels × {side b, both sides} gives 312,795
renders. FNMR is measured at the DigiFace cal_far_1pct threshold, pooled over levels and modes:

| Family | AdaFace AUC / FNMR | ArcFace AUC / FNMR | DigiFace Test-OOD FNMR (Ada / Arc), same threshold | LFW FNMR, high severity, both sides (Ada / Arc) |
|---|---|---|---|---|
| F1 mask | 0.9984 / 2.9 % | 0.9982 / 3.7 % | 10.8 % / 14.5 % | 2.6 % / 2.9 % |
| F2 sunglasses | 0.9992 / 1.1 % | 0.9994 / 0.9 % | 7.6 % / 9.8 % | 1.2 % / 1.0 % |
| F3 hand / object | 0.9988 / 4.3 % | 0.9989 / 4.0 % | 10.1 % / 12.9 % | 13.7 % / 13.3 % |
| F4 random patch | 0.9952 / **16.6 %** | 0.9940 / **16.6 %** | 21.6 % / 25.1 % | **48.0 % / 48.7 %** |
| F5 extreme lighting (non-physical) | 0.9985 / 4.2 % | 0.9945 / 7.7 % | 6.8 % / 9.8 % | 12.2 % / 21.0 % |

- **Same ordering of difficulty as DigiFace.** Random patch is the hardest family and severity is
  monotone.
- **Lower absolute degradation on real photos.** This is partly the conservative synthetic threshold and
  partly real faces carrying more identity information.
- Figures: `lfw_vs_digiface_occlusion_fnmr.svg`, `lfw_derived_occlusion_heatmap.svg`. The DigiFace mixture
  (25 % clean–clean, cross-family pairs) differs from the LFW design (single family, one or both sides),
  so the comparison is directional, not exact.

### 7.5 Risk control on real photos: frozen DigiFace rules of the fold where the family was held out

Primary targets (sFMR ≤ 5 %, sFNMR ≤ 5 %):

| Family | `global` sFNMR (Ada / Arc), coverage | `stratified_vis` (oracle) sFNMR, coverage | `global_v2` sFNMR | `stratified_vis_v2` sFNMR, coverage |
|---|---|---|---|---|
| F1 mask | 0.64 % / 0.32 %, 0.99 / 0.90 | 0.23 % / NA*, 0.83 / 0.00 | 0.81 % / 0.85 % | 0.31 % / 0.30 % |
| F2 sunglasses | 0.34 % / 0.20 % | 0.37 % / 0.26 % | 0.38 % / 0.33 % | 0.47 % / 0.45 % |
| F3 hand / object | 0.53 % / 0.40 % | 0.23 % / 0.28 % | 0.93 % / 0.82 % | 1.32 % / 0.99 % |
| **F4 random patch** | **7.46 % (violated) / 5.28 % (point above)**, 0.97 / 0.96 | **0.35 % / 1.72 %**, 0.43 / 0.87 | **9.96 % / 7.46 % (both violated)** | 4.92 % / 3.25 % (within), 0.96 / 0.94 |
| F5 lighting | 0.60 % / 2.37 % | 0.21 % / 0.60 % | 1.25 % / 3.11 % | 0.39 % / 1.15 % |

\* Every LFW mask render has co-visibility below fold F1's stratum edge: 0.683 from DigiFace val,
against an LFW median of 0.49 (the synthetic mask covers more of real crops). All F1 pairs therefore fall in the
`vis_low` stratum. ArcFace's v1 arm certified no rule there, so it abstains on all of them.
Correct, but uninformative.

- **sFMR ≤ 0.08 % in every cell.**
- **The Phase 2 C3 finding reproduces on real photographs.**
  - Calibration that never saw random patches fails on them: sFNMR of 5.3–10.0 % against a 5 % target.
  - Oracle-visibility stratification restores control at a coverage cost (0.43 / 0.87 vs 0.97 / 0.96).
  - The v2 stratified arm also stays within target here, at coverage 0.96 / 0.94.
- Results at sFMR ≤ 1 % show the same pattern (`lfw_final.json`).

### 7.6 Mask false matches: post-hoc exploratory audit (run `e97048b9c65f`, NOT pre-registered)

The pre-registered analysis could not test this. At the DigiFace threshold almost no LFW impostor is
accepted in any condition: mask-both FMR is 0.00 % for AdaFace and ≤ 0.07 % for ArcFace.

The post-hoc audit compares impostor pairs with the same occlusion on both sides against the same pairs
clean (`reports/experiments/lfw_posthoc_mask.json`):

| Occlusion (high level) | Impostor score shift, Ada / Arc (95 % CI) | FMR at DevTrain EER threshold, occluded vs clean (Ada) | (Arc) |
|---|---|---|---|
| F1 mask | +0.0078 [0.0045, 0.0110] / +0.0103 [0.0072, 0.0133] | 1.68 % vs 0.27 % | 2.93 % vs 0.57 % |
| F2 sunglasses | +0.0203 / +0.0153 (CIs exclude 0) | 1.48 % vs 0.27 % | 1.51 % vs 0.57 % |
| F4 random patch | +0.0110 / +0.0112 (CIs exclude 0) | 1.28 % vs 0.27 % | 1.31 % vs 0.57 % |
| F3 hand / object | +0.0030 / +0.0040 | 0.30 % vs 0.27 % | 0.77 % vs 0.57 % |
| F5 lighting | +0.0014 (CI includes 0) / +0.0044 | 0.30 % vs 0.27 % | 1.51 % vs 0.57 % |

- **Masks raise false matches on real photographs too.** This replicates the DigiFace finding
  qualitatively.
- **The effect is not mask-specific.** Identical sunglasses or patches on both faces also raise impostor
  similarity. The mechanism is the shared occluder content.
- **Caveats.** Post-hoc, at an in-domain threshold with identity overlap (§0), one replication. It needs
  pre-registration in Phase 3.

### 7.7 C1 on real photos (fit-free, exploratory)

Among derived-occluded pairs, errors at the DigiFace threshold are ranked as follows:

| Signal | AdaFace AUROC | ArcFace AUROC |
|---|---|---|
| Low feature norm (quality) | 0.70–0.92 | 0.52–0.90 |
| Low oracle co-visibility | 0.56–0.83 | 0.55–0.82 |
| Co-visibility within quality deciles | 0.46–0.63 | 0.51–0.74 |

- Within quality deciles, co-visibility is highest for hand/object and random patch; it is absent for
  lighting with AdaFace.
- This is consistent with DigiFace: quality dominates, and co-visibility adds a residual,
  family-dependent signal.

### 7.8 Limitations of the real-world evaluation

1. **Identity overlap.** Probable overlap with WebFace4M makes clean accuracy optimistic (§0).
2. **Synthetic occluders.** The occlusions are synthetic overlays on real photographs. They are not
   physical occlusions (no 3-D geometry, shadows, or occluder-induced detection or alignment failure),
   and they were applied after alignment.
3. **No detector re-run on occluded images.** Detector-inclusive occlusion results would need the
   detector to run on occluded photographs. This was not done.
4. **Size of LFW.** 3,000 impostors support FAR 1 % but not 0.1 %. The 6,000 pairs reuse identities;
   intervals are identity-clustered.
5. **Licensing.** LFW was used under user-reported academic authorisation (D-043), not a dataset-owner
   licence.
