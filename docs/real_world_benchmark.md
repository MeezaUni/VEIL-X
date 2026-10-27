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

All Phase 2 measurements come from DigiFace-1M, a **synthetic** dataset, with **synthetic**
occlusions. No statement about real-world robustness is made. The real-world gap is listed as an
open limitation in `docs/statistical_audit.md`.
