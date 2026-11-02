# Physical-occlusion dataset readiness (Phase 3, gate 3H)

**Status: documentation only. No physical-occlusion dataset has been downloaded, requested or
deserialised.** Using any of them is a separate owner decision (not a Phase 3 prerequisite).
Checked 2026-10-27; licence statements below must be re-verified at the source before any use.

## 1. Why it matters

Every Phase 3 occlusion is synthetic: rendered by `veilx.occlusion`, and on LFW composited onto
real photographs. The learned visibility estimator (VE), the co-visibility features and the
occlusion-aware calibration have **never seen a physical occluder**. SIM-013
(`reports/simulations/noisy_strata_validity.json`) shows the specific risk: if the estimator
*overestimates* visibility on a new kind of occluder, pairs move into a looser stratum. In the
full-scale run this broke the per-stratum guarantee in 20 % of replicates (± 2.3 % MC SE), even
though pooled risk stayed within target. Physical occluders (real masks, sunglasses, hands,
hair, scarves) are the natural test of whether that happens.

## 2. Candidates

| Dataset | What it contains (as stated by the source) | Access | Issues found |
|---|---|---|---|
| **ROF, Real World Occluded Faces** (Erakın, Demir, Ekenel, BIOSIG 2021; [GitHub](https://github.com/ekremerakin/RealWorldOccludedFaces)) | 180 identities; 3,195 neutral, 1,686 sunglasses and 678 masked images; 70 identities have both occlusion types, 110 have sunglasses only | Public repository; images are produced by a `download.py` script **from pickle files** | (a) **Pickle distribution:** loading the data means deserialising third-party pickles, which our rules forbid ("never deserialise untrusted pickles"). (b) **Licence conflict:** the repository shows an MIT licence, while a third-party listing reports CC BY-NC-ND 4.0. Unresolved. (c) Images were collected from Google Image Search (web-crawled; no consent statement found). |
| **AR Face Database** (Martínez & Benavente; [catalogue entry](https://service.tib.eu/ldmservice/dataset/ar-face-database)) | Controlled studio images with real sunglasses and scarves, plus illumination variation, in two sessions | Request to the original authors (CVC/UAB); one catalogue notes the contact could not be reached | Access is uncertain; studio conditions; small and old. Licence/terms only via the request. |

Both are small (on the order of 100–200 identities). Any confirmatory claim would therefore have
wide identity-level intervals. Treat them as **external follow-up** evidence, never as
confirmatory evidence.

## 3. What would be needed before use (owner checklist)

1. **Authorisation:** written owner/supervisor approval naming the dataset and the purpose, as
   for LFW (D-043).
2. **Licence:** resolve the ROF MIT vs CC BY-NC-ND discrepancy with the authors. If ND applies,
   no derivatives (crops/occlusion edits) may be redistributed; local research use only.
3. **Pickle safety (ROF):** never call `pickle.load` / `np.load(allow_pickle=True)` on the
   files. Acceptable routes, in order of preference:
   (i) ask the authors for the original JPEGs;
   (ii) inspect the pickles with `pickletools.dis` (which does not execute them) and convert
   them in a throw-away, network-less container using a restricted unpickler. That unpickler
   must whitelist only the NumPy array-reconstruction globals and should be reviewed before it
   is run.
4. **Privacy:** these are photographs of real people. Keep them under `VEILX_DATA_ROOT`, never
   commit images, identity names, manifests or embeddings, and publish aggregates only (same
   rules as LFW).

## 4. Technical readiness of the pipeline

| Component | Ready? | Note |
|---|---|---|
| Detection + 5-point alignment (SCRFD-10G, Umeyama to the ArcFace template) | Yes | Built and audited for LFW (`veilx.detect`). Occluders lower detection rates, so report detector failures separately, as for LFW. |
| Frozen embeddings (AdaFace / ArcFace IR-101) | Yes | `scripts/lfw_embed.py` pattern (part shards, resume, memory monitor). |
| Learned visibility maps | Yes | Five fold models. On real occluders, **use the fold whose held-out family matches**: F1 for masks, F2 for sunglasses. That makes the occluder unseen by the estimator. |
| Ground-truth visibility | **No** | Physical datasets have no pixel masks. The oracle arm (A5) and map-accuracy metrics cannot be computed. Only decision-level metrics are available, plus optional hand annotation of a small subset. |
| Fixed thresholds / calibration | Yes | Use the frozen Phase 3 rules (`phase3_frozen.json`) without re-tuning. |
| Protocol | Draft | Genuine pairs: neutral vs occluded image of the same identity. Impostors: occluded vs other identities, matched on occlusion type. Identity-level jackknife. Pre-register before the first look. |

## 5. Recommendation

Physical-occlusion evaluation is the most informative next external test of C3. It should be a
**Phase 4 item**, gated on steps 1–3. Until then, every Phase 3 statement about occlusion is
explicitly limited to synthetic occluders (DigiFace renders and LFW composites).

## 6. Phase 4H update (2026-11-01): wider search and recommendations

Documentation only. Nothing was downloaded, requested or deserialised. All statements are as
stated by the sources and must be re-verified before any request.

| Dataset | Real occlusions | Identities / protocol | Access and terms (as stated) | Fit for VEIL-X |
|---|---|---|---|---|
| **IJB-C** (IARPA Janus Benchmark C; [NIST face challenges](http://www.nist.gov/programs-projects/face-challenges), [Idiap interface notes](https://www.idiap.ch/software/bob/docs/bob/bob.db.ijbc/stable/)) | Natural occlusions in a large share of faces, with **annotated occlusion locations** (spatial occlusion grid) | Defined 1:1 verification and identification protocols | Distributed by NIST under Creative Commons variants; eligibility-gated request | **Best candidate.** Real occlusion annotations allow an occlusion-stratified evaluation with a standard protocol. Needs an access request and a licence review. Possible identity overlap with web-crawled backbone training data must be checked |
| **RMFRD / RMFD** ([GitHub](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)) | Real masks: 5,000 masked + 90,000 unmasked faces of 525 people; a verification subset of 426 people, 3,589 + 3,589 pairs | Identity-organised; verification pairs | **No licence stated** in the repository; web-crawled; mirrors on Google Drive / Baidu (`.rar`) | Masks only. The licence and consent status is unclear, so the owner must clarify before any use |
| **MFR2** ([MaskTheFace](https://gitee.com/Egrt/MaskTheFace)) | Real masks; 53 identities, 269 images, 848 pairs | `pairs.txt` | Web-collected; distributed through a tool download; no explicit licence found | Too small for confirmatory claims; at most a smoke test |
| **CelebA** ([TFDS catalogue](https://www.tensorflow.org/datasets/catalog/celeb_a)) | Mild real occluders through the `Eyeglasses` / `Wearing_Hat` attributes; 10,177 identities | Identity labels; no verification protocol (pairs would be built) | Non-commercial research agreement | Mild occlusion only, with attribute-level (not pixel) labels. Celebrity identities may overlap the backbone training data |
| ROF, AR Face | See §2 | See §2 | Pickles + licence conflict (ROF); on request (AR) | Unchanged |

**Recommendation (Phase 5 or later, owner decision).**
1. Request **IJB-C** access for an occlusion-stratified 1:1 evaluation of the frozen A3 system.
   Use its occlusion annotations to compare quality-stratified and visibility-aware decisions on
   **physical** occluders.
2. For masks specifically, ask the RMFRD authors for terms first.
3. Before any of these, check the identity overlap with WebFace4M (backbone training), as was done
   for LFW in Phase 2.
4. The never-deserialise-pickles rule stays.

## 7. Phase 6 status and actionable access checklist (2026-11-02)

**Status: real physical-occlusion validation is FUTURE WORK, not completed evidence.**
- No physical-occlusion dataset was requested, downloaded or evaluated in any phase.
- Every VEIL-X occlusion result is on synthetic occluders: DigiFace renders, LFW composites, and
  the evaluation-only E1 hair-strand renders.
- Nothing here was re-verified online in Phase 6. Access terms must be checked at the source
  when the request is made.

**Checklist for an IJB-C request**, the preferred benchmark. The owner or supervisor carries it
out; Claude does not submit requests.

1. **Eligibility and terms.** Read the current NIST / IARPA Janus distribution page and its
   licence. Record:
   - who may request it (institutional affiliation; individual vs lab);
   - permitted uses (non-commercial research);
   - redistribution limits (no images, crops or embeddings outside the lab);
   - publication and citation requirements;
   - retention and deletion obligations.
2. **Institutional approval.** Obtain the supervisor's written approval naming IJB-C and the
   purpose: occlusion-stratified 1:1 evaluation of frozen systems. Add the institutional ethics
   or data-protection review if the university requires it for face data. Record it in
   `docs/decision_log.md`, as was done for LFW (D-043).
3. **Request content.** Include:
   - the project summary (non-commercial academic research);
   - the exact use: frozen-model evaluation only, no training;
   - storage: an encrypted local disk, outside any repository;
   - who has access;
   - the retention period;
   - the commitment to publish aggregates only.
4. **Before any score is computed:**
   - **file safety:** verify archive integrity and file formats (images and CSV / text metadata
     only; no pickles or executables);
   - **training-data overlap:** IJB-C subjects are public figures, and an overlap with WebFace4M
     (backbone training data) is likely. It must be stated as a limitation, as for LFW;
   - **pre-registration:** pre-register a protocol, as `docs/phase6_protocol.md` did. It should
     cover:
     - a 1:1 template or image-pair protocol;
     - strata from IJB-C's occlusion annotations;
     - the frozen A3 systems with no re-tuning;
     - sFMR and sFNMR, coverage, identity-clustered intervals;
     - a single logged access.
5. **Alternatives if IJB-C is unavailable:**
   - **RMFRD** (masks): only after the authors confirm terms.
   - **MFR2:** a smoke test only, too small for claims.
   - **ROF:** only via original JPEGs or a reviewed restricted unpickler; never `pickle.load`.
   - None of these replaces a licensed, annotated benchmark.
