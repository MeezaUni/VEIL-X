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
