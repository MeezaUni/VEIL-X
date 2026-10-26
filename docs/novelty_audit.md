# VEIL-X — Novelty Audit

> Labels follow `research_landscape.md` §0. Paper IDs (P#) refer to that document.
> Scores in this document are **the author's judgements [D]**, not measurements.

## 1. Brief ideas vs. verified prior art

| Idea in the brief | Closest verified prior art | Verdict |
|---|---|---|
| ArcFace-style pretrained embeddings | P1, P2, P3 | **Not novel.** Used as frozen infrastructure. |
| Occlusion-aware features / feature masking | P6 PDSN, P7 FROM, P11 G2D | **Not novel.** |
| Regional reliability modelling + adaptive fusion | P12 Zhu & Ma (3-region reliability MLP), P6 (grid-cell masks) | **Not novel.** |
| Periocular emphasis under masks | P13 PLGSA (periocular landmark attention) | **Not novel.** |
| Occlusion-conditioned decision threshold | P13 PLGSA (threshold raised with predicted occlusion severity) | **Not novel.** P13 is a small-data preprint, but it is direct prior art and must be cited. |
| Patch-level matching with frozen features | P9 DeepFace-EMD | **Not novel.** |
| Embedding uncertainty / quality | P17 PFE, P18 DUL, P19 SER-FIQ, P20 CR-FIQA, P21–P23 | **Not novel.** |
| ArcFace + InfoNCE / contrastive | P12 (identity-guided contrastive), P22 UCFace (supervised contrastive) | **Not novel**, and not needed for the RQ. **Dropped [D].** |
| Adaptive experts (MoE) | No verified paper was collected in this phase. The brief states this idea exists. | **Not novel**, and not justified by the RQ. **Dropped [D].** |
| Selective prediction / reject option | P25, P26 (general); FIQA error-vs-reject evaluation (P19, P20) | **Not novel** as a concept. |
| Finite-sample risk control | P27 LTT, P28 CRC (general ML) | **Not novel** as a method. |
| Risk-controlled abstention in face verification **evaluated under held-out occlusion families**, with a pair-level co-visibility signal | **No direct match found** in this search | **Possibly open. Low-to-moderate novelty confidence** (see §4). |

**Conclusion:** VEIL-X's components are all established. The only defensible novelty is a
**question and an evaluation**, not a module: *does pair-level visibility evidence make abstention
and risk control more reliable on occlusion types that the calibration never saw, relative to the
quality-based rejection that strong backbones already provide?*

---

## 2. Candidate research contributions

### C1 — Pair-level co-visibility as an abstention signal
**Claim form [H]:** an abstention score that uses (a) the fraction of facial area visible in
*both* images and (b) co-visibility-weighted local similarity from frozen spatial features yields
a lower AUGRC on held-out occlusion families than the best per-image quality baseline (feature
norm, CR-FIQA) and the score-margin baseline, using the same frozen backbone.

- Builds on P9 (frozen spatial features), P12/P6 (regional reliability) and P20/P2 (quality
  baselines).
- Differs from prior work in operating on the **pair**, targeting **abstention** rather than raw
  accuracy, and using **held-out occlusion families** as the test.
- Requires training a visibility segmenter (small) and a fusion/abstention head (tiny).

### C2 — Visibility-weighted regional fusion for higher raw TAR@FAR
**Claim form [H]:** fusing global cosine with co-visibility-weighted region similarities raises
non-selective TAR@FAR on occluded pairs.

- Heavily overlaps P6, P7, P9, P12 and P13. Gains, if any, are incremental, and SOTA methods that
  fine-tune the backbone (P7, P11) are likely stronger on this axis. That is an expectation, not a
  measurement.
- Kept as a **secondary** outcome of the same pipeline (hypothesis H3 in the evaluation
  protocol), not as a headline contribution.

### C3 — Measuring and reducing risk-control failure under occlusion shift
**Claim form [H]:** (i) FMR/FNMR guarantees obtained with LTT on seen occlusion families are
violated on unseen families, and this is measurable. (ii) Visibility-stratified (Mondrian-style)
calibration reduces the violation, at a coverage cost that is reported.

- Part (i) is an empirical characterisation. It is valuable even if (ii) fails, because it
  quantifies how badly a "calibrated" face verifier misbehaves under a realistic shift.
- Depends on C1's visibility signal for part (ii) only.
- Cheap: uses cached embeddings and runs on CPU.

### Scoring (1 = poor, 5 = strong; author judgement [D])

| Criterion | C1 Co-visibility abstention | C2 Regional fusion accuracy | C3 Risk control under shift |
|---|---|---|---|
| Novelty confidence | 2.5: components known; pair-level + held-out-family framing not found | 1.5: direct overlap with P6, P7, P9, P12, P13 | 3: no face-verification risk-control-under-shift paper found; general CP-under-shift literature exists but was not exhaustively searched |
| Scientific value | 4: answers whether visibility adds information beyond quality | 2: incremental; confounded by backbone choice | 4: a negative or positive result is informative to practitioners |
| Engineering feasibility (1 dev, RTX 4060) | 4: frozen backbone + small segmenter + LR/MLP head | 4: same pipeline | 5: post-hoc on cached scores |
| Compute cost (5 = cheapest) | 4: one embedding pass per image/occlusion variant; segmenter training is small | 4 | 5 |
| Testability | 5: clear baselines and paired bootstrap on AUGRC | 4: standard TAR@FAR | 5: violation rate is directly observable |
| **Total (/25)** | **19.5** | **15.5** | **22** |

---

## 3. Recommendation [D]

> **Approved 2026-10-26 (`decision_log.md` D-001/D-002):** **C3 is the primary study and C1
> is secondary.** Regional fusion (C2) is optional. The paragraphs below are the original
> Phase 0 recommendation. Where they call C1 "core", read "secondary". Any novelty claim
> remains qualified: "to our knowledge, within the search logged in `lit_search_log.md`".
> A negative result for C1 or C3-ii is an acceptable, reportable outcome.

**Original recommendation: C3 + C1 combined into one study.**

> *Under held-out occlusion families and real occlusions (ROF), how much do risk-controlled face
> verification decisions violate their target error rates, and does a pair-level co-visibility
> signal, compared with per-image quality and score margin, (a) improve the risk-coverage
> trade-off and (b) reduce guarantee violations when used for stratified calibration?*

- C3(i) is the **minimum publishable unit**. It needs only frozen backbones, quality baselines and
  LTT, which is the MVP.
- C1 is the **core experimental contribution** on top of the MVP.
- C2 is reported as a secondary metric, without a novelty claim.

**Why this combination is the most defensible:**
1. It is falsifiable with pre-declared comparisons (`evaluation_protocol.md`).
2. A null result is still a result: "visibility does not beat feature norm" is useful to know.
3. It avoids competing with fine-tuned SOTA occluded-FR models on raw accuracy, which a single
   developer with 8 GB of VRAM cannot fairly do.

---

## 4. Novelty risks and required pre-claim checks

| Risk | Severity | Mitigation |
|---|---|---|
| A paper combining conformal/LTT risk control with face verification under distribution shift exists but was not surfaced by this search | **High** | Before writing any "first to" sentence: search Google Scholar, Semantic Scholar and IEEE Xplore for {"conformal" OR "risk control" OR "selective"} × {"face verification" OR "biometric"}, plus IJCB/BIOSIG/FG/TBIOM/TIFS 2023–2026 proceedings. Log queries and dates in `docs/lit_search_log.md`. **Do not use "first" language at all unless that search is clean.** |
| Quality/uncertainty literature (FIQA) already evaluates "error-vs-reject" under occlusion | Medium | The FIQA evaluation protocol is image-level reject. Frame VEIL-X as a pair-level extension and cite P19 and P20 explicitly. |
| PLGSA (P13) already conditions decisions on occlusion severity | Medium | Implement an occlusion-severity-adaptive threshold as an explicit **baseline (B5)**. |
| Reviewers see C1 as "just another reliability-weighted fusion" | Medium | The headline metric is AUGRC/violation, not TAR. Ablate the visibility signal against oracle visibility and against quality. |
| Gains come only from synthetic occlusion artefacts (the synthetic-to-real gap, P31) | High | ROF real-occlusion test is mandatory. A claim with no ROF effect must be explicitly scoped to synthetic occlusions. |
| Backbone pretraining overlap inflates absolute numbers | Medium | Report only relative comparisons with a fixed backbone. State the limitation. Add the DigiFace-1M identities as an overlap-free check (with a domain-gap caveat). |

## 5. Allowed and disallowed claim language

- **Allowed:** "We study…", "We evaluate…", "Under protocol X, method A achieved lower AUGRC than
  B (95% CI …)". The last form only after measurement.
- **Disallowed until the evidence exists:** "novel architecture", "first", "state-of-the-art",
  "robust to unseen occlusions", "guaranteed error rate" (guarantees hold only under
  exchangeability; say so), and any percentage improvement not produced by `eval/`.
