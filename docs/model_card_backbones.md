# Frozen Backbones: Model Card and Verification (Phase 2)

Inference only: no training and no fine-tuning. Weights live in the private data root and are
never committed. Source of truth: `src/veilx/models/spec.py` (pinned values) and
`reports/models/model_verification.json` (measured checks).

## 1. Checkpoints

| | AdaFace IR-101 (primary) | ArcFace IR-101 (secondary) |
|---|---|---|
| Source | HF `minchul/cvlface_adaface_ir101_webface4m` | HF `minchul/cvlface_arcface_ir101_webface4m` |
| Pinned revision | `f2b38d9e24bfe301490d8dd081d8924b102333dd` | `5846a5e96fbe1094952a414c78ce5b057c92d013` |
| Weights file | `model.safetensors` (tensor-only; the pickled `model.pt` was **not** downloaded) | same |
| SHA-256 (= Hub LFS record) | `21adb6220e8799a0e658f16946df9649c7269f432fe9810a7b9c4ad1241080a8` | `e90e0c8e24c94409fe53c2254a1343ef54f48a3a66b885b8ff69aefa21660ee7` |
| Size | 260,980,552 B | 260,997,808 B |
| Architecture | CVLface IR-101 (`models/iresnet/model.py`, 3/13/30/3 basic IR blocks, BN-Dropout-FC-BN1d head) | InsightFace iresnet100 (`models/iresnet_insightface/model.py`) |
| Parameters | 65,150,912 | 65,156,160 |
| Training data / loss | WebFace4M / AdaFace (arXiv:2204.00964) | WebFace4M / ArcFace (arXiv:1801.07698) |
| Input | RGB, 112×112, `(x/255 − 0.5)/0.5`, NCHW float | same |
| Alignment | ArcFace 5-point template, 112×112 | same |
| Output | 512-d; L2-normalised by VEIL-X, with the pre-normalisation norm kept as a quality proxy | same |
| Similarity | cosine | cosine |
| Licence | CVLface code MIT. The weights carry no licence file; the card asks users to follow the training-data licence (WebFace: non-commercial research). Used for non-commercial academic research only | same |

**Why this ArcFace (D-037).** It has the same architecture family, the same training data and
the same release pipeline as the AdaFace checkpoint. The AdaFace vs ArcFace comparison
therefore isolates the training loss. It also loads from safetensors in pure PyTorch, with no
ONNX runtime. The InsightFace `buffalo_l` w600k_r50 release (ResNet-50, WebFace600K) was the
alternative; it was not needed because the detector-inclusive LFW path is blocked.

**Preprocessing conventions verified, not assumed.** The CVLface configs state
`color_space: RGB` and use the test transform `ToTensor() + Normalize(0.5, 0.5)`. The original
AdaFace repository used BGR; the CVLface re-release does not. RGB vs BGR is also checked
empirically (§2).

## 2. Verification (MEASURED, 20,000 DigiFace **validation** pairs, protocol v0.4.0)

`scripts/verify_models.py`; numbers in `reports/models/model_verification.json`.

| Check | AdaFace | ArcFace |
|---|---|---|
| Strict load: exact key set, shapes and dtypes; `strict=True`; loaded-tensor fingerprint = checkpoint fingerprint | pass (917 tensors) | pass (925 tensors) |
| Equality with the **authors' own module** (built from the pinned reference code, same weights), fp32, max abs diff on random and real inputs, CPU and CUDA | 0.0 | 0.0 |
| CPU vs CUDA fp32, max abs diff of raw (unnormalised; real-image norms ≈ 20–25) outputs | ≈ 2e-3 | ≈ 3e-3 |
| Determinism: two CUDA fp32 passes | bit-identical | bit-identical |
| Val AUC, RGB vs BGR input (20,000 pairs) | **0.99881** vs 0.99817 | **0.99810** vs 0.99732 |
| fp16 autocast vs fp32: min cosine, max pair-score diff, val AUC | 0.99989, 0.0027, 0.99881 | 0.99985, 0.0019, 0.99809 |
| bf16 autocast vs fp32: min cosine, max pair-score diff | 0.99413, 0.0112 | 0.99620, 0.0096 |

**Channel order: an honest note.** An earlier 2,000-pair check ordered ArcFace RGB > BGR on
one sample and BGR > RGB on another (differences ≈ 0.002, within sampling noise). On 20,000
pairs RGB is ahead for both backbones, consistent with the authors' configuration. The
decision rests on the configuration; the empirical check is supporting evidence, not a formal
test. DigiFace renders are only weakly colour-dependent, so the gap is small.

The authors' loader calls `load_state_dict(strict=False)` and only prints mismatches. VEIL-X
never does this: any missing, unexpected or mis-shaped tensor raises `ModelLoadError`.

**Precision decision (D-040).** Extraction uses **fp16** autocast. It agrees with fp32 about
4–5× more closely than bf16 on pair scores (max diff 0.002–0.003 vs 0.010–0.011), with an
identical validation AUC, and is faster than fp32. Calibration and test embeddings use the
same precision, so the frozen system (weights + preprocessing + fp16) is identical
across splits.
