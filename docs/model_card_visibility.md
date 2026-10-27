# Visibility Estimator (VisNet): Model Card (Phase 3)

Trained by VEIL-X for research on occlusion-aware verification. Five fold models, each trained
**without** one occlusion family so that family can be evaluated as unseen. Weights are private
(`VEILX_DATA_ROOT/models/phase3/ve/<fold>/best.safetensors`); only training summaries are
public (`reports/experiments/phase3_ve_training_<fold>.json`, which include the weights'
SHA-256). Accuracy results are in `docs/phase3_results.md` §2 (gate 3B).

## 1. What it is

| | |
|---|---|
| Task | Per-cell visible-face probability on a 28×28 grid over the aligned 112×112 crop; also gives a global visible fraction (mean over face cells) and region estimates (eyes, nose, mouth, lower face) |
| Architecture | `veilx.vis.model.VisNet(width=32)`: a U-Net-lite with encoder 112→7 (32/64/128/256/256 channels), a transposed-conv decoder to 28×28 with skips, and a 1×1 head producing logits. 3,604,385 parameters |
| Input | RGB 112×112 aligned crop, `(x/255 − 0.5)/0.5` (same as the backbones) |
| Target | Soft visible fraction per 4×4-pixel cell from the occlusion engine's ground-truth mask (`veilx.vis.regions.visibility_target`) |
| Loss | BCE on the soft targets; face cells weight 1.0, other cells 0.25 |
| Training data | DigiFace **p3_train** identities only. Occlusions are rendered on the fly by the engine from the four families **seen** in that fold; 25 % of samples are clean |
| Selection | Fixed 10,000-render p3_val set (seen families), early stopping on validation loss (patience 4), best checkpoint kept |
| Optimiser | AdamW lr 1e-3, wd 1e-4, cosine schedule; batch 128; 50k samples/epoch; ≤ 30 epochs; fp16 AMP; seed 20261027 |
| Not used | Face-recognition backbones (frozen and separate), p3_test, held-out family, LFW |

## 2. Training record (measured; epochs 0-indexed)

| Fold (held-out family) | Epochs run | Best epoch | Best val loss | Val IoU at best | Val global MAE at best | fp16 vs fp32 max \|Δp\| | Wall |
|---|---|---|---|---|---|---|---|
| F1 mask | 9 | 4 | 0.0369 | 0.900 | 0.0130 | 0.013 | 10 min |
| F2 sunglasses | 9 | 4 | 0.0356 | 0.917 | 0.0111 | 0.015 | 11 min |
| F3 hand/object | 15 | 10 | 0.0306 | 0.922 | 0.0093 | 0.016 | 17 min |
| F4 random patch | 30 | 29 | 0.0221 | 0.948 | 0.0038 | 0.027 | 32 min |
| F5 extreme lighting | 30 | 29 | 0.0196 | 0.972 | 0.0005 | 0.008 | 31 min |

Per-epoch values are in each fold's `history`. Validation losses, IoU and MAE are
**not comparable across folds**, because each fold's validation set contains a different set
of (seen) families.

**Observations.**
- **F4 and F5 hit the 30-epoch cap still improving** (best = last epoch); they are not
  converged by the early-stopping criterion.
- Validation loss is **noisy**, with single-epoch spikes up to ~2× (e.g. F1 epochs 7–8,
  F3 epoch 11). Checkpoint selection makes the saved models robust to this, but it indicates
  some optimisation instability at lr 1e-3. More epochs or a lower learning rate are Phase 4
  options; the Phase 3 protocol fixed these values beforehand and they were not changed.
- Peak GPU memory was 1.3 GB and peak process-tree RAM 3.5 GB (memory-monitored).

## 3. Intended use and limits

- **Use:** research input to pairwise co-visibility features, fusion models and stratified
  calibration in VEIL-X Phase 3.
- **Not** a face detector, a segmentation model for arbitrary images or a liveness or
  occlusion-attack detector.
- **Trained on synthetic faces and synthetic occluders only.** Behaviour on physical
  occluders is unknown (see `docs/physical_occlusion_readiness.md`). An estimator that
  *overestimates* visibility on a new occluder breaks the per-stratum calibration guarantee
  (SIM-013).
- The **ground-truth oracle** maps used for comparison come from the occlusion engine and are
  never available at deployment.
- Weights derive from DigiFace-1M renders (non-commercial research licence); they are not
  redistributed.
