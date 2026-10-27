"""Facial visibility estimation (Phase 3): targets, regions, metrics, heuristic and model.

``regions``, ``metrics``, ``heuristic`` and ``covis`` are numpy-only; ``model`` needs torch.
The ground-truth occlusion mask is a TRAINING TARGET and an ORACLE reference only - it is never
an input of the estimator.
"""
