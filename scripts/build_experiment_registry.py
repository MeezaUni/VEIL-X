"""Machine-readable registry of Phase 2 experiment runs -> reports/experiments/registry.json.

One entry per run artefact that exists: id, purpose, script, evidence type, split(s) touched,
protocol version, dataset and model ids, git commit recorded by the run, run id / frozen-config
hash where applicable, artefact path and SHA-256. Missing artefacts are listed as ``pending``.

    python scripts/build_experiment_registry.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
MODELS = ["adaface_ir101_webface4m", "arcface_ir101_webface4m"]

ENTRIES: list[dict[str, Any]] = [
    {
        "id": "P2-EXP-001",
        "purpose": "Backbone verification: strict load, equality with authors' modules, channel "
        "order, precision agreement, determinism, throughput",
        "script": "scripts/verify_models.py --pairs 20000",
        "artefact": "reports/models/model_verification.json",
        "splits": ["val"],
    },
    {
        "id": "P2-EXP-002",
        "purpose": "Extraction pilot on GPU (fp16): throughput, latency, memory",
        "script": "scripts/extract_embeddings.py --pilot 2048",
        "artefact": "reports/experiments/extraction_pilot_cuda.json",
        "splits": ["val", "test (images only; no scores computed)"],
    },
    {
        "id": "P2-EXP-003",
        "purpose": "Extraction pilot on CPU (fp32): CPU fallback throughput",
        "script": "scripts/extract_embeddings.py --pilot 128 --device cpu --precision fp32 "
        "--kinds clean",
        "artefact": "reports/experiments/extraction_pilot_cpu.json",
        "splits": ["(clean images of evaluation pair sets; no scores computed)"],
    },
    {
        "id": "P2-EXP-004",
        "purpose": "Full embedding extraction (clean, 20 LOFO groups, stress), both backbones",
        "script": "scripts/extract_embeddings.py",
        "artefact": "reports/experiments/extraction_run.json",
        "splits": ["val", "cal", "test (images only; no scores computed)"],
    },
    {
        "id": "P2-EXP-005",
        "purpose": "Develop stage: thresholds, LTT candidate orders, strata edges, LTT fits "
        "(validation + calibration only)",
        "script": "scripts/evaluate_phase2.py develop",
        "artefact": "reports/experiments/phase2_develop.json",
        "splits": ["val", "cal"],
    },
    {
        "id": "P2-EXP-006",
        "purpose": "Final evaluation (single logged test run): clean baselines, LOFO degradation, "
        "realised selective risk, C1 exploratory, stress, audit",
        "script": "scripts/evaluate_phase2.py final --final",
        "artefact": "reports/experiments/phase2_final.json",
        "splits": ["test"],
    },
    {
        "id": "P2-EXP-007",
        "purpose": "Reproducibility audit: logged rerun of the DigiFace final with the same freeze",
        "script": "scripts/evaluate_phase2.py final --final --note 'reproducibility audit'",
        "artefact": "reports/experiments/phase2_reproducibility.json",
        "splits": ["test"],
    },
    {
        "id": "P2-EXP-008",
        "purpose": "LFW detection/alignment (SCRFD det_10g) and detector-free fixed crop",
        "script": "scripts/lfw_detect_align.py",
        "artefact": "reports/experiments/lfw_detection.json",
        "splits": ["lfw (all images; no scores)"],
        "dataset_ids": ["kaggle_jessicali9530_lfw"],
    },
    {
        "id": "P2-EXP-009",
        "purpose": "LFW embedding extraction (clean det/fixed, 312,795 derived occlusions)",
        "script": "scripts/lfw_embed.py --part-size 16384",
        "artefact": "reports/experiments/lfw_extraction.json",
        "splits": ["lfw (images only; no scores)"],
        "dataset_ids": ["kaggle_jessicali9530_lfw"],
    },
    {
        "id": "P2-EXP-010",
        "purpose": "LFW final (single logged run): official View-2 clean, DigiFace rule transfer, "
        "derived synthetic occlusion, held-out-family risk control, C1",
        "script": "scripts/evaluate_lfw.py --final",
        "artefact": "reports/experiments/lfw_final.json",
        "splits": ["lfw_view2"],
        "dataset_ids": ["kaggle_jessicali9530_lfw"],
    },
    {
        "id": "P2-EXP-011",
        "purpose": "POST-HOC EXPLORATORY shared-occluder false-match audit on LFW",
        "script": "scripts/lfw_posthoc_mask.py",
        "artefact": "reports/experiments/lfw_posthoc_mask.json",
        "splits": ["lfw_view2"],
        "dataset_ids": ["kaggle_jessicali9530_lfw"],
    },
    {
        "id": "P2-SIM-011",
        "purpose": "Paired-difference interval coverage (clean margin)",
        "script": "scripts/run_simulations.py --only paired_difference_coverage",
        "artefact": "reports/simulations/paired_difference_coverage.json",
        "splits": ["(synthetic)"],
        "dataset_ids": [],
    },
    {
        "id": "P2-SIM-012",
        "purpose": "Validity of the adaptive LTT ordering procedure (D-042)",
        "script": "scripts/run_simulations.py --only adaptive_ordering_validity",
        "artefact": "reports/simulations/adaptive_ordering_validity.json",
        "splits": ["(synthetic)"],
        "dataset_ids": [],
    },
    {
        "id": "P3-SIM-013",
        "purpose": "Stratified LTT validity with predicted (noisy/biased) visibility strata",
        "script": "scripts/run_simulations.py --only noisy_strata_validity",
        "artefact": "reports/simulations/noisy_strata_validity.json",
        "splits": ["(synthetic)"],
        "dataset_ids": [],
    },
    {
        "id": "P2-CALC-001",
        "purpose": "Calibration capacity (exact binomial) at protocol v0.4.0",
        "script": "scripts/capacity_report.py",
        "artefact": "reports/data/capacity_v0.4.0.json",
        "splits": ["cal (pair counts only)"],
    },
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    out: list[dict[str, Any]] = []
    for e in ENTRIES:
        path = ROOT / e["artefact"]
        rec = {"dataset_ids": ["digiface1m"], "model_ids": MODELS} | dict(e)
        if not path.exists():
            out.append(rec | {"status": "pending"})
            continue
        data = json.loads(path.read_text("utf-8"))
        prov = data.get("provenance", {})
        git = prov.get("git", {}) if isinstance(prov, dict) else {}
        rec |= {
            "status": "complete",
            "evidence_type": data.get("evidence_type", "real_measured_model_result"),
            "protocol_version": data.get("protocol_version"),
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            "run_id": data.get("run_id"),
            "frozen_config_sha256": data.get("frozen_config_sha256") or data.get("sha256"),
            "artefact_sha256": _sha(path),
            "created_utc": prov.get("timestamp_utc") if isinstance(prov, dict) else None,
        }
        if e["id"] == "P2-CALC-001":
            rec["model_ids"] = []
        out.append(rec)
    reg = {
        "schema": "veilx-experiment-registry/1",
        "phase": 2,
        "scope": "DigiFace-1M (synthetic faces) and LFW Kaggle v4 (real photographs, D-043); "
        "all occlusions are synthetic",
        "test_run_log": "results/test_runs.jsonl (private, git-ignored)",
        "experiments": out,
    }
    path = EXP / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(reg, fh, indent=2)
    print(
        f"wrote {path.relative_to(ROOT)} ({sum(r['status'] == 'complete' for r in out)} complete)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
