"""Research-integrity checks on the evidence registry and committed simulation reports."""

from __future__ import annotations

import json

import pytest

from veilx.evidence import (
    REQUIRED_MEASURED_PROVENANCE,
    EvidenceRecord,
    EvidenceType,
    load_evidence_registry,
)

from .conftest import ROOT

REGISTRY = load_evidence_registry(ROOT / "reports" / "evidence_registry.yaml")
SIM_DIR = ROOT / "reports" / "simulations"


def test_every_simulation_report_is_registered_and_labelled() -> None:
    registered = {r.artefact for r in REGISTRY.of_type(EvidenceType.SIMULATION)}
    files = sorted(SIM_DIR.glob("*.json"))
    assert files, "simulation reports missing"
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        assert rel in registered, f"{rel} not registered"
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["evidence_type"] == "synthetic_simulation"
        assert "SIMULATED" in data["disclaimer"] and "NOT evidence" in data["disclaimer"]
    for art in registered:
        assert (ROOT / str(art)).is_file(), f"registered artefact missing: {art}"


def test_no_measured_results_without_provenance() -> None:
    # Phase 2: measured results exist; each carries full provenance and a committed artefact
    measured = REGISTRY.of_type(EvidenceType.MEASURED)
    assert measured
    for rec in measured:
        assert set(rec.provenance) >= REQUIRED_MEASURED_PROVENANCE
        assert rec.artefact and (ROOT / rec.artefact).exists()
    with pytest.raises(ValueError, match="provenance"):
        EvidenceRecord(
            id="MEAS-001",
            type=EvidenceType.MEASURED,
            title="t",
            statement="TAR measured on ROF",
            source="x",
            created="2026-10-26",
        )


def test_type_rules() -> None:
    with pytest.raises(ValueError, match="simulated"):
        EvidenceRecord(
            id="SIM-999",
            type=EvidenceType.SIMULATION,
            title="t",
            statement="coverage was 95%",
            source="x",
            artefact="reports/simulations/x.json",
            created="2026-10-26",
        )
    with pytest.raises(ValueError, match="URL"):
        EvidenceRecord(
            id="PUB-999",
            type=EvidenceType.PUBLISHED,
            title="t",
            statement="some published finding",
            source="memory",
            created="2026-10-26",
        )


def test_data_statistics_are_not_performance_claims() -> None:
    for r in REGISTRY.of_type(EvidenceType.DATA_STATISTIC):
        assert (ROOT / str(r.artefact)).is_file(), r.id
    with pytest.raises(ValueError, match="performance"):
        EvidenceRecord(
            id="DATA-999",
            type=EvidenceType.DATA_STATISTIC,
            title="t",
            statement="the backbone reaches TAR@FAR 0.99 on the test split",
            source="x",
            artefact="reports/data/x.json",
            created="2026-10-26",
        )


def test_public_data_summary_has_no_identifiers() -> None:
    text = (ROOT / "reports" / "data" / "digiface_phase1b_summary.json").read_text(encoding="utf-8")
    assert "digiface1m:" not in text and "relpath" not in text and "pair_id" not in text
    data = json.loads(text)
    assert data["evidence_type"] == "real_data_preparation_statistics"


def test_simulation_statements_match_reports() -> None:
    """Spot-check that registry statements quote the committed numbers."""
    shift = json.loads((SIM_DIR / "guarantee_under_shift.json").read_text(encoding="utf-8"))
    assert shift["results"]["severity=0.25"]["violation_rate"] == pytest.approx(0.955)
    stat = next(r for r in REGISTRY.records if r.id == "SIM-004").statement
    assert "95.5%" in stat
    power = json.loads((SIM_DIR / "clean_tolerance_power.json").read_text(encoding="utf-8"))
    assert power["results"]["n_identities=1000"]["power_noninferiority"] == pytest.approx(0.68)
