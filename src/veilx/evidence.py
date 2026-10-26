"""Research-integrity registry: every claim-bearing artefact is typed by evidence category.

Categories (never interchangeable):

* ``published_result``           — a result reported in a cited paper (needs a source URL/DOI);
* ``mathematical_calculation``   — exact calculation (e.g. binomial sample-size bounds);
* ``synthetic_simulation``       — Monte Carlo on synthetic scores; never face-recognition evidence;
* ``real_measured_model_result`` — measured on real data with a real model (needs full provenance);
* ``real_data_statistic``        — measured facts about prepared data (counts, integrity, splits);
  never a statement about model performance;
* ``unverified_hypothesis``      — a claim not yet tested.

The registry lives in ``reports/evidence_registry.yaml``; tests check that every simulation report
on disk is registered as ``synthetic_simulation`` and carries the SIMULATED disclaimer, and that
no ``real_measured_model_result`` exists without provenance.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from veilx.errors import ConfigError

REQUIRED_MEASURED_PROVENANCE = frozenset(
    {"git_commit", "dataset_ids", "model_ids", "split", "protocol_version", "final_run_log_entry"}
)


class EvidenceType(StrEnum):
    PUBLISHED = "published_result"
    CALCULATION = "mathematical_calculation"
    SIMULATION = "synthetic_simulation"
    MEASURED = "real_measured_model_result"
    DATA_STATISTIC = "real_data_statistic"  # facts about prepared data; never model performance
    HYPOTHESIS = "unverified_hypothesis"


_PERFORMANCE_WORDS = ("tar@", "tar ", "far ", "eer", "auc", "accuracy", "fnmr", "fmr ")


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Z]+-\d{3}$")
    type: EvidenceType
    title: str
    statement: str = Field(min_length=10)
    source: str
    artefact: str | None = None
    created: str
    provenance: dict[str, str | list[str]] = Field(default_factory=dict)
    caveats: str = ""

    @model_validator(mode="after")
    def _rules(self) -> EvidenceRecord:
        if self.type is EvidenceType.PUBLISHED and not self.source.startswith(("http", "doi:")):
            raise ValueError(f"{self.id}: published results need a URL or DOI source")
        if self.type is EvidenceType.SIMULATION:
            if not (self.artefact or "").startswith("reports/simulations/"):
                raise ValueError(f"{self.id}: simulation artefacts live in reports/simulations/")
            if "simulat" not in self.statement.lower():
                raise ValueError(f"{self.id}: simulation statements must say they are simulated")
        if self.type is EvidenceType.MEASURED:
            missing = REQUIRED_MEASURED_PROVENANCE - set(self.provenance)
            if missing:
                raise ValueError(f"{self.id}: measured results need provenance {sorted(missing)}")
        if self.type is EvidenceType.DATA_STATISTIC:
            if not (self.artefact or "").startswith("reports/data/"):
                raise ValueError(f"{self.id}: data statistics live in reports/data/")
            text = f" {self.statement.lower()} "
            if any(w in text for w in _PERFORMANCE_WORDS):
                raise ValueError(f"{self.id}: data statistics may not state model performance")
        return self


class EvidenceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    records: tuple[EvidenceRecord, ...]

    @model_validator(mode="after")
    def _unique(self) -> EvidenceRegistry:
        ids = [r.id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique")
        return self

    def of_type(self, t: EvidenceType) -> tuple[EvidenceRecord, ...]:
        return tuple(r for r in self.records if r.type is t)


def load_evidence_registry(path: Path) -> EvidenceRegistry:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvidenceRegistry.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path.name}: invalid evidence registry ({exc})") from exc
