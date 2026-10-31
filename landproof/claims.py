"""Falsifiable safety claims and landing certificates."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"      # evidence backend found no instance inside radius + buffer
    REFUTED = "REFUTED"          # evidence places an instance inside radius + buffer
    UNVERIFIABLE = "UNVERIFIABLE"  # e.g. zone not fully in view, backend error


class HazardClaim(BaseModel):
    claim_id: str
    zone_id: str
    phrase: str
    hazard_class: str
    assertion: Literal["absent"] = "absent"
    source: Literal["spark", "verifier_mandatory", "glimmer", "human", "scripted"] = "spark"
    rationale: str = ""

    def text(self, radius_m: float, buffer_m: float) -> str:
        return f"no '{self.phrase}' within {radius_m + buffer_m:.1f} m of {self.zone_id} centre"


class ClaimResult(BaseModel):
    claim: HazardClaim
    status: ClaimStatus
    required_m: float
    nearest_m: float | None = None
    instances_seen: int = 0
    violating_ids: list[str] = Field(default_factory=list)
    evidence_backend: str = ""
    note: str = ""


class Certificate(BaseModel):
    cert_id: str
    zone_id: str
    verdict: Literal["PASS", "FAIL"]
    frame_sha256: str
    issued_t: float
    altitude_m: float
    gsd_m: float
    center_px: tuple[float, float]
    radius_m: float
    results: list[ClaimResult]
    mandatory_added: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    evidence_backend: str = ""
    models: dict = Field(default_factory=dict)
    ledger_hash: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def short(self) -> str:
        ok = sum(r.status == ClaimStatus.SUPPORTED for r in self.results)
        return f"{self.cert_id} {self.zone_id} {self.verdict} ({ok}/{len(self.results)} claims supported)"
