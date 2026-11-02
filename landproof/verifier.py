"""Deterministic claim verifier: the only component allowed to approve a landing zone.

Rules (in priority order):
  1. Every mandatory hazard class is checked, even if the LLM did not claim it.
  2. A claim "no X near zone Z" is REFUTED if any SAM 3.1 instance of X lies within
     (zone radius + class buffer) of the zone centre.
  3. The zone must lie fully inside the frame (otherwise UNVERIFIABLE).
  4. PASS only if every claim is SUPPORTED.  The LLM can add claims, never remove them.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from .claims import Certificate, ClaimResult, ClaimStatus, HazardClaim
from .evidence import Instance, frame_digest
from .ledger import Ledger
from .meta.sam import Region, SamBackend
from .scorer import Zone
from .vocab import Vocabulary

_cert_counter = itertools.count(1)


class EvidenceCache:
    """Memoise SAM calls per (frame, phrase, region) — certificates reuse evidence."""

    def __init__(self, backend: SamBackend):
        self.backend = backend
        self._mem: dict[tuple, list[Instance]] = {}

    def get(self, image: np.ndarray, digest: str, phrase: str, region: Region | None) -> list[Instance]:
        key = (digest, phrase, None if region is None else (region.x0, region.y0, region.x1, region.y1, region.out_px))
        if key not in self._mem:
            self._mem[key] = self.backend.segment(image, phrase, region)
        return self._mem[key]


class Verifier:
    def __init__(self, vocab: Vocabulary, sam: SamBackend, cfg: dict, ledger: Ledger | None = None):
        self.vocab = vocab
        self.evidence = EvidenceCache(sam)
        self.cfg = cfg
        v = cfg["verifier"]
        self.min_conf = float(v["min_confidence"])
        self.crop_zoom = bool(v["crop_zoom"])
        self.crop_context_m = float(v["crop_context_m"])
        self.crop_size_px = int(v["crop_size_px"])
        # zoomed re-checks only where small objects hide (people, animals, vehicles, wires, debris);
        # large classes (water, roads, buildings, trees) reuse the cached full-frame evidence
        self.crop_classes = set(v.get("crop_zoom_classes", ["person", "animal", "vehicle", "wire_pole", "obstacle"]))
        self.ledger = ledger
        self.models = dict(cfg.get("models", {}))

    # ------------------------------------------------------------------
    def mandatory_claims(self, zone_id: str, existing: list[HazardClaim]) -> list[HazardClaim]:
        # a mandatory class is covered only by its best validated phrase: a weaker phrase the
        # LLM chose ("puddle") must not be the sole evidence that a class (water) is absent
        covered = {c.phrase for c in existing}
        added = []
        for hc in self.vocab.mandatory_classes:
            if hc.default_phrase not in covered:
                added.append(
                    HazardClaim(
                        claim_id=f"{zone_id}-m-{hc.name}",
                        zone_id=zone_id,
                        phrase=hc.default_phrase,
                        hazard_class=hc.name,
                        source="verifier_mandatory",
                        rationale="mandatory class enforced by verifier",
                    )
                )
        return added

    def _region(self, zone: Zone, gsd_m: float, shape) -> Region | None:
        if not self.crop_zoom:
            return None
        max_buf = max(c.buffer_m for c in self.vocab.hazard_classes)
        half = (zone.radius_m + max_buf + self.crop_context_m) / gsd_m
        cx, cy = zone.center_px
        h, w = shape[:2]
        r = Region(int(cx - half), int(cy - half), int(math.ceil(cx + half)), int(math.ceil(cy + half)), self.crop_size_px).clip(w, h)
        # Only zoom if it actually magnifies; otherwise use the full frame (cheaper cache reuse).
        return r if r.scale() > 1.05 else None

    def check_claim(self, image, digest, claim: HazardClaim, zone: Zone, gsd_m: float, region) -> ClaimResult:
        hc = self.vocab.class_of(claim.phrase)
        required = zone.radius_m + hc.buffer_m
        try:
            insts = self.evidence.get(image, digest, claim.phrase, region)
        except Exception as e:  # backend failure is never treated as evidence of absence
            return ClaimResult(claim=claim, status=ClaimStatus.UNVERIFIABLE, required_m=required, note=f"evidence error: {e}")
        insts = [i for i in insts if i.confidence is None or i.confidence >= self.min_conf]
        nearest = math.inf
        violating = []
        for inst in insts:
            d_m = inst.distance_px(zone.center_px) * gsd_m
            nearest = min(nearest, d_m)
            if d_m < required:
                violating.append(inst.object_id)
        status = ClaimStatus.REFUTED if violating else ClaimStatus.SUPPORTED
        backend = insts[0].backend if insts else self.evidence.backend.name
        return ClaimResult(
            claim=claim,
            status=status,
            required_m=round(required, 2),
            nearest_m=None if math.isinf(nearest) else round(nearest, 2),
            instances_seen=len(insts),
            violating_ids=violating,
            evidence_backend=backend,
        )

    def verify_zone(
        self,
        image: np.ndarray,
        zone: Zone,
        claims: list[HazardClaim],
        gsd_m: float,
        altitude_m: float,
        t: float | None = None,
        digest: str | None = None,
    ) -> Certificate:
        digest = digest or frame_digest(image)
        h, w = image.shape[:2]
        for c in claims:
            self.vocab.class_of(c.phrase)  # raises if the phrase is not vetted
        added = self.mandatory_claims(zone.zone_id, claims)
        all_claims = list(claims) + added
        region = self._region(zone, gsd_m, image.shape)

        results = [self.check_claim(image, digest, c, zone, gsd_m, region if c.hazard_class in self.crop_classes else None) for c in all_claims]

        reasons = []
        cx, cy = zone.center_px
        r_px = zone.radius_m / gsd_m
        if cx - r_px < 0 or cy - r_px < 0 or cx + r_px > w or cy + r_px > h:
            reasons.append("zone not fully inside the frame")
        for r in results:
            if r.status == ClaimStatus.REFUTED:
                reasons.append(f"REFUTED: {r.claim.phrase} at {r.nearest_m} m < {r.required_m} m ({', '.join(r.violating_ids[:3])})")
            elif r.status == ClaimStatus.UNVERIFIABLE:
                reasons.append(f"UNVERIFIABLE: {r.claim.phrase} ({r.note})")
        verdict = "PASS" if not reasons else "FAIL"
        backends = sorted({r.evidence_backend for r in results if r.evidence_backend})
        cert = Certificate(
            cert_id=f"C{next(_cert_counter):04d}",
            zone_id=zone.zone_id,
            verdict=verdict,
            frame_sha256=digest,
            issued_t=float(t if t is not None else time.time()),
            altitude_m=float(altitude_m),
            gsd_m=float(gsd_m),
            center_px=(float(cx), float(cy)),
            radius_m=float(zone.radius_m),
            results=results,
            mandatory_added=[c.hazard_class for c in added],
            reasons=reasons,
            evidence_backend="+".join(backends),
            models={k: v for k, v in self.models.items() if k in ("sam", "spark", "glimmer")},
        )
        if self.ledger is not None:
            e = self.ledger.append("certificate", cert.model_dump(mode="json"), t_sim=t)
            cert.ledger_hash = e["hash"]
        return cert
