"""Contingency-atlas agent: Muse Spark plans, SAM 3.1 supplies evidence, the verifier decides.

Tools exposed to Muse Spark (Chat Completions tool calling, tool_choice=auto):
  segment_concepts(phrases[])        -> SAM 3.1 instance counts / areas per phrase
  propose_zones()                    -> scored candidate zones (+ an overlay image message)
  submit_claims(zone_id, absent[], rationale) -> verifier certificate (PASS/FAIL + reasons)
  finalize_atlas(ranked_zone_ids[], summary)  -> only PASS-certified zones are accepted

Phrases are JSON-schema enums of the vetted vocabulary, so the model cannot invent prompts.
A scripted planner with the same tools runs offline (tests, no API key).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from ..atlas import Atlas, AtlasEntry, pixel_to_world
from ..claims import Certificate, HazardClaim
from ..evidence import frame_digest
from ..geometry import Airframe, Camera
from ..ledger import Ledger
from ..scorer import Zone, ZoneScorer
from ..verifier import Verifier
from ..viz.overlay import zones_overlay
from ..vocab import Vocabulary
from . import prompts


@dataclass
class Session:
    rgb: np.ndarray
    digest: str
    gsd: float
    alt: float
    radius_m: float
    pose_est: tuple
    t: float
    class_masks: dict = field(default_factory=dict)
    landable: np.ndarray | None = None
    segmented: dict = field(default_factory=dict)   # phrase -> summary
    zones: dict = field(default_factory=dict)       # zone_id -> Zone
    certs: dict = field(default_factory=dict)       # zone_id -> Certificate
    claims: dict = field(default_factory=dict)      # zone_id -> [phrases]
    final: list | None = None
    trace: list = field(default_factory=list)


class AtlasAgent:
    def __init__(self, vocab: Vocabulary, cfg: dict, verifier: Verifier, camera: Camera, airframe: Airframe,
                 planner: str = "scripted", spark=None, ledger: Ledger | None = None, min_zones: int = 3):
        self.vocab, self.cfg, self.verifier, self.cam, self.air = vocab, cfg, verifier, camera, airframe
        self.scorer = ZoneScorer(vocab.buffers_m(), cfg)
        self.planner, self.spark, self.ledger = planner, spark, ledger
        self.min_zones = min_zones
        self.max_turns = int(cfg["atlas"]["max_agent_turns"])

    # ================================================================== tools
    def tool_segment_concepts(self, s: Session, phrases: list[str]) -> dict:
        out = {}
        for p in phrases[:10]:
            hc = self.vocab.class_of(p)
            insts = self.verifier.evidence.get(s.rgb, s.digest, p, None)
            insts = [i for i in insts if i.confidence is None or i.confidence >= self.verifier.min_conf]
            if hc.id == 0:
                if s.landable is None:
                    s.landable = np.zeros(s.rgb.shape[:2], bool)
                target = s.landable
            else:
                target = s.class_masks.setdefault(hc.id, np.zeros(s.rgb.shape[:2], bool))
            for i in insts:
                i.paint(target)
            areas = sorted((i.area_px * s.gsd**2 for i in insts), reverse=True)
            out[p] = {"class": hc.name, "instances": len(insts), "total_area_m2": round(sum(areas), 1), "largest_m2": round(areas[0], 1) if areas else 0.0}
            s.segmented[p] = out[p]
        return {"segmented": out, "classes_covered": sorted({self.vocab.class_of(p).name for p in s.segmented})}

    def tool_propose_zones(self, s: Session) -> dict:
        auto = []
        covered = {self.vocab.class_of(p).id for p in s.segmented}
        for hc in self.vocab.hazard_classes:  # never score without the mandatory classes
            if hc.mandatory and hc.id not in covered:
                auto.append(hc.default_phrase)
        if auto:
            self.tool_segment_concepts(s, auto)
        masks = {cid: m for cid, m in s.class_masks.items()} or {self.vocab.hazard_classes[0].id: np.zeros(s.rgb.shape[:2], bool)}
        zones = self.scorer.score(masks, s.gsd, s.radius_m, landable_mask=s.landable)
        s.zones = {z.zone_id: z for z in zones}
        return {
            "auto_segmented_mandatory": auto,
            "zones": [
                {"zone_id": z.zone_id, "center_px": [round(z.center_px[0]), round(z.center_px[1])], "clearance_m": round(z.clearance_m, 1),
                 "score": round(z.score, 3), "landable_fraction": round(z.components.get("landable", 0.5), 2)}
                for z in zones
            ],
            "note": "clearance = distance to the nearest buffered hazard among segmented concepts only",
        }

    def tool_submit_claims(self, s: Session, zone_id: str, absent_phrases: list[str], rationale: str = "") -> dict:
        z = s.zones.get(zone_id)
        if z is None:
            return {"error": f"unknown zone {zone_id}; call propose_zones first"}
        claims = [
            HazardClaim(claim_id=f"{zone_id}-{k}", zone_id=zone_id, phrase=p, hazard_class=self.vocab.class_of(p).name,
                        source="spark" if self.planner == "spark" else "scripted", rationale=rationale[:300])
            for k, p in enumerate(dict.fromkeys(absent_phrases))
        ]
        cert = self.verifier.verify_zone(s.rgb, z, claims, s.gsd, s.alt, t=s.t, digest=s.digest)
        s.certs[zone_id], s.claims[zone_id] = cert, [c.phrase for c in claims]
        return {
            "zone_id": zone_id,
            "certificate": cert.cert_id,
            "verdict": cert.verdict,
            "verifier_added_mandatory": cert.mandatory_added,
            "claims": [{"phrase": r.claim.phrase, "status": r.status.value, "nearest_m": r.nearest_m, "required_m": r.required_m} for r in cert.results],
            "reasons": cert.reasons,
        }

    def tool_finalize_atlas(self, s: Session, ranked_zone_ids: list[str], summary: str = "") -> dict:
        ok = [z for z in ranked_zone_ids if z in s.certs and s.certs[z].passed]
        bad = [z for z in ranked_zone_ids if z not in ok]
        if not ok and any(c.passed for c in s.certs.values()):
            return {"error": "none of the listed zones has a PASS certificate", "certified": [k for k, c in s.certs.items() if c.passed]}
        s.final = ok
        return {"accepted": ok, "rejected_not_certified": bad, "summary": summary[:500]}

    # ================================================================== schemas
    def tool_schemas(self) -> list[dict]:
        hazard_enum = self.vocab.all_phrases
        return [
            {"type": "function", "function": {
                "name": "segment_concepts",
                "description": "Run SAM 3.1 on the keyframe for each phrase. Returns instance counts and areas. Max 10 phrases per call.",
                "parameters": {"type": "object", "properties": {"phrases": {"type": "array", "items": {"type": "string", "enum": hazard_enum}, "maxItems": 10}}, "required": ["phrases"]}}},
            {"type": "function", "function": {
                "name": "propose_zones",
                "description": "Score candidate touchdown zones from all hazards segmented so far (mandatory classes are added automatically). You will then receive an overlay image.",
                "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {
                "name": "submit_claims",
                "description": "Assert hazards ABSENT near a zone. The verifier checks each against SAM 3.1 evidence and returns PASS/FAIL.",
                "parameters": {"type": "object", "properties": {
                    "zone_id": {"type": "string"},
                    "absent_phrases": {"type": "array", "items": {"type": "string", "enum": hazard_enum}, "minItems": 1},
                    "rationale": {"type": "string"}}, "required": ["zone_id", "absent_phrases", "rationale"]}}},
            {"type": "function", "function": {
                "name": "finalize_atlas",
                "description": "Finish. List certified zone ids best-first.",
                "parameters": {"type": "object", "properties": {"ranked_zone_ids": {"type": "array", "items": {"type": "string"}}, "summary": {"type": "string"}}, "required": ["ranked_zone_ids", "summary"]}}},
        ]

    def _dispatch(self, s: Session, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if name == "segment_concepts":
                res = self.tool_segment_concepts(s, list(args.get("phrases", [])))
            elif name == "propose_zones":
                res = self.tool_propose_zones(s)
            elif name == "submit_claims":
                res = self.tool_submit_claims(s, args["zone_id"], list(args.get("absent_phrases", [])), args.get("rationale", ""))
            elif name == "finalize_atlas":
                res = self.tool_finalize_atlas(s, list(args.get("ranked_zone_ids", [])), args.get("summary", ""))
            else:
                res = {"error": f"unknown tool {name}"}
        except KeyError as e:
            res = {"error": str(e)}
        s.trace.append({"tool": name, "args": args, "result": res, "ms": round((time.perf_counter() - t0) * 1000)})
        if self.ledger is not None:
            self.ledger.append("agent_tool", {"tool": name, "args": args, "result": res}, t_sim=s.t)
        return res

    # ================================================================== planners
    def _run_scripted(self, s: Session):
        phrases = [c.default_phrase for c in self.vocab.hazard_classes] + [self.vocab.by_name["landable"].default_phrase]
        self._dispatch(s, "segment_concepts", {"phrases": phrases})
        self._dispatch(s, "propose_zones", {})
        for zid in list(s.zones):
            absent = [c.default_phrase for c in self.vocab.hazard_classes]
            self._dispatch(s, "submit_claims", {"zone_id": zid, "absent_phrases": absent, "rationale": "scripted planner: all classes"})
        ranked = [z for z in s.zones if s.certs.get(z) and s.certs[z].passed]
        self._dispatch(s, "finalize_atlas", {"ranked_zone_ids": ranked, "summary": "scripted"})

    def _run_spark(self, s: Session):
        from ..meta.spark import image_part

        vw, vh = s.rgb.shape[1] * s.gsd, s.rgb.shape[0] * s.gsd
        msgs = [
            {"role": "system", "content": prompts.ATLAS_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompts.ATLAS_USER.format(alt=s.alt, gsd=s.gsd, vw=vw, vh=vh, radius=s.radius_m, vocab=self.vocab.summary_for_llm(), min_zones=self.min_zones)},
                image_part(s.rgb)]},
        ]
        tools = self.tool_schemas()
        nudged = False
        for _turn in range(self.max_turns):
            msg = self.spark.chat(msgs, tools=tools)
            calls = msg.tool_calls or []
            if not calls:
                if s.final is not None:
                    break
                if nudged:
                    break
                msgs.append({"role": "assistant", "content": msg.content or ""})
                msgs.append({"role": "user", "content": "Please continue using the tools, and finish with finalize_atlas."})
                nudged = True
                continue
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            show_overlay = False
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                res = self._dispatch(s, c.function.name, args)
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)})
                show_overlay |= c.function.name in ("propose_zones", "submit_claims")
            if show_overlay and s.zones:
                ov = zones_overlay(s.rgb, list(s.zones.values()), certs=s.certs)
                msgs.append({"role": "user", "content": [{"type": "text", "text": "Overlay: cyan = candidate, green = PASS, red = FAIL."}, image_part(ov)]})
            if s.final is not None:
                break
        if s.final is None:  # model ran out of turns: fall back to every PASS certificate, verifier-ranked
            ranked = sorted((z for z, c in s.certs.items() if c.passed), key=lambda z: -s.zones[z].score)
            self._dispatch(s, "finalize_atlas", {"ranked_zone_ids": ranked, "summary": "auto-finalized after turn budget"})

    # ================================================================== entry point
    def build(self, rgb: np.ndarray, pose_est: tuple, t: float = 0.0) -> Atlas:
        x, y, alt, yaw = pose_est
        gsd = self.cam.gsd(alt, rgb.shape[1])
        s = Session(rgb=rgb, digest=frame_digest(rgb), gsd=gsd, alt=alt, radius_m=self.air.required_radius_m(alt), pose_est=pose_est, t=t)
        if self.planner == "spark":
            self._run_spark(s)
        else:
            self._run_scripted(s)
        entries = []
        for zid in s.final or []:
            z = s.zones[zid]
            entries.append(AtlasEntry(z, s.certs[zid], pixel_to_world(z.center_px, pose_est, gsd, rgb.shape[1], rgb.shape[0]), s.claims.get(zid, [])))
        rejected = [{"zone_id": zid, "cert": c.cert_id, "reasons": c.reasons} for zid, c in s.certs.items() if not c.passed]
        atlas = Atlas(entries, rgb.copy(), pose_est, gsd, t, rejected, s.trace, sorted(s.segmented), self.verifier.evidence.backend.name, self.planner)
        if self.ledger is not None:
            self.ledger.append("atlas", {"zones": [e.zone_id for e in entries], "certs": [e.cert.cert_id for e in entries], "rejected": rejected,
                                         "planner": self.planner, "evidence": atlas.evidence_backend}, t_sim=t)
        return atlas
