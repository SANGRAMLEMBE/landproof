"""End-to-end emergency-landing scenario: atlas -> baseline refuted -> closed-loop descent.

Used by scripts/05_descent_demo.py and the tests. Produces an MP4 with the HUD, the
hash-chained decision ledger, and a JSON summary scored against ground truth.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agents.atlas_agent import AtlasAgent
from .bench.strategies import FlattestPatch
from .claims import HazardClaim
from .descent.watch import DescentWatch, State
from .geometry import Airframe, Camera
from .ledger import Ledger, verify_chain
from .meta.sam import OracleSam
from .scorer import Zone
from .sim.replay import Actor, Pose, ReplaySim
from .sim.scene import World
from .verifier import Verifier
from .viz.hud import Hud
from .viz.overlay import zones_overlay
from .viz.video import VideoWriter
from .vocab import Vocabulary


@dataclass
class Scenario:
    name: str
    start: Pose
    actors: list[Actor] = field(default_factory=list)
    link_down_t: float | None = None
    battery_start: float = 6.0
    battery_drain_pct_s: float = 0.04
    max_t: float = 150.0
    fps: float = 15.0
    seed: int = 1
    human_abort_t: float | None = None


def monsoon_scenario(link_down_t: float | None = 45.0) -> Scenario:
    """Battery critical over a flooded Chennai junction; a pedestrian walks into zone #1."""
    return Scenario(
        name="monsoon-junction",
        start=Pose(80.0, 60.0, 80.0, 0.0),
        actors=[Actor("person", [(131.0, 18.0), (117.0, 30.0), (104.0, 41.0)], speed_mps=1.4, start_t=25.0, name="pedestrian")],
        link_down_t=link_down_t,
    )


class _OracleRecert:
    """Registers the current GT frame so OracleSam can answer (tests / offline demo only)."""

    def __init__(self, verifier: Verifier, sam):
        self.v, self.sam, self.label = verifier, sam, None

    def __call__(self, rgb, zone, gsd, alt, t, prev):
        if isinstance(self.sam, OracleSam):
            self.sam.register(rgb, self.label)
        claims = [HazardClaim(claim_id=f"{zone.zone_id}-r{k}", zone_id=zone.zone_id, phrase=p, hazard_class=self.v.vocab.class_of(p).name, source="spark")
                  for k, p in enumerate(dict.fromkeys(r.claim.phrase for r in prev.results))]
        return self.v.verify_zone(rgb, zone, claims, gsd, alt, t=t)


def run_scenario(world: World, sc: Scenario, cfg: dict, vocab: Vocabulary, *, sam, student, planner: str = "scripted", spark=None,
                 glimmer=None, out_dir: str | Path = "runs/demo", render: bool = True, intro: bool = True, mode_label: str = "SIMULATION",
                 playback_speed: float = 1.5, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam, air = Camera.from_cfg(cfg), Airframe.from_cfg(cfg)
    ledger = Ledger(out / "ledger.jsonl")
    sim = ReplaySim(world, cam, vocab, sc.start, sc.actors, fps=sc.fps, seed=sc.seed)
    obs = sim.observe()
    if isinstance(sam, OracleSam):
        sam.register(obs.rgb, obs.label)
    verifier = Verifier(vocab, sam, cfg, ledger)

    # ---------------------------------------------------------------- 1. contingency atlas (built during healthy flight)
    t0 = time.perf_counter()
    agent = AtlasAgent(vocab, cfg, verifier, cam, air, planner=planner, spark=spark, ledger=ledger)
    atlas = agent.build(obs.rgb, obs.pose_est.as_tuple(), t=0.0)
    atlas_s = time.perf_counter() - t0
    if verbose:
        print(f"[atlas] {len(atlas.entries)} certified, {len(atlas.rejected)} rejected in {atlas_s:.1f}s  planner={planner} evidence={atlas.evidence_backend}")

    # ---------------------------------------------------------------- 2. what a flatness heuristic would do (refuted)
    gsd0 = cam.gsd(obs.pose_est.alt)
    base = FlattestPatch(air).pick(obs.rgb, gsd0, obs.pose_est.alt, {})
    r0 = air.required_radius_m(obs.pose_est.alt)
    bz = Zone("BASELINE", base.xy, r0, r0 / gsd0, 0.0, 0.0, True)
    base_cert = verifier.verify_zone(obs.rgb, bz, [HazardClaim(claim_id="B-0", zone_id="BASELINE", phrase="standing water", hazard_class="water", source="scripted")], gsd0, obs.pose_est.alt, t=0.0)
    if verbose:
        print(f"[baseline] flattest patch at {tuple(round(v) for v in base.xy)} flatness={base.info['flatness']} -> {base_cert.verdict}: {base_cert.reasons[:1]}")

    hud = Hud(cam.width, cam.height, mode=mode_label, evidence=atlas.evidence_backend, planner=f"{planner} | atlas {atlas_s:.1f}s")
    writer = VideoWriter(out / "landing.mp4", sc.fps * playback_speed, hud.size) if render else None
    if writer and intro:
        _intro(writer, hud, obs, atlas, base, base_cert, bz, sc)

    # ---------------------------------------------------------------- 3. emergency: closed-loop descent
    recert = _OracleRecert(verifier, sam)
    edge = glimmer
    watch = DescentWatch(vocab, cfg, cam, air, atlas, student, ledger, recertifier=recert, edge_gate=edge, transit_alt_m=None)
    ledger.append("emergency", {"cause": "battery critical", "battery_pct": sc.battery_start}, t_sim=0.0)
    states, abort_t, first_threat_gt, zone_true = [], None, None, {}
    per_frame_ms = []
    while not watch.done and sim.t < sc.max_t:
        link_up = sc.link_down_t is None or sim.t < sc.link_down_t
        if sc.human_abort_t is not None and abs(sim.t - sc.human_abort_t) < 0.5 / sc.fps:
            watch.human_command("abort")
        recert.label = obs.label
        if hasattr(edge, "set_label"):
            edge.set_label(obs.label)
        f0 = time.perf_counter()
        cmd, rep = watch.step(obs, link_up)
        per_frame_ms.append((time.perf_counter() - f0) * 1000)
        if rep.state in (State.HOLD, State.ABORT) and abort_t is None:
            abort_t = sim.t
        if rep.state == State.DESCEND and rep.zone_id not in zone_true:
            zone_true[rep.zone_id] = (obs.pose_true.x, obs.pose_true.y, obs.pose_true.alt)  # drone is servoed over the zone
        if first_threat_gt is None and watch.entry is not None and watch.entry.zone_id in zone_true:
            zx, zy, za = zone_true[watch.entry.zone_id]
            reach = air.required_radius_m(za) + vocab.by_name["person"].buffer_m
            if any(math.dist(xy, (zx, zy)) < reach for xy in obs.actors_xy.values()):
                first_threat_gt = sim.t
        if verbose:
            for e in rep.events:
                print(f"  t={sim.t:6.2f} alt={obs.pose_true.alt:5.1f}  {e}")
        if writer:
            batt = max(0.0, sc.battery_start - sc.battery_drain_pct_s * sim.t)
            writer.write(hud.render(obs.rgb, rep, battery_pct=batt))
        states.append(rep.state.value)
        obs = sim.step(cmd)
    if writer:
        writer.close()

    gt = sim.touchdown_safety(vocab, air.diameter_m / 2 + air.touchdown_margin_m) if watch.done else {"safe": None, "note": "did not land"}
    ok, bad = verify_chain(ledger.entries)
    summary = {
        "scenario": sc.name,
        "planner": planner,
        "evidence_backend": atlas.evidence_backend,
        "atlas": {"certified": [e.zone_id for e in atlas.entries], "rejected": atlas.rejected, "build_s": round(atlas_s, 2),
                  "scene_phrases": atlas.scene_phrases, "tool_calls": len(atlas.trace)},
        "baseline_flattest_patch": {"verdict": base_cert.verdict, "reasons": base_cert.reasons},
        "landed": watch.done,
        "final_state": watch.state.value,
        "zones_failed": sorted(watch.failed),
        "touchdown_gt": gt,
        "sim_time_s": round(sim.t, 1),
        "first_hold_or_abort_t": abort_t,
        "gt_person_enters_buffer_t": first_threat_gt,
        "reaction_lead_s": None if (abort_t is None or first_threat_gt is None) else round(first_threat_gt - abort_t, 2),
        "watch_ms_per_frame_mean": round(float(np.mean(per_frame_ms)), 1) if per_frame_ms else None,
        "ledger_entries": len(ledger.entries),
        "ledger_chain_ok": ok,
        "video": str(out / "landing.mp4") if render else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _intro(writer: VideoWriter, hud: Hud, obs, atlas, base, base_cert, bz, sc: Scenario):
    """~6 s intro: the flat-texture pick, its refutation, then the certified atlas."""
    from PIL import Image, ImageDraw

    from .viz.hud import C_BAD, C_OK, draw_pad, font

    n = int(sc.fps * 2.2)
    rgb = obs.rgb
    pad = np.full((hud.ch, hud.size[0] - hud.cw, 3), (14, 17, 22), np.uint8)

    def panel_text(lines):
        p = Image.fromarray(pad.copy())
        d = ImageDraw.Draw(p)
        y = 20
        for txt, size, col in lines:
            d.text((18, y), txt, font=font(size, True), fill=col)
            y += size + 12
        return np.array(p)

    # (a) heuristic landing on the flattest patch
    f = rgb.copy()
    draw_pad(f, base.xy, bz.radius_px * 1.6, C_OK)
    img = Image.fromarray(f)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, hud.cw, 38], fill=(0, 0, 0))
    d.text((16, 7), f"BATTERY {sc.battery_start:.0f}% - MUST LAND NOW.  Flattest area: {base.info['flatness']:.0%} smooth -> LANDING", font=font(19, True), fill=C_OK)
    frame_a = np.concatenate([np.array(img), panel_text([("Typical failsafe", 22, (230, 234, 240)), ("land on the flattest", 16, (140, 150, 165)), ("open patch", 16, (140, 150, 165))])], axis=1)
    for _ in range(n):
        writer.write(frame_a)

    # (b) the claim is refuted by SAM 3.1 evidence
    f = rgb.copy()
    draw_pad(f, base.xy, bz.radius_px * 1.6, C_BAD)
    img = Image.fromarray(f)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, hud.cw, 38], fill=(0, 0, 0))
    reason = base_cert.reasons[0] if base_cert.reasons else "PASS"
    d.text((16, 7), "CLAIM REFUTED: " + reason.replace("REFUTED: ", "")[:70], font=font(19, True), fill=C_BAD)
    lines = [("Claim checked", 22, (230, 234, 240)), ("'no standing water'", 16, (140, 150, 165)), (base_cert.verdict, 30, C_BAD if not base_cert.passed else C_OK),
             (f"evidence: {base_cert.evidence_backend}"[:40], 13, (140, 150, 165))]
    frame_b = np.concatenate([np.array(img), panel_text(lines)], axis=1)
    for _ in range(n):
        writer.write(frame_b)

    # (c) the certified contingency atlas
    ov = zones_overlay(rgb, [e.zone for e in atlas.entries], certs={e.zone_id: e.cert for e in atlas.entries})
    img = Image.fromarray(ov)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, hud.cw, 38], fill=(0, 0, 0))
    d.text((16, 7), f"CONTINGENCY ATLAS: {len(atlas.entries)} zones certified, {len(atlas.rejected)} rejected", font=font(19, True), fill=C_OK)
    lines = [("Atlas (pre-computed)", 22, (230, 234, 240))] + [(f"{e.zone_id}  {e.cert.cert_id}  PASS", 15, C_OK) for e in atlas.entries[:8]]
    frame_c = np.concatenate([np.array(img), panel_text(lines)], axis=1)
    for _ in range(n):
        writer.write(frame_c)
