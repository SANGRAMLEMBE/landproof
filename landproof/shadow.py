"""Shadow mode on real footage: certify a zone on the first frame, then watch the descent.

Nothing is flown. The HUD shows what LandProof decides on every real frame, and the
ledger records what it WOULD have commanded. This is how real drone video (e.g. a
teammate walking into the zone) is evaluated honestly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .agents.atlas_agent import AtlasAgent
from .descent.watch import DescentWatch, State
from .geometry import Airframe, Camera
from .ledger import Ledger, verify_chain
from .sim.footage import FootageConfig, FootageSource
from .verifier import Verifier
from .viz.hud import Hud
from .viz.video import VideoWriter
from .vocab import Vocabulary


def run_shadow(fc: FootageConfig, cfg: dict, vocab: Vocabulary, *, sam, student, planner="scripted", spark=None, zone: str | None = None,
               out_dir="runs/shadow", link_down_t: float | None = None, glimmer=None, playback_speed: float = 1.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam, air = Camera.from_cfg(cfg), Airframe.from_cfg(cfg)
    src = FootageSource(fc)
    ledger = Ledger(out / "ledger.jsonl")
    verifier = Verifier(vocab, sam, cfg, ledger)
    it = iter(src)
    idx0, first = next(it)
    atlas = AtlasAgent(vocab, cfg, verifier, cam, air, planner=planner, spark=spark, ledger=ledger).build(first.rgb, first.pose_est.as_tuple(), t=0.0)
    if not atlas.entries:
        raise RuntimeError(f"no zone certified on the first frame; rejected: {atlas.rejected}")
    # the pilot descended over the zone nearest the image centre (or the one named)
    centre = (cam.width / 2, cam.height / 2)
    entry = next((e for e in atlas.entries if e.zone_id == zone), None) or min(atlas.entries, key=lambda e: math.dist(e.zone.center_px, centre))
    atlas.entries = [entry] + [e for e in atlas.entries if e is not entry]

    watch = DescentWatch(vocab, cfg, cam, air, atlas, student, ledger, edge_gate=glimmer)
    watch.entry = entry
    events: list[str] = []
    watch._lock_zone(first, events)
    watch.state = State.DESCEND
    ledger.append("shadow_start", {"video": fc.video, "zone": entry.zone_id, "cert": entry.cert.cert_id}, t_sim=0.0)

    hud = Hud(cam.width, cam.height, mode="REPLAY OF REAL FOOTAGE - SHADOW MODE (nothing flown)", evidence=atlas.evidence_backend, planner=f"{planner}")
    writer = VideoWriter(out / "shadow.mp4", src.fps / fc.stride * playback_speed, hud.size)
    decisions = []
    for idx, obs in [(idx0, first)] + list(it):
        if hasattr(student, "set_frame"):
            student.set_frame(idx)
        link_up = link_down_t is None or obs.t < link_down_t
        cmd, rep = watch.step(obs, link_up)
        rep.events = events + rep.events if idx == idx0 else rep.events
        for e in rep.events:
            decisions.append({"t": round(obs.t, 2), "frame": idx, "event": e})
        writer.write(hud.render(obs.rgb, rep))
    writer.close()
    ok, _ = verify_chain(ledger.entries)
    summary = {"video": fc.video, "zone": entry.zone_id, "certificate": entry.cert.cert_id, "evidence": atlas.evidence_backend,
               "decisions": decisions, "final_state": watch.state.value, "ledger_chain_ok": ok, "output": str(out / "shadow.mp4")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
