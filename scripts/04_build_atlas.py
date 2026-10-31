#!/usr/bin/env python3
"""Build a contingency atlas on one nadir image and save the overlay + agent trace.

  python scripts/04_build_atlas.py --image my_frame.jpg --alt 60 --planner spark --evidence meta-api
  python scripts/04_build_atlas.py --synthetic --planner scripted --evidence oracle      # offline
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from landproof.agents.atlas_agent import AtlasAgent  # noqa: E402
from landproof.backends import make_sam, make_spark  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.data.datasets import camera_frame, load_rgb  # noqa: E402
from landproof.geometry import Airframe, Camera  # noqa: E402
from landproof.ledger import Ledger  # noqa: E402
from landproof.meta.client import METER  # noqa: E402
from landproof.verifier import Verifier  # noqa: E402
from landproof.viz.overlay import zones_overlay  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alt", type=float, default=None, help="altitude (m) the image was taken from")
    ap.add_argument("--gsd", type=float, default=None, help="or the source image GSD (m/px)")
    ap.add_argument("--planner", default="spark", choices=["spark", "scripted"])
    ap.add_argument("--evidence", default="meta-api")
    ap.add_argument("--out", default="runs/atlas")
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    cam, air = Camera.from_cfg(cfg), Airframe.from_cfg(cfg)
    sam = make_sam(a.evidence, vocab, cfg)
    if a.synthetic:
        from landproof.sim.replay import Pose, ReplaySim
        from landproof.sim.scene import make_world

        sim = ReplaySim(make_world(vocab), cam, vocab, Pose(80, 60, a.alt or 80, 0))
        obs = sim.observe()
        rgb, alt = obs.rgb, obs.pose_est.alt
        if hasattr(sam, "register"):
            sam.register(rgb, obs.label)
    else:
        src = load_rgb(Path(a.image))
        if a.gsd:
            rgb, _, gsd = camera_frame(src, None, a.gsd, cam.width, cam.height)
            alt = cam.altitude_for_gsd(gsd, rgb.shape[1])
        else:
            rgb, _, _ = camera_frame(src, None, 1.0, cam.width, cam.height)
            alt = a.alt or 60.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.jsonl")
    agent = AtlasAgent(vocab, cfg, Verifier(vocab, sam, cfg, ledger), cam, air, planner=a.planner, spark=make_spark(a.planner, cfg), ledger=ledger)
    atlas = agent.build(rgb, (0.0, 0.0, alt, 0.0))
    all_zones = [e.zone for e in atlas.entries]
    ov = zones_overlay(rgb, all_zones, certs={e.zone_id: e.cert for e in atlas.entries})
    cv2.imwrite(str(out / "atlas.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    (out / "trace.json").write_text(json.dumps(atlas.trace, indent=1, default=str))
    (out / "atlas.json").write_text(json.dumps({"certified": [{"zone": e.zone.to_dict(), "cert": e.cert.model_dump(mode="json")} for e in atlas.entries],
                                                "rejected": atlas.rejected, "scene_phrases": atlas.scene_phrases}, indent=1, default=str))
    print(f"altitude {alt:.1f} m, {len(atlas.entries)} certified, {len(atlas.rejected)} rejected, {len(atlas.trace)} tool calls")
    for t in atlas.trace:
        print(f"  {t['tool']}({json.dumps(t['args'])[:110]})")
    print(f"-> {out}/atlas.png  spend {METER.report()}")


if __name__ == "__main__":
    main()
