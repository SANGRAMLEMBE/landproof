#!/usr/bin/env python3
"""Closed-loop emergency landing in the replay simulator, rendered with the HUD.

Offline, no keys (oracle evidence - labelled as such on every frame):
  python scripts/05_descent_demo.py --evidence oracle --student oracle --glimmer oracle

The real stack:
  python scripts/05_descent_demo.py --evidence meta-api --planner spark --student checkpoints/student_sam.pt --glimmer remote

On your own orthophoto / mosaic (labels needed only for ground-truth scoring + oracle mode):
  python scripts/05_descent_demo.py --world-image ortho.jpg --world-labels ortho_labels.png --world-gsd 0.03 --start 120,80 ...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from landproof.backends import make_glimmer, make_sam, make_spark, make_student  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.demo import monsoon_scenario, run_scenario  # noqa: E402
from landproof.budget import BUDGET as METER  # noqa: E402
from landproof.sim.replay import Pose  # noqa: E402
from landproof.sim.scene import World, make_world  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="oracle")
    ap.add_argument("--planner", default="scripted", choices=["spark", "scripted"])
    ap.add_argument("--student", default="oracle", help="'oracle' or a checkpoint path")
    ap.add_argument("--glimmer", default="oracle", help="none | oracle | remote")
    ap.add_argument("--link-down", type=float, default=45.0, help="sim time the cloud link drops (s); <0 = never")
    ap.add_argument("--no-intruder", action="store_true")
    ap.add_argument("--world-image")
    ap.add_argument("--world-labels")
    ap.add_argument("--world-gsd", type=float)
    ap.add_argument("--start", default=None, help="x,y in metres")
    ap.add_argument("--alt", type=float, default=80.0)
    ap.add_argument("--speed", type=float, default=1.5, help="video playback speed")
    ap.add_argument("--voice-at", type=float, default=None, help="sim time of the operator's spoken override")
    ap.add_argument("--voice-wav", help="16-bit WAV of the operator speaking (transcribed by Muse Voice Transcribe)")
    ap.add_argument("--voice-lang", default=None, help="e.g. ta, hi, en")
    ap.add_argument("--voice-text", help="skip transcription and use this text (offline)")
    ap.add_argument("--out", default="runs/demo")
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    if a.world_image:
        img = cv2.cvtColor(cv2.imread(a.world_image), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(a.world_labels, cv2.IMREAD_UNCHANGED) if a.world_labels else np.zeros(img.shape[:2], np.uint8)
        world = World(img, lab, a.world_gsd, (img.shape[1] * a.world_gsd / 2, img.shape[0] * a.world_gsd / 2), Path(a.world_image).stem)
    else:
        world = make_world(vocab)
    sc = monsoon_scenario(None if a.link_down < 0 else a.link_down)
    if a.start:
        x, y = map(float, a.start.split(","))
        sc.start = Pose(x, y, a.alt, 0.0)
    else:
        sc.start = Pose(*world.start_xy_m, a.alt, 0.0)
    if a.no_intruder:
        sc.actors = []
    if a.voice_at is not None:
        sc.human_abort_t = a.voice_at
        if a.voice_wav:
            from landproof.config import load_env
            from landproof.meta.voice import transcribe_wav

            load_env()
            sc.voice_text = transcribe_wav(a.voice_wav, a.voice_lang)
            print(f"[voice] Muse Voice Transcribe heard: {sc.voice_text!r}")
        else:
            sc.voice_text, sc.voice_source = a.voice_text, "typed (offline)"
    mode = "SIMULATION (oracle evidence)" if a.evidence == "oracle" else f"SIMULATION ({a.evidence} evidence)"
    s = run_scenario(world, sc, cfg, vocab, sam=make_sam(a.evidence, vocab, cfg), student=make_student(a.student, cfg), planner=a.planner,
                     spark=make_spark(a.planner, cfg), glimmer=make_glimmer(a.glimmer, vocab), out_dir=a.out, mode_label=mode, playback_speed=a.speed)
    print(json.dumps({k: s[k] for k in ("landed", "touchdown_gt", "zones_failed", "reaction_lead_s", "ledger_chain_ok", "video")}, indent=1, default=str))
    print("spend:", METER.report())


if __name__ == "__main__":
    main()
