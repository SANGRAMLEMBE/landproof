#!/usr/bin/env python3
"""Run LandProof in shadow mode on real drone footage (nothing is flown).

  # 1. precompute SAM 3.1 video tracks for movers (~$0.20 per 1000 frames per phrase)
  # 2. certify on frame 0, watch every frame, render HUD
  python scripts/08_shadow_mode.py --video DJI_0042.MP4 --srt DJI_0042.SRT --planner spark --stride 2
  python scripts/08_shadow_mode.py --video clip.mp4 --alt-start 35 --alt-end 4 --student checkpoints/student_sam.pt
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from landproof.backends import make_glimmer, make_sam, make_spark, make_student  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.budget import BUDGET as METER  # noqa: E402
from landproof.shadow import run_shadow  # noqa: E402
from landproof.sim.footage import FootageConfig, PrecomputedSamStudent  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402

MOVERS = ["person", "car", "motorcycle", "dog", "cow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--srt")
    ap.add_argument("--alt-start", type=float)
    ap.add_argument("--alt-end", type=float)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--planner", default="spark", choices=["spark", "scripted"])
    ap.add_argument("--student", default="sam-video", help="'sam-video' (precomputed SAM 3.1 tracks) or a checkpoint")
    ap.add_argument("--glimmer", default="none")
    ap.add_argument("--link-down", type=float, default=None)
    ap.add_argument("--zone", default=None)
    ap.add_argument("--out", default="runs/shadow")
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    sam = make_sam("meta-api", vocab, cfg)
    if a.student == "sam-video":
        cache = Path(a.out) / "sam_video_tracks.pkl"
        if cache.exists():
            tracks = pickle.loads(cache.read_bytes())
        else:
            cap = cv2.VideoCapture(a.video)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            METER.preflight({"sam-3.1/video_frame": max(50, n) * len(MOVERS), "sam-3.1/image": 30}, f"shadow mode ({n} frames x {len(MOVERS)} movers)")
            tracks = {p: sam.segment_video(a.video, p) for p in MOVERS}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(tracks))
        cap = cv2.VideoCapture(a.video)
        vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        student = PrecomputedSamStudent(tracks, vocab, (vw, vh), (cfg["camera"]["width"], cfg["camera"]["height"]))
    else:
        student = make_student(a.student, cfg)
    fc = FootageConfig(a.video, a.srt, a.alt_start, a.alt_end, a.stride, cfg["camera"]["width"], cfg["camera"]["height"], a.start_s, a.end_s)
    s = run_shadow(fc, cfg, vocab, sam=sam, student=student, planner=a.planner, spark=make_spark(a.planner, cfg), zone=a.zone, out_dir=a.out,
                   link_down_t=a.link_down, glimmer=make_glimmer(a.glimmer, vocab))
    print(json.dumps({k: s[k] for k in ("zone", "certificate", "final_state", "ledger_chain_ok", "output")}, indent=1))
    for d in s["decisions"]:
        print(f"  t={d['t']:6.2f}  {d['event']}")
    print("spend:", METER.report())


if __name__ == "__main__":
    main()
