#!/usr/bin/env python3
"""LandBench: compare landing-site strategies against ground truth.

  python scripts/07_landbench.py --source synthetic --n 60                       # offline, oracle evidence
  python scripts/07_landbench.py --source floodnet --root data/FloodNet --split test --n 100 --evidence meta-api --student checkpoints/student_sam.pt
  python scripts/07_landbench.py --source icg --root data/semantic_drone --n 80 --evidence meta-api
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from landproof.agents.atlas_agent import AtlasAgent  # noqa: E402
from landproof.backends import make_sam, make_spark, make_student  # noqa: E402
from landproof.bench.landbench import Frame, run_bench  # noqa: E402
from landproof.bench.strategies import BlindLand, FlattestPatch, LandProofStrategy, MaskScorer  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.data.datasets import DEFAULT_GSD, camera_frame, list_samples, load_label, load_rgb  # noqa: E402
from landproof.geometry import Airframe, Camera  # noqa: E402
from landproof.budget import BUDGET as METER  # noqa: E402
from landproof.meta.sam import OracleSam  # noqa: E402
from landproof.verifier import Verifier  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def synthetic_frames(vocab, cam, n, seed=0):
    from landproof.sim.replay import Pose, ReplaySim
    from landproof.sim.scene import make_world

    rng = np.random.default_rng(seed)
    frames = []
    for w_seed in range(max(1, n // 20)):
        world = make_world(vocab, seed=7 + w_seed)
        sim = ReplaySim(world, cam, vocab, Pose(80, 60, 40, 0))
        for k in range(min(20, n - len(frames))):
            alt = rng.uniform(25, 70)
            x, y = rng.uniform(30, 130), rng.uniform(25, 95)
            rgb, label, _ = sim.render(Pose(x, y, alt, rng.uniform(0, 360)), 0)
            frames.append(Frame(f"syn{w_seed}_{k}", rgb, label, cam.gsd(alt), alt, "synthetic"))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gsd", type=float, default=None)
    ap.add_argument("--evidence", default="oracle")
    ap.add_argument("--planner", default="scripted", choices=["spark", "scripted"])
    ap.add_argument("--student", default=None)
    ap.add_argument("--out", default="runs/landbench")
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    cam, air = Camera.from_cfg(cfg), Airframe.from_cfg(cfg)
    if a.source == "synthetic":
        frames = synthetic_frames(vocab, cam, a.n)
    else:
        gsd0 = a.gsd or DEFAULT_GSD[a.source]
        frames = []
        for s in [s for s in list_samples(a.source, a.root, a.split) if s.label_path][: a.n]:
            src = load_rgb(s.image_path)
            rgb, lab, gsd = camera_frame(src, load_label(s, vocab, src.shape[:2]), gsd0, cam.width, cam.height)
            frames.append(Frame(s.sample_id, rgb, lab, gsd, cam.altitude_for_gsd(gsd, rgb.shape[1]), a.source))
    print(f"{len(frames)} frames from {a.source}")

    if a.evidence == "meta-api":  # ~10 full-frame phrases + 3 zones x 5 zoomed claims per frame; cached frames are free
        METER.preflight({"sam-3.1/image": len(frames) * 25, "usd": len(frames) * (0.08 if a.planner == "spark" else 0.0)}, f"LandBench {len(frames)} frames")
    sam = make_sam(a.evidence, vocab, cfg)
    spark = make_spark(a.planner, cfg)

    def agent_factory(rgb, ctx):
        if isinstance(sam, OracleSam):
            sam.register(rgb, ctx["gt"])
        return AtlasAgent(vocab, cfg, Verifier(vocab, sam, cfg), cam, air, planner=a.planner, spark=spark)

    fixed = [c.name for c in vocab.hazard_classes if c.fixed_vocab]
    strategies = [BlindLand(), FlattestPatch(air),
                  MaskScorer("fixed-vocab (UAVid classes, perfect masks)", vocab, cfg, air, lambda rgb, ctx: ctx["gt"], classes=fixed)]
    if a.student:
        st = make_student(a.student, cfg)
        strategies.append(MaskScorer("student (open vocab, distilled)", vocab, cfg, air, lambda rgb, ctx: st.predict(rgb)))
    strategies.append(LandProofStrategy(agent_factory))
    summary = run_bench(frames, strategies, vocab, air.diameter_m / 2 + air.touchdown_margin_m, out_dir=a.out)
    print((Path(a.out) / "landbench_table.md").read_text())
    print(f"evidence: {a.evidence}   spend {METER.report()}")


if __name__ == "__main__":
    main()
