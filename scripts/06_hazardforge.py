#!/usr/bin/env python3
"""HazardForge: Muse Image invents traps, SAM 3.1 verifies them, strategies get scored.

  python scripts/06_hazardforge.py --frames data/forge_frames/*.jpg --gsd 0.05 --rounds 2 --student checkpoints/student_sam.pt --spark
Writes runs/forge/round{N}/traps.json, a failure table, and train_pairs/ (trap images + SAM labels) for student v2.
Budget: ~$0.01 per Muse Image edit + ~$0.005 SAM per trap.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from landproof.agents.atlas_agent import AtlasAgent  # noqa: E402
from landproof.backends import make_sam, make_spark, make_student  # noqa: E402
from landproof.bench.strategies import FlattestPatch, LandProofStrategy, MaskScorer  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.data.autolabel import autolabel_image, choose_phrases  # noqa: E402
from landproof.data.datasets import camera_frame, load_rgb  # noqa: E402
from landproof.forge.hazardforge import SEED_TRAPS, HazardForge, failure_table  # noqa: E402
from landproof.geometry import Airframe, Camera  # noqa: E402
from landproof.budget import BUDGET as METER  # noqa: E402
from landproof.meta.image import MuseImage  # noqa: E402
from landproof.verifier import Verifier  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--gsd", type=float, required=True, help="source image GSD (m/px)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--per-round", type=int, default=8)
    ap.add_argument("--student", default=None, help="checkpoint to red-team (optional)")
    ap.add_argument("--spark", action="store_true", help="Muse Spark designs round 2+ (else deterministic mutation)")
    ap.add_argument("--out", default="runs/forge")
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    cam, air = Camera.from_cfg(cfg), Airframe.from_cfg(cfg)
    sam = make_sam("meta-api", vocab, cfg)
    spark = make_spark("spark", cfg) if a.spark else None
    forge = HazardForge(vocab, sam, MuseImage(), a.out, spark=spark)
    paths = [p for pat in a.frames for p in glob.glob(pat)]
    frames = []
    for p in paths:
        rgb, _, gsd = camera_frame(load_rgb(Path(p)), None, a.gsd, cam.width, cam.height)
        frames.append((Path(p).stem, rgb, gsd, cam.altitude_for_gsd(gsd, rgb.shape[1])))

    strategies = {"flattest-patch": FlattestPatch(air), "landproof": LandProofStrategy(lambda rgb, ctx: AtlasAgent(vocab, cfg, Verifier(vocab, sam, cfg), cam, air, "scripted"))}
    if a.student:
        st = make_student(a.student, cfg)
        strategies["student"] = MaskScorer("student", vocab, cfg, air, lambda rgb, ctx: st.predict(rgb))

    # per trap: 1 Muse Image edit + ~2 SAM checks + 2 atlases (~25 each) + trap labelling (~20)
    METER.preflight({"muse-image-1.0/image": a.rounds * a.per_round, "sam-3.1/image": a.rounds * a.per_round * 72,
                     "usd": 0.04 * a.rounds if a.spark else 0.0}, f"HazardForge {a.rounds}x{a.per_round} traps")
    specs = [dict(s) for s in SEED_TRAPS][: a.per_round]
    pair_dir = Path(a.out) / "train_pairs"
    (pair_dir / "images").mkdir(parents=True, exist_ok=True)
    (pair_dir / "labels").mkdir(parents=True, exist_ok=True)
    all_tables = []
    for r in range(1, a.rounds + 1):
        traps = []
        for k, spec in enumerate(specs):
            name, rgb, gsd, alt = frames[k % len(frames)]
            # put the trap where a clean scene would have landed: the top certified zone
            atlas = AtlasAgent(vocab, cfg, Verifier(vocab, sam, cfg), cam, air, "scripted").build(rgb, (0, 0, alt, 0))
            if not atlas.entries:
                continue
            trap, frame, hazard = forge.make_trap(rgb, atlas.entries[0].zone.center_px, gsd, spec, f"r{r}_{k:02d}_{name}", r)
            if trap.accepted:
                forge.evaluate(trap, frame, hazard, gsd, alt, strategies, air)
                label, _ = autolabel_image(frame, sam, vocab, choose_phrases(vocab, 2))
                cv2.imwrite(str(pair_dir / "images" / f"{trap.trap_id}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(pair_dir / "labels" / f"{trap.trap_id}.png"), label)
            traps.append(trap)
            print(f"[round {r}] {trap.trap_id} '{trap.phrase}': {'ACCEPTED' if trap.accepted else 'rejected - ' + trap.reject_reason} {trap.results}")
        table = failure_table(traps)
        all_tables.append({"round": r, **table})
        forge.save(traps, f"round{r}_traps.json")
        print(f"[round {r}] {json.dumps(table)}  spend {METER.report()}")
        specs = forge.next_round(traps, a.per_round)
    (Path(a.out) / "failure_tables.json").write_text(json.dumps(all_tables, indent=1))


if __name__ == "__main__":
    main()
