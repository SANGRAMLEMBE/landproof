#!/usr/bin/env python3
"""Go/no-go: can this machine reach every Meta model LandProof needs?

  python scripts/00_check_access.py            # models list + SAM 3.1 + Muse Spark (~$0.01)
  python scripts/00_check_access.py --image    # also one Muse Image edit (~$0.01)
  python scripts/00_check_access.py --glimmer  # also ping the Glimmer server (GLIMMER_BASE_URL)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.config import load_config, load_env  # noqa: E402
from landproof.meta.client import METER, openai_client  # noqa: E402
from landproof.sim.scene import make_world  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def step(name, fn):
    t = time.perf_counter()
    try:
        out = fn()
        print(f"[ OK ] {name} ({time.perf_counter() - t:.1f}s) {out}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--glimmer", action="store_true")
    a = ap.parse_args()
    load_env()
    cfg, vocab = load_config(), Vocabulary.load()
    w = make_world(vocab, gsd=0.1)
    crop = w.image[500:900, 700:1200]  # junction with cars, synthetic
    results = []

    results.append(step("list models", lambda: [m.id for m in openai_client().models.list().data]))

    def sam():
        from landproof.meta.sam import MetaApiSam

        insts = MetaApiSam(cache_dir=None).segment(crop, "car")
        return f"{len(insts)} instance(s) of 'car' (synthetic image; 0-3 is plausible)"

    results.append(step("sam-3.1 segmentation", sam))

    def spark():
        from landproof.meta.spark import SparkClient, image_part

        s = SparkClient(model=cfg["models"]["spark"], effort="low", max_tokens=3000)
        return s.json([{"role": "user", "content": [{"type": "text", "text": "Name up to 3 hazards a landing drone should avoid in this image."}, image_part(crop)]}],
                      {"type": "object", "properties": {"hazards": {"type": "array", "items": {"type": "string"}}}, "required": ["hazards"]})

    results.append(step("muse-spark-1.3 vision + structured output", spark))

    if a.image:
        def img():
            from landproof.meta.image import MuseImage

            out = MuseImage().edit([crop], "Add a parked yellow auto rickshaw on the road, seen from directly above. Keep everything else unchanged.")
            return f"edited image {out.shape}"

        results.append(step("muse-image-1.0 edit", img))
    if a.glimmer:
        def gl():
            from landproof.meta.glimmer import GlimmerGate

            return GlimmerGate()(crop, (250, 200), 40, 0.1)

        results.append(step("muse-glimmer (vLLM)", gl))
    print(f"\nspend this run: {METER.report()}")
    print("ALL GOOD" if all(results) else "Some checks failed - see README > Kill criteria for fallbacks.")


if __name__ == "__main__":
    main()
