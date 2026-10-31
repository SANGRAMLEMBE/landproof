#!/usr/bin/env python3
"""Score every vocabulary phrase against labeled frames; write configs/hazards.validated.yaml.

  python scripts/01_validate_vocab.py --dataset floodnet --root data/FloodNet --n 20
  python scripts/01_validate_vocab.py --dataset icg --root data/semantic_drone --n 20 --merge

A phrase is kept if IoU >= 0.30, or (for small dynamic classes) recall >= 0.5 with
precision >= 0.3. The best phrase of each class is always kept (flagged weak if below).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.backends import make_sam  # noqa: E402
from landproof.config import CONFIG_DIR, load_config  # noqa: E402
from landproof.data.autolabel import phrase_quality  # noqa: E402
from landproof.data.datasets import list_samples, load_label, load_rgb  # noqa: E402
from landproof.meta.client import METER  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--evidence", default="meta-api")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--merge", action="store_true", help="merge with an existing hazards.validated.yaml")
    a = ap.parse_args()
    cfg = load_config()
    vocab = Vocabulary.load(CONFIG_DIR / "hazards.yaml")
    sam = make_sam(a.evidence, vocab, cfg)
    samples = [s for s in list_samples(a.dataset, a.root, a.split) if s.label_path][: a.n]
    if not samples:
        sys.exit("no labeled samples found - check --root/--split")
    print(f"{len(samples)} frames x {len(vocab.all_phrases)} phrases = {len(samples) * len(vocab.all_phrases)} SAM calls (~${len(samples) * len(vocab.all_phrases) * 0.0025:.2f})")

    acc = {p: [] for p in vocab.all_phrases}
    for s in samples:
        rgb = load_rgb(s.image_path, a.max_side)
        gt = load_label(s, vocab, rgb.shape[:2])
        present = set(np.unique(gt).tolist())
        for p in vocab.all_phrases:
            if vocab.class_of(p).id not in present:
                continue  # only score phrases whose class appears in this frame
            acc[p].append(phrase_quality(rgb, gt, p, sam, vocab))
        print(f"  {s.sample_id}: done ({METER.report()})")

    prev = Vocabulary.load(CONFIG_DIR / "hazards.validated.yaml") if a.merge and (CONFIG_DIR / "hazards.validated.yaml").exists() else None
    for c in vocab.classes:
        stats = {}
        for p in c.phrases:
            rows = acc[p]
            agg = {k: float(np.nanmean([r[k] for r in rows])) if rows else float("nan") for k in ("iou", "precision", "recall")}
            agg["n"] = len(rows)
            if prev and p in prev.by_name[c.name].stats:
                agg = {"dataset": a.dataset, **agg, "previous": prev.by_name[c.name].stats[p]}
            stats[p] = {k: (None if isinstance(v, float) and math.isnan(v) else (round(v, 4) if isinstance(v, float) else v)) for k, v in agg.items()}
        small = c.dynamic
        keep = [p for p in c.phrases if stats[p]["n"] and ((stats[p]["iou"] or 0) >= 0.30 or (small and (stats[p]["recall"] or 0) >= 0.5 and (stats[p]["precision"] or 0) >= 0.3))]
        if not keep:
            scored = [p for p in c.phrases if stats[p]["n"]]
            keep = [max(scored, key=lambda p: stats[p]["iou"] or 0)] if scored else c.phrases[:1]
            stats[keep[0]]["weak"] = True
        c.stats = {p: stats[p] for p in c.phrases}
        print(f"{c.name:10s} keep {keep}  dropped {[p for p in c.phrases if p not in keep]}")
        c.phrases = keep
    vocab.dump(CONFIG_DIR / "hazards.validated.yaml")
    print(f"wrote {CONFIG_DIR / 'hazards.validated.yaml'}   spend {METER.report()}")


if __name__ == "__main__":
    main()
