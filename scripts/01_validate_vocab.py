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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.backends import make_sam  # noqa: E402
from landproof.config import CONFIG_DIR, load_config  # noqa: E402
from landproof.data.autolabel import phrase_quality  # noqa: E402
from landproof.data.datasets import list_samples, load_label, load_rgb  # noqa: E402
from landproof.budget import BUDGET as METER  # noqa: E402
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
    ap.add_argument("--workers", type=int, default=6, help="parallel SAM calls per frame")
    a = ap.parse_args()
    cfg = load_config()
    vocab = Vocabulary.load(CONFIG_DIR / "hazards.yaml")
    sam = make_sam(a.evidence, vocab, cfg)
    samples = [s for s in list_samples(a.dataset, a.root, a.split) if s.label_path][: a.n]
    if not samples:
        sys.exit("no labeled samples found - check --root/--split")
    print(f"{len(samples)} frames x {len(vocab.all_phrases)} phrases = {len(samples) * len(vocab.all_phrases)} SAM calls (~${len(samples) * len(vocab.all_phrases) * 0.0025:.2f})")

    if a.evidence == "meta-api":  # only phrases whose class is present in a frame are scored (~70%)
        METER.preflight({"sam-3.1/image": len(samples) * len(vocab.all_phrases) * 0.7}, "vocabulary validation")
    acc = {p: [] for p in vocab.all_phrases}
    for s in samples:
        rgb = load_rgb(s.image_path, a.max_side)
        gt = load_label(s, vocab, rgb.shape[:2])
        present = set(np.unique(gt).tolist())
        todo = [p for p in vocab.all_phrases if vocab.class_of(p).id in present]  # only phrases whose class appears in this frame
        with ThreadPoolExecutor(a.workers) as ex:
            for p, q in zip(todo, ex.map(lambda p: phrase_quality(rgb, gt, p, sam, vocab), todo)):
                acc[p].append(q)
        print(f"  {s.sample_id}: done ({METER.report()})")

    prev = None
    if a.merge and (CONFIG_DIR / "hazards.validated.yaml").exists():
        prev = {c.name: c.stats for c in Vocabulary.load(CONFIG_DIR / "hazards.validated.yaml").classes}
    for c in vocab.classes:
        stats = {}
        for p in c.phrases:
            rows = list(acc[p])
            old = (prev or {}).get(c.name, {}).get(p) or {}
            # merge with an earlier dataset: weight each metric by its number of frames
            n_old = int(old.get("n") or 0)
            agg = {}
            for k in ("iou", "precision", "recall"):
                vals = [r[k] for r in rows if not math.isnan(r[k])]
                tot, n = sum(vals), len(vals)
                if n_old and old.get(k) is not None:
                    tot, n = tot + old[k] * n_old, n + n_old
                agg[k] = round(tot / n, 4) if n else None
            agg["n"] = len(rows) + n_old
            agg["datasets"] = sorted(set(old.get("datasets", [])) | ({a.dataset} if rows else set()))
            stats[p] = agg
        scored = [p for p in c.phrases if stats[p]["n"]]
        if not scored:
            # no labelled example of this class in the data: keep every phrase, flagged unvalidated
            keep = list(c.phrases)
            for p in keep:
                stats[p]["unvalidated"] = True
        else:
            keep = [p for p in scored if (stats[p]["iou"] or 0) >= 0.30 or (c.dynamic and (stats[p]["recall"] or 0) >= 0.5 and (stats[p]["precision"] or 0) >= 0.3)]
            if not keep:
                keep = [max(scored, key=lambda p: stats[p]["iou"] or 0)]
                stats[keep[0]]["weak"] = True
            keep += [p for p in c.phrases if not stats[p]["n"]]  # untested phrases are not punished
        c.stats = {p: stats[p] for p in c.phrases}
        print(f"{c.name:10s} keep {keep}  dropped {[p for p in c.phrases if p not in keep]}")
        c.phrases = keep
    vocab.dump(CONFIG_DIR / "hazards.validated.yaml")
    print(f"wrote {CONFIG_DIR / 'hazards.validated.yaml'}   spend {METER.report()}")


if __name__ == "__main__":
    main()
