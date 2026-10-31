#!/usr/bin/env python3
"""SAM 3.1 auto-labelling of a whole dataset split (no human annotation).

  python scripts/02_autolabel.py --dataset floodnet --root data/FloodNet --split train --out data/pseudo/floodnet_train --limit 400
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.backends import make_sam  # noqa: E402
from landproof.config import load_config  # noqa: E402
from landproof.data.autolabel import autolabel_dataset  # noqa: E402
from landproof.data.datasets import list_samples  # noqa: E402
from landproof.meta.client import METER  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--evidence", default="meta-api")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    samples = list_samples(a.dataset, a.root, a.split)[: a.limit]
    n_calls = len(samples) * len(vocab.all_phrases)
    print(f"{len(samples)} images x {len(vocab.all_phrases)} phrases = {n_calls} SAM calls (~${n_calls * 0.0025:.2f}); cached calls are free")
    rows = autolabel_dataset(samples, make_sam(a.evidence, vocab, cfg), vocab, a.out, a.max_side, a.workers)
    print(f"labelled {len(rows)} images -> {a.out}   spend {METER.report()}")


if __name__ == "__main__":
    main()
