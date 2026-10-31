#!/usr/bin/env python3
"""Train the onboard student on SAM 3.1 pseudo-labels (or GT, for the comparison), evaluate on GT.

  # zero-annotator student: SAM labels for training, human labels only for evaluation
  python scripts/03_train_student.py --train-dir data/pseudo/floodnet_train --val-dataset floodnet --val-root data/FloodNet --out checkpoints/student_sam.pt
  # reference student trained on human labels
  python scripts/03_train_student.py --train-dataset floodnet --train-root data/FloodNet --val-dataset floodnet --val-root data/FloodNet --out checkpoints/student_gt.pt
  # add HazardForge traps (round-1 failures) for student v2
  python scripts/03_train_student.py --train-dir data/pseudo/floodnet_train --extra-dir runs/forge/train_pairs --out checkpoints/student_v2.pt ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from landproof.data.datasets import list_samples, load_label, load_rgb  # noqa: E402
from landproof.perception.train import measure_fps, train_student  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402


def dir_pairs(d):
    d = Path(d)
    return [(p, d / "labels" / (p.stem + ".png")) for p in sorted((d / "images").glob("*.jpg")) if (d / "labels" / (p.stem + ".png")).exists()]


def gt_pairs(dataset, root, split, vocab, limit, cache_dir):
    out = []
    cache = Path(cache_dir)
    (cache / "images").mkdir(parents=True, exist_ok=True)
    (cache / "labels").mkdir(parents=True, exist_ok=True)
    for s in [s for s in list_samples(dataset, root, split) if s.label_path][:limit]:
        ip, lp = cache / "images" / f"{s.sample_id.replace('/', '_')}.jpg", cache / "labels" / f"{s.sample_id.replace('/', '_')}.png"
        if not lp.exists():
            rgb = load_rgb(s.image_path, 1536)
            cv2.imwrite(str(ip), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(lp), load_label(s, vocab, rgb.shape[:2]))
        out.append((ip, lp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", help="pseudo-label dir from 02_autolabel.py")
    ap.add_argument("--train-dataset")
    ap.add_argument("--train-root")
    ap.add_argument("--extra-dir", action="append", default=[], help="additional image/label dirs (e.g. HazardForge traps)")
    ap.add_argument("--val-dataset", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vocab = Vocabulary.load()
    if a.train_dir:
        train, tag = dir_pairs(a.train_dir), "sam-pseudo-labels"
    else:
        train, tag = gt_pairs(a.train_dataset, a.train_root, "train", vocab, a.limit, f"cache/gt_{a.train_dataset}_train"), "ground-truth"
    for d in a.extra_dir:
        train += dir_pairs(d)
    val = gt_pairs(a.val_dataset, a.val_root, a.val_split, vocab, None, f"cache/gt_{a.val_dataset}_{a.val_split}")
    print(f"train {len(train)} pairs ({tag}), val {len(val)} GT pairs")
    rep = train_student(train, val, vocab.num_classes, a.out, a.epochs, a.batch_size, a.lr, a.crop, workers=a.workers,
                        extra_meta={"train_labels": tag, "extra": a.extra_dir})
    print(f"best val mIoU (vs human labels): {rep['best_val_miou']:.4f}")
    print("speed:", measure_fps(a.out))


if __name__ == "__main__":
    main()
