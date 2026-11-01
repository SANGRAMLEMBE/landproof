#!/usr/bin/env python3
"""SAM 3.1 auto-labelling of a dataset split (no human annotation), inside the budget cap.

  # 1) ~$1: does packed-video labelling agree with per-image labelling? (writes cache/video_ab.json)
  python scripts/02_autolabel.py --dataset floodnet --root data/FloodNet --split train --ab-check 10
  # 2) bulk: phrases that passed the A/B check use video mode, the rest image mode
  python scripts/02_autolabel.py --dataset floodnet --root data/FloodNet --split train --out data/pseudo/floodnet_train --limit 300 --mode auto

--per-class N keeps the N best validated phrases per class (default 2 -> 20 phrases).
Cached calls are free; re-running resumes where it stopped.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.backends import make_sam  # noqa: E402
from landproof.budget import BUDGET  # noqa: E402
from landproof.config import REPO_ROOT, load_config  # noqa: E402
from landproof.data.autolabel import ab_check, autolabel_dataset, choose_phrases, estimate_usd  # noqa: E402
from landproof.data.datasets import list_samples, load_rgb  # noqa: E402
from landproof.vocab import Vocabulary  # noqa: E402

AB_FILE = REPO_ROOT / "cache" / "video_ab.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--mode", default="auto", choices=["image", "video", "auto"])
    ap.add_argument("--ab-check", type=int, default=0, help="run the image-vs-video agreement check on N images and exit")
    ap.add_argument("--ab-threshold", type=float, default=0.7)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--evidence", default="meta-api")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg, vocab = load_config(), Vocabulary.load()
    sam = make_sam(a.evidence, vocab, cfg)
    phrases = choose_phrases(vocab, a.per_class)
    samples = list_samples(a.dataset, a.root, a.split)[: a.limit]
    if not samples:
        sys.exit("no images found - check --root/--split")
    print(f"{len(phrases)} phrases: {phrases}")

    if a.ab_check:
        imgs = [load_rgb(s.image_path, a.max_side) for s in samples[: a.ab_check]]
        BUDGET.preflight({"sam-3.1/image": len(imgs) * len(phrases), "sam-3.1/video_frame": len(phrases) * max(50, 3 * len(imgs))}, "A/B check")
        rep = ab_check(imgs, sam, vocab, phrases)
        AB_FILE.parent.mkdir(parents=True, exist_ok=True)
        AB_FILE.write_text(json.dumps(rep, indent=1))
        ok = [p for p, r in rep.items() if r["agreement_iou"] is not None and r["agreement_iou"] >= a.ab_threshold]
        for p, r in rep.items():
            print(f"  {p:28s} agreement {r['agreement_iou']}  inst image/video {r['instances_image']}/{r['instances_video']}  {'VIDEO OK' if p in ok else 'image mode'}")
        print(f"-> {AB_FILE}: {len(ok)}/{len(phrases)} phrases can use video mode.  {BUDGET.report()}")
        return

    if not a.out:
        sys.exit("--out is required")
    if a.mode == "image":
        video = []
    elif a.mode == "video":
        video = phrases
    else:
        if not AB_FILE.exists():
            sys.exit("--mode auto needs cache/video_ab.json: run with --ab-check 10 first (or use --mode image)")
        rep = json.loads(AB_FILE.read_text())
        video = [p for p in phrases if (rep.get(p) or {}).get("agreement_iou") is not None and rep[p]["agreement_iou"] >= a.ab_threshold]
    todo = [s for s in samples if not (Path(a.out) / "labels" / f"{s.sample_id.replace('/', '_')}.png").exists()]
    est = estimate_usd(len(todo), [p for p in phrases if p not in video], video, a.chunk)
    BUDGET.preflight(est, f"autolabel {len(todo)} images ({len(video)} video-mode phrases)")
    rows = autolabel_dataset(samples, sam, vocab, a.out, a.max_side, a.workers, phrases, video, a.chunk)
    print(f"labelled {len(rows)} images -> {a.out}   {BUDGET.report()}")


if __name__ == "__main__":
    main()
