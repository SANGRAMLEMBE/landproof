"""SAM 3.1 auto-labelling: text concepts -> pixel label maps, no human annotators.

Pixels no phrase claims are IGNORE (255), not "background": the student should not
learn that whatever SAM missed is safe ground.

Two ways to call SAM 3.1:
  image mode  one Responses call per (image, phrase)        $0.0025
  video mode  images packed into one clip, one call per phrase, billed per frame
              ($0.0002/frame, 3 frames per image incl. separators) -> ~$0.0006 per (image, phrase)
Video mode tracks at most 16 objects per frame, so `ab_check` measures per phrase whether
video-mode masks agree with image-mode masks before you trust it for that phrase.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..evidence import union_mask
from ..masks import class_masks_to_label_map, iou
from ..meta.sam import SamBackend
from ..vocab import IGNORE, Vocabulary

FRAMES_PER_IMAGE = 3  # image + 2 black separators


def choose_phrases(vocab: Vocabulary, per_class: int = 2) -> list[str]:
    """Best `per_class` phrases per class by validated IoU (or file order if not validated)."""
    out = []
    for c in vocab.classes:
        ranked = sorted(c.phrases, key=lambda p: -((c.stats.get(p) or {}).get("iou") or 0.0)) if c.stats else list(c.phrases)
        out += ranked[:per_class]
    return out


def compose(per_phrase: dict[str, list], vocab: Vocabulary, shape, min_conf: float = 0.3) -> tuple[np.ndarray, dict]:
    masks: dict[int, np.ndarray] = {}
    stats = {}
    for p, insts in per_phrase.items():
        insts = [i for i in insts if i.confidence is None or i.confidence >= min_conf]
        cid = vocab.class_of(p).id
        m = union_mask(insts, shape)
        masks[cid] = masks.get(cid, np.zeros(shape, bool)) | m
        stats[p] = {"instances": len(insts), "frac": round(float(m.mean()), 4)}
    label = class_masks_to_label_map(masks, vocab, shape)
    stats["_ignore_frac"] = round(float((label == IGNORE).mean()), 4)
    return label, stats


def autolabel_image(rgb: np.ndarray, sam: SamBackend, vocab: Vocabulary, phrases: list[str] | None = None, min_conf: float = 0.3) -> tuple[np.ndarray, dict]:
    phrases = phrases or vocab.all_phrases
    return compose({p: sam.segment(rgb, p) for p in phrases}, vocab, rgb.shape[:2], min_conf)


def estimate_usd(n_images: int, image_phrases: list[str], video_phrases: list[str], chunk: int = 50) -> dict:
    n_chunks = math.ceil(n_images / chunk) if n_images else 0
    frames = n_chunks * len(video_phrases) * max(50, min(chunk, n_images) * FRAMES_PER_IMAGE)
    return {"sam-3.1/image": n_images * len(image_phrases), "sam-3.1/video_frame": frames}


def autolabel_dataset(samples, sam: SamBackend, vocab: Vocabulary, out_dir: str | Path, max_side: int = 1536, workers: int = 6,
                      phrases: list[str] | None = None, video_phrases: list[str] | None = None, chunk: int = 50) -> list[dict]:
    """Label `samples`; phrases in `video_phrases` use the packed-video path, the rest image mode."""
    from .datasets import load_rgb

    phrases = phrases or vocab.all_phrases
    video_phrases = [p for p in (video_phrases or []) if p in phrases]
    image_phrases = [p for p in phrases if p not in video_phrases]
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    todo = [s for s in samples if not (out / "labels" / f"{s.sample_id.replace('/', '_')}.png").exists()]
    rows = [{"id": s.sample_id, "cached": True} for s in samples if s not in todo]

    for c0 in range(0, len(todo), chunk):
        batch = todo[c0 : c0 + chunk]
        imgs = [load_rgb(s.image_path, max_side) for s in batch]
        vid = sam.segment_images_as_video(imgs, video_phrases) if video_phrases else {}

        def one(k):
            per = {p: vid[p][k] for p in video_phrases}
            for p in image_phrases:
                per[p] = sam.segment(imgs[k], p)
            label, stats = compose(per, vocab, imgs[k].shape[:2])
            name = batch[k].sample_id.replace("/", "_")
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out / "labels" / f"{name}.png"), label)
            return {"id": batch[k].sample_id, **stats}

        with ThreadPoolExecutor(workers) as ex:
            rows += list(ex.map(one, range(len(batch))))
        print(f"  labelled {min(c0 + chunk, len(todo))}/{len(todo)}", flush=True)
    (out / "manifest.json").write_text(json.dumps({"phrases": phrases, "video_phrases": video_phrases, "rows": rows}, indent=1))
    return rows


def ab_check(images: list[np.ndarray], sam, vocab: Vocabulary, phrases: list[str], min_conf: float = 0.3) -> dict:
    """Per phrase: IoU between image-mode and video-mode masks (union over images)."""
    vid = sam.segment_images_as_video(images, phrases)
    report = {}
    for p in phrases:
        ious, n_img, n_vid = [], 0, 0
        for k, im in enumerate(images):
            a = [i for i in sam.segment(im, p) if i.confidence is None or i.confidence >= min_conf]
            b = [i for i in vid[p][k] if i.confidence is None or i.confidence >= min_conf]
            n_img, n_vid = n_img + len(a), n_vid + len(b)
            ma, mb = union_mask(a, im.shape[:2]), union_mask(b, im.shape[:2])
            if ma.any() or mb.any():
                ious.append(iou(ma, mb))
        report[p] = {"agreement_iou": round(float(np.mean(ious)), 3) if ious else None, "instances_image": n_img, "instances_video": n_vid}
    return report


def phrase_quality(rgb: np.ndarray, gt: np.ndarray, phrase: str, sam: SamBackend, vocab: Vocabulary) -> dict:
    """IoU / precision / recall of one phrase against the GT mask of its class."""
    cid = vocab.class_of(phrase).id
    valid = gt != IGNORE
    pred = union_mask(sam.segment(rgb, phrase), rgb.shape[:2]) & valid
    g = (gt == cid) & valid
    tp = float((pred & g).sum())
    return {
        "iou": iou(pred, g) if (pred | g).any() else float("nan"),
        "precision": tp / pred.sum() if pred.sum() else float("nan"),
        "recall": tp / g.sum() if g.sum() else float("nan"),
        "gt_px": int(g.sum()),
    }
