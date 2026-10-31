"""SAM 3.1 auto-labelling: text concepts -> pixel label maps, no human annotators.

Pixels no phrase claims are IGNORE (255), not "background": the student should not
learn that whatever SAM missed is safe ground.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..evidence import union_mask
from ..masks import class_masks_to_label_map, iou
from ..meta.sam import SamBackend
from ..vocab import IGNORE, Vocabulary


def autolabel_image(rgb: np.ndarray, sam: SamBackend, vocab: Vocabulary, phrases: list[str] | None = None, min_conf: float = 0.3) -> tuple[np.ndarray, dict]:
    phrases = phrases or vocab.all_phrases
    masks: dict[int, np.ndarray] = {}
    stats = {}
    for p in phrases:
        insts = [i for i in sam.segment(rgb, p) if i.confidence is None or i.confidence >= min_conf]
        cid = vocab.class_of(p).id
        m = union_mask(insts, rgb.shape[:2])
        masks[cid] = masks.get(cid, np.zeros(rgb.shape[:2], bool)) | m
        stats[p] = {"instances": len(insts), "frac": round(float(m.mean()), 4)}
    label = class_masks_to_label_map(masks, vocab, rgb.shape[:2])
    stats["_ignore_frac"] = round(float((label == IGNORE).mean()), 4)
    return label, stats


def autolabel_dataset(samples, sam: SamBackend, vocab: Vocabulary, out_dir: str | Path, max_side: int = 1536, workers: int = 6, phrases=None) -> list[dict]:
    from .datasets import load_rgb

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    def one(s):
        lp = out / "labels" / f"{s.sample_id.replace('/', '_')}.png"
        ip = out / "images" / f"{s.sample_id.replace('/', '_')}.jpg"
        if lp.exists() and ip.exists():
            return {"id": s.sample_id, "cached": True}
        rgb = load_rgb(s.image_path, max_side)
        label, stats = autolabel_image(rgb, sam, vocab, phrases)
        cv2.imwrite(str(ip), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(lp), label)
        return {"id": s.sample_id, **stats}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, samples))
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))
    return rows


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
