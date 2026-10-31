"""Mask utilities shared by the scorer, verifier, student and benchmark."""

from __future__ import annotations

import cv2
import numpy as np

from .vocab import IGNORE, Vocabulary

# When pseudo-labels overlap, the more dangerous class wins.
PRIORITY = ["person", "animal", "vehicle", "wire_pole", "water", "obstacle", "tree", "building", "road", "landable"]


def label_map_to_class_masks(label: np.ndarray, vocab: Vocabulary, hazards_only: bool = True) -> dict[int, np.ndarray]:
    classes = vocab.hazard_classes if hazards_only else vocab.classes
    return {c.id: label == c.id for c in classes}


def class_masks_to_label_map(masks: dict[int, np.ndarray], vocab: Vocabulary, shape: tuple[int, int]) -> np.ndarray:
    """Compose per-class masks into one label map; unclaimed pixels are IGNORE."""
    label = np.full(shape, IGNORE, dtype=np.uint8)
    order = [vocab.by_name[n].id for n in reversed(PRIORITY) if n in vocab.by_name]
    for cid in order:  # low priority first, high priority overwrites
        m = masks.get(cid)
        if m is not None:
            label[m] = cid
    return label


def downsample_any(mask: np.ndarray, factor: int) -> np.ndarray:
    """Conservative downsample: a cell is set if any source pixel in it is set."""
    if factor <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    H, W = int(np.ceil(h / factor)), int(np.ceil(w / factor))
    pad = np.zeros((H * factor, W * factor), dtype=bool)
    pad[:h, :w] = mask
    return pad.reshape(H, factor, W, factor).any(axis=(1, 3))


def resize_nearest(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def confusion_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore: int = IGNORE) -> tuple[float, np.ndarray]:
    valid = gt != ignore
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    k = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
    cm = np.bincount(num_classes * g[k] + p[k], minlength=num_classes**2).reshape(num_classes, num_classes)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = inter / union
    present = union > 0
    return float(np.nanmean(per_class[present])) if present.any() else float("nan"), per_class
