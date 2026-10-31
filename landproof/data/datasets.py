"""Public aerial datasets mapped onto LandProof's classes.

Datasets are NOT redistributed in this repo (licences below). Download them yourself
and point --root at the folder. Folder layouts follow the official releases; if your
copy differs, pass --images/--labels globs.

| dataset            | licence                    | view      | used for                         |
|--------------------|----------------------------|-----------|----------------------------------|
| FloodNet (Track 1) | CC BY-SA 4.0               | nadir     | flood traps, bench, demo stills  |
| ICG Semantic Drone | research only, no redistr. | nadir     | bench (5-30 m), people, dogs     |
| UAVid              | CC BY-NC-SA 4.0            | oblique   | student pre-training only        |
| UDD6               | research / non-commercial  | nadir-ish | student training                 |
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..vocab import IGNORE, Vocabulary

# dataset class index -> LandProof class name (None = ignore)
FLOODNET = {0: None, 1: "building", 2: "building", 3: "water", 4: "road", 5: "water", 6: "tree", 7: "vehicle", 8: "water", 9: "landable"}
ICG = {0: None, 1: "landable", 2: "landable", 3: "landable", 4: "landable", 5: "water", 6: "obstacle", 7: "water", 8: "tree", 9: "building",
       10: "building", 11: "building", 12: "building", 13: "obstacle", 14: "wire_pole", 15: "person", 16: "animal", 17: "vehicle", 18: "vehicle",
       19: "tree", 20: "tree", 21: "landable", 22: "obstacle", 23: None}
UAVID_RGB = {(0, 0, 0): None, (128, 0, 0): "building", (128, 64, 128): "road", (0, 128, 0): "tree", (128, 128, 0): "landable",
             (64, 0, 128): "vehicle", (192, 0, 192): "vehicle", (64, 64, 0): "person"}
UDD6_RGB = {(0, 0, 0): None, (102, 102, 156): "building", (128, 64, 128): "road", (107, 142, 35): "tree", (0, 0, 142): "vehicle", (70, 70, 70): "building"}
UDD6_IDX = {0: None, 1: "building", 2: "road", 3: "tree", 4: "vehicle", 5: "building"}

# Approximate ground sample distance of the full-resolution images (m/px). Override with --gsd.
DEFAULT_GSD = {"floodnet": 0.015, "icg": 0.01, "uavid": None, "udd6": 0.025}


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    label_path: Path | None
    dataset: str


def _glob(root: Path, pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / pattern), recursive=True))


def list_samples(dataset: str, root: str | Path, split: str = "train", images: str | None = None, labels: str | None = None) -> list[Sample]:
    root = Path(root)
    ds = dataset.lower()
    if images:
        imgs = _glob(root, images)
        labs = _glob(root, labels) if labels else []
    elif ds == "floodnet":
        imgs = _glob(root, f"**/{split}/{split}-org-img/*.jpg")
        labs = _glob(root, f"**/{split}/{split}-label-img/*_lab.png")
    elif ds == "icg":
        imgs = _glob(root, "**/original_images/*.jpg")
        labs = _glob(root, "**/label_images_semantic/*.png")
    elif ds == "uavid":
        imgs = _glob(root, f"**/uavid_{split}/seq*/Images/*.png")
        labs = _glob(root, f"**/uavid_{split}/seq*/Labels/*.png")
    elif ds == "udd6":
        imgs = _glob(root, f"**/{split}/src/*.*")
        labs = _glob(root, f"**/{split}/gt/*.png")
    else:
        raise ValueError(f"unknown dataset {dataset}")
    by_stem = {}
    for p in labs:
        stem = p.stem.replace("_lab", "")
        key = f"{p.parent.parent.name}/{stem}" if ds == "uavid" else stem
        by_stem[key] = p
    out = []
    for p in imgs:
        key = f"{p.parent.parent.name}/{p.stem}" if ds == "uavid" else p.stem
        out.append(Sample(key, p, by_stem.get(key), ds))
    return out


def load_rgb(path: Path, max_side: int | None = None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max_side and max(rgb.shape[:2]) > max_side:
        s = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return rgb


def load_label(sample: Sample, vocab: Vocabulary, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load a dataset label and remap it to LandProof class ids (IGNORE = 255)."""
    raw = cv2.imread(str(sample.label_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(sample.label_path)
    out = np.full(raw.shape[:2], IGNORE, np.uint8)
    ids = {c.name: c.id for c in vocab.classes}
    if raw.ndim == 3:  # colour-coded
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        table = UAVID_RGB if sample.dataset == "uavid" else UDD6_RGB
        for color, name in table.items():
            if name is None:
                continue
            m = np.all(rgb == np.array(color, np.uint8), axis=-1)
            out[m] = ids[name]
    else:
        table = {"floodnet": FLOODNET, "icg": ICG, "udd6": UDD6_IDX}[sample.dataset]
        for k, name in table.items():
            if name is not None:
                out[raw == k] = ids[name]
    if shape_hw and out.shape != tuple(shape_hw):
        out = cv2.resize(out, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return out


def camera_frame(rgb: np.ndarray, label: np.ndarray | None, src_gsd: float, cam_w: int = 960, cam_h: int = 720):
    """Resize a nadir image to a camera-sized frame; returns (rgb, label, gsd)."""
    h, w = rgb.shape[:2]
    s = min(cam_w / w, cam_h / h)
    nw, nh = round(w * s), round(h * s)
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    l = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST) if label is not None else None
    return r, l, src_gsd / s
