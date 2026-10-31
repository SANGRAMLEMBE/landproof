"""Distilled onboard student: LR-ASPP MobileNetV3-Large trained on SAM 3.1 pseudo-labels.

torchvision code and ImageNet backbone weights are BSD-3 licensed, so the student has
no non-commercial strings attached (unlike SegFormer's NVIDIA weights or Ultralytics'
AGPL). Class 0 = landable surface, 1..9 = hazards, 255 = ignore (no SAM phrase fired).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class StudentBackend:
    name = "student"

    def predict(self, rgb: np.ndarray, gt_for_oracle_only: np.ndarray | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class OracleStudent(StudentBackend):
    """Ground-truth 'student' for offline logic tests, with realistic failure modes."""

    name = "oracle-student"

    def __init__(self, min_area_px: int = 0, miss_prob: float = 0.0, seed: int = 0):
        self.min_area_px = min_area_px
        self.miss_prob = miss_prob
        self.rng = np.random.default_rng(seed)

    def predict(self, rgb, gt_for_oracle_only=None):
        if gt_for_oracle_only is None:
            raise ValueError("OracleStudent needs the GT label (tests only)")
        lab = gt_for_oracle_only.copy()
        if self.min_area_px or self.miss_prob:
            from scipy import ndimage

            for cid in np.unique(lab):
                if cid == 0 or cid == 255:
                    continue
                comp, n = ndimage.label(lab == cid)
                for k in range(1, n + 1):
                    m = comp == k
                    if m.sum() < self.min_area_px or self.rng.random() < self.miss_prob:
                        lab[m] = 0
        return lab


def build_student(num_classes: int, pretrained_backbone: bool = True):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    wb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    try:
        return lraspp_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=num_classes)
    except Exception as e:  # offline: no access to download.pytorch.org
        if wb is None:
            raise
        print(f"WARNING: could not fetch ImageNet backbone weights ({type(e).__name__}); training from scratch")
        return lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)


def preprocess(rgb: np.ndarray, input_px: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    s = input_px / max(h, w)
    nh, nw = int(round(h * s / 16) * 16), int(round(w * s / 16) * 16)
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1), s


class TorchStudent(StudentBackend):
    name = "student-lraspp"

    def __init__(self, checkpoint: str | Path, device: str | None = None, input_px: int = 512):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.num_classes = int(ck["num_classes"])
        self.model = build_student(self.num_classes, pretrained_backbone=False)
        self.model.load_state_dict(ck["state_dict"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.input_px = input_px
        self.torch = torch
        self.last_ms = float("nan")

    def predict(self, rgb, gt_for_oracle_only=None):
        torch = self.torch
        x, _ = preprocess(rgb, self.input_px)
        t0 = time.perf_counter()
        with torch.inference_mode():
            xt = torch.from_numpy(x)[None].to(self.device)
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(xt)["out"]
            else:
                out = self.model(xt)["out"]
            pred = out.argmax(1)[0].to(torch.uint8).cpu().numpy()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
