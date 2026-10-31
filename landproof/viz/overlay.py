"""Static overlays: numbered candidate zones (shown to Muse Spark) and label maps."""

from __future__ import annotations

import cv2
import numpy as np

from ..vocab import IGNORE

PALETTE = {
    0: (0, 0, 0),         # landable (not drawn)
    1: (255, 64, 64),     # person
    2: (255, 160, 0),     # vehicle
    3: (255, 220, 120),   # animal
    4: (40, 140, 255),    # water
    5: (255, 0, 200),     # wire_pole
    6: (200, 120, 255),   # obstacle
    7: (40, 200, 90),     # tree
    8: (160, 160, 160),   # building
    9: (120, 100, 140),   # road
}


def label_overlay(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.45, skip=(0, IGNORE)) -> np.ndarray:
    out = rgb.copy()
    color = np.zeros_like(rgb)
    m = np.zeros(label.shape, bool)
    for cid, c in PALETTE.items():
        if cid in skip:
            continue
        sel = label == cid
        color[sel] = c
        m |= sel
    out[m] = (out[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return out


def zones_overlay(rgb: np.ndarray, zones, gsd: float | None = None, certs: dict | None = None) -> np.ndarray:
    out = rgb.copy()
    th = max(2, out.shape[1] // 480)
    for z in zones:
        c = (0, 255, 255)
        if certs and z.zone_id in certs:
            c = (60, 255, 120) if certs[z.zone_id].passed else (255, 70, 70)
        cx, cy = int(z.center_px[0]), int(z.center_px[1])
        cv2.circle(out, (cx, cy), int(z.radius_px), c, th, cv2.LINE_AA)
        fs = max(0.6, out.shape[1] / 1400)
        cv2.putText(out, z.zone_id, (cx + int(z.radius_px) + 4, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(out, z.zone_id, (cx + int(z.radius_px) + 4, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    return out
