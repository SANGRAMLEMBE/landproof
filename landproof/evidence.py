"""Segmentation evidence in the same shape SAM 3.1 returns it: a box-local raster.

Keeping masks box-local (instead of full-frame) lets the verifier work on 24-megapixel
orthophotos without allocating a full-size array per instance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Instance:
    object_id: str
    phrase: str
    bounds: tuple[float, float, float, float]  # left, top, right, bottom (frame px, half-open)
    raster: np.ndarray                         # bool (rows, cols) covering `bounds`
    confidence: float | None = None
    frame_index: int | None = None
    backend: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def area_px(self) -> float:
        l, t, r, b = self.bounds
        if self.raster.size == 0:
            return 0.0
        return float(self.raster.mean() * (r - l) * (b - t))

    def pixel_coords(self, boundary_only: bool = True) -> np.ndarray:
        """Frame-space (x, y) coordinates of mask pixels (centres)."""
        m = self.raster.astype(np.uint8)
        if boundary_only and m.shape[0] > 2 and m.shape[1] > 2:
            er = cv2.erode(m, np.ones((3, 3), np.uint8))
            m = m & (1 - er)
        ys, xs = np.nonzero(m)
        l, t, r, b = self.bounds
        rows, cols = self.raster.shape
        sx = (r - l) / max(cols, 1)
        sy = (b - t) / max(rows, 1)
        return np.stack([l + (xs + 0.5) * sx, t + (ys + 0.5) * sy], axis=1)

    def contains(self, xy: tuple[float, float]) -> bool:
        l, t, r, b = self.bounds
        x, y = xy
        if not (l <= x < r and t <= y < b):
            return False
        rows, cols = self.raster.shape
        j = int((x - l) / (r - l) * cols)
        i = int((y - t) / (b - t) * rows)
        return bool(self.raster[min(i, rows - 1), min(j, cols - 1)])

    def distance_px(self, xy: tuple[float, float]) -> float:
        if self.raster.size == 0 or not self.raster.any():
            return float("inf")
        if self.contains(xy):
            return 0.0
        pts = self.pixel_coords(boundary_only=True)
        if len(pts) == 0:
            pts = self.pixel_coords(boundary_only=False)
        d = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])
        return float(d.min())

    def paint(self, canvas: np.ndarray, value=True) -> np.ndarray:
        """Paint this instance into a full-frame array (in place)."""
        h, w = canvas.shape[:2]
        l, t, r, b = self.bounds
        x0, y0 = int(round(l)), int(round(t))
        x1, y1 = int(round(r)), int(round(b))
        if x1 <= x0 or y1 <= y0:
            return canvas
        rs = cv2.resize(self.raster.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return canvas
        sub = rs[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        canvas[cy0:cy1, cx0:cx1][sub] = value
        return canvas

    def transformed(self, offset_xy: tuple[float, float], scale: float) -> "Instance":
        """Map from a (cropped, resized) image back to the parent frame."""
        l, t, r, b = self.bounds
        ox, oy = offset_xy
        nb = (ox + l / scale, oy + t / scale, ox + r / scale, oy + b / scale)
        return Instance(self.object_id, self.phrase, nb, self.raster, self.confidence, self.frame_index, self.backend, dict(self.meta))

    @classmethod
    def from_full_mask(cls, mask: np.ndarray, object_id: str, phrase: str, backend: str, confidence=None) -> "Instance | None":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return cls(object_id, phrase, (float(x0), float(y0), float(x1), float(y1)), mask[y0:y1, x0:x1].copy(), confidence, None, backend)


def union_mask(instances: list[Instance], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for inst in instances:
        inst.paint(m)
    return m


def frame_digest(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()
