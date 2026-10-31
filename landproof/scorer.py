"""Landing-zone scorer.

clearance(x) = min over hazard classes c of ( distance from x to the nearest c pixel - buffer_c )

A point is a feasible touchdown centre when clearance(x) >= the airframe's required
radius. Per-class buffers mean a person needs 5 m of keep-out while a puddle needs
1.5 m, which a single dilated "obstacle" mask cannot express.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .geometry import disk_mask
from .masks import downsample_any


@dataclass
class Zone:
    zone_id: str
    center_px: tuple[float, float]  # (x, y) in the source image
    radius_m: float                 # required touchdown radius (before buffers)
    radius_px: float
    clearance_m: float              # distance to the nearest buffered hazard
    score: float
    feasible: bool
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center_px"] = [round(self.center_px[0], 1), round(self.center_px[1], 1)]
        return d


@dataclass
class ClearanceResult:
    clearance_m: np.ndarray   # on the work grid
    work_factor: int
    work_gsd: float
    limiting_class: np.ndarray  # class id that limits clearance at each cell (-1 = border)


def clearance_map(
    class_masks: dict[int, np.ndarray],
    buffers_m: dict[int, float],
    gsd_m: float,
    work_gsd_m: float = 0.10,
    include_border: bool = True,
) -> ClearanceResult:
    any_mask = next(iter(class_masks.values()))
    h, w = any_mask.shape
    factor = max(1, int(round(work_gsd_m / gsd_m)))
    wgsd = gsd_m * factor
    H, W = int(math.ceil(h / factor)), int(math.ceil(w / factor))

    clearance = np.full((H, W), np.inf, dtype=np.float32)
    limiting = np.full((H, W), -2, dtype=np.int16)
    for cid, m in class_masks.items():
        small = downsample_any(m, factor)
        if not small.any():
            continue
        d = ndimage.distance_transform_edt(~small).astype(np.float32) * wgsd
        c = d - float(buffers_m.get(cid, 0.0)) - wgsd / 2.0
        better = c < clearance
        clearance[better] = c[better]
        limiting[better] = cid

    if include_border:
        yy, xx = np.mgrid[0:H, 0:W]
        border = np.minimum.reduce([xx, yy, (W - 1) - xx, (H - 1) - yy]).astype(np.float32) * wgsd
        better = border < clearance
        clearance[better] = border[better]
        limiting[better] = -1
    return ClearanceResult(clearance, factor, wgsd, limiting)


class ZoneScorer:
    def __init__(self, buffers_m: dict[int, float], cfg: dict):
        self.buffers_m = buffers_m
        s = cfg["scorer"]
        self.max_zones = int(s["max_zones"])
        self.min_sep_m = float(s["min_separation_m"])
        self.work_gsd = float(s["work_gsd_m"])
        self.weights = dict(s["weights"])

    def score(
        self,
        class_masks: dict[int, np.ndarray],
        gsd_m: float,
        required_radius_m: float,
        landable_mask: np.ndarray | None = None,
        image: np.ndarray | None = None,
        current_xy: tuple[float, float] | None = None,
    ) -> list[Zone]:
        cr = clearance_map(class_masks, self.buffers_m, gsd_m, self.work_gsd)
        clr, f, wgsd = cr.clearance_m, cr.work_factor, cr.work_gsd
        H, W = clr.shape
        h, w = next(iter(class_masks.values())).shape
        if current_xy is None:
            current_xy = (w / 2.0, h / 2.0)

        feasible = clr >= required_radius_m
        if not feasible.any():
            return []

        sep_cells = max(3, int(round(self.min_sep_m / wgsd)) | 1)
        finite = np.where(np.isfinite(clr), clr, 1e6)
        peaks = (finite == ndimage.maximum_filter(finite, size=sep_cells, mode="nearest")) & feasible
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            ys, xs = np.nonzero(feasible)

        smooth = None
        if image is not None and self.weights.get("smoothness", 0) > 0:
            smooth = texture_energy(image)

        diag = math.hypot(w, h)
        r_px = required_radius_m / gsd_m
        cands = []
        for y, x in zip(ys, xs):
            cx, cy = (x + 0.5) * f, (y + 0.5) * f
            margin = float(clr[y, x]) - required_radius_m
            comp = {
                "clearance": float(min(margin / max(required_radius_m * 2.0, 1e-6), 1.0)),
                "proximity": float(1.0 - math.hypot(cx - current_xy[0], cy - current_xy[1]) / diag),
                "landable": 0.5,
                "smoothness": 0.0,
            }
            if landable_mask is not None or smooth is not None:
                disk = _disk_window(landable_mask if landable_mask is not None else smooth, (cx, cy), r_px)
                if landable_mask is not None:
                    comp["landable"] = float(disk[0][disk[1]].mean()) if disk[1].any() else 0.0
                if smooth is not None:
                    sw = _disk_window(smooth, (cx, cy), r_px)
                    comp["smoothness"] = float(1.0 - sw[0][sw[1]].mean()) if sw[1].any() else 0.0
            score = sum(self.weights.get(k, 0.0) * v for k, v in comp.items())
            cands.append((score, cx, cy, float(clr[y, x]), comp))

        cands.sort(key=lambda t: -t[0])
        zones: list[Zone] = []
        for score, cx, cy, c_m, comp in cands:
            if any(math.hypot(cx - z.center_px[0], cy - z.center_px[1]) * gsd_m < self.min_sep_m for z in zones):
                continue
            zones.append(
                Zone(
                    zone_id=f"Z{len(zones) + 1}",
                    center_px=(float(cx), float(cy)),
                    radius_m=float(required_radius_m),
                    radius_px=float(r_px),
                    clearance_m=float(c_m),
                    score=float(score),
                    feasible=True,
                    components=comp,
                )
            )
            if len(zones) >= self.max_zones:
                break
        return zones


def texture_energy(image: np.ndarray) -> np.ndarray:
    """Normalised local gradient energy in [0, 1] (0 = perfectly smooth)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3)
    return np.clip(e / (np.percentile(e, 99) + 1e-6), 0, 1)


def _disk_window(arr: np.ndarray, center: tuple[float, float], r: float):
    h, w = arr.shape[:2]
    cx, cy = center
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    win = arr[y0:y1, x0:x1]
    m = disk_mask(win.shape[:2], (cx - x0, cy - y0), r)
    return win, m
