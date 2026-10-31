"""Camera and landing geometry for a nadir (straight-down) camera."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    hfov_deg: float = 70.0
    width: int = 960
    height: int = 720

    def gsd(self, altitude_m: float, width_px: int | None = None) -> float:
        """Ground sample distance (metres per pixel) at a given altitude."""
        w = width_px or self.width
        ground_width = 2.0 * max(altitude_m, 0.05) * math.tan(math.radians(self.hfov_deg) / 2.0)
        return ground_width / w

    def altitude_for_gsd(self, gsd_m: float, width_px: int | None = None) -> float:
        w = width_px or self.width
        return gsd_m * w / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    def ground_extent(self, altitude_m: float) -> tuple[float, float]:
        g = self.gsd(altitude_m)
        return self.width * g, self.height * g

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Camera":
        c = cfg["camera"]
        return cls(hfov_deg=float(c["hfov_deg"]), width=int(c["width"]), height=int(c["height"]))


@dataclass(frozen=True)
class Airframe:
    diameter_m: float = 0.9
    touchdown_margin_m: float = 1.0
    position_error_m_per_10m: float = 0.3

    def required_radius_m(self, altitude_m: float) -> float:
        """Radius of the disc that must be hazard-free (before per-class buffers)."""
        pos_err = self.position_error_m_per_10m * max(altitude_m, 0.0) / 10.0
        return self.diameter_m / 2.0 + self.touchdown_margin_m + pos_err

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Airframe":
        a = cfg["airframe"]
        return cls(
            diameter_m=float(a["diameter_m"]),
            touchdown_margin_m=float(a["touchdown_margin_m"]),
            position_error_m_per_10m=float(a["position_error_m_per_10m"]),
        )


def disk_mask(shape: tuple[int, int], center_xy: tuple[float, float], radius_px: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    p = np.hstack([pts, ones]) @ H.T
    return p[:, :2] / p[:, 2:3]


def homography_scale(H: np.ndarray, at_xy: tuple[float, float] = (0.0, 0.0)) -> float:
    """Approximate isotropic scale of H around a point (sqrt of local Jacobian det)."""
    x, y = at_xy
    p = apply_homography(H, np.array([[x, y], [x + 1, y], [x, y + 1]]))
    j = np.array([p[1] - p[0], p[2] - p[0]])
    return float(math.sqrt(abs(np.linalg.det(j))))
