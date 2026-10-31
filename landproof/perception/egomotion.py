"""Ego-motion compensation for a moving nadir camera (new code written for LandProof).

Frame-to-frame homographies from ORB features + RANSAC, with features on moving objects
(people, vehicles, animals) masked out so a walking person cannot drag the estimate.
When vision is degraded (low texture, motion blur) we fall back to a similarity
transform predicted from telemetry, and report it, instead of silently drifting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Registration:
    H: np.ndarray | None      # maps points in `src` image to `dst` image
    inliers: int
    source: str               # "vision" | "telemetry" | "failed"
    reproj_px: float = float("nan")


class EgoMotion:
    def __init__(self, n_features: int = 2000, min_inliers: int = 25, work_width: int = 640, ratio: float = 0.75):
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.min_inliers = min_inliers
        self.work_width = work_width
        self.ratio = ratio

    def _prep(self, rgb: np.ndarray, exclude: np.ndarray | None):
        s = min(1.0, self.work_width / rgb.shape[1])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if s < 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(gray)
        mask = None
        if exclude is not None:
            ex = cv2.resize(exclude.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            ex = cv2.dilate(ex, np.ones((9, 9), np.uint8))
            mask = (1 - ex).astype(np.uint8) * 255
        kp, des = self.orb.detectAndCompute(gray, mask)
        return kp, des, s

    def register(self, src: np.ndarray, dst: np.ndarray, exclude_src=None, exclude_dst=None) -> Registration:
        kp1, d1, s1 = self._prep(src, exclude_src)
        kp2, d2, s2 = self._prep(dst, exclude_dst)
        if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
            return Registration(None, 0, "failed")
        pairs = self.matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return Registration(None, len(good), "failed")
        p1 = np.float32([kp1[m.queryIdx].pt for m in good]) / s1
        p2 = np.float32([kp2[m.trainIdx].pt for m in good]) / s2
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0 / min(s1, s2), maxIters=2000, confidence=0.995)
        if H is None:
            return Registration(None, 0, "failed")
        n = int(inl.sum())
        proj = cv2.perspectiveTransform(p1[inl.ravel() == 1].reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.median(np.linalg.norm(proj - p2[inl.ravel() == 1], axis=1))) if n else float("nan")
        if n < self.min_inliers or not _plausible(H):
            return Registration(None, n, "failed", err)
        return Registration(H, n, "vision", err)


def _plausible(H: np.ndarray) -> bool:
    """Reject degenerate homographies (flips, extreme shear/scale, strong perspective)."""
    A = H[:2, :2] / H[2, 2]
    det = np.linalg.det(A)
    if det <= 0.05 or det > 20:
        return False
    if abs(H[2, 0]) > 2e-3 or abs(H[2, 1]) > 2e-3:
        return False
    return True


def telemetry_homography(pose_src, pose_dst, camera, width: int, height: int) -> np.ndarray:
    """Similarity mapping src-image px -> dst-image px predicted from two poses.

    pose = (x_m, y_m, alt_m, yaw_deg) in a local metric frame (x east, y south).
    """
    def img_to_world(p):
        x, y, alt, yaw = p
        g = camera.gsd(alt, width)
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        R = np.array([[c, -s], [s, c]]) * g
        t = np.array([x, y]) - R @ np.array([width / 2, height / 2])
        M = np.eye(3)
        M[:2, :2], M[:2, 2] = R, t
        return M

    A = img_to_world(pose_src)
    B = img_to_world(pose_dst)
    return np.linalg.inv(B) @ A
