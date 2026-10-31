"""Top-down sprites (person, car, cow) drawn at any ground sample distance.

Sprites paint both the RGB image and the label map, so every simulated frame has
pixel-exact ground truth.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SHIRTS = [(200, 40, 40), (240, 200, 40), (40, 90, 200), (230, 230, 230), (30, 140, 70), (220, 90, 160)]


def _ellipse(img, label, cid, center, axes_px, angle_deg, color, shadow=True):
    cx, cy = center
    ax = (max(1, int(round(axes_px[0]))), max(1, int(round(axes_px[1]))))
    if shadow:
        off = max(1, int(round(0.6 * ax[0])))
        cv2.ellipse(img, (int(cx + off), int(cy + off)), ax, angle_deg, 0, 360, (35, 40, 35), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(round(cx)), int(round(cy))), ax, angle_deg, 0, 360, color, -1, cv2.LINE_AA)
    if label is not None:
        cv2.ellipse(label, (int(round(cx)), int(round(cy))), ax, angle_deg, 0, 360, int(cid), -1)


def draw_person(img, label, cid, center_px, gsd, heading_deg=0.0, color=None, lying=False):
    color = color or SHIRTS[int(abs(center_px[0] * 7 + center_px[1] * 3)) % len(SHIRTS)]
    if lying:
        _ellipse(img, label, cid, center_px, (0.85 / gsd, 0.22 / gsd), heading_deg, color)
    else:
        _ellipse(img, label, cid, center_px, (0.24 / gsd, 0.14 / gsd), heading_deg + 90, color)
    # head (dark hair), slightly forward of shoulders
    hx = center_px[0] + math.cos(math.radians(heading_deg)) * 0.05 / gsd
    hy = center_px[1] + math.sin(math.radians(heading_deg)) * 0.05 / gsd
    r = max(1, int(round(0.1 / gsd)))
    cv2.circle(img, (int(round(hx)), int(round(hy))), r, (25, 20, 18), -1, cv2.LINE_AA)
    if label is not None:
        cv2.circle(label, (int(round(hx)), int(round(hy))), r, int(cid), -1)


def draw_car(img, label, cid, center_px, gsd, heading_deg=0.0, color=(180, 180, 190)):
    L, W = 4.4 / gsd, 1.8 / gsd
    rect = ((float(center_px[0]), float(center_px[1])), (L, W), heading_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    sh = box + int(round(0.4 / gsd))
    cv2.fillPoly(img, [sh], (40, 42, 40), cv2.LINE_AA)
    cv2.fillPoly(img, [box], color, cv2.LINE_AA)
    # windshield
    wrect = ((float(center_px[0] + math.cos(math.radians(heading_deg)) * 0.7 / gsd), float(center_px[1] + math.sin(math.radians(heading_deg)) * 0.7 / gsd)), (0.9 / gsd, 1.5 / gsd), heading_deg)
    cv2.fillPoly(img, [cv2.boxPoints(wrect).astype(np.int32)], (40, 55, 70), cv2.LINE_AA)
    if label is not None:
        cv2.fillPoly(label, [box], int(cid))


def draw_cow(img, label, cid, center_px, gsd, heading_deg=0.0):
    _ellipse(img, label, cid, center_px, (1.0 / gsd, 0.38 / gsd), heading_deg, (235, 230, 220))
    hx = center_px[0] + math.cos(math.radians(heading_deg)) * 1.05 / gsd
    hy = center_px[1] + math.sin(math.radians(heading_deg)) * 1.05 / gsd
    _ellipse(img, label, cid, (hx, hy), (0.25 / gsd, 0.18 / gsd), heading_deg, (120, 90, 70), shadow=False)
    # patches
    cv2.circle(img, (int(center_px[0]), int(center_px[1])), max(1, int(0.2 / gsd)), (90, 70, 60), -1, cv2.LINE_AA)


DRAWERS = {"person": draw_person, "vehicle": draw_car, "animal": draw_cow}
