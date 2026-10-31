"""Procedural top-down world with pixel-exact labels (offline tests + fallback demo).

The layout is built around the two traps that LandProof exists to catch:

* a monsoon-flooded field and waterlogged road: the largest, flattest, smoothest open
  area in view. A texture/flatness heuristic picks it; UAVid's fixed vocabulary has no
  water class, so a fixed-class model also picks it.
* an open grass field crossed by a power line: dry and obstacle-free for any fixed
  8-class model, lethal for a drone.

The genuinely safe park sits further from the start position, partly occupied by a cow.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..vocab import Vocabulary
from .sprites import draw_car, draw_cow, draw_person


@dataclass
class World:
    image: np.ndarray       # RGB uint8 orthophoto
    label: np.ndarray       # uint8 class ids (see configs/hazards.yaml)
    gsd: float              # metres per pixel
    start_xy_m: tuple[float, float]
    name: str = "synthetic"

    @property
    def size_m(self) -> tuple[float, float]:
        return self.image.shape[1] * self.gsd, self.image.shape[0] * self.gsd

    def m2px(self, xy_m):
        return (xy_m[0] / self.gsd, xy_m[1] / self.gsd)


def _noise(rng, h, w, scales=(64, 16, 4), weights=(0.6, 0.3, 0.1)):
    out = np.zeros((h, w), np.float32)
    for s, wt in zip(scales, weights):
        small = rng.standard_normal((max(2, h // s), max(2, w // s))).astype(np.float32)
        out += wt * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / (out.max() + 1e-6)


def _fill(img, mask, base_rgb, rng, amp=18.0, scales=(48, 12, 3)):
    h, w = mask.shape
    n = _noise(rng, h, w, scales)
    for c in range(3):
        ch = img[..., c].astype(np.float32)
        ch[mask] = np.clip(base_rgb[c] + (n[mask] - 0.5) * 2 * amp, 0, 255)
        img[..., c] = ch.astype(np.uint8)


def make_world(vocab: Vocabulary | None = None, seed: int = 7, gsd: float = 0.08) -> World:
    vocab = vocab or Vocabulary.load()
    ID = {c.name: c.id for c in vocab.classes}
    rng = np.random.default_rng(seed)
    W_m, H_m = 160.0, 120.0
    w, h = int(W_m / gsd), int(H_m / gsd)
    P = lambda v: int(round(v / gsd))  # metres -> px

    img = np.zeros((h, w, 3), np.uint8)
    label = np.full((h, w), ID["landable"], np.uint8)

    # --- ground: grass with soil patches
    _fill(img, np.ones((h, w), bool), (78, 124, 54), rng, amp=22)
    soil = _noise(rng, h, w, (160, 40), (0.7, 0.3)) > 0.72
    _fill(img, soil, (132, 112, 78), rng, amp=14)

    # --- roads (horizontal + vertical) with lane dashes
    road = np.zeros((h, w), bool)
    road[P(56) : P(63), :] = True
    road[:, P(76) : P(83)] = True
    _fill(img, road, (88, 88, 92), rng, amp=8, scales=(32, 8, 2))
    label[road] = ID["road"]
    for x in range(0, w, P(6)):
        cv2.rectangle(img, (x, P(59.4)), (x + P(3), P(59.7)), (225, 225, 215), -1)
    for y in range(0, h, P(6)):
        cv2.rectangle(img, (P(79.4), y), (P(79.7), y + P(3)), (225, 225, 215), -1)

    # --- building blocks
    def building(x0, y0, x1, y1, roof):
        cv2.rectangle(img, (P(x0) + P(1.2), P(y0) + P(1.2)), (P(x1) + P(1.2), P(y1) + P(1.2)), (40, 45, 40), -1)
        m = np.zeros((h, w), bool)
        m[P(y0) : P(y1), P(x0) : P(x1)] = True
        _fill(img, m, roof, rng, amp=10, scales=(16, 6, 2))
        cv2.line(img, (P(x0), P((y0 + y1) / 2)), (P(x1), P((y0 + y1) / 2)), tuple(int(c * 0.8) for c in roof), max(1, P(0.3)))
        label[m] = ID["building"]

    roofs = [(170, 80, 60), (150, 150, 145), (190, 110, 80), (120, 125, 135), (200, 190, 170)]
    for i, (x0, y0) in enumerate([(28, 22), (42, 22), (56, 24), (28, 36), (44, 38), (58, 38)]):
        building(x0, y0, x0 + 11, y0 + 10, roofs[i % len(roofs)])
    # water tanks on two roofs (obstacles)
    for cx, cy in [(33, 27), (62, 43)]:
        cv2.circle(img, (P(cx), P(cy)), P(0.9), (25, 25, 28), -1, cv2.LINE_AA)
        cv2.circle(label, (P(cx), P(cy)), P(0.9), ID["obstacle"], -1)

    # --- monsoon flood: waterlogged road + flooded field (smooth, flat, inviting)
    flood = np.zeros((h, w), np.uint8)
    pts = np.array([[92, 55], [134, 54], [136, 66], [133, 97], [112, 100], [97, 96], [95, 66]], np.float32) / gsd
    cv2.fillPoly(flood, [pts.astype(np.int32)], 1)
    flood = cv2.GaussianBlur(flood.astype(np.float32), (0, 0), P(1.5)) > 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx / w * 0.6 + yy / h * 0.4).astype(np.float32)
    for c, (base, sky) in enumerate(zip((116, 104, 82), (150, 160, 170))):
        ch = img[..., c].astype(np.float32)
        ch[flood] = base * (1 - 0.25 * grad[flood]) + sky * 0.25 * grad[flood] + rng.normal(0, 1.5, flood.sum())
        img[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
    label[flood] = ID["water"]

    # --- trees (clusters along roads and around the park)
    def tree(cx, cy, r):
        cv2.circle(img, (P(cx + 0.6 * r), P(cy + 0.6 * r)), P(r), (30, 45, 28), -1, cv2.LINE_AA)
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (P(cx), P(cy)), P(r), 1, -1)
        m = m.astype(bool)
        _fill(img, m, (40, 92, 38), rng, amp=26, scales=(8, 3, 1))
        label[m] = ID["tree"]

    for cx, cy, r in [(88, 22, 2.6), (88, 30, 2.2), (88, 38, 2.4), (88, 46, 2.0), (136, 24, 2.5), (136, 34, 2.8),
                      (24, 70, 2.2), (24, 82, 2.6), (72, 104, 2.4), (100, 104, 2.0), (66, 66, 1.8), (138, 46, 2.1)]:
        tree(cx, cy, r)

    # --- power line across the dry field (thin, easy to miss; deadly)
    p0, p1 = (27.0, 69.0), (73.0, 99.0)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    wire = np.zeros((h, w), np.uint8)
    for off in (-0.35, 0.35):
        a = (P(p0[0]), P(p0[1] + off))
        b = (P(p1[0]), P(p1[1] + off))
        cv2.line(img, a, b, (35, 35, 35), max(1, P(0.06)), cv2.LINE_AA)
        cv2.line(wire, a, b, 1, max(2, P(0.3)))
    for p in (p0, mid, p1):
        cv2.circle(img, (P(p[0] + 0.8), P(p[1] + 0.8)), P(0.5), (30, 35, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (P(p[0]), P(p[1])), P(0.35), (120, 115, 105), -1, cv2.LINE_AA)
        cv2.circle(wire, (P(p[0]), P(p[1])), P(0.6), 1, -1)
    label[wire.astype(bool)] = ID["wire_pole"]

    # --- obstacles: blue tarpaulin and debris in the park corner
    cv2.rectangle(img, (P(124), P(41)), (P(131), P(47)), (40, 90, 175), -1)
    cv2.rectangle(label, (P(124), P(41)), (P(131), P(47)), ID["obstacle"], -1)
    for _ in range(25):
        cx, cy = rng.uniform(117, 122), rng.uniform(44, 49)
        cv2.circle(img, (P(cx), P(cy)), P(rng.uniform(0.2, 0.5)), (110, 100, 95), -1)
        cv2.circle(label, (P(cx), P(cy)), P(0.5), ID["obstacle"], -1)

    # --- vehicles on roads, people at the flood edge, a cow in the park
    for (cx, cy, hd, col) in [(79.5, 40, 90, (200, 200, 205)), (79.5, 74, 90, (160, 30, 30)), (60, 58, 0, (230, 230, 230)),
                              (104, 61, 0, (40, 70, 140)), (20, 61, 180, (30, 30, 30)), (79.5, 108, 270, (220, 180, 40))]:
        draw_car(img, label, ID["vehicle"], (P(cx), P(cy)), gsd, hd, col)
    for (cx, cy) in [(94, 66), (95.2, 67.1), (73, 52), (140, 70)]:
        draw_person(img, label, ID["person"], (P(cx), P(cy)), gsd, rng.uniform(0, 360))
    draw_cow(img, label, ID["animal"], (P(98), P(30)), gsd, 20)
    draw_cow(img, label, ID["animal"], (P(102), P(36)), gsd, 200)

    return World(image=img, label=label, gsd=gsd, start_xy_m=(80.0, 60.0), name="synthetic-chennai-monsoon")
