"""Ground-frame multi-object tracker with time-to-zone-entry prediction.

Detections are mapped into a fixed ground reference (the keyframe captured on arrival
over the zone, via the ego-motion homography) so tracks keep their identity while the
camera descends, drifts and yaws. Velocity lets the drone hold *before* a person walks
into the zone instead of reacting after.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    cls: int
    xy_m: tuple[float, float]  # zone-centric ground coordinates (metres)
    radius_m: float
    px: tuple[float, float]    # current-frame pixel centroid (for drawing)


@dataclass
class Track:
    track_id: int
    cls: int
    xy_m: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius_m: float = 0.3
    hits: int = 1
    last_t: float = 0.0
    px: tuple[float, float] = (0.0, 0.0)
    history: list = field(default_factory=list)  # [(t, x, y)] over the last `window_s`

    def fit_velocity(self, window_s: float, min_span_s: float = 0.6) -> None:
        """Least-squares velocity over a time window: robust to per-frame ego-motion jitter."""
        h = [p for p in self.history if self.last_t - p[0] <= window_s]
        self.history = h
        if len(h) < 5 or h[-1][0] - h[0][0] < min_span_s:
            self.vel = np.zeros(2)
            return
        a = np.array(h)
        t = a[:, 0] - a[:, 0].mean()
        denom = float(t @ t)
        self.vel = np.array([float(t @ (a[:, 1] - a[:, 1].mean())) / denom, float(t @ (a[:, 2] - a[:, 2].mean())) / denom])

    def time_to_enter(self, radius_m: float) -> float:
        """Smallest t >= 0 at which |xy + vel t| <= radius_m (inf if never)."""
        p, v = self.xy_m, self.vel
        if np.hypot(*p) <= radius_m:
            return 0.0
        a = float(v @ v)
        if a < 1e-6:
            return math.inf
        b = 2.0 * float(p @ v)
        c = float(p @ p) - radius_m**2
        disc = b * b - 4 * a * c
        if disc < 0:
            return math.inf
        t1 = (-b - math.sqrt(disc)) / (2 * a)
        t2 = (-b + math.sqrt(disc)) / (2 * a)
        ts = [t for t in (t1, t2) if t >= 0]
        return min(ts) if ts else math.inf

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.vel))


class GroundTracker:
    def __init__(self, gate_m: float = 3.0, max_age_s: float = 1.0, window_s: float = 1.5):
        self.gate_m = gate_m
        self.max_age_s = max_age_s
        self.window_s = window_s
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def reset(self):
        self.tracks = []

    def update(self, dets: list[Detection], t: float) -> list[Track]:
        live = [tr for tr in self.tracks if t - tr.last_t <= self.max_age_s]
        if live and dets:
            cost = np.full((len(live), len(dets)), 1e6)
            for i, tr in enumerate(live):
                pred = tr.xy_m + tr.vel * (t - tr.last_t)
                for j, d in enumerate(dets):
                    if d.cls != tr.cls:
                        continue
                    dist = float(np.hypot(*(np.array(d.xy_m) - pred)))
                    if dist <= self.gate_m:
                        cost[i, j] = dist
            rows, cols = linear_sum_assignment(cost)
        else:
            rows, cols = np.array([], int), np.array([], int)
        matched_t, matched_d = set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            tr, d = live[i], dets[j]
            new = np.array(d.xy_m, float)
            tr.xy_m, tr.radius_m, tr.hits, tr.last_t, tr.px = new, d.radius_m, tr.hits + 1, t, d.px
            tr.history.append((t, new[0], new[1]))
            tr.fit_velocity(self.window_s)
            matched_t.add(i)
            matched_d.add(j)
        for j, d in enumerate(dets):
            if j not in matched_d:
                live.append(Track(next(self._ids), d.cls, np.array(d.xy_m, float), radius_m=d.radius_m, last_t=t, px=d.px, history=[(t, d.xy_m[0], d.xy_m[1])]))
        self.tracks = live
        return [tr for tr in self.tracks if tr.last_t == t]
