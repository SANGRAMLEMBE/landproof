"""Closed-loop replay-descent simulator.

Renders what a nadir camera sees from any pose over an orthophoto (synthetic world,
FloodNet/ICG image, or your own drone mosaic), composites moving actors at camera
resolution, and integrates simple drone kinematics with wind drift and GPS error.
The flight logic never sees ground truth; GT is only used for scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import Camera
from ..vocab import Vocabulary
from .scene import World
from .sprites import DRAWERS


@dataclass
class Actor:
    kind: str                       # "person" | "vehicle" | "animal"
    path_m: list[tuple[float, float]]
    speed_mps: float = 1.3
    start_t: float = 0.0
    name: str = ""

    def position(self, t: float) -> tuple[tuple[float, float], float] | None:
        """(xy_m, heading_deg) at time t, or None before start. Stops at path end."""
        if t < self.start_t:
            return None
        dist = (t - self.start_t) * self.speed_mps
        pts = self.path_m
        for a, b in zip(pts[:-1], pts[1:]):
            seg = math.dist(a, b)
            if dist <= seg or (a, b) == (pts[-2], pts[-1]):
                f = min(1.0, dist / seg) if seg else 1.0
                heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), heading
            dist -= seg
        return pts[-1], 0.0


@dataclass
class Pose:
    x: float
    y: float
    alt: float
    yaw: float = 0.0

    def as_tuple(self):
        return (self.x, self.y, self.alt, self.yaw)


@dataclass
class Command:
    target_xy: tuple[float, float] | None = None   # where to fly (telemetry frame)
    vel_xy: tuple[float, float] | None = None      # or a direct horizontal velocity (m/s)
    vz: float = 0.0                                # +up, m/s
    max_speed: float = 4.0


@dataclass
class Observation:
    t: float
    rgb: np.ndarray
    label: np.ndarray            # GT (never used by flight logic)
    pose_true: Pose
    pose_est: Pose               # telemetry estimate (GPS + baro/rangefinder)
    actors_xy: dict = field(default_factory=dict)


class ReplaySim:
    def __init__(self, world: World, camera: Camera, vocab: Vocabulary, start: Pose, actors: list[Actor] | None = None,
                 fps: float = 15.0, wind_mps: float = 0.4, gps_sigma_m: float = 0.8, yaw_wobble_deg: float = 2.5, seed: int = 0):
        self.world, self.cam, self.vocab = world, camera, vocab
        self.actors = actors or []
        self.dt = 1.0 / fps
        self.t = 0.0
        self.pose = Pose(*start.as_tuple())
        self.rng = np.random.default_rng(seed)
        self.wind = np.zeros(2)
        self.wind_mps = wind_mps
        self.gps_bias = self.rng.normal(0, gps_sigma_m, 2)
        self.gps_sigma = gps_sigma_m
        self.yaw0 = start.yaw
        self.yaw_wobble = yaw_wobble_deg
        self.ids = {c.name: c.id for c in vocab.classes}

    # ------------------------------------------------------------------ rendering
    def _affine(self, pose: Pose) -> tuple[np.ndarray, float]:
        g_cam = self.cam.gsd(pose.alt)
        s = self.world.gsd / g_cam
        c, sn = math.cos(math.radians(-pose.yaw)), math.sin(math.radians(-pose.yaw))
        R = np.array([[c, -sn], [sn, c]]) * s
        cw = np.array([pose.x / self.world.gsd, pose.y / self.world.gsd])
        cc = np.array([self.cam.width / 2.0, self.cam.height / 2.0])
        M = np.zeros((2, 3))
        M[:, :2] = R
        M[:, 2] = cc - R @ cw
        return M, g_cam

    def world_to_cam(self, pose: Pose, xy_m) -> tuple[float, float]:
        M, _ = self._affine(pose)
        p = M @ np.array([xy_m[0] / self.world.gsd, xy_m[1] / self.world.gsd, 1.0])
        return float(p[0]), float(p[1])

    def render(self, pose: Pose, t: float) -> tuple[np.ndarray, np.ndarray, dict]:
        M, g_cam = self._affine(pose)
        size = (self.cam.width, self.cam.height)
        interp = cv2.INTER_LINEAR if g_cam > self.world.gsd * 0.7 else cv2.INTER_CUBIC
        rgb = cv2.warpAffine(self.world.image, M, size, flags=interp, borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(self.world.label, M, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        if g_cam < self.world.gsd * 0.6:  # restore sensor-like fine texture when upsampling
            noise = self.rng.normal(0, 4.0, rgb.shape[:2]).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (0, 0), 0.8)[..., None]
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        actors_xy = {}
        for i, a in enumerate(self.actors):
            st = a.position(t)
            if st is None:
                continue
            (x, y), hd = st
            actors_xy[a.name or f"{a.kind}{i}"] = (x, y)
            p = M @ np.array([x / self.world.gsd, y / self.world.gsd, 1.0])
            if -50 <= p[0] <= size[0] + 50 and -50 <= p[1] <= size[1] + 50:
                DRAWERS[a.kind](rgb, label, self.ids[a.kind], (p[0], p[1]), g_cam, hd - pose.yaw)
        return rgb, label, actors_xy

    # ------------------------------------------------------------------ dynamics
    def observe(self) -> Observation:
        rgb, label, actors = self.render(self.pose, self.t)
        est = Pose(self.pose.x + self.gps_bias[0], self.pose.y + self.gps_bias[1], self.pose.alt + self.rng.normal(0, 0.05), self.pose.yaw + self.rng.normal(0, 0.3))
        return Observation(self.t, rgb, label, Pose(*self.pose.as_tuple()), est, actors)

    def step(self, cmd: Command) -> Observation:
        dt = self.dt
        if cmd.vel_xy is not None:
            v = np.array(cmd.vel_xy, float)
        elif cmd.target_xy is not None:
            # controller flies toward target expressed in the *telemetry* frame
            est = np.array([self.pose.x + self.gps_bias[0], self.pose.y + self.gps_bias[1]])
            d = np.array(cmd.target_xy) - est
            dist = np.hypot(*d)
            v = d / dist * min(cmd.max_speed, dist * 1.2) if dist > 1e-6 else np.zeros(2)
        else:
            v = np.zeros(2)
        sp = np.hypot(*v)
        if sp > cmd.max_speed:
            v = v / sp * cmd.max_speed
        self.wind = 0.95 * self.wind + self.rng.normal(0, self.wind_mps * 0.3, 2)
        self.pose.x += (v[0] + self.wind[0] * 0.3) * dt
        self.pose.y += (v[1] + self.wind[1] * 0.3) * dt
        self.pose.alt = max(0.0, self.pose.alt + cmd.vz * dt)
        self.pose.yaw = self.yaw0 + self.yaw_wobble * math.sin(self.t * 0.7) + 0.4 * math.sin(self.t * 2.3)
        self.gps_bias = 0.98 * self.gps_bias + self.rng.normal(0, self.gps_sigma * 0.05, 2)
        self.t += dt
        return self.observe()

    # ------------------------------------------------------------------ ground truth scoring
    def touchdown_safety(self, vocab: Vocabulary, airframe_radius_m: float, person_clear_m: float = 5.0) -> dict:
        """Score the true touchdown point against ground truth (static map + actors)."""
        g = self.world.gsd
        cx, cy = self.pose.x / g, self.pose.y / g
        r = int(math.ceil(airframe_radius_m / g)) + 1
        H, W = self.world.label.shape
        x0, x1, y0, y1 = max(0, int(cx) - r), min(W, int(cx) + r + 1), max(0, int(cy) - r), min(H, int(cy) + r + 1)
        win = self.world.label[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (airframe_radius_m / g) ** 2
        hazard_ids = {c.id: c.name for c in vocab.hazard_classes}
        hit = sorted({hazard_ids[int(v)] for v in np.unique(win[disk]) if int(v) in hazard_ids})
        near = []
        for a in self.actors:
            st = a.position(self.t)
            if st and a.kind == "person" and math.dist(st[0], (self.pose.x, self.pose.y)) < person_clear_m:
                near.append(a.name or a.kind)
        return {"safe": not hit and not near, "static_hazards_under_footprint": hit, "people_within_m": near,
                "touchdown_xy_m": (round(self.pose.x, 2), round(self.pose.y, 2))}
