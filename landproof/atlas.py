"""Contingency atlas: certified landing zones prepared *before* an emergency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .claims import Certificate
from .scorer import Zone


@dataclass
class AtlasEntry:
    zone: Zone                       # pixel coords in the atlas keyframe
    cert: Certificate
    world_xy_est: tuple[float, float]  # telemetry-frame metres
    claims_phrases: list[str] = field(default_factory=list)

    @property
    def zone_id(self) -> str:
        return self.zone.zone_id


@dataclass
class Atlas:
    entries: list[AtlasEntry]
    keyframe_rgb: np.ndarray
    keyframe_pose_est: tuple[float, float, float, float]  # x, y, alt, yaw
    gsd: float
    built_t: float
    rejected: list[dict] = field(default_factory=list)   # zones that failed certification (+why)
    trace: list[dict] = field(default_factory=list)       # agent tool calls, for the README/video
    scene_phrases: list[str] = field(default_factory=list)
    evidence_backend: str = ""
    planner: str = ""

    def usable(self, t: float, ttl_s: float, exclude: set[str]) -> list[AtlasEntry]:
        return [e for e in self.entries if e.cert.passed and e.zone_id not in exclude and (t - e.cert.issued_t) <= ttl_s]


def pixel_to_world(px: tuple[float, float], pose_est, gsd: float, width: int, height: int) -> tuple[float, float]:
    x, y, _alt, yaw = pose_est
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dx, dy = (px[0] - width / 2.0) * gsd, (px[1] - height / 2.0) * gsd
    return (x + c * dx - s * dy, y + s * dx + c * dy)
