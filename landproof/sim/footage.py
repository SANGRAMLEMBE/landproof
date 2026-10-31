"""Real drone footage as an observation source ("shadow mode").

The descent watch runs on recorded frames and logs what it WOULD command; nothing is
flown autonomously. Telemetry comes from DJI .SRT sidecar files when available
(latitude / longitude / rel_alt), otherwise from a constant or linear altitude profile.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .replay import Observation, Pose

_SRT_FIELDS = {
    "lat": re.compile(r"latitude\s*:\s*([-\d.]+)"),
    "lon": re.compile(r"longt?itude\s*:\s*([-\d.]+)"),
    "rel_alt": re.compile(r"rel_alt\s*:\s*([-\d.]+)"),
    "alt": re.compile(r"\baltitude\s*:\s*([-\d.]+)"),
}


def parse_dji_srt(path: str | Path) -> list[dict]:
    """Parse a DJI .SRT telemetry file into one dict per subtitle block."""
    text = Path(path).read_text(errors="ignore")
    out = []
    for block in re.split(r"\n\s*\n", text):
        rec = {}
        for k, rx in _SRT_FIELDS.items():
            m = rx.search(block)
            if m:
                rec[k] = float(m.group(1))
        if rec:
            out.append(rec)
    return out


def latlon_to_local(lat, lon, lat0, lon0) -> tuple[float, float]:
    """Equirectangular approximation: x east, y SOUTH (image-like), metres."""
    R = 6378137.0
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = -math.radians(lat - lat0) * R
    return x, y


@dataclass
class FootageConfig:
    video: str
    srt: str | None = None
    alt_start: float | None = None   # used when there is no SRT
    alt_end: float | None = None
    stride: int = 1                  # use every n-th frame
    cam_w: int = 960
    cam_h: int = 720
    start_s: float = 0.0
    end_s: float | None = None


class FootageSource:
    def __init__(self, fc: FootageConfig):
        self.fc = fc
        self.cap = cv2.VideoCapture(fc.video)
        if not self.cap.isOpened():
            raise FileNotFoundError(fc.video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.telemetry = parse_dji_srt(fc.srt) if fc.srt else []
        self.origin = None
        if self.telemetry and "lat" in self.telemetry[0]:
            self.origin = (self.telemetry[0]["lat"], self.telemetry[0]["lon"])

    def _pose(self, idx: int) -> Pose:
        if self.telemetry:
            r = self.telemetry[min(idx, len(self.telemetry) - 1)]
            alt = r.get("rel_alt", r.get("alt", 30.0))
            x, y = (latlon_to_local(r["lat"], r["lon"], *self.origin) if self.origin and "lat" in r else (0.0, 0.0))
            return Pose(x, y, alt, 0.0)
        a0 = self.fc.alt_start or 30.0
        a1 = self.fc.alt_end if self.fc.alt_end is not None else a0
        f = idx / max(1, self.n - 1)
        return Pose(0.0, 0.0, a0 + (a1 - a0) * f, 0.0)

    def _fit(self, bgr):
        h, w = bgr.shape[:2]
        target = self.fc.cam_w / self.fc.cam_h
        if w / h > target:  # crop width (e.g. 16:9 -> 4:3)
            nw = int(h * target)
            x0 = (w - nw) // 2
            bgr = bgr[:, x0 : x0 + nw]
        else:
            nh = int(w / target)
            y0 = (h - nh) // 2
            bgr = bgr[y0 : y0 + nh]
        return cv2.cvtColor(cv2.resize(bgr, (self.fc.cam_w, self.fc.cam_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    def __iter__(self):
        idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.fc.start_s * self.fps))
        idx = int(self.fc.start_s * self.fps)
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            t = idx / self.fps
            if self.fc.end_s is not None and t > self.fc.end_s:
                break
            if (idx - int(self.fc.start_s * self.fps)) % self.fc.stride == 0:
                pose = self._pose(idx)
                yield idx, Observation(t - self.fc.start_s, self._fit(bgr), None, pose, pose, {})
            idx += 1


class PrecomputedSamStudent:
    """Per-frame hazard maps from SAM 3.1 *video* tracking, precomputed offline.

    A stand-in for the distilled student when replaying real footage (labelled as such on
    the HUD). Built from MetaApiSam.segment_video results: {phrase: {frame_idx: [Instance]}}.
    """

    name = "sam-3.1-video (precomputed)"

    def __init__(self, tracks_by_phrase: dict, vocab, video_wh: tuple[int, int], cam_wh: tuple[int, int]):
        self.tracks, self.vocab = tracks_by_phrase, vocab
        self.vw, self.vh = video_wh
        self.cw, self.ch = cam_wh
        self.frame_idx = 0

    def set_frame(self, idx: int):
        self.frame_idx = idx

    def predict(self, rgb, gt_for_oracle_only=None):
        target = self.cw / self.ch
        if self.vw / self.vh > target:
            nw = int(self.vh * target)
            x0, y0, sw, sh = (self.vw - nw) // 2, 0, nw, self.vh
        else:
            nh = int(self.vw / target)
            x0, y0, sw, sh = 0, (self.vh - nh) // 2, self.vw, nh
        full = np.zeros((self.vh, self.vw), np.uint8)
        for phrase, by_frame in self.tracks.items():
            cid = self.vocab.class_of(phrase).id
            canvas = np.zeros((self.vh, self.vw), bool)
            for inst in by_frame.get(self.frame_idx, []):
                inst.paint(canvas)
            full[canvas] = cid
        crop = full[y0 : y0 + sh, x0 : x0 + sw]
        return cv2.resize(crop, (self.cw, self.ch), interpolation=cv2.INTER_NEAREST)
