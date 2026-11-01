"""Demo HUD: camera view with the landing pad, tracks and alerts, plus a certificate panel.

Design rules (responsible-AI review): an "H" landing pad, never crosshairs; people are
outlined, never labelled as targets; every frame says whether it is SIMULATION or REPLAY
and which evidence backend produced the certificates.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..claims import ClaimStatus
from ..descent.watch import FrameReport, State
from .overlay import label_overlay

PANEL_W = 340
C_BG = (14, 17, 22)
C_TEXT = (230, 234, 240)
C_DIM = (140, 150, 165)
C_OK = (70, 220, 120)
C_WARN = (255, 190, 40)
C_BAD = (255, 70, 70)
C_CYAN = (60, 220, 255)

STATE_COLOR = {
    State.TRANSIT: C_CYAN, State.RECERTIFY: C_CYAN, State.DESCEND: C_OK, State.HOLD: C_WARN, State.ABORT: C_BAD,
    State.DIVERT: C_WARN, State.COMMIT: C_OK, State.TOUCHDOWN: C_OK, State.FAILSAFE_HOVER: C_BAD,
}

_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"]


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "arialbd.ttf" if bold else "arial.ttf", "Arial Bold.ttf" if bold else "Arial.ttf"]
    try:
        import matplotlib

        _FONT_DIRS.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))
    except ImportError:
        pass
    for d in _FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=8)
def font_indic(size: int):
    """A font with Tamil/Devanagari glyphs if the OS has one (Windows: Nirmala UI)."""
    cands = ["C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/NirmalaB.ttf", "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
             "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf", "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
             "/Library/Fonts/Arial Unicode.ttf"]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return font(size, True)


def _dashed_circle(img, c, r, color, th=2, dash_deg=12):
    for a in range(0, 360, dash_deg * 2):
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(r), int(r)), 0, a, a + dash_deg, color, th, cv2.LINE_AA)


def draw_pad(img, c, r, color, buffer_r=None):
    cx, cy = int(c[0]), int(c[1])
    r = max(int(r), 6)
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
    cv2.circle(img, (cx, cy), r, color, 3, cv2.LINE_AA)
    if buffer_r:
        _dashed_circle(img, (cx, cy), buffer_r, color, 1)
    s = max(4, int(r * 0.45))
    t = max(2, s // 4)
    cv2.line(img, (cx - s // 2, cy - s), (cx - s // 2, cy + s), color, t, cv2.LINE_AA)
    cv2.line(img, (cx + s // 2, cy - s), (cx + s // 2, cy + s), color, t, cv2.LINE_AA)
    cv2.line(img, (cx - s // 2, cy), (cx + s // 2, cy), color, t, cv2.LINE_AA)


class Hud:
    def __init__(self, cam_w: int, cam_h: int, mode: str = "SIMULATION", evidence: str = "", planner: str = "", show_student: bool = True):
        self.cw, self.ch = cam_w, cam_h
        self.mode, self.evidence, self.planner = mode, evidence, planner
        self.show_student = show_student
        self.log: list[tuple[float, str]] = []
        self.banner: tuple[str, tuple, float] | None = None
        self.battery = None

    @property
    def size(self):
        return self.cw + PANEL_W, self.ch

    def render(self, rgb: np.ndarray, rep: FrameReport, battery_pct: float | None = None, extra_banner: str | None = None) -> np.ndarray:
        cam = rgb.copy()
        if self.show_student and rep.student_label is not None:
            cam = label_overlay(cam, rep.student_label, alpha=0.28, skip=(0, 255, 7, 8, 9))
        col = STATE_COLOR.get(rep.state, C_CYAN)
        if rep.zone_px is not None:
            draw_pad(cam, rep.zone_px, rep.radius_px, col, rep.buffer_px or None)
        movers = [tr for tr in rep.tracks if tr["speed"] >= 0.4 or (tr["tte"] is not None and tr["tte"] < 8)]
        for tr in movers:
            px = tuple(int(v) for v in tr["px"])
            danger = tr["tte"] is not None and tr["tte"] < 4.0
            c = C_BAD if danger else C_WARN
            cv2.circle(cam, px, 14, c, 2, cv2.LINE_AA)
        img = Image.fromarray(cam)
        d = ImageDraw.Draw(img)
        for tr in movers:
            px = tr["px"]
            txt = f"#{tr['id']} {tr['cls']} {tr['speed']:.1f} m/s"
            if tr["tte"] is not None and tr["tte"] < 10:
                txt += f"  enters in {tr['tte']:.1f}s"
            d.text((px[0] + 18, px[1] - 10), txt, font=font(15, True), fill=C_TEXT, stroke_width=3, stroke_fill=(0, 0, 0))

        for e in rep.events:
            self.log.append((rep.t, e))
            if e.startswith(("ABORT", "HOLD", "DIVERT", "TOUCHDOWN", "FAILSAFE")):
                self.banner = (e.split(":")[0] + (" - " + e.split(":", 1)[1].strip() if ":" in e else ""), STATE_COLOR.get(State(e.split(":")[0]), C_WARN), rep.t)
        # top banner
        if not rep.link_up:
            self._bar(d, 0, "LINK LOST - EDGE MODE (student + Muse Glimmer)", C_WARN)
        if self.banner and rep.t - self.banner[2] < 3.5:
            self._bar(d, 40 if not rep.link_up else 0, self.banner[0].upper()[:70], self.banner[1])
        if extra_banner:
            d.rectangle([0, self.ch - 84, self.cw, self.ch - 46], fill=(0, 0, 0))
            d.rectangle([0, self.ch - 84, 8, self.ch - 46], fill=C_CYAN)
            d.text((20, self.ch - 78), extra_banner[:80], font=font_indic(21), fill=C_CYAN)
        d.text((14, self.ch - 30), f"{self.mode}  |  t = {rep.t:5.1f} s", font=font(16, True), fill=C_TEXT, stroke_width=3, stroke_fill=(0, 0, 0))
        cam = np.array(img)

        panel = self._panel(rep, battery_pct)
        return np.concatenate([cam, panel], axis=1)

    def _bar(self, d: ImageDraw.ImageDraw, y: int, text: str, color):
        d.rectangle([0, y, self.cw, y + 38], fill=(0, 0, 0))
        d.rectangle([0, y, 8, y + 38], fill=color)
        d.text((20, y + 7), text, font=font(21, True), fill=color)

    def _panel(self, rep: FrameReport, battery_pct):
        p = Image.new("RGB", (PANEL_W, self.ch), C_BG)
        d = ImageDraw.Draw(p)
        x = 18
        d.text((x, 14), "LandProof", font=font(26, True), fill=C_TEXT)
        d.text((x, 46), "proof-carrying emergency landing", font=font(13), fill=C_DIM)
        col = STATE_COLOR.get(rep.state, C_CYAN)
        d.rounded_rectangle([x, 74, PANEL_W - 18, 112], radius=8, fill=tuple(int(c * 0.25) for c in col), outline=col, width=2)
        d.text((x + 12, 82), rep.state.value, font=font(20, True), fill=col)

        y = 126
        batt = f"{battery_pct:.0f}%" if battery_pct is not None else "-"
        rows = [("ALT", f"{rep.alt:5.1f} m"), ("BATTERY", batt), ("LINK", "UP" if rep.link_up else "DOWN"), ("ZONE", rep.zone_id or "-"),
                ("EGO-MOTION", f"{rep.ego_source or '-'} {rep.ego_inliers or ''}")]
        for k, v in rows:
            d.text((x, y), k, font=font(13), fill=C_DIM)
            vc = C_BAD if (k == "LINK" and not rep.link_up) or (k == "BATTERY" and battery_pct is not None and battery_pct < 10) else C_TEXT
            d.text((x + 120, y - 2), v, font=font(16, True), fill=vc)
            y += 24

        y += 6
        d.line([x, y, PANEL_W - 18, y], fill=(50, 56, 66), width=1)
        y += 10
        cert = rep.cert
        if cert is not None:
            d.text((x, y), f"CERTIFICATE {cert.cert_id}", font=font(14, True), fill=C_TEXT)
            d.text((PANEL_W - 80, y), cert.verdict, font=font(14, True), fill=C_OK if cert.passed else C_BAD)
            y += 22
            for r in cert.results[:9]:
                ok = r.status == ClaimStatus.SUPPORTED
                mark = "\u2713" if ok else "\u2717"
                txt = f"{mark} no {r.claim.phrase}"
                d.text((x, y), txt[:30], font=font(13, not ok), fill=C_OK if ok else C_BAD)
                if r.nearest_m is not None:
                    d.text((PANEL_W - 78, y), f"{r.nearest_m:.0f}/{r.required_m:.0f} m", font=font(12), fill=C_DIM)
                y += 19
            y += 4
            d.text((x, y), f"evidence: {cert.evidence_backend or self.evidence}"[:44], font=font(11), fill=C_DIM)
            y += 18
        if rep.glimmer:
            g = rep.glimmer
            d.text((x, y), "Muse Glimmer (edge): " + ("VETO" if g.get("veto") else "agrees"), font=font(13, True), fill=C_BAD if g.get("veto") else C_OK)
            y += 20
        if rep.threat and (rep.threat.get("inside") or rep.threat.get("buffer") or rep.threat.get("predicted")):
            th = rep.threat
            msg = th.get("inside") and f"INSIDE: {th['inside']}" or th.get("buffer") and f"NEAR: {th['buffer']}" or f"PREDICTED: {th['predicted']} in {th['predicted_tte']:.1f}s"
            d.text((x, y), msg[:40], font=font(13, True), fill=C_BAD)
            y += 20

        y = max(y + 6, self.ch - 210)
        d.line([x, y, PANEL_W - 18, y], fill=(50, 56, 66), width=1)
        y += 8
        d.text((x, y), "DECISION LOG (hash-chained)", font=font(12, True), fill=C_DIM)
        y += 20
        for t, e in self.log[-7:]:
            d.text((x, y), f"{t:5.1f}  {e}"[:46], font=font(11), fill=C_TEXT)
            y += 17
        d.text((x, self.ch - 24), f"planner: {self.planner}"[:44], font=font(11), fill=C_DIM)
        return np.array(p)
