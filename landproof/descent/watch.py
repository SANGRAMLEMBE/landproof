"""Descent watch: the onboard state machine that flies an emergency landing.

TRANSIT -> RECERTIFY -> DESCEND <-> HOLD -> COMMIT -> TOUCHDOWN
                              \\-> ABORT -> DIVERT -> TRANSIT (next certified zone)
                                                \\-> FAILSAFE_HOVER (atlas exhausted)

Authority order when signals disagree: human > deterministic verifier > student > LLM.
Nothing in this file calls an LLM. Muse Spark only ever *prepares* the atlas; Muse
Glimmer can only veto (never approve) when the cloud link is down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from ..atlas import Atlas, AtlasEntry
from ..claims import Certificate
from ..geometry import Airframe, Camera, apply_homography
from ..ledger import Ledger
from ..perception.egomotion import EgoMotion, telemetry_homography
from ..perception.student import StudentBackend
from ..perception.tracker import Detection, GroundTracker
from ..scorer import Zone
from ..sim.replay import Command, Observation
from ..vocab import Vocabulary


class State(str, Enum):
    TRANSIT = "TRANSIT"
    RECERTIFY = "RECERTIFY"
    DESCEND = "DESCEND"
    HOLD = "HOLD"
    ABORT = "ABORT"
    DIVERT = "DIVERT"
    COMMIT = "COMMIT"
    TOUCHDOWN = "TOUCHDOWN"
    FAILSAFE_HOVER = "FAILSAFE_HOVER"


@dataclass
class FrameReport:
    t: float
    state: State
    zone_id: str | None
    zone_px: tuple[float, float] | None
    radius_px: float
    buffer_px: float
    alt: float
    gsd: float
    link_up: bool
    tracks: list[dict] = field(default_factory=list)
    threat: dict | None = None
    events: list[str] = field(default_factory=list)
    ego_source: str = ""
    ego_inliers: int = 0
    glimmer: dict | None = None
    cert: Certificate | None = None
    student_label: np.ndarray | None = None


# Recertifier: (rgb, zone_in_current_frame, gsd, alt, t, previous_cert) -> Certificate
Recertifier = Callable[[np.ndarray, Zone, float, float, float, Certificate], Certificate]
# EdgeGate: (rgb, zone_px, radius_px, gsd) -> {"veto": bool, "checks": [...], "backend": str}
EdgeGate = Callable[[np.ndarray, tuple[float, float], float, float], dict]


class DescentWatch:
    def __init__(self, vocab: Vocabulary, cfg: dict, camera: Camera, airframe: Airframe, atlas: Atlas,
                 student: StudentBackend, ledger: Ledger | None = None,
                 recertifier: Recertifier | None = None, edge_gate: EdgeGate | None = None,
                 recert_latency_s: float = 1.5, edge_latency_s: float = 4.0, transit_alt_m: float | None = None):
        self.vocab, self.cfg, self.cam, self.air, self.atlas = vocab, cfg, camera, airframe, atlas
        self.student, self.ledger = student, ledger
        self.recertifier, self.edge_gate = recertifier, edge_gate
        self.recert_latency_s, self.edge_latency_s = recert_latency_s, edge_latency_s
        L = cfg["landing"]
        self.commit_alt = float(L["commit_altitude_m"])
        self.hold_s = float(L["hold_seconds"])
        self.horizon_s = float(L["predicted_entry_horizon_s"])
        self.climb = float(L["abort_climb_rate_mps"])
        self.rate_hi, self.rate_lo, self.slow_below = float(L["descent_rate_high_mps"]), float(L["descent_rate_low_mps"]), float(L["slow_below_m"])
        W = cfg["watch"]
        self.confirm_frames = int(W["intrusion_frames"])
        self.ttl = float(cfg["atlas"]["certificate_ttl_s"])
        self.ego = EgoMotion(min_inliers=int(W["egomotion_min_inliers"]))
        self.tracker = GroundTracker()
        self.transit_alt = transit_alt_m
        self.dyn_ids = vocab.dynamic_ids
        self.hazard_ids = {c.id for c in vocab.hazard_classes}
        self.buffers = vocab.buffers_m()

        self.state = State.DIVERT  # picks the first zone on the first step
        self.entry: AtlasEntry | None = None
        self.failed: set[str] = set()
        self.prev: Observation | None = None
        self.prev_excl: np.ndarray | None = None
        self.K0_rgb = None
        self.zone_px_K0 = None
        self.gsd_K0 = None
        self.H_cur_K0 = np.eye(3)
        self.state_since = 0.0
        self.hold_since = None
        self.clear_since = None
        self.confirm = 0
        self.abort_alt = None
        self.pending = None
        self.frame_i = 0
        self.human_abort = False
        self.human_note = ""
        self.done = False
        self.outcome: dict = {}
        self.yaw_K0 = 0.0
        self._last_obs = None
        self.last_glimmer = None
        self.threat_points: list[tuple[float, float]] = []
        self.divert_clearance_m = float(cfg["watch"].get("divert_threat_clearance_m", 12.0))

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, payload: dict, t: float, events: list[str], msg: str):
        events.append(msg)
        if self.ledger is not None:
            self.ledger.append(kind, payload, t_sim=t)

    def _goto(self, s: State, t: float, events: list[str], why: str, **extra):
        if s == State.ABORT and self._last_obs is not None:
            self._remember_threats(self._last_obs)
        if s != self.state:
            self._log("transition", {"from": self.state.value, "to": s.value, "why": why, "zone": self.entry.zone_id if self.entry else None, **extra}, t, events, f"{s.value}: {why}")
            self.state = s
            self.state_since = t

    def human_command(self, cmd: str, note: str = ""):
        """Human operator always wins (e.g. a spoken 'stop' transcribed by Muse Voice Transcribe)."""
        if cmd.lower().strip() in {"abort", "stop", "climb"}:
            self.human_abort = True
            self.human_note = note

    def _servo(self, zone_px, gsd, yaw_deg, gain=0.8, vmax=2.5):
        ex = (zone_px[0] - self.cam.width / 2.0) * gsd
        ey = (zone_px[1] - self.cam.height / 2.0) * gsd
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        v = np.array([c * ex - s * ey, s * ex + c * ey]) * gain
        n = np.hypot(*v)
        return tuple(v / n * vmax) if n > vmax else tuple(v)

    def _descent_rate(self, alt):
        return self.rate_lo if alt < self.slow_below else self.rate_hi

    # ------------------------------------------------------------------ main step
    def step(self, obs: Observation, link_up: bool) -> tuple[Command, FrameReport]:
        t, rgb = obs.t, obs.rgb
        self._last_obs = obs
        alt = max(obs.pose_est.alt, 0.05)
        gsd = self.cam.gsd(alt)
        events: list[str] = []
        rep = FrameReport(t, self.state, None, None, 0.0, 0.0, alt, gsd, link_up)
        label = self.student.predict(rgb, gt_for_oracle_only=obs.label)
        rep.student_label = label
        excl = np.isin(label, list(self.dyn_ids))
        cmd = Command(vz=0.0)

        if self.human_abort and self.state not in (State.ABORT, State.TOUCHDOWN):
            self.human_abort = False
            self.failed.add(self.entry.zone_id if self.entry else "")
            self.abort_alt = alt + 8.0
            self._goto(State.ABORT, t, events, "human override" + (f": {self.human_note}" if self.human_note else ""))

        # ---------------------------------------------------------- DIVERT: choose next zone
        if self.state == State.DIVERT:
            options = self.atlas.usable(t, self.ttl, self.failed)
            if not options:
                self._goto(State.FAILSAFE_HOVER, t, events, "atlas exhausted - hover and request human")
            elif not self.failed:
                self.entry = options[0]  # first choice: the planner's ranking
                self._goto(State.TRANSIT, t, events, f"target {self.entry.zone_id} (cert {self.entry.cert.cert_id})")
            else:
                self.entry, note = self._choose_divert(options, obs)
                self._goto(State.TRANSIT, t, events, f"target {self.entry.zone_id} (cert {self.entry.cert.cert_id}){note}")

        # ---------------------------------------------------------- TRANSIT
        if self.state == State.TRANSIT:
            tx, ty = self.entry.world_xy_est
            dist = math.hypot(tx - obs.pose_est.x, ty - obs.pose_est.y)
            vz = 0.0
            if self.transit_alt and alt < self.transit_alt - 0.5:
                vz = self.climb
            cmd = Command(target_xy=(tx, ty), vz=vz, max_speed=5.0)
            if dist < 1.0:
                self._lock_zone(obs, events)
                self._goto(State.RECERTIFY, t, events, "arrived; re-checking zone with fresh evidence")
                self.pending = None

        # ---------------------------------------------------------- RECERTIFY (hover)
        elif self.state == State.RECERTIFY:
            zone_px = apply_homography(self.H_cur_K0, [self.zone_px_K0])[0]
            if self.pending is None:
                self.pending = {"t0": t, "rgb": rgb.copy(), "zone_px": tuple(zone_px), "gsd": gsd, "alt": alt, "label": label}
            latency = self.recert_latency_s if link_up else self.edge_latency_s
            cmd = Command(vel_xy=self._servo(zone_px, gsd, obs.pose_est.yaw), vz=0.0)
            if t - self.pending["t0"] >= latency:
                ok, why, extra = self._recertify(self.pending, link_up, t, rep)
                if ok:
                    self._goto(State.DESCEND, t, events, why)
                else:
                    self.failed.add(self.entry.zone_id)
                    self._goto(State.DIVERT, t, events, why, **extra)
                self.pending = None

        # ---------------------------------------------------------- DESCEND / HOLD / COMMIT
        elif self.state in (State.DESCEND, State.HOLD, State.COMMIT):
            zone_px = self._update_egomotion(obs, excl, rep)
            r_m = self.air.required_radius_m(alt)
            threat = self._assess(label, zone_px, gsd, r_m, t)
            rep.threat = threat
            cmd = Command(vel_xy=self._servo(zone_px, gsd, obs.pose_est.yaw), vz=0.0)
            committed = alt <= self.commit_alt

            if committed and self.state != State.COMMIT:
                self._goto(State.COMMIT, t, events, f"below {self.commit_alt:g} m - committed")
            if self.state == State.COMMIT:
                cmd.vz = -0.6
                if alt <= 0.06:
                    self._goto(State.TOUCHDOWN, t, events, "touchdown")
                    self.done = True
            elif threat["inside"]:
                self.confirm += 1
                if self.confirm >= self.confirm_frames:
                    self.failed.add(self.entry.zone_id)
                    self.abort_alt = alt + 8.0
                    self._goto(State.ABORT, t, events, f"{threat['inside']} inside the zone", threat=_jsonable(threat))
            elif threat["buffer"] or threat["predicted"]:
                self.confirm += 1
                if self.confirm >= self.confirm_frames:
                    if self.state != State.HOLD:
                        self.hold_since = t
                        reason = threat["buffer"] or f"predicted entry in {threat['predicted_tte']:.1f} s ({threat['predicted']})"
                        self._goto(State.HOLD, t, events, reason, threat=_jsonable(threat))
                    elif t - self.hold_since > self.hold_s:
                        self.failed.add(self.entry.zone_id)
                        self.abort_alt = alt + 8.0
                        self._goto(State.ABORT, t, events, f"zone did not clear in {self.hold_s:g} s")
            else:
                self.confirm = 0
                if self.state == State.HOLD:
                    self.clear_since = self.clear_since or t
                    if t - self.clear_since >= 1.0:
                        self._goto(State.DESCEND, t, events, "zone clear again - resume")
                        self.clear_since = None
                else:
                    cmd.vz = -self._descent_rate(alt)
            if self.state == State.HOLD and (threat["inside"] or threat["buffer"] or threat["predicted"]):
                self.clear_since = None
            rep.zone_px, rep.radius_px = tuple(zone_px), r_m / gsd
            rep.buffer_px = (r_m + self.buffers[self.vocab.by_name["person"].id]) / gsd

        # ---------------------------------------------------------- ABORT (climb)
        elif self.state == State.ABORT:
            zone_px = self._update_egomotion(obs, excl, rep)
            cmd = Command(vel_xy=(0.0, 0.0), vz=self.climb)
            rep.zone_px, rep.radius_px = tuple(zone_px), self.air.required_radius_m(alt) / gsd
            if alt >= (self.abort_alt or alt):
                self._goto(State.DIVERT, t, events, "abort climb complete")

        elif self.state == State.FAILSAFE_HOVER:
            cmd = Command(vel_xy=(0.0, 0.0), vz=0.0)

        rep.state = self.state
        rep.zone_id = self.entry.zone_id if self.entry else None
        if self.state in (State.DIVERT, State.TRANSIT):
            self.last_glimmer = None
        rep.glimmer = rep.glimmer or self.last_glimmer
        if rep.zone_px is None and self.zone_px_K0 is not None and self.state in (State.RECERTIFY,):
            zp = apply_homography(self.H_cur_K0, [self.zone_px_K0])[0]
            rep.zone_px, rep.radius_px = tuple(zp), self.air.required_radius_m(alt) / gsd
        rep.events = events
        rep.cert = self.entry.cert if self.entry else None
        rep.tracks = [
            {"id": tr.track_id, "cls": self.vocab.by_id[tr.cls].name, "px": tr.px, "speed": round(tr.speed, 2),
             "tte": tr.time_to_enter(self.air.required_radius_m(alt) + self.buffers.get(tr.cls, 0.0))}
            for tr in self.tracker.tracks if tr.last_t == t
        ]
        self.prev, self.prev_excl = obs, excl
        self.frame_i += 1
        return cmd, rep

    # ------------------------------------------------------------------ divert selection
    def _remember_threats(self, obs: Observation):
        """Store where tracked movers are heading (world frame) so DIVERT avoids their path."""
        if self.entry is None:
            return
        zx, zy = self.entry.world_xy_est
        c, s = math.cos(math.radians(self.yaw_K0)), math.sin(math.radians(self.yaw_K0))
        R = np.array([[c, -s], [s, c]])
        for tr in self.tracker.tracks:
            for dt in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
                p = R @ (tr.xy_m + tr.vel * dt)
                self.threat_points.append((zx + p[0], zy + p[1]))

    def _choose_divert(self, options: list[AtlasEntry], obs: Observation):
        here = (obs.pose_est.x, obs.pose_est.y)
        def clear_of_threats(e):
            return all(math.dist(e.world_xy_est, p) >= self.divert_clearance_m for p in self.threat_points)
        safe = [e for e in options if clear_of_threats(e)]
        note = ""
        if self.threat_points and len(safe) < len(options):
            note = f"; skipped {len(options) - len(safe)} zone(s) on the intruder's path"
        pool = safe or options
        # emergency: prefer the nearest certified zone, then the better-scored one
        best = min(pool, key=lambda e: (round(math.dist(here, e.world_xy_est) / 10.0), -e.zone.score))
        return best, note

    # ------------------------------------------------------------------ zone locking
    def _lock_zone(self, obs: Observation, events: list[str]):
        """Find the atlas zone in the current frame: vision first, telemetry fallback."""
        kf = self.atlas.keyframe_rgb
        H_tel = telemetry_homography(self.atlas.keyframe_pose_est, obs.pose_est.as_tuple(), self.cam, self.cam.width, self.cam.height)
        reg = self.ego.register(kf, obs.rgb)
        if reg.H is not None:
            zp = apply_homography(reg.H, [self.entry.zone.center_px])[0]
            src = f"vision lock ({reg.inliers} inliers)"
        else:
            # coarse-to-fine: warp the keyframe into the current view with telemetry, then register
            warped = cv2.warpPerspective(kf, H_tel, (obs.rgb.shape[1], obs.rgb.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            valid = cv2.warpPerspective(np.ones(kf.shape[:2], np.uint8), H_tel, (obs.rgb.shape[1], obs.rgb.shape[0])) == 0
            reg2 = self.ego.register(warped, obs.rgb, exclude_src=valid)
            if reg2.H is not None:
                zp = apply_homography(reg2.H @ H_tel, [self.entry.zone.center_px])[0]
                src = f"vision lock, coarse-to-fine ({reg2.inliers} inliers)"
            else:
                zp = apply_homography(H_tel, [self.entry.zone.center_px])[0]
                src = "telemetry lock (vision registration failed)"
        self.K0_rgb, self.zone_px_K0, self.gsd_K0 = obs.rgb.copy(), tuple(zp), self.cam.gsd(max(obs.pose_est.alt, 0.05))
        self.yaw_K0 = obs.pose_est.yaw
        self.H_cur_K0 = np.eye(3)
        self.tracker.reset()
        self.confirm = 0
        events.append(f"{self.entry.zone_id} located by {src}")

    def _update_egomotion(self, obs: Observation, excl: np.ndarray, rep: FrameReport):
        if self.prev is not None:
            reg = self.ego.register(self.prev.rgb, obs.rgb, self.prev_excl, excl)
            if reg.H is not None:
                H_cur_prev, rep.ego_source, rep.ego_inliers = reg.H, "vision", reg.inliers
            else:
                H_cur_prev = telemetry_homography(self.prev.pose_est.as_tuple(), obs.pose_est.as_tuple(), self.cam, self.cam.width, self.cam.height)
                rep.ego_source, rep.ego_inliers = "telemetry", reg.inliers
            self.H_cur_K0 = H_cur_prev @ self.H_cur_K0
        # re-anchor the keyframe when the view has zoomed in a lot (keeps tracks well-conditioned)
        gsd_now = self.cam.gsd(max(obs.pose_est.alt, 0.05))
        zone_px = apply_homography(self.H_cur_K0, [self.zone_px_K0])[0]
        if self.gsd_K0 / gsd_now > 2.0:
            Hk = self.H_cur_K0
            ang = math.atan2(Hk[1, 0], Hk[0, 0])
            c, s = math.cos(-ang), math.sin(-ang)
            Rm = np.array([[c, -s], [s, c]])
            for tr in self.tracker.tracks:  # express tracks in the new keyframe orientation
                tr.xy_m, tr.vel = Rm.T @ tr.xy_m, Rm.T @ tr.vel
            self.K0_rgb, self.zone_px_K0, self.gsd_K0, self.H_cur_K0 = obs.rgb.copy(), tuple(zone_px), gsd_now, np.eye(3)
            self.yaw_K0 = obs.pose_est.yaw
        return zone_px

    # ------------------------------------------------------------------ threat assessment
    def _assess(self, label, zone_px, gsd, r_m, t) -> dict:
        out = {"inside": None, "buffer": None, "predicted": None, "predicted_tte": math.inf, "nearest": {}}
        cx, cy = zone_px
        H_K0_cur = np.linalg.inv(self.H_cur_K0)
        dets = []
        min_area_px = {cid: (0.03 if cid in self.dyn_ids else 0.15) / (gsd * gsd) for cid in self.hazard_ids}
        for cid in self.hazard_ids:
            m = label == cid
            if not m.any():
                continue
            name = self.vocab.by_id[cid].name
            buf = self.buffers.get(cid, 0.0)
            comp, n = ndimage.label(m)
            if n == 0:
                continue
            idx = np.arange(1, n + 1)
            areas = ndimage.sum(m, comp, idx)
            coms = ndimage.center_of_mass(m, comp, idx)
            slices = ndimage.find_objects(comp)
            for k, area, com, sl in zip(idx, areas, coms, slices):
                if area < max(2.0, min_area_px[cid]) or sl is None:
                    continue
                ys, xs = np.nonzero(comp[sl] == k)
                ys, xs = ys + sl[0].start, xs + sl[1].start
                d_m = float(np.min(np.hypot(xs - cx, ys - cy))) * gsd
                out["nearest"][name] = min(out["nearest"].get(name, math.inf), round(d_m, 2))
                if d_m < r_m:
                    out["inside"] = out["inside"] or name
                elif cid in self.dyn_ids and d_m < r_m + buf:
                    out["buffer"] = out["buffer"] or f"{name} {d_m:.1f} m from centre"
                if cid in self.dyn_ids:
                    py, px = com
                    k0 = apply_homography(H_K0_cur, [(px, py)])[0]
                    xy_m = ((k0[0] - self.zone_px_K0[0]) * self.gsd_K0, (k0[1] - self.zone_px_K0[1]) * self.gsd_K0)
                    dets.append(Detection(cid, xy_m, math.sqrt(area / math.pi) * gsd, (float(px), float(py))))
        for tr in self.tracker.update(dets, t):
            if tr.hits < 6 or tr.speed < 0.5:
                continue
            tte = tr.time_to_enter(r_m + self.buffers.get(tr.cls, 0.0))
            if 0 < tte < self.horizon_s and tte < out["predicted_tte"]:
                out["predicted"], out["predicted_tte"] = f"{self.vocab.by_id[tr.cls].name} #{tr.track_id}", tte
        return out

    # ------------------------------------------------------------------ recertification
    def _recertify(self, p: dict, link_up: bool, t: float, rep: FrameReport):
        r_m = self.air.required_radius_m(p["alt"])
        zone = Zone(self.entry.zone_id, p["zone_px"], r_m, r_m / p["gsd"], 0.0, 0.0, True)
        if link_up and self.recertifier is not None:
            cert = self.recertifier(p["rgb"], zone, p["gsd"], p["alt"], t, self.entry.cert)
            rep.cert = cert
            if cert.passed:
                self.entry.cert = cert
                return True, f"fresh certificate {cert.cert_id} PASS ({cert.evidence_backend} evidence)", {}
            return False, f"fresh certificate {cert.cert_id} FAIL: {'; '.join(cert.reasons[:2])}", {"cert": cert.cert_id}
        # link down: student must be clear AND the edge model must not object
        threat = self._assess(p["label"], p["zone_px"], p["gsd"], r_m, t)
        if threat["inside"] or threat["buffer"]:
            return False, f"student sees {threat['inside'] or threat['buffer']} (link down)", {}
        if self.edge_gate is None:
            return True, "link down, no edge model: student-only check passed (cached certificate)", {}
        g = self.edge_gate(p["rgb"], p["zone_px"], r_m / p["gsd"], p["gsd"])
        rep.glimmer = self.last_glimmer = g
        if g.get("veto"):
            return False, f"Muse Glimmer veto: {g.get('reason', '')}", {"glimmer": g}
        return True, f"link down: student clear + Muse Glimmer agrees ({g.get('backend', 'glimmer')})", {}


def _jsonable(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in d.items()}
