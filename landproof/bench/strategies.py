"""Landing-site selection strategies compared in LandBench.

Every strategy sees one nadir frame and returns a touchdown pixel (or abstains).
Ground truth is only used by the scorer in landbench.py, never by a strategy
(except the explicitly named 'oracle' evidence used for offline smoke tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import Airframe, Camera
from ..scorer import ZoneScorer, texture_energy
from ..vocab import Vocabulary


@dataclass
class Pick:
    xy: tuple[float, float] | None   # None = abstain (hover / ask human)
    claimed_safe: bool
    info: dict


class Strategy:
    name = "base"

    def pick(self, rgb: np.ndarray, gsd: float, alt: float, ctx: dict) -> Pick:  # pragma: no cover
        raise NotImplementedError


class BlindLand(Strategy):
    """Today's LAND failsafe: descend straight down."""

    name = "blind-LAND"

    def pick(self, rgb, gsd, alt, ctx):
        return Pick((rgb.shape[1] / 2, rgb.shape[0] / 2), True, {})


class FlattestPatch(Strategy):
    """Classic heuristic: the smoothest, most uniform patch of sufficient size."""

    name = "flattest-patch"

    def __init__(self, airframe: Airframe):
        self.air = airframe

    def pick(self, rgb, gsd, alt, ctx):
        r_px = self.air.required_radius_m(alt) / gsd
        e = texture_energy(rgb)
        import cv2

        k = max(3, int(r_px * 2) | 1)
        local = cv2.blur(e, (k, k))
        m = int(math.ceil(r_px)) + 2
        local[:m, :], local[-m:, :], local[:, :m], local[:, -m:] = 1, 1, 1, 1
        y, x = np.unravel_index(np.argmin(local), local.shape)
        flat = 1.0 - float(local[y, x])
        return Pick((float(x), float(y)), True, {"flatness": round(flat, 3)})


class MaskScorer(Strategy):
    """Scorer over hazard masks from some source (student, fixed-vocab, SAM...)."""

    def __init__(self, name: str, vocab: Vocabulary, cfg: dict, airframe: Airframe, mask_source, classes: list[str] | None = None):
        self.name = name
        self.vocab = vocab
        self.air = airframe
        self.classes = classes
        self.source = mask_source  # callable(rgb, ctx) -> label map
        sub = {c.id: c.buffer_m for c in vocab.hazard_classes if classes is None or c.name in classes}
        self.scorer = ZoneScorer(sub, cfg)
        self.ids = list(sub)

    def pick(self, rgb, gsd, alt, ctx):
        label = self.source(rgb, ctx)
        masks = {cid: label == cid for cid in self.ids}
        zones = self.scorer.score(masks, gsd, self.air.required_radius_m(alt), landable_mask=(label == 0))
        if not zones:
            return Pick(None, False, {"reason": "no feasible zone"})
        return Pick(zones[0].center_px, True, {"zone": zones[0].zone_id, "clearance_m": round(zones[0].clearance_m, 2)})


class LandProofStrategy(Strategy):
    """Full stack: atlas agent (planner + SAM 3.1 evidence + verifier). Abstains if nothing certifies."""

    name = "landproof"

    def __init__(self, agent_factory):
        self.agent_factory = agent_factory  # callable(rgb, ctx) -> AtlasAgent ready for this frame

    def pick(self, rgb, gsd, alt, ctx):
        agent = self.agent_factory(rgb, ctx)
        atlas = agent.build(rgb, (0.0, 0.0, alt, 0.0))
        if not atlas.entries:
            return Pick(None, False, {"reason": "no zone certified", "rejected": len(atlas.rejected)})
        e = atlas.entries[0]
        return Pick(e.zone.center_px, True, {"zone": e.zone_id, "cert": e.cert.cert_id, "rejected": len(atlas.rejected)})
