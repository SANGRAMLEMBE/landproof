"""Muse Glimmer edge gate (open weights, served by vLLM — see modal_apps/glimmer_server.py).

Glimmer runs on the ground-station / edge box when the cloud link is down. It can only
VETO a zone (any critical hazard 'present' or 'unsure' -> veto); it can never approve
a zone the student model flags. Its 82% hallucination rate on open-ended knowledge
questions (Artificial Analysis) is why it gets a narrow, structured, visual yes/no task.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from .client import to_data_url, with_retries

CHECK_CLASSES = ["person", "vehicle", "animal", "standing water", "power line or pole", "other obstacle"]
CRITICAL = {"person", "vehicle", "animal", "standing water", "power line or pole"}

SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard": {"type": "string", "enum": CHECK_CLASSES},
                    "verdict": {"type": "string", "enum": ["present", "absent", "unsure"]},
                    "evidence": {"type": "string"},
                },
                "required": ["hazard", "verdict", "evidence"],
                "additionalProperties": False,
            },
        },
        "overall": {"type": "string", "enum": ["clear", "not_clear"]},
    },
    "required": ["checks", "overall"],
    "additionalProperties": False,
}

PROMPT = (
    "Reasoning strength: low.\n"
    "You are a safety checker on an emergency-landing drone. The image is a straight-down camera crop. "
    "The cyan circle is the planned touchdown area ({diam:.1f} m across). For each hazard type, say whether it is "
    "present INSIDE or TOUCHING the circle: {classes}. Answer 'unsure' if you cannot tell. "
    "Do not describe anything outside the circle. Return JSON only."
)


def zone_crop(rgb: np.ndarray, zone_px, radius_px: float, context: float = 2.2, out_px: int = 768) -> np.ndarray:
    h, w = rgb.shape[:2]
    half = int(max(radius_px * context, 24))
    cx, cy = int(zone_px[0]), int(zone_px[1])
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    crop = rgb[y0:y1, x0:x1].copy()
    s = out_px / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    cv2.circle(crop, (int((cx - x0) * s), int((cy - y0) * s)), int(radius_px * s), (0, 255, 255), 2, cv2.LINE_AA)
    return crop


class GlimmerGate:
    backend = "muse-glimmer@vllm"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, timeout: float = 60.0):
        from openai import OpenAI

        self.base_url = base_url or os.environ.get("GLIMMER_BASE_URL", "")
        if not self.base_url:
            raise RuntimeError("GLIMMER_BASE_URL not set (deploy modal_apps/glimmer_server.py first)")
        self.model = model or os.environ.get("GLIMMER_MODEL", "muse-glimmer")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key or os.environ.get("GLIMMER_API_KEY", "EMPTY"), timeout=timeout, max_retries=0)

    def __call__(self, rgb, zone_px, radius_px, gsd) -> dict:
        crop = zone_crop(rgb, zone_px, radius_px)
        text = PROMPT.format(diam=2 * radius_px * gsd, classes=", ".join(CHECK_CLASSES))
        t0 = time.perf_counter()
        resp = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": to_data_url(crop)}}]}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                response_format={"type": "json_schema", "json_schema": {"name": "zone_check", "schema": SCHEMA}},
            ),
            attempts=2,
        )
        latency = time.perf_counter() - t0
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {"veto": True, "reason": "unparseable edge-model output (fail safe)", "backend": self.backend, "latency_s": latency}
        return decide(data, self.backend, latency)


def decide(data: dict, backend: str, latency: float | None = None) -> dict:
    flagged = [c for c in data.get("checks", []) if c.get("hazard") in CRITICAL and c.get("verdict") in ("present", "unsure")]
    missing = CRITICAL - {c.get("hazard") for c in data.get("checks", [])}
    veto = bool(flagged) or bool(missing) or data.get("overall") == "not_clear"
    reason = "; ".join(f"{c['hazard']} {c['verdict']}" for c in flagged) or ("incomplete answer" if missing else ("model says not clear" if veto else ""))
    return {"veto": veto, "reason": reason, "checks": data.get("checks", []), "backend": backend, "latency_s": latency}


class OracleGlimmer:
    """GT stand-in for offline tests (tagged backend='oracle-glimmer')."""

    backend = "oracle-glimmer"
    MAP = {"person": "person", "vehicle": "vehicle", "animal": "animal", "standing water": "water", "power line or pole": "wire_pole", "other obstacle": "obstacle"}

    def __init__(self, vocab):
        self.vocab = vocab
        self.label = None

    def set_label(self, label):
        self.label = label

    def __call__(self, rgb, zone_px, radius_px, gsd) -> dict:
        h, w = self.label.shape
        yy, xx = np.ogrid[:h, :w]
        disk = (xx - zone_px[0]) ** 2 + (yy - zone_px[1]) ** 2 <= radius_px**2
        present = set(np.unique(self.label[disk]).tolist())
        checks = []
        for k, name in self.MAP.items():
            cid = self.vocab.by_name[name].id
            checks.append({"hazard": k, "verdict": "present" if cid in present else "absent", "evidence": "ground truth"})
        return decide({"checks": checks, "overall": "clear"}, self.backend, 0.0)
