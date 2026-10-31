"""Muse Image 1.0 (POST /v1/images/edits and /v1/images/generations)."""

from __future__ import annotations

import cv2
import numpy as np
import requests

from .client import METER, api_key, base_url, decode_b64_image, to_data_url, with_retries


def _size_for(w: int, h: int) -> str:
    # size sets the aspect ratio (not exact pixels); keep the source aspect
    s = 1024 / max(w, h)
    return f"{int(round(w * s / 16) * 16)}x{int(round(h * s / 16) * 16)}"


class MuseImage:
    def __init__(self, model: str = "muse-image-1.0", timeout: float = 240.0):
        self.model, self.timeout = model, timeout

    def _post(self, path: str, body: dict) -> np.ndarray:
        def call():
            r = requests.post(
                f"{base_url()}{path}",
                headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            if r.status_code >= 400:
                err = requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
                err.response = r
                raise err
            return r.json()

        data = with_retries(call, attempts=4)
        METER.add("muse-image-1.0/image")
        return decode_b64_image(data["data"][0]["b64_json"])

    def edit(self, images: list[np.ndarray], prompt: str, keep_size: bool = True) -> np.ndarray:
        h, w = images[0].shape[:2]
        body = {
            "model": self.model,
            "prompt": prompt,
            "images": [{"image_url": to_data_url(im, fmt="PNG")} for im in images],
            "n": 1,
            "size": _size_for(w, h),
            "response_format": "b64_json",
            "output_format": "png",
        }
        out = self._post("/images/edits", body)
        if keep_size and out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        return out

    def generate(self, prompt: str, size: str = "1024x1024") -> np.ndarray:
        return self._post("/images/generations", {"model": self.model, "prompt": prompt, "n": 1, "size": size, "response_format": "b64_json", "output_format": "png"})
