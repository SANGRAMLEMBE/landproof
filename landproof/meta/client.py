"""Meta Model API client factory, retries and image encoding. Spend is tracked in landproof.budget."""

from __future__ import annotations

import base64
import io
import os
import random
import time

import numpy as np
from PIL import Image

DEFAULT_BASE = "https://api.meta.ai/v1"

from ..budget import BUDGET, PRICES  # noqa: F401  (re-exported)


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def base_url() -> str:
    return os.environ.get("META_API_BASE", DEFAULT_BASE).rstrip("/")


def openai_client(timeout: float = 120.0):
    from openai import OpenAI

    return OpenAI(base_url=base_url(), api_key=api_key(), timeout=timeout, max_retries=0)


def with_retries(fn, *, attempts: int = 5, base_delay: float = 1.5, retry_on=(429, 500, 502, 503, 504)):
    """Call fn() with exponential backoff on rate limits / server errors."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # openai.APIStatusError, requests.HTTPError, timeouts
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            transient = status in retry_on or status is None and "timeout" in type(e).__name__.lower()
            if not transient or i == attempts - 1:
                raise
            last = e
            time.sleep(base_delay * (2**i) + random.random())
    raise last  # pragma: no cover


def to_data_url(image: np.ndarray | Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def decode_b64_image(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


# Back-compat name: every script prints METER.report(); it is the persistent budget guard.
METER = BUDGET
