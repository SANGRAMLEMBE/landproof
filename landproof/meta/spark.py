"""Muse Spark 1.3 wrapper (Chat Completions, tool calling, structured output)."""

from __future__ import annotations

import json

import numpy as np

from .client import METER, openai_client, to_data_url, with_retries


def image_part(rgb: np.ndarray, max_side: int = 1280) -> dict:
    import cv2

    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return {"type": "image_url", "image_url": {"url": to_data_url(rgb)}}


class SparkClient:
    def __init__(self, model: str = "muse-spark-1.3", effort: str = "low", max_tokens: int = 6000, timeout: float = 180.0):
        if "contributor" in model:
            print("WARNING: contributor-tier traffic may be used by Meta for training; never send footage of people.")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.client = openai_client(timeout=timeout)

    def _usage(self, resp):
        u = getattr(resp, "usage", None)
        if u is not None:
            METER.add("muse-spark-1.3/in", getattr(u, "prompt_tokens", 0) or 0)
            METER.add("muse-spark-1.3/out", getattr(u, "completion_tokens", 0) or 0)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, effort: str | None = None, **kw):
        args = dict(model=self.model, messages=messages, reasoning_effort=effort or self.effort, max_tokens=self.max_tokens, **kw)
        if tools:
            args["tools"] = tools
            args["parallel_tool_calls"] = True
        resp = with_retries(lambda: self.client.chat.completions.create(**args))
        self._usage(resp)
        return resp.choices[0].message

    def json(self, messages: list[dict], schema: dict, name: str = "result", effort: str | None = None) -> dict:
        args = dict(
            model=self.model,
            messages=messages,
            reasoning_effort=effort or self.effort,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        )
        resp = with_retries(lambda: self.client.chat.completions.create(**args))
        self._usage(resp)
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("Muse Spark returned empty content - raise max_tokens (reasoning tokens share the budget)")
        return json.loads(content)
