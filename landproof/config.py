"""Configuration loading (YAML + environment)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load configs/default.yaml, optionally merged with another YAML and a dict."""
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_update(cfg, overrides)
    spark_env = os.environ.get("LANDPROOF_SPARK_MODEL")
    if spark_env:
        cfg["models"]["spark"] = spark_env
    return cfg


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
