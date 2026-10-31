"""Construct evidence / student / edge / planner backends from CLI strings."""

from __future__ import annotations

from .config import load_env
from .vocab import Vocabulary


def make_sam(kind: str, vocab: Vocabulary, cfg: dict):
    kind = kind.lower()
    if kind in ("meta", "meta-api", "api"):
        load_env()
        from .meta.sam import MetaApiSam

        return MetaApiSam(model=cfg["models"]["sam"])
    if kind in ("local", "open-weights"):
        from .meta.sam import LocalSam

        return LocalSam()
    if kind == "oracle":
        from .meta.sam import OracleSam

        return OracleSam(vocab)
    raise ValueError(f"unknown evidence backend {kind!r} (meta-api | local | oracle)")


def make_student(kind: str, cfg: dict):
    if kind == "oracle":
        from .perception.student import OracleStudent

        return OracleStudent(min_area_px=4)
    from .perception.student import TorchStudent

    return TorchStudent(kind, input_px=int(cfg["watch"]["student_input_px"]))


def make_glimmer(kind: str, vocab: Vocabulary):
    if kind in ("none", ""):
        return None
    if kind == "oracle":
        from .meta.glimmer import OracleGlimmer

        return OracleGlimmer(vocab)
    load_env()
    from .meta.glimmer import GlimmerGate

    return GlimmerGate()


def make_spark(kind: str, cfg: dict):
    if kind != "spark":
        return None
    load_env()
    from .meta.spark import SparkClient

    return SparkClient(model=cfg["models"]["spark"], effort=cfg["models"]["spark_effort_planning"], max_tokens=int(cfg["models"]["spark_max_tokens"]))
