"""Vetted hazard vocabulary.

The vocabulary is the contract between the LLM and the rest of the system: Muse Spark
can only name phrases that appear here (JSON-schema enum), and every phrase can carry
validation statistics measured against labeled frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import CONFIG_DIR

IGNORE = 255


@dataclass
class HazardClass:
    name: str
    id: int
    buffer_m: float
    mandatory: bool
    dynamic: bool
    hazard: bool
    fixed_vocab: bool
    phrases: list[str]
    # phrase -> {"iou": .., "precision": .., "recall": .., "n": ..} from validation
    stats: dict[str, dict] = field(default_factory=dict)

    @property
    def default_phrase(self) -> str:
        if self.stats:
            return max(self.phrases, key=lambda p: self.stats.get(p, {}).get("iou", -1.0))
        return self.phrases[0]


class Vocabulary:
    def __init__(self, classes: list[HazardClass]):
        self.classes = sorted(classes, key=lambda c: c.id)
        self.by_name = {c.name: c for c in self.classes}
        self.by_id = {c.id: c for c in self.classes}
        self.phrase_to_class: dict[str, HazardClass] = {}
        for c in self.classes:
            for p in c.phrases:
                if p in self.phrase_to_class:
                    raise ValueError(f"phrase {p!r} is listed under two classes")
                self.phrase_to_class[p] = c

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Vocabulary":
        """Load the validated vocabulary if it exists, else the raw one."""
        if path is None:
            validated = CONFIG_DIR / "hazards.validated.yaml"
            path = validated if validated.exists() else CONFIG_DIR / "hazards.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        classes = []
        for c in raw["classes"]:
            classes.append(
                HazardClass(
                    name=c["name"],
                    id=int(c["id"]),
                    buffer_m=float(c.get("buffer_m", 0.0)),
                    mandatory=bool(c.get("mandatory", False)),
                    dynamic=bool(c.get("dynamic", False)),
                    hazard=bool(c.get("hazard", True)),
                    fixed_vocab=bool(c.get("fixed_vocab", False)),
                    phrases=list(c["phrases"]),
                    stats=dict(c.get("stats", {}) or {}),
                )
            )
        return cls(classes)

    def dump(self, path: str | Path) -> None:
        data = {
            "classes": [
                {
                    "name": c.name,
                    "id": c.id,
                    "buffer_m": c.buffer_m,
                    "mandatory": c.mandatory,
                    "dynamic": c.dynamic,
                    "hazard": c.hazard,
                    "fixed_vocab": c.fixed_vocab,
                    "phrases": c.phrases,
                    "stats": c.stats,
                }
                for c in self.classes
            ]
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------ queries
    @property
    def num_classes(self) -> int:
        return max(c.id for c in self.classes) + 1

    @property
    def hazard_classes(self) -> list[HazardClass]:
        return [c for c in self.classes if c.hazard]

    @property
    def mandatory_classes(self) -> list[HazardClass]:
        return [c for c in self.classes if c.mandatory]

    @property
    def dynamic_ids(self) -> set[int]:
        return {c.id for c in self.classes if c.dynamic}

    @property
    def hazard_phrases(self) -> list[str]:
        return [p for c in self.hazard_classes for p in c.phrases]

    @property
    def all_phrases(self) -> list[str]:
        return [p for c in self.classes for p in c.phrases]

    def class_of(self, phrase: str) -> HazardClass:
        try:
            return self.phrase_to_class[phrase]
        except KeyError as e:
            raise KeyError(f"phrase {phrase!r} is not in the vetted vocabulary") from e

    def buffers_m(self) -> dict[int, float]:
        return {c.id: c.buffer_m for c in self.hazard_classes}

    def subset(self, names: list[str]) -> "Vocabulary":
        return Vocabulary([c for c in self.classes if c.name in names])

    def summary_for_llm(self) -> str:
        lines = []
        for c in self.hazard_classes:
            tag = "MANDATORY" if c.mandatory else "optional"
            lines.append(f"- {c.name} ({tag}, keep-out {c.buffer_m:g} m): " + ", ".join(c.phrases))
        return "\n".join(lines)
