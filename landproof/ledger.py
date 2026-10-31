"""Tamper-evident, hash-chained decision log (JSONL).

Every entry stores sha256(prev_hash || canonical_json(entry_without_hash)). Changing any
past decision breaks every hash after it, which `verify_chain` detects.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.entries = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def append(self, kind: str, payload: dict, t_sim: float | None = None) -> dict:
        with self._lock:
            entry = {
                "seq": len(self.entries),
                "kind": kind,
                "wall_time": round(time.time(), 3),
                "t_sim": None if t_sim is None else round(float(t_sim), 3),
                "payload": json.loads(canonical(payload)),
                "prev_hash": self.head,
            }
            entry["hash"] = hashlib.sha256((entry["prev_hash"] + canonical(entry)).encode()).hexdigest()
            self.entries.append(entry)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(canonical(entry) + "\n")
            return entry

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e["kind"] == kind]


def verify_chain(entries: list[dict]) -> tuple[bool, int | None]:
    """Return (ok, index_of_first_bad_entry)."""
    prev = GENESIS
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "hash"}
        if body.get("prev_hash") != prev:
            return False, i
        h = hashlib.sha256((prev + canonical(body)).encode()).hexdigest()
        if h != e.get("hash"):
            return False, i
        prev = h
    return True, None
