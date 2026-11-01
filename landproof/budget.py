"""Persistent spend guard for the Meta Model API credits.

Every paid call goes through `BUDGET.reserve(...)` *before* the request (estimated cost x
safety factor) and `BUDGET.add(...)` after it (actual units). The running total lives in
cache/spend.json, so the cap holds across scripts, restarts and days.

  LANDPROOF_BUDGET_USD=130   hard cap (default 130)
  python scripts/spend.py    show the ledger

Prices are public list prices (Oct 2026). Your Meta Billing page is the source of truth:
also set a Spend Limit there.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

PRICES = {
    "sam-3.1/image": 2.50 / 1000,
    "sam-3.1/video_frame": 0.20 / 1000,
    "muse-spark-1.3/in": 1.25 / 1e6,
    "muse-spark-1.3/out": 4.25 / 1e6,
    "muse-image-1.0/image": 0.01,
    "muse-voice-transcribe-1.0/second": 0.18 / 3600,
}
SAFETY = 1.25  # reservations over-estimate by 25%


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path: str | Path | None = None, cap_usd: float | None = None):
        from .config import REPO_ROOT

        self.path = Path(path or os.environ.get("LANDPROOF_SPEND_FILE", REPO_ROOT / "cache" / "spend.json"))
        self._cap_arg = cap_usd
        self._lock = threading.Lock()
        self.session_usd = 0.0
        self.session_calls: dict[str, float] = {}
        self._state = None

    @property
    def cap(self) -> float:
        """Read lazily so a value from .env (loaded after import) is honoured."""
        return float(self._cap_arg if self._cap_arg is not None else os.environ.get("LANDPROOF_BUDGET_USD", "130"))

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        if self._state is None:
            try:
                self._state = json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                self._state = {"total_usd": 0.0, "by_key": {}, "units": {}, "log": []}
        return self._state

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=1))
        os.replace(tmp, self.path)

    @property
    def total(self) -> float:
        with self._lock:
            return float(self._load()["total_usd"])

    @property
    def remaining(self) -> float:
        return self.cap - self.total

    # ------------------------------------------------------------------ API
    def cost(self, key: str, units: float) -> float:
        return PRICES.get(key, 0.0) * units

    def reserve(self, key: str, units: float = 1.0, usd: float | None = None) -> None:
        """Raise BudgetExceeded if this call could push spend over the cap."""
        est = (usd if usd is not None else self.cost(key, units)) * SAFETY
        with self._lock:
            total = float(self._load()["total_usd"])
            if total + est > self.cap:
                raise BudgetExceeded(
                    f"budget cap ${self.cap:.2f} would be exceeded: spent ${total:.2f}, this call ~${est:.3f} ({key}). "
                    f"Raise LANDPROOF_BUDGET_USD only if you are sure.")

    def add(self, key: str, units: float = 1.0, usd: float | None = None, note: str = "") -> float:
        c = usd if usd is not None else self.cost(key, units)
        with self._lock:
            st = self._load()
            st["total_usd"] = float(st["total_usd"]) + c
            st["by_key"][key] = float(st["by_key"].get(key, 0.0)) + c
            st["units"][key] = float(st["units"].get(key, 0.0)) + units
            if note or c > 0.05:
                st["log"] = (st["log"] + [{"t": round(time.time()), "key": key, "units": units, "usd": round(c, 5), "note": note}])[-500:]
            self.session_usd += c
            self.session_calls[key] = self.session_calls.get(key, 0) + units
            self._save()
        return c

    def preflight(self, items: dict[str, float], label: str = "") -> float:
        """Print and check an estimate for a whole job: {price_key: units, "usd": extra}. Returns est USD."""
        est = sum(self.cost(k, u) for k, u in items.items() if k != "usd") + float(items.get("usd", 0.0))
        print(f"[budget] {label} estimate ~${est:.2f} | spent ${self.total:.2f} of ${self.cap:.2f} cap | remaining ${self.remaining:.2f}")
        if est * SAFETY > self.remaining:
            raise BudgetExceeded(f"{label}: estimated ${est:.2f} (x{SAFETY} safety) exceeds remaining ${self.remaining:.2f}. Use --limit to shrink the job.")
        return est

    def report(self) -> str:
        parts = ", ".join(f"{k}: {v:g}" for k, v in sorted(self.session_calls.items()))
        return f"this run ~${self.session_usd:.3f} ({parts}) | all-time ${self.total:.2f} / cap ${self.cap:.2f}"


BUDGET = Budget()
