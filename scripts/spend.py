#!/usr/bin/env python3
"""Show what LandProof has spent on the Meta Model API (cache/spend.json) against the cap.

  python scripts/spend.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landproof.budget import BUDGET  # noqa: E402


def main():
    st = BUDGET._load()
    print(f"spent ${st['total_usd']:.2f} of ${BUDGET.cap:.2f} cap  (remaining ${BUDGET.remaining:.2f})")
    for k, v in sorted(st["by_key"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} ${v:8.3f}   units {st['units'].get(k, 0):g}")
    if st["log"]:
        print("recent large items:")
        for e in st["log"][-10:]:
            print("  ", json.dumps(e))
    print("Cross-check with the Billing page on dev.meta.ai (that is the source of truth).")


if __name__ == "__main__":
    main()
