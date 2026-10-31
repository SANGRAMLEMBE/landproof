"""LandBench scoring.

Primary metric: FALSE-SAFE RATE = picks that ground truth says are unsafe / picks made.
A strategy that abstains (hover, ask a human) is not penalised as unsafe, but its
abstention rate is reported so "never land" cannot win.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..vocab import IGNORE, Vocabulary
from .strategies import Strategy


@dataclass
class Frame:
    frame_id: str
    rgb: np.ndarray
    gt: np.ndarray
    gsd: float
    alt: float
    source: str


def gt_safety(gt: np.ndarray, xy, gsd: float, vocab: Vocabulary, footprint_m: float, person_clear_m: float = 5.0) -> dict:
    h, w = gt.shape
    cx, cy = xy
    r = footprint_m / gsd
    rp = person_clear_m / gsd
    R = int(math.ceil(max(r, rp))) + 1
    x0, x1, y0, y1 = max(0, int(cx) - R), min(w, int(cx) + R + 1), max(0, int(cy) - R), min(h, int(cy) + R + 1)
    win = gt[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    hazard = {c.id: c.name for c in vocab.hazard_classes}
    under = {hazard[int(v)] for v in np.unique(win[d2 <= r * r]) if int(v) in hazard}
    pid = vocab.by_name["person"].id
    if (win[d2 <= rp * rp] == pid).any():
        under.add("person")
    out_of_view = cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h
    unknown = float((win[d2 <= r * r] == IGNORE).mean()) if (d2 <= r * r).any() else 1.0
    return {"safe": not under and not out_of_view, "hazards": sorted(under), "out_of_view": out_of_view, "unlabeled_frac": round(unknown, 3)}


def run_bench(frames: list[Frame], strategies: list[Strategy], vocab: Vocabulary, footprint_m: float, out_dir: str | Path | None = None, ctx_fn=None) -> dict:
    rows = []
    for f in frames:
        ctx = ctx_fn(f) if ctx_fn else {"gt": f.gt}
        for s in strategies:
            p = s.pick(f.rgb, f.gsd, f.alt, ctx)
            if p.xy is None:
                rows.append({"frame": f.frame_id, "source": f.source, "strategy": s.name, "picked": False, "safe": None, "hazards": "", "info": json.dumps(p.info)})
                continue
            g = gt_safety(f.gt, p.xy, f.gsd, vocab, footprint_m)
            rows.append({"frame": f.frame_id, "source": f.source, "strategy": s.name, "picked": True, "safe": g["safe"], "hazards": ",".join(g["hazards"]) or ("out_of_view" if g["out_of_view"] else ""),
                         "x": round(p.xy[0], 1), "y": round(p.xy[1], 1), "info": json.dumps(p.info)})
    summary = summarize(rows, [s.name for s in strategies])
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "landbench_rows.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            wr.writeheader()
            wr.writerows(rows)
        (out / "landbench_summary.json").write_text(json.dumps(summary, indent=2))
        (out / "landbench_table.md").write_text(to_markdown(summary))
    return summary


def summarize(rows: list[dict], order: list[str]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)
    out = {}
    for name in order:
        rs = by[name]
        n = len(rs)
        picked = [r for r in rs if r["picked"]]
        safe = sum(1 for r in picked if r["safe"])
        unsafe = len(picked) - safe
        causes = Counter(h for r in picked if not r["safe"] for h in (r["hazards"] or "unknown").split(","))
        out[name] = {
            "frames": n,
            "safe_landing_rate": round(safe / n, 3) if n else None,
            "false_safe_rate": round(unsafe / len(picked), 3) if picked else 0.0,
            "abstain_rate": round(1 - len(picked) / n, 3) if n else None,
            "unsafe_causes": dict(causes.most_common()),
        }
    return out


def to_markdown(summary: dict) -> str:
    lines = ["| strategy | frames | safe-landing rate | false-safe rate | abstain | top unsafe causes |", "|---|---|---|---|---|---|"]
    for k, v in summary.items():
        causes = ", ".join(f"{c} ({n})" for c, n in list(v["unsafe_causes"].items())[:3]) or "-"
        lines.append(f"| {k} | {v['frames']} | {v['safe_landing_rate']:.0%} | {v['false_safe_rate']:.0%} | {v['abstain_rate']:.0%} | {causes} |")
    return "\n".join(lines) + "\n"
