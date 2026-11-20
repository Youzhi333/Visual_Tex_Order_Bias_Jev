"""Two-level bootstrap for the mean accuracy delta across cells and items.

    python research2visualtexorder/two_level_stats.py

Cells are resampled, then items within each selected cell. This is a
conservative check on the cell-level interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"


def main() -> int:
    cells = []
    for path in sorted(RESULTS.glob("*_4pos.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data["raw"]["metrics"]["positions"]
        truth = [it["answer"] for it in data["items"]]
        pre = data["raw"]["preds"][names[0]]
        suf = data["raw"]["preds"][names[-1]]
        delta = np.asarray([(s == t) - (p == t) for p, s, t in zip(pre, suf, truth)],
                           dtype=float)
        cells.append((data["model"], delta))
    rows = []
    for label, test in (("Qwen", lambda m: "Qwen" in m),
                        ("Non-Qwen", lambda m: "Qwen" not in m),
                        ("All", lambda m: True)):
        group = [d for m, d in cells if test(m)]
        if not group:
            continue
        rng = np.random.default_rng(0)
        means = []
        for _ in range(2000):
            chosen = rng.integers(0, len(group), len(group))
            vals = []
            for idx in chosen:
                arr = group[idx]
                vals.append(arr[rng.integers(0, len(arr), len(arr))].mean())
            means.append(np.mean(vals))
        lo, hi = np.percentile(means, [2.5, 97.5])
        point = float(np.mean([d.mean() for d in group]))
        rows.append({"family": label, "cells": len(group), "delta": point,
                     "ci": [float(lo), float(hi)]})
        print(label, len(group), round(point, 3), [round(lo, 3), round(hi, 3)])
    (RESULTS / "analysis_two_level_stats.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'analysis_two_level_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
