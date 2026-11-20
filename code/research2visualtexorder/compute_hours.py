"""Sum the recorded GPU seconds in the result JSONs.

    python research2visualtexorder/compute_hours.py

Readout files store per-position seconds under each section; attention files
store seconds per placement. Reports by section and a grand total. Model
loading is not included (the JSONs do not record it).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def seconds_of(data: dict) -> tuple[float, str]:
    positions = data.get("positions")
    if isinstance(positions, dict) and positions:
        first = next(iter(positions.values()))
        if isinstance(first, dict) and "layers" in first:
            total = sum(float(p.get("seconds", 0.0))
                        for p in positions.values() if isinstance(p, dict))
            return total, "attention"
    total, kind = 0.0, "readout"
    for section in ("raw", "grey", "pointwise"):
        block = data.get(section)
        if isinstance(block, dict) and isinstance(block.get("seconds"), dict):
            total += sum(float(v) for v in block["seconds"].values())
    return total, kind


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?",
                        default=str(ROOT / "results" / "2026-11-12"))
    args = parser.parse_args()
    root = Path(args.root)
    totals = defaultdict(float)
    counts = defaultdict(int)
    files = 0
    for path in sorted(root.glob("*.json")):
        if path.name.startswith(("analysis_", "attention_heads")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "model" not in data:
            continue
        seconds, kind = seconds_of(data)
        if seconds <= 0:
            continue
        totals[kind] += seconds
        counts[kind] += 1
        files += 1
    grand = sum(totals.values())
    for kind, value in sorted(totals.items()):
        print(f"{kind:10s} {counts[kind]:3d} files  {value / 60:8.1f} GPU-min "
              f"({value / 3600:5.2f} h)")
    print(f"{'total':10s} {files:3d} files  {grand / 60:8.1f} GPU-min "
          f"({grand / 3600:5.2f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
