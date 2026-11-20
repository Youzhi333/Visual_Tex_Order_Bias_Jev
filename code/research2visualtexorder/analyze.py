"""Summarize modality-order result JSONs.

    python research2visualtexorder/analyze.py research2visualtexorder/results/2026-11-12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(values, digits=4):
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--sections", default="raw,grey,pointwise")
    args = parser.parse_args()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    files = []
    for path in args.paths:
        p = Path(path)
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "dataset" not in data or "model" not in data:
            continue
        print(f"\n{path}")
        print(f"  {data['model']} / {data['dataset']}  n={data['n']}  "
              f"k={data.get('k_counts')}  positions={data.get('positions')}")
        for section in sections:
            if section not in data:
                continue
            metrics = data[section]["metrics"]
            print(f"  {section:9s} flip={fmt(metrics['flip'])} tv={fmt(metrics['tv'])} "
                  f"instability={metrics['instability']:.4f}")
            acc = [metrics["per_position"][p]["accuracy"] for p in metrics["positions"]]
            print(f"            acc={fmt(acc)}")
            best = None
            for key, value in metrics.get("dof", {}).items():
                if key.startswith("prefix_"):
                    continue
                if best is None or value["flip_vs_prefix"] < best[1]["flip_vs_prefix"]:
                    best = (key, value)
            if best:
                print(f"            best-dof={best[0]} acc={best[1]['accuracy']:.4f} "
                      f"flip={best[1]['flip_vs_prefix']:.4f} tv={best[1]['tv_vs_prefix']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
