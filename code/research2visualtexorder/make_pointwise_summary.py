"""Family means for the pointwise control.

    python research2visualtexorder/make_pointwise_summary.py

Reads the *_pointwise.json files (A-OKVQA and EuroSAT, all seven checkpoints)
and prints the suffix accuracy, suffix flip and suffix ECE used by the
"Pointwise control" rows of tab:fuse.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
GROUPS = {
    "Qwen": ("Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct",
             "Qwen3-VL-8B-Instruct", "Qwen2.5-VL-7B-Instruct"),
    "Non-Qwen": ("Idefics3-8B-Llama3", "llava-onevision-qwen2-7b-ov-hf",
                 "llava-v1.6-mistral-7b-hf"),
}
DATASETS = ("aokvqa", "eurosat")


def cell(slug: str, dataset: str) -> tuple[float, float, float] | None:
    path = RESULTS / f"{slug}_{dataset}_pointwise.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["pointwise"]["metrics"]
    suffix = data["positions"][-1]
    return (m["per_position"][suffix]["accuracy"], m["flip"][-1],
            m["per_position"][suffix]["ece"])


def main() -> int:
    for group, models in GROUPS.items():
        rows = [cell(m, d) for m in models for d in DATASETS]
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f"% {group}: no pointwise files")
            continue
        n = len(rows)
        acc = sum(r[0] for r in rows) / n
        flip = sum(r[1] for r in rows) / n
        ece = sum(r[2] for r in rows) / n
        print(f"% {group}: {n} cells (A-OKVQA/EuroSAT)")
        print(f"Pointwise control & {acc:.3f} & {flip:.3f} & {ece:.3f} & $K$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
