"""Placement-invariant reference over the five token-level positions.

    python research2visualtexorder/token_invariant.py

Uses the existing token-sweep JSONs (prefix, r=0.25/0.5/0.75, suffix); no
option rotation and no new model runs. Writes analysis_token_invariant.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"


def main() -> int:
    best = {}
    for path in sorted(RESULTS.glob("*_token*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "raw" not in data or "model" not in data:
            continue
        key = (data["model"], data["dataset"])
        if key in best and "_token6" not in path.name:
            continue
        best[key] = (data, path)
    rows = []
    for (_, _), (data, path) in sorted(best.items()):
        names = data["raw"]["metrics"]["positions"]
        items = data["items"]
        truth = [it["answer"] for it in items]
        probs = np.stack([np.asarray(data["raw"]["probs"][n], dtype=float)
                          for n in names], axis=1)
        bar = probs.mean(axis=1)
        inv = [items[i]["options"][int(j)] for i, j in enumerate(bar.argmax(axis=1))]
        suffix = data["raw"]["preds"][names[-1]]
        prefix = data["raw"]["preds"][names[0]]
        rows.append({
            "source": path.name, "model": data["model"], "dataset": data["dataset"],
            "n": data["n"], "n_positions": len(names),
            "invariant_accuracy": float(np.mean([a == t for a, t in zip(inv, truth)])),
            "suffix_flip_vs_invariant": float(np.mean(
                [a != b for a, b in zip(suffix, inv)])),
            "suffix_flip_vs_prefix": float(np.mean(
                [a != b for a, b in zip(suffix, prefix)])),
        })
    out = {"description": "average over five token-level image placements; "
                          "option listing fixed", "rows": rows}
    dest = RESULTS / "analysis_token_invariant.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for label, test in (("Qwen", lambda m: "Qwen" in m),
                        ("Non-Qwen", lambda m: "Qwen" not in m)):
        group = [r for r in rows if test(r["model"])]
        print(label, len(group),
              "suffix vs invariant", round(np.mean(
                  [r["suffix_flip_vs_invariant"] for r in group]), 3),
              "vs prefix", round(np.mean(
                  [r["suffix_flip_vs_prefix"] for r in group]), 3))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
