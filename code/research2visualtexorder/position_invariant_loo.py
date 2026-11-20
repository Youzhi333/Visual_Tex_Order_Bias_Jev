"""Leave-one-out placement-invariant reference.

    python research2visualtexorder/position_invariant_loo.py

For each position r, average the other three image placements and compare r to
that average. This avoids including r in its own reference. No option rotation.
Writes analysis_position_invariant_loo.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
POSITIONS = ("prefix", "middle", "question", "suffix")


def main() -> int:
    rows = []
    for path in sorted(RESULTS.glob("*_4pos.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data["raw"]["metrics"]["positions"]
        if len(names) != 4:
            continue
        items = data["items"]
        probs = np.stack([np.asarray(data["raw"]["probs"][n], dtype=float)
                          for n in names], axis=1)
        per = {}
        for i, name in enumerate(names):
            others = [j for j in range(4) if j != i]
            bar = probs[:, others, :].mean(axis=1)
            inv = [items[k]["options"][int(j)] for k, j in enumerate(bar.argmax(axis=1))]
            pred = data["raw"]["preds"][name]
            per[POSITIONS[i]] = {
                "flip": float(np.mean([a != b for a, b in zip(pred, inv)])),
                "tv": float(0.5 * np.abs(probs[:, i] - bar).sum(axis=1).mean()),
            }
        rows.append({"source": path.name, "model": data["model"],
                     "dataset": data["dataset"], "per_position": per})
    out = {"description": "leave-one-out image-placement average; option listing fixed",
           "rows": rows}
    (RESULTS / "analysis_position_invariant_loo.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for label, test in (("Qwen", lambda m: "Qwen" in m),
                        ("Non-Qwen", lambda m: "Qwen" not in m)):
        group = [r for r in rows if test(r["model"])]
        print(label, len(group),
              {p: round(float(np.mean([r["per_position"][p]["flip"] for r in group])), 3)
               for p in POSITIONS})
    print(f"wrote {RESULTS / 'analysis_position_invariant_loo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
