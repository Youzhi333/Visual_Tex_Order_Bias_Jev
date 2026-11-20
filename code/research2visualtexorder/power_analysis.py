"""Approximate minimum detectable paired accuracy difference at n=300.

    python research2visualtexorder/power_analysis.py

For paired binary outcomes, n ~ (z_.975 + z_.8)^2 * pi_d / delta^2, where pi_d
is the discordance rate. We report the implied delta for each cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
Z = (1.959964 + 0.841621) ** 2


def main() -> int:
    rows = []
    for path in sorted(RESULTS.glob("*_4pos.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data["raw"]["metrics"]["positions"]
        truth = [it["answer"] for it in data["items"]]
        pre = data["raw"]["preds"][names[0]]
        suf = data["raw"]["preds"][names[-1]]
        cp = np.asarray([a == t for a, t in zip(pre, truth)], dtype=bool)
        cs = np.asarray([a == t for a, t in zip(suf, truth)], dtype=bool)
        disc = float(np.mean(cp != cs))
        mdd = float(np.sqrt(Z * max(disc, 1e-6) / len(truth)))
        rows.append({"model": data["model"], "dataset": data["dataset"],
                     "n": len(truth), "discordance": disc, "mdd": mdd})
    out = {"assumptions": "paired binary, alpha=.05 two-sided, power=.8",
           "rows": rows}
    (RESULTS / "analysis_power.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")
    for label, test in (("Qwen", lambda m: "Qwen" in m),
                        ("Non-Qwen", lambda m: "Qwen" not in m)):
        group = [r for r in rows if test(r["model"])]
        vals = np.asarray([r["mdd"] for r in group])
        print(label, "MDD range", round(vals.min(), 3), round(vals.max(), 3),
              "mean", round(vals.mean(), 3))
    print(f"wrote {RESULTS / 'analysis_power.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
