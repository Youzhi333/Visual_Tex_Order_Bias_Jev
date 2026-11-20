"""2x2 label-permutation decomposition for Qwen3-VL-4B.

    python research2visualtexorder/label_interaction.py

Position effect = suffix vs prefix flip under identity or permuted letters.
Label effect = identity vs permuted-letter flip at the same position. Option
order is fixed; this is not an option-order experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
TAG = "Qwen3-VL-4B-Instruct"


def main() -> int:
    rows = []
    for dataset in ("aokvqa", "eurosat"):
        raw = json.loads((RESULTS / f"{TAG}_{dataset}_4pos.json").read_text())
        perm = json.loads((RESULTS / f"label_perm_{TAG}_{dataset}.json").read_text())
        truth = [it["answer"] for it in raw["items"]]
        id_pre = raw["raw"]["preds"]["prefix"]
        id_suf = raw["raw"]["preds"]["suffix"]
        p_pre = perm["positions"]["prefix"]["preds_permuted"]
        p_suf = perm["positions"]["suffix"]["preds_permuted"]

        def flip(a, b):
            return float(np.mean([x != y for x, y in zip(a, b)]))

        def acc(pred):
            return float(np.mean([p == t for p, t in zip(pred, truth)]))

        rows.append({
            "dataset": dataset,
            "position_identity": flip(id_suf, id_pre),
            "position_permuted": flip(p_suf, p_pre),
            "label_prefix": flip(p_pre, id_pre),
            "label_suffix": flip(p_suf, id_suf),
            "accuracy": {"id_prefix": acc(id_pre), "id_suffix": acc(id_suf),
                         "perm_prefix": acc(p_pre), "perm_suffix": acc(p_suf)},
        })
    out = {"model": TAG, "control": "letter permutation, option order fixed",
           "rows": rows}
    dest = RESULTS / "analysis_label_interaction.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for row in rows:
        print(row["dataset"],
              "position", round(row["position_identity"], 3),
              "->", round(row["position_permuted"], 3),
              "label", round(row["label_prefix"], 3),
              round(row["label_suffix"], 3))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
