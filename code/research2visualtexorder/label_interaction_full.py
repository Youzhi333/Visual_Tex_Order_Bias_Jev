"""Full-set 2x2 label-permutation decomposition (Qwen3-VL-4B).

    python research2visualtexorder/label_interaction_full.py
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
    for label_path in sorted(RESULTS.glob("label_perm_*_full.json")):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        slug = label["model"].rstrip("/").split("/")[-1]
        dataset = label["dataset"]
        raw_path = RESULTS / f"{slug}_{dataset}_full.json"
        if not raw_path.exists():
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        names = raw["raw"]["metrics"]["positions"]
        n = label["n"]
        id_pre = raw["raw"]["preds"][names[0]][:n]
        id_suf = raw["raw"]["preds"][names[-1]][:n]
        p_pre = label["positions"]["prefix"]["preds_permuted"][:n]
        p_suf = label["positions"]["suffix"]["preds_permuted"][:n]

        def flip(a, b):
            return float(np.mean([x != y for x, y in zip(a, b)]))

        rows.append({
            "model": label["model"], "dataset": dataset, "n": n,
            "label_prefix": flip(p_pre, id_pre),
            "label_suffix": flip(p_suf, id_suf),
            "position_identity": flip(id_suf, id_pre),
            "position_permuted": flip(p_suf, p_pre),
        })
    out = {"model": TAG, "rows": rows}
    (RESULTS / "analysis_label_interaction_full.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for row in rows:
        print(row)
    print(f"wrote {RESULTS / 'analysis_label_interaction_full.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
