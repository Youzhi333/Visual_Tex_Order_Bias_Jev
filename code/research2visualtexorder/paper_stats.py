"""Paired statistics and offline baselines for the modality-order paper.

    python research2visualtexorder/paper_stats.py

Reads the committed ``_4pos.json`` files only. Reports paired bootstrap
intervals, exact McNemar tests, and the fusion baselines computed from cached
per-item distributions. Writes analysis_paper_stats.json and a LaTeX table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
POSITIONS = ("prefix", "middle", "question", "suffix")


def boot_ci(values, n_boot=4000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ece_of(probs, correct, bins=10):
    conf = np.asarray(probs.max(axis=1), dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def mcnemar(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def logmean(probs):
    """Geometric mean over axis 0, normalized over the option axis."""
    z = np.log(np.clip(probs, 1e-12, None)).mean(axis=0)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def pred_of(probs, items):
    idx = probs.argmax(axis=1)
    return [items[i]["options"][int(j)] for i, j in enumerate(idx)]


def acc_of(probs, truth, items):
    pred = pred_of(probs, items)
    return np.asarray([p == t for p, t in zip(pred, truth)], dtype=float)


def read_cell(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["raw"]["metrics"]["positions"]
    if len(names) != len(POSITIONS):
        return {}
    items = data["items"]
    truth = [it["answer"] for it in items]
    p = np.stack([np.asarray(data["raw"]["probs"][n], dtype=float) for n in names], axis=1)
    preds = [data["raw"]["preds"][n] for n in names]
    correct = {
        POSITIONS[i]: np.asarray([preds[i][j] == truth[j] for j in range(len(truth))],
                                 dtype=float)
        for i in range(len(POSITIONS))
    }
    flips = {
        POSITIONS[i]: np.asarray([preds[i][j] != preds[0][j] for j in range(len(truth))],
                                 dtype=float)
        for i in range(len(POSITIONS))
    }
    delta = correct["suffix"] - correct["prefix"]
    z75 = (0.25 * np.log(np.clip(p[:, 3], 1e-12, None)) +
           0.75 * np.log(np.clip(p[:, 0], 1e-12, None)))
    z75 = z75 - z75.max(axis=1, keepdims=True)
    fuse75 = np.exp(z75)
    fuse75 = fuse75 / fuse75.sum(axis=1, keepdims=True)
    p50 = np.exp(0.5 * (np.log(np.clip(p[:, 3], 1e-12, None)) +
                        np.log(np.clip(p[:, 0], 1e-12, None))))
    p50 = p50 / p50.sum(axis=1, keepdims=True)
    dof_correct = acc_of(fuse75, truth, items)
    baseline_probs = {
        "all_mean": p.mean(axis=1),
        "all_logmean": logmean(p.transpose(1, 0, 2)),
        "prefix_suffix_mean": (p[:, 0] + p[:, 3]) / 2.0,
        "dof_beta0.5": p50,
        "dof_beta0.75": fuse75,
    }
    baselines = {k: acc_of(v, truth, items) for k, v in baseline_probs.items()}
    baseline_ece = {k: ece_of(v, baselines[k]) for k, v in baseline_probs.items()}
    prefix_pred = pred_of(p[:, 0], items)
    baseline_flips = {
        k: float(np.mean([a != b for a, b in zip(pred_of(v, items), prefix_pred)]))
        for k, v in baseline_probs.items()
    }
    return {
        "source": path.name,
        "model": data["model"],
        "dataset": data["dataset"],
        "n": int(len(truth)),
        "prefix_accuracy": float(correct["prefix"].mean()),
        "suffix_accuracy": float(correct["suffix"].mean()),
        "accuracy_delta": float(delta.mean()),
        "accuracy_delta_ci": boot_ci(delta),
        "mcnemar_prefix_suffix": mcnemar(correct["prefix"], correct["suffix"]),
        "suffix_flip": float(flips["suffix"].mean()),
        "suffix_flip_ci": boot_ci(flips["suffix"]),
        "flip_diff_suffix_question": float(flips["suffix"].mean() - flips["question"].mean()),
        "flip_diff_ci": boot_ci(flips["suffix"] - flips["question"]),
        "baselines": {k: float(v.mean()) for k, v in baselines.items()},
        "baseline_flips": baseline_flips,
        "baseline_ece": baseline_ece,
        "dof_mcnemar": mcnemar(correct["prefix"], dof_correct),
        "ece": {POSITIONS[i]: data["raw"]["metrics"]["per_position"][names[i]]["ece"]
                for i in range(len(POSITIONS))},
        "instability": data["raw"]["metrics"]["instability"],
        "label_entropy": {
            POSITIONS[i]: data["raw"]["metrics"]["per_position"][names[i]]["label_marginal_entropy"]
            for i in range(len(POSITIONS))
        },
    }


def holm(pvalues):
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=str(RESULTS / "analysis_paper_stats.json"))
    parser.add_argument("--tex", default=str(ROOT / "paper/tables/stats.tex"))
    args = parser.parse_args()
    files = sorted(Path(args.results).glob("*_4pos.json"))
    rows = [r for r in (read_cell(p) for p in files) if r]
    if not rows:
        raise SystemExit("no _4pos files")
    for family, matches in (
        ("all", lambda m: True),
        ("qwen", lambda m: "Qwen" in m),
        ("nonqwen", lambda m: "Qwen" not in m),
    ):
        group = [r for r in rows if matches(r["model"])]
        for r in group:
            r[f"{family}_mcnemar_holm"] = 1.0
        adjusted = holm([r["mcnemar_prefix_suffix"] for r in group])
        for r, p in zip(group, adjusted):
            r[f"{family}_mcnemar_holm"] = float(p)
    payload = {"description": "paired item-level statistics; no option rotation",
               "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def summary(label, group):
        if not group:
            return
        delta = np.asarray([r["accuracy_delta"] for r in group])
        lo, hi = boot_ci(delta)
        flip = np.asarray([r["suffix_flip"] for r in group])
        sig = sum(r["mcnemar_prefix_suffix"] < 0.05 for r in group)
        holm_key = {"all": "all_mcnemar_holm",
                    "qwen": "qwen_mcnemar_holm",
                    "nonqwen": "nonqwen_mcnemar_holm"}.get(label)
        sig_holm = sum((r[holm_key] if holm_key else r["mcnemar_prefix_suffix"]) < 0.05
                       for r in group)
        print(f"{label:8s} cells={len(group):2d}  mean acc delta={delta.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]  "
              f"mean suffix flip={flip.mean():.3f}  McNemar p<.05 {sig}/{len(group)} "
              f"(Holm {sig_holm}/{len(group)})")
        base = {k: np.mean([r["baselines"][k] for r in group])
                for k in group[0]["baselines"]}
        print("          baselines " + " ".join(f"{k}={v:.3f}" for k, v in base.items()))

    summary("qwen", [r for r in rows if "Qwen" in r["model"]])
    summary("nonqwen", [r for r in rows if "Qwen" not in r["model"]])
    summary("all", rows)
    lines = [
        "% generated by paper_stats.py from the *_4pos.json files",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Family & cells & mean $\\Delta$acc [95\\% CI] & mean suffix flip & "
        "McNemar $p<.05$ & Holm \\\\",
        "\\midrule",
    ]
    for label, group, key in (
        ("All", rows, "all_mcnemar_holm"),
        ("Qwen", [r for r in rows if "Qwen" in r["model"]], "qwen_mcnemar_holm"),
        ("Non-Qwen", [r for r in rows if "Qwen" not in r["model"]],
         "nonqwen_mcnemar_holm"),
    ):
        deltas = np.asarray([r["accuracy_delta"] for r in group])
        lo, hi = boot_ci(deltas)
        flip = np.mean([r["suffix_flip"] for r in group])
        sig = sum(r["mcnemar_prefix_suffix"] < 0.05 for r in group)
        sig_holm = sum(r[key] < 0.05 for r in group)
        lines.append(f"{label} & {len(group)} & {deltas.mean():+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}] & {flip:.3f} & "
                     f"{sig}/{len(group)} & {sig_holm}/{len(group)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}\nwrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
