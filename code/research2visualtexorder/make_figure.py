"""Plot the position sweep from the four-model result JSONs.

Run with the system Python (matplotlib), not the project venv:

    python3 research2visualtexorder/make_figure.py

Writes paper/figures/fig_curve.pdf and .png. One series per panel; the bars are
95% bootstrap CIs over the 24 model-dataset cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
FIG_DIR = ROOT / "paper" / "figures"
MODELS = (
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
)
DATASETS = ("aokvqa", "eurosat", "scienceqa", "cvbench", "pets", "mmmu")
KEYS = ("prefix", "middle", "question", "suffix")
LABELS = ("prefix", "pre-Q", "pre-O", "suffix")
INK = "#333333"
MUTED = "#666666"
GRID = "#d9d9d9"
BLUE = "#0072B2"      # Okabe-Ito blue
ORANGE = "#D55E00"    # Okabe-Ito vermillion


def bootstrap_ci(values, n_boot=4000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def collect():
    flips, accs = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for model in MODELS:
        for dataset in DATASETS:
            path = RESULTS / f"{model}_{dataset}_4pos.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))["raw"]["metrics"]
            index = {p: i for i, p in enumerate(metrics["positions"])}
            for key in KEYS:
                flips[key].append(100.0 * metrics["flip"][index[key]])
                accs[key].append(100.0 * metrics["per_position"][key]["accuracy"])
    return flips, accs


def panel(ax, values, color, ylabel, title, fmt):
    means = [np.mean(values[k]) for k in KEYS]
    cis = [bootstrap_ci(values[k]) for k in KEYS]
    x = np.arange(len(KEYS))
    lower = [m - c[0] for m, c in zip(means, cis)]
    upper = [c[1] - m for m, c in zip(means, cis)]
    ax.errorbar(x, means, yerr=[lower, upper], color=color, linewidth=1.5,
                marker="o", markersize=4.5, capsize=2.5, elinewidth=1.0)
    for xi, m in zip(x, means):
        ax.annotate(fmt.format(m), (xi, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=6.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    ax.set_title(title, fontsize=8.5, fontweight="bold", color=INK, loc="left")
    ax.tick_params(axis="both", labelsize=7, colors=MUTED, length=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    flips, accs = collect()
    n_cells = len(flips["prefix"])
    acc_change = {
        key: [a - b for a, b in zip(accs[key], accs["prefix"])]
        for key in KEYS
    }
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.15), constrained_layout=True)
    panel(axes[0], flips, BLUE, "Modality flip (%)", "(a) decision flips", "{:.1f}%")
    panel(axes[1], acc_change, ORANGE, "Accuracy change (pp)", "(b) accuracy vs prefix", "{:+.1f}")
    axes[0].set_ylim(0, max(np.mean(flips[k]) for k in KEYS) + 8)
    axes[1].axhline(0.0, color=GRID, linewidth=0.8)
    axes[1].set_ylim(min(np.mean(acc_change[k]) for k in KEYS) - 4.0, 3.0)
    fig.text(0.5, -0.04, f"Mean over {n_cells} model-dataset cells; bars are 95% bootstrap CIs",
             ha="center", fontsize=6.0, color=MUTED)
    fig.savefig(FIG_DIR / "fig_curve.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_curve.png", dpi=300, bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'fig_curve.pdf'} ({n_cells} cells)")
    for key in KEYS:
        print(f"  {key:8s} flip={np.mean(flips[key]):5.1f}%  "
              f"acc_delta={np.mean(acc_change[key]):+5.1f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
