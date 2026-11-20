"""Plot the position sweep for every checkpoint, not just the Qwen grid.

    python3 research2visualtexorder/make_family_figure.py

Two panels: modality flip relative to prefix and accuracy change relative to
prefix, averaged over the datasets each model has. Writes
paper/figures/fig_family_curve.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
FIG_DIR = ROOT / "paper" / "figures"
DATASETS = ("aokvqa", "scienceqa", "cvbench", "eurosat", "pets", "mmmu")
KEYS = ("prefix", "middle", "question", "suffix")
LABELS = ("prefix", "pre-Q", "pre-O", "suffix")
INK = "#333333"
MUTED = "#666666"
GRID = "#d9d9d9"
# model slug, legend label, colour, line style
MODELS = (
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL-2B", "#0072B2", "-"),
    ("Qwen3-VL-4B-Instruct", "Qwen3-VL-4B", "#56B4E9", "-"),
    ("Qwen3-VL-8B-Instruct", "Qwen3-VL-8B", "#009E73", "-"),
    ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B", "#CC79A7", "-"),
    ("Idefics3-8B-Llama3", "Idefics3-8B", "#E69F00", "--"),
    ("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OneVision-7B", "#D55E00", "--"),
    ("llava-v1.6-mistral-7b-hf", "LLaVA-NeXT-7B", "#000000", ":"),
)


def collect(slug: str):
    flips, accs = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for dataset in DATASETS:
        path = RESULTS / f"{slug}_{dataset}_4pos.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["raw"]["metrics"]
        names = metrics["positions"]
        for key, index in zip(KEYS, range(len(KEYS))):
            flips[key].append(metrics["flip"][index])
            accs[key].append(metrics["per_position"][names[index]]["accuracy"])
    return flips, accs


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    x = np.arange(len(KEYS))
    for slug, label, color, style in MODELS:
        flips, accs = collect(slug)
        if not flips["prefix"]:
            continue
        flip = [100.0 * float(np.mean(flips[k])) for k in KEYS]
        base = flip[0]
        flip = [v - base for v in flip]
        acc = [100.0 * float(np.mean(accs[k])) for k in KEYS]
        acc = [v - acc[0] for v in acc]
        axes[0].plot(x, flip, color=color, linestyle=style, linewidth=1.4,
                     marker="o", markersize=3.2, label=label)
        axes[1].plot(x, acc, color=color, linestyle=style, linewidth=1.4,
                     marker="o", markersize=3.2)
    for ax, ylabel, title in zip(
        axes,
        ("modality flip vs prefix (\\%)", "accuracy change vs prefix (pp)"),
        ("(a) Decision flip", "(b) Accuracy"),
    ):
        ax.axhline(0.0, color="#888888", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=6.8, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=7.4, color=INK)
        ax.set_title(title, fontsize=8.2, fontweight="bold", color=INK, loc="left")
        ax.tick_params(axis="both", labelsize=6.8, colors=MUTED, length=2)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=6.4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_family_curve.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'fig_family_curve.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
