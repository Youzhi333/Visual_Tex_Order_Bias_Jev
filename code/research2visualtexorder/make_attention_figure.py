"""Plot the every-layer attention gap from analysis_attention_heads.json.

    python3 research2visualtexorder/make_attention_figure.py

One small panel per model; x is relative depth, y is suffix minus prefix image
attention at the answer position in percentage points. Writes
paper/figures/fig_attention_layers.pdf and .png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
FIG_DIR = ROOT / "paper" / "figures"
SUMMARY = RESULTS / "analysis_attention_heads.json"
INK = "#333333"
MUTED = "#666666"
GRID = "#d9d9d9"
BLUE = "#0072B2"
ORANGE = "#D55E00"
ZERO = "#888888"

ORDER = [
    "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen2.5-VL-7B",
    "Idefics3-8B", "LLaVA-OneVision-7B", "LLaVA-NeXT-7B",
]
DATASETS = {"aokvqa": ("A-OKVQA", BLUE), "eurosat": ("EuroSAT", ORANGE)}
TITLE = {
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "HuggingFaceM4/Idefics3-8B-Llama3": "Idefics3-8B",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-NeXT-7B",
}


def model_name(summary: dict) -> str:
    for key, label in TITLE.items():
        if key.rstrip("/").split("/")[-1].lower() in summary["model"].lower():
            return label
    return summary["model"].rstrip("/").split("/")[-1]


def main() -> int:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cells = {}
    for summary in data["summaries"]:
        cells.setdefault(model_name(summary), {})[summary["dataset"]] = summary
    models = [m for m in ORDER if m in cells] + [m for m in cells if m not in ORDER]
    ncols = 4
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.9 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, model in enumerate(models):
        ax = axes.flat[i]
        ax.set_visible(True)
        ax.axhline(0.0, color=ZERO, linewidth=0.7)
        for dataset, (label, color) in DATASETS.items():
            if dataset not in cells[model]:
                continue
            summary = cells[model][dataset]
            layers = summary["per_layer"]
            depth = np.array([g["layer"] for g in layers], dtype=float)
            depth = depth / max(depth.max(), 1.0)
            gap = np.array([100.0 * g["gap"] for g in layers])
            ax.plot(depth, gap, color=color, linewidth=1.3, label=label)
        ax.set_title(model, fontsize=7.6, fontweight="bold", color=INK, loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=6, color=MUTED)
        ax.tick_params(axis="y", labelsize=6, colors=MUTED, length=2)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if i % ncols == 0:
            ax.set_ylabel("suffix $-$ prefix (pp)", fontsize=6.8, color=INK)
        if i // ncols == nrows - 1 or i + ncols >= len(models):
            ax.set_xlabel("relative depth", fontsize=6.8, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.99, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_attention_layers.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'fig_attention_layers.pdf'} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
