"""Heatmaps of per-head, per-layer image-attention change (suffix - prefix).

    python3 research2visualtexorder/make_head_figure.py

Reads the ``attention_*_all.json`` files directly. Each panel is one model and
dataset: rows are layers, columns are heads, color is the change in image
attention at the answer position. Writes paper/figures/fig_attention_heads.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
FIG_DIR = ROOT / "paper" / "figures"
INK = "#333333"
MUTED = "#666666"

ORDER = [
    "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen2.5-VL-7B",
    "Idefics3-8B", "LLaVA-OneVision-7B", "LLaVA-NeXT-7B",
]
DATASETS = {"aokvqa": "A-OKVQA", "eurosat": "EuroSAT"}
TITLE = {
    "qwen3vl2b": "Qwen3-VL-2B",
    "qwen3vl4b": "Qwen3-VL-4B",
    "qwen3vl8b": "Qwen3-VL-8B",
    "qwen25vl7b": "Qwen2.5-VL-7B",
    "idefics3-8b-llama3": "Idefics3-8B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-NeXT-7B",
}


def label_for(path: Path, data: dict) -> str:
    name = path.stem.lower()
    for key, label in TITLE.items():
        if key in name:
            return label
    return data["model"].rstrip("/").split("/")[-1]


def matrix(data: dict):
    prefix = data["positions"]["prefix"]["layers"]
    suffix = data["positions"]["suffix"]["layers"]
    layers = sorted(set(prefix) & set(suffix), key=int)
    rows = []
    for layer in layers:
        p = np.asarray(prefix[layer]["head_means"], dtype=float)
        s = np.asarray(suffix[layer]["head_means"], dtype=float)
        rows.append(s - p)
    return np.asarray(rows)


def main() -> int:
    cells = {}
    for path in sorted(RESULTS.glob("attention_*_all.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["dataset"] not in DATASETS:
            continue
        label = label_for(path, data)
        cells[(label, data["dataset"])] = matrix(data)
    models = [m for m in ORDER if any(k[0] == m for k in cells)]
    models += [m for m in {k[0] for k in cells} if m not in models]
    panels = [(m, d) for d in DATASETS for m in models if (m, d) in cells]
    if not panels:
        raise SystemExit("no attention_*_all.json files")
    vmax = max(float(v.max()) for v in cells.values())
    vmin = min(float(v.min()) for v in cells.values())
    norm = TwoSlopeNorm(vmin=min(vmin, -0.05), vcenter=0.0, vmax=max(vmax, 0.05))
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.55 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    image = None
    for i, (model, dataset) in enumerate(panels):
        ax = axes.flat[i]
        ax.set_visible(True)
        array = cells[(model, dataset)]
        image = ax.imshow(array, aspect="auto", cmap="RdBu_r", norm=norm,
                          interpolation="nearest")
        ax.set_title(f"{model} / {DATASETS[dataset]}", fontsize=6.6,
                     color=INK, loc="left")
        ax.set_yticks(np.arange(0, array.shape[0], max(1, array.shape[0] // 4)))
        ax.set_xticks(np.arange(0, array.shape[1], max(1, array.shape[1] // 4)))
        ax.tick_params(axis="both", labelsize=5.2, colors=MUTED, length=1.5)
        ax.set_ylabel("layer", fontsize=5.6, color=MUTED)
        ax.set_xlabel("head", fontsize=5.6, color=MUTED)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if image is not None:
        cax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
        bar = fig.colorbar(image, cax=cax)
        bar.set_label("suffix $-$ prefix image attention", fontsize=6.2, color=INK)
        bar.ax.tick_params(labelsize=5.4, colors=MUTED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_attention_heads.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'fig_attention_heads.pdf'} ({len(panels)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
