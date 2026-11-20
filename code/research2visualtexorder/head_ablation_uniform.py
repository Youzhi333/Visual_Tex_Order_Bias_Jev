"""Held-out, uniform head-set ablation.

    python research2visualtexorder/head_ablation_uniform.py

Selection uses the first 50 probe items: top-10 heads by suffix-minus-prefix
image attention. Ablation is evaluated on the held-out 50 items. The protocol is
identical across models; only the selected heads differ.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402
from head_ablation import run  # noqa: E402

RESULTS = _ROOT / "research2visualtexorder/results/2026-11-12"
MODELS = (
    ("Qwen/Qwen3-VL-2B-Instruct", "qwen3vl2b"),
    ("Qwen/Qwen3-VL-4B-Instruct", "qwen3vl4b"),
    ("Qwen/Qwen3-VL-8B-Instruct", "qwen3vl8b"),
    ("HuggingFaceM4/Idefics3-8B-Llama3", "idefics3-8b-llama3"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf",
     "llava-onevision-qwen2-7b-ov-hf"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25vl7b"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "llava-v1.6-mistral-7b-hf"),
)
TOP_K = 10


def select_heads(path: Path):
    data = json.loads(path.read_text())
    prefix = data["positions"]["prefix"]["layers"]
    suffix = data["positions"]["suffix"]["layers"]
    layers = sorted(set(prefix) & set(suffix), key=int)
    deltas = []
    for layer in layers:
        p = np.asarray(prefix[layer]["head_values"], dtype=float)[:50]
        s = np.asarray(suffix[layer]["head_values"], dtype=float)[:50]
        delta = s.mean(axis=0) - p.mean(axis=0)
        for head, value in enumerate(delta):
            deltas.append((float(value), int(layer), int(head)))
    deltas.sort(reverse=True)
    top = [(layer, head) for _, layer, head in deltas[:TOP_K]]
    pool = [(layer, head) for _, layer, head in deltas[TOP_K:]]
    rng = random.Random(0)
    rng.shuffle(pool)
    return top, pool[:TOP_K]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa",
                        choices=["aokvqa", "scienceqa", "cvbench", "eurosat",
                                 "pets", "mmmu"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--models", default="")
    parser.add_argument("--out", default=str(RESULTS / "analysis_head_uniform.json"))
    args = parser.parse_args()
    image_dir = args.image_dir or str(
        _ROOT / ("research/visual_bias/images" if args.dataset == "aokvqa"
                 else "research/visual_bias/stage2_images"))
    items = load_items(args.dataset, limit=args.limit, seed=0, image_dir=image_dir)
    evaluate = items[50:100]
    truth = [it["answer"] for it in evaluate]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for model, tag in MODELS:
        if wanted and model not in wanted:
            continue
        top, rnd = select_heads(RESULTS / f"attention_{tag}_{args.dataset}_all.json")
        backend = HFVLMBackend(model, device="cuda", dtype="bfloat16", batch_size=1,
                               max_pixels=448 * 448)
        row = {"model": model, "dataset": args.dataset, "n": len(evaluate),
               "top_heads": [list(h) for h in top],
               "random_heads": [list(h) for h in rnd], "conditions": {}}
        for placement in ("prefix", "suffix"):
            base, _ = run(backend, evaluate, placement)
            tops, _ = run(backend, evaluate, placement, top)
            rands, _ = run(backend, evaluate, placement, rnd)
            row["conditions"][placement] = {}
            for label, preds in (("baseline", base), ("top", tops), ("random", rands)):
                acc = float(np.mean([p == t for p, t in zip(preds, truth)]))
                flip = float(np.mean([p != b for p, b in zip(preds, base)]))
                row["conditions"][placement][label] = {"accuracy": acc, "flip": flip}
            print(model.split("/")[-1], placement,
                  "top", row["conditions"][placement]["top"],
                  "random", row["conditions"][placement]["random"], flush=True)
        rows.append(row)
        del backend
        import torch

        torch.cuda.empty_cache()
    dest = Path(args.out) if args.out else (
        RESULTS / f"analysis_head_uniform_{args.dataset}.json")
    dest.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
