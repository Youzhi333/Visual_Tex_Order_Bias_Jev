"""Causal head-set ablation for the suffix effect.

    python research2visualtexorder/head_ablation.py

Zeros the attention heads above 50% suffix image attention (or a matched random
set) at the output-projection input and measures the prefix/suffix readout.
Works for Qwen, Idefics3 and LLaVA via a generic layer lookup.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Decider, Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TOP = (0, 14)      # layer, head from attention_qwen3vl4b_aokvqa_all.json
RANDOM = (18, 5)


def run(backend, items, placement, heads=()):
    backend.image_placement = placement
    decider = Decider(backend, level="raw", prior="none", shared_prefix=False)
    hooks = []
    by_layer = {}
    for layer, index in heads:
        by_layer.setdefault(layer, []).append(index)
    for layer, indices in by_layer.items():
        attn = None
        pattern = re.compile(rf"(?:^|\.)layers\.{layer}\.self_attn$")
        for name, module in backend.model.named_modules():
            if pattern.search(name) and "vision" not in name.lower():
                attn = module
                break
        if attn is None:
            raise RuntimeError(f"attention layer {layer} not found")
        dim = attn.head_dim

        def hook_fn(module, args, indices=tuple(indices), dim=dim):
            x = args[0]
            for index in indices:
                x[..., index * dim:(index + 1) * dim] = 0
            return (x,) + args[1:]

        hooks.append(attn.o_proj.register_forward_pre_hook(hook_fn))
    preds, confs = [], []
    try:
        for item in items:
            q = Question.choice(item["question"], item["options"], name="answer")
            state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
            d = decider.decide(state, [q])["answer"]
            preds.append(d.argmax)
            confs.append(float(d.confidence))
    finally:
        for hook in hooks:
            hook.remove()
    return preds, np.asarray(confs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default="aokvqa", choices=["aokvqa", "eurosat"])
    parser.add_argument("--attention-json", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    slug = args.model.rstrip("/").split("/")[-1]
    image_dir = args.image_dir or str(
        _ROOT / ("research/visual_bias/images" if args.dataset == "aokvqa"
                 else "research/visual_bias/stage2_images"))
    attn_path = Path(args.attention_json) if args.attention_json else (
        _ROOT / "research2visualtexorder/results/2026-11-12" /
        f"attention_{slug.lower().replace('-instruct', '').replace('-', '').replace('_', '')}"
        f"_{args.dataset}_all.json")
    items = load_items(args.dataset, limit=args.limit, seed=0, image_dir=image_dir)
    truth = [it["answer"] for it in items]
    backend = HFVLMBackend(args.model, device="cuda", dtype="bfloat16", batch_size=1,
                           max_pixels=448 * 448)
    attn_data = json.loads(attn_path.read_text())
    layers = attn_data["positions"]["suffix"]["layers"]
    top_heads = []
    all_heads = []
    for layer, values in layers.items():
        for head, value in enumerate(values["head_means"]):
            pair = (int(layer), int(head))
            all_heads.append(pair)
            if value > 0.5:
                top_heads.append(pair)
    rng = random.Random(0)
    random_heads = [h for h in all_heads if h not in top_heads]
    rng.shuffle(random_heads)
    random_heads = random_heads[:len(top_heads)]
    out = {"model": args.model, "dataset": args.dataset, "n": len(items),
           "top_heads": [list(h) for h in top_heads],
           "random_heads": [list(h) for h in random_heads],
           "conditions": {}}
    for placement in ("prefix", "suffix"):
        base, _ = run(backend, items, placement)
        top, _ = run(backend, items, placement, top_heads)
        rnd, _ = run(backend, items, placement, random_heads)
        out["conditions"][placement] = {}
        for label, preds in (("baseline", base), ("top_set", top),
                             ("random_set", rnd)):
            acc = float(np.mean([p == t for p, t in zip(preds, truth)]))
            flip = float(np.mean([p != b for p, b in zip(preds, base)]))
            out["conditions"][placement][label] = {"accuracy": acc, "flip": flip}
            print(placement, label, "acc", round(acc, 3), "flip", round(flip, 3))
    dest = Path(args.out) if args.out else (
        _ROOT / "research2visualtexorder/results/2026-11-12" /
        f"analysis_head_set_ablation_{slug}_{args.dataset}.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
