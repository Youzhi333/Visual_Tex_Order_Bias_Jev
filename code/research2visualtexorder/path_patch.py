"""Answer-position path patch for the Qwen3-VL-4B suffix readout.

    python research2visualtexorder/path_patch.py

Selects the top-10 heads by suffix-minus-prefix attention on items 1--50, then
on held-out items 51--100 replaces the suffix attention output for those heads
at the answer token with the prefix value. If suffix decisions move toward
prefix, the image-position path is mediated by those heads at that token.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Question  # noqa: E402
from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402
from anyjev.readout import resolve_labels  # noqa: E402
from label_permutation import read  # noqa: E402

RESULTS = _ROOT / "research2visualtexorder/results/2026-11-12"
MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def attn_module(model, layer):
    pattern = re.compile(rf"(?:^|\.)layers\.{layer}\.self_attn$")
    for name, module in model.named_modules():
        if pattern.search(name) and "vision" not in name.lower():
            return module
    raise RuntimeError(f"layer {layer} not found")


def select():
    data = json.loads((RESULTS / "attention_qwen3vl4b_aokvqa_all.json").read_text())
    p = data["positions"]["prefix"]["layers"]
    s = data["positions"]["suffix"]["layers"]
    ranked = []
    for layer in p:
        pm = np.asarray(p[layer]["head_values"])[:50].mean(axis=0)
        sm = np.asarray(s[layer]["head_values"])[:50].mean(axis=0)
        for head, value in enumerate(sm - pm):
            ranked.append((float(value), int(layer), int(head)))
    ranked.sort(reverse=True)
    return [(layer, head) for _, layer, head in ranked[:10]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default=str(RESULTS / "analysis_path_patch.json"))
    args = parser.parse_args()
    items = load_items("aokvqa", limit=args.limit, seed=0,
                       image_dir=str(_ROOT / "research/visual_bias/images"))
    evaluate = items[50:100]
    truth = [it["answer"] for it in evaluate]
    heads = select()
    backend = HFVLMBackend(MODEL, device="cuda", dtype="bfloat16", batch_size=1,
                           max_pixels=448 * 448)
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    pre_pred, suf_pred, patched = [], [], []
    for item in evaluate:
        q = Question.choice(item["question"], item["options"], name="answer")
        labels, ids = resolve_labels(backend.tokenizer, q)
        captured = {}
        hooks = []
        for layer, head_list in by_layer.items():
            attn = attn_module(backend.model, layer)
            dim = attn.head_dim

            def cap(module, args, layer=layer, head_list=tuple(head_list), dim=dim):
                captured[layer] = args[0][:, -1, :].clone()

            hooks.append(attn.o_proj.register_forward_pre_hook(cap))
        p_pre = read(backend, item, "prefix", labels, ids, False)
        for hook in hooks:
            hook.remove()
        p_suf = read(backend, item, "suffix", labels, ids, False)
        hooks = []
        for layer, head_list in by_layer.items():
            attn = attn_module(backend.model, layer)
            dim = attn.head_dim
            src = captured[layer]

            def patch(module, args, head_list=tuple(head_list), dim=dim, src=src):
                x = args[0]
                for head in head_list:
                    x[:, -1, head * dim:(head + 1) * dim] = src[:, head * dim:(head + 1) * dim]
                return (x,) + args[1:]

            hooks.append(attn.o_proj.register_forward_pre_hook(patch))
        p_patch = read(backend, item, "suffix", labels, ids, False)
        for hook in hooks:
            hook.remove()
        pre_pred.append(item["options"][int(p_pre.argmax())])
        suf_pred.append(item["options"][int(p_suf.argmax())])
        patched.append(item["options"][int(p_patch.argmax())])
    out = {
        "model": MODEL, "dataset": "aokvqa", "n": len(evaluate),
        "heads": [list(h) for h in heads],
        "prefix_accuracy": float(np.mean([p == t for p, t in zip(pre_pred, truth)])),
        "suffix_accuracy": float(np.mean([p == t for p, t in zip(suf_pred, truth)])),
        "patched_accuracy": float(np.mean([p == t for p, t in zip(patched, truth)])),
        "suffix_flip": float(np.mean([a != b for a, b in zip(suf_pred, pre_pred)])),
        "patched_flip": float(np.mean([a != b for a, b in zip(patched, pre_pred)])),
        "patch_vs_suffix": float(np.mean([a != b for a, b in zip(patched, suf_pred)])),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
