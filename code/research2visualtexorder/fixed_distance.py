"""Fixed-distance structural-boundary experiment.

    python research2visualtexorder/fixed_distance.py

Two prompts put the image on opposite sides of the option list, but padding
after the image in the suffix condition equalizes the token distance from the
last image token to the answer. A difference between the two conditions is then
not a distance difference. Qwen3-VL-4B, 100 A-OKVQA items.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import _ID_MARK, HFVLMBackend  # noqa: E402
from anyjev.readout import PromptSpec, build_prompt, resolve_labels  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def prepare(backend, item, mode, filler=0, keep=False, filler_text=" the"):
    question = Question.choice(item["question"], item["options"], name="answer")
    state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
    spec = build_prompt(render_state(state), question, list(range(question.k)))
    if mode == "pre_options":
        user = spec.user.replace("Options:", IMAGE_SPLIT + "Options:", 1)
    else:
        user = spec.user + IMAGE_SPLIT + (filler_text * filler)
    backend.image_placement = "split"
    new_spec = PromptSpec(system=spec.system, user=user, split=spec.split)
    prompt, marker = backend.render_chat_parts(new_spec)
    key = marker.rsplit(_ID_MARK, 1)[1]
    stored = backend._pending[key] if keep else backend._pending.pop(key)
    prompt, images = stored
    enc = backend._inputs([prompt], [list(images)])
    image_token = getattr(backend.processor, "image_token", None) or "<|image_pad|>"
    image_id = backend.tokenizer.convert_tokens_to_ids(image_token)
    ids = enc["input_ids"][0].tolist()
    positions = [i for i, t in enumerate(ids) if t == image_id]
    distance = int(len(ids) - 1 - max(positions))
    return marker, distance


def read(backend, item, mode, filler=0, filler_text=" the"):
    question = Question.choice(item["question"], item["options"], name="answer")
    labels, ids = resolve_labels(backend.tokenizer, question)
    marker, distance = prepare(backend, item, mode, filler, keep=True,
                               filler_text=filler_text)
    lp = backend.next_token_logprobs([marker], [list(ids)])[0]
    z = np.asarray(lp, dtype=float)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum(), distance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa", choices=["aokvqa", "eurosat"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--filler", default=" the",
                        help="filler string repeated after the image")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    image_dir = args.image_dir or str(
        _ROOT / ("research/visual_bias/images" if args.dataset == "aokvqa"
                 else "research/visual_bias/stage2_images"))
    items = load_items(args.dataset, limit=args.limit, seed=0, image_dir=image_dir)
    truth = [it["answer"] for it in items]
    backend = HFVLMBackend(args.model, device="cuda", dtype="bfloat16", batch_size=1,
                           max_pixels=448 * 448)
    flips, dists, preds_a, preds_b = [], [], [], []
    for item in items:
        _, target = prepare(backend, item, "pre_options", filler_text=args.filler)
        filler = target
        _, got = prepare(backend, item, "suffix", filler, filler_text=args.filler)
        for _ in range(3):
            if got == target:
                break
            filler += target - got
            filler = max(0, filler)
            _, got = prepare(backend, item, "suffix", filler,
                             filler_text=args.filler)
        pa, da = read(backend, item, "pre_options", filler_text=args.filler)
        pb, db = read(backend, item, "suffix", filler, filler_text=args.filler)
        pred_a = item["options"][int(pa.argmax())]
        pred_b = item["options"][int(pb.argmax())]
        preds_a.append(pred_a)
        preds_b.append(pred_b)
        flips.append(pred_a != pred_b)
        dists.append((da, db))
    del backend
    import torch

    torch.cuda.empty_cache()
    out = {
        "model": args.model, "dataset": args.dataset, "n": len(items),
        "conditions": ["image before options", "image after cue plus filler"],
        "filler": args.filler,
        "flip": float(np.mean(flips)),
        "accuracy_pre_options": float(np.mean([p == t for p, t in zip(preds_a, truth)])),
        "accuracy_padded_suffix": float(np.mean([p == t for p, t in zip(preds_b, truth)])),
        "distance_pre_options": float(np.mean([d[0] for d in dists])),
        "distance_padded_suffix": float(np.mean([d[1] for d in dists])),
        "max_distance_gap": int(max(abs(a - b) for a, b in dists)),
    }
    dest = Path(args.out) if args.out else (
        _ROOT / "research2visualtexorder/results/2026-11-12" /
        f"analysis_fixed_distance_{args.dataset}.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
