"""Validate the input_ids splice on a template that forces the image first.

    python research2visualtexorder/splice_check.py \
        --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset aokvqa

Renders the prefix prompt, then moves the image-token block in `input_ids` to
the question, options, or answer-cue boundary. Prints the image span before and
after; PASS means the block moved and the sequence length is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from anyjev import Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import _ID_MARK, HFVLMBackend  # noqa: E402
from anyjev.readout import build_prompt  # noqa: E402
from anyjev.state import render_state  # noqa: E402
from visual_datasets import load_items  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--dataset", default="aokvqa")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    args = parser.parse_args()

    items = load_items(args.dataset, limit=args.limit, seed=args.seed, image_dir=args.image_dir)
    item = items[0]
    backend = HFVLMBackend(args.model, device=args.device, dtype=args.dtype,
                           batch_size=1, max_pixels=args.max_pixels)
    image_id = getattr(backend.processor, "image_token_id", None)
    if image_id is None:
        token = getattr(backend.processor, "image_token", None)
        image_id = backend.tokenizer.convert_tokens_to_ids(token)
    image_id = int(image_id)

    question = Question.choice(item["question"], item["options"], name="answer")
    state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
    spec = build_prompt(render_state(state), question, list(range(question.k)))
    backend.image_placement = "prefix"
    prompt, marker = backend.render_chat_parts(spec)
    key = marker.rsplit(_ID_MARK, 1)[1]
    stored = backend._pending.pop(key)
    prompt, images = stored
    enc = backend._inputs([prompt], [list(images)])
    ids0 = enc["input_ids"][0].tolist()
    positions0 = [i for i, token in enumerate(ids0) if token == image_id]
    span0 = (min(positions0), max(positions0)) if positions0 else None
    print(f"prefix image_tokens={len(positions0)} span={span0} length={len(ids0)}", flush=True)

    failures = []
    for splice in ("middle", "question", "suffix"):
        backend.image_placement = "splice"
        backend.image_splice = splice
        enc2 = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in enc.items()}
        try:
            enc2 = backend._splice_encodings(enc2)
        except ValueError as exc:
            print(f"{splice:8s} FAIL {exc}", flush=True)
            failures.append(splice)
            continue
        ids = enc2["input_ids"][0].tolist()
        positions = [i for i, token in enumerate(ids) if token == image_id]
        span = (min(positions), max(positions)) if positions else None
        multiset_ok = Counter(ids) == Counter(ids0)
        image_ok = len(positions) == len(positions0)
        pixel_ok = True
        if "pixel_values" in enc and "pixel_values" in enc2:
            a, b = enc["pixel_values"], enc2["pixel_values"]
            pixel_ok = a.shape == b.shape and bool((a == b).all())
        ok = (span is not None and span != span0 and len(ids) == len(ids0)
              and multiset_ok and image_ok and pixel_ok)
        print(f"{splice:8s} span={span} length={len(ids)} "
              f"multiset={multiset_ok} images={image_ok} pixels={pixel_ok} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(splice)

    if failures:
        print(f"splice check failed: {failures}", flush=True)
        return 1
    print("splice check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
