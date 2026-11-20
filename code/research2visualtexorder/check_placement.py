"""Phase 1: verify the three image placements on a real processor.

    python research2visualtexorder/check_placement.py \
        --model Qwen/Qwen3-VL-2B-Instruct --dataset aokvqa --limit 1

For each placement this prints the character order of the image placeholder
relative to the state, question and options, the expanded image-token span, and
one real decision. It exits non-zero if a placement is not where it should be.
No dataset loop and no metrics; this is the gate before the pilot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from anyjev import Decider, Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402
from anyjev.readout import build_prompt  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402
from visual_datasets import load_items  # noqa: E402


def state_text_for(item: dict, placement: str) -> str:
    text = item.get("state", "")
    if placement == "middle":
        return (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
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
    image_token = getattr(backend.processor, "image_token", None) or "<|image_pad|>"
    image_id = backend.tokenizer.convert_tokens_to_ids(image_token)

    failures = []
    for placement in ("prefix", "middle", "question", "suffix"):
        backend.image_placement = "split" if placement in ("middle", "question") else placement
        text = state_text_for(item, placement)
        state = VisualState(text=text, images=tuple(item["images"]))
        question_text = item["question"] + (IMAGE_SPLIT if placement == "question" else "")
        question = Question.choice(question_text, item["options"], name="answer")
        spec = build_prompt(render_state(state), question, list(range(question.k)))
        prompt, _ = backend.render_chat_parts(spec)

        positions = {
            "vision": prompt.find("<|vision_start|>"),
            "state": prompt.find("State:"),
            "question": prompt.find("Question:"),
            "options": prompt.find("Options:"),
        }
        enc = backend._inputs([prompt], [[backend._load(item["images"][0])]])
        ids = enc["input_ids"][0].tolist()
        image_positions = [i for i, token in enumerate(ids) if token == image_id]
        span = (min(image_positions), max(image_positions)) if image_positions else None

        decision = Decider(backend, shared_prefix=False, prior="none").decide(
            state, [question], level="raw")["answer"]
        print(f"{placement:6s} positions={positions} image_tokens={len(image_positions)} "
              f"span={span} decision={decision.argmax!r} conf={decision.confidence:.3f}", flush=True)

        vision = positions["vision"]
        if placement == "prefix":
            ok = 0 <= vision < positions["state"]
        elif placement == "middle":
            ok = positions["state"] < vision < positions["question"]
        elif placement == "question":
            ok = positions["question"] < vision < positions["options"]
        else:
            ok = positions["options"] < vision
        ok = ok and bool(image_positions) and image_id >= 0
        print(f"       {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(placement)
            start = max(0, vision - 80) if vision >= 0 else 0
            print("       prompt excerpt:", repr(prompt[start:start + 220]), flush=True)

    token_failures = []
    token_starts = []
    base_question = Question.choice(item["question"], item["options"], name="answer")
    state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        backend.image_placement = "token"
        backend.image_position = fraction
        spec = build_prompt(render_state(state), base_question, list(range(base_question.k)))
        prompt, _ = backend.render_chat_parts(spec)
        enc = backend._inputs([prompt], [[backend._load(item["images"][0])]])
        ids = enc["input_ids"][0].tolist()
        image_positions = [i for i, token in enumerate(ids) if token == image_id]
        start = min(image_positions) if image_positions else None
        token_starts.append(start)
        print(f"token {fraction:.2f} image_tokens={len(image_positions)} start={start}", flush=True)
        if not image_positions:
            token_failures.append(fraction)
    if any(start is None for start in token_starts):
        token_failures.append("missing image span")
    elif any(second < first for first, second in zip(token_starts, token_starts[1:])):
        token_failures.append("non-monotone span")
    if token_failures:
        print(f"token placement failed: {token_failures}", flush=True)
        failures.extend(str(item) for item in token_failures)
    else:
        print("token placement passed", flush=True)

    if failures:
        print(f"placement check failed: {failures}", flush=True)
        return 1
    print("placement check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
