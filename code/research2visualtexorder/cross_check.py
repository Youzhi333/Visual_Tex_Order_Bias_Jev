"""Cross-family placement check: print where the image placeholder lands.

    python research2visualtexorder/cross_check.py \
        --model llava-hf/llava-onevision-qwen2-7b-ov-hf --dataset aokvqa

This does not assert a token span (LLaVA/Idefics expand the placeholder
differently from Qwen). It checks the character order of the image placeholder
relative to State, Question and Options, and the monotonicity of the token-level
positions. A FAIL means the processor template does not preserve the placement;
that family is reported as a limitation, not run in the grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from anyjev import Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402
from anyjev.readout import build_prompt  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402
from visual_datasets import load_items  # noqa: E402


def find_image_token(processor, prompt: str) -> tuple[str, int]:
    candidates = [getattr(processor, "image_token", None), "<image>", "<|image_pad|>"]
    for token in candidates:
        if token and token in prompt:
            return token, prompt.find(token)
    return "", -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
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

    failures = []
    for placement in ("prefix", "middle", "question", "suffix"):
        backend.image_placement = "split" if placement in ("middle", "question") else placement
        text = item.get("state", "")
        if placement == "middle":
            text = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
        state = VisualState(text=text, images=tuple(item["images"]))
        question_text = item["question"] + (IMAGE_SPLIT if placement == "question" else "")
        question = Question.choice(question_text, item["options"], name="answer")
        prompt, _ = backend.render_chat_parts(
            build_prompt(render_state(state), question, list(range(question.k))))
        token, image_pos = find_image_token(backend.processor, prompt)
        positions = {
            "image": image_pos,
            "state": prompt.find("State:"),
            "question": prompt.find("Question:"),
            "options": prompt.find("Options:"),
        }
        if placement == "prefix":
            ok = 0 <= image_pos < positions["state"]
        elif placement == "middle":
            ok = positions["state"] < image_pos < positions["question"]
        elif placement == "question":
            ok = positions["question"] < image_pos < positions["options"]
        else:
            ok = positions["options"] < image_pos
        print(f"{placement:8s} token={token!r} positions={positions} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(placement)

    starts = []
    for fraction in (0.0, 0.5, 1.0):
        backend.image_placement = "token"
        backend.image_position = fraction
        state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
        question = Question.choice(item["question"], item["options"], name="answer")
        prompt, _ = backend.render_chat_parts(
            build_prompt(render_state(state), question, list(range(question.k))))
        _, image_pos = find_image_token(backend.processor, prompt)
        starts.append(image_pos)
        print(f"token {fraction:.1f} image={image_pos}", flush=True)
    monotone = all(second >= first for first, second in zip(starts, starts[1:]))
    print(f"token monotone: {'PASS' if monotone else 'FAIL'}", flush=True)
    if not monotone:
        failures.append("token")

    if failures:
        print(f"cross-family placement failed: {failures}", flush=True)
        return 1
    print("cross-family placement passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
