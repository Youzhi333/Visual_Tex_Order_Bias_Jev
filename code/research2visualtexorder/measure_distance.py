"""Token distance from the image to the answer position, without a forward.

    python research2visualtexorder/measure_distance.py

For a small sample, render each placement, encode it, and report the number of
tokens between the last image token and the answer position, plus the prompt
length. This is the confound the paper's four placements vary; it names the
distance instead of assuming it.
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
from anyjev.readout import build_prompt  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402

POSITIONS = ("prefix", "middle", "question", "suffix")


def set_placement(backend, placement, is_llava):
    if placement == "prefix":
        backend.image_placement = "prefix"
    elif is_llava:
        backend.image_placement = "splice"
        backend.image_splice = placement
    elif placement in ("middle", "question"):
        backend.image_placement = "split"
    else:
        backend.image_placement = "suffix"


def measure(backend, item, placement, is_llava):
    text = item.get("state", "")
    if placement == "middle" and not is_llava:
        text = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
    question_text = item["question"]
    if placement == "question" and not is_llava:
        question_text = question_text + IMAGE_SPLIT
    question = Question.choice(question_text, item["options"], name="answer")
    state = VisualState(text=text, images=tuple(item["images"]))
    set_placement(backend, placement, is_llava)
    spec = build_prompt(render_state(state), question, list(range(question.k)))
    prompt, marker = backend.render_chat_parts(spec)
    key = marker.rsplit(_ID_MARK, 1)[1]
    stored = backend._pending.pop(key)
    prompt, images = stored
    enc = backend._inputs([prompt], [list(images)])
    if is_llava and placement != "prefix":
        enc = backend._splice_encodings(enc)
    image_token = getattr(backend.processor, "image_token", None) or "<|image_pad|>"
    image_id = backend.tokenizer.convert_tokens_to_ids(image_token)
    ids = enc["input_ids"][0].tolist()
    image_positions = [i for i, token in enumerate(ids) if token == image_id]
    if not image_positions:
        return None
    return {
        "distance": int(len(ids) - 1 - image_positions[-1]),
        "prompt_tokens": int(len(ids)),
        "image_tokens": int(len(image_positions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa")
    parser.add_argument("--models", default=",".join((
        "Qwen/Qwen3-VL-4B-Instruct",
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    parser.add_argument(
        "--out",
        default=str(_ROOT / "research2visualtexorder/results/2026-11-12/"
                    "analysis_distance.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    args = parser.parse_args()
    items = load_items(args.dataset, limit=args.limit, seed=0, image_dir=args.image_dir)
    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        is_llava = model.startswith("llava-hf/")
        backend = HFVLMBackend(model, device=args.device, dtype=args.dtype,
                               batch_size=1, max_pixels=args.max_pixels)
        per_position = {}
        for placement in POSITIONS:
            values = [measure(backend, item, placement, is_llava) for item in items]
            values = [v for v in values if v]
            per_position[placement] = {
                "distance": float(np.mean([v["distance"] for v in values])),
                "prompt_tokens": float(np.mean([v["prompt_tokens"] for v in values])),
                "image_tokens": float(np.mean([v["image_tokens"] for v in values])),
            }
        rows.append({"model": model, "dataset": args.dataset, "n": len(items),
                     "positions": per_position})
        print(f"{model}: " + " ".join(
            f"{p}={per_position[p]['distance']:.0f}" for p in POSITIONS), flush=True)
        del backend
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    short = {
        "Qwen/Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
        "HuggingFaceM4/Idefics3-8B-Llama3": "Idefics3-8B",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
        "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-NeXT-7B",
    }
    lines = [
        "% generated by measure_distance.py on 50 A-OKVQA items",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & prefix & pre-Q & pre-O & suffix \\\\",
        "\\midrule",
    ]
    for row in rows:
        p = row["positions"]
        lines.append(f"{short.get(row['model'], row['model'])} & "
                     f"{p['prefix']['distance']:.0f} & {p['middle']['distance']:.0f} & "
                     f"{p['question']['distance']:.0f} & {p['suffix']['distance']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = Path(__file__).parent / "paper/tables/distance.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}\nwrote {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
