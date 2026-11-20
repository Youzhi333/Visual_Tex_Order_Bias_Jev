"""Independent content-order check for the LLaVA suffix splice.

    python research2visualtexorder/splice_order_check.py

Builds the prompt by replacing a text placeholder in LLaVA's own chat template
with text followed by an image placeholder, so the image is after the text at
the tokenizer level. Compares that path with the input_ids splice. This is not
the model's official single-turn template (which forces the image first), but
it is an independent order-preserving path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import HFVLMBackend  # noqa: E402
from anyjev.readout import build_prompt, resolve_labels  # noqa: E402
from anyjev.state import VISUAL_MARK_RE, render_state  # noqa: E402
from label_permutation import read as splice_read  # noqa: E402

MODELS = ("llava-hf/llava-onevision-qwen2-7b-ov-hf",
          "llava-hf/llava-v1.6-mistral-7b-hf")


def native_order(backend, item, labels, ids):
    question = Question.choice(item["question"], item["options"], name="answer")
    state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
    spec = build_prompt(render_state(state), question, list(range(question.k)),
                        labels=labels)
    text = VISUAL_MARK_RE.sub("", spec.user)
    template = backend.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "__TEXT__"}]}],
        tokenize=False)
    prompt = template.replace("__TEXT__", text + "<image>")
    enc = backend._inputs([prompt], [list(item["images"])])
    enc = {k: (v.to(backend.device) if hasattr(v, "to") else v) for k, v in enc.items()}
    import torch

    with torch.inference_mode():
        logits = backend.model(**enc).logits[:, -1, :].float()
    lp = torch.log_softmax(logits, dim=-1)[0, torch.as_tensor(ids, device=logits.device)]
    z = lp.cpu().numpy().astype(float)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    parser.add_argument("--out", default=str(_ROOT / "research2visualtexorder/results/2026-11-12/analysis_splice_order.json"))
    args = parser.parse_args()
    items = load_items("aokvqa", limit=args.limit, seed=0, image_dir=args.image_dir)
    rows = []
    for model in MODELS:
        backend = HFVLMBackend(model, device="cuda", dtype="bfloat16", batch_size=1,
                               max_pixels=448 * 448)
        agree, tvs = 0, []
        for item in items:
            question = Question.choice(item["question"], item["options"], name="answer")
            labels, ids = resolve_labels(backend.tokenizer, question)
            ps = splice_read(backend, item, "suffix", labels, ids, True)
            pn = native_order(backend, item, labels, ids)
            agree += (item["options"][int(ps.argmax())] ==
                      item["options"][int(pn.argmax())])
            tvs.append(0.5 * np.abs(ps - pn).sum())
        rows.append({"model": model, "n": len(items),
                     "agreement": agree / len(items),
                     "mean_tv": float(np.mean(tvs))})
        print(model, "agreement", round(agree / len(items), 3),
              "TV", round(float(np.mean(tvs)), 3), flush=True)
        del backend
        import torch

        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
