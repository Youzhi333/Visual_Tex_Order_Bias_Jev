"""Attention probe: image attention at the answer position, every layer and head.

    python research2visualtexorder/attention_probe.py \
        --dataset aokvqa --limit 100 --layers all \
        --image-dir research/visual_bias/images

Loads the model with eager attention so `output_attentions=True` returns the
maps. For every selected layer it stores the per-head image-attention means and
the head mean/std/max/min. LLaVA checkpoints use the input_ids splice for the
middle/question/suffix placements because their content-part template forces
the image first.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Question, VisualState  # noqa: E402
from anyjev.backends.hf_vlm import _ID_MARK, HFVLMBackend  # noqa: E402
from anyjev.readout import build_prompt, resolve_labels  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402

POSITIONS = ("prefix", "middle", "question", "suffix")


def resolve_layers(attentions, requested):
    n = len(attentions)
    if requested == "all":
        return list(range(n))
    out = []
    for layer in requested:
        index = layer if layer >= 0 else n + layer
        if 0 <= index < n and index not in out:
            out.append(index)
    return out


def _set_placement(backend, placement: str, is_llava: bool):
    if placement == "prefix":
        backend.image_placement = "prefix"
    elif is_llava:
        backend.image_placement = "splice"
        backend.image_splice = placement
    elif placement in ("middle", "question"):
        backend.image_placement = "split"
    else:
        backend.image_placement = "suffix"


def image_mass(backend, item, placement: str, requested_layers, is_llava: bool):
    import torch

    text = item.get("state", "")
    if placement == "middle" and not is_llava:
        text = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
    _set_placement(backend, placement, is_llava)
    question_text = item["question"]
    if placement == "question" and not is_llava:
        question_text = question_text + IMAGE_SPLIT
    question = Question.choice(question_text, item["options"], name="answer")
    state = VisualState(text=text, images=tuple(item["images"]))
    spec = build_prompt(render_state(state), question, list(range(question.k)))
    prompt, marker = backend.render_chat_parts(spec)
    key = marker.rsplit(_ID_MARK, 1)[1]
    stored = backend._pending.pop(key)
    prompt, images = stored
    enc = backend._inputs([prompt], [list(images)])
    enc = {k: (v.to(backend.device) if hasattr(v, "to") else v) for k, v in enc.items()}
    if is_llava and placement != "prefix":
        enc = backend._splice_encodings(enc)
    image_token = getattr(backend.processor, "image_token", None) or "<|image_pad|>"
    image_id = backend.tokenizer.convert_tokens_to_ids(image_token)
    ids = enc["input_ids"][0].tolist()
    image_positions = [i for i, token in enumerate(ids) if token == image_id]
    if not image_positions:
        return {}, "no image tokens"
    labels, label_ids = resolve_labels(backend.tokenizer, question)
    label_id_set = {int(x) for x in label_ids}
    for label in labels:
        for variant in (label, " " + label, "\n" + label):
            encoded = backend.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                label_id_set.add(int(encoded[-1]))
    option_positions = [i for i, token in enumerate(ids) if token in label_id_set]
    with torch.inference_mode():
        out = backend.model(**enc, output_attentions=True, use_cache=False)
    if out.attentions is None:
        return {}, "no attentions"
    layers = resolve_layers(out.attentions, requested_layers)
    result = {}
    device = out.attentions[0].device
    index = torch.as_tensor(image_positions, device=device)
    option_index = torch.as_tensor(option_positions, device=device)
    for layer in layers:
        attn = out.attentions[layer][0]              # [heads, seq, seq]
        row = attn[:, -1, :]                          # answer position
        denom = row.sum(dim=1).clamp_min(1e-12)
        head_mass = row[:, index].sum(dim=1) / denom  # [heads]
        values = head_mass.float().cpu().numpy()
        if option_positions:
            option_mass = row[:, option_index].sum(dim=1) / denom
            option_values = option_mass.float().cpu().numpy()
            share = head_mass / (head_mass + option_mass).clamp_min(1e-12)
            share_values = share.float().cpu().numpy()
        else:
            option_values = np.zeros_like(values)
            share_values = np.zeros_like(values)
        result[str(layer)] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
            "min": float(values.min()),
            "heads": values.tolist(),
            "option_mean": float(option_values.mean()),
            "option_heads": option_values.tolist(),
            "share_mean": float(share_values.mean()),
            "share_heads": share_values.tolist(),
        }
    return result, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="all",
                        help="'all' or comma-separated indices, negative allowed")
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    parser.add_argument("--out", default=str(_ROOT / "research2visualtexorder/results/2026-11-12/attention.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--single-view", action="store_true",
                        help="disable Idefics3 image splitting; needed for "
                             "32-head eager attention fits")
    parser.add_argument("--downscale", type=int, default=None,
                        help="resize each image so its longest side is at most "
                             "this many pixels before the processor")
    parser.add_argument("--bf16-attention", action="store_true",
                        help="softmax the eager attention map in bf16 instead of "
                             "float32; halves the peak map memory")
    parser.add_argument("--square", action="store_true",
                        help="with --downscale, resize to a square instead of "
                             "preserving aspect; avoids extra anyres tiles")
    args = parser.parse_args()

    requested = args.layers if args.layers == "all" else [
        int(x) for x in args.layers.split(",") if x.strip()]
    is_llava = args.model.startswith("llava-hf/")
    items = load_items(args.dataset, limit=args.limit, seed=args.seed, image_dir=args.image_dir)
    if args.downscale:
        import tempfile

        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="anyjev_attn_"))
        for item in items:
            resized = []
            for j, path in enumerate(item["images"]):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if args.square:
                        image = image.resize((args.downscale, args.downscale))
                    else:
                        image.thumbnail((args.downscale, args.downscale))
                    out = tmp / f"{item['id']}_{j}.png"
                    image.save(out)
                    resized.append(str(out))
            item["images"] = tuple(resized)
    backend = HFVLMBackend(args.model, device=args.device, dtype=args.dtype, batch_size=1,
                           max_pixels=args.max_pixels, attn_implementation="eager")
    if args.single_view:
        image_processor = getattr(backend.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False
    import torch
    import transformers

    if args.bf16_attention:
        import torch.nn.functional as F
        import transformers.models.clip.modeling_clip as clip_modeling
        import transformers.models.mistral.modeling_mistral as mistral
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.mistral.modeling_mistral import repeat_kv

        original_clip = clip_modeling.eager_attention_forward

        def eager_bf16(module, query, key, value, attention_mask, scaling,
                       dropout=0.0, **kwargs):
            if not hasattr(module, "num_key_value_groups"):
                return original_clip(module, query, key, value, attention_mask,
                                     scaling, dropout=dropout, **kwargs)
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        ALL_ATTENTION_FUNCTIONS["eager"] = eager_bf16
        mistral.eager_attention_forward = eager_bf16

    result = {"dataset": args.dataset, "model": args.model, "n": len(items),
              "requested_layers": args.layers,
              "env": {
                  "date": datetime.now().isoformat(timespec="seconds"),
                  "python": platform.python_version(),
                  "torch": torch.__version__,
                  "transformers": transformers.__version__,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "cuda": torch.version.cuda,
                  "dtype": args.dtype,
                  "batch_size": 1,
                  "max_pixels": args.max_pixels,
                  "single_view": bool(args.single_view),
                  "downscale": args.downscale,
                  "square": bool(args.square),
                  "bf16_attention": bool(args.bf16_attention),
                  "attention": "eager",
                  "device": args.device,
              },
              "positions": {}}
    for placement in POSITIONS:
        per_layer = {}
        notes = set()
        started = time.time()
        for item in items:
            masses, note = image_mass(backend, item, placement, requested, is_llava)
            notes.add(note)
            torch.cuda.empty_cache()
            for layer, values in masses.items():
                per_layer.setdefault(layer, []).append(values)
        summary = {}
        for layer, rows in per_layer.items():
            head_matrix = np.asarray([r["heads"] for r in rows], dtype=float)
            option_matrix = np.asarray([r["option_heads"] for r in rows], dtype=float)
            share_matrix = np.asarray([r["share_heads"] for r in rows], dtype=float)
            summary[layer] = {
                "mean": float(np.mean([r["mean"] for r in rows])),
                "mean_std": float(np.mean([r["std"] for r in rows])),
                "mean_max": float(np.mean([r["max"] for r in rows])),
                "mean_min": float(np.mean([r["min"] for r in rows])),
                "head_means": head_matrix.mean(axis=0).tolist(),
                "head_std": head_matrix.std(axis=0).tolist(),
                "head_values": head_matrix.tolist(),
                "share_values": share_matrix.tolist(),
                "option_mean": float(np.mean([r["option_mean"] for r in rows])),
                "option_head_means": option_matrix.mean(axis=0).tolist(),
                "share_mean": float(np.mean([r["share_mean"] for r in rows])),
                "share_head_means": share_matrix.mean(axis=0).tolist(),
                "n": len(rows),
            }
        result["positions"][placement] = {
            "layers": summary, "notes": sorted(notes),
            "seconds": time.time() - started}
        print(f"{placement:8s} " + " ".join(
            f"L{layer}:{values['mean']:.4f}" for layer, values in summary.items()), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
