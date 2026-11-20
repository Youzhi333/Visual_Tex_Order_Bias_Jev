"""Label-permutation control: permute the letters, not the options.

    python research2visualtexorder/label_permutation.py --dataset aokvqa

The option order is fixed. For each item we assign a random permutation of the
letter labels to the same options, read the label logits, and compare the
result with the identity assignment. This isolates a letter preference from the
option identity; it is not an option-order or option-rotation experiment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from visual_datasets import load_items  # noqa: E402

from anyjev import Question, VisualState  # noqa: E402
from anyjev.readout import build_prompt, resolve_labels  # noqa: E402
from anyjev.state import IMAGE_SPLIT, render_state  # noqa: E402

POSITIONS = ("prefix", "middle", "question", "suffix")


def pad(probs, max_k):
    out = np.zeros((len(probs), max_k), dtype=float)
    for i, p in enumerate(probs):
        out[i, :len(p)] = p
    return out


def set_placement(backend, placement, is_llava):
    if placement == "prefix":
        backend.image_placement = "prefix"
    elif is_llava:
        backend.image_placement = "splice"
        backend.image_splice = placement
    elif placement in ("middle", "question"):
        backend.image_placement = "split"
    else:
        backend.image_placement = placement


def read(backend, item, placement, labels, ids, is_llava):
    text = item.get("state", "")
    if placement == "middle" and not is_llava:
        text = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
    question_text = item["question"]
    if placement == "question" and not is_llava:
        question_text = question_text + IMAGE_SPLIT
    question = Question.choice(question_text, item["options"], name="answer")
    state = VisualState(text=text, images=tuple(item["images"]))
    set_placement(backend, placement, is_llava)
    spec = build_prompt(render_state(state), question, list(range(question.k)),
                        labels=labels)
    prompt, marker = backend.render_chat_parts(spec)
    lp = backend.next_token_logprobs([marker], [list(ids)])[0]
    z = np.asarray(lp, dtype=float)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa",
                        choices=["aokvqa", "scienceqa", "cvbench", "eurosat",
                                 "pets", "mmmu"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--perm-only", action="store_true",
                        help="read the identity condition from the committed "
                             "_4pos.json and run only the permuted condition")
    parser.add_argument("--full", action="store_true",
                        help="use the full evaluation set and the _full.json "
                             "identity condition")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    args = parser.parse_args()
    if args.full:
        args.limit = 0
    image_dir = args.image_dir or str(
        _ROOT / ("research/visual_bias/images" if args.dataset == "aokvqa"
                 else "research/visual_bias/stage2_images"))
    items = load_items(args.dataset, limit=args.limit, seed=args.seed,
                       image_dir=image_dir)
    is_llava = args.model.startswith("llava-hf/")
    raw = None
    if args.perm_only:
        slug = args.model.rstrip("/").split("/")[-1]
        suffix = "_full" if args.full else "_4pos"
        raw_path = (_ROOT / "research2visualtexorder/results/2026-11-12" /
                    f"{slug}_{args.dataset}{suffix}.json")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if len(raw["items"]) < len(items):
            raise SystemExit(f"raw item count {len(raw['items'])} < {len(items)}")
    from anyjev.backends.hf_vlm import HFVLMBackend

    backend = HFVLMBackend(args.model, device=args.device, dtype=args.dtype,
                           batch_size=1, max_pixels=args.max_pixels)
    truth = [it["answer"] for it in items]
    max_k = max(len(it["options"]) for it in items)
    out = {"dataset": args.dataset, "model": args.model, "n": len(items),
           "control": "letter permutation with fixed option order",
           "env": {"date": datetime.now().isoformat(timespec="seconds"),
                   "dtype": args.dtype, "max_pixels": args.max_pixels},
           "positions": {}}
    raw_positions = raw["raw"]["metrics"]["positions"] if raw is not None else POSITIONS
    for j, placement in enumerate(POSITIONS):
        perm_probs, id_probs, perm_pred, id_pred = [], [], [], []
        raw_name = raw_positions[j]
        for i, item in enumerate(items):
            labels, label_ids = resolve_labels(
                backend.tokenizer,
                Question.choice(item["question"], item["options"]))
            identity = list(labels)
            id_by_label = dict(zip(labels, label_ids))
            identity_ids = [id_by_label[label] for label in identity]
            rng = random.Random(args.seed + i)
            perm = list(identity)
            rng.shuffle(perm)
            while perm == identity and len(identity) > 1:
                rng.shuffle(perm)
            perm_ids = [id_by_label[label] for label in perm]
            if raw is not None:
                p_id = np.asarray(raw["raw"]["probs"][raw_name][i], dtype=float)
                id_pred.append(raw["raw"]["preds"][raw_name][i])
            else:
                p_id = read(backend, item, placement, identity, identity_ids,
                            is_llava)
                id_pred.append(item["options"][int(p_id.argmax())])
            p_perm = read(backend, item, placement, perm, perm_ids, is_llava)
            id_probs.append(p_id)
            perm_probs.append(p_perm)
            perm_pred.append(item["options"][int(p_perm.argmax())])
        id_probs = pad(id_probs, max_k)
        perm_probs = pad(perm_probs, max_k)
        flip = np.asarray([a != b for a, b in zip(id_pred, perm_pred)], dtype=float)
        tv = 0.5 * np.abs(id_probs - perm_probs).sum(axis=1)
        correct_id = np.asarray([p == t for p, t in zip(id_pred, truth)], dtype=float)
        correct_perm = np.asarray([p == t for p, t in zip(perm_pred, truth)], dtype=float)
        out["positions"][placement] = {
            "flip": float(flip.mean()),
            "tv": float(tv.mean()),
            "identity_accuracy": float(correct_id.mean()),
            "permuted_accuracy": float(correct_perm.mean()),
            "preds_identity": id_pred,
            "preds_permuted": perm_pred,
        }
        print(f"{placement:8s} flip={flip.mean():.3f} tv={tv.mean():.3f} "
              f"id_acc={correct_id.mean():.3f} perm_acc={correct_perm.mean():.3f}",
              flush=True)
    dest = Path(args.out) if args.out else (
        _ROOT / "research2visualtexorder/results/2026-11-12" /
        f"label_perm_{args.model.split('/')[-1]}_{args.dataset}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
