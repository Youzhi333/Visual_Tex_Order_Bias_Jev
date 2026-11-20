"""Label and position bias of the training-free readout on a visual choice task.

    python scripts/eval_visual_bias.py --full \
        --models Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct,Qwen/Qwen3-VL-8B-Instruct

Dataset: A-OKVQA validation (4-way VQA). The official repository allenai/aokvqa is
Apache-2.0; the images are the COCO images and keep COCO's image terms. Downloaded at
run time, never vendored; evaluation only.

Every item has its own question, so AnyJev's per-question batch prior cannot pool
across items. The content-free prior used here is the visual analogue: the same
question and options with the image replaced by a uniform grey field. The prior is
applied after the readout (`p / q^0.75`) as a diagnostic, not inside L0.

Measures:
  * raw and L0 accuracy, ECE, cov@5%, and the grey-prior-corrected variants;
  * position bias: answer flip under full option reversal (the README's `flip`), the
    cyclic-shift disagreement recorded by the readout, and probability-level changes
    (total variation, KL, log-margin change, correct-answer probability change);
  * label/vision prior: the grey-image position distribution, how much the answer
    moves when the image is removed, and what dividing that prior out does;
  * bootstrap 95% intervals over items for the key effects, including L0 - raw.
"""
import argparse
import gc
import json
import platform
import random
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from anyjev import Decider, Question, VisualState

_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


# ---------------------------------------------------------------- metrics
def macro_f1(y_true, y_pred, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def ece(decisions, truth, bins=10):
    conf = np.asarray([d.confidence for d in decisions])
    correct = np.asarray([d.argmax == t for d, t in zip(decisions, truth)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def cov_at_risk(decisions, truth, risk=0.05):
    conf = np.asarray([d.confidence for d in decisions])
    correct = np.asarray([d.argmax == t for d, t in zip(decisions, truth)], dtype=float)
    order = np.argsort(-conf)
    running = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(running >= 1.0 - risk)[0]
    return float((ok.max() + 1) / len(order)) if len(ok) else 0.0


def tv_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(p - q).sum())


def kl_divergence(p, q):
    p = np.clip(np.asarray(p, dtype=float), EPS, None)
    q = np.clip(np.asarray(q, dtype=float), EPS, None)
    return float((p * np.log(p / q)).sum())


def log_margin(probs):
    p = np.sort(np.asarray(probs, dtype=float))[::-1]
    if p.size < 2:
        return 0.0
    return float(np.log(max(p[0], EPS)) - np.log(max(p[1], EPS)))


def align_reversed(probs):
    """Reversed option order -> original order (rev[j] is original[K-1-j])."""
    return np.asarray(probs, dtype=float)[::-1]


def pair_stats(orig, rev, choices, answer, align=True):
    """Probability-level comparison for one item.

    `rev` is the reversed-option run when `align=True`; when options were sorted to a
    canonical order before both reads, pass `align=False` and the probs are already
    in the same option order.
    """
    p = np.asarray(orig.probs, dtype=float)
    q = align_reversed(rev.probs) if align else np.asarray(rev.probs, dtype=float)
    idx = choices.index(answer)
    m0, m1 = log_margin(p), log_margin(q)
    return {
        "tv": tv_distance(p, q),
        "kl": kl_divergence(p, q),
        "margin": m0,
        "margin_reversed": m1,
        "margin_change": m0 - m1,
        "abs_margin_change": abs(m0 - m1),
        "p_correct": float(p[idx]),
        "p_correct_reversed": float(q[idx]),
        "correct_change": float(p[idx] - q[idx]),
    }


def boot_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_summary(values, seed=0):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "ci95": boot_ci(v, seed=seed)}


def summarise(decisions, truth):
    return {
        "acc": float(np.mean([d.argmax == t for d, t in zip(decisions, truth)])),
        "macro_f1": macro_f1(list(truth), [d.argmax for d in decisions], sorted(set(truth))),
        "ece": ece(decisions, truth),
        "cov@5%": cov_at_risk(decisions, truth),
        "mean_confidence": float(np.mean([d.confidence for d in decisions])),
        "mean_answer_mass": float(np.mean([d.diagnostics.get("answer_mass", 1.0) for d in decisions])),
    }


def position_prior(decisions, k=None):
    rows = [d.probs for d in decisions if k is None or len(d.probs) == k]
    return [float(x) for x in np.mean(rows, axis=0)] if rows else []


def prior_entropy(prior):
    p = np.clip(np.asarray(prior, dtype=float), EPS, None)
    return float(-(p * np.log(p)).sum())


def divide_prior(decisions, priors, strength=0.75):
    """p_corrected proportional to p / q^strength, renormalised."""
    out = []
    for d, q in zip(decisions, priors):
        z = np.asarray(d.probs, dtype=float) / np.power(np.clip(q, EPS, None), strength)
        out.append(replace(d, probs=z / z.sum()))
    return out


def model_slug(model):
    return model.split("/")[-1].replace(".", "_")


def default_batch(model):
    name = model.lower()
    if "8b" in name or "7b" in name:
        return 1
    if "4b" in name or "3b" in name:
        return 2
    return 4


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="single model (backward compatible)")
    ap.add_argument("--models", default=None, help="comma-separated model ids; overrides --model")
    ap.add_argument("--dataset", default="HuggingFaceM4/A-OKVQA")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--full", action="store_true", help="use the whole evaluation split")
    ap.add_argument("--limit", type=int, default=200, help="used when --full is absent")
    ap.add_argument("--seed", type=int, default=0, help="subset seed when --full is absent")
    ap.add_argument("--batch-size", type=int, default=None, help="default: model-size dependent")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    ap.add_argument("--out-dir", default=str(_ROOT / "research/visual_bias/aokvqa_full"))
    ap.add_argument("--out", default=None, help="single-model output path; overrides --out-dir naming")
    ap.add_argument("--no-grey", action="store_true", help="skip the grey-image control")
    args = ap.parse_args()

    import torch
    import transformers
    from datasets import load_dataset

    import anyjev
    from anyjev.backends.hf_vlm import HFVLMBackend

    models = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
              else [args.model or "Qwen/Qwen3-VL-4B-Instruct"])

    rows = load_dataset(args.dataset, split=args.split)
    order = list(range(len(rows)))
    if not args.full:
        random.Random(args.seed).shuffle(order)
        order = order[: args.limit]
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i in order:
        row = rows[i]
        choices = list(dict.fromkeys(row["choices"]))
        if len(choices) < 2:
            continue
        qid = str(row["question_id"])
        path = image_dir / f"{qid}.png"
        if not path.exists():
            row["image"].convert("RGB").save(path)
        items.append({"id": qid, "question": row["question"], "choices": choices,
                      "answer": row["choices"][int(row["correct_choice_idx"])], "image": str(path)})
    ks = Counter(len(it["choices"]) for it in items)
    modal_k = ks.most_common(1)[0][0]
    truth = [it["answer"] for it in items]
    print(f"dataset={args.dataset} split={args.split} full={args.full} items={len(items)} "
          f"k_counts={dict(ks)} models={models}", flush=True)

    combined = {
        "dataset": args.dataset, "split": args.split, "full": args.full,
        "n": len(items), "k_counts": dict(ks), "modal_k": modal_k,
        "seed": None if args.full else args.seed, "prior_strength": args.strength,
        "models": {},
    }

    def run(backend, level, reversed_options=False, grey=False):
        decider = Decider(backend, shared_prefix=False, prior="none")
        out, t0 = [], time.time()
        for it in items:
            choices = it["choices"][::-1] if reversed_options else it["choices"]
            q = Question.choice(it["question"], choices, name="answer")
            image = it["image"]
            if grey:
                from PIL import Image
                image = str(image_dir / f"{it['id']}_grey.png")
                if not Path(image).exists():
                    with Image.open(it["image"]) as im:
                        Image.new("RGB", im.size, (128, 128, 128)).save(image)
            out.append(decider.decide(VisualState(images=(image,)), [q], level=level)["answer"])
        return out, time.time() - t0

    for model in models:
        batch_size = args.batch_size if args.batch_size is not None else default_batch(model)
        backend = HFVLMBackend(model, device=args.device, dtype=args.dtype,
                               batch_size=batch_size, max_pixels=args.max_pixels)

        result = {
            "model": model, "dataset": args.dataset, "split": args.split,
            "license": "A-OKVQA official repository Apache-2.0; COCO images keep COCO's image terms",
            "full": args.full, "n": len(items), "k_counts": dict(ks), "modal_k": modal_k,
            "seed": None if args.full else args.seed, "prior_strength": args.strength,
            "env": {
                "date": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(), "anyjev": anyjev.__version__,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "dtype": args.dtype, "batch_size": batch_size, "max_pixels": args.max_pixels,
            },
            "metrics": {}, "position_prior": {},
        }

        runs, reversed_runs = {}, {}
        for level in ("raw", "L0"):
            runs[level], secs = run(backend, level)
            result["metrics"][level] = {**summarise(runs[level], truth), "seconds": secs}
            result["position_prior"][level] = position_prior(runs[level], modal_k)
            if level == "L0":
                result["metrics"]["cyclic_flip_raw"] = float(
                    np.mean([d.diagnostics.get("order_flip_raw", 0.0) for d in runs[level]]))
                result["metrics"]["cyclic_flip_l0"] = float(
                    np.mean([d.diagnostics.get("order_flip_l0", 0.0) for d in runs[level]]))
            print(f"{model} {level:3s} {result['metrics'][level]}", flush=True)

        for level in ("raw", "L0"):
            reversed_runs[level], secs = run(backend, level, reversed_options=True)
            result["metrics"][f"{level}_reversed"] = {
                **summarise(reversed_runs[level], truth), "seconds": secs,
                "flip_vs_original": float(np.mean([a.argmax != b.argmax
                                                   for a, b in zip(runs[level], reversed_runs[level])])),
            }
            print(f"{model} {level} reversed {result['metrics'][f'{level}_reversed']}", flush=True)

        # probability-level bias. Compute this and checkpoint BEFORE the grey pass:
        # the grey pass is where the libcuda crash happened, and raw/L0/reversed are
        # already complete at this point.
        grey = None
        raw_pair = [pair_stats(a, b, it["choices"], it["answer"])
                    for a, b, it in zip(runs["raw"], reversed_runs["raw"], items)]
        l0_pair = [pair_stats(a, b, it["choices"], it["answer"])
                   for a, b, it in zip(runs["L0"], reversed_runs["L0"], items)]
        for level, pairs in (("raw", raw_pair), ("L0", l0_pair)):
            result["metrics"][level].update({
                "tv_mean": float(np.mean([p["tv"] for p in pairs])),
                "kl_mean": float(np.mean([p["kl"] for p in pairs])),
                "margin_change_mean": float(np.mean([p["margin_change"] for p in pairs])),
                "abs_margin_change_mean": float(np.mean([p["abs_margin_change"] for p in pairs])),
                "p_correct_change_mean": float(np.mean([p["correct_change"] for p in pairs])),
            })

        correct_raw = np.asarray([d.argmax == t for d, t in zip(runs["raw"], truth)], dtype=int)
        correct_l0 = np.asarray([d.argmax == t for d, t in zip(runs["L0"], truth)], dtype=int)
        flip_raw = np.asarray([a.argmax != b.argmax for a, b in zip(runs["raw"], reversed_runs["raw"])], dtype=int)
        flip_l0 = np.asarray([a.argmax != b.argmax for a, b in zip(runs["L0"], reversed_runs["L0"])], dtype=int)
        result["bootstrap"] = {
            "raw_acc": boot_summary(correct_raw),
            "L0_acc": boot_summary(correct_l0),
            "acc_delta": boot_summary(correct_l0 - correct_raw),
            "raw_flip": boot_summary(flip_raw),
            "L0_flip": boot_summary(flip_l0),
            "flip_delta": boot_summary(flip_l0 - flip_raw),
            "raw_tv": boot_summary([p["tv"] for p in raw_pair]),
            "L0_tv": boot_summary([p["tv"] for p in l0_pair]),
            "raw_margin_change": boot_summary([p["margin_change"] for p in raw_pair]),
            "L0_margin_change": boot_summary([p["margin_change"] for p in l0_pair]),
        }

        if args.out and len(models) == 1:
            out = Path(args.out)
        else:
            out = out_dir / f"aokvqa_{model_slug(model)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        def build_per_item(with_grey: bool):
            return [
                {
                    "id": it["id"], "k": len(it["choices"]), "answer": it["answer"],
                    "raw": runs["raw"][i].argmax, "L0": runs["L0"][i].argmax,
                    "raw_reversed": reversed_runs["raw"][i].argmax,
                    "L0_reversed": reversed_runs["L0"][i].argmax,
                    "raw_grey": None if not with_grey else grey[i].argmax,
                    "raw_probs": [float(x) for x in runs["raw"][i].probs],
                    "L0_probs": [float(x) for x in runs["L0"][i].probs],
                    "raw_reversed_aligned": [float(x) for x in align_reversed(reversed_runs["raw"][i].probs)],
                    "L0_reversed_aligned": [float(x) for x in align_reversed(reversed_runs["L0"][i].probs)],
                    "raw_grey_probs": None if not with_grey else [float(x) for x in grey[i].probs],
                    "raw_confidence": runs["raw"][i].confidence, "L0_confidence": runs["L0"][i].confidence,
                    "raw_pair": raw_pair[i], "L0_pair": l0_pair[i],
                }
                for i, it in enumerate(items)
            ]

        result["per_item"] = build_per_item(False)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote checkpoint {out}", flush=True)

        if not args.no_grey:
            grey, secs = run(backend, "raw", grey=True)
            result["metrics"]["raw_grey_image"] = {
                **summarise(grey, truth), "seconds": secs,
                "flip_vs_real": float(np.mean([a.argmax != b.argmax for a, b in zip(runs["raw"], grey)])),
            }
            result["position_prior"]["raw_grey_image"] = position_prior(grey, modal_k)
            result["position_prior"]["raw_grey_entropy"] = prior_entropy(position_prior(grey, modal_k))
            for level in ("raw", "L0"):
                corrected = divide_prior(runs[level], [d.probs for d in grey], args.strength)
                result["metrics"][f"{level}_grey_prior"] = summarise(corrected, truth)
            result["per_item"] = build_per_item(True)
            print(f"{model} raw grey {result['metrics']['raw_grey_image']}", flush=True)

        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {out}", flush=True)

        combined["models"][model] = {
            "metrics": result["metrics"], "bootstrap": result["bootstrap"],
            "position_prior": result["position_prior"], "env": result["env"],
            "out": str(out),
        }
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = out_dir / "aokvqa_grid_summary.json"
    summary_path.write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
