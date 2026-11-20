"""Run the focused modality-order study for one dataset and one or more models.

    python research2visualtexorder/run_order.py \
        --dataset aokvqa --models Qwen/Qwen3-VL-4B-Instruct --limit 300 \
        --image-dir research/visual_bias/images --grey --pointwise

The readout is the AnyJev raw level (`level="raw"`, `prior="none"`): one
prefill, no generation. The image position is the only manipulated variable:
`prefix` (image-first), `middle` (between the state and the question), and
`suffix` (image-last). The option listing is fixed per item.

Every run writes a JSON with the per-item distributions and the metrics; the
paper tables are filled from those files only. Nothing is sampled.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from anyjev import Decider, Question, VisualState  # noqa: E402
from anyjev.state import IMAGE_SPLIT  # noqa: E402
from anyjev.visual_order import (  # noqa: E402
    dual_order_fuse,
    modality_flip,
    modality_instability,
    modality_tv,
)
from eval_visual_bias import boot_ci  # noqa: E402
from eval_visual_pointwise import label_id_sets, score_pointwise  # noqa: E402
from visual_datasets import load_items  # noqa: E402

TOKEN_POSITIONS = {"token0.25": 0.25, "token0.5": 0.5, "token0.75": 0.75}
SPLICE_POSITIONS = {"splice_middle": "middle", "splice_question": "question",
                    "splice_suffix": "suffix"}
SPLICE_TOKEN_POSITIONS = {"splice_token0.25": 0.25, "splice_token0.5": 0.5,
                          "splice_token0.75": 0.75}
POSITIONS = ("prefix", "middle", "question", "suffix", *TOKEN_POSITIONS,
             *SPLICE_POSITIONS, *SPLICE_TOKEN_POSITIONS)
LICENSES = {
    "aokvqa": "A-OKVQA official repository Apache-2.0; COCO images keep COCO's image terms",
    "scienceqa": "ScienceQA CC-BY-SA-4.0 (HF card); official repository MIT",
    "mmmu": "MMMU Apache-2.0",
    "cvbench": "CV-Bench Apache-2.0",
    "eurosat": "EuroSAT MIT (official repository)",
    "pets": "Oxford-IIIT Pets CC-BY-SA-4.0",
}


def model_slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def default_batch(model: str) -> int:
    return 4 if "2B" in model else 1


def _set_placement(backend, placement: str) -> None:
    if placement in SPLICE_TOKEN_POSITIONS:
        backend.image_placement = "splice_token"
        backend.image_position = SPLICE_TOKEN_POSITIONS[placement]
    elif placement in SPLICE_POSITIONS:
        backend.image_placement = "splice"
        backend.image_splice = SPLICE_POSITIONS[placement]
    elif placement in TOKEN_POSITIONS:
        backend.image_placement = "token"
        backend.image_position = TOKEN_POSITIONS[placement]
    else:
        backend.image_placement = "split" if placement in ("middle", "question") else placement


def state_for(item: dict, placement: str, images=None) -> VisualState:
    text = item.get("state", "")
    if placement == "middle":
        text = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
    return VisualState(text=text, images=tuple(images if images is not None else item["images"]))


def grey_images(item: dict, image_dir: Path, grey_dir: Path) -> tuple[str, ...]:
    from PIL import Image

    grey_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for j, path in enumerate(item["images"]):
        candidates = [image_dir / f"{item['id']}_grey_{j}.png",
                      image_dir / f"{item['id']}_grey.png"]
        grey = next((c for c in candidates if c.exists()), None)
        if grey is None:
            grey = grey_dir / f"{item['id']}_grey_{j}.png"
            if not grey.exists():
                with Image.open(path) as im:
                    Image.new("RGB", im.size, (128, 128, 128)).save(grey)
        out.append(str(grey))
    return tuple(out)


def ece_from_conf(conf, correct, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def entropy(p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return float(-(p * np.log(p)).sum())


def pad_probs(probs, max_k: int) -> np.ndarray:
    """Pad per-item distributions to a common K with zeros, preserving option order."""
    out = np.zeros((len(probs), max_k), dtype=np.float64)
    for i, p in enumerate(probs):
        out[i, :len(p)] = p
    return out


def summarize(probs, preds, truth, confs) -> dict:
    correct = np.asarray([p == t for p, t in zip(preds, truth)], dtype=float)
    return {
        "n": int(correct.size),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [float(x) for x in boot_ci(correct)],
        "ece": ece_from_conf(confs, correct),
        "mean_confidence": float(np.asarray(confs, dtype=float).mean()),
    }


def evaluate(probs_by_pos, preds_by_pos, confs_by_pos, truth, gold_idx, modal_mask,
             modal_k, positions) -> dict:
    p_by_r = np.stack([probs_by_pos[p] for p in positions], axis=1)
    out = {
        "positions": list(positions),
        "flip": modality_flip(p_by_r).tolist(),
        "tv": modality_tv(p_by_r).tolist(),
        "instability": modality_instability(p_by_r),
        "per_position": {},
        "dof": {},
    }
    for i, placement in enumerate(positions):
        summary = summarize(probs_by_pos[placement], preds_by_pos[placement],
                            truth, confs_by_pos[placement])
        marginal = p_by_r[modal_mask, i, :modal_k].mean(axis=0)
        summary["label_marginal_entropy"] = entropy(marginal)
        summary["label_marginal"] = marginal.tolist()
        out["per_position"][placement] = summary

    prefix_idx = p_by_r[:, 0, :].argmax(axis=1)
    for i, placement in enumerate(positions):
        for beta in (0.5, 0.75):
            fused = dual_order_fuse(p_by_r[:, i, :], p_by_r[:, 0, :], beta)
            pred_idx = fused.argmax(axis=1)
            correct = (pred_idx == gold_idx).astype(float)
            out["dof"][f"{placement}_beta{beta}"] = {
                "accuracy": float(correct.mean()),
                "accuracy_ci95": [float(x) for x in boot_ci(correct)],
                "flip_vs_prefix": float((pred_idx != prefix_idx).mean()),
                "tv_vs_prefix": float(0.5 * np.abs(fused - p_by_r[:, 0, :]).sum(axis=1).mean()),
                "ece": ece_from_conf(fused.max(axis=1), correct),
            }
    return out


def run_raw(decider, backend, items, placement: str):
    _set_placement(backend, placement)
    probs, preds, confs = [], [], []
    t0 = time.time()
    for item in items:
        text = item["question"] + IMAGE_SPLIT if placement == "question" else item["question"]
        question = Question.choice(text, item["options"], name="answer")
        decision = decider.decide(state_for(item, placement), [question], level="raw")["answer"]
        probs.append(np.asarray(decision.probs, dtype=np.float64))
        preds.append(decision.argmax)
        confs.append(float(decision.confidence))
    return probs, preds, np.asarray(confs), time.time() - t0


def run_pointwise(backend, items, placement: str):
    _set_placement(backend, placement)
    yes_ids, no_ids = label_id_sets(backend.tokenizer)
    probs, preds = [], []
    t0 = time.time()
    for item in items:
        item2 = dict(item)
        if placement == "middle":
            text = item.get("state", "")
            item2["state"] = (text + "\n" + IMAGE_SPLIT) if text else IMAGE_SPLIT
        elif placement == "question":
            item2["question"] = item["question"] + IMAGE_SPLIT
        _, pi, _ = score_pointwise(backend, item2, yes_ids, no_ids, with_options=False)
        probs.append(np.asarray(pi, dtype=np.float64))
        preds.append(item["options"][int(np.argmax(pi))])
    return probs, preds, time.time() - t0


def section(probs, preds, confs, secs, truth, gold_idx, modal_mask, modal_k,
            positions) -> dict:
    return {
        "probs": {p: probs[p].tolist() for p in positions},
        "preds": {p: preds[p] for p in positions},
        "confidences": {p: confs[p].tolist() for p in positions},
        "seconds": secs,
        "metrics": evaluate(probs, preds, confs, truth, gold_idx, modal_mask, modal_k,
                            positions),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["aokvqa", "scienceqa", "mmmu", "cvbench", "eurosat", "pets"])
    parser.add_argument("--models", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="comma-separated model ids")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--listing", default="caller",
                        choices=["caller", "reversed", "canonical"],
                        help="option listing; the answer text is unchanged")
    parser.add_argument("--positions", default="prefix,middle,suffix")
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/stage2_images"))
    parser.add_argument("--grey-dir", default=None)
    parser.add_argument("--out-dir", default=str(_ROOT / "research2visualtexorder/results"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--grey", action="store_true", help="also run the grey-image control")
    parser.add_argument("--pointwise", action="store_true", help="also run the pointwise control")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--data-root", default="/tmp/anyjev_data")
    parser.add_argument("--smoke", action="store_true", help="limit to 10 items")
    args = parser.parse_args()

    positions = tuple(p.strip() for p in args.positions.split(",") if p.strip())
    if not positions or any(p not in POSITIONS for p in positions):
        raise SystemExit(f"positions must be drawn from {POSITIONS}")
    if "prefix" not in positions:
        raise SystemExit("the metrics use prefix as the reference; include it")

    items = load_items(args.dataset, limit=args.limit, seed=args.seed, image_dir=args.image_dir)
    if args.smoke:
        items = items[:10]
    if not items:
        raise SystemExit("no items loaded")
    if args.listing != "caller":
        for item in items:
            options = list(item["options"])
            item["options"] = options[::-1] if args.listing == "reversed" else sorted(options)
    truth = [item["answer"] for item in items]
    gold_idx = np.asarray([item["options"].index(item["answer"]) for item in items])
    k_counts = Counter(len(item["options"]) for item in items)
    modal_k = k_counts.most_common(1)[0][0]
    max_k = max(len(item["options"]) for item in items)
    modal_mask = np.asarray([len(item["options"]) == modal_k for item in items])
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y-%m-%d")
    grey_dir = Path(args.grey_dir) if args.grey_dir else out_dir / "grey"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    import torch
    import transformers

    import anyjev
    from anyjev.backends.hf_vlm import HFVLMBackend

    print(f"dataset={args.dataset} n={len(items)} k_counts={dict(k_counts)} "
          f"modal_k={modal_k} positions={positions} models={models}", flush=True)

    for model in models:
        batch_size = args.batch_size if args.batch_size is not None else default_batch(model)
        backend = HFVLMBackend(model, device=args.device, dtype=args.dtype,
                               batch_size=batch_size, max_pixels=args.max_pixels)
        decider = Decider(backend, shared_prefix=False, prior="none")
        payload = {
            "dataset": args.dataset,
            "model": model,
            "license": LICENSES.get(args.dataset, "see THIRD_PARTY.md"),
            "n": len(items),
            "k_counts": dict(k_counts),
            "modal_k": modal_k,
            "max_k": max_k,
            "limit": args.limit,
            "seed": args.seed,
            "listing": args.listing,
            "positions": list(positions),
            "smoke": bool(args.smoke),
            "env": {
                "date": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(),
                "anyjev": anyjev.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda": torch.version.cuda,
                "dtype": args.dtype,
                "batch_size": batch_size,
                "max_pixels": args.max_pixels,
                "device": args.device,
            },
            "items": [{"id": it["id"], "answer": it["answer"], "options": it["options"]}
                      for it in items],
        }
        suffix = "" if args.listing == "caller" else f"_{args.listing}"
        out = (Path(args.out) if args.out
               else out_dir / f"{model_slug(model)}_{args.dataset}{suffix}.json")

        probs, preds, confs, secs = {}, {}, {}, {}
        for placement in positions:
            probs[placement], preds[placement], confs[placement], secs[placement] = run_raw(
                decider, backend, items, placement)
            probs[placement] = pad_probs(probs[placement], max_k)
            summary = summarize(probs[placement], preds[placement], truth, confs[placement])
            print(f"{model} raw {placement:6s} acc={summary['accuracy']:.4f} "
                  f"conf={summary['mean_confidence']:.3f} secs={secs[placement]:.1f}", flush=True)
        payload["raw"] = section(probs, preds, confs, secs, truth, gold_idx, modal_mask,
                                 modal_k, positions)
        write_json(out, payload)
        print(f"raw flip={payload['raw']['metrics']['flip']} "
              f"tv={payload['raw']['metrics']['tv']} "
              f"instability={payload['raw']['metrics']['instability']:.4f}", flush=True)

        if args.grey:
            g_probs, g_preds, g_confs, g_secs = {}, {}, {}, {}
            for placement in positions:
                grey = [dict(item, images=grey_images(item, image_dir, grey_dir)) for item in items]
                g_probs[placement], g_preds[placement], g_confs[placement], g_secs[placement] = run_raw(
                    decider, backend, grey, placement)
                g_probs[placement] = pad_probs(g_probs[placement], max_k)
                summary = summarize(g_probs[placement], g_preds[placement], truth, g_confs[placement])
                print(f"{model} grey {placement:6s} acc={summary['accuracy']:.4f} "
                      f"secs={g_secs[placement]:.1f}", flush=True)
            payload["grey"] = section(g_probs, g_preds, g_confs, g_secs, truth, gold_idx,
                                      modal_mask, modal_k, positions)
            write_json(out, payload)
            print(f"grey flip={payload['grey']['metrics']['flip']}", flush=True)

        if args.pointwise:
            p_probs, p_preds, p_confs, p_secs = {}, {}, {}, {}
            for placement in positions:
                p_probs[placement], p_preds[placement], p_secs[placement] = run_pointwise(
                    backend, items, placement)
                p_probs[placement] = pad_probs(p_probs[placement], max_k)
                p_confs[placement] = p_probs[placement].max(axis=1)
                summary = summarize(p_probs[placement], p_preds[placement], truth, p_confs[placement])
                print(f"{model} pointwise {placement:6s} acc={summary['accuracy']:.4f} "
                      f"secs={p_secs[placement]:.1f}", flush=True)
            payload["pointwise"] = section(p_probs, p_preds, p_confs, p_secs, truth, gold_idx,
                                           modal_mask, modal_k, positions)
            write_json(out, payload)
            print(f"pointwise flip={payload['pointwise']['metrics']['flip']}", flush=True)

        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
