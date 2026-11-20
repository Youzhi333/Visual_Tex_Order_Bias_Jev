"""Pointwise visual readout: score each option alone, then softmax over options.

For every option o the model sees the state, the question and that option only:

    f(x, q, o) = log P(yes | x, q, o) - log P(no | x, q, o)
    pi(o)      = softmax_o f(x, q, o)

No option is seen next to another, so pi is invariant to the listing order by
construction. A check scores N items a second time with the options reversed and
reports the maximum difference.

Two batch-prior variants are computed offline from the same scores:
  * per-question: the running mean of pi for the same Question key (so it only
    activates where the question and options repeat, e.g. Pets and EuroSAT);
  * pooled: the mean pi of each option *text* over every item on which it appears
    (labels appearing fewer than --min-prior-n times are left uncorrected).

    python scripts/eval_visual_pointwise.py --dataset aokvqa \
        --models Qwen/Qwen3-VL-4B-Instruct --limit 300
"""
import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from eval_visual_bias import default_batch, model_slug, summarise

from anyjev import Decision, Question, VisualState
from anyjev.state import render_state

_ROOT = Path(__file__).resolve().parents[1]
POINTWISE_SYSTEM = (
    "You are a decision function. You will be given a state, a question and one option. "
    "Reply with yes or no only: no words, no punctuation, no explanation."
)


def build_pointwise_user(state_text: str, question: str, option: str,
                         options=None) -> str:
    """The option is scored alone; optional context lists the full option set."""
    prefix = f"{state_text}\n\nQuestion: {question}\n"
    if options:
        prefix += "Options:\n" + "\n".join(f"- {o}" for o in options) + "\n"
    return (prefix + f"Option: {option}\n"
            "Is this option the correct answer? Answer yes or no.")


def logsumexp(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    m = float(v.max())
    return m + float(np.log(np.exp(v - m).sum()))


def pointwise_probs(logodds) -> np.ndarray:
    z = np.asarray(logodds, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def apply_prob_prior(pi, prior, strength) -> np.ndarray:
    z = np.asarray(pi, dtype=float) / np.power(np.clip(np.asarray(prior, float), 1e-12, None),
                                               strength)
    return z / z.sum()


def label_id_sets(tokenizer, yes="yes", no="no"):
    """Single-token ids for yes/no, including capitalised and leading-space forms."""
    def ids_for(word):
        ids = set()
        for cand in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
        if not ids:
            raise RuntimeError(f"no single token for {word!r}")
        return sorted(ids)

    yes_ids, no_ids = ids_for(yes), ids_for(no)
    if set(yes_ids) & set(no_ids):
        raise RuntimeError("yes and no share a token id")
    return yes_ids, no_ids


def score_pointwise(backend, item, yes_ids, no_ids, with_options=False):
    """Return (f per option, pi, mean yes/no mass) for one item."""
    state = render_state(VisualState(text=item.get("state", ""), images=tuple(item["images"])))
    prompts = []
    for option in item["options"]:
        spec = SimpleNamespace(system=POINTWISE_SYSTEM,
                               user=build_pointwise_user(state, item["question"], option,
                                                         item["options"] if with_options else None))
        pre, suf = backend.render_chat_parts(spec)
        prompts.append(pre + suf)
    lps = backend.next_token_logprobs(prompts, [yes_ids + no_ids] * len(prompts))
    f, mass = [], []
    for lp in lps:
        ly = logsumexp(lp[:len(yes_ids)])
        ln = logsumexp(lp[len(yes_ids):])
        f.append(ly - ln)
        mass.append(float(np.exp(ly) + np.exp(ln)))
    return np.asarray(f), pointwise_probs(f), float(np.mean(mass))


def run_priors(pis, keys, option_lists, strength, min_n):
    """Per-question running prior and pooled per-label prior, both offline."""
    per_key = {}
    batch, active = [], []
    for pi, key in zip(pis, keys):
        s, c = per_key.get(key, (np.zeros_like(pi), 0))
        s, c = s + pi, c + 1
        per_key[key] = (s, c)
        use = c >= min_n
        active.append(bool(use))
        batch.append(apply_prob_prior(pi, s / c, strength) if use else pi.copy())

    sums, counts = defaultdict(lambda: 0.0), defaultdict(int)
    for pi, opts in zip(pis, option_lists):
        for o, p in zip(opts, pi):
            sums[o] += float(p)
            counts[o] += 1
    prior = {o: sums[o] / counts[o] for o in sums if counts[o] >= min_n}
    pooled = []
    for pi, opts in zip(pis, option_lists):
        q = np.asarray([prior.get(o, 1.0) for o in opts], dtype=float)
        # leave uncorrected labels exactly as they are: a prior of 1 is a no-op
        pooled.append(apply_prob_prior(pi, q, strength))
    return batch, active, pooled, prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["aokvqa", "scienceqa", "mmmu", "cvbench", "eurosat", "pets",
                             "aokvqa-k8", "aokvqa-k16"])
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--min-prior-n", type=int, default=8)
    ap.add_argument("--with-options", action="store_true",
                    help="show the full option list as context while scoring one option")
    ap.add_argument("--check-invariance", type=int, default=5,
                    help="items rescored with the options reversed (0 = skip)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--data-root", default="/tmp/anyjev_data")
    ap.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/stage2_images"))
    ap.add_argument("--out-dir", default=str(_ROOT / "research/visual_bias/pilot_prior"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from visual_datasets import load_items

    import anyjev
    from anyjev.backends.hf_vlm import HFVLMBackend

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset, limit=args.limit, seed=args.seed,
                       image_dir=str(_ROOT / "research/visual_bias/stage2_images"),
                       data_root=args.data_root)
    truth = [it["answer"] for it in items]
    print(f"dataset={args.dataset} items={len(items)} readout=pointwise models={models}",
          flush=True)

    for model in models:
        batch_size = args.batch_size if args.batch_size is not None else default_batch(model)
        backend = HFVLMBackend(model, device=args.device, dtype=args.dtype,
                               batch_size=batch_size, max_pixels=args.max_pixels)
        yes_ids, no_ids = label_id_sets(backend.tokenizer)

        t0 = time.time()
        pis, scores, masses, keys, opts_all = [], [], [], [], []
        questions = []
        for it in items:
            q = Question.choice(it["question"], it["options"], name="answer")
            f, pi, mass = score_pointwise(backend, it, yes_ids, no_ids, args.with_options)
            pis.append(pi)
            scores.append(f)
            masses.append(mass)
            keys.append(q.key)
            opts_all.append(list(it["options"]))
            questions.append(q)

        batch_pis, batch_active, pooled_pis, pooled_prior = run_priors(
            pis, keys, opts_all, args.strength, args.min_prior_n)

        def decisions(plist, methods):
            out = []
            for q, p, method, mass in zip(questions, plist, methods, masses):
                out.append(Decision(q, np.asarray(p), "L0",
                                    {"readout": "pointwise", "prior_method": method,
                                     "answer_mass": mass}))
            return out

        variants = {
            "pointwise": decisions(pis, ["none"] * len(pis)),
            "pointwise_batch": decisions(
                batch_pis, ["batch" if a else "none" for a in batch_active]),
            "pointwise_pooled": decisions(pooled_pis, ["pooled"] * len(pis)),
        }

        inv = None
        if args.check_invariance:
            diffs = []
            for idx, it in enumerate(items[:args.check_invariance]):
                rev = dict(it)
                rev["options"] = list(it["options"])[::-1]
                _, pi_rev, _ = score_pointwise(backend, rev, yes_ids, no_ids, args.with_options)
                diffs.append(float(np.abs(np.asarray(pis[idx]) - pi_rev[::-1]).max()))
            inv = {"n": len(diffs),
                   "max_abs_diff_vs_reversed": float(max(diffs)) if diffs else None}

        result = {
            "dataset": args.dataset, "model": model,
            "readout": "pointwise_context" if args.with_options else "pointwise",
            "n": len(items), "limit": args.limit, "seed": args.seed,
            "env": {
                "date": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(), "anyjev": anyjev.__version__,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "dtype": args.dtype, "batch_size": batch_size, "max_pixels": args.max_pixels,
                "strength": args.strength, "min_prior_n": args.min_prior_n,
                "system": POINTWISE_SYSTEM, "yes_ids": yes_ids, "no_ids": no_ids,
            },
            "metrics": {name: {**summarise(d, truth), "seconds": time.time() - t0}
                        for name, d in variants.items()},
            "mean_yes_no_mass": float(np.mean(masses)),
            "batch_prior_active_items": int(sum(batch_active)),
            "invariance": inv,
            "pooled_prior_labels": len(pooled_prior),
            "pooled_prior_values": {k: float(v) for k, v in sorted(
                pooled_prior.items(), key=lambda kv: -kv[1])[:50]},
            "per_item": [
                {"id": it["id"], "k": len(it["options"]), "answer": it["answer"],
                 "options": list(it["options"]), "f": [float(x) for x in f],
                 "pointwise": [float(x) for x in variants["pointwise"][i].probs],
                 "pointwise_batch": [float(x) for x in variants["pointwise_batch"][i].probs],
                 "pointwise_pooled": [float(x) for x in variants["pointwise_pooled"][i].probs],
                 "batch_prior_active": batch_active[i],
                 "pooled_labels_corrected": sum(o in pooled_prior for o in it["options"]),
                 "yes_no_mass": masses[i]}
                for i, (it, f) in enumerate(zip(items, scores))
            ],
        }
        out = Path(args.out) if args.out and len(models) == 1 else \
            out_dir / f"{args.dataset}_{model_slug(model)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        for name, met in result["metrics"].items():
            print(f"{model} {name} {met}", flush=True)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
