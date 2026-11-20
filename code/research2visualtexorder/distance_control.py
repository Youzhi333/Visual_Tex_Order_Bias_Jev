"""Distance-controlled analysis of the token-level sweep.

    python research2visualtexorder/distance_control.py

Encodes each token-level prompt (no model forward) to measure the distance from
the last image token to the answer, then fits a cluster-robust logistic model
of flip on distance plus position dummies. If position dummies remain
significant after distance, the categorical slot has an effect beyond distance.
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
from anyjev.state import render_state  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
RESULTS = _ROOT / "research2visualtexorder/results/2026-11-12"


def distance(backend, item, fraction):
    backend.image_placement = "token"
    backend.image_position = fraction
    question = Question.choice(item["question"], item["options"], name="answer")
    state = VisualState(text=item.get("state", ""), images=tuple(item["images"]))
    spec = build_prompt(render_state(state), question, list(range(question.k)))
    prompt, marker = backend.render_chat_parts(spec)
    key = marker.rsplit(_ID_MARK, 1)[1]
    prompt, images = backend._pending.pop(key)
    enc = backend._inputs([prompt], [list(images)])
    image_token = getattr(backend.processor, "image_token", None) or "<|image_pad|>"
    image_id = backend.tokenizer.convert_tokens_to_ids(image_token)
    ids = enc["input_ids"][0].tolist()
    positions = [i for i, t in enumerate(ids) if t == image_id]
    return int(len(ids) - 1 - positions[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aokvqa")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--image-dir", default=str(_ROOT / "research/visual_bias/images"))
    parser.add_argument("--out", default=str(RESULTS / "analysis_distance_control.json"))
    args = parser.parse_args()
    path = RESULTS / f"{MODEL.split('/')[-1]}_{args.dataset}_token.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["raw"]["metrics"]["positions"]
    fractions = {"prefix": 0.0, "token0.25": 0.25, "token0.5": 0.5,
                 "token0.75": 0.75, "suffix": 1.0}
    items = load_items(args.dataset, limit=args.limit, seed=0,
                       image_dir=args.image_dir)
    backend = HFVLMBackend(MODEL, device="cuda", dtype="bfloat16", batch_size=1,
                           max_pixels=448 * 448)
    prefix_pred = data["raw"]["preds"][names[0]]
    X, y, groups, labels = [], [], [], []
    for i, item in enumerate(items):
        for name in names[1:]:
            d = distance(backend, item, fractions[name])
            pred = data["raw"]["preds"][name][i]
            X.append(d)
            y.append(int(pred != prefix_pred[i]))
            groups.append(i)
            labels.append(name)
    del backend
    import torch

    torch.cuda.empty_cache()
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    dummies = np.zeros((len(y), len(names) - 2))
    for row, name in enumerate(labels):
        if name in names[1:-1]:
            dummies[row, names[1:-1].index(name)] = 1.0
    design = np.column_stack([X, dummies, np.ones(len(y))])
    model = sm.Logit(y, design).fit(disp=0, cov_type="cluster",
                                    cov_kwds={"groups": np.asarray(groups)})
    names_out = ["distance"] + [f"{n}_vs_suffix" for n in names[1:-1]] + ["intercept"]
    out = {"dataset": args.dataset, "model": MODEL, "positions": names,
           "coefficients": {n: {"coef": float(model.params[i]),
                                "se": float(model.bse[i]),
                                "p": float(model.pvalues[i])}
                            for i, n in enumerate(names_out)},
           "n_rows": int(len(y)), "n_items": int(len(items)),
           "note": "position dummies are relative to suffix; cluster-robust by item"}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for i, n in enumerate(names_out):
        print(f"{n:20s} coef={model.params[i]:+.4f} se={model.bse[i]:.4f} "
              f"p={model.pvalues[i]:.3g}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
