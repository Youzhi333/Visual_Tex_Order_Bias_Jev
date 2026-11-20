"""Print LaTeX rows for the paper tables from committed result JSONs.

    python research2visualtexorder/make_tables.py research2visualtexorder/results/2026-11-12

Per dataset, the row is the mean over the model files that exist. Only reads
files; no model and no GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = ("aokvqa", "eurosat", "scienceqa", "cvbench", "pets", "mmmu")
LABELS = {
    "aokvqa": "A-OKVQA",
    "eurosat": "EuroSAT",
    "scienceqa": "ScienceQA",
    "cvbench": "CV-Bench",
    "pets": "Pets",
    "mmmu": "MMMU",
}
POSITIONS = ("prefix", "middle", "question", "suffix")
DEFAULT_MODELS = (
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
)
SHORT = {
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Idefics3-8B-Llama3": "Idefics3-8B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-NeXT-7B",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def values(metrics, key, positions):
    """Map a run's positions onto canonical names by order.

    LLaVA runs use ``splice_middle``/``splice_question``/``splice_suffix`` and
    token runs use ``token0.25`` etc.; the order is always prefix to suffix.
    """
    if len(metrics["positions"]) != len(positions):
        raise ValueError(f"expected {len(positions)} positions, "
                         f"got {metrics['positions']}")
    return dict(zip(positions, metrics[key]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--per-model", action="store_true",
                        help="also print one row per model and dataset")
    parser.add_argument("--full", action="store_true",
                        help="print full-set rows instead of the 300-item grid")
    args = parser.parse_args()
    root = Path(args.root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows, sources, details = [], [], []
    for dataset in DATASETS:
        per_model = []
        for model in models:
            path = root / f"{model}_{dataset}_4pos.json"
            if not path.exists():
                continue
            data = load(path)
            metrics = data["raw"]["metrics"]
            flip = values(metrics, "flip", POSITIONS)
            tv = values(metrics, "tv", POSITIONS)
            acc = {canon: metrics["per_position"][actual]["accuracy"]
                   for canon, actual in zip(POSITIONS, metrics["positions"])}
            per_model.append((flip, tv, acc))
            details.append((model, dataset, flip, tv, acc))
            sources.append(path.name)
        if not per_model:
            continue
        rows.append((
            dataset,
            {p: float(np.mean([m[0][p] for m in per_model])) for p in POSITIONS},
            {p: float(np.mean([m[1][p] for m in per_model])) for p in POSITIONS},
            {p: float(np.mean([m[2][p] for m in per_model])) for p in POSITIONS},
        ))
    if not rows:
        raise SystemExit(f"no model files under {root}")
    print(f"% models seen: {len(set(s.split('_')[0] for s in sources))}; "
          f"per-dataset mean over available model files")

    print("% Table flip/TV")
    for dataset, flip, tv, _ in rows:
        print(f"{LABELS[dataset]} & {flip['middle']:.3f} & {flip['question']:.3f} & "
              f"{flip['suffix']:.3f} & {tv['middle']:.3f} & {tv['question']:.3f} & "
              f"{tv['suffix']:.3f} \\\\")
    mean_flip = np.mean([[flip[p] for p in POSITIONS] for _, flip, _, _ in rows], axis=0)
    mean_tv = np.mean([[tv[p] for p in POSITIONS] for _, _, tv, _ in rows], axis=0)
    print(f"Mean & {mean_flip[1]:.3f} & {mean_flip[2]:.3f} & {mean_flip[3]:.3f} & "
          f"{mean_tv[1]:.3f} & {mean_tv[2]:.3f} & {mean_tv[3]:.3f} \\\\")

    print("\n% Table accuracy")
    for dataset, _, _, acc in rows:
        print(f"{LABELS[dataset]} & {acc['prefix']:.3f} & {acc['middle']:.3f} & "
              f"{acc['question']:.3f} & {acc['suffix']:.3f} \\\\")
    mean_acc = np.mean([[acc[p] for p in POSITIONS] for _, _, _, acc in rows], axis=0)
    print(f"Mean & {mean_acc[0]:.3f} & {mean_acc[1]:.3f} & {mean_acc[2]:.3f} & "
          f"{mean_acc[3]:.3f} \\\\")

    print("\n% Sources")
    for name in sources:
        print(f"% {name}")

    if args.per_model:
        print("\n% Per-model rows")
        for model, dataset, flip, tv, acc in details:
            print(f"{SHORT.get(model, model)} & {LABELS[dataset]} & "
                  f"{acc['prefix']:.3f} & {acc['middle']:.3f} & "
                  f"{acc['question']:.3f} & {acc['suffix']:.3f} & "
                  f"{flip['suffix']:.3f} \\\\")

    print("\n% Pilot controls (Qwen3-VL-4B; A-OKVQA, EuroSAT)")
    pilot_model = "Qwen3-VL-4B-Instruct"
    for dataset in ("aokvqa", "eurosat"):
        path = root / f"{pilot_model}_{dataset}.json"
        if not path.exists():
            continue
        data = load(path)
        for section in ("grey", "pointwise"):
            if section not in data:
                continue
            metrics = data[section]["metrics"]
            flip = values(metrics, "flip", metrics["positions"])
            tv = values(metrics, "tv", metrics["positions"])
            acc = {p: metrics["per_position"][p]["accuracy"] for p in metrics["positions"]}
            suffix = metrics["positions"][-1]
            print(f"% {dataset} {section}: suffix flip={flip[suffix]:.3f} "
                  f"tv={tv[suffix]:.3f} acc={acc[suffix]:.3f} "
                  f"({path.name})")
        dof = data["raw"]["metrics"]["dof"]
        for key in ("middle_beta0.75", "suffix_beta0.75"):
            if key in dof:
                value = dof[key]
                print(f"% {dataset} raw {key}: acc={value['accuracy']:.3f} "
                      f"flip={value['flip_vs_prefix']:.3f} "
                      f"tv={value['tv_vs_prefix']:.3f} ({path.name})")

    if args.full:
        print("\n% Full sets")
        for model in models:
            for dataset in ("aokvqa", "eurosat", "pets"):
                path = root / f"{model}_{dataset}_full.json"
                if not path.exists():
                    continue
                data = load(path)
                metrics = data["raw"]["metrics"]
                flip = values(metrics, "flip", POSITIONS)
                tv = values(metrics, "tv", POSITIONS)
                acc = {canon: metrics["per_position"][actual]["accuracy"]
                       for canon, actual in zip(POSITIONS, metrics["positions"])}
                print(f"{SHORT.get(model, model)} & {LABELS[dataset]} & "
                      f"{data['n']:,} & {acc['prefix']:.3f} & {acc['suffix']:.3f} & "
                      f"{flip['suffix']:.3f} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
