"""Per-cell readout metrics shared by the aggregate tables.

Every readout is computed from one committed _4pos.json file: the four
placement distributions, their subsets and dual-order fusion. Each value is
(accuracy, flip relative to prefix, ECE).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_stats import acc_of, ece_of, logmean, pred_of

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "2026-11-12"
READOUTS = (
    ("Prefix", "prefix"),
    ("Pre-question", "pre_q"),
    ("Pre-options", "pre_o"),
    ("Suffix", "suffix"),
    ("All-position mean", "all_mean"),
    ("All-position log-mean", "all_logmean"),
    ("Prefix+suffix mean", "ps"),
    ("P+Q+O mean", "p_q_o"),
    ("Q+O mean", "q_o"),
    ("P+Q mean", "p_q"),
    ("P+O mean", "p_o"),
    ("M+M mean", "m_m"),
    ("Dual-order, $\\beta{=}0.5$", "dof50"),
    ("Dual-order, $\\beta{=}0.75$", "dof75"),
)


def _softmax_log(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def cell(slug: str, dataset: str) -> dict[str, tuple[float, float, float]]:
    data = json.loads((RESULTS / f"{slug}_{dataset}_4pos.json").read_text(encoding="utf-8"))
    names = data["raw"]["metrics"]["positions"]
    items = data["items"]
    truth = [it["answer"] for it in items]
    p = np.stack([np.asarray(data["raw"]["probs"][n], dtype=float) for n in names], axis=1)
    logp = np.log(np.clip(p, 1e-12, None))
    prefix_pred = pred_of(p[:, 0], items)
    all_logmean = logmean(p.transpose(1, 0, 2))
    probs = {
        "prefix": p[:, 0],
        "pre_q": p[:, 1],
        "pre_o": p[:, 2],
        "suffix": p[:, 3],
        "all_mean": p.mean(axis=1),
        "all_logmean": all_logmean,
        "ps": (p[:, 0] + p[:, 3]) / 2.0,
        "p_q_o": p[:, :3].mean(axis=1),
        "q_o": p[:, 1:3].mean(axis=1),
        "p_q": p[:, :2].mean(axis=1),
        "p_o": p[:, [0, 2]].mean(axis=1),
        "m_m": (p.mean(axis=1) + all_logmean) / 2.0,
        "dof50": _softmax_log(0.5 * (logp[:, 0] + logp[:, 3])),
        "dof75": _softmax_log(0.75 * logp[:, 0] + 0.25 * logp[:, 3]),
    }
    out = {}
    for _, key in READOUTS:
        pred = pred_of(probs[key], items)
        correct = acc_of(probs[key], truth, items)
        flip = float(np.mean([a != b for a, b in zip(pred, prefix_pred)]))
        out[key] = (float(correct.mean()), flip, ece_of(probs[key], correct))
    return out


def pointwise_cell(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["pointwise"]["metrics"]
    suffix = data["positions"][-1]
    return (metrics["per_position"][suffix]["accuracy"], metrics["flip"][-1],
            metrics["per_position"][suffix]["ece"])


def pointwise_metrics(slug: str, dataset: str) -> dict[str, float] | None:
    """Pointwise prefix/suffix accuracy, ECE, and flips against raw prefix."""
    pw_path = RESULTS / f"{slug}_{dataset}_pointwise.json"
    raw_path = RESULTS / f"{slug}_{dataset}_4pos.json"
    if not pw_path.exists():
        return None
    data = json.loads(pw_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if "pointwise" not in data:
        return None
    metrics = data["pointwise"]["metrics"]
    names = data["positions"]
    pre, suf = names[0], names[-1]
    raw_names = raw["raw"]["metrics"]["positions"]
    raw_pre = raw["raw"]["preds"][raw_names[0]]
    pre_pred = data["pointwise"]["preds"][pre]
    suf_pred = data["pointwise"]["preds"][suf]
    n = len(raw_pre)
    return {
        "pre_acc": metrics["per_position"][pre]["accuracy"],
        "pre_ece": metrics["per_position"][pre]["ece"],
        "pre_flip": sum(a != b for a, b in zip(pre_pred, raw_pre)) / n,
        "suf_acc": metrics["per_position"][suf]["accuracy"],
        "suf_ece": metrics["per_position"][suf]["ece"],
        "suf_flip": sum(a != b for a, b in zip(suf_pred, raw_pre)) / n,
        "suf_flip_within": metrics["flip"][-1],
    }
