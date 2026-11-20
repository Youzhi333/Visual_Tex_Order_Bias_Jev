"""Modality-order metrics for a one-prefill label readout.

These are the layer above the backend, next to `anyjev.calibrate.permute`:
they compare distributions that were read under different image positions.
`probs` are aligned by option before any comparison. `listing[j]` is the index
of the option shown at slot `j`, so `align_positions` converts slot-order
probabilities into option-order probabilities. Batch functions take the batch
axis first.

Nothing here samples tokens or touches a model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-12


def align_positions(probs, listing: Sequence[int]) -> np.ndarray:
    """Convert slot-order probabilities to option-order probabilities.

    `probs[..., j]` is the probability of the option shown at slot `j`;
    `listing[j]` is that option's index. The result `out[..., i]` is the
    probability of option `i`, for any number of leading batch axes.
    """
    p = np.asarray(probs, dtype=np.float64)
    listing = list(listing)
    if p.shape[-1] != len(listing):
        raise ValueError(f"probs has {p.shape[-1]} slots but listing has {len(listing)} options")
    out = np.zeros_like(p)
    for j, i in enumerate(listing):
        out[..., i] = p[..., j]
    return out


def _as_batch3(p) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[None]
    if p.ndim != 3:
        raise ValueError("expected [N, R, K] or [R, K]")
    return p


def modality_flip(p_by_r, reference: int = 0) -> np.ndarray:
    """Fraction of items whose argmax differs from the reference position.

    `p_by_r`: [N, R, K] aligned by option. Returns `F(r)` for every `r`.
    """
    p = _as_batch3(p_by_r)
    winners = p.argmax(axis=-1)
    return (winners != winners[:, [reference]]).mean(axis=0)


def modality_tv(p_by_r, reference: int = 0) -> np.ndarray:
    """Mean total-variation distance from the reference position.

    `p_by_r`: [N, R, K] aligned by option. Returns `T(r)` for every `r`, in [0, 1].
    """
    p = _as_batch3(p_by_r).astype(np.float64)
    return 0.5 * np.abs(p - p[:, [reference], :]).sum(axis=-1).mean(axis=0)


def modality_instability(p_by_r) -> float:
    """Mean argmax flip over all unordered pairs of positions.

    One number per batch; 0 means every position agrees on every item.
    """
    p = _as_batch3(p_by_r)
    winners = p.argmax(axis=-1)
    values = []
    for r in range(p.shape[1]):
        for s in range(r + 1, p.shape[1]):
            values.append(float((winners[:, r] != winners[:, s]).mean()))
    return float(np.mean(values)) if values else 0.0


def flip_between(p_a, p_b) -> float:
    """Fraction of items whose argmax differs between two aligned distributions."""
    a = np.asarray(p_a)
    b = np.asarray(p_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float((a.argmax(axis=-1) != b.argmax(axis=-1)).mean())


def tv_between(p_a, p_b) -> float:
    """Mean total-variation distance between two aligned distributions."""
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(0.5 * np.abs(a - b).sum(axis=-1).mean())


def logmean(p_by_order) -> np.ndarray:
    """Average distributions in log space and renormalize.

    `p_by_order` has an order axis at `-2`: [..., R, K] -> [..., K]. This is the
    combining rule that cancels an additive position effect exactly.
    """
    p = np.asarray(p_by_order, dtype=np.float64)
    z = np.log(np.clip(p, EPS, None)).mean(axis=-2)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def dual_order_fuse(p_student, p_teacher, beta: float = 0.5) -> np.ndarray:
    """Fuse two aligned distributions in log space.

    `beta` is the weight on the teacher: 0 returns the student, 1 returns the
    teacher, 0.5 is the log-mean. This is the readout-time analogue of a
    prefix-teacher / order-student fusion; no decoding is involved.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    s = np.clip(np.asarray(p_student, dtype=np.float64), EPS, None)
    t = np.clip(np.asarray(p_teacher, dtype=np.float64), EPS, None)
    if s.shape != t.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {t.shape}")
    z = (1.0 - beta) * np.log(s) + beta * np.log(t)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
