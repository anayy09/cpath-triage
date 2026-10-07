"""
src/triage/weighted.py

Inverse-probability-weighted versions of the routing metrics, for samples drawn
with known, unequal inclusion probabilities.

The 1,800-patch consistency subsets were not drawn at random from their split:
_select_patches in scripts/run_consistency.py takes up to 100 patches the
full-scale V3 run got right and 100 it got wrong from each class. That changes
both the outcome prevalence and the class mix, so an AUC computed on the subset
describes the subset, not the cohort. Because the draw is uniform inside each
(class, full-scale correctness) cell, every patch has an exactly known inclusion
probability pi, and weighting by 1/pi recovers the cohort-level quantity.

The weighted selective-accuracy AUC is defined on the Horvitz-Thompson
pseudo-population in which patch i stands for w_i = 1/pi_i cohort patches. The
routing budget is a fraction of that weighted mass, and a group of tied
confidence values straddling the cut is included fractionally, which is the
expectation over uniformly random tie orders in the replicated population. With
every weight equal to 1 this is algebraically the same quantity as the closed
form in src/triage/router.py, and tests/test_weighted.py holds the two to 1e-12.

Public API:
    weighted_expected_curve(confidences, correct, weights, n_points) -> dict
    weighted_auc(confidences, correct, weights) -> float
    weighted_accuracy(correct, weights) -> float
    weighted_auroc(scores, correct, weights) -> float
    stratified_bootstrap_indices(strata, n_boot, seed) -> np.ndarray
    simple_bootstrap_indices(n, n_boot, seed) -> np.ndarray
    summarize_diffs(diffs, point) -> dict
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from src.triage.router import _auc_from_curve


def _budget_masses(total: float, n_points: int) -> list[float]:
    # router._confirm_sizes rounds (1 - b) * N to a whole patch count. Rounding
    # the target mass the same way keeps the weights=1 case identical to it, and
    # on a pseudo-population of roughly 10,000 units it moves nothing that shows.
    budgets = np.linspace(0.0, 1.0, n_points)
    return [min(float(round((1.0 - b) * total)), total) for b in budgets]


def weighted_expected_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    n_points: int = 101,
) -> dict:
    """
    Expected auto-confirm accuracy against routing budget on a weighted sample.

    Args:
        confidences: 1-D float, higher means more certain.
        correct:     1-D bool/int outcome the signal routes.
        weights:     1-D positive float, 1/pi per patch.
        n_points:    budgets evenly spaced on [0, 1], as in the router.

    Returns:
        Dict with budgets, auto_confirm_acc (NaN where nothing is confirmed),
        auc (same trapezoid convention as the router) and total_weight.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (len(conf) == len(corr) == len(w)):
        raise ValueError("confidences, correct and weights must have equal length")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")

    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    _, starts = np.unique(-sorted_conf, return_index=True)
    group_mass = np.add.reduceat(w[order], starts)
    group_correct = np.add.reduceat((w * corr)[order], starts)
    cum_mass = np.concatenate([[0.0], np.cumsum(group_mass)])
    cum_correct = np.concatenate([[0.0], np.cumsum(group_correct)])
    total = float(cum_mass[-1])
    m = len(group_mass)

    targets = np.asarray(_budget_masses(total, n_points))
    j = np.clip(np.searchsorted(cum_mass, targets, side="left"), 1, m)
    slots = targets - cum_mass[j - 1]
    rate = group_correct[j - 1] / group_mass[j - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = (cum_correct[j - 1] + slots * rate) / targets
    acc = [float(v) if t > 0 else float("nan") for v, t in zip(vals, targets)]

    return {
        "budgets": np.linspace(0.0, 1.0, n_points).tolist(),
        "auto_confirm_acc": acc,
        "auc": _auc_from_curve(acc),
        "total_weight": total,
    }


def weighted_auc(confidences: np.ndarray, correct: np.ndarray, weights: np.ndarray) -> float:
    """Selective-accuracy AUC on the weighted pseudo-population."""
    return float(weighted_expected_curve(confidences, correct, weights)["auc"])


def weighted_accuracy(correct: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted accuracy, which is also the AUC of random routing.

    Under a uniformly random order every prefix has expected accuracy equal to
    the overall rate, so the random reference needs no Monte Carlo.
    """
    w = np.asarray(weights, dtype=float)
    return float((w * np.asarray(correct, dtype=float)).sum() / w.sum())


def weighted_auroc(scores: np.ndarray, correct: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Mann-Whitney AUROC of score for correctness, ties at one half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=bool)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    _, starts = np.unique(s_sorted, return_index=True)
    pos = np.add.reduceat(np.where(y[order], w[order], 0.0), starts)
    neg = np.add.reduceat(np.where(y[order], 0.0, w[order]), starts)
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    total_pos, total_neg = pos.sum(), neg.sum()
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    return float(((pos * neg_below).sum() + 0.5 * (pos * neg).sum()) / (total_pos * total_neg))


def simple_bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """
    Paired resampling of patches, drawn the way every published interval was.

    The scripts behind Tables 10 and 11 seed one generator and call
    integers(0, n, n) per resample, so the same seed reproduces their draws.
    """
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, size=n) for _ in range(n_boot)])


def stratified_bootstrap_indices(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """
    Resample with replacement inside each stratum, keeping stratum sizes fixed.

    The selection was a stratified draw, so resampling inside the strata mirrors
    the design and holds every inclusion weight constant across resamples.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty((n_boot, len(strata)), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
    return out


def summarize_diffs(diffs: np.ndarray, point: float) -> dict:
    """
    Interval, standard error and a normal-approximation p for one contrast.

    The point estimate is the plug-in difference of the two full-sample values,
    rounded once, as in every other table; the bootstrap supplies the interval
    and the standard error. The p value is two-sided from point/SE and exists so
    a Holm adjustment can be applied across the family of headline contrasts.
    """
    d = np.asarray(diffs, dtype=float)
    lo, hi = np.percentile(d, [2.5, 97.5])
    se = float(d.std(ddof=1))
    p = float(2.0 * norm.sf(abs(point) / se)) if se > 0 else float("nan")
    return {
        "gap_point": round(float(point), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "point_inside_ci": bool(lo <= round(float(point), 4) <= hi),
        "bootstrap_mean": round(float(d.mean()), 4),
        "se": round(se, 4),
        "p_normal": float(f"{p:.3g}") if not math.isnan(p) else None,
        "n_boot": len(d),
        # Unrounded, so a table printing three decimals rounds once rather than twice.
        "exact": {"gap_point": float(point), "ci": [float(lo), float(hi)]},
    }
