"""
src/triage/router.py

Uncertainty-aware routing policy for the cpath-triage pipeline.

The routing policy works as follows:
  Given N patch predictions with associated confidence scores:
  1. Rank patches by confidence (highest = most certain).
  2. Auto-confirm the top (1 - budget) fraction (high-confidence cases).
  3. Route the bottom budget fraction to the simulated specialist queue.

Evaluation curve (risk-coverage / accuracy-rejection curve):
  For routing budgets from 0 to 1, report accuracy on the auto-confirmed set.
  A good routing policy shows increasing auto-confirm accuracy as budget grows --
  the specialist handles progressively harder cases while routine cases are cleared.

Three policies are compared:
  - calibrated: route by calibrated uncertainty (1 - calibrated_confidence)
  - raw:        route by raw uncertainty (1 - raw_confidence)
  - random:     route a random fraction (averaged over multiple seeds)

Public API:
    risk_coverage_curve(confidences, correct, n_points) -> dict
    random_routing_curve(correct, n_points, n_trials, seed) -> dict
    operating_point(confidences, correct, budget) -> dict
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_points: int = 101,
) -> dict:
    """
    Compute the accuracy-vs-routing-budget curve for a confidence-based policy.

    At each budget b, the policy routes the b*N lowest-confidence patches to the
    specialist and auto-confirms the (1-b)*N highest-confidence patches.

    Args:
        confidences: 1-D float array of confidence scores (higher = more certain).
        correct:     1-D bool/int array (1 if prediction is correct).
        n_points:    Number of evenly-spaced budget values in [0, 1].

    Returns:
        Dict with keys:
            budgets:              list of float, routing fraction
            auto_confirm_acc:     list of float (NaN at budget=1.0 where N=0)
            n_auto_confirmed:     list of int
            auc:                  float, area under the curve (trapezoid)
    """
    n = len(confidences)
    budgets = np.linspace(0.0, 1.0, n_points)

    # Sort descending by confidence: index 0 = most confident
    order = np.argsort(confidences)[::-1]
    sorted_correct = correct[order].astype(float)

    acc_list: list[float] = []
    n_auto_list: list[int] = []

    for b in budgets:
        n_confirm = int(round((1.0 - b) * n))
        if n_confirm == 0:
            acc_list.append(float("nan"))
        else:
            acc_list.append(float(sorted_correct[:n_confirm].mean()))
        n_auto_list.append(n_confirm)

    # AUC over the non-NaN prefix (budget < 1.0)
    valid = [a for a in acc_list if not (isinstance(a, float) and a != a)]
    auc = float(np.trapz(valid, dx=1.0 / max(len(valid) - 1, 1))) if len(valid) > 1 else float("nan")

    return {
        "budgets": budgets.tolist(),
        "auto_confirm_acc": acc_list,
        "n_auto_confirmed": n_auto_list,
        "auc": auc,
    }


def random_routing_curve(
    correct: np.ndarray,
    n_points: int = 101,
    n_trials: int = 30,
    seed: int = 42,
) -> dict:
    """
    Random routing baseline: route a uniformly random subset to the specialist.

    Averaged over n_trials random orderings to smooth out variance.

    Args:
        correct:   1-D bool/int array.
        n_points:  Number of budget values.
        n_trials:  Number of random permutations to average over.
        seed:      Base random seed.

    Returns:
        Same dict structure as risk_coverage_curve, plus std_acc.
    """
    n = len(correct)
    budgets = np.linspace(0.0, 1.0, n_points)
    rng = np.random.default_rng(seed)

    all_accs = np.full((n_trials, n_points), fill_value=np.nan)

    for t in range(n_trials):
        perm = rng.permutation(n)
        sorted_c = correct[perm].astype(float)
        for i, b in enumerate(budgets):
            n_confirm = int(round((1.0 - b) * n))
            if n_confirm > 0:
                all_accs[t, i] = sorted_c[:n_confirm].mean()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_acc = np.nanmean(all_accs, axis=0)
        std_acc  = np.nanstd(all_accs,  axis=0)

    acc_list = [float(v) if not np.isnan(v) else float("nan") for v in mean_acc]
    valid = [a for a in acc_list if not (isinstance(a, float) and a != a)]
    auc = float(np.trapz(valid, dx=1.0 / max(len(valid) - 1, 1))) if len(valid) > 1 else float("nan")

    return {
        "budgets": budgets.tolist(),
        "auto_confirm_acc": acc_list,
        "std_acc": std_acc.tolist(),
        "n_auto_confirmed": [int(round((1.0 - b) * n)) for b in budgets],
        "auc": auc,
    }


def operating_point(
    confidences: np.ndarray,
    correct: np.ndarray,
    budget: float,
) -> dict:
    """
    Report metrics at a specific routing budget.

    Args:
        confidences: 1-D float confidence array.
        correct:     1-D bool/int correct array.
        budget:      Fraction of patches routed to specialist (0 to 1).

    Returns:
        Dict with budget, n_routed, n_auto_confirmed, auto_confirm_acc, error_rate.
    """
    n = len(confidences)
    n_route = int(round(budget * n))
    n_confirm = n - n_route

    order = np.argsort(confidences)[::-1]
    sorted_correct = correct[order].astype(float)

    if n_confirm == 0:
        acc = float("nan")
    else:
        acc = float(sorted_correct[:n_confirm].mean())

    return {
        "budget": float(budget),
        "n_total": int(n),
        "n_routed": int(n_route),
        "n_auto_confirmed": int(n_confirm),
        "auto_confirm_acc": acc,
        "auto_confirm_error": float("nan") if acc != acc else round(1.0 - acc, 4),
    }
