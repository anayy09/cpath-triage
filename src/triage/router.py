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

Tie handling
------------
Verbalized VLM confidence is coarse: on the 1,800-patch consistency subset one
signal takes 10 distinct values with 1,538 patches at exactly 0.95, and MedGemma's
full-scale confidence takes 4 distinct values with 96% at one of them. When most
pairs are tied, the ranking is mostly undetermined and any AUC that resolves ties
by row order is reporting a property of the file, not of the model. An earlier
version of this module ranked with a plain descending argsort, which orders tied
values by row index. Every signal record written under results/ now stores the
row-order value beside the permutation-invariant one so the size of that defect
stays quotable: on the 1,800-patch consistency subset the same consistency score
gives 0.4585 by row order and 0.4307 by the tie expectation.

Three tie policies are therefore available, all of which agree when there are no
ties:
  - "expected" : the exact expectation over uniformly random tie orders, computed
                 in closed form. Permutation-invariant by construction.
  - "random"   : Monte Carlo over n_repeats random tie orders, which additionally
                 gives the spread the ordering induces. The default, because that
                 spread is worth reporting rather than hiding.
  - "row_order": the original behaviour, kept so published numbers can be
                 reproduced and the size of the defect can be quoted.

Public API:
    risk_coverage_curve(confidences, correct, n_points, tie_break, n_repeats, seed) -> dict
    random_routing_curve(correct, n_points, n_trials, seed) -> dict
    operating_point(confidences, correct, budget, tie_break) -> dict
    tie_statistics(confidences) -> dict
"""

from __future__ import annotations

import math

import numpy as np

# np.trapz was removed in numpy 2.0. The project pins numpy>=1.26 with no upper
# bound, so resolve the name once at import rather than at every call site.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

TIE_BREAK_CHOICES = ("random", "expected", "row_order")

# Matches the published curve definition: the AUC is a trapezoid over the budgets
# where at least one patch is auto-confirmed, rescaled to unit width. Budget 1.0
# auto-confirms nothing and contributes NaN, so it is dropped.
def _auc_from_curve(acc_list: list[float]) -> float:
    valid = [a for a in acc_list if not math.isnan(a)]
    if len(valid) <= 1:
        return float("nan")
    return float(_trapezoid(valid, dx=1.0 / max(len(valid) - 1, 1)))


def _confirm_sizes(n: int, budgets: np.ndarray) -> list[int]:
    return [round((1.0 - b) * n) for b in budgets]


def tie_statistics(confidences: np.ndarray) -> dict:
    """
    Summarise how tie-dominated a confidence signal is.

    tie_fraction is the probability that two patches drawn without replacement
    share a confidence value. At 0.9 essentially all of the ranking is arbitrary
    and any AUC is pinned near the random baseline as a matter of arithmetic.

    Returns:
        Dict with n, n_distinct, tie_fraction, largest_tie_group_fraction.
    """
    conf = np.asarray(confidences, dtype=float)
    n = len(conf)
    if n < 2:
        return {
            "n": int(n),
            "n_distinct": len(np.unique(conf)) if n else 0,
            "tie_fraction": float("nan"),
            "largest_tie_group_fraction": float("nan"),
        }
    _, counts = np.unique(conf, return_counts=True)
    tied_pairs = float((counts * (counts - 1) / 2).sum())
    total_pairs = n * (n - 1) / 2
    return {
        "n": int(n),
        "n_distinct": len(counts),
        "tie_fraction": float(tied_pairs / total_pairs),
        "largest_tie_group_fraction": float(counts.max() / n),
    }


def _expected_prefix_accuracy(
    conf: np.ndarray, correct: np.ndarray, sizes: list[int]
) -> list[float]:
    """
    Exact expected auto-confirm accuracy at each prefix size, over random tie orders.

    Group the patches by confidence value in descending order. A prefix of size k
    always contains every patch whose confidence is strictly above the value at the
    cut, and fills its remaining slots uniformly at random from the tied group that
    straddles the cut. The expected number correct in those slots is the group's
    correct fraction times the number of slots, so the whole curve is available in
    closed form without sampling.
    """
    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    sorted_correct = correct[order].astype(float)

    # Group boundaries of equal-confidence runs in the descending order.
    values, starts, counts = np.unique(-sorted_conf, return_index=True, return_counts=True)
    del values
    group_correct = np.add.reduceat(sorted_correct, starts)

    cum_count = np.concatenate([[0], np.cumsum(counts)])          # C_0 .. C_m
    cum_correct = np.concatenate([[0.0], np.cumsum(group_correct)])  # S_0 .. S_m

    acc: list[float] = []
    for k in sizes:
        if k <= 0:
            acc.append(float("nan"))
            continue
        # Index of the group that straddles the cut at k.
        j = int(np.searchsorted(cum_count, k, side="left"))
        j = max(j, 1)
        full = cum_correct[j - 1]
        slots = k - cum_count[j - 1]
        rate = group_correct[j - 1] / counts[j - 1]
        acc.append(float((full + slots * rate) / k))
    return acc


def _row_order_prefix_accuracy(
    conf: np.ndarray, correct: np.ndarray, sizes: list[int]
) -> list[float]:
    """Original behaviour: descending argsort, so ties fall in reverse row order."""
    order = np.argsort(conf)[::-1]
    sorted_correct = correct[order].astype(float)
    return [
        float(sorted_correct[:k].mean()) if k > 0 else float("nan")
        for k in sizes
    ]


def risk_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_points: int = 101,
    *,
    tie_break: str = "random",
    n_repeats: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compute the accuracy-vs-routing-budget curve for a confidence-based policy.

    At each budget b, the policy routes the b*N lowest-confidence patches to the
    specialist and auto-confirms the (1-b)*N highest-confidence patches.

    Args:
        confidences: 1-D float array of confidence scores (higher = more certain).
        correct:     1-D bool/int array (1 if prediction is correct).
        n_points:    Number of evenly-spaced budget values in [0, 1].
        tie_break:   One of TIE_BREAK_CHOICES. See the module docstring.
        n_repeats:   Monte Carlo draws when tie_break="random".
        seed:        RNG seed for the Monte Carlo draws.

    Returns:
        Dict with keys:
            budgets:              list of float, routing fraction
            auto_confirm_acc:     list of float (NaN at budget=1.0 where N=0)
            n_auto_confirmed:     list of int
            auc:                  float, area under the curve (trapezoid)
            auc_sd:               float, sd of the AUC across tie orders
            auc_min, auc_max:     float, the smallest and largest AUC observed
                                  across the sampled tie orders. Their difference
                                  is how far two legitimate orderings of the same
                                  predictions can land apart, measured rather
                                  than argued. NaN unless tie_break="random".
            auc_expected:         float, exact expectation over random tie orders
            auc_row_order:        float, the pre-fix value, for comparison
            tie_fraction:         float, probability two patches tie
            n_distinct:           int, distinct confidence values
            tie_break, n_repeats, seed: the settings used
    """
    if tie_break not in TIE_BREAK_CHOICES:
        raise ValueError(f"tie_break must be one of {TIE_BREAK_CHOICES}, got {tie_break!r}")

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct)
    if len(conf) != len(corr):
        raise ValueError(f"length mismatch: {len(conf)} confidences, {len(corr)} correct")

    n = len(conf)
    budgets = np.linspace(0.0, 1.0, n_points)
    sizes = _confirm_sizes(n, budgets)

    stats = tie_statistics(conf)
    expected_acc = _expected_prefix_accuracy(conf, corr, sizes)
    auc_expected = _auc_from_curve(expected_acc)
    auc_row_order = _auc_from_curve(_row_order_prefix_accuracy(conf, corr, sizes))

    auc_min = auc_max = float("nan")
    if tie_break == "expected":
        acc_list, auc, auc_sd = expected_acc, auc_expected, 0.0
    elif tie_break == "row_order":
        acc_list = _row_order_prefix_accuracy(conf, corr, sizes)
        auc, auc_sd = auc_row_order, 0.0
    else:
        rng = np.random.default_rng(seed)
        corr_f = corr.astype(float)
        aucs = np.empty(n_repeats, dtype=float)
        acc_sum = np.zeros(len(sizes), dtype=float)
        for r in range(n_repeats):
            # Shuffle first, then sort on confidence alone: a stable sort of a
            # shuffled array places tied values in uniformly random order.
            perm = rng.permutation(n)
            order = perm[np.argsort(-conf[perm], kind="stable")]
            sorted_correct = corr_f[order]
            csum = np.concatenate([[0.0], np.cumsum(sorted_correct)])
            acc_r = [float(csum[k] / k) if k > 0 else float("nan") for k in sizes]
            aucs[r] = _auc_from_curve(acc_r)
            acc_sum += np.array([0.0 if math.isnan(a) else a for a in acc_r])
        auc = float(aucs.mean())
        auc_sd = float(aucs.std(ddof=1)) if n_repeats > 1 else 0.0
        auc_min, auc_max = float(aucs.min()), float(aucs.max())
        mean_acc = acc_sum / n_repeats
        acc_list = [
            float(m) if k > 0 else float("nan") for m, k in zip(mean_acc, sizes)
        ]

    return {
        "budgets": budgets.tolist(),
        "auto_confirm_acc": acc_list,
        "n_auto_confirmed": sizes,
        "auc": auc,
        "auc_sd": auc_sd,
        "auc_min": auc_min,
        "auc_max": auc_max,
        "auc_expected": auc_expected,
        "auc_row_order": auc_row_order,
        "tie_fraction": stats["tie_fraction"],
        "n_distinct": stats["n_distinct"],
        "tie_break": tie_break,
        "n_repeats": int(n_repeats) if tie_break == "random" else 0,
        "seed": int(seed),
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
            n_confirm = round((1.0 - b) * n)
            if n_confirm > 0:
                all_accs[t, i] = sorted_c[:n_confirm].mean()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_acc = np.nanmean(all_accs, axis=0)
        std_acc  = np.nanstd(all_accs,  axis=0)

    acc_list = [float(v) for v in mean_acc]

    return {
        "budgets": budgets.tolist(),
        "auto_confirm_acc": acc_list,
        "std_acc": std_acc.tolist(),
        "n_auto_confirmed": [round((1.0 - b) * n) for b in budgets],
        "auc": _auc_from_curve(acc_list),
    }


def operating_point(
    confidences: np.ndarray,
    correct: np.ndarray,
    budget: float,
    *,
    tie_break: str = "expected",
) -> dict:
    """
    Report metrics at a specific routing budget.

    Defaults to the exact tie expectation rather than Monte Carlo: a single cutoff
    has a closed-form answer, so there is nothing to sample.

    Args:
        confidences: 1-D float confidence array.
        correct:     1-D bool/int correct array.
        budget:      Fraction of patches routed to specialist (0 to 1).
        tie_break:   "expected" or "row_order".

    Returns:
        Dict with budget, n_routed, n_auto_confirmed, auto_confirm_acc, error_rate,
        plus the tie diagnostics.
    """
    if tie_break not in ("expected", "row_order"):
        raise ValueError(f"tie_break must be 'expected' or 'row_order', got {tie_break!r}")

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct)
    n = len(conf)
    n_route = round(budget * n)
    n_confirm = n - n_route

    if tie_break == "expected":
        acc = _expected_prefix_accuracy(conf, corr, [n_confirm])[0]
    else:
        acc = _row_order_prefix_accuracy(conf, corr, [n_confirm])[0]

    stats = tie_statistics(conf)
    return {
        "budget": float(budget),
        "n_total": int(n),
        "n_routed": int(n_route),
        "n_auto_confirmed": int(n_confirm),
        "auto_confirm_acc": acc,
        "auto_confirm_error": float("nan") if math.isnan(acc) else round(1.0 - acc, 4),
        "tie_break": tie_break,
        "tie_fraction": stats["tie_fraction"],
        "n_distinct": stats["n_distinct"],
    }
