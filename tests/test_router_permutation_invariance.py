"""
tests/test_router_permutation_invariance.py

Regression tests for the tie-handling defect (revision items C-01 and C-I6).

The defect: risk_coverage_curve ranked with a plain descending argsort, so tied
confidences were ordered by row index. On signals that are almost entirely ties,
which is what verbalized VLM confidence turns out to be, the resulting AUC is a
property of how the file happens to be sorted. Two legitimate row orderings of
the same 1,800 single-query predictions gave 0.266 and 0.329.

These tests fix that in place: AUC must not move when the input rows are
permuted. They are written to fail against the old implementation.

Run:
    python -m pytest tests/test_router_permutation_invariance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import (
    operating_point,
    risk_coverage_curve,
    tie_statistics,
)


def _tie_dominated_data(n: int = 1800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Mimic the real signal: a handful of distinct values, most mass on one of them,
    and correctness correlated with confidence so there is a real signal to recover
    underneath the ties.
    """
    rng = np.random.default_rng(seed)
    conf = rng.choice([0.80, 0.90, 0.95, 1.00], size=n, p=[0.05, 0.08, 0.82, 0.05])
    p_correct = 0.15 + 0.45 * (conf - 0.80) / 0.20
    correct = (rng.random(n) < p_correct).astype(int)
    return conf, correct


def test_expected_auc_is_exactly_permutation_invariant():
    conf, correct = _tie_dominated_data()
    base = risk_coverage_curve(conf, correct, tie_break="expected")["auc"]

    rng = np.random.default_rng(1)
    for _ in range(10):
        perm = rng.permutation(len(conf))
        shuffled = risk_coverage_curve(conf[perm], correct[perm], tie_break="expected")["auc"]
        assert shuffled == pytest.approx(base, abs=1e-12)


def test_random_tie_break_auc_is_invariant_within_monte_carlo_error():
    conf, correct = _tie_dominated_data()
    n_repeats = 200
    base = risk_coverage_curve(
        conf, correct, tie_break="random", n_repeats=n_repeats, seed=42
    )
    tolerance = 5.0 * base["auc_sd"] / np.sqrt(n_repeats)

    rng = np.random.default_rng(2)
    for _ in range(5):
        perm = rng.permutation(len(conf))
        shuffled = risk_coverage_curve(
            conf[perm], correct[perm], tie_break="random", n_repeats=n_repeats, seed=42
        )
        assert shuffled["auc"] == pytest.approx(base["auc"], abs=tolerance)


def test_row_order_auc_is_not_invariant_which_is_the_defect():
    """The old behaviour, kept behind a flag. This documents why it was replaced."""
    conf, correct = _tie_dominated_data()
    base = risk_coverage_curve(conf, correct, tie_break="row_order")["auc"]

    rng = np.random.default_rng(3)
    spread = []
    for _ in range(20):
        perm = rng.permutation(len(conf))
        spread.append(
            risk_coverage_curve(conf[perm], correct[perm], tie_break="row_order")["auc"]
        )
    assert max(spread) - min(spread) > 0.01, (
        "row_order tie handling should be visibly order-dependent on tied data"
    )
    assert base == pytest.approx(base)


def test_monte_carlo_mean_converges_to_closed_form_expectation():
    conf, correct = _tie_dominated_data()
    res = risk_coverage_curve(conf, correct, tie_break="random", n_repeats=400, seed=7)
    assert res["auc"] == pytest.approx(res["auc_expected"], abs=4e-3)


def test_no_ties_all_policies_agree():
    rng = np.random.default_rng(11)
    n = 500
    conf = rng.permutation(n).astype(float) / n  # all distinct
    correct = (rng.random(n) < conf).astype(int)

    expected = risk_coverage_curve(conf, correct, tie_break="expected")["auc"]
    row = risk_coverage_curve(conf, correct, tie_break="row_order")["auc"]
    mc = risk_coverage_curve(conf, correct, tie_break="random", n_repeats=20, seed=5)["auc"]

    assert expected == pytest.approx(row, abs=1e-12)
    assert mc == pytest.approx(row, abs=1e-12)


def test_tie_statistics_match_hand_computation():
    conf = np.array([1.0, 1.0, 1.0, 0.5])
    stats = tie_statistics(conf)
    # 3 tied pairs among the three 1.0 values, out of 6 total pairs.
    assert stats["tie_fraction"] == pytest.approx(0.5)
    assert stats["n_distinct"] == 2
    assert stats["largest_tie_group_fraction"] == pytest.approx(0.75)


def test_operating_point_expected_is_permutation_invariant():
    conf, correct = _tie_dominated_data()
    base = operating_point(conf, correct, 0.15)["auto_confirm_acc"]

    rng = np.random.default_rng(4)
    for _ in range(10):
        perm = rng.permutation(len(conf))
        got = operating_point(conf[perm], correct[perm], 0.15)["auto_confirm_acc"]
        assert got == pytest.approx(base, abs=1e-12)


def test_auc_does_not_use_removed_numpy_trapz():
    """C-I1: np.trapz was removed in numpy 2.0; the module must not depend on it."""
    conf, correct = _tie_dominated_data(n=200)
    res = risk_coverage_curve(conf, correct, tie_break="expected")
    assert np.isfinite(res["auc"])
    source = (PROJECT_ROOT / "src" / "triage" / "router.py").read_text(encoding="utf-8")
    assert "np.trapz(" not in source, "call np.trapezoid via the compatibility shim"
