"""
tests/test_weighted.py

The weighted routing metrics must collapse to the published unweighted ones when
every weight is 1, must not depend on row order, and must treat an integer
weight exactly like that many copies of the patch.

Run:
    python -m pytest tests/test_weighted.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import risk_coverage_curve
from src.triage.weighted import weighted_auc, weighted_auroc, weighted_expected_curve


def _tied_signal(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Few distinct values, most mass on one, like verbalized confidence.
    conf = rng.choice([0.8, 0.9, 0.95, 1.0], size=n, p=[0.05, 0.1, 0.8, 0.05])
    correct = rng.random(n) < 0.4
    return conf, correct


@pytest.mark.parametrize("n,seed", [(1800, 0), (2001, 1), (37, 2), (10004, 3)])
def test_unit_weights_equal_router_expectation(n: int, seed: int) -> None:
    conf, correct = _tied_signal(n, seed)
    expected = risk_coverage_curve(conf, correct, tie_break="expected")["auc"]
    got = weighted_auc(conf, correct, np.ones(n))
    assert abs(got - expected) < 1e-12


def test_curve_points_equal_router_with_unit_weights() -> None:
    conf, correct = _tied_signal(500, 4)
    a = risk_coverage_curve(conf, correct, tie_break="expected")["auto_confirm_acc"]
    b = weighted_expected_curve(conf, correct, np.ones(500))["auto_confirm_acc"]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def test_row_permutation_invariance() -> None:
    conf, correct = _tied_signal(1800, 5)
    w = np.random.default_rng(6).uniform(1.0, 30.0, size=1800)
    base = weighted_auc(conf, correct, w)
    perm = np.random.default_rng(7).permutation(1800)
    assert abs(weighted_auc(conf[perm], correct[perm], w[perm]) - base) < 1e-12
    auroc = weighted_auroc(conf, correct, w)
    assert abs(weighted_auroc(conf[perm], correct[perm], w[perm]) - auroc) < 1e-12


def test_integer_weight_equals_replication() -> None:
    conf, correct = _tied_signal(300, 8)
    reps = np.random.default_rng(9).integers(1, 5, size=300)
    expanded_conf = np.repeat(conf, reps)
    expanded_corr = np.repeat(correct, reps)
    replicated = risk_coverage_curve(expanded_conf, expanded_corr, tie_break="expected")["auc"]
    assert abs(weighted_auc(conf, correct, reps.astype(float)) - replicated) < 1e-12


def test_auroc_unit_weights_matches_midrank() -> None:
    from scipy.stats import mannwhitneyu

    conf, correct = _tied_signal(800, 10)
    u = mannwhitneyu(conf[correct], conf[~correct]).statistic
    midrank = u / (correct.sum() * (~correct).sum())
    assert abs(weighted_auroc(conf, correct, np.ones(800)) - midrank) < 1e-12
