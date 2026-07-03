"""
src/eval/calibration.py

Calibration metrics and temperature scaling for the cpath-triage pipeline.

Two calibration contexts exist here:
  VLM (scalar confidence): the model emits a single float c in [0,1].
    Treat c as P(correct); calibrate with one-parameter Platt/logit scaling.
  CNN (full softmax): the model emits K logits.
    Calibrate with standard temperature scaling (Guo et al., 2017).

Neither context uses the test split. All fitting happens on a calibration
partition carved from val.

Public API:
    ece_score, mce_score, brier_binary, brier_multiclass
    reliability_diagram_data
    TemperatureScaler
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


# ── Scalar metrics ────────────────────────────────────────────────────────────

def ece_score(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error.

    Bins predictions by confidence, measures the weighted average absolute
    gap between mean bin confidence and bin accuracy.

    Args:
        confidences: 1-D float array in [0, 1].
        correct:     1-D bool/int array (1 if prediction was correct).
        n_bins:      Number of equal-width bins covering [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(confidences[mask].mean() - correct[mask].astype(float).mean())
        ece += mask.sum() * gap
    return float(ece / n)


def adaptive_ece_score(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Adaptive (equal-mass) Expected Calibration Error.

    Nixon et al. (2019) showed equal-width ECE is sensitive to the binning
    scheme; equal-mass bins put the same number of samples in each bin, which
    removes the empty/near-empty high-confidence bins that dominate the
    equal-width estimate. Reported alongside the equal-width ece_score so the
    calibration conclusions can be shown to hold across schemes.

    Args:
        confidences: 1-D float array in [0, 1].
        correct:     1-D bool/int array (1 if prediction was correct).
        n_bins:      Number of equal-count bins.

    Returns:
        Adaptive ECE as a float in [0, 1].
    """
    n = len(confidences)
    if n == 0:
        return float("nan")
    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correct[order].astype(float)
    # Split indices into n_bins near-equal groups.
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        c = conf_sorted[lo:hi]
        y = corr_sorted[lo:hi]
        ece += len(c) * abs(c.mean() - y.mean())
    return float(ece / n)


def mce_score(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Maximum Calibration Error: the worst single bin gap.

    Args:
        confidences: 1-D float array in [0, 1].
        correct:     1-D bool/int array.
        n_bins:      Number of equal-width bins.

    Returns:
        MCE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(confidences[mask].mean() - correct[mask].astype(float).mean())
        mce = max(mce, gap)
    return float(mce)


def brier_binary(confidences: np.ndarray, correct: np.ndarray) -> float:
    """
    Brier score treating calibration as a binary P(correct) problem.
    Lower is better; perfect calibration = 0.

    Args:
        confidences: 1-D float array in [0, 1].
        correct:     1-D bool/int array.
    """
    return float(np.mean((confidences - correct.astype(float)) ** 2))


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multiclass Brier score (mean squared error over one-hot targets).
    Lower is better.

    Args:
        probs:  (N, K) float array of class probabilities.
        labels: (N,) int array of true class indices.
    """
    n, k = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ── Reliability diagram data ──────────────────────────────────────────────────

def reliability_diagram_data(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """
    Compute per-bin statistics for a reliability diagram.

    Returns a dict with:
        bin_centers:     list of float (mid-point of each bin range)
        bin_confidences: list of float (mean predicted confidence in bin; NaN if empty)
        bin_accuracies:  list of float (fraction correct in bin; NaN if empty)
        bin_counts:      list of int
        ece:             float
        mce:             float
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        centers.append(float((lo + hi) / 2))
        if mask.sum() == 0:
            confs.append(float("nan"))
            accs.append(float("nan"))
            counts.append(0)
        else:
            confs.append(float(confidences[mask].mean()))
            accs.append(float(correct[mask].astype(float).mean()))
            counts.append(int(mask.sum()))

    return {
        "bin_centers": centers,
        "bin_confidences": confs,
        "bin_accuracies": accs,
        "bin_counts": counts,
        "ece": ece_score(confidences, correct, n_bins),
        "mce": mce_score(confidences, correct, n_bins),
    }


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaler:
    """
    One-parameter temperature scaling.

    Two modes:
      fit_scalar / transform_scalar: for VLM scalar confidence c in [0,1].
        Converts c to log-odds, divides by T, back to probability. Minimises
        binary NLL on a (confidence, correct) calibration set.

      fit_logits / transform_logits: for CNN K-dimensional logit arrays.
        Standard temperature scaling from Guo et al. (2017): divides logit
        vector by T, applies softmax. Minimises cross-entropy via LBFGS.

    After fitting, T > 1 means the model was overconfident; T < 1 means
    it was underconfident.

    Args:
        n_bins: Bins used in ECE reporting after fitting.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.T = 1.0
        self.n_bins = n_bins

    # ---- VLM scalar mode ----

    def fit_scalar(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
    ) -> float:
        """
        Find T minimising binary NLL on (confidence, correct) pairs.

        Args:
            confidences: 1-D float in [0, 1] (raw model confidence).
            correct:     1-D bool/int (1 = prediction was correct).

        Returns:
            Fitted temperature T.
        """
        eps = 1e-7
        c = np.clip(confidences, eps, 1.0 - eps)
        logits = np.log(c / (1.0 - c))   # log-odds
        y = correct.astype(float)

        def nll(T: float) -> float:
            T = max(T, 1e-3)
            p = 1.0 / (1.0 + np.exp(-logits / T))
            p = np.clip(p, eps, 1.0 - eps)
            return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        result = minimize_scalar(nll, bounds=(0.05, 200.0), method="bounded")
        self.T = float(result.x)
        return self.T

    def transform_scalar(self, confidences: np.ndarray) -> np.ndarray:
        """Apply fitted T to scalar VLM confidences."""
        eps = 1e-7
        c = np.clip(confidences, eps, 1.0 - eps)
        logits = np.log(c / (1.0 - c))
        return (1.0 / (1.0 + np.exp(-logits / self.T))).astype(np.float32)

    # ---- CNN logit mode ----

    def fit_logits(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Find T minimising cross-entropy on (logits, labels) via LBFGS.

        Args:
            logits: (N, K) float array of pre-softmax scores.
            labels: (N,)  int array of true class indices.

        Returns:
            Fitted temperature T.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        T_param = nn.Parameter(torch.ones(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([T_param], lr=0.1, max_iter=100)
        ce_loss = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_t / T_param.clamp(min=1e-3)
            loss = ce_loss(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.T = float(T_param.item())
        return self.T

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to a (N, K) logit array.

        Returns:
            (N, K) float32 softmax probability array.
        """
        scaled = logits / max(self.T, 1e-3)
        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)
