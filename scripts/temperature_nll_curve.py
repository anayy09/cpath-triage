"""
scripts/temperature_nll_curve.py

The binary negative log-likelihood that the VLM temperature fit minimises,
evaluated as a curve in T rather than argued about in general.

Why this exists. Methods used to say that whenever every confidence exceeds the
aggregate accuracy, binary NLL decreases monotonically in T. That is false as a
general statement: at 40% accuracy, 0.99 on the correct cases and 0.70 on the
errors, NLL is 0.726 at T = 1, 0.595 at T = 2 and 0.690 at T = 200, so it has an
interior minimum. The conclusion the paper draws (T = 200 is where the search
stopped, not an estimate) does not need the general claim. It needs the curve
for the two models actually fitted, which this script computes on exactly the
calibration partitions scripts/calibrate.py fits on.

It also records a condition that is true, and checks it numerically rather
than asserting it. Write z_i = logit(c_i) and y_i for correctness. Then

    dNLL/dT = -(1 / (N T^2)) * sum_i z_i (sigmoid(z_i / T) - y_i).

If every z_i > 0, then sigmoid(z_i / T) > 1/2 for all T, so each term is at least
z_i (1/2 - y_i), and the sum is strictly positive whenever

    sum over errors of z_i  >=  sum over correct predictions of z_i,

which makes NLL strictly decreasing in T on the whole half-line. For a single
shared confidence c > 1/2 this reduces to accuracy <= 1/2. In the other
direction, as T grows the sign of the derivative tends to the sign of
sum_i z_i (y_i - 1/2), so if the log-odds mass on correct predictions exceeds
that on errors, NLL is increasing for large T and the minimiser is finite. The
counterexample is of that kind.

Outputs:
    results/calibration/nll_curve.json
    results/calibration/nll_curve.png  (and .pdf)

Usage:
    python scripts/temperature_nll_curve.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VLMS = ("medgemma-27b-it", "gemma-3-27b-it")
LABELS = {"medgemma-27b-it": "MedGemma-27b-it", "gemma-3-27b-it": "Gemma-3-27b-it"}
# Okabe-Ito, so the two curves stay distinguishable in grey and for colour-blind readers.
COLORS = {"medgemma-27b-it": "#D55E00", "gemma-3-27b-it": "#0072B2"}
SEARCH_BOUNDS = (0.05, 200.0)
EPS = 1e-7
T_GRID = np.logspace(math.log10(0.05), 5.0, 4000)
# Floating-point slack for "decreasing": at very large T successive values agree
# to ~1e-16 and rounding can produce a spurious positive step.
MONO_TOL = 1e-12


def binary_nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    """Binary NLL of sigmoid(logit / T), clipped exactly as src/eval/calibration.py does."""
    p = 1.0 / (1.0 + np.exp(-logits / T))
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def exact_nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    """
    The same objective without the probability clip, via log-sum-exp.

    The clip in calibration.py flattens the loss of confidently wrong cases at
    very small T while leaving confidently right ones free to move, which can
    produce a spurious increase near T = 0.05. The mathematical condition is
    about the unclipped function, so it is checked on this one.
    """
    s = logits / T
    return float(np.mean(y * np.logaddexp(0.0, -s) + (1.0 - y) * np.logaddexp(0.0, s)))


def to_logits(conf: np.ndarray) -> np.ndarray:
    c = np.clip(conf, EPS, 1.0 - EPS)
    return np.log(c / (1.0 - c))


def curve(
    logits: np.ndarray, y: np.ndarray, grid: np.ndarray = T_GRID, clipped: bool = True
) -> np.ndarray:
    f = binary_nll if clipped else exact_nll
    return np.array([f(logits, y, T) for T in grid])


def monotone_decreasing(values: np.ndarray, tol: float = MONO_TOL) -> bool:
    return bool(np.all(np.diff(values) <= tol))


def condition_statistics(logits: np.ndarray, y: np.ndarray) -> dict:
    """The quantities the sufficient and the large-T necessary conditions are stated in."""
    pos = logits > 0
    err_mass = float(logits[(y == 0) & pos].sum())
    cor_mass = float(logits[(y == 1) & pos].sum())
    return {
        "n": len(logits),
        "n_logit_positive": int(pos.sum()),
        "n_logit_zero": int((logits == 0).sum()),
        "n_logit_negative": int((logits < 0).sum()),
        "logodds_mass_on_errors": round(err_mass, 4),
        "logodds_mass_on_correct": round(cor_mass, 4),
        "sufficient_condition_holds": bool(pos.all() and err_mass >= cor_mass),
        "large_T_derivative_sign_statistic": round(float(np.sum(logits * (0.5 - y))), 4),
        "note": (
            "sufficient_condition_holds requires every confidence above 0.5; when some "
            "logits are zero or negative the condition does not apply and the curve "
            "is checked numerically instead."
        ),
    }


def calibration_partition(model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The exact calibration set calibrate.py fits T on: parse failures dropped,
    default_rng(seed) permutation, first 80%.
    """
    path = PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_parquet(path)
    df = df[df["pred_label"] != "unknown"].copy()
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(float)
    conf = df["pred_confidence"].to_numpy(float)
    idx = np.random.default_rng(seed).permutation(len(df))
    cal = idx[: int(len(df) * 0.8)]
    return conf[cal], correct[cal]


def counterexample() -> dict:
    """Reviewer 3's configuration: 40% accuracy, 0.99 on correct, 0.70 on errors."""
    conf = np.array([0.99] * 4 + [0.70] * 6)
    y = np.array([1.0] * 4 + [0.0] * 6)
    z = to_logits(conf)
    vals = curve(z, y, clipped=False)
    i = int(np.argmin(vals))
    return {
        "accuracy": 0.4,
        "confidence_correct": 0.99,
        "confidence_errors": 0.70,
        "nll_at": {str(T): round(binary_nll(z, y, T), 4) for T in (1.0, 2.0, 200.0)},
        "grid_minimiser_T": round(float(T_GRID[i]), 3),
        "grid_minimum_nll": round(float(vals[i]), 4),
        "monotone_decreasing_on_grid": monotone_decreasing(vals),
        "conditions": condition_statistics(z, y),
    }


def randomized_check(seed: int, n_instances: int = 2000) -> dict:
    """
    Numerical check of the sufficient condition on random instances.

    Draw confidences above 0.5 and binary outcomes at random, keep instances
    where the condition holds, and confirm every one of them is monotone on the
    grid. Instances where it fails are counted separately, to show the condition
    is not vacuous: some of those have interior minima.
    """
    rng = np.random.default_rng(seed)
    grid = np.logspace(math.log10(0.05), 4.0, 600)
    held = held_monotone = held_clipped_monotone = failed = failed_nonmonotone = 0
    for _ in range(n_instances):
        n = int(rng.integers(5, 60))
        conf = rng.uniform(0.501, 0.999, size=n)
        y = (rng.uniform(size=n) < rng.uniform(0.05, 0.95)).astype(float)
        z = to_logits(conf)
        stats = condition_statistics(z, y)
        vals = curve(z, y, grid, clipped=False)
        if stats["sufficient_condition_holds"]:
            held += 1
            held_monotone += monotone_decreasing(vals)
            held_clipped_monotone += monotone_decreasing(curve(z, y, grid))
        else:
            failed += 1
            failed_nonmonotone += not monotone_decreasing(vals)
    return {
        "n_instances": n_instances,
        "condition_held": held,
        "condition_held_and_monotone": held_monotone,
        "condition_held_and_clipped_objective_monotone": held_clipped_monotone,
        "condition_failed": failed,
        "condition_failed_and_not_monotone": failed_nonmonotone,
        "grid": "600 log-spaced T in [0.05, 1e4]",
        "objective": (
            "exact (unclipped) NLL; the clipped objective of calibration.py can rise "
            "spuriously near T = 0.05 when some logits are large, which is why the "
            "clipped count can be lower"
        ),
    }


def plot(curves: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(85 / 25.4, 60 / 25.4))
    for model, vals in curves.items():
        ax.plot(T_GRID, vals, color=COLORS[model], lw=1.2, label=LABELS[model])
    ax.axhline(math.log(2), color="0.35", ls="--", lw=0.8, label=r"$\ln 2$ (limit as $T\to\infty$)")
    ax.axvline(SEARCH_BOUNDS[1], color="0.35", ls=":", lw=0.8, label="search bound $T=200$")
    ax.set_xscale("log")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Binary NLL (calibration set)")
    ax.set_ylim(0.65, 2.5)
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed}")

    models: dict[str, dict] = {}
    curves: dict[str, np.ndarray] = {}
    for model in VLMS:
        conf, y = calibration_partition(model, args.seed)
        z = to_logits(conf)
        vals = curve(z, y)
        curves[model] = vals
        in_bounds = T_GRID <= SEARCH_BOUNDS[1]
        models[model] = {
            "n_cal": len(y),
            "accuracy": round(float(y.mean()), 4),
            "min_confidence": float(conf.min()),
            "fraction_confidence_at_or_below_half": round(float((conf <= 0.5).mean()), 4),
            "nll_at": {str(T): round(binary_nll(z, y, T), 5) for T in (1.0, 200.0, 1e4)},
            "ln2": round(math.log(2), 5),
            "monotone_decreasing_on_search_interval": monotone_decreasing(vals[in_bounds]),
            "monotone_decreasing_on_full_grid": monotone_decreasing(vals),
            "exact_objective_monotone_decreasing_on_full_grid": monotone_decreasing(
                curve(z, y, clipped=False)),
            "grid": "4000 log-spaced T in [0.05, 1e5]",
            "conditions": condition_statistics(z, y),
        }
        print(f"{model}: n={len(y)} acc={y.mean():.4f} NLL(1)={models[model]['nll_at']['1.0']} "
              f"NLL(200)={models[model]['nll_at']['200.0']} NLL(1e4)={models[model]['nll_at']['10000.0']} "
              f"monotone={models[model]['monotone_decreasing_on_full_grid']} "
              f"sufficient={models[model]['conditions']['sufficient_condition_holds']}")

    ce = counterexample()
    print(f"counterexample NLL at T=1,2,200: {ce['nll_at']}  minimiser T={ce['grid_minimiser_T']}")
    rc = randomized_check(args.seed)
    print(f"randomized check: {rc}")

    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "nll_curve.json"
    out_json.write_text(json.dumps({
        "seed": args.seed,
        "optimiser_bounds": list(SEARCH_BOUNDS),
        "objective": "binary NLL of sigmoid(logit(c) / T), clipped at 1e-7, as in src/eval/calibration.py",
        "partition": "calibrate.py calibration set: parse failures dropped, default_rng(seed) permutation, first 80%",
        "general_claim_withdrawn": (
            "All confidences exceeding aggregate accuracy does not make NLL monotone in T; "
            "see counterexample."
        ),
        "sufficient_condition": (
            "If every confidence exceeds 0.5 and the summed log-odds over incorrect "
            "predictions is at least the summed log-odds over correct predictions, NLL is "
            "strictly decreasing in T for all T > 0, so a bounded search runs to its upper "
            "bound. For one shared confidence above 0.5 this is accuracy <= 0.5."
        ),
        "large_T_condition": (
            "As T grows, sign(dNLL/dT) tends to -sign(sum_i logit(c_i) (0.5 - y_i)). If the "
            "log-odds mass on correct predictions exceeds that on errors, NLL increases at "
            "large T and the minimiser is finite."
        ),
        "models": models,
        "counterexample": ce,
        "randomized_check_of_sufficient_condition": rc,
    }, indent=2), encoding="utf-8")
    out_png = out_dir / "nll_curve.png"
    plot(curves, out_png)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
