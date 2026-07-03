"""
scripts/routing_ci.py

Paired-bootstrap confidence intervals for the routing (selective-accuracy) AUC
gaps reported in Table 3. No API calls: reads the full-scale predictions already
on disk.

Why this exists: Table 3 reported point estimates of the selective-accuracy AUC
for calibrated routing and for random routing, but no uncertainty on the gap.
A reviewer cannot tell whether "MedGemma 0.395 vs 0.403 random" differs from
noise. This script bootstraps the gap (calibrated minus random) with patch
identity paired within each resample, exactly as the consistency experiment
already does for Table 6b.

Metric orientation (see paper Section on the routing policy): the reported
quantity is the area under the auto-confirm-accuracy vs coverage curve. Higher
is better. A non-informative (random) confidence signal integrates to the
overall accuracy, so the natural reference for the gap is the base accuracy of
the set. We confirm empirically that random-routing AUC equals base accuracy
(to within averaging noise) and use base accuracy as the paired random
reference inside the bootstrap.

Outputs:
    results/routing/routing_auc_ci.json

Usage:
    python scripts/routing_ci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import risk_coverage_curve, random_routing_curve

N_BOOT = 1000
SEED = 42


def acc_coverage_auc(conf: np.ndarray, correct: np.ndarray, n_points: int = 101) -> float:
    """
    Area under the auto-confirm-accuracy vs coverage curve.

    Replicates src.triage.router.risk_coverage_curve's AUC exactly (same budget
    grid, same rounding, same trapezoid over the non-empty prefix) but via a
    single cumulative sum so the bootstrap stays cheap. Verified against the
    router implementation in main() before use.
    """
    n = len(conf)
    order = np.argsort(conf)[::-1]
    csum = np.cumsum(correct[order].astype(float))
    budgets = np.linspace(0.0, 1.0, n_points)
    accs = []
    for b in budgets:
        n_confirm = int(round((1.0 - b) * n))
        accs.append(csum[n_confirm - 1] / n_confirm if n_confirm > 0 else np.nan)
    accs = np.array(accs, dtype=float)
    valid = accs[~np.isnan(accs)]
    if len(valid) <= 1:
        return float("nan")
    return float(np.trapezoid(valid, dx=1.0 / (len(valid) - 1)))


def load_signal(model: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidence, correct) arrays for a model/split from saved predictions."""
    if model == "resnet18_64px":
        # CNN test predictions carry precomputed confidence and correct columns.
        path = PROJECT_ROOT / "results" / "cnn" / model / f"{split}_predictions.parquet"
        df = pd.read_parquet(path)
        return df["confidence"].to_numpy(float), df["correct"].to_numpy(bool)

    # VLMs: full-scale predictions. Parse failures ("unknown") are kept and
    # scored as incorrect, matching the paper's routing evaluation.
    suffix = "V3_full" if split == "val" else "V3_full_test"
    path = PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet"
    df = pd.read_parquet(path)
    conf = df["pred_confidence"].to_numpy(float)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    return conf, correct


def bootstrap_gap(conf: np.ndarray, correct: np.ndarray) -> dict:
    """
    Paired bootstrap of (selective-accuracy AUC) minus (random-routing AUC).

    Random-routing AUC equals base accuracy in expectation, so within each
    resample the random reference is the resample's mean correctness. Patch
    identity is held paired by resampling indices once per iteration.
    """
    rng = np.random.default_rng(SEED)
    n = len(correct)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        c, y = conf[idx], correct[idx]
        diffs[i] = acc_coverage_auc(c, y) - y.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": N_BOOT,
    }


def main() -> int:
    targets = [
        ("medgemma-27b-it", "val"),
        ("medgemma-27b-it", "test"),
        ("gemma-3-27b-it", "val"),
        ("gemma-3-27b-it", "test"),
        ("resnet18_64px", "test"),
    ]

    results: dict[str, dict] = {}
    for model, split in targets:
        conf, correct = load_signal(model, split)

        # Point estimates. Cross-check the fast AUC against the router module.
        cal_auc_router = risk_coverage_curve(conf, correct)["auc"]
        cal_auc_fast = acc_coverage_auc(conf, correct)
        assert abs(cal_auc_router - cal_auc_fast) < 1e-9, (
            f"AUC mismatch for {model}/{split}: {cal_auc_router} vs {cal_auc_fast}"
        )
        rnd_auc = random_routing_curve(correct, seed=SEED)["auc"]
        base_acc = float(correct.mean())

        boot = bootstrap_gap(conf, correct)

        key = f"{model}|{split}"
        results[key] = {
            "model": model,
            "split": split,
            "n": int(len(correct)),
            "base_accuracy": round(base_acc, 4),
            "cal_auc": round(float(cal_auc_router), 4),
            "random_auc": round(float(rnd_auc), 4),
            "gap_point": round(float(cal_auc_router - rnd_auc), 4),
            "gap_bootstrap": boot,
        }
        print(
            f"{key:28s} n={len(correct):5d}  cal={cal_auc_router:.4f}  "
            f"rnd={rnd_auc:.4f} (base_acc={base_acc:.4f})  "
            f"gap={boot['mean_diff']:+.4f} [{boot['ci_2.5']:+.4f}, {boot['ci_97.5']:+.4f}]  "
            f"{'excludes 0' if boot['excludes_zero'] else 'includes 0'}"
        )

    out_dir = PROJECT_ROOT / "results" / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "routing_auc_ci.json"
    with open(out_path, "w") as f:
        json.dump({"n_boot": N_BOOT, "seed": SEED, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
