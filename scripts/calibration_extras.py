"""
scripts/calibration_extras.py

Calibration metrics beyond single-scheme equal-width ECE. No API calls: reads
the same predictions and reproduces the same held-out evaluation partition as
scripts/calibrate.py, then adds adaptive (equal-mass) ECE and the Brier score so
the paper does not rest on one binning choice (the reviewer's point, and the one
Nixon et al. warn about).

For each model it reports, on the held-out evaluation partition (Table 2) and on
the full external test set (Table 5):
  - equal-width ECE  (recomputed; must match the canonical calibration_params)
  - adaptive ECE     (equal-mass bins)
  - Brier score      (binary P(correct))

Outputs:
    results/calibration/calibration_extras.json

Usage:
    python scripts/calibration_extras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration import ece_score, adaptive_ece_score, brier_binary

SEED = 42
N_CLASSES = 9


def _metrics(conf: np.ndarray, correct: np.ndarray) -> dict:
    return {
        "n": int(len(correct)),
        "ece_equal_width": round(float(ece_score(conf, correct)), 4),
        "ece_adaptive": round(float(adaptive_ece_score(conf, correct)), 4),
        "brier": round(float(brier_binary(conf, correct)), 4),
    }


def _eval_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce calibrate.py's 80/20 held-out evaluation partition."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cal_n = int(len(df) * 0.8)
    return df.iloc[idx[cal_n:]]


def vlm_extras(model: str) -> dict:
    zroot = PROJECT_ROOT / "results" / "zeroshot" / model
    val = pd.read_parquet(zroot / "V3_full" / "predictions.parquet")
    val = val[val["pred_label"] != "unknown"].copy()
    val["correct"] = val["pred_label"] == val["true_label_name"]
    ev = _eval_partition(val)

    test = pd.read_parquet(zroot / "V3_full_test" / "predictions.parquet")
    test = test[test["pred_label"] != "unknown"].copy()
    test["correct"] = test["pred_label"] == test["true_label_name"]

    return {
        "eval_partition": _metrics(ev["pred_confidence"].to_numpy(float), ev["correct"].to_numpy(bool)),
        "test_full": _metrics(test["pred_confidence"].to_numpy(float), test["correct"].to_numpy(bool)),
    }


def cnn_extras(model_key: str) -> dict:
    croot = PROJECT_ROOT / "results" / "cnn" / model_key
    prob_cols = [f"prob_{c}" for c in range(N_CLASSES)]

    val = pd.read_parquet(croot / "val_predictions.parquet")
    ev = _eval_partition(val)
    ev_conf = ev[prob_cols].to_numpy(float).max(axis=1)
    ev_corr = (ev["pred_label_idx"].to_numpy(int) == ev["true_label_idx"].to_numpy(int))

    test = pd.read_parquet(croot / "test_predictions.parquet")
    test_conf = test[prob_cols].to_numpy(float).max(axis=1)
    test_corr = (test["pred_label_idx"].to_numpy(int) == test["true_label_idx"].to_numpy(int))

    return {
        "eval_partition": _metrics(ev_conf, ev_corr),
        "test_full": _metrics(test_conf, test_corr),
    }


def main() -> int:
    out = {
        "medgemma-27b-it": vlm_extras("medgemma-27b-it"),
        "gemma-3-27b-it": vlm_extras("gemma-3-27b-it"),
        "resnet18_64px": cnn_extras("resnet18_64px"),
        "resnet18_224px": cnn_extras("resnet18_224px"),
    }

    for model, parts in out.items():
        for part, m in parts.items():
            print(
                f"{model:16s} {part:15s} n={m['n']:5d}  "
                f"ECE(width)={m['ece_equal_width']:.4f}  "
                f"ECE(adaptive)={m['ece_adaptive']:.4f}  Brier={m['brier']:.4f}"
            )

    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_path = out_dir / "calibration_extras.json"
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "results": out}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
