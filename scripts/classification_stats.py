"""
scripts/classification_stats.py

Per-class F1, weighted-F1, and Wilson-score accuracy intervals for all four
models on val and test. No API calls: reads saved predictions.

Two reviewer points: (a) Table 1 reports CIs without naming the method, and (b)
macro-F1 is dominated by the near-zero classes so the full per-class breakdown
and a weighted-F1 should be shown. This script produces both from the same
prediction files the paper's accuracy/macro-F1 numbers come from, and uses
Wilson score intervals (better than the Wald interval at these class-collapsed
proportions).

Outputs:
    results/classification_stats.json

Usage:
    python scripts/classification_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
Z = 1.959963984540054  # 97.5th percentile of the standard normal


def wilson_interval(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - half), float(center + half)


def stats_from(true_names: list[str], pred_names: list[str]) -> dict:
    n = len(true_names)
    k = int(sum(t == p for t, p in zip(true_names, pred_names)))
    lo, hi = wilson_interval(k, n)
    per_class = f1_score(true_names, pred_names, average=None, labels=ALL_LABELS, zero_division=0)
    return {
        "n": n,
        "accuracy": round(k / n, 4),
        "wilson_ci": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(float(f1_score(true_names, pred_names, average="macro", labels=ALL_LABELS, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(true_names, pred_names, average="weighted", labels=ALL_LABELS, zero_division=0)), 4),
        "per_class_f1": {ALL_LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class)},
    }


def vlm_stats(model: str, split: str) -> dict:
    suffix = "V3_full" if split == "val" else "V3_full_test"
    df = pd.read_parquet(PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet")
    return stats_from(df["true_label_name"].tolist(), df["pred_label"].tolist())


def cnn_stats(model_key: str, split: str) -> dict:
    df = pd.read_parquet(PROJECT_ROOT / "results" / "cnn" / model_key / f"{split}_predictions.parquet")
    # Map indices to label names for a consistent per-class table.
    true_names = [LABEL_NAMES[i] for i in df["true_label_idx"].to_numpy(int)]
    pred_names = [LABEL_NAMES[i] for i in df["pred_label_idx"].to_numpy(int)]
    return stats_from(true_names, pred_names)


def main() -> int:
    out = {
        "medgemma-27b-it": {"val": vlm_stats("medgemma-27b-it", "val"), "test": vlm_stats("medgemma-27b-it", "test")},
        "gemma-3-27b-it": {"val": vlm_stats("gemma-3-27b-it", "val"), "test": vlm_stats("gemma-3-27b-it", "test")},
        "resnet18_64px": {"val": cnn_stats("resnet18_64px", "val"), "test": cnn_stats("resnet18_64px", "test")},
        "resnet18_224px": {"val": cnn_stats("resnet18_224px", "val"), "test": cnn_stats("resnet18_224px", "test")},
    }

    for model, splits in out.items():
        for split, s in splits.items():
            print(f"{model:16s} {split:4s}  acc={s['accuracy']:.4f} "
                  f"CI[{s['wilson_ci'][0]:.4f},{s['wilson_ci'][1]:.4f}]  "
                  f"macroF1={s['macro_f1']:.4f}  weightedF1={s['weighted_f1']:.4f}")

    out_path = PROJECT_ROOT / "results" / "classification_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Per-class F1 table (val) for the two VLMs, printed for the manuscript table.
    print("\nPer-class F1 (val / test):")
    print(f"{'class':<40}{'MG val':>8}{'MG test':>9}{'G3 val':>8}{'G3 test':>9}")
    for lbl in ALL_LABELS:
        mgv = out["medgemma-27b-it"]["val"]["per_class_f1"][lbl]
        mgt = out["medgemma-27b-it"]["test"]["per_class_f1"][lbl]
        g3v = out["gemma-3-27b-it"]["val"]["per_class_f1"][lbl]
        g3t = out["gemma-3-27b-it"]["test"]["per_class_f1"][lbl]
        print(f"{lbl:<40}{mgv:>8.3f}{mgt:>9.3f}{g3v:>8.3f}{g3t:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
