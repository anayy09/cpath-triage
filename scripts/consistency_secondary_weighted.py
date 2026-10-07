"""
scripts/consistency_secondary_weighted.py

Cohort-level estimates for the two consistency results that consistency_population.py
does not cover. No API calls, no images.

1. The V5 variant family. Its 1,800 validation patches were drawn as 200 per class
   with no stratification on correctness (the full-scale predictions were not yet
   available when it ran), so each patch's inclusion probability is 200 divided by
   its class's size in the validation split. Weighting by the inverse gives the
   cohort-level routing AUCs, with a paired bootstrap that resamples inside each
   class. Output: results/consistency/medgemma-27b-it/V5/weighted_routing.json.

2. Accuracy by agreement level for the V3 family on both splits. The subset
   accuracies at 5/5 and at 2/5 or fewer depend on how many correct and incorrect
   patches the stratified draw took, so they are restated with the inclusion
   weights from consistency_selection_weights.py. Output:
   results/consistency/medgemma-27b-it/{V3,V3_test}/accuracy_by_agreement.json.

Usage:
    python scripts/consistency_secondary_weighted.py [--n-boot 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.weighted import (
    stratified_bootstrap_indices,
    summarize_diffs,
    weighted_accuracy,
    weighted_auc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "medgemma-27b-it"
CONS_ROOT = PROJECT_ROOT / "results" / "consistency" / MODEL
FULL_VAL = PROJECT_ROOT / "results" / "zeroshot" / MODEL / "V3_full" / "predictions.parquet"


def _v5_weights(df: pd.DataFrame) -> np.ndarray:
    full = pd.read_parquet(FULL_VAL)
    class_size = full["true_label_name"].value_counts()
    drawn = df["true_label_name"].value_counts()
    # A class drawn in full would have probability 1; none is, but guard anyway.
    pi = (drawn / class_size).clip(upper=1.0)
    return (1.0 / df["true_label_name"].map(pi)).to_numpy(float)


def v5_weighted(n_boot: int, seed: int) -> dict:
    path = CONS_ROOT / "V5" / "predictions.parquet"
    df = pd.read_parquet(path)
    w = _v5_weights(df)
    cons = df["consistency_score"].to_numpy(float)
    mean5 = df["mean_textual_conf"].to_numpy(float)
    corr = df["is_correct"].to_numpy(bool)

    def stats(idx: np.ndarray) -> tuple[float, float, float]:
        rnd = weighted_accuracy(corr[idx], w[idx])
        return (weighted_auc(cons[idx], corr[idx], w[idx]) - rnd,
                weighted_auc(mean5[idx], corr[idx], w[idx]) - rnd,
                weighted_auc(cons[idx], corr[idx], w[idx])
                - weighted_auc(mean5[idx], corr[idx], w[idx]))

    full_idx = np.arange(len(df))
    point = stats(full_idx)
    boots = np.array([stats(ix) for ix in stratified_bootstrap_indices(
        df["true_label_name"].to_numpy(), n_boot, seed)])
    names = ("consistency_minus_random", "mean5_conf_minus_random",
             "consistency_minus_mean5_conf")
    return {
        "model": MODEL, "version": "V5", "split": "val", "seed": seed, "n_boot": n_boot,
        "n_patches": len(df), "cohort_size_weighted": float(w.sum()),
        "design": "200 patches per class, uniform within class, no correctness stratification",
        "auc_weighted": {
            "consistency|modal": round(weighted_auc(cons, corr, w), 4),
            "mean5_conf|modal": round(weighted_auc(mean5, corr, w), 4),
            "random|modal": round(weighted_accuracy(corr, w), 4),
        },
        "contrasts_weighted": {n: summarize_diffs(boots[:, k], point[k])
                               for k, n in enumerate(names)},
    }


def accuracy_by_agreement(split_dir: str) -> dict:
    d = CONS_ROOT / split_dir
    df = pd.read_parquet(d / "predictions.parquet")
    wt = pd.read_parquet(d / "weights.parquet")
    if list(wt["arr_idx"]) != list(df["arr_idx"]):
        raise SystemExit(f"weights.parquet is not aligned with predictions in {d}")
    w = wt["weight"].to_numpy(float)
    corr = df["is_correct"].to_numpy(bool)
    k_agree = np.rint(df["consistency_score"].to_numpy(float) * 5).astype(int)
    levels = {"5_of_5": k_agree == 5, "4_of_5": k_agree == 4, "3_of_5": k_agree == 3,
              "2_or_fewer_of_5": k_agree <= 2}
    out = {}
    for name, m in levels.items():
        out[name] = {
            "n_subset": int(m.sum()),
            "cohort_share": round(float(w[m].sum() / w.sum()), 4),
            "accuracy_subset": round(float(corr[m].mean()), 4),
            "accuracy_cohort": round(weighted_accuracy(corr[m], w[m]), 4),
        }
    return {"model": MODEL, "version": "V3", "split_dir": split_dir, "levels": out,
            "mean_confidence_without_variant_2": _mean_conf_without_v2(df, w, wt["cell"])}


def _mean_conf_without_v2(df: pd.DataFrame, w: np.ndarray, cells: pd.Series,
                          n_boot: int = 1000, seed: int = 42) -> dict:
    # Variant 2 is the only variant whose own confidence discriminates, so this
    # asks whether the averaged confidence's small cohort-level gain is its doing.
    confs = np.array([json.loads(c) for c in df["raw_confs"]], dtype=float)
    corr = df["is_correct"].to_numpy(bool)
    mean5 = confs.mean(axis=1)
    mean4 = confs[:, [0, 1, 3, 4]].mean(axis=1)

    def gaps(ix: np.ndarray) -> tuple[float, float]:
        rnd = weighted_accuracy(corr[ix], w[ix])
        return (weighted_auc(mean5[ix], corr[ix], w[ix]) - rnd,
                weighted_auc(mean4[ix], corr[ix], w[ix]) - rnd)

    point = gaps(np.arange(len(df)))
    boots = np.array([gaps(ix) for ix in stratified_bootstrap_indices(
        cells.to_numpy(), n_boot, seed)])
    return {"mean5_conf_minus_random": summarize_diffs(boots[:, 0], point[0]),
            "mean4_conf_without_v2_minus_random": summarize_diffs(boots[:, 1], point[1])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger.info("seed=%d n_boot=%d", args.seed, args.n_boot)

    out = CONS_ROOT / "V5" / "weighted_routing.json"
    out.write_text(json.dumps(v5_weighted(args.n_boot, args.seed), indent=2))
    print(f"Saved: {out}")
    for split_dir in ("V3", "V3_test"):
        out = CONS_ROOT / split_dir / "accuracy_by_agreement.json"
        out.write_text(json.dumps(accuracy_by_agreement(split_dir), indent=2))
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
