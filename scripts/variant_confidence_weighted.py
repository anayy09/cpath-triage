"""
scripts/variant_confidence_weighted.py

Each prompt variant's own verbalized confidence, as a single-query signal routing
that variant's own answers, weighted to the whole split. No API calls, no images.

Why this exists. The claim that MedGemma's uninformative confidence belongs to the
V3 prompt rather than to the model rested on the V5 family, which ran on a
different sample. The K=5 runs hold a sharper test on identical patches: under
the same session and the same images, one variant (2, the restructured V4-style
prompt) may yield a confidence that ranks while the others do not. This script
measures that per variant, for every family that stored per-variant answers:

    V3       validation    selection weights of the V3 run
    V3_test  external      selection weights of the V3 external run
    V3W      validation    the V3 validation weights (same patches, same order)
    V3W_test external      the V3 external weights, once that run exists
    V5       validation    class-share weights (200 per class, no correctness strata)

Per variant: cohort accuracy, weighted confidence-correctness AUROC (ties at one
half), the selective-accuracy AUC of routing its own answers by its own
confidence, and the gap over random routing of the same answers (the weighted
accuracy), with a paired bootstrap inside the selection strata. Unweighted subset
AUROC is reported beside it for comparison with per_variant.json.

Output: results/consistency/medgemma-27b-it/variant_confidence_weighted.json

Usage:
    python scripts/variant_confidence_weighted.py [--n-boot 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from confidence_information import auroc_midrank
from consistency_secondary_weighted import _v5_weights

from src.triage.weighted import (
    stratified_bootstrap_indices,
    summarize_diffs,
    weighted_accuracy,
    weighted_auc,
    weighted_auroc,
)

CONS = PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it"
OUT = CONS / "variant_confidence_weighted.json"
K = 5
RUNS = (
    ("V3", "val", "V3"),
    ("V3_test", "test", "V3_test"),
    ("V3W", "val", "V3"),
    ("V3W_test", "test", "V3_test"),
    ("V5", "val", None),
)


def load_run(run: str, weights_from: str | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(CONS / run / "predictions.parquet")
    if weights_from is None:
        w = _v5_weights(df)
        strata = df["true_label_name"].to_numpy()
    else:
        wt = pd.read_parquet(CONS / weights_from / "weights.parquet")
        if wt["arr_idx"].tolist() != df["arr_idx"].tolist():
            raise SystemExit(f"{run}: patches differ from the {weights_from} selection")
        w = wt["weight"].to_numpy(float)
        strata = wt["cell"].to_numpy()
    return df, w, strata


def per_variant(df: pd.DataFrame, w: np.ndarray, strata: np.ndarray, n_boot: int, seed: int
                ) -> dict:
    labels = np.array([json.loads(s) for s in df["raw_labels"]], dtype=object)
    confs = np.array([json.loads(s) for s in df["raw_confs"]], dtype=float)
    true = df["true_label_name"].to_numpy()
    idx_matrix = stratified_bootstrap_indices(strata, n_boot, seed)
    out = {}
    for k in range(K):
        correct = labels[:, k] == true
        conf = confs[:, k]

        def gap(ix: np.ndarray, conf=conf, correct=correct) -> float:
            return weighted_auc(conf[ix], correct[ix], w[ix]) - weighted_accuracy(correct[ix], w[ix])

        def auroc(ix: np.ndarray, conf=conf, correct=correct) -> float:
            return weighted_auroc(conf[ix], correct[ix], w[ix])

        full = np.arange(len(df))
        g_point, a_point = gap(full), auroc(full)
        g_boot = np.array([gap(ix) for ix in idx_matrix])
        a_boot = np.array([auroc(ix) for ix in idx_matrix])
        lo, hi = np.percentile(a_boot, [2.5, 97.5])
        out[f"variant_{k}"] = {
            "cohort_accuracy": round(weighted_accuracy(correct, w), 4),
            "cohort_auroc": round(a_point, 4),
            "cohort_auroc_ci": [round(float(lo), 4), round(float(hi), 4)],
            "cohort_auroc_exact": a_point,
            "cohort_selective_auc": round(weighted_auc(conf, correct, w), 4),
            "cohort_gap_vs_random": summarize_diffs(g_boot, g_point),
            "cohort_gap_exact": g_point,
            "subset_accuracy": round(float(correct.mean()), 4),
            "subset_auroc": round(auroc_midrank(conf, correct), 4),
            "n_distinct_confidences": len(np.unique(conf)),
            "mean_confidence": round(float(conf.mean()), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"seed={args.seed}  n_boot={args.n_boot}")

    result = {"seed": args.seed, "n_boot": args.n_boot, "runs": {}, "pending": []}
    for run, split, weights_from in RUNS:
        if not (CONS / run / "predictions.parquet").exists():
            result["pending"].append(run)
            print(f"{run}: pending (no predictions.parquet)")
            continue
        df, w, strata = load_run(run, weights_from)
        result["runs"][run] = {
            "split": split,
            "weights": "class share" if weights_from is None else f"selection weights of {weights_from}",
            "n_patches": len(df),
            "variants": per_variant(df, w, strata, args.n_boot, args.seed),
        }
        for k, v in result["runs"][run]["variants"].items():
            g = v["cohort_gap_vs_random"]
            print(f"{run:9s} {k}: acc {v['cohort_accuracy']:.3f}  AUROC {v['cohort_auroc']:.3f} "
                  f"{v['cohort_auroc_ci']}  gap {g['gap_point']:+.3f} [{g['ci_2.5']:.3f}, "
                  f"{g['ci_97.5']:.3f}]  subset AUROC {v['subset_auroc']:.3f}")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
