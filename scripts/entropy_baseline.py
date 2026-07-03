"""
scripts/entropy_baseline.py

Predictive-entropy-over-K routing baseline for the consistency experiment. No
API calls: reads the K=5 per-patch label lists already saved by
run_consistency.py.

Why this exists: the paper routes by the consistency score (fraction of K
queries agreeing with the mode). Predictive entropy over the K-label
distribution is the standard alternative uncertainty signal and a reviewer
asked for it explicitly. Consistency uses only the modal count; entropy uses the
full shape of the label distribution (e.g. a 3-1-1 split and a 3-2 split share a
modal count of 3 but differ in entropy). We compute the entropy signal, its
selective-accuracy AUC routing the modal-vote correctness, and a paired
bootstrap of consistency-minus-entropy on the identical patches.

Outputs:
    results/consistency/{model}/{version}/entropy_baseline.json

Usage:
    python scripts/entropy_baseline.py
    python scripts/entropy_baseline.py --version V5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import risk_coverage_curve, random_routing_curve

SEED = 42
N_BOOT = 1000


def predictive_entropy_bits(labels: list[str]) -> float:
    """Shannon entropy (bits) of the empirical label distribution over K queries."""
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def bootstrap_auc_diff(
    sig_a: np.ndarray, corr_a: np.ndarray,
    sig_b: np.ndarray, corr_b: np.ndarray,
) -> dict:
    """Paired bootstrap CI for AUC(A) - AUC(B), patch identity paired by index."""
    rng = np.random.default_rng(SEED)
    n = len(corr_a)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        auc_a = risk_coverage_curve(sig_a[idx], corr_a[idx])["auc"]
        auc_b = risk_coverage_curve(sig_b[idx], corr_b[idx])["auc"]
        diffs[i] = auc_a - auc_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": N_BOOT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Entropy-over-K routing baseline")
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="test analyzes the {version}_test consistency run.")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(":", "_")
    version_dir = args.version + ("_test" if args.split == "test" else "")
    cons_dir = PROJECT_ROOT / "results" / "consistency" / model_slug / version_dir
    df = pd.read_parquet(cons_dir / "predictions.parquet")

    labels_per_patch = [json.loads(s) for s in df["raw_labels"]]
    entropy = np.array([predictive_entropy_bits(lbls) for lbls in labels_per_patch], dtype=float)
    # Route by certainty: low entropy = confident = auto-confirm, so the signal
    # is negative entropy (higher = more certain), matching the router's
    # descending-confidence convention.
    entropy_signal = -entropy

    modal_correct = df["is_correct"].to_numpy(bool)
    consistency = df["consistency_score"].to_numpy(float)
    mean_conf = df["mean_textual_conf"].to_numpy(float)

    auc_entropy = risk_coverage_curve(entropy_signal, modal_correct)["auc"]
    auc_consistency = risk_coverage_curve(consistency, modal_correct)["auc"]
    auc_meanconf = risk_coverage_curve(mean_conf, modal_correct)["auc"]
    auc_random = random_routing_curve(modal_correct, seed=SEED)["auc"]

    boot_cons_vs_entropy = bootstrap_auc_diff(consistency, modal_correct, entropy_signal, modal_correct)
    boot_entropy_vs_meanconf = bootstrap_auc_diff(entropy_signal, modal_correct, mean_conf, modal_correct)
    boot_entropy_vs_random = bootstrap_auc_diff(
        entropy_signal, modal_correct,
        # A "random" signal paired per-patch: shuffle once with the base seed.
        np.random.default_rng(SEED).permutation(entropy_signal), modal_correct,
    )

    result = {
        "model": args.model,
        "version": args.version,
        "n_patches": int(len(df)),
        "mean_entropy_bits": round(float(entropy.mean()), 4),
        "routing_auc": {
            "consistency": round(float(auc_consistency), 4),
            "entropy_over_k": round(float(auc_entropy), 4),
            "mean_textual_conf": round(float(auc_meanconf), 4),
            "random": round(float(auc_random), 4),
        },
        "bootstrap": {
            "consistency_minus_entropy": boot_cons_vs_entropy,
            "entropy_minus_mean_conf": boot_entropy_vs_meanconf,
            "entropy_minus_random": boot_entropy_vs_random,
        },
    }

    out_path = cons_dir / "entropy_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== Entropy-over-K baseline | {args.model} {args.version} (n={len(df)}) ===")
    print(f"  AUC consistency:      {auc_consistency:.4f}")
    print(f"  AUC entropy-over-K:   {auc_entropy:.4f}")
    print(f"  AUC mean-textual-conf:{auc_meanconf:.4f}")
    print(f"  AUC random:           {auc_random:.4f}")
    print(f"  consistency - entropy: {boot_cons_vs_entropy['mean_diff']:+.4f} "
          f"[{boot_cons_vs_entropy['ci_2.5']:+.4f}, {boot_cons_vs_entropy['ci_97.5']:+.4f}] "
          f"{'excludes 0' if boot_cons_vs_entropy['excludes_zero'] else 'includes 0'}")
    print(f"  entropy - mean_conf:   {boot_entropy_vs_meanconf['mean_diff']:+.4f} "
          f"[{boot_entropy_vs_meanconf['ci_2.5']:+.4f}, {boot_entropy_vs_meanconf['ci_97.5']:+.4f}] "
          f"{'excludes 0' if boot_entropy_vs_meanconf['excludes_zero'] else 'includes 0'}")
    print(f"  entropy - random:      {boot_entropy_vs_random['mean_diff']:+.4f} "
          f"[{boot_entropy_vs_random['ci_2.5']:+.4f}, {boot_entropy_vs_random['ci_97.5']:+.4f}] "
          f"{'excludes 0' if boot_entropy_vs_random['excludes_zero'] else 'includes 0'}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
