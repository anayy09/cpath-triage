"""
scripts/confidence_information.py

Direct measurement of how much information the VLM verbalized confidence carries
about correctness. No API calls: reads the full-scale predictions already on
disk.

Why this exists: the draft inferred "negligible mutual information with
correctness" from the temperature-scaling fit saturating at T=200. That is an
overclaim and is contradicted by the paper's own routing result (Gemma-3
confidence routes above random, so its confidence cannot carry zero information).
Temperature scaling is monotone: it cannot create or destroy ranking
information, so a saturating T only says scaling cannot help, not that the
signal is empty. Here we measure the information directly:

  1. AUROC of confidence as a correctness detector (correct = positive class),
     with mid-rank tie handling and a bootstrap interval.
  2. Mutual information between binned confidence and the binary correctness
     outcome, in bits, against a permutation null.

Revision additions (items C-06, C-08, C-14b):
  - The confidence distribution and its tie fraction, because a signal that takes
    four distinct values with 96% of the mass on one of them has an AUROC pinned
    near 0.5 as a matter of arithmetic rather than as a finding about ranking.
  - Mean confidence and accuracy by *predicted* class. The submitted paper
    explained MedGemma's uninformative confidence by saying high confidence
    concentrates on a systematically wrong class (MUS). That mechanism predicts
    AUROC near 0.33; the measured value is 0.497. The per-class table shows why:
    the confidence is close to constant across classes, so there is no
    high-confidence subset to be wrong about.
  - Within-class (per predicted class) AUROC, which tests whether a model's
    apparent confidence signal is just a proxy for which class it predicted.
  - An explicit permutation null for MI, since the plug-in estimator has a
    positive chance floor that scales with n and with the number of occupied
    bins. Reporting MI against that null replaces arguing about the floor.

Reported on the full val and full test sets (the sets routed in Table 4) and on
the held-out evaluation partition (the set used for Table 3 ECE), so the number
lines up with each table.

Outputs:
    results/calibration/confidence_information.json
    results/calibration/confidence_by_class.json

Usage:
    python scripts/confidence_information.py [--n-boot 1000] [--n-perm 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import tie_statistics

SEED = 42
N_BINS = 15
N_BOOT = 1000
N_PERM = 1000
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")


def binary_entropy_bits(y: np.ndarray) -> float:
    """Shannon entropy of a binary array, in bits."""
    p = float(y.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def bin_confidence(conf: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-width binning on [0, 1]. The scheme the paper uses but never stated."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)


def confidence_mi_bits(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Mutual information (bits) between binned confidence and binary correctness.
    sklearn's mutual_info_score is in nats; convert to bits.
    """
    mi_nats = mutual_info_score(bin_confidence(conf, n_bins), correct.astype(int))
    return float(mi_nats / np.log(2))


def auroc_midrank(scores: np.ndarray, positive: np.ndarray) -> float:
    """
    AUROC with mid-rank tie handling (the Mann-Whitney statistic).

    Ties contribute 0.5 rather than being broken by input order. On these signals
    that is the difference between a measurement and an artifact.
    """
    pos = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auroc(scores: np.ndarray, positive: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = auroc_midrank(scores[idx], positive[idx])
    finite = vals[np.isfinite(vals)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_half": bool(lo > 0.5 or hi < 0.5),
        "n_boot": n_boot,
    }


def permutation_null_mi(
    conf: np.ndarray, correct: np.ndarray, n_perm: int, seed: int
) -> dict:
    """
    Null distribution of the plug-in MI under independence.

    The plug-in estimator is positively biased, so a small positive MI is not
    evidence of dependence. Shuffling correctness while holding the confidence
    distribution fixed gives the bias directly, which is a cleaner answer than
    arguing about an analytic chance floor.
    """
    rng = np.random.default_rng(seed)
    observed = confidence_mi_bits(conf, correct)
    y = correct.astype(int).copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = confidence_mi_bits(conf, rng.permutation(y))
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "observed_mi_bits": round(observed, 6),
        "null_mean_mi_bits": round(float(null.mean()), 6),
        "null_p95_mi_bits": round(float(np.percentile(null, 95)), 6),
        "p_value": round(p, 4),
        "significant_at_0.05": bool(p < 0.05),
        "n_perm": n_perm,
        "note": (
            "null_mean is the plug-in estimator's bias at this n and bin "
            "occupancy; the observed value should be read against it, not "
            "against zero."
        ),
    }


def vlm_frame(model: str, split: str) -> pd.DataFrame:
    suffix = "V3_full" if split == "val" else "V3_full_test"
    path = PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet"
    df = pd.read_parquet(path)
    df = df.assign(correct=df["pred_label"] == df["true_label_name"])
    return df


def eval_partition_frame(model: str) -> pd.DataFrame:
    """
    Reproduce the exact held-out evaluation partition used by calibrate.py
    (drop parse failures, np.random.default_rng(42) permutation, last 20%).
    """
    df = pd.read_parquet(
        PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full" / "predictions.parquet"
    )
    df = df[df["pred_label"] != "unknown"].copy()
    df["correct"] = df["pred_label"] == df["true_label_name"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    return df.iloc[idx[int(len(df) * 0.8):]]


def summarize(df: pd.DataFrame, n_boot: int, n_perm: int) -> dict:
    conf = df["pred_confidence"].to_numpy(float)
    correct = df["correct"].to_numpy(bool)
    h = binary_entropy_bits(correct)
    mi = confidence_mi_bits(conf, correct)
    auroc = auroc_midrank(conf, correct)
    ties = tie_statistics(conf)
    occupied = int(len(np.unique(bin_confidence(conf))))
    return {
        "n": int(len(df)),
        "accuracy": round(float(correct.mean()), 4),
        "conf_correct_auroc": round(auroc, 4),
        "conf_correct_auroc_bootstrap": bootstrap_auroc(conf, correct, n_boot, SEED),
        "auroc_tie_handling": "mid-rank (Mann-Whitney)",
        "tie_fraction": round(float(ties["tie_fraction"]), 4),
        "n_distinct_confidences": ties["n_distinct"],
        "largest_tie_group_fraction": round(float(ties["largest_tie_group_fraction"]), 4),
        "mi_bits": round(mi, 5),
        "mi_estimator": "plug-in",
        "mi_binning": f"{N_BINS} equal-width bins on [0, 1]",
        "mi_occupied_bins": occupied,
        "mi_permutation_null": permutation_null_mi(conf, correct, n_perm, SEED),
        "h_correct_bits": round(h, 4),
        "mi_normalized": round(mi / h, 4) if h > 0 else None,
    }


def by_predicted_class(df: pd.DataFrame) -> dict:
    """
    Mean confidence and accuracy conditioned on the predicted class, plus the
    within-class AUROC.

    This is the direct test of the mechanism the paper asserted and of Reviewer
    2's worry that an apparent confidence signal is a proxy for class identity.
    """
    rows = {}
    for label, g in df.groupby("pred_label"):
        conf = g["pred_confidence"].to_numpy(float)
        corr = g["correct"].to_numpy(bool)
        rows[str(label)] = {
            "n": int(len(g)),
            "share_of_predictions": round(float(len(g) / len(df)), 4),
            "accuracy": round(float(corr.mean()), 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "sd_confidence": round(float(conf.std(ddof=1)) if len(g) > 1 else 0.0, 4),
            "n_distinct_confidences": int(len(np.unique(conf))),
            "within_class_auroc": (
                round(auroc_midrank(conf, corr), 4) if 0 < corr.sum() < len(corr) else None
            ),
        }
    ordered = dict(sorted(rows.items(), key=lambda kv: -kv[1]["n"]))
    means = [v["mean_confidence"] for v in ordered.values()]
    return {
        "by_predicted_class": ordered,
        "spread_of_mean_confidence_across_classes": round(float(max(means) - min(means)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()

    out: dict[str, dict] = {}
    by_class: dict[str, dict] = {}

    for model in MODELS:
        frames = {
            "val_full": vlm_frame(model, "val"),
            "test_full": vlm_frame(model, "test"),
            "eval_partition": eval_partition_frame(model),
        }
        out[model] = {
            part: summarize(f, args.n_boot, args.n_perm) for part, f in frames.items()
        }
        by_class[model] = {
            part: by_predicted_class(f)
            for part, f in frames.items()
            if part != "eval_partition"
        }
        by_class[model]["confidence_distribution"] = {
            part: {
                str(k): int(v)
                for k, v in sorted(
                    Counter(np.round(f["pred_confidence"].to_numpy(float), 4).tolist()).items(),
                    reverse=True,
                )
            }
            for part, f in frames.items()
            if part != "eval_partition"
        }

        for part, s in out[model].items():
            b = s["conf_correct_auroc_bootstrap"]
            pn = s["mi_permutation_null"]
            print(
                f"{model:16s} {part:15s} n={s['n']:5d}  acc={s['accuracy']:.4f}  "
                f"AUROC={s['conf_correct_auroc']:.4f} "
                f"[{b['ci_2.5']:.4f}, {b['ci_97.5']:.4f}]  "
                f"ties={s['tie_fraction']:.3f} ({s['n_distinct_confidences']} distinct)  "
                f"MI={s['mi_bits']:.5f} vs null {pn['null_mean_mi_bits']:.5f} "
                f"(p={pn['p_value']})"
            )

    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    info_path = out_dir / "confidence_information.json"
    info_path.write_text(
        json.dumps(
            {"n_bins": N_BINS, "seed": SEED, "n_boot": args.n_boot,
             "n_perm": args.n_perm, "results": out},
            indent=2,
        ),
        encoding="utf-8",
    )

    class_path = out_dir / "confidence_by_class.json"
    class_path.write_text(json.dumps({"seed": SEED, "results": by_class}, indent=2), encoding="utf-8")

    print("\n=== Mean confidence by predicted class (validation) ===")
    for model in MODELS:
        rows = by_class[model]["val_full"]["by_predicted_class"]
        spread = by_class[model]["val_full"]["spread_of_mean_confidence_across_classes"]
        print(f"\n{model}  (spread of class mean confidence: {spread:.4f})")
        print(f"  {'pred class':<38}{'n':>7}{'acc':>8}{'meanconf':>10}{'within-AUROC':>14}")
        for label, r in list(rows.items())[:6]:
            wa = r["within_class_auroc"]
            print(f"  {label:<38}{r['n']:>7}{r['accuracy']:>8.3f}{r['mean_confidence']:>10.4f}"
                  f"{(f'{wa:.3f}' if wa is not None else 'n/a'):>14}")

    print(f"\nSaved: {info_path}")
    print(f"Saved: {class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
