"""
scripts/confusability_weighted.py

Confusability-weighted agreement: a dispersion signal over the K=5 responses that
is not a monotone function of the modal fraction (revision item C-02).

Why this exists. The paper offered predictive entropy over K as independent
corroboration of the consistency score. It is not: with K=5 the attainable
entropies are 0.000, 0.722, {0.971, 1.371}, {1.522, 1.922} and 2.322 bits, and
those ranges do not overlap across consecutive modal counts, so negative entropy
is an order-preserving relabelling of the modal fraction. The measured difference
of -0.000 was the identity, not a finding. Reviewer 1 asked for a signal that is
genuinely not a function of the modal count.

This is that signal. Each dissenting response is charged according to how
confusable its label is with the modal label, so two patches that both split 3/2
score differently depending on whether the dissent went to a neighbouring tissue
class or to an unrelated one.

Confusability source. The weights come from the ResNet-18 64 px confusion matrix
on the *external test* split, not the validation split. Validation accuracy is
0.9971, so that confusion matrix is nearly diagonal and carries almost no
confusability information, and it is the same estimate that revision item C-05
flags as leakage-contaminated. The external test matrix (accuracy 0.9109) has
real off-diagonal mass. The matrix is symmetrised and row-normalised, then scaled
so the most confusable pair carries weight 1.

Control. The same analysis is run with a uniform weight matrix, under which the
score becomes an affine function of the modal count and therefore reproduces
plain consistency exactly. Reporting both shows how much of the effect the
weighting contributes.

Outputs:
    results/consistency/{model}/{run}/confusability_weighted.json

Usage:
    python scripts/confusability_weighted.py --split val
    python scripts/confusability_weighted.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES
from src.triage.router import random_routing_curve, risk_coverage_curve

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
N_BOOT = 1000
N_TIE_REPEATS = 200


def build_confusability(cnn_test_pred: Path) -> tuple[np.ndarray, dict]:
    """
    Symmetrised, row-normalised, max-scaled off-diagonal confusability matrix.

    Returns the 9x9 matrix (diagonal 1.0) and a small provenance record.
    """
    df = pd.read_parquet(cnn_test_pred)
    n = len(ALL_LABELS)
    idx = {name: i for i, name in enumerate(ALL_LABELS)}

    counts = np.zeros((n, n), dtype=float)
    for true_name, pred_name in zip(df["true_label_name"], df["pred_label_name"]):
        counts[idx[str(true_name)], idx[str(pred_name)]] += 1.0

    sym = (counts + counts.T) / 2.0
    np.fill_diagonal(sym, 0.0)

    row_sums = sym.sum(axis=1, keepdims=True)
    # A class the CNN never confuses would divide by zero; leave its row at zero,
    # which charges any dissent to it the full amount.
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(row_sums > 0, sym / row_sums, 0.0)

    peak = float(normed.max())
    scaled = normed / peak if peak > 0 else normed
    np.fill_diagonal(scaled, 1.0)

    provenance = {
        "source": str(cnn_test_pred.relative_to(PROJECT_ROOT)),
        "source_accuracy": round(float(df["correct"].mean()), 4),
        "n_samples": len(df),
        "peak_offdiagonal_before_scaling": round(peak, 6),
        "most_confusable_pairs": _top_pairs(scaled, k=5),
    }
    return scaled, provenance


def _top_pairs(m: np.ndarray, k: int = 5) -> list[dict]:
    pairs = []
    n = len(ALL_LABELS)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float((m[i, j] + m[j, i]) / 2.0), ALL_LABELS[i], ALL_LABELS[j]))
    pairs.sort(reverse=True)
    return [{"a": a, "b": b, "weight": round(w, 4)} for w, a, b in pairs[:k]]


def weighted_agreement(
    labels_per_patch: list[list[str]], modal_labels: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Mean weight of each response against the modal label.

    Identical to the consistency score when the off-diagonal weights are zero,
    and strictly finer-grained otherwise.
    """
    idx = {name: i for i, name in enumerate(ALL_LABELS)}
    out = np.empty(len(labels_per_patch), dtype=float)
    for p, (labels, modal) in enumerate(zip(labels_per_patch, modal_labels)):
        m = idx.get(str(modal))
        total = 0.0
        for lab in labels:
            i = idx.get(str(lab))
            if m is None or i is None:
                total += 1.0 if str(lab) == str(modal) else 0.0
            else:
                total += float(weights[m, i])
        out[p] = total / len(labels)
    return out


def bootstrap_diff(
    sig_a: np.ndarray, sig_b: np.ndarray, correct: np.ndarray, n_boot: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (
            risk_coverage_curve(sig_a[idx], correct[idx], tie_break="expected")["auc"]
            - risk_coverage_curve(sig_b[idx], correct[idx], tie_break="expected")["auc"]
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(":", "_")
    run_dir = args.version + ("_test" if args.split == "test" else "")
    cons_dir = PROJECT_ROOT / "results" / "consistency" / model_slug / run_dir
    pred_path = cons_dir / "predictions.parquet"
    cnn_test = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "test_predictions.parquet"

    for p in (pred_path, cnn_test):
        if not p.exists():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    weights, provenance = build_confusability(cnn_test)
    uniform = np.eye(len(ALL_LABELS), dtype=float)

    df = pd.read_parquet(pred_path)
    labels_per_patch = [json.loads(s) for s in df["raw_labels"]]
    modal = df["modal_label"].to_numpy()
    correct = df["is_correct"].to_numpy(bool)
    consistency = df["consistency_score"].to_numpy(float)

    sig_weighted = weighted_agreement(labels_per_patch, modal, weights)
    sig_uniform = weighted_agreement(labels_per_patch, modal, uniform)

    # The uniform control must reproduce the stored consistency score exactly.
    # If it does not, the modal label and the response list disagree and every
    # number below is suspect.
    max_dev = float(np.abs(sig_uniform - consistency).max())

    random_auc = random_routing_curve(correct, seed=args.seed)["auc"]
    curves = {}
    for name, sig in (
        ("confusability_weighted", sig_weighted),
        ("uniform_weight_control", sig_uniform),
        ("consistency_score", consistency),
    ):
        c = risk_coverage_curve(
            sig, correct, tie_break="random", n_repeats=N_TIE_REPEATS, seed=args.seed
        )
        curves[name] = {
            "auc": round(float(c["auc"]), 4),
            "auc_sd_over_tie_orders": round(float(c["auc_sd"]), 4),
            "tie_fraction": round(float(c["tie_fraction"]), 4),
            "n_distinct": int(c["n_distinct"]),
            "gap_vs_random": round(float(c["auc"] - random_auc), 4),
        }

    contrasts = {
        "weighted_minus_consistency": bootstrap_diff(
            sig_weighted, consistency, correct, args.n_boot, args.seed
        ),
        "weighted_minus_random": bootstrap_diff(
            sig_weighted,
            np.random.default_rng(args.seed).permutation(sig_weighted),
            correct,
            args.n_boot,
            args.seed,
        ),
    }

    result = {
        "model": args.model,
        "version": args.version,
        "split": args.split,
        "n_patches": len(df),
        "seed": args.seed,
        "confusability_matrix": {
            "labels": ALL_LABELS,
            "matrix": [[round(float(v), 6) for v in row] for row in weights],
            **provenance,
        },
        "uniform_control_reproduces_consistency": {
            "max_abs_deviation": round(max_dev, 12),
            "exact": bool(max_dev < 1e-12),
        },
        "random_routing_auc": round(float(random_auc), 4),
        "signals": curves,
        "contrasts": contrasts,
    }

    out_path = cons_dir / "confusability_weighted.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== Confusability-weighted agreement | {args.model} {args.version} "
          f"{args.split} (n={len(df)}) ===")
    print(f"  confusability from: {provenance['source']} "
          f"(accuracy {provenance['source_accuracy']})")
    print("  most confusable pairs: " + ", ".join(
        f"{p['a']}/{p['b']} {p['weight']:.2f}" for p in provenance["most_confusable_pairs"][:3]
    ))
    print(f"  uniform control reproduces consistency exactly: "
          f"{result['uniform_control_reproduces_consistency']['exact']} "
          f"(max dev {max_dev:.2e})")
    print()
    for name, c in curves.items():
        print(f"  {name:<26} AUC {c['auc']:.4f} (sd {c['auc_sd_over_tie_orders']:.4f}, "
              f"{c['n_distinct']} distinct)  vs random {c['gap_vs_random']:+.4f}")
    print(f"  {'random':<26} AUC {random_auc:.4f}")
    print()
    for name, c in contrasts.items():
        print(f"  {name:<28} {c['mean_diff']:+.4f} [{c['ci_2.5']:+.4f}, {c['ci_97.5']:+.4f}]  "
              f"{'excludes 0' if c['excludes_zero'] else 'includes 0'}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
