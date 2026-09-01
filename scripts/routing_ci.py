"""
scripts/routing_ci.py

Paired-bootstrap confidence intervals for the routing (selective-accuracy) AUC
gaps reported in Table 4. No API calls: reads the predictions already on disk.

Why this exists: Table 4 reported point estimates of the selective-accuracy AUC
for calibrated routing and for random routing, but no uncertainty on the gap.
A reviewer cannot tell whether "MedGemma 0.395 vs 0.403 random" differs from
noise. This script bootstraps the gap (calibrated minus random) with patch
identity paired within each resample, exactly as the consistency experiment
already does for Table 8.

Metric orientation (see paper Section on the routing policy): the reported
quantity is the area under the auto-confirm-accuracy vs coverage curve. Higher
is better. A non-informative (random) confidence signal integrates to the
overall accuracy, so the natural reference for the gap is the base accuracy of
the set. We confirm empirically that random-routing AUC equals base accuracy
(to within averaging noise) and use base accuracy as the paired random
reference inside the bootstrap.

Revision changes (items C-01, C-06, C-09):
  - Tie handling is explicit. The previous point estimates ranked tied
    confidences by row order, which on these signals is most of the ranking:
    MedGemma's full-scale confidence takes four distinct values with 96% at one
    of them. The bootstrap was unaffected because resampling incidentally
    randomises tie order, which is exactly why the published gap column (the
    bootstrap mean) did not reconcile with the published AUC columns (row-order
    point estimates) on the two external-test rows.
  - The missing CNN-64 validation cell and both CNN-224 rows are computed.
  - A predicted-class-prior routing baseline is added: rank each patch by the
    validation accuracy of the class the model predicted for it. If a model's
    confidence cannot beat knowing nothing but its own predicted label, that is
    worth reporting plainly.

Outputs:
    results/routing/routing_auc_ci.json

Usage:
    python scripts/routing_ci.py [--n-boot 1000] [--seed 42]
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

from src.triage.router import (
    _trapezoid,
    random_routing_curve,
    risk_coverage_curve,
    tie_statistics,
)

N_BOOT = 1000
SEED = 42
VLM_MODELS = ("medgemma-27b-it", "gemma-3-27b-it")


def acc_coverage_auc(conf: np.ndarray, correct: np.ndarray, n_points: int = 101) -> float:
    """
    Area under the auto-confirm-accuracy vs coverage curve, resolving ties by
    their exact expectation over random tie orders.

    Replicates src.triage.router.risk_coverage_curve(tie_break="expected") on the
    same budget grid with the same rounding and the same trapezoid over the
    non-empty prefix, but inline so the bootstrap stays cheap. Verified against
    the router implementation in main() before use.

    Ties are handled by grouping equal confidences: a prefix of length k always
    contains every patch above the cut and fills the remainder with the tied
    group's mean correctness, which is what a uniformly random tie order gives
    in expectation.
    """
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    sorted_correct = correct[order].astype(float)

    _, starts, counts = np.unique(-sorted_conf, return_index=True, return_counts=True)
    group_correct = np.add.reduceat(sorted_correct, starts)
    cum_count = np.concatenate([[0], np.cumsum(counts)])
    cum_correct = np.concatenate([[0.0], np.cumsum(group_correct)])
    rates = group_correct / counts

    budgets = np.linspace(0.0, 1.0, n_points)
    accs = []
    for b in budgets:
        k = round((1.0 - b) * n)
        if k <= 0:
            accs.append(np.nan)
            continue
        j = max(int(np.searchsorted(cum_count, k, side="left")), 1)
        accs.append((cum_correct[j - 1] + (k - cum_count[j - 1]) * rates[j - 1]) / k)

    accs = np.array(accs, dtype=float)
    valid = accs[~np.isnan(accs)]
    if len(valid) <= 1:
        return float("nan")
    return float(_trapezoid(valid, dx=1.0 / (len(valid) - 1)))


def load_signal(model: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (confidence, correct, predicted_label) for a model/split.

    The predicted label is returned so the class-prior baseline can be built
    without re-reading the parquet.
    """
    if model.startswith("resnet18"):
        path = PROJECT_ROOT / "results" / "cnn" / model / f"{split}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        df = pd.read_parquet(path)
        return (
            df["confidence"].to_numpy(float),
            df["correct"].to_numpy(bool),
            df["pred_label_name"].to_numpy(),
        )

    # VLMs: full-scale predictions. Parse failures ("unknown") are kept and
    # scored as incorrect, matching the paper's routing evaluation.
    suffix = "V3_full" if split == "val" else "V3_full_test"
    path = PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_parquet(path)
    conf = df["pred_confidence"].to_numpy(float)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    return conf, correct, df["pred_label"].to_numpy()


def class_prior_signal(
    pred_labels: np.ndarray, prior: dict[str, float], default: float
) -> np.ndarray:
    """Rank a patch by the historical accuracy of the class the model predicted."""
    return np.array([prior.get(str(p), default) for p in pred_labels], dtype=float)


def bootstrap_gap(
    conf: np.ndarray, correct: np.ndarray, n_boot: int, seed: int
) -> dict:
    """
    Paired bootstrap of (selective-accuracy AUC) minus (random-routing AUC).

    Random-routing AUC equals base accuracy in expectation, so within each
    resample the random reference is the resample's mean correctness. Patch
    identity is held paired by resampling indices once per iteration.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c, y = conf[idx], correct[idx]
        diffs[i] = acc_coverage_auc(c, y) - y.mean()
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
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    targets = [
        ("medgemma-27b-it", "val"),
        ("medgemma-27b-it", "test"),
        ("gemma-3-27b-it", "val"),
        ("gemma-3-27b-it", "test"),
        ("resnet18_64px", "val"),
        ("resnet18_64px", "test"),
        ("resnet18_224px", "val"),
        ("resnet18_224px", "test"),
    ]

    # Per-model class-accuracy prior, estimated on validation and applied to both
    # splits. On validation this is in-sample and therefore optimistic; that is
    # stated rather than corrected, because the baseline only has to be hard
    # enough to be informative about the confidence signal.
    priors: dict[str, tuple[dict[str, float], float]] = {}
    for model in (*VLM_MODELS, "resnet18_64px", "resnet18_224px"):
        _, correct_v, pred_v = load_signal(model, "val")
        df = pd.DataFrame({"pred": pred_v, "correct": correct_v})
        priors[model] = (
            df.groupby("pred")["correct"].mean().to_dict(),
            float(correct_v.mean()),
        )

    results: dict[str, dict] = {}
    for model, split in targets:
        conf, correct, pred_labels = load_signal(model, split)

        # Point estimates. Cross-check the fast AUC against the router module.
        router_out = risk_coverage_curve(conf, correct, tie_break="expected")
        cal_auc = router_out["auc"]
        cal_auc_fast = acc_coverage_auc(conf, correct)
        assert abs(cal_auc - cal_auc_fast) < 1e-9, (
            f"AUC mismatch for {model}/{split}: {cal_auc} vs {cal_auc_fast}"
        )
        mc = risk_coverage_curve(conf, correct, tie_break="random",
                                 n_repeats=200, seed=args.seed)
        rnd_auc = random_routing_curve(correct, seed=args.seed)["auc"]
        base_acc = float(correct.mean())
        ties = tie_statistics(conf)

        boot = bootstrap_gap(conf, correct, args.n_boot, args.seed)

        prior_map, prior_default = priors[model]
        prior_conf = class_prior_signal(pred_labels, prior_map, prior_default)
        prior_auc = acc_coverage_auc(prior_conf, correct)
        prior_boot = bootstrap_gap(prior_conf, correct, args.n_boot, args.seed)

        key = f"{model}|{split}"
        results[key] = {
            "model": model,
            "split": split,
            "n": len(correct),
            "base_accuracy": round(base_acc, 4),
            "cal_auc": round(float(cal_auc), 4),
            "cal_auc_tie_sd": round(float(mc["auc_sd"]), 4),
            "cal_auc_row_order_superseded": round(float(router_out["auc_row_order"]), 4),
            "random_auc": round(float(rnd_auc), 4),
            "gap_point": round(float(cal_auc - rnd_auc), 4),
            "gap_bootstrap": boot,
            "tie_fraction": round(float(ties["tie_fraction"]), 4),
            "n_distinct_confidences": ties["n_distinct"],
            "largest_tie_group_fraction": round(float(ties["largest_tie_group_fraction"]), 4),
            "class_prior_baseline": {
                "auc": round(float(prior_auc), 4),
                "gap_point": round(float(prior_auc - rnd_auc), 4),
                "gap_bootstrap": prior_boot,
                "prior_estimated_on": "val",
                "in_sample": split == "val",
            },
        }
        print(
            f"{key:26s} n={len(correct):5d}  cal={cal_auc:.4f}"
            f" (tie sd {mc['auc_sd']:.4f}, was {router_out['auc_row_order']:.4f})  "
            f"rnd={rnd_auc:.4f}  gap={cal_auc - rnd_auc:+.4f} "
            f"[{boot['ci_2.5']:+.4f}, {boot['ci_97.5']:+.4f}]  "
            f"{'excl 0' if boot['excludes_zero'] else 'incl 0'}  "
            f"ties={ties['tie_fraction']:.3f} ({ties['n_distinct']} distinct)  "
            f"prior={prior_auc:.4f}"
        )

    out_dir = PROJECT_ROOT / "results" / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "routing_auc_ci.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "tie_break": "expected",
                "note": (
                    "cal_auc resolves ties by their exact expectation over random tie "
                    "orders. cal_auc_row_order_superseded is the pre-revision value, "
                    "which ranked tied confidences by row order and is reported only so "
                    "the size of that defect can be quoted."
                ),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
