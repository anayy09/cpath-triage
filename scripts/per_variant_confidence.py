"""
scripts/per_variant_confidence.py

Per-variant accuracy, confidence and confidence-correctness AUROC for the K=5
consistency experiment (revision item C-10).

Why this exists. The paper pooled five confidence values per patch into a
mean-of-5 and then attributed the resulting signal's uselessness to the model.
Reviewer 1 objected that the five values are not measurements on a common scale,
and inspecting the variants shows the objection is right: variant 4 appends an
explicit instruction to lower CONFIDENCE when torn between two classes, which
changes the elicitation rather than the underlying belief. Averaging across that
boundary mixes two different questions.

The five variants, in the order run_consistency.py generates them:
    0  base prompt, unmodified. The in-session single-query baseline.
    1  architectural-focus instruction prepended.
    2  nine classes listed alphabetically, to test listing-order bias.
    3  clinical context added ("colorectal surgical specimen").
    4  uncertainty reminder appended, which explicitly rescales CONFIDENCE.

For each variant this reports accuracy, mean and distribution of the verbalized
confidence, the tie fraction, and the confidence-correctness AUROC with a
bootstrap interval. AUROC uses mid-ranks, so ties are handled explicitly rather
than by whatever order the rows happen to be in.

Outputs:
    results/consistency/{model}/{run}/per_variant.json

Usage:
    python scripts/per_variant_confidence.py --split val
    python scripts/per_variant_confidence.py --split test
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

from src.triage.router import risk_coverage_curve, tie_statistics

N_BOOT = 1000

VARIANT_DESCRIPTIONS = {
    0: "base prompt, unmodified (in-session single-query baseline)",
    1: "architectural-focus instruction prepended",
    2: "class list in alphabetical order",
    3: "clinical context added",
    4: "uncertainty reminder appended (rescales the confidence elicitation)",
}


def auroc_midrank(scores: np.ndarray, positive: np.ndarray) -> float:
    """
    AUROC with mid-rank tie handling, equivalent to the Mann-Whitney statistic.

    Ties contribute 0.5 rather than being resolved by input order, which matters
    here: these confidence signals are mostly ties.
    """
    pos = np.asarray(positive, dtype=bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Replace each tied block with its mean rank.
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
    if not pred_path.exists():
        print(f"Consistency predictions not found: {pred_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(pred_path)
    labels_per_patch = [json.loads(s) for s in df["raw_labels"]]
    confs_per_patch = [json.loads(s) for s in df["raw_confs"]]
    truth = df["true_label_name"].to_numpy()
    k = len(labels_per_patch[0])

    variants: dict[str, dict] = {}
    for v in range(k):
        pred_v = np.array([lab[v] for lab in labels_per_patch])
        conf_v = np.array([c[v] for c in confs_per_patch], dtype=float)
        correct_v = pred_v == truth

        ties = tie_statistics(conf_v)
        auroc = auroc_midrank(conf_v, correct_v)
        boot = bootstrap_auroc(conf_v, correct_v, args.n_boot, args.seed)
        curve = risk_coverage_curve(
            conf_v, correct_v, tie_break="random", n_repeats=200, seed=args.seed
        )
        dist = Counter(np.round(conf_v, 4).tolist())

        variants[f"variant_{v}"] = {
            "description": VARIANT_DESCRIPTIONS.get(v, ""),
            "accuracy": round(float(correct_v.mean()), 4),
            "mean_confidence": round(float(conf_v.mean()), 4),
            "sd_confidence": round(float(conf_v.std(ddof=1)), 4),
            "min_confidence": round(float(conf_v.min()), 4),
            "max_confidence": round(float(conf_v.max()), 4),
            "n_distinct_confidences": ties["n_distinct"],
            "tie_fraction": round(float(ties["tie_fraction"]), 4),
            "largest_tie_group_fraction": round(float(ties["largest_tie_group_fraction"]), 4),
            "confidence_distribution": {
                str(kk): int(vv) for kk, vv in sorted(dist.items(), reverse=True)
            },
            "auroc_conf_vs_correct_midrank": round(float(auroc), 4),
            "auroc_bootstrap": boot,
            "selective_accuracy_auc": round(float(curve["auc"]), 4),
            "selective_accuracy_auc_sd_over_tie_orders": round(float(curve["auc_sd"]), 4),
        }

    mean_conf = df["mean_textual_conf"].to_numpy(float)
    spread = {
        "accuracy_range_across_variants": round(
            float(max(v["accuracy"] for v in variants.values())
                  - min(v["accuracy"] for v in variants.values())), 4
        ),
        "mean_confidence_range_across_variants": round(
            float(max(v["mean_confidence"] for v in variants.values())
                  - min(v["mean_confidence"] for v in variants.values())), 4
        ),
        "pooled_mean_of_5_confidence": round(float(mean_conf.mean()), 4),
    }

    result = {
        "model": args.model,
        "version": args.version,
        "split": args.split,
        "n_patches": len(df),
        "k": k,
        "seed": args.seed,
        "variants": variants,
        "across_variant_spread": spread,
        "note": (
            "Variant 4 appends an instruction to lower CONFIDENCE under class "
            "ambiguity, so its confidence values are elicited on a different scale "
            "from the other four. The mean-of-5 confidence pools across that "
            "boundary, which is why the pooled signal should not be read as a "
            "measurement of the model's confidence quality."
        ),
    }

    out_path = cons_dir / "per_variant.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== Per-variant confidence | {args.model} {args.version} {args.split} "
          f"(n={len(df)}, K={k}) ===")
    print(f"{'variant':<9}{'acc':>8}{'meanconf':>10}{'distinct':>9}{'ties':>8}"
          f"{'AUROC':>8}{'95% CI':>18}")
    for name, v in variants.items():
        b = v["auroc_bootstrap"]
        print(f"{name:<9}{v['accuracy']:>8.4f}{v['mean_confidence']:>10.4f}"
              f"{v['n_distinct_confidences']:>9d}{v['tie_fraction']:>8.3f}"
              f"{v['auroc_conf_vs_correct_midrank']:>8.4f}"
              f"  [{b['ci_2.5']:.4f}, {b['ci_97.5']:.4f}]"
              f"{'' if b['excludes_half'] else '  (incl 0.5)'}")
    print(f"\n  accuracy range across variants:        "
          f"{spread['accuracy_range_across_variants']:.4f}")
    print(f"  mean-confidence range across variants: "
          f"{spread['mean_confidence_range_across_variants']:.4f}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
