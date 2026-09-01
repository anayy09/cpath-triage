"""
scripts/consistency_routing_table.py

Regenerate the routing-signal comparison behind Tables 8 and 9 from the saved
K=5 predictions. No API calls.

This replaces the routing_auc block written by run_consistency.py at inference
time, which used the superseded row-order tie handling. Every AUC here averages
over random tie orders and reports the spread that ordering induces, because
these signals are coarse: the consistency score takes 5 distinct values and
mean-of-5 confidence takes 10, with most of the mass on one of them.

Signals compared (all route the K=5 modal-vote outcome unless noted):
  consistency_score        modal fraction over K responses
  entropy_over_k           negative predictive entropy, an order-preserving
                           relabelling of the modal fraction rather than an
                           independent signal (revision item C-02)
  mean_textual_conf        mean of the five verbalized confidences
  mean_textual_conf_flip   the same signal negated. Reviewer 1 asked for this on
                           the reading that a below-random signal is exploitable;
                           it is reported so the number is on the record
                           (revision item C-10)
  single_query_conf        the variant-0 confidence, routing its own outcome
  random                   uniform random routing baseline

Outputs:
    results/consistency/{model}/{run}/routing_signals.json

Usage:
    python scripts/consistency_routing_table.py --split val
    python scripts/consistency_routing_table.py --split test
    python scripts/consistency_routing_table.py --version V5 --split val
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

from src.triage.router import random_routing_curve, risk_coverage_curve

N_BOOT = 1000
N_TIE_REPEATS = 200


def predictive_entropy_bits(labels: list[str]) -> float:
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def bootstrap_diff(
    sig_a: np.ndarray,
    corr_a: np.ndarray,
    sig_b: np.ndarray,
    corr_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired bootstrap CI for AUC(A) - AUC(B), patch identity paired by index."""
    rng = np.random.default_rng(seed)
    n = len(corr_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (
            risk_coverage_curve(sig_a[idx], corr_a[idx], tie_break="expected")["auc"]
            - risk_coverage_curve(sig_b[idx], corr_b[idx], tie_break="expected")["auc"]
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
    if not pred_path.exists():
        print(f"Consistency predictions not found: {pred_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(pred_path)
    modal_correct = df["is_correct"].to_numpy(bool)
    labels_per_patch = [json.loads(s) for s in df["raw_labels"]]
    confs_per_patch = [json.loads(s) for s in df["raw_confs"]]

    signals: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "consistency_score": (df["consistency_score"].to_numpy(float), modal_correct),
        "entropy_over_k": (
            -np.array([predictive_entropy_bits(x) for x in labels_per_patch]),
            modal_correct,
        ),
        "mean_textual_conf": (df["mean_textual_conf"].to_numpy(float), modal_correct),
        "mean_textual_conf_flip": (-df["mean_textual_conf"].to_numpy(float), modal_correct),
    }

    # The variant-0 query is the unmodified base prompt run inside the same K=5
    # session, so it is the in-session single-query baseline. It routes its own
    # outcome, not the modal vote.
    v0_pred = np.array([x[0] for x in labels_per_patch])
    v0_conf = np.array([c[0] for c in confs_per_patch], dtype=float)
    v0_correct = v0_pred == df["true_label_name"].to_numpy()
    signals["single_query_conf"] = (v0_conf, v0_correct)

    random_auc = random_routing_curve(modal_correct, seed=args.seed)["auc"]
    random_auc_v0 = random_routing_curve(v0_correct, seed=args.seed)["auc"]

    rows: dict[str, dict] = {}
    for name, (sig, corr) in signals.items():
        curve = risk_coverage_curve(
            sig, corr, tie_break="random", n_repeats=N_TIE_REPEATS, seed=args.seed
        )
        ref = random_auc_v0 if name == "single_query_conf" else random_auc
        rows[name] = {
            "auc": round(float(curve["auc"]), 4),
            "auc_sd_over_tie_orders": round(float(curve["auc_sd"]), 4),
            "auc_expected_closed_form": round(float(curve["auc_expected"]), 4),
            "auc_row_order_superseded": round(float(curve["auc_row_order"]), 4),
            "tie_fraction": round(float(curve["tie_fraction"]), 4),
            "n_distinct": int(curve["n_distinct"]),
            "random_reference": round(float(ref), 4),
            "gap_vs_random": round(float(curve["auc"] - ref), 4),
            "routes_outcome": "single_query" if name == "single_query_conf" else "modal_vote",
        }

    # Contrasts the paper makes explicitly.
    cons_sig = signals["consistency_score"][0]
    contrasts = {
        "consistency_minus_mean_textual_conf": bootstrap_diff(
            cons_sig, modal_correct,
            signals["mean_textual_conf"][0], modal_correct,
            args.n_boot, args.seed,
        ),
        "consistency_minus_single_query_conf": bootstrap_diff(
            cons_sig, modal_correct, v0_conf, v0_correct, args.n_boot, args.seed
        ),
        "consistency_minus_entropy": bootstrap_diff(
            cons_sig, modal_correct,
            signals["entropy_over_k"][0], modal_correct,
            args.n_boot, args.seed,
        ),
        "mean_textual_conf_minus_random": bootstrap_diff(
            signals["mean_textual_conf"][0], modal_correct,
            np.random.default_rng(args.seed).permutation(signals["mean_textual_conf"][0]),
            modal_correct, args.n_boot, args.seed,
        ),
        "mean_textual_conf_flip_minus_random": bootstrap_diff(
            signals["mean_textual_conf_flip"][0], modal_correct,
            np.random.default_rng(args.seed).permutation(signals["mean_textual_conf"][0]),
            modal_correct, args.n_boot, args.seed,
        ),
    }

    # Surfaced because the paper's constructive claim is that consistency is the
    # usable signal, so how well it behaves as a probability is relevant
    # (revision item C-I7).
    metrics_path = cons_dir / "metrics.json"
    ece_consistency = None
    if metrics_path.exists():
        ece_consistency = json.loads(metrics_path.read_text()).get(
            "ece_consistency_as_confidence"
        )

    result = {
        "model": args.model,
        "version": args.version,
        "split": args.split,
        "n_patches": len(df),
        "seed": args.seed,
        "tie_handling": {
            "tie_break": "random",
            "n_repeats": N_TIE_REPEATS,
            "note": (
                "Point estimates average the AUC over random tie orders; "
                "auc_row_order_superseded is what the pre-revision code reported, "
                "which ranked tied values by row position."
            ),
        },
        "random_routing_auc": {
            "modal_vote_outcome": round(float(random_auc), 4),
            "single_query_outcome": round(float(random_auc_v0), 4),
        },
        "signals": rows,
        "contrasts": contrasts,
        "ece_consistency_as_confidence": ece_consistency,
    }

    out_path = cons_dir / "routing_signals.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== Routing signals | {args.model} {args.version} {args.split} (n={len(df)}) ===")
    print(f"{'signal':<26}{'AUC':>8}{'sd':>8}{'was':>8}{'ties':>8}{'dist':>6}{'vs rnd':>9}")
    for name, r in rows.items():
        print(f"{name:<26}{r['auc']:>8.4f}{r['auc_sd_over_tie_orders']:>8.4f}"
              f"{r['auc_row_order_superseded']:>8.4f}{r['tie_fraction']:>8.3f}"
              f"{r['n_distinct']:>6d}{r['gap_vs_random']:>+9.4f}")
    print(f"{'random (modal outcome)':<26}{random_auc:>8.4f}")
    print()
    for name, c in contrasts.items():
        print(f"  {name:<42} {c['mean_diff']:+.4f} "
              f"[{c['ci_2.5']:+.4f}, {c['ci_97.5']:+.4f}]  "
              f"{'excludes 0' if c['excludes_zero'] else 'includes 0'}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
