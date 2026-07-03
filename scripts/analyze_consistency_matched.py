"""
scripts/analyze_consistency_matched.py

Matched-patch baseline analysis for the consistency experiment.

Problem this fixes: the consistency experiment's metrics.json compares
modal-vote (K=5) accuracy/F1 against single-query numbers drawn from a
DIFFERENT patch set (the 450-sample balanced pilot) and compares routing AUC
against a textual-confidence AUC computed on the FULL 10,004-patch val set.
Both comparisons conflate the effect of multi-query voting with the effect
of evaluating a different set of patches, which is not a valid isolation of
the voting effect and would not survive review.

Fix: extract the single-query baseline directly from data that already
exists, on the IDENTICAL 1,800-patch subset used for modal voting:

  1. Single-query accuracy/F1: filter the full-scale predictions.parquet
     (e.g. V3_full) down to the same arr_idx values used in the consistency
     run. This is the model's one-shot answer on the exact same patches,
     under the exact same prompt (the consistency run's variant 0 is the
     unmodified base prompt -- identical to what produced the full-scale
     predictions).
  2. Single-query routing AUC: same filtered subset, using the raw textual
     confidence from that one-shot answer, vs random routing computed on
     the same 1,800-sample correct/incorrect array (same seed as elsewhere).
  3. Cross-check: also extract raw_labels[0] / raw_confs[0] from the
     consistency run's own predictions (the base-prompt query within the
     K=5 set) and confirm it is consistent with (1). API calls at
     temperature 0.0 are nearly but not perfectly deterministic, so exact
     agreement is not guaranteed; this is reported, not assumed.

No API calls. Pure pandas filtering and the existing router/calibration
modules.

Outputs:
    results/consistency/{model}/{prompt_version}/matched_baseline.json

Usage:
    python scripts/analyze_consistency_matched.py
    python scripts/analyze_consistency_matched.py --base-prompt-version V3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES
from src.triage.router import risk_coverage_curve, random_routing_curve, operating_point

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    Exact McNemar test (binomial on discordant pairs) for two paired binary
    correctness arrays evaluated on the identical set of items.

    Returns counts of discordant pairs and a two-sided exact p-value testing
    whether A's wins and B's wins among discordant pairs are equally likely.
    """
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_value": 1.0}
    p_value = binomtest(b_only, n_discordant, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n_discordant, "p_value": round(float(p_value), 4)}


def bootstrap_auc_diff(
    conf_a: np.ndarray,
    correct_a: np.ndarray,
    conf_b: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap CI for the difference in risk-coverage AUC between two
    (confidence, outcome) pairs measured on the SAME underlying patches.

    Signal A and signal B may route different outcomes (e.g. consistency
    score routes the modal-vote prediction's correctness, while single-query
    confidence routes that single query's own correctness) -- what stays
    paired across the resample is the patch identity via array position, so
    conf_a/correct_a/conf_b/correct_b must already be aligned by patch
    (e.g. produced from the same merged/joined DataFrame) before calling.
    """
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auc_a = risk_coverage_curve(conf_a[idx], correct_a[idx])["auc"]
        auc_b = risk_coverage_curve(conf_b[idx], correct_b[idx])["auc"]
        diffs[i] = auc_a - auc_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "n_boot": n_boot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched-patch single-query baseline for consistency results")
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--base-prompt-version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Which consistency run to analyze. test looks in "
                             "{version}_test and defaults the single-query baseline to "
                             "the {version}_full_test predictions.")
    parser.add_argument("--full-pred-path", type=Path, default=None,
                        help="Full-scale predictions.parquet for the same prompt version. "
                             "Defaults to results/zeroshot/{model}/{version}_full[_test]/predictions.parquet.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(":", "_")
    version_dir = args.base_prompt_version + ("_test" if args.split == "test" else "")
    cons_dir = PROJECT_ROOT / "results" / "consistency" / model_slug / version_dir
    cons_path = cons_dir / "predictions.parquet"

    if not cons_path.exists():
        print(f"Consistency predictions not found: {cons_path}")
        print("Run scripts/run_consistency.py first.")
        return 1

    full_suffix = f"{args.base_prompt_version}_full" + ("_test" if args.split == "test" else "")
    full_pred_path = args.full_pred_path or (
        PROJECT_ROOT / "results" / "zeroshot" / model_slug
        / full_suffix / "predictions.parquet"
    )
    if not full_pred_path.exists():
        print(f"Full-scale predictions not found: {full_pred_path}")
        return 1

    cons_df = pd.read_parquet(cons_path)
    full_df = pd.read_parquet(full_pred_path)

    n = len(cons_df)
    print(f"Consistency subset: {n} patches")

    # ── (1) and (2): filter the full-scale run down to the same arr_idx values ──
    matched = full_df[full_df["arr_idx"].isin(cons_df["arr_idx"])].copy()
    # arr_idx is unique within a balanced/stratified selection but guard against
    # accidental duplicates from the full-scale all-samples run (there should be none)
    matched = matched.drop_duplicates(subset="arr_idx")
    if len(matched) != n:
        print(f"WARNING: matched {len(matched)} of {n} patches by arr_idx. "
              "Some consistency-run patches were not found in the full-scale predictions.")

    sq_true = matched["true_label_name"].tolist()
    sq_pred = matched["pred_label"].tolist()
    sq_conf = matched["pred_confidence"].to_numpy(float)
    sq_correct = (matched["pred_label"] == matched["true_label_name"]).to_numpy(bool)

    sq_accuracy = accuracy_score(sq_true, sq_pred)
    sq_macro_f1 = f1_score(sq_true, sq_pred, average="macro", labels=ALL_LABELS, zero_division=0)

    # Routing: raw confidence ranking (calibrated and raw give identical ranking
    # under monotone temperature scaling, per Methods 3.4)
    sq_curve = risk_coverage_curve(sq_conf, sq_correct)
    sq_random_curve = random_routing_curve(sq_correct, seed=args.seed)
    sq_op_15 = operating_point(sq_conf, sq_correct, 0.15)
    sq_op_rnd_15 = operating_point(np.full(len(sq_correct), 0.5), sq_correct, 0.15)

    print("\n=== (1)+(2) Single-query baseline, matched to the 1,800-patch subset ===")
    print(f"  n matched:        {len(matched)}")
    print(f"  Accuracy:         {sq_accuracy:.4f}")
    print(f"  Macro F1:         {sq_macro_f1:.4f}")
    print(f"  Routing AUC:      {sq_curve['auc']:.4f}")
    print(f"  Random AUC:       {sq_random_curve['auc']:.4f}")
    print(f"  Auto-confirm@15%: {sq_op_15['auto_confirm_acc']:.4f}  (random: {sq_op_rnd_15['auto_confirm_acc']:.4f})")

    # ── (3) cross-check against the consistency run's own variant-0 (base prompt) ──
    # Build a DataFrame with arr_idx alongside variant-0 label/conf so the
    # comparison against `matched` is joined by arr_idx, not by row position.
    # cons_df and the full-scale run are NOT in the same row order (the
    # consistency run iterates a stratified selection, not arr_idx order), so
    # a zip() on row position silently compares unrelated patches.
    v0_labels, v0_confs = [], []
    for _, row in cons_df.iterrows():
        labels = json.loads(row["raw_labels"])
        confs = json.loads(row["raw_confs"])
        v0_labels.append(labels[0])
        v0_confs.append(confs[0])

    v0_df = pd.DataFrame({
        "arr_idx": cons_df["arr_idx"].to_numpy(),
        "true_label_name": cons_df["true_label_name"].to_numpy(),
        "v0_pred": v0_labels,
        "v0_conf": v0_confs,
    })

    v0_accuracy = accuracy_score(v0_df["true_label_name"], v0_df["v0_pred"])
    v0_macro_f1 = f1_score(v0_df["true_label_name"], v0_df["v0_pred"],
                            average="macro", labels=ALL_LABELS, zero_division=0)

    joined = v0_df.merge(matched[["arr_idx", "pred_label"]], on="arr_idx", how="inner")
    agreement_with_full_scale = (
        float((joined["v0_pred"] == joined["pred_label"]).mean()) if len(joined) > 0 else None
    )

    print("\n=== (3) Cross-check: consistency run's own variant-0 (base prompt) query ===")
    print(f"  Accuracy:  {v0_accuracy:.4f}  (vs filtered full-scale: {sq_accuracy:.4f})")
    print(f"  Macro F1:  {v0_macro_f1:.4f}  (vs filtered full-scale: {sq_macro_f1:.4f})")
    if agreement_with_full_scale is not None:
        print(f"  Label agreement rate with filtered full-scale predictions "
              f"(joined on arr_idx, n={len(joined)}): {agreement_with_full_scale:.4f}")
        print("  (Not expected to be exactly 1.0 -- API inference at temperature 0.0 is")
        print("   nearly but not perfectly deterministic across separate calls.)")

    # ── Load the existing modal-vote / consistency-score numbers for the side-by-side ──
    metrics_path = cons_dir / "metrics.json"
    modal_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    # ── Significance tests, all joined on arr_idx to keep pairing correct ──
    # (1) Is the single-query -> modal-vote accuracy delta distinguishable from noise?
    paired = cons_df[["arr_idx", "is_correct", "consistency_score", "mean_textual_conf"]].merge(
        matched[["arr_idx", "pred_label", "true_label_name", "pred_confidence"]], on="arr_idx", how="inner"
    )
    paired["sq_correct"] = paired["pred_label"] == paired["true_label_name"]

    mcnemar_voting = mcnemar_test(
        paired["is_correct"].to_numpy(bool),   # modal-vote correct
        paired["sq_correct"].to_numpy(bool),   # single-query correct
    )
    print("\n=== Significance: does K=5 voting beat single-query on accuracy? ===")
    print(f"  Modal-only wins: {mcnemar_voting['a_only']}  Single-query-only wins: {mcnemar_voting['b_only']}")
    print(f"  McNemar exact p-value: {mcnemar_voting['p_value']}"
          f"  ({'not significant at alpha=0.05' if mcnemar_voting['p_value'] >= 0.05 else 'significant at alpha=0.05'})")

    # (2a) Strategy comparison: consistency routing the K=5 modal-vote outcome vs
    # single-query confidence routing its own single-query outcome. This answers
    # "if you commit to one strategy end-to-end, which one routes better?"
    boot_strategy = bootstrap_auc_diff(
        paired["consistency_score"].to_numpy(float), paired["is_correct"].to_numpy(bool),
        paired["pred_confidence"].to_numpy(float), paired["sq_correct"].to_numpy(bool),
    )
    print("\n=== Bootstrap CI: consistency (routes modal-vote) minus single-query confidence (routes itself) ===")
    print(f"  Mean diff: {boot_strategy['mean_diff']:+.4f}  "
          f"95% CI: [{boot_strategy['ci_2.5']:+.4f}, {boot_strategy['ci_97.5']:+.4f}]")

    # (2b) Same-outcome comparison: consistency vs mean-of-5 textual confidence,
    # BOTH routing the SAME modal-vote outcome. Isolates whether consistency adds
    # value beyond simply averaging confidence across the 5 queries (matches how
    # Table 6 / metrics.json's "textual_conf" AUC was computed).
    modal_correct = paired["is_correct"].to_numpy(bool)
    boot_same_outcome = bootstrap_auc_diff(
        paired["consistency_score"].to_numpy(float), modal_correct,
        paired["mean_textual_conf"].to_numpy(float), modal_correct,
    )
    print("\n=== Bootstrap CI: consistency-score AUC minus mean-of-5-textual-confidence AUC (same outcome) ===")
    print(f"  Mean diff: {boot_same_outcome['mean_diff']:+.4f}  "
          f"95% CI: [{boot_same_outcome['ci_2.5']:+.4f}, {boot_same_outcome['ci_97.5']:+.4f}]")

    result = {
        "model": args.model,
        "prompt_version": args.base_prompt_version,
        "n_patches": n,
        "n_matched": len(matched),
        "single_query_matched": {
            "accuracy": round(float(sq_accuracy), 4),
            "macro_f1": round(float(sq_macro_f1), 4),
            "routing_auc": round(float(sq_curve["auc"]), 4),
            "random_auc": round(float(sq_random_curve["auc"]), 4),
            "auto_confirm_acc_15pct": round(float(sq_op_15["auto_confirm_acc"]), 4),
            "random_auto_confirm_acc_15pct": round(float(sq_op_rnd_15["auto_confirm_acc"]), 4),
        },
        "single_query_variant0_crosscheck": {
            "accuracy": round(float(v0_accuracy), 4),
            "macro_f1": round(float(v0_macro_f1), 4),
            "agreement_with_filtered_full_scale": (
                round(agreement_with_full_scale, 4) if agreement_with_full_scale is not None else None
            ),
        },
        "modal_vote_k5": {
            "accuracy": modal_metrics.get("modal_accuracy"),
            "macro_f1": modal_metrics.get("modal_macro_f1"),
        },
        "routing_auc_k5": modal_metrics.get("routing_auc", {}),
        "significance": {
            "voting_lift_mcnemar": mcnemar_voting,
            "auc_diff_bootstrap_strategy_comparison": boot_strategy,
            "auc_diff_bootstrap_same_outcome": boot_same_outcome,
        },
    }

    out_path = cons_dir / "matched_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 65}")
    print("Matched comparison (all numbers on the IDENTICAL 1,800-patch subset)")
    print(f"{'=' * 65}")
    print(f"{'Metric':<28} {'Single-query':>14} {'Modal K=5':>12} {'Delta':>10}")
    macro_f1_delta = (modal_metrics.get("modal_macro_f1", float('nan')) - sq_macro_f1)
    acc_delta = (modal_metrics.get("modal_accuracy", float('nan')) - sq_accuracy)
    print(f"{'Accuracy':<28} {sq_accuracy:>14.4f} {modal_metrics.get('modal_accuracy', float('nan')):>12.4f} {acc_delta:>+10.4f}")
    print(f"{'Macro F1':<28} {sq_macro_f1:>14.4f} {modal_metrics.get('modal_macro_f1', float('nan')):>12.4f} {macro_f1_delta:>+10.4f}")
    routing_auc_k5 = modal_metrics.get("routing_auc", {})
    print(f"\n{'Routing AUC':<28} {'Single-query':>14} {'Consistency':>12} {'Random':>10}")
    print(f"{'(matched 1,800 patches)':<28} {sq_curve['auc']:>14.4f} "
          f"{routing_auc_k5.get('consistency', float('nan')):>12.4f} {sq_random_curve['auc']:>10.4f}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
