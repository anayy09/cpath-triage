"""
scripts/consistency_population.py

Cohort-level re-analysis of the K=5 consistency experiment. No API calls, no
images: everything is computed from the saved K=5 responses, the full-scale V3
predictions and the inclusion weights written by consistency_selection_weights.py.

Five analyses, each answering a design objection to the subset-level numbers in
Tables 10 and 11:

1. weighted_routing.json. The subsets were stratified on full-scale V3
   correctness and capped at 200 patches per class, so their outcome prevalence
   and class mix differ from the cohort's. Every routing AUC, accuracy and
   contrast is recomputed with Horvitz-Thompson weights (src/triage/weighted.py),
   which estimates what the same signals would score on the whole split, next to
   the unweighted subset value. The voting gain becomes a cohort-level accuracy
   difference with an interval rather than a McNemar test on the subset.

2. fixed_outcome.json. The published consistency minus single-query contrast
   routes the modal-vote outcome with one signal and the variant-0 outcome with
   the other, so it changes signal and outcome at once. Here every signal routes
   the same outcome, once with modal-vote correctness and once with variant-0
   correctness.

3. leave_variant_out.json. The five variants are not wording-only paraphrases:
   variant 2 is a restructured V4-style prompt and variant 4 instructs the model
   to lower its confidence. Agreement is recomputed with each variant dropped,
   and with only the three V3 edits (variants 0, 1 and 3).

4. confusability_valweights.json. The published confusability weights come from
   a CNN confusion matrix on the external test split, which uses test labels to
   build the score evaluated on that split. Here the weights come from MedGemma's
   own full-scale validation confusion matrix, cross-fitted on validation (the
   1,800 evaluated patches are left out of the matrix) and taken whole for the
   external split, so no test label enters the score.

5. pilot_excluded.json (validation only). The 450-patch pilot that selected V3
   was drawn from validation and 78 of its patches are in this subset. Analyses
   1 and 2 are repeated without them, reweighted to the cohort minus the pilot.

Conventions follow scripts/consistency_routing_table.py: the point estimate is
the closed-form tie expectation, a gap is the plug-in difference rounded once,
and intervals are 1,000-resample paired bootstraps. Unweighted intervals use the
same simple resampling and seed as the published tables, so the published
unweighted intervals reproduce; weighted intervals resample inside the selection
cells, which keeps every weight fixed. The random reference is the exact
expectation of random routing, the accuracy of the routed outcome.

Usage:
    python scripts/consistency_population.py --split val
    python scripts/consistency_population.py --split test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES
from src.triage.router import random_routing_curve, tie_statistics
from src.triage.weighted import (
    simple_bootstrap_indices,
    stratified_bootstrap_indices,
    summarize_diffs,
    weighted_accuracy,
    weighted_auc,
    weighted_auroc,
)

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
K = 5
N_BOOT = 1000
# Variant order as built by run_consistency._build_variants.
VARIANT_NAMES = [
    "0: V3 unmodified",
    "1: architectural-focus instruction prepended",
    "2: V4-style restructured prompt, alphabetical classes, disambiguation rules",
    "3: clinical context added",
    "4: instruction to lower confidence under ambiguity",
]
SUBSETS: dict[str, tuple[int, ...]] = {
    "all_5": (0, 1, 2, 3, 4),
    "drop_0": (1, 2, 3, 4),
    "drop_1": (0, 2, 3, 4),
    "drop_2": (0, 1, 3, 4),
    "drop_3": (0, 1, 2, 4),
    "drop_4": (0, 1, 2, 3),
    "v3_edits_only_0_1_3": (0, 1, 3),
}


def _load_confusability_module():
    # The published construction is reused rather than copied, so the
    # validation-derived matrix differs from the CNN one only in its source.
    path = PROJECT_ROOT / "scripts" / "confusability_weighted.py"
    spec = importlib.util.spec_from_file_location("confusability_weighted", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def confusability_from_counts(counts: np.ndarray) -> np.ndarray:
    """Symmetrise, row-normalise and max-scale, exactly as build_confusability does."""
    sym = (counts + counts.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    row_sums = sym.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(row_sums > 0, sym / row_sums, 0.0)
    peak = float(normed.max())
    scaled = normed / peak if peak > 0 else normed
    np.fill_diagonal(scaled, 1.0)
    return scaled


def confusion_counts(true_names: pd.Series, pred_names: pd.Series) -> np.ndarray:
    idx = {name: i for i, name in enumerate(ALL_LABELS)}
    counts = np.zeros((len(ALL_LABELS), len(ALL_LABELS)), dtype=float)
    for t, p in zip(true_names, pred_names):
        if str(p) in idx:
            counts[idx[str(t)], idx[str(p)]] += 1.0
    return counts


def modal_and_agreement(labels: list[list[str]], keep: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """
    Modal label and agreement fraction over a subset of variants.

    Same rule as run_consistency: parse failures are dropped before counting,
    ties go to the label seen first in variant order (Counter.most_common), and
    the denominator is the number of variants queried.
    """
    modal, agree = [], []
    for row in labels:
        sub = [row[i] for i in keep]
        valid = [lab for lab in sub if lab != "unknown"]
        if valid:
            lab, cnt = Counter(valid).most_common(1)[0]
        else:
            lab, cnt = "unknown", 0
        modal.append(lab)
        agree.append(cnt / len(keep))
    return np.array(modal), np.array(agree, dtype=float)


def entropy_bits(labels: list[str]) -> float:
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate(specs: dict[str, tuple[np.ndarray | None, np.ndarray]], idx: np.ndarray,
             w: np.ndarray) -> dict[str, float]:
    """AUC for every (signal, outcome) spec on one resample; signal None means random."""
    out = {}
    wi = w[idx]
    for name, (sig, corr) in specs.items():
        c = corr[idx]
        out[name] = weighted_accuracy(c, wi) if sig is None else weighted_auc(sig[idx], c, wi)
    return out


def run_bootstrap(specs: dict, w: np.ndarray, idx_matrix: np.ndarray) -> dict[str, np.ndarray]:
    rows = [evaluate(specs, idx, w) for idx in idx_matrix]
    return {k: np.array([r[k] for r in rows]) for k in specs}


def analyse(specs: dict, contrasts: dict[str, tuple[str, str]], w: np.ndarray,
            strata: np.ndarray, n_boot: int, seed: int) -> dict:
    """Unweighted and weighted point estimates, intervals and contrasts for a spec set."""
    n = len(w)
    ones = np.ones(n)
    result: dict = {}
    for label, weights, idx_matrix in (
        ("unweighted_subset", ones, simple_bootstrap_indices(n, n_boot, seed)),
        ("weighted_cohort", w, stratified_bootstrap_indices(strata, n_boot, seed)),
    ):
        full = evaluate(specs, np.arange(n), weights)
        boot = run_bootstrap(specs, weights, idx_matrix)
        block = {"auc": {k: round(v, 4) for k, v in full.items()},
                 "auc_exact": dict(full), "contrasts": {}}
        for cname, (a, b) in contrasts.items():
            # Full precision difference, rounded once inside summarize_diffs.
            block["contrasts"][cname] = summarize_diffs(boot[a] - boot[b], full[a] - full[b])
        result[label] = block
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-from", default=None,
        help="Family whose patch selection this run reused, e.g. V3 for V3W. Its weights, "
             "selection cells and full-scale predictions are used; the alignment check below "
             "refuses to proceed unless the patches are identical and in the same order.",
    )
    args = parser.parse_args()
    print(f"seed={args.seed}  n_boot={args.n_boot}  split={args.split}")

    suffix = "_test" if args.split == "test" else ""
    root = PROJECT_ROOT / "results" / "consistency" / args.model
    cons_dir = root / f"{args.version}{suffix}"
    sel = args.selection_from or args.version
    sel_dir = root / f"{sel}{suffix}"
    paths = {
        "cons": cons_dir / "predictions.parquet",
        "weights": sel_dir / "weights.parquet",
        "cells": sel_dir / "selection_cells.json",
        "val_full": PROJECT_ROOT / "results" / "zeroshot" / args.model / f"{sel}_full" / "predictions.parquet",
        "cnn_test": PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "test_predictions.parquet",
    }
    for p in paths.values():
        if not p.exists():
            print(f"Missing required file: {p} (run consistency_selection_weights.py first)",
                  file=sys.stderr)
            return 1

    df = pd.read_parquet(paths["cons"])
    wdf = pd.read_parquet(paths["weights"])
    if df["arr_idx"].tolist() != wdf["arr_idx"].tolist():
        print("weights.parquet is not aligned with predictions.parquet", file=sys.stderr)
        return 1
    w = wdf["weight"].to_numpy(float)
    strata = wdf["cell"].to_numpy()
    true = df["true_label_name"].to_numpy()
    labels = [json.loads(s) for s in df["raw_labels"]]
    confs = np.array([json.loads(s) for s in df["raw_confs"]], dtype=float)
    if labels and len(labels[0]) != K:
        print("unexpected number of variants", file=sys.stderr)
        return 1

    modal5, agree5 = modal_and_agreement(labels, SUBSETS["all_5"])
    # The recomputed modal label and agreement must match what inference stored.
    if not (np.array_equal(modal5, df["modal_label"].to_numpy())
            and np.allclose(agree5, df["consistency_score"].to_numpy(float), atol=1e-9)):
        print("recomputed modal label or agreement differs from the stored values", file=sys.stderr)
        return 1

    modal_correct = modal5 == true
    v0_correct = np.array([row[0] for row in labels]) == true
    v0_conf = confs[:, 0]
    mean5 = df["mean_textual_conf"].to_numpy(float)
    entropy = -np.array([entropy_bits(row) for row in labels])

    cmod = _load_confusability_module()
    cnn_matrix, cnn_prov = cmod.build_confusability(paths["cnn_test"])
    conf_cnn = cmod.weighted_agreement(labels, modal5, cnn_matrix)

    # Validation-derived matrix: MedGemma's own full-scale confusions. On the
    # validation split the evaluated patches are left out of the matrix, so the
    # score for each patch never uses that patch's label.
    val_full = pd.read_parquet(paths["val_full"])
    if args.split == "val":
        matrix_rows = val_full[~val_full["arr_idx"].isin(set(df["arr_idx"]))]
        matrix_note = "MedGemma V3 full-scale validation, excluding the 1,800 evaluated patches"
    else:
        matrix_rows = val_full
        matrix_note = "MedGemma V3 full-scale validation, all 10,004 patches; no test labels used"
    val_matrix = confusability_from_counts(
        confusion_counts(matrix_rows["true_label_name"], matrix_rows["pred_label"])
    )
    conf_val = cmod.weighted_agreement(labels, modal5, val_matrix)
    conf_uniform = cmod.weighted_agreement(labels, modal5, np.eye(len(ALL_LABELS)))
    uniform_dev = float(np.abs(conf_uniform - agree5).max())

    # ---- 1. weighted routing ------------------------------------------------
    specs1 = {
        "single_query_conf|single_query": (v0_conf, v0_correct),
        "random|single_query": (None, v0_correct),
        "mean5_conf|modal": (mean5, modal_correct),
        "consistency|modal": (agree5, modal_correct),
        "entropy|modal": (entropy, modal_correct),
        "confusability_cnn_test|modal": (conf_cnn, modal_correct),
        "random|modal": (None, modal_correct),
    }
    contrasts1 = {
        "consistency_minus_mean5_conf": ("consistency|modal", "mean5_conf|modal"),
        "consistency_minus_single_query_conf_mixed_outcome": (
            "consistency|modal", "single_query_conf|single_query"),
        "consistency_minus_entropy": ("consistency|modal", "entropy|modal"),
        "confusability_cnn_test_minus_consistency": (
            "confusability_cnn_test|modal", "consistency|modal"),
        "consistency_minus_random": ("consistency|modal", "random|modal"),
        "mean5_conf_minus_random": ("mean5_conf|modal", "random|modal"),
        "single_query_conf_minus_random": ("single_query_conf|single_query", "random|single_query"),
        "confusability_cnn_test_minus_random": ("confusability_cnn_test|modal", "random|modal"),
        "voting_gain_accuracy": ("random|modal", "random|single_query"),
    }
    res1 = analyse(specs1, contrasts1, w, strata, args.n_boot, args.seed)
    ties = {k: round(float(tie_statistics(s)["tie_fraction"]), 4)
            for k, (s, _) in specs1.items() if s is not None}
    distinct = {k: int(tie_statistics(s)["n_distinct"])
                for k, (s, _) in specs1.items() if s is not None}
    auroc = {
        "consistency_vs_modal_correct": {
            "unweighted_subset": round(weighted_auroc(agree5, modal_correct, np.ones(len(w))), 4),
            "weighted_cohort": round(weighted_auroc(agree5, modal_correct, w), 4),
        },
        "single_query_conf_vs_single_query_correct": {
            "unweighted_subset": round(weighted_auroc(v0_conf, v0_correct, np.ones(len(w))), 4),
            "weighted_cohort": round(weighted_auroc(v0_conf, v0_correct, w), 4),
        },
    }
    published_random = {
        "modal_vote_outcome": round(random_routing_curve(modal_correct, seed=args.seed)["auc"], 4),
        "single_query_outcome": round(random_routing_curve(v0_correct, seed=args.seed)["auc"], 4),
    }
    common = {
        "model": args.model, "version": args.version, "split": args.split,
        "seed": args.seed, "n_boot": args.n_boot, "n_patches": len(df),
        "cohort_size_weighted": round(float(w.sum()), 3),
    }
    out1 = {
        **common,
        "estimand": (
            "weighted_cohort estimates each quantity on the whole split using "
            "inverse inclusion-probability weights from the stratified selection; "
            "unweighted_subset is the 1,800-patch subset value as in Tables 10 and 11"
        ),
        "random_reference_note": (
            "random rows are the exact expectation of random routing, i.e. the "
            "accuracy of the routed outcome; the published tables used a 30-draw "
            "Monte Carlo of the same quantity, given in published_random_mc"
        ),
        "published_random_mc": published_random,
        **res1,
        "tie_fraction_subset": ties,
        "n_distinct_subset": distinct,
        "auroc": auroc,
        "accuracy": {
            "single_query": {
                "unweighted_subset": round(float(v0_correct.mean()), 4),
                "weighted_cohort": round(weighted_accuracy(v0_correct, w), 4),
            },
            "modal_vote": {
                "unweighted_subset": round(float(modal_correct.mean()), 4),
                "weighted_cohort": round(weighted_accuracy(modal_correct, w), 4),
            },
        },
    }
    write(cons_dir / "weighted_routing.json", out1)

    # ---- 2. fixed outcome -----------------------------------------------------
    specs2 = {}
    contrasts2 = {}
    for oname, outcome in (("modal", modal_correct), ("single_query", v0_correct)):
        specs2[f"consistency|{oname}"] = (agree5, outcome)
        specs2[f"single_query_conf|{oname}"] = (v0_conf, outcome)
        specs2[f"mean5_conf|{oname}"] = (mean5, outcome)
        specs2[f"random|{oname}"] = (None, outcome)
        contrasts2[f"{oname}_outcome:consistency_minus_single_query_conf"] = (
            f"consistency|{oname}", f"single_query_conf|{oname}")
        contrasts2[f"{oname}_outcome:consistency_minus_mean5_conf"] = (
            f"consistency|{oname}", f"mean5_conf|{oname}")
        contrasts2[f"{oname}_outcome:mean5_conf_minus_single_query_conf"] = (
            f"mean5_conf|{oname}", f"single_query_conf|{oname}")
        for s in ("consistency", "single_query_conf", "mean5_conf"):
            contrasts2[f"{oname}_outcome:{s}_minus_random"] = (f"{s}|{oname}", f"random|{oname}")
    res2 = analyse(specs2, contrasts2, w, strata, args.n_boot, args.seed)
    write(cons_dir / "fixed_outcome.json", {
        **common,
        "note": (
            "every signal routes the same outcome: modal-vote correctness, then "
            "variant-0 (single-query) correctness. The single-query prediction is "
            "variant 0 of the same session."
        ),
        **res2,
    })

    # ---- 3. leave variant out ------------------------------------------------
    specs3, contrasts3, lvo_meta = {}, {}, {}
    for sname, keep in SUBSETS.items():
        modal_s, agree_s = modal_and_agreement(labels, keep)
        corr_s = modal_s == true
        specs3[f"{sname}|consistency"] = (agree_s, corr_s)
        specs3[f"{sname}|random"] = (None, corr_s)
        contrasts3[f"{sname}:consistency_minus_random"] = (
            f"{sname}|consistency", f"{sname}|random")
        lvo_meta[sname] = {
            "variants": [VARIANT_NAMES[i] for i in keep],
            "k": len(keep),
            "n_distinct": int(tie_statistics(agree_s)["n_distinct"]),
            "tie_fraction_subset": round(float(tie_statistics(agree_s)["tie_fraction"]), 4),
            "modal_accuracy_unweighted": round(float(corr_s.mean()), 4),
            "modal_accuracy_weighted": round(weighted_accuracy(corr_s, w), 4),
        }
    res3 = analyse(specs3, contrasts3, w, strata, args.n_boot, args.seed)
    write(cons_dir / "leave_variant_out.json", {
        **common,
        "variant_order": VARIANT_NAMES,
        "note": (
            "agreement is recomputed over each variant subset, with the modal "
            "label and the outcome it routes recomputed from the same subset"
        ),
        "subsets": lvo_meta,
        **res3,
    })

    # ---- 4. test-free confusability weights ---------------------------------
    specs4 = {
        "confusability_valweights|modal": (conf_val, modal_correct),
        "confusability_cnn_test|modal": (conf_cnn, modal_correct),
        "uniform_control|modal": (conf_uniform, modal_correct),
        "consistency|modal": (agree5, modal_correct),
        "random|modal": (None, modal_correct),
    }
    contrasts4 = {
        "valweights_minus_consistency": ("confusability_valweights|modal", "consistency|modal"),
        "valweights_minus_random": ("confusability_valweights|modal", "random|modal"),
        "cnn_test_minus_consistency": ("confusability_cnn_test|modal", "consistency|modal"),
        "valweights_minus_cnn_test": ("confusability_valweights|modal", "confusability_cnn_test|modal"),
    }
    res4 = analyse(specs4, contrasts4, w, strata, args.n_boot, args.seed)
    top = cmod._top_pairs(val_matrix, k=5)
    write(cons_dir / "confusability_valweights.json", {
        **common,
        "matrix_source": matrix_note,
        "matrix_n_patches": len(matrix_rows),
        "matrix_source_accuracy": round(
            float((matrix_rows["pred_label"] == matrix_rows["true_label_name"]).mean()), 4),
        "confusability_matrix": {
            "labels": ALL_LABELS,
            "matrix": [[round(float(v), 6) for v in row] for row in val_matrix],
            "most_confusable_pairs": top,
        },
        "cnn_test_matrix_most_confusable_pairs": cnn_prov["most_confusable_pairs"],
        "uniform_control_reproduces_consistency": {
            "max_abs_deviation": round(uniform_dev, 12), "exact": bool(uniform_dev < 1e-12),
        },
        "n_distinct_subset": {
            "confusability_valweights": int(tie_statistics(conf_val)["n_distinct"]),
            "confusability_cnn_test": int(tie_statistics(conf_cnn)["n_distinct"]),
        },
        "tie_fraction_subset": {
            "confusability_valweights": round(float(tie_statistics(conf_val)["tie_fraction"]), 4),
            "confusability_cnn_test": round(float(tie_statistics(conf_cnn)["tie_fraction"]), 4),
        },
        **res4,
    })

    # ---- 5. pilot excluded ---------------------------------------------------
    if args.split == "val":
        keep_mask = ~wdf["in_pilot"].to_numpy(bool)
        cells = json.loads(paths["cells"].read_text())["cells"]
        w_np = np.array([
            cells[c]["pool_size_excl_pilot"] / cells[c]["n_drawn_excl_pilot"]
            for c in strata[keep_mask]
        ])

        def sub(spec: dict) -> dict:
            return {k: (None if s is None else s[keep_mask], c[keep_mask]) for k, (s, c) in spec.items()}

        res5a = analyse(sub(specs1), contrasts1, w_np, strata[keep_mask], args.n_boot, args.seed)
        res5b = analyse(sub(specs2), contrasts2, w_np, strata[keep_mask], args.n_boot, args.seed)
        write(cons_dir / "pilot_excluded.json", {
            **common,
            "n_patches": int(keep_mask.sum()),
            "n_pilot_removed": int((~keep_mask).sum()),
            "cohort_size_weighted": round(float(w_np.sum()), 3),
            "note": (
                "validation subset without the patches of the 450-patch prompt-"
                "selection pilot; weights re-derived for the cohort minus the pilot"
            ),
            "weighted_routing": res5a,
            "fixed_outcome": res5b,
        })

    print_summary(out1, res2, lvo_meta, res3, res4)
    return 0


def write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def _fmt(c: dict) -> str:
    return (f"{c['gap_point']:+.4f} [{c['ci_2.5']:+.4f}, {c['ci_97.5']:+.4f}]"
            f"{'*' if c['excludes_zero'] else ' '} p={c['p_normal']}")


def print_summary(out1: dict, res2: dict, lvo_meta: dict, res3: dict, res4: dict) -> None:
    for blk in ("unweighted_subset", "weighted_cohort"):
        print(f"\n== {blk}")
        for k, v in out1[blk]["auc"].items():
            print(f"  {k:<40}{v:.4f}")
        for k, v in out1[blk]["contrasts"].items():
            print(f"  {k:<52}{_fmt(v)}")
        for k, v in res2[blk]["contrasts"].items():
            print(f"  {k:<52}{_fmt(v)}")
        for k, v in res3[blk]["contrasts"].items():
            print(f"  {k:<52}{_fmt(v)}")
        for k, v in res4[blk]["contrasts"].items():
            print(f"  {k:<52}{_fmt(v)}")


if __name__ == "__main__":
    raise SystemExit(main())
