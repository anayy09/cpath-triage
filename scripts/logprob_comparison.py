"""
scripts/logprob_comparison.py

Collect the four log-probability runs into the single comparison table the
manuscript quotes (revision item C-11, Decision 1). No API calls: reads the
metrics written by scripts/run_logprob_confidence.py.

Why this exists rather than quoting four files. Decision 1 asks for the
label-token confidence to get "the same treatment the verbalized confidence
gets", across two models and two splits. That is sixteen numbers whose whole
point is the comparison between them, so they should come from one artifact
with one provenance rather than be transcribed from four.

One reporting hazard this script exists to surface. Mutual information here is
the plug-in estimator over 15 equal-width bins, the same scheme used everywhere
else in the paper, and under that scheme MI is *not* comparable between the two
confidence sources. The label-token signal is near-continuous and heavily peaked
near 1.0, so most of its mass falls in the top bin and the estimator sees little
variation, while the verbalized signal takes a handful of values spread across
the range. On Gemma-3 this inverts the ordering: MI ranks the verbalized signal
far above the label-token signal while AUROC ranks them the other way round.

The rank-based statistics do not have this problem, because they depend only on
ordering. So the manuscript leads with AUROC and selective-accuracy AUC, reports
MI with the artifact named, and this script computes the diagnostic that makes
the artifact checkable: the fraction of each signal's mass in its single most
occupied bin, and the number of the 15 bins it occupies at all.

Outputs:
    results/logprob_confidence/comparison.json

Usage:
    python scripts/logprob_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import random_routing_curve, risk_coverage_curve

LP_ROOT = PROJECT_ROOT / "results" / "logprob_confidence"
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")
SPLITS = ("val", "test")
SOURCES = ("label_token_confidence", "verbalized_confidence_same_patches")
N_BINS = 15
N_BOOT = 1000
SEED = 42


def bin_occupancy(x: np.ndarray, n_bins: int = N_BINS) -> dict:
    """How concentrated a signal is under the paper's equal-width binning."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    return {
        "n_bins": n_bins,
        "n_occupied": int((counts > 0).sum()),
        "largest_bin_fraction": round(float(counts.max() / len(x)), 4),
    }



def paired_bootstrap(
    sig_a: np.ndarray,
    sig_b: np.ndarray | None,
    correct: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """
    Interval for a selective-accuracy AUC difference, patches held paired.

    With sig_b given, the estimand is AUC(A) - AUC(B) on identical patches, which
    is what the label-token against verbalized comparison needs. With sig_b None
    the estimand is AUC(A) minus the random-routing reference on the same
    resample, which is the gap-against-random each signal is quoted with.

    Ties are resolved by the closed-form expectation inside every resample, so the
    interval and the point estimate are computing the same quantity. That
    correspondence is the defect revision item C-01 was about.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = correct[idx]
        a = risk_coverage_curve(sig_a[idx], c, tie_break="expected")["auc"]
        b = (risk_coverage_curve(sig_b[idx], c, tie_break="expected")["auc"]
             if sig_b is not None else random_routing_curve(c, seed=seed)["auc"])
        diffs[i] = a - b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_2.5": round(float(lo), 4),
        "ci_97.5": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }


def auroc_midrank(conf: np.ndarray, correct: np.ndarray) -> float:
    """Mann-Whitney AUROC with mid-ranks, so ties are handled explicitly."""
    pos, neg = correct.astype(bool), ~correct.astype(bool)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = rankdata(conf)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def by_predicted_class(df: pd.DataFrame, min_n: int = 50) -> dict:
    """
    Confidence and accuracy per predicted class, plus the within-class AUROC.

    This exists because the pooled AUROC of the label-token signal reverses
    between splits while the within-class rankings do not. MedGemma is most
    confident at the token level on the class it is worst at, and that class
    holds most of its predictions, so pooling inverts a signal that is
    informative inside every class. Reporting only the pooled number would hide
    the mechanism, which is the same mistake the submitted manuscript made about
    the verbalized signal.
    """
    out: dict = {}
    for cls, sub in df.groupby("pred_label"):
        rec = {
            "n": len(sub),
            "share_of_predictions": round(float(len(sub) / len(df)), 4),
            "accuracy": round(float(sub["correct"].mean()), 4),
            "mean_label_token_conf": round(float(sub["logprob_conf"].mean()), 4),
            "mean_verbalized_conf": round(float(sub["verbalized_conf"].mean()), 4),
        }
        if len(sub) >= min_n and sub["correct"].nunique() == 2:
            rec["within_class_auroc_label_token"] = round(
                auroc_midrank(sub["logprob_conf"].to_numpy(float),
                              sub["correct"].to_numpy(bool)), 4)
            rec["within_class_auroc_verbalized"] = round(
                auroc_midrank(sub["verbalized_conf"].to_numpy(float),
                              sub["correct"].to_numpy(bool)), 4)
        out[cls] = rec
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["n"])
    return {
        "min_n_for_within_class_auroc": min_n,
        "classes": dict(ranked),
    }


def main() -> int:
    rows: dict[str, dict] = {}
    missing: list[str] = []

    for model in MODELS:
        for split in SPLITS:
            mpath = LP_ROOT / model / split / "metrics.json"
            ppath = LP_ROOT / model / split / "predictions.parquet"
            key = f"{model}|{split}"
            if not mpath.exists():
                missing.append(str(mpath.relative_to(PROJECT_ROOT)))
                continue
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if SOURCES[0] not in m:
                missing.append(f"{key} (no usable responses)")
                continue

            entry: dict = {
                "model": model,
                "split": split,
                "n": m[SOURCES[0]]["n"],
                "accuracy": m[SOURCES[0]]["accuracy"],
                "sampling": m["sampling"],
                "design": m["design"],
                "n_label_span_missing": m["n_label_span_missing"],
                "n_call_failed": m["n_call_failed"],
                "request_window_utc": m["endpoint"]["request_window_utc"],
            }

            occ = {}
            if ppath.exists():
                df = pd.read_parquet(ppath)
                usable = df[df["logprob_conf"].notna() & ~df["call_failed"]]
                occ = {
                    "label_token_confidence": bin_occupancy(
                        usable["logprob_conf"].to_numpy(float)),
                    "verbalized_confidence_same_patches": bin_occupancy(
                        usable["verbalized_conf"].to_numpy(float)),
                }

            for src in SOURCES:
                s = dict(m[src])
                if src in occ:
                    s["bin_occupancy"] = occ[src]
                entry[src] = s

            lt, vb = m[SOURCES[0]], m[SOURCES[1]]
            entry["label_token_minus_verbalized"] = {
                "auroc": round(lt["auroc_midrank"] - vb["auroc_midrank"], 4),
                "selective_accuracy_auc": round(
                    lt["selective_accuracy_auc"] - vb["selective_accuracy_auc"], 4),
                "ece": round(lt["ece"] - vb["ece"], 4),
                "note": "Positive AUROC and AAUC favour the label-token signal.",
            }

            # Every gap quoted from this table needs an interval, and the two
            # signals live on the same patches, so the comparison is paired.
            if ppath.exists():
                correct = usable["correct"].to_numpy(bool)
                lt_sig = usable["logprob_conf"].to_numpy(float)
                vb_sig = usable["verbalized_conf"].to_numpy(float)
                entry["by_predicted_class"] = by_predicted_class(usable)
                entry["bootstrap"] = {
                    "label_token_minus_random": paired_bootstrap(lt_sig, None, correct),
                    "verbalized_minus_random": paired_bootstrap(vb_sig, None, correct),
                    "label_token_minus_verbalized": paired_bootstrap(lt_sig, vb_sig, correct),
                }
            rows[key] = entry

    if not rows:
        print("No log-probability metrics found. Run scripts/run_logprob_confidence.py first.",
              file=sys.stderr)
        return 1

    out = {
        "n_bins_for_mi_and_ece": N_BINS,
        "mi_comparability_warning": (
            "Mutual information uses the plug-in estimator over 15 equal-width bins, as "
            "elsewhere in the paper. It is not comparable between these two confidence "
            "sources: the label-token signal is near-continuous and peaked near 1.0, so its "
            "mass concentrates in the top bin, while the verbalized signal takes a few values "
            "spread across the range. Compare the rank-based statistics, AUROC and "
            "selective-accuracy AUC, and read the bin_occupancy fields before quoting MI."
        ),
        "runs": rows,
        "missing": missing,
    }
    out_path = LP_ROOT / "comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"=== Label-token vs verbalized confidence ({len(rows)}/4 runs present) ===")
    hdr = f"{'run':28}{'source':12}{'AUROC':>8}{'AAUC':>8}{'rnd':>8}{'gap':>9}{'ECE':>8}{'MI':>9}{'occ':>5}{'top':>7}"
    print(hdr)
    for key, e in rows.items():
        for src, tag in zip(SOURCES, ("label-tok", "verbal")):
            s = e[src]
            occ = s.get("bin_occupancy", {})
            print(f"{key:28}{tag:12}{s['auroc_midrank']:>8.4f}{s['selective_accuracy_auc']:>8.4f}"
                  f"{s['random_routing_auc']:>8.4f}{s['gap_vs_random']:>+9.4f}{s['ece']:>8.4f}"
                  f"{s['mi_bits']:>9.5f}{occ.get('n_occupied', 0):>5}"
                  f"{occ.get('largest_bin_fraction', float('nan')):>7.2f}")
    print()
    for key, e in rows.items():
        b = e.get("bootstrap")
        if not b:
            continue
        print(key)
        for name, c in b.items():
            mark = "excludes 0" if c["excludes_zero"] else "includes 0"
            print(f"   {name:34} {c['mean_diff']:+.4f} "
                  f"[{c['ci_2.5']:+.4f}, {c['ci_97.5']:+.4f}]  {mark}")

    if missing:
        print("\nMissing:")
        for m in missing:
            print(f"  {m}")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
