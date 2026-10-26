"""
scripts/analyze_image_controls.py

Summarise the round-3 image controls (scripts/run_image_controls.py) and the V3W
drift probe (scripts/run_consistency_v3w.py) against the stored real-image V3
K=5 run on the same validation patches. No API calls.

What each block answers:
  no_image  Are outputs identical across repeats with the image withheld (text
            only) or blanked (uniform grey)? If they are, every patch would get the
            same five answers, agreement would be one constant, and it could not
            rank patches at all. Also reports which labels the prompts alone
            produce.
  swap      Do the five answers follow the image or the patch identity? Per-call
            label agreement with the donor's stored labels versus the recipient's,
            modal-label agreement, and the correlation of consistency scores.
  shuffle   With morphology destroyed and colour kept: modal accuracy, accuracy at
            5/5 versus at most 2/5 agreement, and consistency A_AUC against random
            (ties resolved by expectation, the manuscript's convention), each next
            to the real-image values on the same patches, with paired bootstrap
            intervals for the gap over random and for the real-minus-shuffled
            difference of those gaps.
  v3w       If the V3W run exists: variant-0 drift against the stored V3 variant 0
            (same prompt, different session), and V3W consistency A_AUC versus
            random. Full V3W routing tables come from
            consistency_routing_table.py --version V3W.

Random routing is scored at its exact expectation, the base accuracy of the
outcome routed, rather than by sampling orders.

Two scoring rules. The control and V3W runs kept their response text and are
re-scored from the LABEL line (scripts/rescore_labels.py); the stored V3 run did
not and keeps its whole-text labels. A quantity inside one control run uses the
re-scored labels. A comparison with the stored V3 run (swap matches, V3W drift,
real minus shuffled) uses the stored labels on both sides, kept as *_regex
columns, so that re-scoring is not counted as drift or as an image effect.

Output: results/consistency/{model}/controls/summary.json

Usage:
    python scripts/analyze_image_controls.py
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

from src.triage.router import risk_coverage_curve
from src.triage.weighted import (
    stratified_bootstrap_indices,
    summarize_diffs,
    weighted_accuracy,
    weighted_auc,
)

N_BOOT = 1000


def aauc(sig: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage_curve(sig, correct, tie_break="expected")["auc"])


def gap_vs_random(sig: np.ndarray, correct: np.ndarray) -> float:
    return aauc(sig, correct) - float(correct.mean())


def boot(fn, n: int, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        out.append(fn(idx))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def strata_accuracy(cons: np.ndarray, correct: np.ndarray) -> dict:
    hi = cons >= 1.0 - 1e-9
    lo = cons <= 0.4 + 1e-9
    return {
        "acc_5of5": float(correct[hi].mean()) if hi.any() else None, "n_5of5": int(hi.sum()),
        "acc_le2of5": float(correct[lo].mean()) if lo.any() else None, "n_le2of5": int(lo.sum()),
    }


def stored(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as the whole-text parser stored it, for comparisons with the stored V3 run."""
    return df[f"{col}_regex"] if f"{col}_regex" in df.columns else df[col]


def labels_of(df: pd.DataFrame, as_stored: bool = False) -> list[list[str]]:
    col = stored(df, "raw_labels") if as_stored else df["raw_labels"]
    return [json.loads(s) for s in col]


def no_image_block(path: Path) -> dict:
    out = pd.read_parquet(path)
    block: dict = {}
    for cond, g in out.groupby("key"):
        per_variant = {}
        for v, gv in g.groupby("variant"):
            outputs = list(zip(gv["label"], gv["confidence"]))
            per_variant[str(int(v))] = {
                "n": len(gv),
                "distinct_label_conf_pairs": len(set(outputs)),
                "distinct_raw_texts": int(gv["raw_text"].nunique()),
                "labels": gv["label"].value_counts().to_dict(),
            }
        modal_per_variant = [max(d["labels"], key=d["labels"].get) for _, d in sorted(per_variant.items())]
        block[cond] = {
            "per_variant": per_variant,
            "all_repeats_identical": all(d["distinct_label_conf_pairs"] == 1 for d in per_variant.values()),
            "modal_label_per_variant": modal_per_variant,
            "agreement_if_every_patch_got_this_input": max(
                modal_per_variant.count(lbl) for lbl in set(modal_per_variant)
            ) / len(modal_per_variant),
        }
    return block


def rescoring_effect(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """
    The swap run sent the five V3-family prompts with real images and kept its
    response text, so it is the one place where those prompts' agreement signal
    can be scored both ways. The difference estimates how the stored V3 runs,
    which kept no text, would move if they could be re-scored.
    """
    if "consistency_score_regex" not in pred.columns:
        return {"note": "run not re-scored (no *_regex columns)"}
    c1, y1 = pred["consistency_score"].to_numpy(float), pred["is_correct_vs_donor"].to_numpy(bool)
    c0, y0 = pred["consistency_score_regex"].to_numpy(float), pred["is_correct_vs_donor_regex"].to_numpy(bool)

    def diff(i: np.ndarray) -> float:
        return gap_vs_random(c1[i], y1[i]) - gap_vs_random(c0[i], y0[i])

    return {
        "outcome": "modal vote scored against the supplied (donor) image's class; unweighted",
        "gap_vs_random_whole_text_rule": gap_vs_random(c0, y0),
        "gap_vs_random_label_line": gap_vs_random(c1, y1),
        "difference": diff(np.arange(len(pred))),
        "difference_ci95": boot(diff, len(pred), n_boot, seed),
    }


def swap_block(pred: pd.DataFrame, real: pd.DataFrame) -> dict:
    real_by = real.set_index("arr_idx")
    donor = real_by.loc[pred["donor_arr_idx"].to_numpy()]
    recip = real_by.loc[pred["recipient_arr_idx"].to_numpy()]
    lab_swap = labels_of(pred, as_stored=True)
    lab_donor = labels_of(donor)
    lab_recip = labels_of(recip)
    per_call_donor = np.mean([a == b for x, y in zip(lab_swap, lab_donor) for a, b in zip(x, y)])
    per_call_recip = np.mean([a == b for x, y in zip(lab_swap, lab_recip) for a, b in zip(x, y)])
    cons = stored(pred, "consistency_score").to_numpy(float)
    modal = stored(pred, "modal_label").to_numpy()
    return {
        "n_pairs": len(pred),
        "per_call_label_match_donor_stored": float(per_call_donor),
        "per_call_label_match_recipient_stored": float(per_call_recip),
        "modal_match_donor_stored": float((modal == donor["modal_label"].to_numpy()).mean()),
        "modal_match_recipient_stored": float((modal == recip["modal_label"].to_numpy()).mean()),
        "consistency_corr_with_donor_stored": float(np.corrcoef(cons, donor["consistency_score"].to_numpy(float))[0, 1]),
        "consistency_corr_with_recipient_stored": float(np.corrcoef(cons, recip["consistency_score"].to_numpy(float))[0, 1]),
        "modal_accuracy_vs_donor_label": float(pred["is_correct_vs_donor"].mean()),
        "modal_accuracy_vs_recipient_label": float(pred["is_correct_vs_recipient"].mean()),
        "note": "Stored labels come from an earlier session; the same-session drift rate is in the v3w block. "
                "Matches and correlations use the swap run's whole-text labels, as the stored run has only those; "
                "the two modal accuracies use its re-scored labels.",
    }


def weighted_shuffle_block(cs: np.ndarray, ys: np.ndarray, cs0: np.ndarray, ys0: np.ndarray,
                           cr: np.ndarray, yr: np.ndarray,
                           wdf: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """
    Cohort-weighted shuffle contrasts, in the summarize_diffs format of
    consistency_population.py so multiplicity_holm.py reads them the same way.

    The shuffle reuses the V3 selection, so the V3 inclusion weights apply to
    both arms unchanged, and resampling inside the selection cells keeps them
    fixed, as in the published weighted intervals.
    """
    w = wdf["weight"].to_numpy(float)
    idx_matrix = stratified_bootstrap_indices(wdf["cell"].to_numpy(), n_boot, seed)

    def stats(idx: np.ndarray) -> dict[str, float]:
        wi = w[idx]
        return {
            "shuffled_consistency": weighted_auc(cs[idx], ys[idx], wi),
            "shuffled_random": weighted_accuracy(ys[idx], wi),
            "shuffled_consistency_stored": weighted_auc(cs0[idx], ys0[idx], wi),
            "shuffled_random_stored": weighted_accuracy(ys0[idx], wi),
            "real_consistency": weighted_auc(cr[idx], yr[idx], wi),
            "real_random": weighted_accuracy(yr[idx], wi),
        }

    full = stats(np.arange(len(w)))
    rows = [stats(idx) for idx in idx_matrix]
    boot = {k: np.array([r[k] for r in rows]) for k in full}
    sh_gap = boot["shuffled_consistency"] - boot["shuffled_random"]
    sh0_gap = boot["shuffled_consistency_stored"] - boot["shuffled_random_stored"]
    re_gap = boot["real_consistency"] - boot["real_random"]
    full_sh = full["shuffled_consistency"] - full["shuffled_random"]
    full_sh0 = full["shuffled_consistency_stored"] - full["shuffled_random_stored"]
    full_re = full["real_consistency"] - full["real_random"]
    return {
        "auc": {k: round(v, 4) for k, v in full.items()},
        "contrasts": {
            "consistency_minus_random": summarize_diffs(sh_gap, full_sh),
            "real_consistency_minus_random": summarize_diffs(re_gap, full_re),
            "real_minus_shuffled_gap": summarize_diffs(re_gap - sh0_gap, full_re - full_sh0),
            "consistency_minus_random_stored_labels": summarize_diffs(sh0_gap, full_sh0),
        },
        "note": "Horvitz-Thompson weights from V3/weights.parquet; intervals resample inside "
                "the selection cells. consistency_minus_random is the shuffled arm, re-scored; "
                "real_minus_shuffled_gap uses the shuffled arm's whole-text labels, as the real "
                "run has only those.",
    }


def shuffle_block(sh: pd.DataFrame, real: pd.DataFrame, wdf: pd.DataFrame | None,
                  n_boot: int, seed: int) -> dict:
    if wdf is not None:
        if sorted(wdf["arr_idx"]) != sorted(sh["arr_idx"]):
            raise ValueError("shuffle patches differ from the V3 selection; weights do not apply")
        # The V3 selection order makes every bootstrap draw here the same draw
        # that produced the published V3 intervals, so both arms stay paired with them.
        sh = sh.set_index("arr_idx").loc[wdf["arr_idx"].to_numpy()].reset_index()
    real = real.set_index("arr_idx").loc[sh["arr_idx"].to_numpy()].reset_index()
    cs, ys = sh["consistency_score"].to_numpy(float), sh["is_correct"].to_numpy(bool)
    cs0, ys0 = stored(sh, "consistency_score").to_numpy(float), stored(sh, "is_correct").to_numpy(bool)
    cr, yr = real["consistency_score"].to_numpy(float), real["is_correct"].to_numpy(bool)
    n = len(sh)
    weighted = None if wdf is None else weighted_shuffle_block(cs, ys, cs0, ys0, cr, yr, wdf, n_boot, seed)

    def diff(idx: np.ndarray) -> float:
        return gap_vs_random(cr[idx], yr[idx]) - gap_vs_random(cs0[idx], ys0[idx])

    pred_dist = sh["modal_label"].value_counts(normalize=True).round(4).to_dict()
    return {
        "n": n,
        "shuffled": {
            "modal_accuracy": float(ys.mean()),
            "consistency_aauc": aauc(cs, ys),
            "random_aauc": float(ys.mean()),
            "gap_vs_random": gap_vs_random(cs, ys),
            "gap_vs_random_ci95": boot(lambda i: gap_vs_random(cs[i], ys[i]), n, n_boot, seed),
            "mean_consistency": float(cs.mean()),
            "modal_label_distribution": pred_dist,
            **strata_accuracy(cs, ys),
        },
        "real_image_same_patches": {
            "modal_accuracy": float(yr.mean()),
            "consistency_aauc": aauc(cr, yr),
            "gap_vs_random": gap_vs_random(cr, yr),
            "mean_consistency": float(cr.mean()),
            **strata_accuracy(cr, yr),
        },
        "real_minus_shuffled_gap": diff(np.arange(n)),
        "real_minus_shuffled_gap_ci95": boot(diff, n, n_boot, seed),
        "note": "Unweighted, on the outcome-stratified 1,800-patch subset; weighted_cohort applies the V3 inclusion weights to both arms.",
        "weighted_cohort": weighted,
    }


def v3w_block(v3w: pd.DataFrame, real: pd.DataFrame) -> dict:
    real = real.set_index("arr_idx").loc[v3w["arr_idx"].to_numpy()].reset_index()
    v0_new = [x[0] for x in labels_of(v3w, as_stored=True)]
    v0_old = [x[0] for x in labels_of(real)]
    disagree = int(sum(a != b for a, b in zip(v0_new, v0_old)))
    c, y = v3w["consistency_score"].to_numpy(float), v3w["is_correct"].to_numpy(bool)
    return {
        "n": len(v3w),
        "variant0_disagreements_with_stored_v3_variant0": disagree,
        "variant0_disagreement_rate": disagree / len(v3w),
        "modal_accuracy": float(y.mean()),
        "consistency_aauc": aauc(c, y),
        "random_aauc": float(y.mean()),
        "gap_vs_random": gap_vs_random(c, y),
        "mean_consistency": float(c.mean()),
        **strata_accuracy(c, y),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed} n_boot={args.n_boot}")

    base = PROJECT_ROOT / "results" / "consistency" / args.model.replace("/", "_")
    ctrl = base / "controls"
    inputs = {
        "no_image": ctrl / "no_image" / "outputs.parquet",
        "swap": ctrl / "swap" / "predictions.parquet",
        "shuffle": ctrl / "shuffle" / "predictions.parquet",
        "v3w": base / "V3W" / "predictions.parquet",
    }
    present = {k: p for k, p in inputs.items() if p.exists()}
    if not present:
        print("No control outputs found. Expected any of:\n  " + "\n  ".join(str(p) for p in inputs.values())
              + "\nRun scripts/run_image_controls.py and scripts/run_consistency_v3w.py first "
              "(docs/RUNBOOK_SR_R3.md).", file=sys.stderr)
        return 1

    real = pd.read_parquet(base / "V3" / "predictions.parquet")
    summary: dict = {"seed": args.seed, "n_boot": args.n_boot, "missing": sorted(set(inputs) - set(present))}
    if "no_image" in present:
        summary["no_image"] = no_image_block(present["no_image"])
    if "swap" in present:
        swap = pd.read_parquet(present["swap"])
        summary["swap"] = swap_block(swap, real)
        summary["swap"]["rescoring_effect"] = rescoring_effect(swap, args.n_boot, args.seed)
    if "shuffle" in present:
        wpath = base / "V3" / "weights.parquet"
        if not wpath.exists():
            print(f"Missing {wpath}; the shuffle block is unweighted only "
                  "(run consistency_selection_weights.py).", file=sys.stderr)
        wdf = pd.read_parquet(wpath) if wpath.exists() else None
        summary["shuffle"] = shuffle_block(pd.read_parquet(present["shuffle"]), real, wdf,
                                           args.n_boot, args.seed)
    if "v3w" in present:
        summary["v3w"] = v3w_block(pd.read_parquet(present["v3w"]), real)

    ctrl.mkdir(parents=True, exist_ok=True)
    out = ctrl / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:4000])
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
