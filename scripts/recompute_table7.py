"""
scripts/recompute_table7.py

Recompute Table 7 (K=5 majority voting versus a single query) against the
in-session single-query baseline (revision item C-13, and the override in
docs/DECISIONS_SR_R1.md that makes this cost nothing).

The problem Reviewer 1 identified. The published Table 7 compared the K=5 modal
vote against the *stored full-scale* single-query prediction, which was produced
in an earlier session. On validation those two disagree on 54 of 1,800 patches
(3.0%); on the external test split they disagree on 0 of 1,800. Under a common
per-call disagreement rate those two observations are incompatible, so the two
runs did not experience the same endpoint conditions. That matters because 54
drifted labels is the same order as the 60 discordant pairs that produced the
published p = 0.245, meaning the voting comparison was partly measuring session
drift rather than the effect of voting.

The fix needs no new inference. Variant 0 of the K=5 experiment *is* the
unmodified base prompt, issued inside the same session as the other four
variants. Using it as the single-query baseline makes the comparison in-session
by construction and removes the stale prediction entirely.

This writes a new file rather than overwriting matched_baseline.json, because the
difference between the two is evidence for the response letter.

Outputs:
    results/consistency/{model}/{run}/matched_baseline_v2.json

Usage:
    python scripts/recompute_table7.py --split val
    python scripts/recompute_table7.py --split test
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

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test (binomial on discordant pairs) for paired correctness."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_value": 1.0}
    return {
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_disc,
        "p_value": round(float(binomtest(b_only, n_disc, 0.5).pvalue), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(":", "_")
    run_dir = args.version + ("_test" if args.split == "test" else "")
    cons_dir = PROJECT_ROOT / "results" / "consistency" / model_slug / run_dir
    pred_path = cons_dir / "predictions.parquet"
    full_suffix = f"{args.version}_full" + ("_test" if args.split == "test" else "")
    full_path = (
        PROJECT_ROOT / "results" / "zeroshot" / model_slug / full_suffix / "predictions.parquet"
    )
    for p in (pred_path, full_path):
        if not p.exists():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    df = pd.read_parquet(pred_path)
    truth = df["true_label_name"].to_numpy()
    labels_per_patch = [json.loads(s) for s in df["raw_labels"]]
    confs_per_patch = [json.loads(s) for s in df["raw_confs"]]

    v0_pred = np.array([lab[0] for lab in labels_per_patch])
    v0_conf = np.array([c[0] for c in confs_per_patch], dtype=float)
    v0_correct = v0_pred == truth

    modal_pred = df["modal_label"].to_numpy()
    modal_correct = df["is_correct"].to_numpy(bool)

    # The stale baseline, kept only to quantify what changed.
    full_df = pd.read_parquet(full_path)
    stale = (
        full_df[full_df["arr_idx"].isin(df["arr_idx"])]
        .drop_duplicates(subset="arr_idx")
        .set_index("arr_idx")
        .reindex(df["arr_idx"].to_numpy())
    )
    stale_pred = stale["pred_label"].to_numpy()
    stale_correct = stale_pred == truth
    drift = int((stale_pred != v0_pred).sum())

    in_session = {
        "accuracy": round(float(accuracy_score(truth, v0_pred)), 4),
        "macro_f1": round(
            float(f1_score(truth, v0_pred, average="macro", labels=ALL_LABELS, zero_division=0)), 4
        ),
        "mean_confidence": round(float(v0_conf.mean()), 4),
    }
    modal = {
        "accuracy": round(float(accuracy_score(truth, modal_pred)), 4),
        "macro_f1": round(
            float(f1_score(truth, modal_pred, average="macro", labels=ALL_LABELS, zero_division=0)),
            4,
        ),
    }
    stale_stats = {
        "accuracy": round(float(accuracy_score(truth, stale_pred)), 4),
        "macro_f1": round(
            float(f1_score(truth, stale_pred, average="macro", labels=ALL_LABELS, zero_division=0)),
            4,
        ),
    }

    mcnemar_in_session = mcnemar_test(modal_correct, v0_correct)
    mcnemar_stale = mcnemar_test(modal_correct, stale_correct)

    result = {
        "model": args.model,
        "version": args.version,
        "split": args.split,
        "n_patches": len(df),
        "single_query_baseline": "variant_0_in_session",
        "in_session_single_query": in_session,
        "modal_vote_k5": modal,
        "superseded_stale_single_query": stale_stats,
        "session_drift": {
            "n_label_changes_stale_vs_in_session": drift,
            "fraction": round(drift / len(df), 4),
            "note": (
                "Both runs used temperature 0.0. A 3.0% disagreement on validation "
                "against 0.0% on the external test split cannot come from a common "
                "per-call rate, so the two runs did not see identical endpoint "
                "conditions. This is reported rather than explained away."
            ),
        },
        "mcnemar_voting_vs_in_session_single_query": mcnemar_in_session,
        "mcnemar_voting_vs_stale_single_query_superseded": mcnemar_stale,
        "deltas_in_session": {
            "accuracy": round(modal["accuracy"] - in_session["accuracy"], 4),
            "macro_f1": round(modal["macro_f1"] - in_session["macro_f1"], 4),
        },
    }

    out_path = cons_dir / "matched_baseline_v2.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== Table 7 recomputed in-session | {args.model} {args.version} "
          f"{args.split} (n={len(df)}) ===")
    print(f"  session drift, stale vs in-session variant 0: {drift} labels "
          f"({drift / len(df):.2%})")
    print(f"{'':<26}{'accuracy':>10}{'macro F1':>10}")
    print(f"{'single query (in-session)':<26}{in_session['accuracy']:>10.4f}"
          f"{in_session['macro_f1']:>10.4f}")
    print(f"{'single query (stale, old)':<26}{stale_stats['accuracy']:>10.4f}"
          f"{stale_stats['macro_f1']:>10.4f}")
    print(f"{'modal vote K=5':<26}{modal['accuracy']:>10.4f}{modal['macro_f1']:>10.4f}")
    print(f"{'delta (in-session)':<26}{result['deltas_in_session']['accuracy']:>+10.4f}"
          f"{result['deltas_in_session']['macro_f1']:>+10.4f}")
    print()
    m, s = mcnemar_in_session, mcnemar_stale
    print(f"  McNemar vs in-session : modal wins {m['a_only']}, single wins {m['b_only']}, "
          f"p = {m['p_value']}")
    print(f"  McNemar vs stale (old): modal wins {s['a_only']}, single wins {s['b_only']}, "
          f"p = {s['p_value']}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
