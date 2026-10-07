"""
scripts/logprob_censoring_sensitivity.py

How much the label-token results depend on the top-20 cutoff.

Why this exists. The label-token confidence is the predicted class's share of
the probability mass over whichever of the nine classes appear among the 20
alternatives the endpoint returns at the first label token. When a class falls
outside the top 20 it drops out of the denominator, so the number of classes
behind each confidence varies from patch to patch and between models. If that
number tracks correctness, the censoring itself can manufacture or erase a
ranking signal, and a difference between models or cohorts could be a
difference in censoring rather than in the distribution.

The raw alternatives were not stored, only the renormalized confidence and the
count of classes found, so no correction can be computed from disk and no
bound on the missing mass is claimed here. What can be done is to condition on
the censoring: recompute every Table S2 contrast on the patches where all nine
classes were returned, where the denominator is complete up to the probability
of non-label tokens, and within each stratum of the class count that is large
enough to say anything.

Ties are resolved by the closed-form expectation throughout, so the full-sample
rows here can differ from the Table S2 verbalized rows, which were computed as
a 200-order Monte Carlo mean, in the fourth decimal.

Outputs:
    results/logprob_confidence/censoring_sensitivity.json

Usage:
    python scripts/logprob_censoring_sensitivity.py [--n-boot 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from logprob_comparison import auroc_midrank, paired_bootstrap

from src.triage.router import (
    risk_coverage_curve,
    tie_statistics,
)

LP_ROOT = PROJECT_ROOT / "results" / "logprob_confidence"
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")
SPLITS = ("val", "test")
# Below this a bootstrap interval on a selective-accuracy AUC difference is too
# wide to inform anything; strata smaller than this get point estimates only.
MIN_N_FOR_CI = 300
MIN_N_FOR_POINT = 100


def load(model: str, split: str) -> pd.DataFrame:
    path = LP_ROOT / model / split / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_parquet(path)
    return df[df["logprob_conf"].notna() & ~df["call_failed"]].copy()


def contrasts(df: pd.DataFrame, n_boot: int, seed: int, with_ci: bool) -> dict:
    """The Table S2 statistics on one subset."""
    y = df["correct"].to_numpy(bool)
    lt = df["logprob_conf"].to_numpy(float)
    vb = df["verbalized_conf"].to_numpy(float)
    out: dict = {"n": len(df), "accuracy": round(float(y.mean()), 4)}
    if y.all() or not y.any():
        out["note"] = "single outcome class; no ranking statistic defined"
        return out
    rnd = float(y.mean())  # random routing AUC is the base accuracy exactly
    a_lt = risk_coverage_curve(lt, y, tie_break="expected")["auc"]
    a_vb = risk_coverage_curve(vb, y, tie_break="expected")["auc"]
    out.update({
        "random_auc": round(float(rnd), 4),
        "label_token": {"auroc": round(auroc_midrank(lt, y), 4), "aauc": round(float(a_lt), 4),
                        "tie_fraction": round(float(tie_statistics(lt)["tie_fraction"]), 4)},
        "verbalized": {"auroc": round(auroc_midrank(vb, y), 4), "aauc": round(float(a_vb), 4),
                       "tie_fraction": round(float(tie_statistics(vb)["tie_fraction"]), 4)},
    })
    for name, point, a, b in (
        ("label_token_minus_random", a_lt - rnd, lt, None),
        ("verbalized_minus_random", a_vb - rnd, vb, None),
        ("label_token_minus_verbalized", a_lt - a_vb, lt, vb),
    ):
        rec: dict = {"gap_point": round(float(point), 4)}
        if with_ci:
            bt = paired_bootstrap(a, b, y, n_boot, seed)
            rec.update({"ci": [bt["ci_2.5"], bt["ci_97.5"]], "excludes_zero": bt["excludes_zero"]})
        out[name] = rec
    return out


def censoring_profile(df: pd.DataFrame) -> dict:
    """How the class count is distributed and whether it tracks correctness."""
    k = df["n_classes_in_topk"].to_numpy(int)
    y = df["correct"].to_numpy(bool)
    by_k = df.groupby("n_classes_in_topk").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                               mean_label_token_conf=("logprob_conf", "mean"))
    by_pred = df.groupby("pred_label").agg(n=("correct", "size"),
                                           mean_n_classes=("n_classes_in_topk", "mean"))
    return {
        "mean_n_classes": round(float(k.mean()), 3),
        "distribution": {int(i): int(r["n"]) for i, r in by_k.iterrows()},
        "accuracy_by_n_classes": {int(i): round(float(r["accuracy"]), 4) for i, r in by_k.iterrows()},
        "mean_label_token_conf_by_n_classes": {int(i): round(float(r["mean_label_token_conf"]), 4)
                                               for i, r in by_k.iterrows()},
        # Fewer surviving classes means a more peaked distribution, so a negative
        # count works as a confidence signal of its own; its AUROC says how much
        # the censoring is entangled with correctness.
        "auroc_of_minus_n_classes_for_correctness": round(auroc_midrank(-k.astype(float), y), 4),
        "fraction_all_nine_present": round(float((k == 9).mean()), 4),
        "mean_n_classes_by_predicted_class": {
            str(i): {"n": int(r["n"]), "mean_n_classes": round(float(r["mean_n_classes"]), 3)}
            for i, r in by_pred.sort_values("n", ascending=False).iterrows()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed} n_boot={args.n_boot}")

    design = json.loads((LP_ROOT / "design_pilot.json").read_text(encoding="utf-8"))
    runs: dict = {}
    for model in MODELS:
        for split in SPLITS:
            df = load(model, split)
            key = f"{model}|{split}"
            strata: dict = {}
            for k, sub in df.groupby("n_classes_in_topk"):
                if len(sub) >= MIN_N_FOR_POINT:
                    strata[int(k)] = contrasts(sub, args.n_boot, args.seed, len(sub) >= MIN_N_FOR_CI)
            runs[key] = {
                "censoring_profile": censoring_profile(df),
                "full_sample": contrasts(df, args.n_boot, args.seed, True),
                "all_nine_classes_present": contrasts(df[df["n_classes_in_topk"] == 9],
                                                      args.n_boot, args.seed, True),
                "by_n_classes_stratum": strata,
            }
            f9 = runs[key]["all_nine_classes_present"]
            fs = runs[key]["full_sample"]
            print(f"\n{key}: mean classes {runs[key]['censoring_profile']['mean_n_classes']}, "
                  f"nine-of-nine n={f9['n']} acc={f9['accuracy']}")
            for c in ("label_token_minus_random", "verbalized_minus_random",
                      "label_token_minus_verbalized"):
                print(f"  {c:30} full {fs[c]['gap_point']:+.4f} {fs[c]['ci']}   "
                      f"nine {f9[c]['gap_point']:+.4f} {f9[c]['ci']}")

    pilot_means = {m: design["results"][m]["classes_matched_in_topk"]["mean"] for m in MODELS}
    out = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "tie_break": "expected (closed form) for every AUC",
        "missing_mass_bound": (
            "Not computable from disk. The runs stored only the renormalized confidence and "
            "the number of classes found among the top-20 alternatives, not the alternatives' "
            "log-probabilities, so the probability of a class outside the top 20 cannot be "
            "bounded here."
        ),
        "strata_rules": {"min_n_for_point": MIN_N_FOR_POINT, "min_n_for_ci": MIN_N_FOR_CI},
        "design_pilot_mean_classes_note": (
            "The 8.28 and 6.72 quoted in Methods are the 50-patch design pilot means "
            "(design_pilot.json); the full-run means are in censoring_profile."
        ),
        "design_pilot_mean_classes": pilot_means,
        "runs": runs,
    }
    out_path = LP_ROOT / "censoring_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
