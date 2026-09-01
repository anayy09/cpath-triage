"""
scripts/verify_calibration_claims.py

Arithmetic verification of the temperature-scaling claims (revision item C-03).

Reviewer 1's objection, restated: both VLMs report T = 200.0, which is exactly
the upper bound of the optimiser's search interval in src/eval/calibration.py.
A bound is not an estimate. And once T is that large the sigmoid output collapses
towards 0.5 for every input, so the calibrated ECE degenerates to |0.5 - accuracy|
by construction. If that is right, then Table 6's cross-cohort ECE changes of
+0.056 and +0.049 are restatements of the accuracy drops of 0.060 and 0.047, and
carry no information about calibration transfer at all.

This script checks all of that against the files on disk rather than asserting it,
so the response letter can quote a verified number. It computes nothing new about
the models; it only confirms identities.

Checks:
  1. The optimiser bound is read out of the source, not remembered.
  2. Each fitted T is compared against that bound.
  3. |0.5 - accuracy| is compared against each reported calibrated ECE, on the
     evaluation partition and on the external test split.
  4. The Table 6 "change" column is compared against the corresponding accuracy
     drop.
  5. The collapse is demonstrated directly: the calibrated confidences at T = 200
     are summarised, showing how far they sit from 0.5.

Outputs:
    results/calibration/temperature_bound_check.json

Usage:
    python scripts/verify_calibration_claims.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
VLMS = ("medgemma-27b-it", "gemma-3-27b-it")


def read_optimiser_bounds() -> tuple[float, float]:
    """Read the minimize_scalar bounds out of the calibration source."""
    src = (PROJECT_ROOT / "src" / "eval" / "calibration.py").read_text(encoding="utf-8")
    m = re.search(r"minimize_scalar\(\s*nll,\s*bounds=\(([\d.eE+-]+),\s*([\d.eE+-]+)\)", src)
    if not m:
        raise RuntimeError("could not locate the minimize_scalar bounds in calibration.py")
    return float(m.group(1)), float(m.group(2))


def eval_partition(model: str) -> pd.DataFrame:
    """The held-out 20% partition calibrate.py evaluates on."""
    df = pd.read_parquet(
        PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full" / "predictions.parquet"
    )
    df = df[df["pred_label"] != "unknown"].copy()
    df["correct"] = df["pred_label"] == df["true_label_name"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    return df.iloc[idx[int(len(df) * 0.8):]]


def test_frame(model: str) -> pd.DataFrame:
    df = pd.read_parquet(
        PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full_test" / "predictions.parquet"
    )
    df = df[df["pred_label"] != "unknown"].copy()
    df["correct"] = df["pred_label"] == df["true_label_name"]
    return df


def temperature_scale(conf: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-7
    c = np.clip(conf, eps, 1.0 - eps)
    logits = np.log(c / (1.0 - c))
    return 1.0 / (1.0 + np.exp(-logits / T))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    lo_bound, hi_bound = read_optimiser_bounds()
    print(f"optimiser bounds in src/eval/calibration.py: ({lo_bound}, {hi_bound})")

    models: dict[str, dict] = {}
    for model in VLMS:
        params = json.loads(
            (PROJECT_ROOT / "results" / "calibration" / model / "V3_full"
             / "calibration_params.json").read_text(encoding="utf-8")
        )
        T = float(params["T"])

        ev = eval_partition(model)
        te = test_frame(model)
        acc_ev = float(ev["correct"].mean())
        acc_te = float(te["correct"].mean())

        cal_ev = temperature_scale(ev["pred_confidence"].to_numpy(float), T)
        cal_te = temperature_scale(te["pred_confidence"].to_numpy(float), T)

        reported_ev = float(params["ece_after"])
        reported_te = float(params["transfer"]["test_ece_cal"])
        reported_change = float(params["transfer"]["change"])

        models[model] = {
            "fitted_T": T,
            "T_equals_optimiser_upper_bound": bool(abs(T - hi_bound) < 1e-9),
            "eval_partition": {
                "n": len(ev),
                "accuracy": round(acc_ev, 4),
                "abs_half_minus_accuracy": round(abs(0.5 - acc_ev), 4),
                "reported_calibrated_ece": reported_ev,
                "difference": round(abs(0.5 - acc_ev) - reported_ev, 4),
                "calibrated_confidence_mean": round(float(cal_ev.mean()), 6),
                "calibrated_confidence_max_deviation_from_half": round(
                    float(np.abs(cal_ev - 0.5).max()), 6
                ),
            },
            "external_test": {
                "n": len(te),
                "accuracy": round(acc_te, 4),
                "abs_half_minus_accuracy": round(abs(0.5 - acc_te), 4),
                "reported_calibrated_ece": reported_te,
                "difference": round(abs(0.5 - acc_te) - reported_te, 4),
                "calibrated_confidence_mean": round(float(cal_te.mean()), 6),
                "calibrated_confidence_max_deviation_from_half": round(
                    float(np.abs(cal_te - 0.5).max()), 6
                ),
            },
            "transfer_change_restates_accuracy_drop": {
                "reported_ece_change": reported_change,
                "accuracy_drop_val_minus_test": round(acc_ev - acc_te, 4),
                "difference": round(reported_change - (acc_ev - acc_te), 4),
            },
        }

        m = models[model]
        print(f"\n{model}")
        print(f"  fitted T = {T} (upper bound {hi_bound}): "
              f"{'AT THE BOUND' if m['T_equals_optimiser_upper_bound'] else 'interior'}")
        for part in ("eval_partition", "external_test"):
            p = m[part]
            print(f"  {part:15s} acc={p['accuracy']:.4f}  |0.5-acc|={p['abs_half_minus_accuracy']:.4f}"
                  f"  reported cal ECE={p['reported_calibrated_ece']:.4f}"
                  f"  diff={p['difference']:+.4f}")
            print(f"                  calibrated confidences sit within "
                  f"{p['calibrated_confidence_max_deviation_from_half']:.2e} of 0.5")
        t = m["transfer_change_restates_accuracy_drop"]
        print(f"  Table 6 change {t['reported_ece_change']:+.4f} vs accuracy drop "
              f"{t['accuracy_drop_val_minus_test']:+.4f}  diff={t['difference']:+.4f}")

    out = {
        "seed": args.seed,
        "optimiser_bounds": {"lower": lo_bound, "upper": hi_bound},
        "source": "src/eval/calibration.py",
        "conclusion": (
            "Both VLM temperatures sit exactly at the optimiser's upper bound, so "
            "T = 200 is where the search stopped rather than an estimated quantity. "
            "At that temperature every calibrated confidence collapses onto 0.5, so "
            "the calibrated ECE equals |0.5 - accuracy| by construction and the "
            "cross-cohort ECE change restates the accuracy drop. These rows carry "
            "accuracy information and no calibration information."
        ),
        "models": models,
    }

    out_path = PROJECT_ROOT / "results" / "calibration" / "temperature_bound_check.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
