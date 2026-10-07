"""
scripts/parse_failure_check.py

How Gemma-3's out-of-vocabulary answers are treated, artifact by artifact, and
what Tables 5 and 9 would read if they were kept and scored as incorrect.

Why this exists. Methods says Gemma-3's parse failures are scored as incorrect
rather than excluded. That is true of accuracy (classification_stats.py), of
routing (routing_ci.py) and of the full-split AUROC (confidence_information.py),
but calibrate.py and calibration_extras.py drop them before the 80/20 split. The
eight responses are not unreadable: each names a label outside the nine classes
(for example "fungal hyphae" or "liver") and states a confidence, so keeping
them as confident errors is well defined. This script measures the effect
rather than leaving the sentence to be checked by hand.

Keeping the rows changes the permutation length, so the held-out partition is
not the published one plus two rows; it is the partition calibrate.py would
have drawn from 10,004 rows. Both are reported.

Outputs:
    results/calibration/parse_failure_check.json

Usage:
    python scripts/parse_failure_check.py [--seed 42]
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

from src.eval.calibration import (
    TemperatureScaler,
    adaptive_ece_score,
    brier_binary,
    ece_score,
)

MODEL = "gemma-3-27b-it"
ZS = PROJECT_ROOT / "results" / "zeroshot" / MODEL


def frame(split: str) -> pd.DataFrame:
    suffix = "V3_full" if split == "val" else "V3_full_test"
    df = pd.read_parquet(ZS / suffix / "predictions.parquet")
    return df.assign(correct=df["pred_label"] == df["true_label_name"])


def tables_5_and_9(val: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """calibrate.py's procedure on whatever rows it is given."""
    idx = np.random.default_rng(seed).permutation(len(val))
    cut = int(len(val) * 0.8)
    cal, ev = val.iloc[idx[:cut]], val.iloc[idx[cut:]]
    scaler = TemperatureScaler()
    T = scaler.fit_scalar(cal["pred_confidence"].to_numpy(float), cal["correct"].to_numpy(bool))
    ec, ey = ev["pred_confidence"].to_numpy(float), ev["correct"].to_numpy(bool)
    tc, ty = test["pred_confidence"].to_numpy(float), test["correct"].to_numpy(bool)
    val_cal = ece_score(scaler.transform_scalar(ec), ey)
    test_cal = ece_score(scaler.transform_scalar(tc), ty)
    return {
        "n_cal": len(cal), "n_eval": len(ev), "n_test": len(test),
        "table5": {"ece_raw": round(float(ece_score(ec, ey)), 4),
                   "ece_adaptive": round(float(adaptive_ece_score(ec, ey)), 4),
                   "brier": round(float(brier_binary(ec, ey)), 4),
                   "T": round(float(T), 4),
                   "ece_calibrated": round(float(val_cal), 4)},
        "table9": {"val_ece_cal": round(float(val_cal), 4),
                   "test_ece_raw": round(float(ece_score(tc, ty)), 4),
                   "test_ece_cal": round(float(test_cal), 4),
                   "change": round(float(test_cal - val_cal), 4)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed}")

    val, test = frame("val"), frame("test")
    failures = {
        split: df.loc[df["pred_label"] == "unknown", ["arr_idx", "true_label_name", "pred_confidence"]]
        .assign(raw_label=df.loc[df["pred_label"] == "unknown", "raw_text"]
                .str.extract(r"LABEL:\s*([^\n]+)")[0].str.strip())
        .to_dict(orient="records")
        for split, df in (("val", val), ("test", test))
    }
    keep = val["pred_label"] != "unknown"
    keep_t = test["pred_label"] != "unknown"
    excluded = tables_5_and_9(val[keep], test[keep_t], args.seed)
    retained = tables_5_and_9(val, test, args.seed)

    pc = json.loads((PROJECT_ROOT / "results" / "calibration" / MODEL / "V3_full"
                     / "calibration_params.json").read_text(encoding="utf-8"))
    out = {
        "seed": args.seed,
        "model": MODEL,
        "failures": failures,
        "what_the_failures_are": (
            "out-of-vocabulary labels with a parsed confidence, not unreadable responses; "
            "pred_label is set to 'unknown' and the stated confidence is kept"
        ),
        "treatment_by_artifact": {
            "classification_stats.py (Tables 3, 4, 7 accuracy and F1)": "kept, scored incorrect",
            "routing_ci.py (Table 6)": "kept, scored incorrect, ranked by stated confidence",
            "confidence_information.py full-split AUROC and MI": "kept, scored incorrect",
            "confidence_information.py evaluation-partition rows": "excluded",
            "calibrate.py (Tables 5 and 9)": "excluded before the 80/20 permutation",
            "calibration_extras.py (Table 5 adaptive ECE, Brier)": "excluded",
            "run_logprob_confidence.py (validation sample = evaluation partition)": "excluded",
        },
        "published": {"table5": {"ece_raw": pc["ece_before"], "brier": pc["brier_before"],
                                 "T": pc["T"], "ece_calibrated": pc["ece_after"]},
                      "table9": pc["transfer"]},
        "recomputed_excluded": excluded,
        "recomputed_retained_as_incorrect": retained,
    }
    out_path = PROJECT_ROOT / "results" / "calibration" / "parse_failure_check.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("published", "recomputed_excluded",
                                          "recomputed_retained_as_incorrect")}, indent=1))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
