"""
scripts/nll_clip_sensitivity.py

How the log-odds sums and the binary NLL of Section 3.3 depend on the clip that
turns a verbalized confidence of 100 into a finite log-odds. No API calls.

src/eval/calibration.py clips confidences to [eps, 1 - eps] with eps = 1e-7, so a
stated 100 becomes log-odds 16.1. MedGemma states 100 on a few percent of
patches, which inflates the summed log-odds and the NLL at T = 1. The fitted
temperature rests on the sufficient condition (every confidence above 0.5 and the
log-odds summed over errors at least the sum over correct predictions) and on the
measured curve, so both are recomputed here for clips from 1e-2 to 1e-7.

Output: results/calibration/nll_clip_sensitivity.json

Usage:
    python scripts/nll_clip_sensitivity.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from temperature_nll_curve import (
    T_GRID,
    calibration_partition,
    condition_statistics,
    exact_nll,
    monotone_decreasing,
)

OUT = PROJECT_ROOT / "results" / "calibration" / "nll_clip_sensitivity.json"
CLIPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"seed={args.seed}")

    out: dict = {"seed": args.seed, "clips": list(CLIPS), "models": {}}
    for model in MODELS:
        conf, y = calibration_partition(model, args.seed)
        at_100 = conf >= 1.0
        rows = {}
        for eps in CLIPS:
            c = np.clip(conf, eps, 1.0 - eps)
            z = np.log(c / (1.0 - c))
            vals = np.array([exact_nll(z, y, T) for T in T_GRID])
            cond = condition_statistics(z, y)
            rows[f"{eps:g}"] = {
                "logodds_at_100": round(float(np.log((1 - eps) / eps)), 3),
                "logodds_mass_on_errors": cond["logodds_mass_on_errors"],
                "logodds_mass_on_correct": cond["logodds_mass_on_correct"],
                "mean_logodds_errors": round(float(z[y == 0].mean()), 3),
                "sufficient_condition_holds": cond["sufficient_condition_holds"],
                "nll_T1": round(exact_nll(z, y, 1.0), 4),
                "nll_T200": round(exact_nll(z, y, 200.0), 4),
                "monotone_decreasing_on_grid": monotone_decreasing(vals),
                "grid_minimiser_T": round(float(T_GRID[int(np.argmin(vals))]), 2),
            }
            r = rows[f"{eps:g}"]
            print(f"{model:16s} eps {eps:g}: errors {r['logodds_mass_on_errors']:.0f} "
                  f"correct {r['logodds_mass_on_correct']:.0f} holds {r['sufficient_condition_holds']} "
                  f"NLL(1) {r['nll_T1']:.3f} NLL(200) {r['nll_T200']:.4f} "
                  f"monotone {r['monotone_decreasing_on_grid']}")
        out["models"][model] = {
            "n_calibration": len(conf),
            "n_at_100": int(at_100.sum()),
            "share_at_100": round(float(at_100.mean()), 4),
            "n_at_100_errors": int((at_100 & (y == 0)).sum()),
            "n_at_100_correct": int((at_100 & (y == 1)).sum()),
            "by_clip": rows,
        }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
