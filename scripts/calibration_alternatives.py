"""
scripts/calibration_alternatives.py

Post-hoc calibrators beyond log-odds temperature scaling, for the two VLMs'
verbalized confidence. No API calls.

Why this exists. Temperature scaling saturates at the upper end of its search
interval for both VLMs, which leaves open whether a less restricted map would do
better. Every map here is fitted on the calibration split that calibrate.py uses
(parse failures dropped, default_rng(seed) permutation, first 80%) and evaluated
on the held-out 20% and, without refitting, on the full external split:

    raw          the stated confidence
    temperature  sigmoid(logit(c) / T), T in [0.05, 200]  (as in the paper)
    platt        sigmoid(a * logit(c) + b), slope and intercept by unpenalized
                 maximum likelihood
    isotonic     monotone non-decreasing step function (pool adjacent violators)
    constant     the calibration split's accuracy for every patch, a reference
                 that is calibrated in the large and carries no ranking at all

Metrics: ECE (15 equal-width bins), adaptive ECE (15 equal-mass bins), Brier,
binary NLL, AUROC (mid-ranks), and Murphy's decomposition of the Brier score
over the distinct calibrated values: Brier = reliability - resolution +
uncertainty. Verbalized confidence takes a handful of values, so grouping by
value is exact rather than binned. Resolution is what a recalibration cannot
create: a map that is monotone in the stated number can merge groups but never
split them, so it can lower reliability toward zero and can only lose resolution.

Output: results/calibration/calibration_alternatives.json

Usage:
    python scripts/calibration_alternatives.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from confidence_information import auroc_midrank

from src.eval.calibration import TemperatureScaler, adaptive_ece_score, ece_score

OUT = PROJECT_ROOT / "results" / "calibration" / "calibration_alternatives.json"
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")
EPS = 1e-7  # the clip calibration.py applies before the logit


def frame(model: str, split_tag: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(PROJECT_ROOT / "results" / "zeroshot" / model / f"V3_full{split_tag}"
                         / "predictions.parquet")
    df = df[df["pred_label"] != "unknown"]
    return (df["pred_confidence"].to_numpy(float),
            (df["pred_label"] == df["true_label_name"]).to_numpy(float))


def logit(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, EPS, 1 - EPS)
    return np.log(c / (1 - c))


def murphy(p: np.ndarray, y: np.ndarray) -> dict:
    base = y.mean()
    keys = np.round(p, 9)
    rel = res = 0.0
    for v in np.unique(keys):
        m = keys == v
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return {"reliability": round(float(rel), 4), "resolution": round(float(res), 4),
            "uncertainty": round(float(base * (1 - base)), 4),
            "n_distinct": len(np.unique(keys))}


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    q = np.clip(p, EPS, 1 - EPS)
    return {
        "n": len(y),
        "accuracy": round(float(y.mean()), 4),
        "ece": round(ece_score(p, y.astype(bool), n_bins=15), 4),
        "adaptive_ece": round(adaptive_ece_score(p, y.astype(bool), n_bins=15), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "nll": round(float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), 4),
        "auroc": round(auroc_midrank(p, y.astype(bool)), 4),
        "brier_decomposition": murphy(p, y),
    }


def fit_maps(c: np.ndarray, y: np.ndarray) -> tuple[dict, dict]:
    ts = TemperatureScaler()
    ts.fit_scalar(c, y)
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    platt.fit(logit(c).reshape(-1, 1), y)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(c, y)
    acc = float(y.mean())
    maps = {
        "raw": lambda x: x,
        "temperature": lambda x: ts.transform_scalar(x),
        "platt": lambda x: platt.predict_proba(logit(x).reshape(-1, 1))[:, 1],
        "isotonic": lambda x: iso.predict(x),
        "constant": lambda x: np.full(len(x), acc),
    }
    params = {
        "temperature_T": round(float(ts.T), 4),
        "platt_slope": round(float(platt.coef_[0, 0]), 4),
        "platt_intercept": round(float(platt.intercept_[0]), 4),
        "isotonic_map": {f"{v:g}": round(float(iso.predict([v])[0]), 4) for v in np.unique(c)},
        "constant": round(acc, 4),
    }
    return maps, params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"seed={args.seed}")

    out: dict = {"seed": args.seed, "models": {}}
    for model in MODELS:
        c_all, y_all = frame(model, "")
        idx = np.random.default_rng(args.seed).permutation(len(c_all))
        n_cal = int(len(c_all) * 0.8)
        cal, ev = idx[:n_cal], idx[n_cal:]
        c_test, y_test = frame(model, "_test")
        maps, params = fit_maps(c_all[cal], y_all[cal])
        res = {"n_calibration": int(n_cal), "params": params, "eval": {}, "external": {}}
        for name, f in maps.items():
            res["eval"][name] = metrics(np.asarray(f(c_all[ev]), float), y_all[ev])
            res["external"][name] = metrics(np.asarray(f(c_test), float), y_test)
            e, x = res["eval"][name], res["external"][name]
            print(f"{model:16s} {name:11s} eval ECE {e['ece']:.3f} Brier {e['brier']:.3f} "
                  f"NLL {e['nll']:.3f} AUROC {e['auroc']:.3f} res {e['brier_decomposition']['resolution']:.4f}"
                  f" | ext ECE {x['ece']:.3f} Brier {x['brier']:.3f} NLL {x['nll']:.3f}")
        out["models"][model] = res
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
