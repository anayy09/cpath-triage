"""
scripts/budget_cost_table.py

Auto-confirm accuracy at fixed specialist budgets for every routing signal, with
the number of model queries each signal costs per patch and the measured cost of
a query. No API calls, no images.

Why this exists. The selective-accuracy AUC summarizes a whole curve; a reader
asking what a signal buys at a given referral budget, and at what inference
cost, needs points on that curve. Budgets 0.10, 0.20 and 0.30 (fraction routed
to the specialist) bracket the 0.15 operating point of Table 3.

Signals and their samples:
    full splits     MedGemma and Gemma-3 verbalized confidence (1 query), zero-shot
                    PLIP (local, no query), ResNet-18 external only (leakage on
                    validation); ties resolved by expectation, as in Table 3
    K=5 subsets     MedGemma single-query confidence (1 query), mean-of-5
                    confidence and agreement (5 queries), V3W agreement (5 queries,
                    validation until the external run exists); cohort-weighted with
                    the selection weights, as in Table 8

Query cost comes from the per-call logs the round-3 and round-4 runs keep
(calls.jsonl: latency, prompt and completion tokens); the earlier runs did not
log per call.

Output: results/routing/budget_cost_table.json

Usage:
    python scripts/budget_cost_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_paper_figures import _cnn_curves

from src.triage.router import risk_coverage_curve
from src.triage.weighted import weighted_accuracy, weighted_expected_curve

RES = PROJECT_ROOT / "results"
CONS = RES / "consistency" / "medgemma-27b-it"
OUT = RES / "routing" / "budget_cost_table.json"
BUDGETS = (0.10, 0.20, 0.30)
CALL_LOGS = {
    "V3W validation": CONS / "V3W" / "calls.jsonl",
    "V3W external": CONS / "V3W_test" / "calls.jsonl",
    "pixel-shuffle control": CONS / "controls" / "shuffle" / "calls.jsonl",
}


def at_budgets(curve: dict) -> dict:
    b = np.round(np.array(curve["budgets"]), 4)
    acc = np.array(curve["auto_confirm_acc"], dtype=float)
    return {f"{x:.2f}": round(float(acc[int(np.flatnonzero(b == round(x, 4))[0])]), 4)
            for x in BUDGETS}


def full_split_rows() -> dict:
    rows = {}
    for model in ("medgemma-27b-it", "gemma-3-27b-it"):
        for split, tag in (("val", ""), ("test", "_test")):
            df = pd.read_parquet(RES / "zeroshot" / model / f"V3_full{tag}" / "predictions.parquet")
            corr = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
            c = risk_coverage_curve(df["pred_confidence"].to_numpy(float), corr, tie_break="expected")
            rows[f"{model} verbalized|{split}"] = {
                "queries_per_patch": 1, "sample": "full split", "base_accuracy": round(float(corr.mean()), 4),
                "auto_confirm_accuracy": at_budgets(c)}
    for split in ("val", "test"):
        p = RES / "plip" / f"predictions_zeroshot_ensemble_{split}.parquet"
        df = pd.read_parquet(p)
        corr = df["correct"].to_numpy(bool)
        c = risk_coverage_curve(df["confidence"].to_numpy(float), corr, tie_break="expected")
        rows[f"PLIP zero-shot|{split}"] = {
            "queries_per_patch": 0, "sample": "full split, local model",
            "base_accuracy": round(float(corr.mean()), 4), "auto_confirm_accuracy": at_budgets(c)}
    cnn_T = json.loads((RES / "calibration" / "resnet18_64px" / "calibration_params.json").read_text())["T"]
    d = _cnn_curves("test", cnn_T)
    rows["ResNet-18 64 px|test"] = {
        "queries_per_patch": 0, "sample": "full split, local model",
        "base_accuracy": round(d["acc"], 4), "auto_confirm_accuracy": at_budgets(d["curve"])}
    return rows


def k5_rows() -> dict:
    rows = {}
    for split, run, wdir in (("val", "V3", "V3"), ("test", "V3_test", "V3_test"),
                             ("val", "V3W", "V3"), ("test", "V3W_test", "V3_test")):
        p = CONS / run / "predictions.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        w = pd.read_parquet(CONS / wdir / "weights.parquet")["weight"].to_numpy(float)
        true = df["true_label_name"].to_numpy()
        modal_corr = (df["modal_label"].to_numpy() == true)
        labels = [json.loads(s) for s in df["raw_labels"]]
        confs = np.array([json.loads(s) for s in df["raw_confs"]], dtype=float)
        v0_corr = np.array([r[0] for r in labels]) == true
        signals = {"agreement": (df["consistency_score"].to_numpy(float), modal_corr, 5)}
        if run.startswith("V3") and not run.startswith("V3W"):
            signals["single-query confidence"] = (confs[:, 0], v0_corr, 1)
            signals["mean-of-5 confidence"] = (df["mean_textual_conf"].to_numpy(float), modal_corr, 5)
        fam = "V3W" if run.startswith("V3W") else "V3"
        for name, (sig, corr, q) in signals.items():
            c = weighted_expected_curve(sig, corr, w)
            rows[f"MedGemma {fam} {name}|{split}"] = {
                "queries_per_patch": q, "sample": "K=5 subset, cohort-weighted",
                "base_accuracy": round(weighted_accuracy(corr, w), 4),
                "auto_confirm_accuracy": at_budgets(c)}
    return rows


def call_costs() -> dict:
    out = {}
    for name, path in CALL_LOGS.items():
        if not path.exists():
            continue
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        lat = np.array([r["latency_s"] for r in recs if r.get("latency_s") is not None], dtype=float)
        pt = np.array([r.get("prompt_tokens") or 0 for r in recs], dtype=float)
        ct = np.array([r.get("completion_tokens") or 0 for r in recs], dtype=float)
        ts = pd.to_datetime([r["ts_utc"] for r in recs])
        hours = (ts.max() - ts.min()).total_seconds() / 3600
        out[name] = {
            "n_calls": len(recs),
            "latency_s_median": round(float(np.median(lat)), 2),
            "latency_s_p90": round(float(np.percentile(lat, 90)), 2),
            "prompt_tokens_mean": round(float(pt.mean()), 1),
            "completion_tokens_mean": round(float(ct.mean()), 1),
            "wall_clock_hours": round(hours, 2),
            "calls_per_minute": round(len(recs) / (hours * 60), 1) if hours > 0 else None,
        }
    return out


def main() -> int:
    result = {"budgets": list(BUDGETS), "rows": {**full_split_rows(), **k5_rows()},
              "query_cost": call_costs()}
    for k, v in result["rows"].items():
        a = v["auto_confirm_accuracy"]
        print(f"{k:48s} q={v['queries_per_patch']}  base {v['base_accuracy']:.3f}  "
              + "  ".join(f"b{b}: {a[b]:.3f}" for b in a))
    for k, v in result["query_cost"].items():
        print(k, v)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
