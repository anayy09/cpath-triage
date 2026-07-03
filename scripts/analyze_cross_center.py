"""
scripts/analyze_cross_center.py

Stage 6: cross-center generalization analysis.

Loads all val and test predictions, applies the val-fitted calibration (T
values from Stage 4) to the test split without re-fitting, and computes:
  - Classification accuracy/F1 table (val vs test degradation)
  - Calibration transfer: ECE before/after on test with val-fitted T
  - Risk-coverage curves on test for each model and policy
  - Operating-point comparison at 15% routing budget

Outputs (under results/stage6/):
    cross_center_table.json       -- all metrics in one place
    cross_center_table.md         -- markdown table for the paper
    calibration_transfer.json     -- ECE on test before/after T scaling
    routing_test_vlm.png          -- routing curves (VLM, test split)
    routing_test_cnn.png          -- routing curves (CNN, test split)
    routing_test_comparison.png   -- all models calibrated policy on test

Usage:
    python scripts/analyze_cross_center.py [--budget 0.15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration import TemperatureScaler, reliability_diagram_data, ece_score
from src.triage.router import risk_coverage_curve, random_routing_curve, operating_point

OUT_DIR = PROJECT_ROOT / "results" / "stage6"
CAL_DIR = PROJECT_ROOT / "results" / "calibration"

# Prediction parquets
PATHS = {
    "medgemma_val":   PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3" / "predictions.parquet",
    "medgemma_test":  PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3_test" / "predictions.parquet",
    "gemma3_val":     PROJECT_ROOT / "results" / "zeroshot" / "gemma-3-27b-it" / "V3" / "predictions.parquet",
    "gemma3_test":    PROJECT_ROOT / "results" / "zeroshot" / "gemma-3-27b-it" / "V3_test" / "predictions.parquet",
    "cnn_val":        PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "val_predictions.parquet",
    "cnn_test":       PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "test_predictions.parquet",
    "cal_medgemma":   CAL_DIR / "medgemma-27b-it" / "V3" / "calibration_params.json",
    "cal_cnn":        CAL_DIR / "resnet18_64px" / "calibration_params.json",
}

N_LOGIT_COLS = 9


def _load_vlm_split(path: Path, T: float) -> dict:
    df = pd.read_parquet(path)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    raw_conf = df["pred_confidence"].to_numpy(float)
    scaler = TemperatureScaler()
    scaler.T = T
    cal_conf = scaler.transform_scalar(raw_conf)
    acc = correct.mean()
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(
        df["true_label_name"].tolist(), df["pred_label"].tolist(),
        average="macro", zero_division=0,
    )
    return {"correct": correct, "raw_conf": raw_conf, "cal_conf": cal_conf,
            "acc": float(acc), "macro_f1": float(macro_f1)}


def _load_cnn_split(path: Path, T: float) -> dict:
    df = pd.read_parquet(path)
    correct = df["correct"].to_numpy(bool)
    raw_conf = df["confidence"].to_numpy(float)
    logit_cols = [f"logit_{c}" for c in range(N_LOGIT_COLS)]
    logits = df[logit_cols].to_numpy(float)
    scaler = TemperatureScaler()
    scaler.T = T
    cal_probs = scaler.transform_logits(logits)
    cal_conf = cal_probs.max(axis=1)
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(
        df["true_label_idx"].tolist(), df["pred_label_idx"].tolist(),
        average="macro", zero_division=0,
    )
    return {"correct": correct, "raw_conf": raw_conf, "cal_conf": cal_conf,
            "acc": float(correct.mean()), "macro_f1": float(macro_f1)}


def _calibration_metrics(conf: np.ndarray, correct: np.ndarray, label: str) -> dict:
    data = reliability_diagram_data(conf, correct)
    return {"label": label, "ece": round(data["ece"], 4), "mce": round(data["mce"], 4)}


def _routing_curves(d: dict, seed: int) -> dict:
    return {
        "calibrated": risk_coverage_curve(d["cal_conf"], d["correct"]),
        "raw":        risk_coverage_curve(d["raw_conf"], d["correct"]),
        "random":     random_routing_curve(d["correct"], seed=seed),
    }


def _plot_curves(curves: dict, title: str, out_path: Path, budget: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"calibrated": ("#D65F5F", "-"), "raw": ("#4878CF", "-"), "random": ("#888888", "--")}
    for key, (color, ls) in styles.items():
        c = curves[key]
        b = np.array(c["budgets"])
        a = np.array(c["auto_confirm_acc"], dtype=float)
        v = ~np.isnan(a)
        ax.plot(b[v], a[v], ls, color=color, lw=1.8,
                label=f"{key} (AUC={c['auc']:.3f})")
    ax.axvline(budget, color="black", lw=1, ls=":", alpha=0.6,
               label=f"{int(budget*100)}% budget")
    ax.set_xlabel("Fraction routed")
    ax.set_ylabel("Auto-confirm accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(model_curves: dict, out_path: Path, budget: float) -> None:
    """Calibrated policy for all models on test split."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"MedGemma V3": "#D65F5F", "Gemma 3 V3": "#E88F2A", "ResNet-18 64px": "#4878CF"}
    for name, curves in model_curves.items():
        c = curves["calibrated"]
        b = np.array(c["budgets"])
        a = np.array(c["auto_confirm_acc"], dtype=float)
        v = ~np.isnan(a)
        ax.plot(b[v], a[v], "-", color=colors.get(name, "#888888"), lw=2,
                label=f"{name} (AUC={c['auc']:.3f})")
    ax.axvline(budget, color="black", lw=1, ls=":", alpha=0.6,
               label=f"{int(budget*100)}% budget")
    ax.set_xlabel("Fraction routed")
    ax.set_ylabel("Auto-confirm accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Risk-coverage (calibrated, test split)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6: cross-center analysis")
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    missing = [k for k, p in PATHS.items() if not p.exists()]
    if missing:
        print("Missing files:", missing)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PATHS["cal_medgemma"]) as f:
        cal_mg  = json.load(f)
    with open(PATHS["cal_cnn"]) as f:
        cal_cnn = json.load(f)
    T_mg = cal_mg["T"]
    T_cnn = cal_cnn["T"]

    # ── Load all splits ──
    mg_val  = _load_vlm_split(PATHS["medgemma_val"],  T_mg)
    mg_test = _load_vlm_split(PATHS["medgemma_test"], T_mg)
    g3_val  = _load_vlm_split(PATHS["gemma3_val"],    T_mg)   # no separate Gemma3 cal; use same T
    g3_test = _load_vlm_split(PATHS["gemma3_test"],   T_mg)
    cn_val  = _load_cnn_split(PATHS["cnn_val"],  T_cnn)
    cn_test = _load_cnn_split(PATHS["cnn_test"], T_cnn)

    # ── Cross-center table ──
    table = {
        "MedGemma V3":    {"val_acc": mg_val["acc"],  "test_acc": mg_test["acc"],
                           "val_f1":  mg_val["macro_f1"], "test_f1": mg_test["macro_f1"]},
        "Gemma 3 27B V3": {"val_acc": g3_val["acc"],  "test_acc": g3_test["acc"],
                           "val_f1":  g3_val["macro_f1"], "test_f1": g3_test["macro_f1"]},
        "ResNet-18 64px": {"val_acc": cn_val["acc"],  "test_acc": cn_test["acc"],
                           "val_f1":  cn_val["macro_f1"], "test_f1": cn_test["macro_f1"]},
    }
    for name, row in table.items():
        row["delta_acc"] = round(row["test_acc"] - row["val_acc"], 4)
        row["delta_f1"]  = round(row["test_f1"]  - row["val_f1"],  4)

    print("\n=== Cross-center classification table ===")
    hdr = f"{'Model':<22} {'Val acc':>8} {'Test acc':>9} {'d acc':>7} {'Val F1':>7} {'Test F1':>8} {'d F1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, row in table.items():
        print(f"{name:<22} {row['val_acc']:>8.4f} {row['test_acc']:>9.4f} "
              f"{row['delta_acc']:>+7.4f} {row['val_f1']:>7.4f} {row['test_f1']:>8.4f} {row['delta_f1']:>+7.4f}")

    with open(OUT_DIR / "cross_center_table.json", "w") as f:
        json.dump(table, f, indent=2)

    # Markdown table for paper
    md = ["| Model | Val acc | Test acc | d acc | Val F1 | Test F1 | d F1 |",
          "|---|---|---|---|---|---|---|"]
    for name, row in table.items():
        md.append(f"| {name} | {row['val_acc']:.4f} | {row['test_acc']:.4f} | "
                  f"{row['delta_acc']:+.4f} | {row['val_f1']:.4f} | {row['test_f1']:.4f} | "
                  f"{row['delta_f1']:+.4f} |")
    (OUT_DIR / "cross_center_table.md").write_text("\n".join(md))

    # ── Calibration transfer ──
    cal_transfer = {}
    for name, raw_c, cal_c, correct in [
        ("MedGemma V3 (val)",   mg_val["raw_conf"],  mg_val["cal_conf"],  mg_val["correct"]),
        ("MedGemma V3 (test)",  mg_test["raw_conf"], mg_test["cal_conf"], mg_test["correct"]),
        ("Gemma 3 V3 (val)",    g3_val["raw_conf"],  g3_val["cal_conf"],  g3_val["correct"]),
        ("Gemma 3 V3 (test)",   g3_test["raw_conf"], g3_test["cal_conf"], g3_test["correct"]),
        ("ResNet-18 64px (val)",  cn_val["raw_conf"],  cn_val["cal_conf"],  cn_val["correct"]),
        ("ResNet-18 64px (test)", cn_test["raw_conf"], cn_test["cal_conf"], cn_test["correct"]),
    ]:
        cal_transfer[name] = {
            "ece_before": round(ece_score(raw_c, correct), 4),
            "ece_after":  round(ece_score(cal_c, correct), 4),
        }

    print("\n=== Calibration transfer (val-fitted T applied to test) ===")
    for name, m in cal_transfer.items():
        print(f"  {name:<30} ECE before={m['ece_before']:.4f}  ECE after={m['ece_after']:.4f}")
    with open(OUT_DIR / "calibration_transfer.json", "w") as f:
        json.dump({"T_medgemma": T_mg, "T_cnn": T_cnn, "splits": cal_transfer}, f, indent=2)

    # ── Routing on test ──
    print("\n=== Routing on test split ===")
    mg_curves  = _routing_curves(mg_test,  args.seed)
    g3_curves  = _routing_curves(g3_test,  args.seed)
    cnn_curves = _routing_curves(cn_test,  args.seed)

    for label, d, curves in [
        ("MedGemma V3 test",    mg_test,  mg_curves),
        ("Gemma 3 V3 test",     g3_test,  g3_curves),
        ("ResNet-18 64px test", cn_test,  cnn_curves),
    ]:
        op_cal = operating_point(d["cal_conf"], d["correct"], args.budget)
        op_rnd = operating_point(np.full(len(d["correct"]), 0.5), d["correct"], args.budget)
        print(f"  {label}: calibrated acc={op_cal['auto_confirm_acc']:.4f}  "
              f"random acc={op_rnd['auto_confirm_acc']:.4f}  "
              f"AUC cal={curves['calibrated']['auc']:.3f}")

    _plot_curves(mg_curves,  "MedGemma V3 routing (test split)",    OUT_DIR / "routing_test_vlm.png",    args.budget)
    _plot_curves(cnn_curves, "ResNet-18 64px routing (test split)", OUT_DIR / "routing_test_cnn.png",    args.budget)
    _plot_comparison(
        {"MedGemma V3": mg_curves, "Gemma 3 V3": g3_curves, "ResNet-18 64px": cnn_curves},
        OUT_DIR / "routing_test_comparison.png", args.budget,
    )

    print(f"\nAll Stage 6 results: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
