"""
scripts/run_routing.py

Stage 5: evaluate the uncertainty-aware routing policy.

Loads predictions from Stage 2 (VLM) and Stage 3 (CNN), applies calibrated
temperature scaling from Stage 4, and computes risk-coverage curves for three
routing policies:
  - calibrated: route by calibrated uncertainty (1 - temp-scaled confidence)
  - raw:        route by raw uncertainty (1 - original confidence)
  - random:     route a random fraction (averaged over 30 trials)

Headline operating point: budget = 0.15 (route 15% to specialist, auto-confirm 85%).

Outputs (under results/routing/):
    risk_coverage_curves.json     -- all curve data
    risk_coverage_{model}.png     -- per-model figure (3 policies)
    risk_coverage_comparison.png  -- all models + calibrated policy on one plot
    operating_points.json         -- metrics at 15% budget for all models

Usage:
    python scripts/run_routing.py [--budget 0.15] [--seed 42]
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

from src.triage.router import risk_coverage_curve, random_routing_curve, operating_point
from src.eval.calibration import TemperatureScaler

OUT_DIR = PROJECT_ROOT / "results" / "routing"
CAL_DIR = PROJECT_ROOT / "results" / "calibration"

# Paths to prediction parquets
VLM_PRED = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3" / "predictions.parquet"
CNN_PRED  = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "val_predictions.parquet"
CAL_VLM   = CAL_DIR / "medgemma-27b-it" / "V3" / "calibration_params.json"
CAL_CNN   = CAL_DIR / "resnet18_64px" / "calibration_params.json"

N_LOGIT_COLS = 9


def _load_vlm(pred_path: Path, cal_path: Path) -> dict[str, np.ndarray]:
    """Load VLM predictions and apply scalar temperature scaling."""
    df = pd.read_parquet(pred_path)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    raw_conf = df["pred_confidence"].to_numpy(float)

    with open(cal_path) as f:
        params = json.load(f)
    T = params["T"]
    scaler = TemperatureScaler()
    scaler.T = T
    cal_conf = scaler.transform_scalar(raw_conf)

    return {"correct": correct, "raw_conf": raw_conf, "cal_conf": cal_conf, "T": T}


def _load_cnn(pred_path: Path, cal_path: Path) -> dict[str, np.ndarray]:
    """Load CNN predictions and apply logit temperature scaling."""
    df = pd.read_parquet(pred_path)
    correct = df["correct"].to_numpy(bool)
    raw_conf = df["confidence"].to_numpy(float)   # max softmax

    logit_cols = [f"logit_{c}" for c in range(N_LOGIT_COLS)]
    logits = df[logit_cols].to_numpy(float)

    with open(cal_path) as f:
        params = json.load(f)
    T = params["T"]
    scaler = TemperatureScaler()
    scaler.T = T
    cal_probs = scaler.transform_logits(logits)
    cal_conf = cal_probs.max(axis=1)

    return {"correct": correct, "raw_conf": raw_conf, "cal_conf": cal_conf, "T": T}


def _plot_single(
    curves: dict[str, dict],
    title: str,
    out_path: Path,
    headline_budget: float,
) -> None:
    """Plot three routing policies for one model."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"calibrated": "#D65F5F", "raw": "#4878CF", "random": "#888888"}
    labels = {
        "calibrated": f"Calibrated (T={curves['T']:.1f})",
        "raw": "Raw confidence",
        "random": "Random (30-trial avg)",
    }

    for key in ("random", "raw", "calibrated"):
        c = curves[key]
        budgets = np.array(c["budgets"])
        accs = np.array(c["auto_confirm_acc"], dtype=float)
        valid = ~np.isnan(accs)
        ls = "--" if key == "random" else "-"
        ax.plot(budgets[valid], accs[valid], ls, color=colors[key],
                lw=1.8, label=f"{labels[key]} (AUC={c['auc']:.3f})")

        if key == "random" and "std_acc" in c:
            std = np.array(c["std_acc"])
            ax.fill_between(budgets[valid], (accs - std)[valid], (accs + std)[valid],
                            color=colors[key], alpha=0.15)

    # Vertical line at headline operating point
    ax.axvline(headline_budget, color="black", lw=1, ls=":", alpha=0.7,
               label=f"Operating point ({int(headline_budget*100)}% routed)")

    ax.set_xlabel("Fraction routed to specialist", fontsize=11)
    ax.set_ylabel("Accuracy on auto-confirmed set", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(
    all_curves: dict[str, dict],
    out_path: Path,
    headline_budget: float,
) -> None:
    """Calibrated-policy curves for all models on one plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {
        "MedGemma V3":    "#D65F5F",
        "Gemma 3 27B V3": "#E88F2A",
        "ResNet-18 64px": "#4878CF",
    }

    for model_name, curves in all_curves.items():
        c = curves["calibrated"]
        budgets = np.array(c["budgets"])
        accs = np.array(c["auto_confirm_acc"], dtype=float)
        valid = ~np.isnan(accs)
        color = color_map.get(model_name, "#888888")
        ax.plot(budgets[valid], accs[valid], "-", color=color, lw=2.0,
                label=f"{model_name} (AUC={c['auc']:.3f})")

        # Random baseline (use first model's random curve, same correct distribution)
        if "random" in curves:
            r = curves["random"]
            rb = np.array(r["budgets"])
            ra = np.array(r["auto_confirm_acc"], dtype=float)
            rv = ~np.isnan(ra)

    # Plot random baseline once (VLM random, since it's the hardest task)
    first_name = list(all_curves.keys())[0]
    r = all_curves[first_name]["random"]
    rb = np.array(r["budgets"])
    ra = np.array(r["auto_confirm_acc"], dtype=float)
    rv = ~np.isnan(ra)
    ax.plot(rb[rv], ra[rv], "--", color="#888888", lw=1.2, alpha=0.7,
            label=f"Random ({first_name})")

    ax.axvline(headline_budget, color="black", lw=1, ls=":", alpha=0.6,
               label=f"{int(headline_budget*100)}% routing budget")

    ax.set_xlabel("Fraction routed to specialist", fontsize=11)
    ax.set_ylabel("Accuracy on auto-confirmed set", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_title("Risk-coverage curves (calibrated policy)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _compute_curves(correct: np.ndarray, raw_conf: np.ndarray, cal_conf: np.ndarray,
                    T: float, seed: int) -> dict:
    return {
        "calibrated": risk_coverage_curve(cal_conf, correct),
        "raw":        risk_coverage_curve(raw_conf, correct),
        "random":     random_routing_curve(correct, seed=seed),
        "T": T,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5: routing policy evaluation")
    parser.add_argument("--budget", type=float, default=0.15,
                        help="Headline routing budget (fraction to specialist).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    missing = [p for p in [VLM_PRED, CNN_PRED, CAL_VLM, CAL_CNN] if not p.exists()]
    if missing:
        print("Missing files (run prior stages first):")
        for p in missing:
            print(f"  {p}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──
    print("Loading predictions...")
    vlm = _load_vlm(VLM_PRED, CAL_VLM)
    cnn = _load_cnn(CNN_PRED, CAL_CNN)

    print(f"  VLM: {len(vlm['correct'])} samples, accuracy={vlm['correct'].mean():.4f}, T={vlm['T']:.1f}")
    print(f"  CNN: {len(cnn['correct'])} samples, accuracy={cnn['correct'].mean():.4f}, T={cnn['T']:.4f}")

    # ── Compute curves ──
    print("\nComputing risk-coverage curves...")
    vlm_curves = _compute_curves(vlm["correct"], vlm["raw_conf"], vlm["cal_conf"], vlm["T"], args.seed)
    cnn_curves = _compute_curves(cnn["correct"], cnn["raw_conf"], cnn["cal_conf"], cnn["T"], args.seed)

    # ── Operating points ──
    budget = args.budget
    op_vlm_cal = operating_point(vlm["cal_conf"], vlm["correct"], budget)
    op_vlm_raw = operating_point(vlm["raw_conf"], vlm["correct"], budget)
    op_vlm_rnd = operating_point(np.full(len(vlm["correct"]), 0.5), vlm["correct"], budget)
    op_cnn_cal = operating_point(cnn["cal_conf"], cnn["correct"], budget)
    op_cnn_raw = operating_point(cnn["raw_conf"], cnn["correct"], budget)
    op_cnn_rnd = operating_point(np.full(len(cnn["correct"]), 0.5), cnn["correct"], budget)

    print(f"\nOperating point (budget={budget:.0%}):")
    print(f"  VLM calibrated:  auto-confirm acc={op_vlm_cal['auto_confirm_acc']:.4f}  "
          f"error={op_vlm_cal['auto_confirm_error']:.4f}")
    print(f"  VLM raw:         auto-confirm acc={op_vlm_raw['auto_confirm_acc']:.4f}  "
          f"error={op_vlm_raw['auto_confirm_error']:.4f}")
    print(f"  VLM random:      auto-confirm acc={op_vlm_rnd['auto_confirm_acc']:.4f}  "
          f"error={op_vlm_rnd['auto_confirm_error']:.4f}")
    print(f"  CNN calibrated:  auto-confirm acc={op_cnn_cal['auto_confirm_acc']:.4f}  "
          f"error={op_cnn_cal['auto_confirm_error']:.4f}")
    print(f"  CNN raw:         auto-confirm acc={op_cnn_raw['auto_confirm_acc']:.4f}  "
          f"error={op_cnn_raw['auto_confirm_error']:.4f}")
    print(f"  CNN random:      auto-confirm acc={op_cnn_rnd['auto_confirm_acc']:.4f}  "
          f"error={op_cnn_rnd['auto_confirm_error']:.4f}")

    # ── Save JSON ──
    all_data = {
        "vlm_medgemma_v3": vlm_curves,
        "cnn_resnet18_64px": cnn_curves,
    }
    curves_path = OUT_DIR / "risk_coverage_curves.json"
    with open(curves_path, "w") as f:
        json.dump(all_data, f, indent=2)

    ops_path = OUT_DIR / "operating_points.json"
    with open(ops_path, "w") as f:
        json.dump({
            "headline_budget": budget,
            "vlm_medgemma_v3": {
                "calibrated": op_vlm_cal, "raw": op_vlm_raw, "random": op_vlm_rnd,
            },
            "cnn_resnet18_64px": {
                "calibrated": op_cnn_cal, "raw": op_cnn_raw, "random": op_cnn_rnd,
            },
        }, f, indent=2)

    # ── Plots ──
    _plot_single(vlm_curves, "MedGemma-27b-it V3 routing policy",
                 OUT_DIR / "risk_coverage_vlm.png", budget)
    _plot_single(cnn_curves, "ResNet-18 64px routing policy",
                 OUT_DIR / "risk_coverage_cnn.png", budget)

    _plot_comparison(
        {
            "MedGemma V3":    vlm_curves,
            "ResNet-18 64px": cnn_curves,
        },
        OUT_DIR / "risk_coverage_comparison.png",
        budget,
    )

    print(f"\nResults: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
