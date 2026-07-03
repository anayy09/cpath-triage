"""
scripts/calibrate.py

Stage 4: measure and improve calibration for VLM and CNN models.

Calibration split strategy:
  VLM: uses the 450-sample balanced val subset already in predictions.parquet.
       Split 80/20 (360 cal / 90 eval) within that file. Small but sufficient
       for a single-parameter fit; limitation is noted in the paper.
  CNN: uses the full val predictions from eval_cnn.py (10,004 samples).
       Split 80/20 (8003 cal / 2001 eval). No test data is touched.

Outputs (under results/calibration/{model_key}/):
    calibration_params.json   -- fitted T, ECE/MCE before and after
    reliability_before.png    -- reliability diagram, raw confidence
    reliability_after.png     -- reliability diagram, calibrated confidence

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --vlm-model gemma-3-27b-it --prompt-version V3
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

from src.eval.calibration import (
    brier_binary, brier_multiclass,
    reliability_diagram_data, TemperatureScaler,
)
from src.data.pathmnist import LABEL_NAMES

N_CLASSES = 9
ALL_LABELS = [LABEL_NAMES[i] for i in range(N_CLASSES)]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reliability(
    data: dict,
    title: str,
    out_path: Path,
    color: str = "#4878CF",
) -> None:
    """Single reliability diagram with gap bars and identity line."""
    centers = np.array(data["bin_centers"])
    accs = np.array(data["bin_accuracies"], dtype=float)
    confs = np.array(data["bin_confidences"], dtype=float)
    counts = np.array(data["bin_counts"])

    valid = ~np.isnan(accs)
    width = centers[1] - centers[0] if len(centers) > 1 else 0.067

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Gap bars (overconfidence shown in red, underconfidence in blue-ish)
    for i in np.where(valid)[0]:
        gap = confs[i] - accs[i]
        gap_color = "#D65F5F" if gap > 0 else "#6ACC65"
        ax.bar(centers[i], abs(gap), width=width * 0.9,
               bottom=min(accs[i], confs[i]),
               color=gap_color, alpha=0.35, zorder=1, label=None)

    # Accuracy bars
    ax.bar(centers[valid], accs[valid], width=width * 0.9,
           color=color, alpha=0.75, zorder=2, label="Accuracy per bin")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

    ece = data["ece"]
    mce = data["mce"]
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE={ece:.3f}  MCE={mce:.3f}", fontsize=10)
    ax.legend(fontsize=9)

    # Annotate bin counts for the largest bins
    for i in np.where(valid)[0]:
        if counts[i] > 0:
            ax.text(centers[i], min(accs[i], 0.05), f"{counts[i]}",
                    ha="center", va="bottom", fontsize=6, color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_combined_reliability(
    before: dict, after: dict, title: str, out_path: Path
) -> None:
    """Side-by-side before/after reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, label in zip(axes, [before, after], ["Before calibration", "After calibration"]):
        centers = np.array(data["bin_centers"])
        accs = np.array(data["bin_accuracies"], dtype=float)
        valid = ~np.isnan(accs)
        width = centers[1] - centers[0] if len(centers) > 1 else 0.067
        ax.bar(centers[valid], accs[valid], width=width * 0.9,
               color="#4878CF", alpha=0.75, label="Accuracy per bin")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nECE={data['ece']:.3f}  MCE={data['mce']:.3f}")
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── VLM calibration ───────────────────────────────────────────────────────────

def calibrate_vlm(
    model_key: str,
    pred_path: Path,
    out_dir: Path,
    cal_fraction: float = 0.8,
    seed: int = 42,
    test_pred_path: Path | None = None,
) -> dict:
    """
    Calibrate a VLM model from its predictions.parquet.

    The calibration split is an 80/20 split of pred_path. Pass the 450-sample
    pilot predictions.parquet for the small pilot fit, or the {version}_full
    predictions.parquet for the full-scale fit used in the paper's Table 2.

    If test_pred_path is given, also computes cross-center transfer: applies
    the val-fitted T to the full test set without re-fitting (paper Table 5).
    No API calls are made.

    Returns dict of all metrics for PROGRESS.md.
    """
    df = pd.read_parquet(pred_path)
    # Exclude "unknown" parse failures from calibration (there should be none)
    df = df[df["pred_label"] != "unknown"].copy()
    df["correct"] = df["pred_label"] == df["true_label_name"]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cal_n = int(len(df) * cal_fraction)
    cal_idx, eval_idx = idx[:cal_n], idx[cal_n:]

    cal_df = df.iloc[cal_idx]
    ev_df  = df.iloc[eval_idx]

    cal_conf = cal_df["pred_confidence"].to_numpy(dtype=float)
    cal_corr = cal_df["correct"].to_numpy(bool)
    ev_conf  = ev_df["pred_confidence"].to_numpy(dtype=float)
    ev_corr  = ev_df["correct"].to_numpy(bool)

    # ECE before calibration (on eval split)
    before = reliability_diagram_data(ev_conf, ev_corr)

    # Fit temperature T on calibration split
    scaler = TemperatureScaler()
    T = scaler.fit_scalar(cal_conf, cal_corr)

    # ECE after calibration (on eval split)
    ev_conf_cal = scaler.transform_scalar(ev_conf)
    after = reliability_diagram_data(ev_conf_cal, ev_corr)

    print(f"\n{model_key}")
    print(f"  N cal/eval:    {cal_n} / {len(ev_df)}")
    print(f"  Temperature T: {T:.4f}")
    print(f"  ECE before:    {before['ece']:.4f}")
    print(f"  ECE after:     {after['ece']:.4f}")
    print(f"  MCE before:    {before['mce']:.4f}")
    print(f"  MCE after:     {after['mce']:.4f}")

    # Brier score (binary P(correct)) on eval split
    brier_before = brier_binary(ev_conf, ev_corr)
    brier_after  = brier_binary(ev_conf_cal, ev_corr)
    print(f"  Brier before:  {brier_before:.4f}")
    print(f"  Brier after:   {brier_after:.4f}")

    # Plots
    plot_combined_reliability(before, after, f"{model_key} calibration", out_dir / "reliability_combined.png")
    plot_reliability(before, f"{model_key} (raw)", out_dir / "reliability_before.png")
    plot_reliability(after, f"{model_key} (T={T:.2f})", out_dir / "reliability_after.png", color="#D65F5F")

    params = {
        "model_key": model_key,
        "calibration_type": "scalar_temperature",
        "T": round(T, 4),
        "n_cal": int(cal_n),
        "n_eval": int(len(ev_df)),
        "ece_before": round(before["ece"], 4),
        "ece_after":  round(after["ece"],  4),
        "mce_before": round(before["mce"], 4),
        "mce_after":  round(after["mce"],  4),
        "brier_before": round(brier_before, 4),
        "brier_after":  round(brier_after,  4),
    }

    if test_pred_path is not None:
        test_df = pd.read_parquet(test_pred_path)
        test_df = test_df[test_df["pred_label"] != "unknown"].copy()
        test_df["correct"] = test_df["pred_label"] == test_df["true_label_name"]

        test_conf = test_df["pred_confidence"].to_numpy(dtype=float)
        test_corr = test_df["correct"].to_numpy(bool)
        test_ece_raw = reliability_diagram_data(test_conf, test_corr)["ece"]
        test_conf_cal = scaler.transform_scalar(test_conf)
        test_ece_cal = reliability_diagram_data(test_conf_cal, test_corr)["ece"]

        params["transfer"] = {
            "n_test": int(len(test_df)),
            "val_ece_cal": round(after["ece"], 4),
            "test_ece_raw": round(float(test_ece_raw), 4),
            "test_ece_cal": round(float(test_ece_cal), 4),
            "change": round(float(test_ece_cal) - after["ece"], 4),
        }
        print(f"  Transfer: val_ece_cal={after['ece']:.4f}  "
              f"test_ece_raw={test_ece_raw:.4f}  test_ece_cal={test_ece_cal:.4f}")

    with open(out_dir / "calibration_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_dir}")
    return params


# ── CNN calibration ───────────────────────────────────────────────────────────

def calibrate_cnn(
    pred_path: Path,
    out_dir: Path,
    cal_fraction: float = 0.8,
    seed: int = 42,
    model_key: str = "resnet18_64px",
    test_pred_path: Path | None = None,
) -> dict:
    """
    Calibrate a CNN using its full val predictions from eval_cnn.py / eval_cnn_224px.py.

    Uses logit columns (logit_0..logit_8) for temperature scaling and
    max-softmax confidence for reliability diagrams. If test_pred_path is
    given, also computes cross-center transfer (paper Table 5) by applying
    the val-fitted T to the test logits without re-fitting.
    """
    df = pd.read_parquet(pred_path)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cal_n = int(len(df) * cal_fraction)
    cal_idx, eval_idx = idx[:cal_n], idx[cal_n:]

    logit_cols = [f"logit_{c}" for c in range(N_CLASSES)]
    prob_cols  = [f"prob_{c}"  for c in range(N_CLASSES)]

    cal_df = df.iloc[cal_idx]
    ev_df  = df.iloc[eval_idx]

    cal_logits = cal_df[logit_cols].to_numpy(dtype=float)
    cal_labels = cal_df["true_label_idx"].to_numpy(int)
    ev_logits  = ev_df[logit_cols].to_numpy(dtype=float)
    ev_labels  = ev_df["true_label_idx"].to_numpy(int)
    ev_probs   = ev_df[prob_cols].to_numpy(dtype=float)
    ev_conf    = ev_probs.max(axis=1)
    ev_corr    = (ev_df["pred_label_idx"].to_numpy(int) == ev_labels)

    before = reliability_diagram_data(ev_conf, ev_corr)

    scaler = TemperatureScaler()
    T = scaler.fit_logits(cal_logits, cal_labels)

    ev_probs_cal = scaler.transform_logits(ev_logits)
    ev_conf_cal  = ev_probs_cal.max(axis=1)
    ev_corr_cal  = (ev_probs_cal.argmax(axis=1) == ev_labels)

    after = reliability_diagram_data(ev_conf_cal, ev_corr_cal)

    brier_before = brier_multiclass(ev_probs, ev_labels)
    brier_after  = brier_multiclass(ev_probs_cal, ev_labels)

    print(f"\n{model_key} (CNN)")
    print(f"  N cal/eval:    {cal_n} / {len(ev_df)}")
    print(f"  Temperature T: {T:.4f}")
    print(f"  ECE before:    {before['ece']:.4f}")
    print(f"  ECE after:     {after['ece']:.4f}")
    print(f"  MCE before:    {before['mce']:.4f}")
    print(f"  MCE after:     {after['mce']:.4f}")
    print(f"  Brier before:  {brier_before:.4f}")
    print(f"  Brier after:   {brier_after:.4f}")

    plot_combined_reliability(before, after, f"{model_key} calibration", out_dir / "reliability_combined.png")
    plot_reliability(before, f"{model_key} (raw softmax)", out_dir / "reliability_before.png")
    plot_reliability(after, f"{model_key} (T={T:.2f})", out_dir / "reliability_after.png", color="#D65F5F")

    params = {
        "model_key": model_key,
        "calibration_type": "logit_temperature",
        "T": round(T, 4),
        "n_cal": int(cal_n),
        "n_eval": int(len(ev_df)),
        "ece_before": round(before["ece"], 4),
        "ece_after":  round(after["ece"],  4),
        "mce_before": round(before["mce"], 4),
        "mce_after":  round(after["mce"],  4),
        "brier_before": round(brier_before, 4),
        "brier_after":  round(brier_after,  4),
    }

    if test_pred_path is not None:
        test_df = pd.read_parquet(test_pred_path)
        test_logits = test_df[logit_cols].to_numpy(dtype=float)
        test_labels = test_df["true_label_idx"].to_numpy(int)
        test_probs_raw = test_df[prob_cols].to_numpy(dtype=float)
        test_conf_raw = test_probs_raw.max(axis=1)
        test_corr_raw = (test_df["pred_label_idx"].to_numpy(int) == test_labels)
        test_ece_raw = reliability_diagram_data(test_conf_raw, test_corr_raw)["ece"]

        test_probs_cal = scaler.transform_logits(test_logits)
        test_conf_cal = test_probs_cal.max(axis=1)
        test_corr_cal = (test_probs_cal.argmax(axis=1) == test_labels)
        test_ece_cal = reliability_diagram_data(test_conf_cal, test_corr_cal)["ece"]

        params["transfer"] = {
            "n_test": int(len(test_df)),
            "val_ece_cal": round(after["ece"], 4),
            "test_ece_raw": round(float(test_ece_raw), 4),
            "test_ece_cal": round(float(test_ece_cal), 4),
            "change": round(float(test_ece_cal) - after["ece"], 4),
        }
        print(f"  Transfer: val_ece_cal={after['ece']:.4f}  "
              f"test_ece_raw={test_ece_raw:.4f}  test_ece_cal={test_ece_cal:.4f}")

    with open(out_dir / "calibration_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_dir}")
    return params


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4: calibration analysis")
    parser.add_argument("--vlm-model",      default="medgemma-27b-it")
    parser.add_argument("--prompt-version", default="V3")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument(
        "--full-scale", action="store_true",
        help=(
            "Fit calibration on the full-scale {version}_full val predictions "
            "(MedGemma and Gemma-3, plus CNN-64) instead of the 450-sample pilot, "
            "and compute cross-center transfer to the full test set. This is the "
            "fit used for paper Tables 2 and 5. Output goes to results/calibration/"
            "{model}/{version}_full/."
        ),
    )
    args = parser.parse_args()

    cal_root = PROJECT_ROOT / "results" / "calibration"

    if args.full_scale:
        zeroshot_root = PROJECT_ROOT / "results" / "zeroshot"
        full_scale_models = [
            ("medgemma-27b-it", "medgemma-27b-it"),
            ("gemma-3-27b-it", "gemma-3-27b-it"),
        ]
        for model_name, model_slug in full_scale_models:
            val_path = zeroshot_root / model_slug / f"{args.prompt_version}_full" / "predictions.parquet"
            test_path = zeroshot_root / model_slug / f"{args.prompt_version}_full_test" / "predictions.parquet"
            if not val_path.exists():
                print(f"Skipping {model_name}: {val_path} not found.")
                continue
            calibrate_vlm(
                model_key=f"{model_name}_{args.prompt_version}_full",
                pred_path=val_path,
                out_dir=cal_root / model_slug / f"{args.prompt_version}_full",
                seed=args.seed,
                test_pred_path=test_path if test_path.exists() else None,
            )

        cnn64_val = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "val_predictions.parquet"
        cnn64_test = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "test_predictions.parquet"
        if cnn64_val.exists():
            calibrate_cnn(
                pred_path=cnn64_val,
                out_dir=cal_root / "resnet18_64px",
                seed=args.seed,
                model_key="resnet18_64px",
                test_pred_path=cnn64_test if cnn64_test.exists() else None,
            )

        cnn224_val = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px" / "val_predictions.parquet"
        cnn224_test = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px" / "test_predictions.parquet"
        if cnn224_val.exists():
            calibrate_cnn(
                pred_path=cnn224_val,
                out_dir=cal_root / "resnet18_224px",
                seed=args.seed,
                model_key="resnet18_224px",
                test_pred_path=cnn224_test if cnn224_test.exists() else None,
            )
        else:
            print(f"\nSkipping CNN-224 calibration: {cnn224_val} not found.")
            print("Run scripts/eval_cnn_224px.py first to enable it.")

        print("\nFull-scale calibration complete.")
        print(f"Results under: {cal_root}")
        return 0

    model_slug = args.vlm_model.replace("/", "_").replace(":", "_")
    vlm_pred_path = (
        PROJECT_ROOT / "results" / "zeroshot" / model_slug
        / args.prompt_version / "predictions.parquet"
    )
    cnn_pred_path = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "val_predictions.parquet"

    # Check prerequisites
    if not vlm_pred_path.exists():
        print(f"VLM predictions not found: {vlm_pred_path}")
        print("Run scripts/run_zeroshot.py first.")
        return 1
    if not cnn_pred_path.exists():
        print(f"CNN val predictions not found: {cnn_pred_path}")
        print("Run scripts/eval_cnn.py first.")
        return 1

    # Calibrate VLM (450-sample pilot fit)
    vlm_key = f"{args.vlm_model}_{args.prompt_version}"
    calibrate_vlm(
        model_key=vlm_key,
        pred_path=vlm_pred_path,
        out_dir=cal_root / model_slug / args.prompt_version,
        seed=args.seed,
    )

    # Calibrate CNN-64
    calibrate_cnn(
        pred_path=cnn_pred_path,
        out_dir=cal_root / "resnet18_64px",
        seed=args.seed,
        model_key="resnet18_64px",
    )

    # Calibrate CNN-224 if its predictions are available (optional)
    cnn224_pred_path = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px" / "val_predictions.parquet"
    if cnn224_pred_path.exists():
        calibrate_cnn(
            pred_path=cnn224_pred_path,
            out_dir=cal_root / "resnet18_224px",
            seed=args.seed,
            model_key="resnet18_224px",
        )
    else:
        print(f"\nSkipping CNN-224 calibration: {cnn224_pred_path} not found.")
        print("Run scripts/eval_cnn_224px.py first to enable it.")

    print("\nStage 4 calibration complete.")
    print(f"Results under: {cal_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
