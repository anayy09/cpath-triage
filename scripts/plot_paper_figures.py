"""
scripts/plot_paper_figures.py

Generates two figures for the manuscript that were computed as part of the
analysis (Tables 3 and 4) but never saved as standalone plots:

  1. Full-scale risk-coverage (routing) comparison, val + test panels, for
     MedGemma-27b-it, Gemma-3-27b-it, and ResNet-18 64px. Reproduces the
     AUC/operating-point numbers in Table 3 exactly.
  2. Cross-center accuracy bar chart (val vs. test) for all four models,
     visualizing the resolution paradox in Table 4 (CNN-224 degrades more
     than CNN-64 despite matching VLM input resolution).

No new experiments are run. Both figures are built entirely from existing,
already-verified results files:
  results/zeroshot/{model}/V3_full{_test}/predictions.parquet
  results/calibration/{model}/V3_full/calibration_params.json
  results/cnn/resnet18_64px/{val,test}_predictions.parquet
  results/zeroshot/*/metrics.json, results/cnn/*/metrics.json

Outputs:
    results/routing/risk_coverage_fullscale_comparison.png
    results/stage6/cross_center_accuracy_bars.png

Usage:
    python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration import TemperatureScaler
from src.triage.router import operating_point, risk_coverage_curve

ZS = PROJECT_ROOT / "results" / "zeroshot"
CAL = PROJECT_ROOT / "results" / "calibration"
CNN = PROJECT_ROOT / "results" / "cnn"


def _vlm_curves(model_slug: str, split_tag: str, T: float, seed: int = 42) -> dict:
    path = ZS / model_slug / f"V3_full{split_tag}" / "predictions.parquet"
    df = pd.read_parquet(path)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    raw_conf = df["pred_confidence"].to_numpy(float)
    scaler = TemperatureScaler()
    scaler.T = T
    cal_conf = scaler.transform_scalar(raw_conf)
    curve = risk_coverage_curve(cal_conf, correct, tie_break="expected")
    op15 = operating_point(cal_conf, correct, 0.15)
    return {"curve": curve, "op15": op15, "n": len(df), "acc": float(correct.mean())}


def _cnn_curves(split: str, T: float, seed: int = 42) -> dict:
    path = CNN / "resnet18_64px" / f"{split}_predictions.parquet"
    df = pd.read_parquet(path)
    logit_cols = [f"logit_{c}" for c in range(9)]
    logits = df[logit_cols].to_numpy(float)
    labels = df["true_label_idx"].to_numpy(int)
    scaler = TemperatureScaler()
    scaler.T = T
    probs = scaler.transform_logits(logits)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)
    conf = probs.max(axis=1)
    curve = risk_coverage_curve(conf, correct, tie_break="expected")
    op15 = operating_point(conf, correct, 0.15)
    return {"curve": curve, "op15": op15, "n": len(df), "acc": float(correct.mean())}


def plot_fullscale_routing() -> Path:
    mg_T = json.loads((CAL / "medgemma-27b-it" / "V3_full" / "calibration_params.json").read_text())["T"]
    g3_T = json.loads((CAL / "gemma-3-27b-it" / "V3_full" / "calibration_params.json").read_text())["T"]
    cnn_T = json.loads((CAL / "resnet18_64px" / "calibration_params.json").read_text())["T"]

    data = {
        # No CNN on validation: its training and validation patches share slides,
        # so its validation routing is leakage-inflated and the paper reports CNN
        # routing on the external cohort only.
        "val": {
            "MedGemma-27b-it": _vlm_curves("medgemma-27b-it", "", mg_T),
            "Gemma-3-27b-it": _vlm_curves("gemma-3-27b-it", "", g3_T),
        },
        "test": {
            "MedGemma-27b-it": _vlm_curves("medgemma-27b-it", "_test", mg_T),
            "Gemma-3-27b-it": _vlm_curves("gemma-3-27b-it", "_test", g3_T),
            "ResNet-18 64px": _cnn_curves("test", cnn_T),
        },
    }

    colors = {"MedGemma-27b-it": "#D65F5F", "Gemma-3-27b-it": "#E8A33D", "ResNet-18 64px": "#4878CF"}
    # Darker shades for the dashed references, which a pale amber would lose in print.
    dark = {"MedGemma-27b-it": "#A33A3A", "Gemma-3-27b-it": "#B07415", "ResNet-18 64px": "#2D559B"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, split in zip(axes, ["val", "test"]):
        # Highest AUC first, so the legend lists models in the order the curves sit.
        for name, d in sorted(data[split].items(), key=lambda kv: -kv[1]["curve"]["auc"]):
            c = d["curve"]
            budgets = np.array(c["budgets"])
            accs = np.array(c["auto_confirm_acc"], dtype=float)
            valid = ~np.isnan(accs)
            short = name.split("-27b")[0].split(" 64px")[0]
            ax.plot(budgets[valid], accs[valid], "-", color=colors[name], lw=1.8,
                     label=f"{short}, AUC {c['auc']:.3f}")
            # Random routing confirms prefixes whose expected accuracy is the base
            # rate at every budget, so each model's reference is a flat line.
            ax.axhline(d["acc"], color=dark[name], lw=1.1, ls="--",
                       label=f"{short}, random {d['acc']:.3f}")
        ax.axvline(0.15, color="black", lw=0.9, ls=":", alpha=0.7)
        # Anchored to the axes rather than the data, at a height where no curve
        # runs in that panel. No table number: the manuscript renumbers tables.
        ann_y = 0.27 if split == "val" else 0.40
        ax.annotate("b = 0.15\noperating point", xy=(0.15, ann_y - 0.02),
                    xycoords=("data", "axes fraction"), xytext=(0.22, ann_y),
                    textcoords=("data", "axes fraction"), fontsize=9, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
        ax.set_xlabel("Fraction routed to specialist (budget b)", fontsize=10)
        ax.set_ylabel("Auto-confirm accuracy", fontsize=10)
        ax.set_xlim(0, 0.99)  # the last budget with a confirmed set, where A_AUC ends
        ax.yaxis.set_major_locator(MultipleLocator(0.05 if split == "val" else 0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{'Validation (NCT-CRC)' if split == 'val' else 'Test (CRC-VAL-HE-7K)'}", fontsize=11)
        ax.text(-0.12, 1.04, "a" if split == "val" else "b", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        # Each panel's empty region, so no curve runs under the legend.
        if split == "val":
            ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.89), frameon=False)
        else:
            ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.56, 0.64), frameon=False)

    # No in-figure title: the caption carries it.
    fig.tight_layout()
    out_path = PROJECT_ROOT / "results" / "routing" / "risk_coverage_fullscale_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print for verification against Table 3
    print("Routing figure verification against Table 3:")
    for split in ["val", "test"]:
        for name, d in data[split].items():
            print(f"  {name:20s} {split:5s} AUC={d['curve']['auc']:.4f}  "
                  f"random_AUC={d['acc']:.4f}  op15={d['op15']['auto_confirm_acc']:.4f}")
    return out_path


def plot_cross_center_bars() -> Path:
    mg_val = json.loads((ZS / "medgemma-27b-it" / "V3_full" / "metrics.json").read_text())
    mg_test = json.loads((ZS / "medgemma-27b-it" / "V3_full_test" / "metrics.json").read_text())
    g3_val = json.loads((ZS / "gemma-3-27b-it" / "V3_full" / "metrics.json").read_text())
    g3_test = json.loads((ZS / "gemma-3-27b-it" / "V3_full_test" / "metrics.json").read_text())
    cnn64 = json.loads((CNN / "resnet18_64px" / "metrics.json").read_text())
    cnn224 = json.loads((CNN / "resnet18_224px" / "metrics.json").read_text())

    models = ["MedGemma-27b-it", "Gemma-3-27b-it", "ResNet-18 64px", "ResNet-18 224px"]
    val_acc = [mg_val["accuracy"], g3_val["accuracy"], cnn64["val_accuracy"], cnn224["val_accuracy"]]
    test_acc = [mg_test["accuracy"], g3_test["accuracy"], cnn64["test_accuracy"], cnn224["test_accuracy"]]

    # Error bars on the CNN rows only. The VLM rows are full-split evaluations of
    # a fixed remote model, so there is no seed to vary; the CNNs were trained
    # three times each and the spread belongs on the figure that carries the
    # resolution comparison.
    val_err = [0.0, 0.0, 0.0, 0.0]
    test_err = [0.0, 0.0, 0.0, 0.0]
    seeds_path = CNN / "multiseed_summary.json"
    if seeds_path.exists():
        seeds = json.loads(seeds_path.read_text())["resolutions"]
        for i, key in ((2, "64px"), (3, "224px")):
            res = seeds.get(key, {})
            if res.get("n_present", 0) > 1:
                val_err[i] = res["val_accuracy"]["sd"]
                test_err[i] = res["test_accuracy"]["sd"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ebar = {"ecolor": "#333333", "capsize": 3, "error_kw": {"elinewidth": 1.0}}
    b1 = ax.bar(x - width / 2, val_acc, width, label="Validation (NCT-CRC)", color="#4878CF",
                yerr=val_err, **ebar)
    b2 = ax.bar(x + width / 2, test_acc, width, label="Test (CRC-VAL-HE-7K)", color="#D65F5F",
                yerr=test_err, **ebar)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)

    for i, (v, t) in enumerate(zip(val_acc, test_acc)):
        ax.annotate(f"$\\Delta$={t - v:+.3f}", (x[i], max(v, t) + 0.05),
                    ha="center", va="bottom", fontsize=8, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Cohort shift: validation versus external test accuracy", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out_path = PROJECT_ROOT / "results" / "stage6" / "cross_center_accuracy_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nCross-center figure verification against Table 4:")
    for name, v, t, ve, te in zip(models, val_acc, test_acc, val_err, test_err):
        bar = f"  {name:20s} val={v:.4f}  test={t:.4f}  delta={t - v:+.4f}"
        print(bar + (f"  (seed sd: val {ve:.4f}, test {te:.4f})" if te else ""))
    return out_path


def main() -> int:
    p1 = plot_fullscale_routing()
    print(f"\nSaved: {p1}")
    p2 = plot_cross_center_bars()
    print(f"Saved: {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
