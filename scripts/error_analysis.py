"""
scripts/error_analysis.py

Stage 7: error analysis on MedGemma val and test predictions.

Produces:
    results/stage7/routed_patch_montage.png   -- 5x9 grid of the highest-uncertainty
                                                 patches per class (those routed at 15%)
    results/stage7/per_class_f1_comparison.png -- val vs test F1 per class, bar chart
    results/stage7/confusion_class_accuracy.png -- per-class accuracy heatmap val vs test
    results/stage7/error_summary.json          -- structured per-class metrics

Loads 224px val images (~1.5 GB) briefly for the montage, then frees them.

Usage:
    python scripts/error_analysis.py [--budget 0.15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import (
    LABEL_NAMES, LABEL_ABBREV, load_split_arrays, arr_to_pil,
)
from src.eval.calibration import TemperatureScaler

OUT_DIR = PROJECT_ROOT / "results" / "stage7"
ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
N_CLASSES = 9


def _load_vlm_df(path: Path, T: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["correct"] = df["pred_label"] == df["true_label_name"]
    raw_conf = df["pred_confidence"].to_numpy(float)
    scaler = TemperatureScaler()
    scaler.T = T
    df["cal_confidence"] = scaler.transform_scalar(raw_conf)
    df["uncertainty"] = 1.0 - df["cal_confidence"]
    return df


def per_class_metrics(df: pd.DataFrame) -> dict:
    """Compute per-class precision, recall, F1, accuracy from predictions."""
    from sklearn.metrics import precision_recall_fscore_support
    true = df["true_label_name"].tolist()
    pred = df["pred_label"].tolist()
    p, r, f, s = precision_recall_fscore_support(
        true, pred, labels=ALL_LABELS, zero_division=0
    )
    result = {}
    for i, label in enumerate(ALL_LABELS):
        mask = df["true_label_name"] == label
        acc = df.loc[mask, "correct"].mean() if mask.sum() > 0 else float("nan")
        result[label] = {
            "precision": round(float(p[i]), 4),
            "recall":    round(float(r[i]), 4),
            "f1":        round(float(f[i]), 4),
            "accuracy":  round(float(acc),  4),
            "support":   int(s[i]),
        }
    return result


def plot_per_class_f1(val_metrics: dict, test_metrics: dict, out_path: Path) -> None:
    """Grouped bar chart: val vs test F1 per class."""
    abbrevs = [LABEL_ABBREV[i] for i in range(N_CLASSES)]
    val_f1  = [val_metrics[LABEL_NAMES[i]]["f1"]  for i in range(N_CLASSES)]
    test_f1 = [test_metrics[LABEL_NAMES[i]]["f1"] for i in range(N_CLASSES)]

    x = np.arange(N_CLASSES)
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, val_f1,  w, label="Val (same center)",   color="#4878CF", alpha=0.85)
    ax.bar(x + w/2, test_f1, w, label="Test (cross-center)", color="#D65F5F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(abbrevs, fontsize=10)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("MedGemma per-class F1: val vs test (CRC-VAL-HE-7K)")
    ax.legend()

    # Full class names on secondary axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL_NAMES[i] for i in range(N_CLASSES)],
                        fontsize=7, rotation=35, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_routed_montage(
    df_val: pd.DataFrame,
    budget: float,
    out_path: Path,
) -> None:
    """
    5 x 9 montage of the highest-uncertainty patches per class.

    Loads 224px val images (~1.5 GB) to retrieve the actual patch content,
    selects the top-5 most uncertain predictions for each tissue class,
    then frees the image array immediately.
    """
    n_show = 5
    print("  Loading 224px val images for montage (~1.5 GB)...", flush=True)
    val_images, _ = load_split_arrays("val")

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(n_show, N_CLASSES, figure=fig, hspace=0.05, wspace=0.05)

    for col, label_idx in enumerate(range(N_CLASSES)):
        label_name = LABEL_NAMES[label_idx]
        # Sort predictions for this class by descending uncertainty
        class_df = df_val[df_val["true_label_name"] == label_name].sort_values(
            "uncertainty", ascending=False
        ).head(n_show)

        for row, (_, pred_row) in enumerate(class_df.iterrows()):
            ax = fig.add_subplot(gs[row, col])
            arr_idx = int(pred_row["arr_idx"])
            img = arr_to_pil(val_images[arr_idx])
            ax.imshow(img)
            # Red border if wrong prediction, green if correct
            edge_color = "#6ACC65" if pred_row["correct"] else "#D65F5F"
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(
                    f"{LABEL_ABBREV[label_idx]}\n{label_name}",
                    fontsize=7, pad=3,
                )

    del val_images  # free ~1.5 GB

    fig.suptitle(
        "Highest-uncertainty patches per class (MedGemma V3 val)\n"
        "Green border = correct, red = wrong",
        fontsize=10, y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_comparison(
    df_val: pd.DataFrame, df_test: pd.DataFrame, out_path: Path
) -> None:
    """Side-by-side confusion matrices for val and test."""
    abbrevs = [LABEL_ABBREV[i] for i in range(N_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, df, title in zip(axes, [df_val, df_test],
                              ["Val (same center)", "Test (cross-center)"]):
        cm = confusion_matrix(
            df["true_label_name"], df["pred_label"],
            labels=ALL_LABELS, normalize="true",
        )
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(abbrevs, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(abbrevs, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"MedGemma V3 -- {title}", fontsize=10)
        for r in range(N_CLASSES):
            for c in range(N_CLASSES):
                v = cm[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7: error analysis")
    parser.add_argument("--budget", type=float, default=0.15)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_path  = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3" / "predictions.parquet"
    test_path = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3_test" / "predictions.parquet"
    cal_path  = PROJECT_ROOT / "results" / "calibration" / "medgemma-27b-it" / "V3" / "calibration_params.json"

    for p in [val_path, test_path, cal_path]:
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    with open(cal_path) as f:
        T = json.load(f)["T"]

    print(f"Loading predictions (T={T})...")
    df_val  = _load_vlm_df(val_path,  T)
    df_test = _load_vlm_df(test_path, T)

    # ── Per-class metrics ──
    val_metrics  = per_class_metrics(df_val)
    test_metrics = per_class_metrics(df_test)

    print("\nPer-class F1 (val | test):")
    print(f"  {'Class':<45} {'Val F1':>7}  {'Test F1':>7}  {'delta':>7}")
    print("  " + "-" * 68)
    for label in ALL_LABELS:
        vf = val_metrics[label]["f1"]
        tf = test_metrics[label]["f1"]
        print(f"  {label:<45} {vf:>7.4f}  {tf:>7.4f}  {tf-vf:>+7.4f}")

    with open(OUT_DIR / "error_summary.json", "w") as f:
        json.dump({"T": T, "val": val_metrics, "test": test_metrics}, f, indent=2)

    # ── Figures ──
    print("\nGenerating per-class F1 comparison figure...")
    plot_per_class_f1(val_metrics, test_metrics, OUT_DIR / "per_class_f1_comparison.png")

    print("\nGenerating side-by-side confusion matrices...")
    plot_confusion_comparison(df_val, df_test, OUT_DIR / "confusion_val_vs_test.png")

    print("\nGenerating routed patch montage...")
    plot_routed_montage(df_val, args.budget, OUT_DIR / "routed_patch_montage.png")

    print(f"\nStage 7 error analysis complete. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
