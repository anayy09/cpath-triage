"""
scripts/summarize_data.py

Stage 1: Characterize the PathMNIST dataset.

Memory-conscious design:
  - Class counts use only label arrays (<1 MB each).
  - The montage loads val images once (~1.5 GB), extracts 9 patches, then
    frees the array. Train images are never loaded.

Outputs:
    results/data_summary.json               -- class counts per split
    results/figures/class_distribution.png  -- grouped bar chart
    results/figures/tissue_montage.png      -- one sample per class

Usage:
    python scripts/summarize_data.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display required; avoids GUI overhead
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import (
    LABEL_NAMES,
    LABEL_ABBREV,
    load_labels,
    load_split_arrays,
    arr_to_pil,
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def build_summary() -> dict:
    """Collect class counts per split using label arrays only (no images)."""
    summary: dict = {"splits": {}, "train_val_total": {}}

    for split in ("train", "val", "test"):
        print(f"  Loading {split} labels...", flush=True)
        labels = load_labels(split)
        unique, counts = np.unique(labels, return_counts=True)
        count_map = dict(zip(unique.tolist(), counts.tolist()))
        summary["splits"][split] = {
            LABEL_NAMES[k]: count_map.get(k, 0) for k in sorted(LABEL_NAMES)
        }
        summary["splits"][split]["_total"] = int(labels.size)

    for idx, name in LABEL_NAMES.items():
        summary["train_val_total"][name] = (
            summary["splits"]["train"].get(name, 0)
            + summary["splits"]["val"].get(name, 0)
        )

    return summary


def plot_class_distribution(summary: dict, out_path: Path) -> None:
    """Grouped bar chart: counts per class for each split."""
    classes = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    abbrevs = [LABEL_ABBREV[i] for i in range(len(LABEL_NAMES))]
    x = np.arange(len(classes))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 5))
    splits = ["train", "val", "test"]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    for split, color, offset in zip(splits, colors, [-width, 0, width]):
        vals = [summary["splits"][split].get(name, 0) for name in classes]
        ax.bar(x + offset, vals, width, label=split, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(abbrevs, fontsize=10)
    ax.set_ylabel("Sample count")
    ax.set_title("PathMNIST class distribution by split")
    ax.legend()

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=7, rotation=35, ha="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tissue_montage(seed: int, out_path: Path) -> None:
    """
    3x3 grid showing one representative patch per tissue class.

    Loads val images (~1.5 GB) once, picks one index per class, converts
    those 9 patches to PIL, then releases the full array.
    """
    print("  Loading val labels...", flush=True)
    val_labels = load_labels("val")

    # Find one example index per class from val labels (no images yet)
    import random
    rng = random.Random(seed)
    indices = list(range(len(val_labels)))
    rng.shuffle(indices)

    class_example: dict[int, int] = {}
    for idx in indices:
        lbl = int(val_labels[idx])
        if lbl not in class_example:
            class_example[lbl] = idx
        if len(class_example) == len(LABEL_NAMES):
            break

    print("  Loading val images to extract montage patches (~1.5 GB)...", flush=True)
    val_images, _ = load_split_arrays("val")  # (10004, 224, 224, 3) uint8

    # Extract the 9 needed patches and immediately free the large array
    patches = {lbl: arr_to_pil(val_images[idx]) for lbl, idx in class_example.items()}
    del val_images  # release ~1.5 GB

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.15)

    for pos, label_idx in enumerate(sorted(patches)):
        ax = fig.add_subplot(gs[pos // 3, pos % 3])
        ax.imshow(patches[label_idx])
        ax.set_title(f"{LABEL_ABBREV[label_idx]}\n{LABEL_NAMES[label_idx]}", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        "PathMNIST tissue classes (val split, 224 px H&E patches)",
        fontsize=12, y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: dataset summary and montage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building class distribution summary (labels only, no images)...")
    summary = build_summary()

    for split, counts in summary["splits"].items():
        total = counts.pop("_total")
        print(f"\n{split} ({total} samples):")
        for name, n in counts.items():
            print(f"  {name:45s} {n:6d}")
        counts["_total"] = total

    out_json = RESULTS_DIR / "data_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")

    plot_class_distribution(summary, FIGURES_DIR / "class_distribution.png")

    print("\nBuilding tissue montage...")
    plot_tissue_montage(args.seed, FIGURES_DIR / "tissue_montage.png")

    print("\nStage 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
