"""
scripts/figures/make_error_montage.py

Supplementary figure: validation patches MedGemma-27b-it labelled smooth muscle
under V3, by true class, beside patches it labelled smooth muscle correctly.

Why this exists. Under V3 most mucus (84%) and normal mucosa (90%) patches, and
the largest share of stroma patches (42%), are called smooth muscle. Showing
those patches next to true smooth muscle lets a reader judge how much of the
confusion is visual similarity at 224 px. Patches are drawn at random (seed 42)
within each row's cell, not chosen by eye, and are shown at native resolution
(interpolation "none" embeds the 224 px arrays unresampled in the PDF).

Reads only the 20 patches it draws, through a memory map of the uncompressed
224 px archive. NCT-CRC-HE-100K is CC BY 4.0; the caption cites it.

Outputs:
    paper/latex/figures/figS_error_montage.pdf (and .png)
    results/figures/error_montage_indices.json   which arr_idx went where

Usage:
    python scripts/figures/make_error_montage.py [--seed 42] [--per-row 5]
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import NPZ_PATH, missing_data_message

PRED = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3_full" / "predictions.parquet"
FIG_DIR = PROJECT_ROOT / "paper" / "latex" / "figures"
IDX_OUT = PROJECT_ROOT / "results" / "figures" / "error_montage_indices.json"
ROWS = (
    ("mucus", "smooth muscle", "True MUC, called MUS"),
    ("normal colon mucosa", "smooth muscle", "True NORM, called MUS"),
    ("cancer-associated stroma", "smooth muscle", "True STR, called MUS"),
    ("smooth muscle", "smooth muscle", "True MUS, called MUS"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-row", type=int, default=5)
    args = ap.parse_args()
    print(f"seed={args.seed}")
    if not NPZ_PATH.exists():
        print(missing_data_message(NPZ_PATH), file=sys.stderr)
        return 1

    df = pd.read_parquet(PRED)
    rng = np.random.default_rng(args.seed)
    picks: dict[str, list[dict]] = {}
    for true, pred, title in ROWS:
        cell = df[(df["true_label_name"] == true) & (df["pred_label"] == pred)]
        chosen = cell.iloc[np.sort(rng.choice(len(cell), size=args.per_row, replace=False))]
        picks[title] = [{"arr_idx": int(r.arr_idx), "confidence": float(r.pred_confidence)}
                        for r in chosen.itertuples()]
        print(f"{title}: {len(cell)} patches in cell, drew {[p['arr_idx'] for p in picks[title]]}")

    images = np.load(NPZ_PATH, mmap_mode="r")["val_images"]

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})
    n_rows, n_cols = len(ROWS), args.per_row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.25 + 1.3, n_rows * 1.3))
    for r, (title, items) in enumerate(picks.items()):
        for c, item in enumerate(items):
            ax = axes[r, c]
            ax.imshow(np.asarray(images[item["arr_idx"]]), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{item['arr_idx']}, {item['confidence']:.2f}", fontsize=6.5, labelpad=2)
        axes[r, 0].set_ylabel(title.replace(", ", ",\n"), fontsize=7.5, rotation=0, ha="right",
                              va="center", labelpad=6)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = FIG_DIR / "figS_error_montage.pdf", FIG_DIR / "figS_error_montage.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    IDX_OUT.parent.mkdir(parents=True, exist_ok=True)
    IDX_OUT.write_text(json.dumps({"seed": args.seed, "source": "PathMNIST validation, 224 px",
                                   "rows": picks}, indent=1), encoding="utf-8")
    print(f"Saved: {pdf}\nSaved: {png}\nSaved: {IDX_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
