"""
scripts/replot_saved_figures.py

Redraw the three manuscript figures whose only other source is a billed or GPU
run, from the outputs those runs saved: the two zero-shot confusion matrices
(manuscript Figures 3 and 4, files fig2 and fig3) and the 224 px CNN training
curve (manuscript Figure 2, file fig8).

Why this exists. run_zeroshot.py draws the confusion matrix at the end of a
full-split API run, and train_cnn_224px.py draws the training curve at the end
of a GPU training run, so without this script a reader of the deposit could
not regenerate either figure without repeating the run. Both runs saved what
the plot needs (predictions.parquet, and the per-epoch history in
metrics.json), and this script calls the same plotting functions on them, so
the output is the published figure by construction rather than a lookalike.

Outputs (overwritten):
    results/zeroshot/medgemma-27b-it/V3_full/confusion_matrix.png
    results/zeroshot/gemma-3-27b-it/V3_full/confusion_matrix.png
    results/cnn/resnet18_224px/training_curve.png

Usage:
    python scripts/replot_saved_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_zeroshot import _plot_confusion_matrix
from train_cnn_224px import _plot_curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Unused; logged for consistency.")
    parser.parse_args()

    written = []
    for model in ("medgemma-27b-it", "gemma-3-27b-it"):
        run = PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full"
        pred = run / "predictions.parquet"
        if not pred.exists():
            print(f"Missing required file: {pred}", file=sys.stderr)
            return 1
        df = pd.read_parquet(pred)
        out = run / "confusion_matrix.png"
        _plot_confusion_matrix(df["true_label_name"].tolist(), df["pred_label"].tolist(), out)
        written.append(out)

    metrics = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px" / "metrics.json"
    if not metrics.exists():
        print(f"Missing required file: {metrics}", file=sys.stderr)
        return 1
    history = json.loads(metrics.read_text(encoding="utf-8"))["training_history"]
    out = metrics.with_name("training_curve.png")
    _plot_curve(history, out)
    written.append(out)

    for p in written:
        print(f"Saved: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
