"""
scripts/eval_cnn.py

Run the trained ResNet-18 on the val (and optionally test) split and save
per-sample predictions, logits, and softmax probabilities.

This output feeds Stage 4 calibration: temperature scaling requires the raw
logits (pre-softmax), which train_cnn.py does not save. eval_cnn.py fills
that gap without re-running training.

Outputs:
    results/cnn/resnet18_64px/val_predictions.parquet
    results/cnn/resnet18_64px/test_predictions.parquet  (if --include-test)

Usage:
    python scripts/eval_cnn.py
    python scripts/eval_cnn.py --include-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES
from scripts.train_cnn import build_resnet18, NPZDataset, _load_64px_data

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
N_CLASSES = 9
MODEL_PATH = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px" / "best_model.pt"
OUT_DIR = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px"


@torch.no_grad()
def run_inference(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run the model on a split and return a DataFrame with one row per sample.

    Columns: sample_idx, true_label_idx, true_label_name, pred_label_idx,
    pred_label_name, confidence (max softmax), correct,
    logit_0..logit_8, prob_0..prob_8.
    """
    ds = NPZDataset(images, labels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits: list[np.ndarray] = []
    model.eval()
    for imgs, _ in loader:
        logits = model(imgs.to(device)).cpu().numpy()
        all_logits.append(logits)

    logits_arr = np.concatenate(all_logits, axis=0)   # (N, 9)

    # Stable softmax
    shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs_arr = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, 9)

    pred_idx = probs_arr.argmax(axis=1)
    confidence = probs_arr.max(axis=1)

    rows = []
    for i, (true_idx, pi, conf) in enumerate(zip(labels, pred_idx, confidence)):
        row = {
            "sample_idx": i,
            "true_label_idx": int(true_idx),
            "true_label_name": LABEL_NAMES[int(true_idx)],
            "pred_label_idx": int(pi),
            "pred_label_name": LABEL_NAMES[int(pi)],
            "confidence": float(conf),
            "correct": int(true_idx) == int(pi),
        }
        for c in range(N_CLASSES):
            row[f"logit_{c}"] = float(logits_arr[i, c])
            row[f"prob_{c}"] = float(probs_arr[i, c])
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN inference -- save logits and probs")
    parser.add_argument("--include-test", action="store_true",
                        help="Also run on the test split (CRC-VAL-HE-7K). "
                             "Only call this when ready for Stage 6 cross-center analysis.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found: {MODEL_PATH}")
        print("Run scripts/train_cnn.py first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the best checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = build_resnet18(N_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    splits = _load_64px_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_splits = ["val"]
    if args.include_test:
        target_splits.append("test")

    for split in target_splits:
        images, labels = splits[split]
        print(f"\nRunning inference on {split} ({len(labels)} samples)...")
        df = run_inference(model, images, labels, args.batch_size, device)

        acc = df["correct"].mean()
        print(f"  Accuracy: {acc:.4f}")

        out_path = OUT_DIR / f"{split}_predictions.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
