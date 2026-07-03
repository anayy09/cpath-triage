"""
scripts/eval_cnn_224px.py

Run the trained 224px ResNet-18 on the val (and optionally test) split and
save per-sample predictions, logits, and softmax probabilities.

This output feeds calibration: temperature scaling requires raw logits
(pre-softmax), which train_cnn_224px.py does not save. Mirrors eval_cnn.py
but for the 224px checkpoint, using PrebuiltDataset (uint8 storage, ~1.5 GB
val + ~1.1 GB test) to avoid the float32 OOM that affected the 224px
training script before its fix.

Outputs:
    results/cnn/resnet18_224px/val_predictions.parquet
    results/cnn/resnet18_224px/test_predictions.parquet  (if --include-test)

Usage:
    python scripts/eval_cnn_224px.py
    python scripts/eval_cnn_224px.py --include-test
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES, load_split_arrays
from scripts.train_cnn_224px import build_resnet18, PrebuiltDataset

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
N_CLASSES = 9
MODEL_PATH = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px" / "best_model.pt"
OUT_DIR = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px"


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
    ds = PrebuiltDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits: list[np.ndarray] = []
    model.eval()
    for imgs, _ in loader:
        logits = model(imgs.to(device)).cpu().numpy()
        all_logits.append(logits)

    logits_arr = np.concatenate(all_logits, axis=0)   # (N, 9)

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
    parser = argparse.ArgumentParser(description="224px CNN inference -- save logits and probs")
    parser.add_argument("--include-test", action="store_true",
                        help="Also run on the test split (CRC-VAL-HE-7K).")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found: {MODEL_PATH}")
        print("Run scripts/train_cnn_224px.py first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = build_resnet18(N_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_splits = ["val"]
    if args.include_test:
        target_splits.append("test")

    for split in target_splits:
        print(f"\nLoading {split} images (224px)...")
        images, labels = load_split_arrays(split)
        print(f"Running inference on {split} ({len(labels)} samples)...")
        df = run_inference(model, images, labels, args.batch_size, device)
        del images

        acc = df["correct"].mean()
        print(f"  Accuracy: {acc:.4f}")

        out_path = OUT_DIR / f"{split}_predictions.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
