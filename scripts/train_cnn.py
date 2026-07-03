"""
scripts/train_cnn.py

Stage 3: Train a ResNet-18 supervised baseline on PathMNIST (64 px).

Why 64 px: the 224 px train split decompresses to ~13.5 GB RAM, exceeding
consumer hardware. 64 px loads the entire train split in ~1.1 GB and is
explicitly allowed by PLAN.md ("drop to 128 px input or a smaller backbone;
document the change"). This gap is named as a limitation in the paper.

AMP is intentionally disabled. ResNet-18 at 64 px in fp32 uses ~300 MB VRAM
at batch_size=64, well within the GTX 1650's 3-4 GB. AMP caused NaN losses
due to fp16 overflow in the initial fc layer; fp32 training is cleaner and
fast enough on this workload.

Performance: ~15-25 s/epoch (GTX 1650) vs 248 s/epoch with the prior
num_workers=0 + per-sample numpy preprocessing approach. The fix is to
pre-convert the full dataset to a CHW float32 tensor once in __init__ so
DataLoader __getitem__ is a single tensor index.

Outputs (under results/cnn/resnet18_64px/):
    best_model.pt      -- best val-accuracy checkpoint
    metrics.json       -- val and test accuracy, macro F1, training curve
    training_curve.png -- loss and accuracy per epoch

Usage:
    python scripts/train_cnn.py [--epochs 40] [--batch-size 64] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
NPZ_64_PATH = DATA_ROOT / "pathmnist_64.npz"
N_CLASSES = 9

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NPZDataset(Dataset):
    """
    Fast Dataset backed by a pre-built CHW float32 tensor.

    Entire split is normalised and transposed once in __init__. After that,
    __getitem__ is a single tensor index plus an optional in-place flip --
    no numpy per-sample work at all, so DataLoader with num_workers=0 is fast.

    Args:
        images: uint8 (N, H, W, C) array.
        labels: int   (N,)          array.
        augment: Random horizontal + vertical flip per sample (train only).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        # Normalise and convert to (N, C, H, W) float32 tensor all at once
        x = images.astype(np.float32) / 255.0   # (N, H, W, C) float32
        x = (x - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))  # (N, C, H, W)
        self.images = torch.from_numpy(x)                   # stays on CPU
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]          # (C, H, W) float32
        if self.augment:
            if random.random() > 0.5:
                img = img.flip(-1)      # horizontal flip
            if random.random() > 0.5:
                img = img.flip(-2)      # vertical flip
        return img, self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 via timm with the final classifier replaced for n_classes."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm not installed. Run: pip install timm") from exc
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=n_classes, in_chans=3)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_64px_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load or download the 64 px PathMNIST npz. Returns {split: (images, labels)}."""
    if not NPZ_64_PATH.exists():
        logger.info("pathmnist_64.npz not found -- downloading via medmnist ...")
        try:
            from medmnist import PathMNIST
        except ImportError as exc:
            raise ImportError("medmnist not installed. Run: pip install medmnist") from exc
        for split in ("train", "val", "test"):
            PathMNIST(split=split, download=True, size=64, root=str(DATA_ROOT))

    logger.info("Loading pathmnist_64.npz ...")
    npz = np.load(NPZ_64_PATH)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        imgs = np.array(npz[f"{split}_images"])
        lbls = np.array(npz[f"{split}_labels"]).squeeze().astype(np.int64)
        splits[split] = (imgs, lbls)
        logger.info("  %s: %d samples, image shape %s", split, len(lbls), imgs.shape[1:])
    npz.close()
    return splits


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping guards against rare spikes during fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ── Utilities ─────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _plot_curve(history: list[dict], out_path: Path) -> None:
    epochs_x = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs_x, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs_x, [h["val_loss"]   for h in history], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Cross-entropy loss")
    ax2.plot(epochs_x, [h["val_acc"] for h in history], label="val acc")
    ax2.plot(epochs_x, [h["val_f1"]  for h in history], label="val macro F1")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3: ResNet-18 baseline (64 px, fp32)")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=8,
                        help="Early stopping: max epochs without val_acc improvement.")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s, VRAM: %.1f GB", props.name, props.total_memory / 1024**3)
    else:
        logger.warning("CUDA not available -- training on CPU will be very slow.")

    out_dir = PROJECT_ROOT / "results" / "cnn" / "resnet18_64px"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    splits = _load_64px_data()
    train_ds = NPZDataset(*splits["train"], augment=True)
    val_ds   = NPZDataset(*splits["val"],   augment=False)
    test_ds  = NPZDataset(*splits["test"],  augment=False)
    logger.info(
        "Datasets built. Train tensor: %s (%.0f MB)",
        tuple(train_ds.images.shape),
        train_ds.images.nbytes / 1024**2,
    )

    # num_workers=0 is fast now because __getitem__ is just a tensor index
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # ── Model ──
    pretrained = not args.no_pretrained
    model = build_resnet18(N_CLASSES, pretrained=pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("ResNet-18 loaded (pretrained=%s, params=%.1fM)", pretrained, n_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max is set to the actual number of epochs we'll run; early stopping may exit earlier
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ──
    best_val_acc = 0.0
    patience_counter = 0
    history: list[dict] = []

    logger.info(
        "Training: epochs=%d, batch=%d, lr=%.1e, patience=%d, seed=%d, fp32",
        args.epochs, args.batch_size, args.lr, args.patience, args.seed,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,     4),
            "val_f1":     round(val_f1,      4),
        })

        logger.info(
            "Epoch %02d/%d | train=%.4f val=%.4f | acc=%.4f f1=%.4f | %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_acc, val_f1, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": val_acc, "val_f1": val_f1},
                out_dir / "best_model.pt",
            )
            logger.info("  -> New best (val_acc=%.4f)", val_acc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Load best checkpoint ──

    # ── Final test evaluation ──
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Best checkpoint: epoch %d (val_acc=%.4f)", ckpt["epoch"], ckpt["val_acc"])

    _, val_acc_f, val_f1_f = evaluate(model, val_loader, criterion, device)
    _, test_acc,  test_f1  = evaluate(model, test_loader, criterion, device)

    metrics = {
        "model": "resnet18",
        "input_size_px": 64,
        "pretrained": pretrained,
        "seed": args.seed,
        "best_epoch": int(ckpt["epoch"]),
        "epochs_trained": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_accuracy":  round(val_acc_f, 4),
        "val_macro_f1":  round(val_f1_f,  4),
        "test_accuracy": round(test_acc,  4),
        "test_macro_f1": round(test_f1,   4),
        "training_history": history,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*55}")
    print(f"ResNet-18  64px  pretrained={pretrained}  seed={args.seed}")
    print(f"Val   accuracy={val_acc_f:.4f}  macro F1={val_f1_f:.4f}")
    print(f"Test  accuracy={test_acc:.4f}  macro F1={test_f1:.4f}")
    print(f"Best epoch: {ckpt['epoch']}  |  Metrics: {metrics_path}")
    print(f"{'='*55}")

    _plot_curve(history, out_dir / "training_curve.png")
    logger.info("Training curve: %s", out_dir / "training_curve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
