"""
scripts/train_cnn_224px.py

ResNet-18 at 224px using memory-mapped loading for the train split.

The train split decompresses to ~13.5 GB RAM. Two modes:
  - mmap mode (default): opens the 224px npz with mmap_mode='r' so the OS
    pages data in on demand. Works well when system RAM >= ~16 GB. With less
    RAM the page-fault rate slows training -- in that case use --subset.
  - subset mode (--subset N): loads a stratified N-sample subset into RAM in
    full (~270 MB at N=20000). Use when mmap is too slow (>5 min/epoch).

Peak RAM is monitored with psutil and printed after training completes.

Outputs (under results/cnn/resnet18_224px/):
    best_model.pt        -- best val-accuracy checkpoint
    metrics.json         -- val and test accuracy, macro F1, training curve
    training_curve.png   -- loss and accuracy per epoch

Usage:
    # Check RAM before committing
    python scripts/train_cnn_224px.py --estimate-only

    # Default: mmap, full train split
    python scripts/train_cnn_224px.py

    # Fallback: 20k stratified subset (fast, lower RAM)
    python scripts/train_cnn_224px.py --subset 20000
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
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import (
    LABEL_NAMES,
    load_labels,
    load_split_arrays,
    load_train_mmap,
)

NPZ_64_PATH = PROJECT_ROOT / "data" / "raw" / "pathmnist_64.npz"

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_CLASSES = 9
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Torch versions used in Upscaled64Dataset (avoids repeated tensor creation per call)
_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Datasets ──────────────────────────────────────────────────────────────────

class MmapTrainDataset(Dataset):
    """
    Dataset that reads individual 224px samples from a memory-mapped npz array.

    The mmap'd array is not loaded into RAM all at once. Each __getitem__ call
    copies one sample from the mmap into a float32 tensor. Use num_workers=0
    to avoid mmap fork issues on Windows.

    Args:
        images: uint8 (N, H, W, 3) mmap'd or regular array.
        labels: int64 (N,) label array.
        augment: Random horizontal + vertical flip per sample (train only).
        indices: Optional index array for subset selection.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        indices: np.ndarray | None = None,
    ) -> None:
        self.images = images
        if indices is not None:
            self.labels = torch.from_numpy(labels[indices].astype(np.int64))
            self.indices = indices
        else:
            self.labels = torch.from_numpy(labels.astype(np.int64))
            self.indices = None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr_idx = int(self.indices[i]) if self.indices is not None else i
        # np.array() copies the mmap slice into a new array in RAM (one sample)
        img = np.array(self.images[arr_idx], dtype=np.float32) / 255.0
        img = (img - _MEAN) / _STD
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.augment:
            if random.random() > 0.5:
                t = t.flip(-1)
            if random.random() > 0.5:
                t = t.flip(-2)
        return t, self.labels[i]


class Upscaled64Dataset(Dataset):
    """
    64px images held as uint8 CHW tensors (~1.1 GB for all 90K samples), normalised
    and bilinearly upscaled to 224px per sample in __getitem__.

    This is the automatic fallback when the 224px mmap loading fails due to limited
    RAM. The upscaling adds ~0.5 ms per sample vs pre-built tensor loading.
    Training with this dataset is documented in metrics.json as 'upscaled_64px_to_224px'.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        target_size: int = 224,
    ) -> None:
        arr = np.ascontiguousarray(images.transpose(0, 3, 1, 2))   # NCHW uint8
        self.images = torch.from_numpy(arr)                         # uint8 -- ~1.1 GB
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].float() / 255.0          # (3, 64, 64) float32
        img = (img - _MEAN_T) / _STD_T
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                     # (3, 224, 224)
        if self.augment:
            if random.random() > 0.5:
                img = img.flip(-1)
            if random.random() > 0.5:
                img = img.flip(-2)
        return img, self.labels[idx]


class PrebuiltDataset(Dataset):
    """
    Val/test dataset backed by a uint8 CHW tensor (4x less RAM than float32).

    Images are held as uint8 (val: 1.5 GB, test: 1.1 GB) and normalised to
    float32 per sample in __getitem__. This avoids OOM when the train mmap
    has already consumed most of the OS page cache.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        arr = np.ascontiguousarray(images.transpose(0, 3, 1, 2))   # NCHW uint8
        self.images = torch.from_numpy(arr)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].float() / 255.0
        img = (img - _MEAN_T) / _STD_T
        return img, self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm not installed. Run: pip install timm") from exc
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=n_classes, in_chans=3)


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


def _peak_ram_gb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / 1024 ** 3


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="ResNet-18 at 224px with mmap or subset loading")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch-size", type=int,   default=32,
                        help="Batch size. 32 is safe for GTX 1650 at 224px.")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=8)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Deprecated -- fallback is now automatic. Ignored if provided.",
    )
    parser.add_argument(
        "--force-mmap", action="store_true",
        help=(
            "Force the 224px native mmap approach regardless of available RAM. "
            "Use when you want valid 224px training data and can tolerate "
            "~10-12 min/epoch on a 16 GB RAM machine. "
            "Upscaled64Dataset (the automatic fallback) has a train/val domain "
            "mismatch and should NOT be used for the paper CNN comparison."
        ),
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="Print RAM usage estimate and exit without training.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    if args.estimate_only:
        avail = psutil.virtual_memory().available / 1024**3
        total = psutil.virtual_memory().total / 1024**3
        print(f"\nSystem RAM: {avail:.1f} GB available / {total:.1f} GB total")
        print("Train split (224px, full): ~13.5 GB decompressed")
        print("  mmap mode: OS pages data on demand; working set ~current usage + page cache")
        print("  subset mode (--subset 20000): ~270 MB for images alone")
        if avail < 16.0:
            print(f"\nWARNING: Available RAM ({avail:.1f} GB) < 16 GB. "
                  f"Consider --subset 20000 to avoid swap thrashing.")
        else:
            print("\nRAM looks sufficient for mmap mode.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s, VRAM: %.1f GB", props.name, props.total_memory / 1024**3)
    else:
        logger.warning("CUDA not available -- training on CPU will be very slow.")

    out_dir = PROJECT_ROOT / "results" / "cnn" / "resnet18_224px"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load train data (auto-fallback) ──
    # The 224px train split decompresses to ~13.5 GB. Effective mmap requires this
    # to be mostly resident in the OS page cache to avoid constant page faults under
    # random-access training. If available RAM < MMAP_MIN_RAM_GB, the mmap path would
    # cause >10 min/epoch (confirmed empirically) and is skipped automatically.
    # Fallback: load the 64px train split (~1.1 GB uint8) and bilinearly upscale to
    # 224px per sample in Upscaled64Dataset.__getitem__. Epoch time ~3-4 min on GTX 1650.
    MMAP_MIN_RAM_GB = 14.0    # need ~13.5 GB resident + overhead
    logger.info("Loading train labels...")
    train_labels = load_labels("train")
    train_ds = None
    train_source = "unknown"

    avail_gb = psutil.virtual_memory().available / 1024**3
    use_mmap = args.force_mmap or (avail_gb >= MMAP_MIN_RAM_GB)

    if use_mmap:
        if args.force_mmap and avail_gb < MMAP_MIN_RAM_GB:
            logger.warning(
                "--force-mmap set: using native 224px mmap despite low RAM "
                "(%.1f GB available). Expect ~10-12 min/epoch due to page faults. "
                "This gives valid training data; Upscaled64Dataset would not.",
                avail_gb,
            )
        else:
            logger.info(
                "Available RAM (%.1f GB) >= %.0f GB: attempting 224px native mmap.",
                avail_gb, MMAP_MIN_RAM_GB,
            )
        try:
            train_images_mmap = load_train_mmap()
            _smoke = np.array(train_images_mmap[0])
            del _smoke
            logger.info(
                "Mmap OK: shape=%s, RAM=%.1f GB", train_images_mmap.shape, _peak_ram_gb()
            )
            train_ds = MmapTrainDataset(train_images_mmap, train_labels, augment=True)
            train_source = "native_224px_mmap"
            logger.info("Train dataset: %d samples (native 224px mmap)", len(train_ds))
        except Exception as exc:
            logger.warning(
                "Mmap failed (%s). Falling back to Upscaled64Dataset.", str(exc)[:120]
            )
    else:
        logger.info(
            "Available RAM (%.1f GB) < %.0f GB required for page-cache-resident mmap. "
            "Using 64px upscaled to 224px. NOTE: this creates a train/val domain "
            "mismatch (blurry train vs sharp native 224px val); use --force-mmap "
            "for a valid 224px CNN comparison.",
            avail_gb, MMAP_MIN_RAM_GB,
        )

    if train_ds is None:
        if not NPZ_64_PATH.exists():
            raise FileNotFoundError(
                f"Fallback 64px npz not found: {NPZ_64_PATH}. "
                "Run scripts/download_data.py or: "
                "python -c \"from medmnist import PathMNIST; "
                "PathMNIST(split='train', download=True, size=64, root='data/raw')\""
            )
        logger.info("Loading 64px train split (~1.1 GB uint8) for upscaled-224px training...")
        npz64 = np.load(NPZ_64_PATH)
        train64_imgs = np.array(npz64["train_images"])
        train64_lbls = np.array(npz64["train_labels"]).squeeze().astype(np.int64)
        npz64.close()
        logger.info(
            "64px train loaded: shape=%s, RAM=%.1f GB", train64_imgs.shape, _peak_ram_gb()
        )
        train_ds = Upscaled64Dataset(train64_imgs, train64_lbls, augment=True, target_size=224)
        del train64_imgs
        train_source = "upscaled_64px_to_224px"
        logger.info(
            "Train dataset: %d samples (64px bilinearly upscaled to 224px), RAM=%.1f GB",
            len(train_ds), _peak_ram_gb(),
        )

    # ── Load val and test (small enough to pre-build) ──
    logger.info("Loading val images...")
    val_images, val_labels = load_split_arrays("val")
    logger.info("Loading test images...")
    test_images, test_labels = load_split_arrays("test")

    val_ds  = PrebuiltDataset(val_images,  val_labels)
    test_ds = PrebuiltDataset(test_images, test_labels)
    del val_images, test_images

    # num_workers=0 required for mmap; also avoids DataLoader pickling issues on Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # ── Model ──
    pretrained = not args.no_pretrained
    model = build_resnet18(N_CLASSES, pretrained=pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "ResNet-18 loaded (pretrained=%s, params=%.1fM, input=224px, batch=%d)",
        pretrained, n_params, args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ──
    best_val_acc = 0.0
    patience_counter = 0
    history: list[dict] = []
    peak_ram = _peak_ram_gb()

    logger.info(
        "Training: epochs=%d, batch=%d, lr=%.1e, patience=%d, seed=%d, fp32, 224px",
        args.epochs, args.batch_size, args.lr, args.patience, args.seed,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        current_ram = _peak_ram_gb()
        peak_ram = max(peak_ram, current_ram)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,     4),
            "val_f1":     round(val_f1,      4),
        })

        logger.info(
            "Epoch %02d/%d | train=%.4f val=%.4f | acc=%.4f f1=%.4f | %.1fs | RAM=%.1fGB",
            epoch, args.epochs, train_loss, val_loss, val_acc, val_f1, elapsed, current_ram,
        )

        if elapsed > 300:
            logger.warning(
                "Epoch took %.0fs (>5 min). Upscaling in __getitem__ is the likely "
                "bottleneck. Training will complete but slowly.", elapsed,
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

    # ── Final evaluation ──
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Best checkpoint: epoch %d (val_acc=%.4f)", ckpt["epoch"], ckpt["val_acc"])

    _, val_acc_f, val_f1_f = evaluate(model, val_loader, criterion, device)
    _, test_acc,  test_f1  = evaluate(model, test_loader, criterion, device)

    metrics = {
        "model": "resnet18",
        "input_size_px": 224,
        "pretrained": pretrained,
        "seed": args.seed,
        "best_epoch": int(ckpt["epoch"]),
        "epochs_trained": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_train_samples": len(train_ds),
        "train_data_source": train_source,
        "val_accuracy":  round(val_acc_f, 4),
        "val_macro_f1":  round(val_f1_f,  4),
        "test_accuracy": round(test_acc,  4),
        "test_macro_f1": round(test_f1,   4),
        "peak_ram_gb":   round(peak_ram,  2),
        "training_history": history,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ResNet-18  224px  pretrained={pretrained}  seed={args.seed}")
    print(f"Train samples: {len(train_ds)} ({train_source})")
    print(f"Val   accuracy={val_acc_f:.4f}  macro F1={val_f1_f:.4f}")
    print(f"Test  accuracy={test_acc:.4f}  macro F1={test_f1:.4f}")
    print(f"Best epoch: {ckpt['epoch']}  |  Peak RAM: {peak_ram:.1f} GB")
    print(f"Results: {metrics_path}")
    print(f"{'='*60}")

    _plot_curve(history, out_dir / "training_curve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
