"""
src/data/pathmnist.py

PathMNIST dataset loader for the cpath-triage pipeline.

Design constraint: the training split images decompress to ~13.5 GB. Never
load train images into RAM. Use load_labels() for class counts and
load_split_arrays() only for the val/test splits when images are needed.

Usage:
    from src.data.pathmnist import (
        load_labels, load_split_arrays, balanced_indices,
        LABEL_NAMES, LABEL_ABBREV,
    )

    # Class counts without touching image data
    labels = load_labels("train")   # shape (N,) int array, <1 MB

    # Load a small split (val or test only; never train)
    images, labels = load_split_arrays("val")  # ~1.5 GB decompressed

    # Balanced index list from labels alone
    idxs = balanced_indices(labels, n_per_class=50, seed=42)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("data/raw")
NPZ_PATH = DATA_ROOT / "pathmnist_224.npz"

# PathMNIST label index to canonical name (MedMNIST v2, Kather et al. classes)
LABEL_NAMES: dict[int, str] = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

LABEL_TO_IDX: dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}

LABEL_ABBREV: dict[int, str] = {
    0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC",
    5: "MUS", 6: "NORM", 7: "STR", 8: "TUM",
}

# Official split sizes for sanity check
EXPECTED_SIZES = {"train": 89_996, "val": 10_004, "test": 7_180}

_VALID_SPLITS = frozenset(EXPECTED_SIZES)


def _npz_key(split: str, kind: str) -> str:
    """Return the npz key, e.g. 'val_images' or 'train_labels'."""
    return f"{split}_{kind}"


def load_labels(split: str, npz_path: Path = NPZ_PATH) -> np.ndarray:
    """
    Load only the label array for a split. Reads < 1 MB regardless of split size.

    Args:
        split: "train", "val", or "test".
        npz_path: Path to pathmnist_224.npz.

    Returns:
        1-D integer array of shape (N,).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {npz_path}. Run scripts/download_data.py first."
        )
    npz = np.load(npz_path)
    labels = npz[_npz_key(split, "labels")].squeeze().astype(np.int32)
    npz.close()
    return labels


def load_split_arrays(
    split: str, npz_path: Path = NPZ_PATH
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both images and labels for a split.

    WARNING: only call this for "val" (~1.5 GB) or "test" (~1.1 GB).
    The train split images decompress to ~13.5 GB and will exhaust RAM.

    Args:
        split: "val" or "test" (train is accepted but will thrash RAM).
        npz_path: Path to pathmnist_224.npz.

    Returns:
        Tuple of (images, labels) where images is uint8 (N, H, W, 3)
        and labels is int32 (N,).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {npz_path}. Run scripts/download_data.py first."
        )
    npz = np.load(npz_path)
    images = npz[_npz_key(split, "images")]   # (N, H, W, 3) uint8
    labels = npz[_npz_key(split, "labels")].squeeze().astype(np.int32)
    # Copy to regular arrays so we can close the npz file handle
    images = np.array(images)
    labels = np.array(labels)
    npz.close()
    return images, labels


def balanced_indices(
    labels: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> list[tuple[int, int, str]]:
    """
    Return a class-balanced list of (dataset_index, label_idx, label_name).

    Works from a labels array alone -- no image loading. If a class has
    fewer than n_per_class samples, all of them are included.

    Args:
        labels: 1-D integer label array (from load_labels).
        n_per_class: Target samples per class.
        seed: Random seed.

    Returns:
        Shuffled list of (index, label_idx, label_name).
    """
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {k: [] for k in LABEL_NAMES}
    for i, lbl in enumerate(labels.tolist()):
        class_indices[int(lbl)].append(i)

    result: list[tuple[int, int, str]] = []
    for class_idx, indices in class_indices.items():
        k = min(n_per_class, len(indices))
        chosen = rng.sample(indices, k)
        for i in chosen:
            result.append((i, class_idx, LABEL_NAMES[class_idx]))

    rng.shuffle(result)
    return result


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a uint8 (H, W, 3) array to a PIL RGB image."""
    img = Image.fromarray(arr.astype(np.uint8))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_train_mmap(npz_path: Path = NPZ_PATH) -> np.ndarray:
    """
    Return the train images as a memory-mapped (or lazily-loaded) array.

    For uncompressed npz files, numpy creates a true memory map so the OS
    handles paging -- peak RAM stays well under the 13.5 GB decompressed size.
    For compressed npz files, numpy decompresses to a temp file first; the
    mmap is still created but requires temporary disk space.

    Callers must use num_workers=0 in any DataLoader wrapping this array to
    avoid mmap fork-safety issues on Windows.

    Args:
        npz_path: Path to pathmnist_224.npz.

    Returns:
        uint8 array of shape (89996, H, W, 3). The array may be a memmap or
        a regular ndarray depending on the npz compression format.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {npz_path}. Run scripts/download_data.py first."
        )
    npz = np.load(npz_path, mmap_mode="r")
    return npz["train_images"]


def stratified_subset(
    labels: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a stratified index array (not shuffled) for a subset of train data.

    Unlike balanced_indices(), this returns raw indices into the original array
    suitable for numpy fancy indexing, not (arr_idx, label_idx, label_name) tuples.

    Args:
        labels: 1-D integer label array.
        n_per_class: Target samples per class.
        seed: Random seed.

    Returns:
        1-D int64 index array of length <= n_per_class * n_classes.
    """
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {k: [] for k in LABEL_NAMES}
    for i, lbl in enumerate(labels.tolist()):
        class_indices[int(lbl)].append(i)

    result: list[int] = []
    for class_idx, indices in class_indices.items():
        k = min(n_per_class, len(indices))
        result.extend(rng.sample(indices, k))

    return np.array(result, dtype=np.int64)
