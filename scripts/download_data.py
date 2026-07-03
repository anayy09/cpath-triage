"""
scripts/download_data.py

Download PathMNIST (MedMNIST v2) at 224 px into data/raw/, verify the splits,
and print the on-disk footprint. PathMNIST source is NCT-CRC-HE-100K (train +
val) and the external test set is CRC-VAL-HE-7K from a different clinical
center. Keep the test set untouched until the cross-center stage.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data/raw")
SIZE_BUDGET_GB = 25.0

# Expected official split sizes for PathMNIST.
EXPECTED = {"train": 89_996, "val": 10_004, "test": 7_180}


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        from medmnist import PathMNIST
        from medmnist.info import INFO
    except ImportError:
        print("medmnist not installed. Run: pip install medmnist")
        return 1

    print("Downloading PathMNIST at 224 px into", DATA_ROOT)
    # size=224 pulls the higher-resolution variant suitable for VLM input.
    # download=True fetches the .npz on first call; splits share one file.
    datasets = {}
    for split in ("train", "val", "test"):
        ds = PathMNIST(split=split, download=True, size=224, root=str(DATA_ROOT))
        datasets[split] = ds
        n = len(ds)
        expected = EXPECTED[split]
        flag = "ok" if n == expected else "MISMATCH"
        print(f"  {split:5s}: {n:>7d} samples (expected {expected}) [{flag}]")
        if n != expected:
            print(f"    split size mismatch for {split}; investigate before proceeding")
            return 1

    info = INFO["pathmnist"]
    print("Classes:", info["label"])

    size_gb = _dir_size_gb(DATA_ROOT)
    print(f"\nOn-disk footprint: {size_gb:.2f} GB (budget {SIZE_BUDGET_GB} GB)")
    if size_gb > SIZE_BUDGET_GB:
        print("OVER BUDGET. Remove the high-res variant or use a smaller size.")
        return 1

    print("Done. Test split (CRC-VAL-HE-7K) is the external center; do not touch it until Stage 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
