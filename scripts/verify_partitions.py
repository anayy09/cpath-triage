"""
scripts/verify_partitions.py

Two verification checks that share one pass over the PathMNIST npz archives.

1. Partition identity (revision item C-17 / Decision 3). The paper compares a
   64 px and a 224 px ResNet-18 on "the same" split. That has to be checked
   rather than asserted: if the two releases shuffled independently, every
   paired comparison between the two resolutions is meaningless. The check is
   element-wise equality of the label arrays for all three splits.

2. Cluster-identifier availability (revision item C-04 / Decision 2). Reviewer 1
   asks for slide-level and patient-level clustered intervals. That requires a
   per-patch slide or patient identifier. This inventories every array stored in
   the archives so the answer is a recorded fact rather than a recollection.

Neither check needs the image arrays, so both archives are opened lazily and only
the label arrays are materialised. The 224 px archive is 12.6 GB; reading its
labels costs a few MB.

Outputs:
    results/data/partition_identity.json

Usage:
    PATHMNIST_DATA_ROOT=/path/to/npz python scripts/verify_partitions.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path(os.environ.get("PATHMNIST_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))
SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 89_996, "val": 10_004, "test": 7_180}


def _load_labels(npz_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with np.load(npz_path) as npz:
        for split in SPLITS:
            out[split] = np.asarray(npz[f"{split}_labels"]).squeeze().astype(np.int64)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Unused; logged for consistency.")
    args = parser.parse_args()

    npz_64 = DATA_ROOT / "pathmnist_64.npz"
    npz_224 = DATA_ROOT / "pathmnist_224.npz"
    for p in (npz_64, npz_224):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            print("Set PATHMNIST_DATA_ROOT to the directory holding the npz files.", file=sys.stderr)
            return 1

    print(f"64 px archive : {npz_64}")
    print(f"224 px archive: {npz_224}")
    print()

    # ── Check 2: what is actually stored in the archives ──
    keys_64 = sorted(np.load(npz_64).files)
    keys_224 = sorted(np.load(npz_224).files)
    expected_keys = sorted(f"{s}_{k}" for s in SPLITS for k in ("images", "labels"))
    identifier_keys_64 = [k for k in keys_64 if k not in expected_keys]
    identifier_keys_224 = [k for k in keys_224 if k not in expected_keys]

    print("Archive contents")
    print(f"  64 px  keys: {keys_64}")
    print(f"  224 px keys: {keys_224}")
    print(f"  keys beyond {{split}}_{{images,labels}}: "
          f"64 px {identifier_keys_64 or 'none'}, 224 px {identifier_keys_224 or 'none'}")
    print()

    # ── Check 1: are the two resolutions the same patches in the same order ──
    labels_64 = _load_labels(npz_64)
    labels_224 = _load_labels(npz_224)

    identity: dict[str, dict] = {}
    all_identical = True
    for split in SPLITS:
        a, b = labels_64[split], labels_224[split]
        same_length = len(a) == len(b)
        elementwise = bool(same_length and np.array_equal(a, b))
        n_mismatch = int((a != b).sum()) if same_length else None
        expected_n = EXPECTED_SIZES[split]
        identity[split] = {
            "n_64px": len(a),
            "n_224px": len(b),
            "n_expected": expected_n,
            "length_matches_expected": bool(len(a) == expected_n == len(b)),
            "labels_elementwise_identical": elementwise,
            "n_label_mismatches": n_mismatch,
            "class_counts_64px": {int(c): int(n) for c, n in zip(*np.unique(a, return_counts=True))},
        }
        all_identical = all_identical and elementwise
        status = "IDENTICAL" if elementwise else f"MISMATCH ({n_mismatch} positions)"
        print(f"  {split:5s} n={len(a):6d} (expected {expected_n:6d})  labels: {status}")

    print()
    if all_identical:
        print("RESULT: the 64 px and 224 px releases index the same patches in the same order.")
        print("        Paired comparisons between the two resolutions are valid.")
    else:
        print("RESULT: the two releases do NOT align. Every paired 64 px vs 224 px")
        print("        comparison in the paper needs re-examination.")

    has_identifiers = bool(identifier_keys_64 or identifier_keys_224)
    if has_identifiers:
        print(f"IDENTIFIERS: extra arrays present: {identifier_keys_64 + identifier_keys_224}")
        print("             Inspect these before concluding clustered intervals are impossible.")
    else:
        print("IDENTIFIERS: absent. The archives store only images and labels, so no")
        print("             per-patch slide or patient identifier is recoverable from them.")

    out = {
        "seed": args.seed,
        "data_root": str(DATA_ROOT),
        "archives": {"64px": str(npz_64), "224px": str(npz_224)},
        "archive_keys": {"64px": keys_64, "224px": keys_224},
        "identifier_keys_beyond_images_and_labels": {
            "64px": identifier_keys_64,
            "224px": identifier_keys_224,
        },
        "cluster_identifiers_available_in_npz": has_identifiers,
        "partition_identity": identity,
        "all_splits_identical": all_identical,
    }

    out_dir = PROJECT_ROOT / "results" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "partition_identity.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
