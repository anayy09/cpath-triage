"""
scripts/consistency_selection_weights.py

Exact inclusion probabilities for the 1,800-patch K=5 consistency subsets.

The subsets were drawn by _select_patches in scripts/run_consistency.py: for each
true class, up to 100 patches the full-scale V3 run classified correctly and up
to 100 it got wrong, topped up from the other pool when one was short. The draw
is uniform inside each (true class, full-scale correctness) cell, so a patch's
inclusion probability is the cell's draw count over the cell's pool size, and
1/pi is its Horvitz-Thompson weight.

Rather than trust that description, the script replays the selection with the
same seed and the same full-scale predictions file and refuses to write
anything unless the replay reproduces the stored 1,800 patch indices exactly.
No images and no API calls are needed.

Outputs:
    results/consistency/{model}/{run}/weights.parquet
        arr_idx, true_label_name, fullscale_correct, cell, pool_size,
        n_drawn, inclusion_prob, weight, in_pilot
    results/consistency/{model}/{run}/selection_cells.json

Usage:
    python scripts/consistency_selection_weights.py --split val
    python scripts/consistency_selection_weights.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES

N_PER_CLASS = 200


def replay_selection(full_pred: pd.DataFrame, n_per_class: int, seed: int) -> list[int]:
    """
    Re-run the stratified branch of run_consistency._select_patches.

    Kept line-for-line equivalent, including drawing from Python lists in file
    order, because numpy's choice consumes the generator differently for
    different pool orders and any deviation changes which patches come out.
    """
    rng = np.random.default_rng(seed)
    chosen_all: list[int] = []
    for class_idx in range(len(LABEL_NAMES)):
        class_name = LABEL_NAMES[class_idx]
        class_df = full_pred[full_pred["true_label_name"] == class_name]
        correct_idx = class_df[class_df["pred_label"] == class_name]["arr_idx"].tolist()
        incorrect_idx = class_df[class_df["pred_label"] != class_name]["arr_idx"].tolist()
        n_half = n_per_class // 2
        chosen_correct = (
            rng.choice(correct_idx, min(n_half, len(correct_idx)), replace=False).tolist()
            if correct_idx else []
        )
        chosen_incorrect = (
            rng.choice(incorrect_idx, min(n_half, len(incorrect_idx)), replace=False).tolist()
            if incorrect_idx else []
        )
        chosen = chosen_correct + chosen_incorrect
        if len(chosen) < n_per_class:
            remaining_pool = [i for i in correct_idx + incorrect_idx if i not in set(chosen)]
            extra = (
                rng.choice(
                    remaining_pool,
                    min(n_per_class - len(chosen), len(remaining_pool)),
                    replace=False,
                ).tolist()
                if remaining_pool else []
            )
            chosen.extend(extra)
        chosen_all.extend(int(i) for i in chosen)
    return chosen_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--version", default="V3")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed}")

    suffix = "_test" if args.split == "test" else ""
    full_path = (PROJECT_ROOT / "results" / "zeroshot" / args.model
                 / f"{args.version}_full{suffix}" / "predictions.parquet")
    cons_dir = PROJECT_ROOT / "results" / "consistency" / args.model / f"{args.version}{suffix}"
    cons_path = cons_dir / "predictions.parquet"
    pilot_path = PROJECT_ROOT / "results" / "zeroshot" / args.model / "V3" / "predictions.parquet"
    for p in (full_path, cons_path, pilot_path):
        if not p.exists():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    full = pd.read_parquet(full_path)
    cons = pd.read_parquet(cons_path)

    replayed = replay_selection(full, N_PER_CLASS, args.seed)
    stored = cons["arr_idx"].astype(int).tolist()
    if replayed != stored:
        same_set = set(replayed) == set(stored)
        print(
            f"Replay does not reproduce the stored selection "
            f"(same set: {same_set}, replayed {len(replayed)}, stored {len(stored)}). "
            "Refusing to write weights.",
            file=sys.stderr,
        )
        return 1

    full = full.assign(fullscale_correct=full["pred_label"] == full["true_label_name"])
    full["cell"] = full["true_label_name"] + "|" + np.where(full["fullscale_correct"], "correct", "incorrect")
    pool = full.groupby("cell").size().rename("pool_size")
    sel = full.set_index("arr_idx").loc[stored].reset_index()
    drawn = sel.groupby("cell").size().rename("n_drawn")
    cells = pd.concat([pool, drawn], axis=1).fillna(0).astype(int)
    cells["inclusion_prob"] = cells["n_drawn"] / cells["pool_size"]

    sel = sel.join(cells, on="cell")
    sel["weight"] = 1.0 / sel["inclusion_prob"]
    # The pilot used to choose V3 drew 450 validation patches; marking them here
    # lets every downstream analysis be rerun without them.
    pilot = set(pd.read_parquet(pilot_path)["arr_idx"].astype(int))
    sel["in_pilot"] = sel["arr_idx"].isin(pilot) if args.split == "val" else False

    out = sel[["arr_idx", "true_label_name", "fullscale_correct", "cell", "pool_size",
               "n_drawn", "inclusion_prob", "weight", "in_pilot"]]
    weights_path = cons_dir / "weights.parquet"
    out.to_parquet(weights_path, index=False)

    # Pool sizes after removing pilot patches, for the pilot-excluded estimand.
    full["in_pilot"] = full["arr_idx"].isin(pilot) if args.split == "val" else False
    pool_np = full[~full["in_pilot"]].groupby("cell").size()
    drawn_np = sel[~sel["in_pilot"]].groupby("cell").size()

    summary = {
        "model": args.model,
        "version": args.version,
        "split": args.split,
        "seed": args.seed,
        "replay_reproduces_stored_selection": True,
        "n_selected": len(stored),
        "cohort_size": len(full),
        "sum_of_weights": round(float(sel["weight"].sum()), 6),
        "n_pilot_in_subset": int(sel["in_pilot"].sum()),
        "cells": {
            c: {
                "pool_size": int(r.pool_size),
                "n_drawn": int(r.n_drawn),
                "inclusion_prob": round(float(r.inclusion_prob), 6),
                "pool_size_excl_pilot": int(pool_np.get(c, 0)),
                "n_drawn_excl_pilot": int(drawn_np.get(c, 0)),
            }
            for c, r in cells.iterrows()
        },
    }
    cells_path = cons_dir / "selection_cells.json"
    cells_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Replay reproduces the stored {len(stored)}-patch selection exactly.")
    print(f"Sum of weights {summary['sum_of_weights']:.1f} against cohort size {len(full)}")
    print(cells.to_string())
    print(f"\nSaved: {weights_path}")
    print(f"Saved: {cells_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
