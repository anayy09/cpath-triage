"""
scripts/cross_center_scalings.py

Class distributions and the competing rescalings of the cross-cohort drop
(revision items C-07 and C-15d).

Reviewer 1's point is that "both VLMs degrade less across centers than either
CNN" is true on one scale and false on others, and that the paper picked the
flattering one without saying so. All four scalings are computed here so the
manuscript can state which is which:

  percentage-point drop   val accuracy minus test accuracy. VLMs look best.
  relative drop           the same divided by val accuracy. CNN-64 looks best.
  above-chance retained   (test - chance) / (val - chance). CNN-64 looks best.
  error ratio             test error rate over val error rate. VLMs look best,
                          decisively, which is the floor effect stated plainly.

Two chance levels are reported, because the paper's 11.1% uniform baseline is not
the right reference for an unbalanced split. The majority-class rate is 14.3% on
validation and 18.6% on the external test set, and a claim of being "well above
chance" should be measured against the one a reader would use.

Outputs:
    results/stage6/cross_center_scalings.json

Usage:
    python scripts/cross_center_scalings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_ABBREV, LABEL_NAMES

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
ABBREV = {LABEL_NAMES[i]: LABEL_ABBREV[i] for i in range(len(LABEL_NAMES))}
UNIFORM_CHANCE = 1.0 / 9.0


def model_accuracy(model: str, split: str) -> float:
    if model.startswith("resnet18"):
        df = pd.read_parquet(
            PROJECT_ROOT / "results" / "cnn" / model / f"{split}_predictions.parquet"
        )
        return float(df["correct"].mean())
    suffix = "V3_full" if split == "val" else "V3_full_test"
    df = pd.read_parquet(
        PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet"
    )
    return float((df["pred_label"] == df["true_label_name"]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Unused; logged for consistency.")
    args = parser.parse_args()

    summary = json.loads(
        (PROJECT_ROOT / "results" / "data_summary.json").read_text(encoding="utf-8")
    )["splits"]

    distributions = {}
    majority = {}
    for split in ("val", "test"):
        counts = {k: v for k, v in summary[split].items() if k != "_total"}
        total = summary[split]["_total"]
        rows = {
            ABBREV[name]: {
                "name": name,
                "n": int(n),
                "share": round(n / total, 4),
            }
            for name, n in counts.items()
        }
        distributions[split] = {
            "n_total": int(total),
            "classes": dict(sorted(rows.items(), key=lambda kv: -kv[1]["n"])),
        }
        top = max(counts.items(), key=lambda kv: kv[1])
        majority[split] = {
            "class": ABBREV[top[0]],
            "name": top[0],
            "n": int(top[1]),
            "rate": round(top[1] / total, 4),
        }

    models = {
        "MedGemma-27b-it": "medgemma-27b-it",
        "Gemma-3-27b-it": "gemma-3-27b-it",
        "ResNet-18 64px": "resnet18_64px",
        "ResNet-18 224px": "resnet18_224px",
    }

    scalings = {}
    for pretty, key in models.items():
        val_acc = model_accuracy(key, "val")
        test_acc = model_accuracy(key, "test")
        drop_pp = val_acc - test_acc

        row = {
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "drop_percentage_points": round(drop_pp, 4),
            "relative_drop": round(drop_pp / val_acc, 4),
            "error_ratio_test_over_val": (
                round((1 - test_acc) / (1 - val_acc), 3) if val_acc < 1.0 else None
            ),
        }
        for label, chance_val, chance_test in (
            ("uniform_1_of_9", UNIFORM_CHANCE, UNIFORM_CHANCE),
            ("majority_class", majority["val"]["rate"], majority["test"]["rate"]),
        ):
            num = test_acc - chance_test
            den = val_acc - chance_val
            row[f"above_chance_retained_{label}"] = (
                round(num / den, 4) if den > 0 else None
            )
            row[f"test_above_chance_{label}"] = round(num, 4)
        scalings[pretty] = row

    ranks = {}
    for metric, better in (
        ("drop_percentage_points", "lower"),
        ("relative_drop", "lower"),
        ("above_chance_retained_uniform_1_of_9", "higher"),
        ("above_chance_retained_majority_class", "higher"),
        ("error_ratio_test_over_val", "lower"),
    ):
        vals = [(m, r[metric]) for m, r in scalings.items() if r.get(metric) is not None]
        vals.sort(key=lambda kv: kv[1], reverse=(better == "higher"))
        ranks[metric] = {"better_is": better, "order_best_to_worst": [m for m, _ in vals]}

    out = {
        "seed": args.seed,
        "class_distributions": distributions,
        "majority_class_baseline": majority,
        "uniform_chance": round(UNIFORM_CHANCE, 4),
        "scalings": scalings,
        "ranking_by_scaling": ranks,
        "note": (
            "The ordering of models reverses between the percentage-point scale and "
            "the relative-drop and above-chance-retained scales. The error-ratio "
            "column shows why: the VLMs start near the floor, so a fixed "
            "percentage-point loss is a much smaller multiple of their error rate "
            "than of a CNN's."
        ),
    }

    out_dir = PROJECT_ROOT / "results" / "stage6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_center_scalings.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Class distribution ===")
    for split in ("val", "test"):
        cls = distributions[split]["classes"]
        top3 = ", ".join(f"{k} {v['share']:.1%}" for k, v in list(cls.items())[:3])
        print(f"  {split:5s} n={distributions[split]['n_total']:6d}  majority "
              f"{majority[split]['class']} {majority[split]['rate']:.2%}   top: {top3}")
    print(f"  uniform chance: {UNIFORM_CHANCE:.2%}")

    print("\n=== Cross-cohort drop under four scalings ===")
    print(f"{'model':<18}{'val':>8}{'test':>8}{'drop pp':>9}{'rel':>8}"
          f"{'abv-ch(1/9)':>13}{'abv-ch(maj)':>13}{'err ratio':>11}")
    for m, r in scalings.items():
        print(f"{m:<18}{r['val_accuracy']:>8.4f}{r['test_accuracy']:>8.4f}"
              f"{r['drop_percentage_points']:>9.4f}{r['relative_drop']:>8.3f}"
              f"{r['above_chance_retained_uniform_1_of_9']:>13.3f}"
              f"{r['above_chance_retained_majority_class']:>13.3f}"
              f"{r['error_ratio_test_over_val']:>11.2f}")

    print("\n=== Who looks best under each scaling ===")
    for metric, r in ranks.items():
        print(f"  {metric:<38} {' > '.join(r['order_best_to_worst'])}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
