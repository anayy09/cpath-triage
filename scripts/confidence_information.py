"""
scripts/confidence_information.py

Direct measurement of how much information the VLM textual confidence carries
about correctness. No API calls: reads the full-scale predictions already on
disk.

Why this exists: the draft inferred "negligible mutual information with
correctness" from the temperature-scaling fit saturating at T=200. That is an
overclaim and is contradicted by the paper's own routing result (Gemma-3
confidence routes above random, so its confidence cannot carry zero information).
Temperature scaling is monotone: it cannot create or destroy ranking
information, so a saturating T only says scaling cannot help, not that the
signal is empty. Here we measure the information directly:

  1. AUROC of confidence as a correctness detector (correct = positive class).
     0.5 = no ranking information; <0.5 = anti-discriminative; >0.5 = usable.
  2. Mutual information between binned confidence and the binary correctness
     outcome, in bits, plus a normalized version (MI / H(correct)).

Reported on the full val and full test sets (the sets routed in Table 3) and on
the held-out evaluation partition (the set used for Table 2 ECE), so the number
lines up with each table.

Outputs:
    results/calibration/confidence_information.json

Usage:
    python scripts/confidence_information.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
N_BINS = 15


def binary_entropy_bits(y: np.ndarray) -> float:
    """Shannon entropy of a binary array, in bits."""
    p = float(y.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def confidence_mi_bits(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Mutual information (bits) between confidence discretized into equal-width
    bins and the binary correctness outcome. sklearn's mutual_info_score is in
    nats; convert to bits.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    mi_nats = mutual_info_score(bin_id, correct.astype(int))
    return float(mi_nats / np.log(2))


def vlm_arrays(model: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) for a VLM on the full val or full test split."""
    suffix = "V3_full" if split == "val" else "V3_full_test"
    df = pd.read_parquet(PROJECT_ROOT / "results" / "zeroshot" / model / suffix / "predictions.parquet")
    conf = df["pred_confidence"].to_numpy(float)
    correct = (df["pred_label"] == df["true_label_name"]).to_numpy(bool)
    return conf, correct


def eval_partition_arrays(model: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the exact held-out evaluation partition used by calibrate.py
    (drop parse failures, np.random.default_rng(42) permutation, last 20%).
    """
    df = pd.read_parquet(PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full" / "predictions.parquet")
    df = df[df["pred_label"] != "unknown"].copy()
    df["correct"] = df["pred_label"] == df["true_label_name"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cal_n = int(len(df) * 0.8)
    ev = df.iloc[idx[cal_n:]]
    return ev["pred_confidence"].to_numpy(float), ev["correct"].to_numpy(bool)


def summarize(conf: np.ndarray, correct: np.ndarray) -> dict:
    y = correct.astype(int)
    h = binary_entropy_bits(correct)
    mi = confidence_mi_bits(conf, correct)
    # AUROC is undefined if only one correctness class is present.
    auroc = float(roc_auc_score(y, conf)) if 0 < y.sum() < len(y) else float("nan")
    return {
        "n": int(len(correct)),
        "accuracy": round(float(correct.mean()), 4),
        "conf_correct_auroc": round(auroc, 4),
        "mi_bits": round(mi, 5),
        "h_correct_bits": round(h, 4),
        "mi_normalized": round(mi / h, 4) if h > 0 else None,
    }


def main() -> int:
    out: dict[str, dict] = {}
    for model in ["medgemma-27b-it", "gemma-3-27b-it"]:
        out[model] = {
            "val_full": summarize(*vlm_arrays(model, "val")),
            "test_full": summarize(*vlm_arrays(model, "test")),
            "eval_partition": summarize(*eval_partition_arrays(model)),
        }
        for part, s in out[model].items():
            print(
                f"{model:16s} {part:15s} n={s['n']:5d}  acc={s['accuracy']:.4f}  "
                f"AUROC(conf->correct)={s['conf_correct_auroc']:.4f}  "
                f"MI={s['mi_bits']:.5f} bits (norm {s['mi_normalized']})"
            )

    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confidence_information.json"
    with open(out_path, "w") as f:
        json.dump({"n_bins": N_BINS, "seed": SEED, "results": out}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
