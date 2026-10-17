"""
scripts/pilot_per_class.py

Per-class F1, recall and prediction share of MedGemma under each prompt version
of the 450-patch pilot (50 validation patches per class, the same patches for
every version). No API calls, no images.

Why this exists. MedGemma's F1 is near zero for mucus, normal mucosa and stroma
under V3, and it over-predicts smooth muscle. Whether that is fixed in the model
or moves with the prompt is answerable from the pilot, which ran five prompt
versions on identical patches: a class that collapses under every version points
at the model, a class that recovers under one points at the prompt.

Output: results/zeroshot/medgemma-27b-it/pilot_per_class.json

Usage:
    python scripts/pilot_per_class.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_ABBREV, LABEL_NAMES

ZS = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it"
OUT = ZS / "pilot_per_class.json"
VERSIONS = ("V2", "V3", "V4", "V4B", "V5")


def main() -> int:
    classes = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    out: dict = {"design": "450 validation patches, 50 per class, identical across versions",
                 "versions": {}}
    ref = None
    for v in VERSIONS:
        df = pd.read_parquet(ZS / v / "predictions.parquet")
        idx = sorted(df["arr_idx"].tolist())
        if ref is None:
            ref = idx
        elif idx != ref:
            raise SystemExit(f"{v} pilot patches differ from {VERSIONS[0]}")
        true, pred = df["true_label_name"].to_numpy(), df["pred_label"].to_numpy()
        per = {}
        for c in classes:
            tp = int(((pred == c) & (true == c)).sum())
            n_pred, n_true = int((pred == c).sum()), int((true == c).sum())
            prec = tp / n_pred if n_pred else 0.0
            rec = tp / n_true if n_true else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            per[LABEL_ABBREV[classes.index(c)]] = {
                "f1": round(f1, 3), "recall": round(rec, 3),
                "prediction_share": round(n_pred / len(df), 3)}
        out["versions"][v] = {"accuracy": round(float((pred == true).mean()), 4),
                              "macro_f1": round(float(np.mean([p["f1"] for p in per.values()])), 4),
                              "per_class": per}
    abbrevs = list(out["versions"]["V3"]["per_class"])
    print("F1        " + " ".join(f"{a:>5s}" for a in abbrevs) + "   MUS share")
    for v, d in out["versions"].items():
        print(f"{v:4s} {d['macro_f1']:.3f} " + " ".join(f"{d['per_class'][a]['f1']:5.2f}" for a in abbrevs)
              + f"   {d['per_class']['MUS']['prediction_share']:.2f}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
