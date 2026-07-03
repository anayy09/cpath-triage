"""
scripts/generate_report.py

Stage 7: generate sample preliminary pathology reports from patch predictions.

Reads MedGemma V3 val predictions, constructs three representative "cases"
(one predominantly TUM, one predominantly NORM, one mixed/uncertain), and
generates a report narrative for each using llama-3.3-70b-instruct.

Outputs:
    results/stage7/sample_reports.json   -- all three report texts
    results/stage7/sample_reports.txt    -- plain-text version

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration import TemperatureScaler
from src.report.generate import build_slide_result, generate_report

OUT_DIR = PROJECT_ROOT / "results" / "stage7"
VAL_PRED = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3" / "predictions.parquet"
CAL_FILE = PROJECT_ROOT / "results" / "calibration" / "medgemma-27b-it" / "V3" / "calibration_params.json"


def _make_case(df: pd.DataFrame, indices, case_id: str) -> tuple:
    """Select rows by position indices and return (subset_df, class_distribution)."""
    sub = df.iloc[indices].copy()
    dist = dict(Counter(sub["pred_label"].tolist()).most_common())
    return sub, dist


def main() -> int:
    if not VAL_PRED.exists():
        print(f"Missing: {VAL_PRED}")
        return 1

    with open(CAL_FILE) as f:
        T = json.load(f)["T"]

    df = pd.read_parquet(VAL_PRED)
    df["correct"] = df["pred_label"] == df["true_label_name"]
    raw_conf = df["pred_confidence"].to_numpy()
    scaler = TemperatureScaler()
    scaler.T = T
    df["cal_confidence"] = scaler.transform_scalar(raw_conf)
    df["uncertainty"] = 1.0 - df["cal_confidence"]

    # Case 1: patches predicted as TUM (tumor-rich slide)
    tum_idx = df.index[df["pred_label"] == "colorectal adenocarcinoma epithelium"][:12].tolist()
    tum_idx = df.index.get_indexer(tum_idx).tolist()

    # Case 2: patches predicted as NORM (normal colon slide)
    norm_idx = df.index[df["pred_label"] == "normal colon mucosa"][:12].tolist()
    norm_idx = df.index.get_indexer(norm_idx).tolist()
    if not norm_idx:  # NORM may be rarely predicted; fall back to LYM
        norm_idx = df.index[df["pred_label"] == "lymphocytes"][:12].tolist()
        norm_idx = df.index.get_indexer(norm_idx).tolist()

    # Case 3: high-uncertainty / mixed patches (those that would be routed)
    uncertain_idx = df.sort_values("uncertainty", ascending=False).head(12).index
    uncertain_idx = df.index.get_indexer(uncertain_idx).tolist()

    cases = [
        (tum_idx,      "CASE-TUM-001", "Tumor-rich case"),
        (norm_idx,     "CASE-NRM-001", "Normal/benign case"),
        (uncertain_idx,"CASE-UNC-001", "High-uncertainty (routed) case"),
    ]

    reports = {}
    lines = []

    for idxs, case_id, description in cases:
        if not idxs:
            print(f"No patches for {case_id}, skipping.")
            continue
        sub_df, dist = _make_case(df, idxs, case_id)
        result = build_slide_result(sub_df, case_id=case_id)

        print(f"\nGenerating report for {case_id} ({description})...")
        print(f"  Patch distribution: {dict(list(dist.items())[:4])}")

        try:
            report_text = generate_report(result, class_distribution=dist)
        except Exception as exc:
            report_text = f"[ERROR: {exc}]"

        reports[case_id] = {
            "description": description,
            "n_patches": len(sub_df),
            "class_distribution": dist,
            "report": report_text,
        }
        lines.append(f"=== {case_id} ({description}) ===\n{report_text}\n")
        print(f"\n{report_text}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sample_reports.json", "w") as f:
        json.dump(reports, f, indent=2)
    (OUT_DIR / "sample_reports.txt").write_text("\n".join(lines))

    print(f"\nReports saved: {OUT_DIR / 'sample_reports.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
