"""
scripts/k_ablation.py

Agreement as a routing signal against the number of prompts K, from saved
answers. No API calls, no images.

Why this exists. The five-variant signal is concentrated in one variant, which
raises the question of how it behaves with fewer or more prompts and across
families. The saved runs allow that without new inference:

    V3 family    variants 0 to 4 (validation and external)
    V3W family   V3 and four wording-only paraphrases (validation; external once run)
    pooled       the five V3 variants and the four V3W paraphrases, nine prompts on
                 the same patches. V3W's own variant 0 is left out because it is
                 byte-identical to V3's variant 0. The two families ran in different
                 sessions (30 of 1,800 validation labels differ between the two
                 copies of variant 0), so pooled agreement mixes session drift into
                 the signal. The V3W run is re-scored from its LABEL lines
                 (scripts/rescore_labels.py) and the V3 run, which kept no response
                 text, is not, so the pool uses V3W's stored whole-text labels to put
                 all nine prompts under one scoring rule. The V3W family alone uses
                 its re-scored labels.

For every K and every subset of K prompts from a family, agreement (the modal
fraction, ties to the first prompt in pool order) routes the subset's modal
vote, and its gap over random routing of that vote is computed with the
selection weights of the split, as in Table 8. Each K is summarized over its
subsets (mean, range), split by whether variant 2 is in the subset where the
family contains it. Intervals come from a paired bootstrap inside the selection
cells for the mean gap over subsets at each K.

Output: results/consistency/medgemma-27b-it/k_ablation.json

Usage:
    python scripts/k_ablation.py [--n-boot 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.weighted import stratified_bootstrap_indices, weighted_accuracy, weighted_auc

CONS = PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it"
OUT = CONS / "k_ablation.json"


def answers(run: str, as_stored: bool = False) -> tuple[pd.DataFrame, list[list[str]]]:
    df = pd.read_parquet(CONS / run / "predictions.parquet")
    col = "raw_labels_regex" if as_stored and "raw_labels_regex" in df.columns else "raw_labels"
    return df, [json.loads(s) for s in df[col]]


def modal_agree(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    # Same rule as consistency_population.modal_and_agreement.
    modal, agree = [], []
    for sub in rows:
        valid = [lab for lab in sub if lab != "unknown"]
        lab, cnt = Counter(valid).most_common(1)[0] if valid else ("unknown", 0)
        modal.append(lab)
        agree.append(cnt / len(sub))
    return np.array(modal), np.array(agree, dtype=float)


def family_pools(split: str) -> dict[str, tuple[list[str], list[list[str]], pd.DataFrame]]:
    """Prompt names and per-patch answers in pool order, for each family available."""
    suffix = "" if split == "val" else "_test"
    pools = {}
    v3_df, v3 = answers(f"V3{suffix}")
    pools["V3"] = ([f"V3.{k}" for k in range(5)], v3, v3_df)
    if (CONS / f"V3W{suffix}" / "predictions.parquet").exists():
        w_df, w = answers(f"V3W{suffix}")
        if w_df["arr_idx"].tolist() != v3_df["arr_idx"].tolist():
            raise SystemExit(f"V3W{suffix} patches differ from V3{suffix}")
        pools["V3W"] = ([f"V3W.{k}" for k in range(5)], w, v3_df)
        _, w_stored = answers(f"V3W{suffix}", as_stored=True)
        pools["pooled"] = ([f"V3.{k}" for k in range(5)] + [f"V3W.{k}" for k in range(1, 5)],
                           [a + b[1:] for a, b in zip(v3, w_stored)], v3_df)
    return pools


def gap(subset_rows: list[list[str]], true: np.ndarray, w: np.ndarray, ix: np.ndarray) -> float:
    modal, agree = modal_agree([subset_rows[i] for i in ix])
    corr = modal == true[ix]
    return weighted_auc(agree, corr, w[ix]) - weighted_accuracy(corr, w[ix])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"seed={args.seed}  n_boot={args.n_boot}")

    out: dict = {"seed": args.seed, "n_boot": args.n_boot, "splits": {}}
    for split in ("val", "test"):
        weights_dir = "V3" if split == "val" else "V3_test"
        wt = pd.read_parquet(CONS / weights_dir / "weights.parquet")
        w = wt["weight"].to_numpy(float)
        idx_matrix = stratified_bootstrap_indices(wt["cell"].to_numpy(), args.n_boot, args.seed)
        out["splits"][split] = {}
        for fam, (names, rows, df) in family_pools(split).items():
            if wt["arr_idx"].tolist() != df["arr_idx"].tolist():
                raise SystemExit(f"{fam} {split}: weights not aligned")
            true = df["true_label_name"].to_numpy()
            full = np.arange(len(df))
            per_k = {}
            for k in range(2, len(names) + 1):
                subsets = list(combinations(range(len(names)), k))
                sub_rows = [[[r[i] for i in s] for r in rows] for s in subsets]
                points = np.array([gap(sr, true, w, full) for sr in sub_rows])
                accs = []
                for sr in sub_rows:
                    modal, _ = modal_agree(sr)
                    accs.append(weighted_accuracy(modal == true, w))
                boot = np.array([np.mean([gap(sr, true, w, ix) for sr in sub_rows])
                                 for ix in idx_matrix])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                entry = {
                    "n_subsets": len(subsets),
                    "mean_gap": round(float(points.mean()), 4),
                    "mean_gap_ci": [round(float(lo), 4), round(float(hi), 4)],
                    "min_gap": round(float(points.min()), 4),
                    "max_gap": round(float(points.max()), 4),
                    "mean_modal_accuracy": round(float(np.mean(accs)), 4),
                    "subsets": {"+".join(names[i] for i in s): round(float(g), 4)
                                for s, g in zip(subsets, points)},
                }
                if "V3.2" in names:
                    has2 = np.array([names.index("V3.2") in s for s in subsets])
                    if has2.any() and (~has2).any():
                        entry["mean_gap_with_variant_2"] = round(float(points[has2].mean()), 4)
                        entry["mean_gap_without_variant_2"] = round(float(points[~has2].mean()), 4)
                per_k[str(k)] = entry
                print(f"{split:4s} {fam:6s} K={k}: {len(subsets):3d} subsets  mean gap "
                      f"{entry['mean_gap']:+.3f} {entry['mean_gap_ci']}  range "
                      f"[{entry['min_gap']:+.3f}, {entry['max_gap']:+.3f}]  "
                      f"acc {entry['mean_modal_accuracy']:.3f}"
                      + (f"  with/without v2 {entry.get('mean_gap_with_variant_2', float('nan')):+.3f}/"
                         f"{entry.get('mean_gap_without_variant_2', float('nan')):+.3f}"
                         if "mean_gap_with_variant_2" in entry else ""))
            out["splits"][split][fam] = {"prompts": names, "by_k": per_k}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
