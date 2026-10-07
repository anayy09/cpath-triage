"""
scripts/figures/extract_fig1_patch.py

Write one real validation patch for Figure 1, so the schematic shows dataset
tissue rather than a rendered image. Runs on the machine that holds PathMNIST.

The patch is drawn from validation patches whose true class is TUM, which
MedGemma, in the stored full-scale V3 run, labelled TUM at verbalized confidence
0.95, and which in the stored K=5 run got TUM from four of the five variants, so
both example outputs in the figure ("TUM, 95" and "4 of 5 agree, score 0.80")
are what the model actually returned for that patch. The default pick is seeded; --contact-sheet
writes a grid of candidates for choosing one by eye, and --arr-idx fixes it.

The array is saved unmodified at its native 224 px. Licence and source go in a
JSON sidecar that the figure caption and Data availability can cite.

Outputs:
    scripts/figures/raw/fig1_patch_tum.png
    scripts/figures/raw/fig1_patch_tum.json
    scripts/figures/raw/fig1_candidates_tum.png   (with --contact-sheet)

Usage:
    python scripts/figures/extract_fig1_patch.py --list
    python scripts/figures/extract_fig1_patch.py --contact-sheet 24
    python scripts/figures/extract_fig1_patch.py --arr-idx 1234
    python scripts/figures/extract_fig1_patch.py            # seeded default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TUM = "colorectal adenocarcinoma epithelium"
PRED = PROJECT_ROOT / "results" / "zeroshot" / "medgemma-27b-it" / "V3_full" / "predictions.parquet"
K5 = PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it" / "V3" / "predictions.parquet"
RAW = PROJECT_ROOT / "scripts" / "figures" / "raw"


def candidates() -> pd.DataFrame:
    # Figure 1 also shows a K=5 example ("4 of 5 agree, score 0.80"), so the
    # patch must be in the consistency subset with that outcome as well; then
    # both example outputs in the figure are what MedGemma returned for it.
    df = pd.read_parquet(PRED)
    c = df[(df.true_label_name == TUM) & (df.pred_label == TUM) & np.isclose(df.pred_confidence, 0.95)]
    k5 = pd.read_parquet(K5)
    k5 = k5[(k5.modal_label == TUM) & np.isclose(k5.consistency_score, 0.8)]
    c = c.merge(k5[["arr_idx", "consistency_score", "raw_labels"]], on="arr_idx")
    return c.sort_values("arr_idx").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arr-idx", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="Print candidate count and the default pick; no data.")
    parser.add_argument("--contact-sheet", type=int, default=0, metavar="N")
    args = parser.parse_args()

    if not PRED.exists():
        print(f"Stored predictions not found: {PRED}", file=sys.stderr)
        return 1
    cand = candidates()
    rng = np.random.default_rng(args.seed)
    default = int(cand.arr_idx.iloc[int(rng.integers(len(cand)))])
    arr_idx = args.arr_idx if args.arr_idx is not None else default
    print(f"seed={args.seed}  candidates={len(cand)}  default arr_idx={default}  chosen={arr_idx}")
    if arr_idx not in set(cand.arr_idx):
        print(f"arr_idx {arr_idx} is not a TUM patch labelled TUM at 0.95 in {PRED} "
              f"with a 4-of-5 TUM modal vote in {K5}", file=sys.stderr)
        return 1
    if args.list:
        return 0

    from src.data.pathmnist import arr_to_pil, load_split_arrays

    images, labels = load_split_arrays("val")
    RAW.mkdir(parents=True, exist_ok=True)

    if args.contact_sheet:
        from PIL import Image, ImageDraw

        picks = cand.arr_idx.to_numpy()[rng.permutation(len(cand))[: args.contact_sheet]]
        cols = 6
        rows = -(-len(picks) // cols)
        sheet = Image.new("RGB", (cols * 224, rows * 240), "white")
        draw = ImageDraw.Draw(sheet)
        for i, a in enumerate(picks):
            x, y = (i % cols) * 224, (i // cols) * 240
            sheet.paste(arr_to_pil(images[int(a)]), (x, y))
            draw.text((x + 4, y + 225), f"arr_idx {int(a)}", fill="black")
        path = RAW / "fig1_candidates_tum.png"
        sheet.save(path)
        print(f"Saved: {path}")
        return 0

    row = cand[cand.arr_idx == arr_idx].iloc[0]
    assert int(labels[arr_idx]) == 8, "label array disagrees with stored predictions"
    png = RAW / "fig1_patch_tum.png"
    arr_to_pil(images[arr_idx]).save(png, format="PNG")
    meta = {
        "file": png.name,
        "arr_idx": int(arr_idx),
        "split": "val (PathMNIST 224 px, MedMNIST+)",
        "true_class": TUM,
        "medgemma_v3_prediction": row.pred_label,
        "medgemma_v3_confidence": float(row.pred_confidence),
        "medgemma_v3_k5_labels": json.loads(row.raw_labels),
        "medgemma_v3_k5_agreement": float(row.consistency_score),
        "pixels": "native 224 x 224, unmodified",
        "source": "NCT-CRC-HE-100K (Kather, Halama and Marx 2018, doi:10.5281/zenodo.1214456), "
                  "distributed as PathMNIST in MedMNIST v2 (Yang et al. 2023) and MedMNIST+",
        "licence": "CC BY 4.0",
        "selection": f"seeded pick (seed {args.seed}) among {len(cand)} candidates"
                     if args.arr_idx is None else "chosen by arr_idx",
    }
    side = RAW / "fig1_patch_tum.json"
    side.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {png}\nSaved: {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
