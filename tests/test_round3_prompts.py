"""
Guards for the round-3 consistency runs: the V3W family really is wording-only
against V3, the shared K=5 summary reproduces the stored V3 run exactly, and the
image-control constructions do what their docstrings say. No API, no image data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.k5 import summarize_k
from src.models.prompts import (
    _V3_FORMAT_BLOCK,
    TISSUE_CLASSES,
    get_prompt,
    get_v3w_variants,
)

V3_RUN = ROOT / "results" / "consistency" / "medgemma-27b-it" / "V3" / "predictions.parquet"


def test_v3w_variant0_is_v3_byte_identical() -> None:
    assert get_v3w_variants()[0] == get_prompt("tissue_classification", version="V3")


def test_v3w_shares_output_block_and_class_order() -> None:
    # V3 lists the classes in TISSUE_CLASSES order; every member must too.
    for p in get_v3w_variants():
        assert p.endswith(_V3_FORMAT_BLOCK)
        # Same blank context slot as V3, so the gap before the format block matches.
        assert p.endswith("\n\n\n\n" + _V3_FORMAT_BLOCK)
        positions = [p.index(f"- {c}:") for c in TISSUE_CLASSES]
        assert positions == sorted(positions)
    assert len(set(get_v3w_variants())) == 5


@pytest.mark.skipif(not V3_RUN.exists(), reason="stored V3 K=5 run not present")
def test_summarize_k_reproduces_stored_v3_run() -> None:
    df = pd.read_parquet(V3_RUN)
    for _, r in df.iterrows():
        s = summarize_k(json.loads(r.raw_labels), json.loads(r.raw_confs), r.true_label_name, 5)
        assert s["modal_label"] == r.modal_label
        assert s["consistency_score"] == r.consistency_score
        assert s["is_correct"] == r.is_correct
        # raw_confs were stored rounded to 3 dp, mean_textual_conf from the unrounded values.
        assert abs(s["mean_textual_conf"] - r.mean_textual_conf) < 1e-3


@pytest.mark.skipif(not V3_RUN.exists(), reason="stored V3 K=5 run not present")
def test_swap_pairs_are_cross_class_and_one_to_one() -> None:
    from run_image_controls import swap_pairs

    pairs = swap_pairs(pd.read_parquet(V3_RUN), seed=42)
    assert len(pairs) == 360
    assert (pairs.recipient_true_idx != pairs.donor_true_idx).all()
    assert set(pairs.recipient_arr_idx) == set(pairs.donor_arr_idx)
    assert pairs.groupby("recipient_true_idx").size().eq(40).all()


def test_shuffle_keeps_colour_multiset_and_is_seeded() -> None:
    from run_image_controls import shuffle_pixels

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    a, b = shuffle_pixels(arr, 7, 42), shuffle_pixels(arr, 7, 42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, shuffle_pixels(arr, 8, 42))
    flat = lambda x: sorted(map(tuple, x.reshape(-1, 3)))
    assert flat(a) == flat(arr)
