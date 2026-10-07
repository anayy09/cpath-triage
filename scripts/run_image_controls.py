"""
scripts/run_image_controls.py

Image controls for the K=5 consistency signal (reviewer item R3.12). MedGemma
only, the V3 K=5 variant family (the one behind the published consistency
results), validation split, billed calls through src/models/client.py.

The five variant prompts are the same for every patch, so the image is the only
per-patch input. Three controls follow from that:

  no-image   Each variant sent with no image, and separately with a uniform
             mid-grey 224 px image, 10 times each (2 x 5 x 10 = 100 calls).
             With the image gone every patch receives the same input, so if the
             outputs are identical across repeats, agreement is a constant and
             carries no per-patch information. This checks determinism and shows
             what the prompts alone produce.
  swap       A class-balanced sample of 360 of the 1,800 subset patches (40 per
             class, seed 42). Each recipient is sent a donor image from a
             different true class: classes are paired by a random derangement and
             donors within a class are shuffled, so every donor is used once
             (360 x 5 = 1,800 calls). If the answers follow the image they
             reproduce the donor's stored labels, not the recipient's.
  shuffle    All 1,800 subset patches with each image's pixel positions randomly
             permuted (RGB triplets kept, per-patch seed from arr_idx), which
             keeps the colour histogram and removes all morphology (9,000 calls).
             If agreement still separates correct from incorrect, the separation
             is carried by colour statistics rather than tissue structure.

Images are the stored uint8 224 px arrays, PNG-encoded with arr_to_pil exactly as
run_consistency.py sent them. Calls run one at a time, temperature 0, max_tokens
128. Every completed call is appended to calls.jsonl, so re-running the same
command resumes; failed calls are retried on the next run.

Outputs (results/consistency/{model}/controls/{no_image,swap,shuffle}/):
    calls.jsonl, run_meta.json, plus
    no_image/outputs.parquet, swap/pairs.csv, swap/predictions.parquet,
    shuffle/predictions.parquet (V3 format, so consistency_routing_table-style
    analysis applies). scripts/analyze_image_controls.py summarises all three.

Usage:
    python scripts/run_image_controls.py no-image --dry-run
    python scripts/run_image_controls.py swap --dry-run
    python scripts/run_image_controls.py shuffle --dry-run
    python scripts/run_image_controls.py shuffle --limit 4      # smoke test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_consistency_v3w import endpoint_meta, git_commit, prompt_hashes

from src.data.pathmnist import LABEL_NAMES
from src.models.k5 import (
    CallLog,
    Job,
    collect,
    log_time_window,
    run_jobs,
    summarize_k,
    utc_now,
    write_run_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("controls")

K = 5
N_REPEATS_NO_IMAGE = 10
SWAP_PER_CLASS = 40
GREY = 128
SIDE = 224


def v3_variants() -> list[str]:
    """The V3 K=5 family, built by the same function that built the stored run."""
    from run_consistency import _build_variants

    return _build_variants("V3")


def shuffle_pixels(arr: np.ndarray, arr_idx: int, seed: int) -> np.ndarray:
    """Permute pixel positions, keeping each RGB triplet, with a per-patch seed."""
    rng = np.random.default_rng([seed, int(arr_idx)])
    flat = arr.reshape(-1, arr.shape[-1])
    return flat[rng.permutation(flat.shape[0])].reshape(arr.shape)


def swap_pairs(subset: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    40 recipients per class, each paired with a donor of a different class.

    A random derangement of the nine classes decides which class donates to
    which; donors within the donor class are shuffled. Every sampled patch is a
    recipient once and a donor once, and no pair shares a true class.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(LABEL_NAMES)
    by_class = {
        c: rng.choice(
            subset.loc[subset.true_label_idx == c, "arr_idx"].to_numpy(), SWAP_PER_CLASS, replace=False
        )
        for c in range(n_classes)
    }
    while True:
        sigma = rng.permutation(n_classes)
        if not np.any(sigma == np.arange(n_classes)):
            break
    rows = []
    for c in range(n_classes):
        donors = rng.permutation(by_class[int(sigma[c])])
        for r_idx, d_idx in zip(by_class[c], donors):
            rows.append({
                "recipient_arr_idx": int(r_idx), "recipient_true_idx": c,
                "recipient_true_name": LABEL_NAMES[c],
                "donor_arr_idx": int(d_idx), "donor_true_idx": int(sigma[c]),
                "donor_true_name": LABEL_NAMES[int(sigma[c])],
            })
    out = pd.DataFrame(rows)
    assert (out.recipient_true_idx != out.donor_true_idx).all()
    assert out.donor_arr_idx.is_unique and out.recipient_arr_idx.is_unique
    return out


def grey_png(path: Path) -> None:
    from src.data.pathmnist import arr_to_pil

    arr_to_pil(np.full((SIDE, SIDE, 3), GREY, dtype=np.uint8)).save(path, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", choices=["no-image", "swap", "shuffle"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="First N patches or pairs only (smoke test).")
    parser.add_argument("--dry-run", action="store_true", help="Counts, a sample prompt and paths; no data, no API.")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    model = args.model or os.environ.get("MEDGEMMA_MODEL", "medgemma-27b-it")
    slug = model.replace("/", "_").replace(":", "_")
    base = PROJECT_ROOT / "results" / "consistency" / slug
    subset_path = base / "V3" / "predictions.parquet"
    out_dir = base / "controls" / args.control.replace("-", "_")
    log = CallLog(out_dir / "calls.jsonl")

    if not subset_path.exists():
        print(f"Stored V3 subset not found: {subset_path}", file=sys.stderr)
        return 1
    subset = pd.read_parquet(subset_path)
    variants = v3_variants()
    assert len(variants) == K

    pairs = None
    if args.control == "no-image":
        jobs = [
            Job(key=cond, variant=v, repeat=r)
            for cond in ("text_only", "grey")
            for v in range(K)
            for r in range(N_REPEATS_NO_IMAGE)
        ]
    elif args.control == "swap":
        pairs = swap_pairs(subset, args.seed)
        if args.limit:
            pairs = pairs.head(args.limit)
        jobs = [Job(key=str(int(r)), variant=v) for r in pairs.recipient_arr_idx for v in range(K)]
    else:
        rows = subset if not args.limit else subset.head(args.limit)
        jobs = [Job(key=str(int(a)), variant=v) for a in rows.arr_idx for v in range(K)]

    n_done = sum(log.done(j) for j in jobs)
    logger.info("control=%s seed=%d model=%s", args.control, args.seed, model)

    if args.dry_run:
        print(f"\n=== {args.control} dry run (no data loaded, no API calls) ===")
        print(f"calls total:     {len(jobs):,}")
        print(f"already logged:  {n_done:,}")
        print(f"calls to make:   {len(jobs) - n_done:,}")
        print(f"prompt sha256/16 (V3 K=5 family): {prompt_hashes(variants)}")
        print(f"outputs under:   {out_dir}")
        if pairs is not None:
            print(f"pairs:           {len(pairs)} recipients; class derangement "
                  f"{sorted(set(zip(pairs.recipient_true_name, pairs.donor_true_name)))}")
            print(pairs.head(3).to_string(index=False))
        if args.control == "shuffle":
            demo = np.arange(SIDE * SIDE * 3, dtype=np.int64).reshape(SIDE, SIDE, 3)
            sh = shuffle_pixels(demo, int(subset.arr_idx.iloc[0]), args.seed)
            ok = np.array_equal(np.sort(sh.reshape(-1, 3), axis=0), np.sort(demo.reshape(-1, 3), axis=0))
            print(f"pixel shuffle keeps the multiset of RGB triplets: {ok}; "
                  f"positions moved: {float((sh != demo).any(axis=-1).mean()):.4f}")
        print(f"\n----- variant 0 (sample) -----\n{variants[0]}")
        return 0

    from src.data.pathmnist import arr_to_pil, load_split_arrays
    from src.models.client import Client

    client = Client(model=model)
    meta = {
        "script": "scripts/run_image_controls.py",
        "control": args.control,
        "family": "V3 K=5",
        "model": model,
        "seed": args.seed,
        "split": "val",
        "prompt_sha256_16": prompt_hashes(variants),
        "temperature": 0.0,
        "max_tokens": 128,
        "git_commit": git_commit(),
        "started_utc": utc_now(),
        "endpoint": endpoint_meta(client, model),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(out_dir / "run_meta.json", meta)
    if pairs is not None:
        pairs.to_csv(out_dir / "pairs.csv", index=False)

    images = None
    if args.control != "no-image":
        logger.info("Loading val images...")
        images, _ = load_split_arrays("val")
        donor_of = dict(zip(pairs.recipient_arr_idx, pairs.donor_arr_idx)) if pairs is not None else {}

    with tempfile.TemporaryDirectory(prefix="cpath_ctrl_") as tmpdir:
        tile = Path(tmpdir) / "patch.png"
        grey = Path(tmpdir) / "grey.png"
        grey_png(grey)
        current = {"key": None}

        def call(job: Job):
            if args.control == "no-image":
                if job.key == "text_only":
                    return client.classify_text_only(variants[job.variant], task="tissue_classification",
                                                     model=model, max_tokens=128)
                path = grey
            else:
                if current["key"] != job.key:
                    a = int(job.key)
                    if args.control == "swap":
                        arr = images[donor_of[a]]
                    else:
                        arr = shuffle_pixels(images[a], a, args.seed)
                    arr_to_pil(arr).save(tile, format="PNG")
                    current["key"] = job.key
                path = tile
            return client.analyze_tiles(
                tile_paths=[path], clinical_context="", task="tissue_classification",
                prompt_text=variants[job.variant], model=model, max_tokens=128,
            )

        made, failed = run_jobs(jobs, call, log, logger=logger)
    logger.info("made=%d failed=%d", made, failed)

    missing = sum(not log.done(j) for j in jobs)
    meta.update({"finished_utc": utc_now(), "call_window": log_time_window(log), "calls_missing": missing})
    write_run_meta(out_dir / "run_meta.json", meta)
    if missing and not args.allow_incomplete:
        print(f"{missing} calls missing; re-run the same command to resume.", file=sys.stderr)
        return 2

    if args.control == "no-image":
        out = pd.DataFrame([log.records[j.log_key] for j in jobs if log.done(j)])
        path = out_dir / "outputs.parquet"
        out.to_parquet(path, index=False)
        print(f"Saved: {path}")
        return 0

    rows = []
    if args.control == "swap":
        for _, p in pairs.iterrows():
            got = collect(log, str(int(p.recipient_arr_idx)), K)
            if got is None:
                continue
            labels, confs = got
            s = summarize_k(labels, confs, p.donor_true_name, K)
            s["is_correct_vs_donor"] = s.pop("is_correct")
            s["is_correct_vs_recipient"] = bool(s["modal_label"] == p.recipient_true_name)
            rows.append({**p.to_dict(), **s})
    else:
        sel = subset if not args.limit else subset.head(args.limit)
        for _, r in sel.iterrows():
            got = collect(log, str(int(r.arr_idx)), K)
            if got is None:
                continue
            labels, confs = got
            rows.append({
                "patch_index": int(r.patch_index), "arr_idx": int(r.arr_idx),
                "true_label_idx": int(r.true_label_idx), "true_label_name": r.true_label_name,
                **summarize_k(labels, confs, r.true_label_name, K),
            })
    path = out_dir / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Saved: {path}")
    print(f"Saved: {out_dir / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
