"""
scripts/run_consistency_v3w.py

K=5 consistency run under the V3W family: the V3 prompt plus four wording-only
rewordings of it (src/models/prompts.get_v3w_variants; texts and notes in
docs/V3W_PARAPHRASES.md). Billed: 1,800 patches x 5 variants = 9,000 calls.

Why this exists: the original V3 family mixes several interventions (a
V4-style structural variant, a clinical context, a confidence instruction), so
agreement across it cannot be attributed to rephrasing. V3W changes wording only.

The patch set is read from the stored V3 run rather than reselected, so the two
runs cover the same 1,800 validation patches in the same order and can be
compared patch by patch. Variant 0 is the unmodified V3 prompt, which makes it a
same-session drift probe against the stored V3 variant-0 labels.

Calls are made one at a time, patch-major, through src/models/client.py, exactly
as run_consistency.py made them (PNG of the stored uint8 224 px array,
temperature 0, max_tokens 128). Each completed call is appended to calls.jsonl;
re-running the same command resumes. predictions.parquet is written only once
every call is logged, in the format of the V3 run, so
scripts/consistency_routing_table.py --version V3W reads it unchanged.

Outputs (results/consistency/{model}/V3W/):
    calls.jsonl          one line per completed call
    predictions.parquet  per-patch K=5 summary, V3 format
    metrics.json         written by run_consistency._compute_and_save_metrics
    run_meta.json        endpoint, model listing, prompt hashes, call window, git commit

Usage:
    python scripts/run_consistency_v3w.py --dry-run
    python scripts/run_consistency_v3w.py
    python scripts/run_consistency_v3w.py --limit 4      # smoke test, 20 calls
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
from src.models.prompts import get_v3w_variants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v3w")

K = 5


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prompt_hashes(prompts: list[str]) -> list[str]:
    return [hashlib.sha256(p.encode("utf-8")).hexdigest()[:16] for p in prompts]


def endpoint_meta(client, model: str) -> dict:
    """What can be recorded about the serving configuration. The endpoint exposes no version."""
    from urllib.parse import urlparse

    try:
        listing = [m for m in client.list_models() if m.get("id") == model]
    except Exception as exc:
        listing = [{"error": str(exc)[:200]}]
    return {"base_url_host": urlparse(os.environ.get("BASE_URL", "")).netloc, "models_entry": listing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Logged only; patch set and prompts are fixed.")
    parser.add_argument(
        "--patches-from", type=Path, default=None,
        help="Stored K=5 predictions whose patch list is reused (default: the V3 validation run).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only the first N patches (smoke test).")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, prompts and paths; no data, no API.")
    parser.add_argument("--show-prompts", action="store_true", help="With --dry-run, print all five prompts.")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Aggregate even if calls are missing, scoring them as run_consistency did (unknown, 0.5).",
    )
    args = parser.parse_args()

    model = args.model or os.environ.get("MEDGEMMA_MODEL", "medgemma-27b-it")
    slug = model.replace("/", "_").replace(":", "_")
    patches_from = args.patches_from or PROJECT_ROOT / "results" / "consistency" / slug / "V3" / "predictions.parquet"
    out_dir = PROJECT_ROOT / "results" / "consistency" / slug / "V3W"
    log_path = out_dir / "calls.jsonl"

    if not patches_from.exists():
        print(f"Patch list not found: {patches_from}", file=sys.stderr)
        return 1
    patches = pd.read_parquet(patches_from)[["arr_idx", "true_label_idx", "true_label_name"]]
    if args.limit:
        patches = patches.head(args.limit)
    patches = patches.reset_index(drop=True)

    variants = get_v3w_variants()
    assert len(variants) == K
    jobs = [Job(key=str(int(a)), variant=v) for a in patches["arr_idx"] for v in range(K)]
    log = CallLog(log_path)
    n_done = sum(log.done(j) for j in jobs)

    logger.info("seed=%d model=%s patches=%d (from %s)", args.seed, model, len(patches), patches_from)
    if args.dry_run:
        print("\n=== V3W dry run (no data loaded, no API calls) ===")
        print(f"model:            {model}")
        print(f"patches:          {len(patches)} from {patches_from}")
        print(f"calls total:      {len(jobs):,}  ({len(patches)} x {K})")
        print(f"already logged:   {n_done:,}")
        print(f"calls to make:    {len(jobs) - n_done:,}")
        print(f"prompt sha256/16: {prompt_hashes(variants)}")
        print(f"outputs:          {log_path}\n                  {out_dir / 'predictions.parquet'}")
        print(f"                  {out_dir / 'metrics.json'}\n                  {out_dir / 'run_meta.json'}")
        if args.show_prompts:
            for i, p in enumerate(variants):
                print(f"\n----- variant {i} -----\n{p}")
        else:
            print(f"\n----- variant 1 (sample) -----\n{variants[1]}")
        return 0

    from src.data.pathmnist import arr_to_pil, load_split_arrays
    from src.models.client import Client

    client = Client(model=model)
    meta = {
        "script": "scripts/run_consistency_v3w.py",
        "family": "V3W",
        "model": model,
        "seed": args.seed,
        "split": "val",
        "n_patches": len(patches),
        "k": K,
        "patches_from": str(patches_from.relative_to(PROJECT_ROOT)) if patches_from.is_relative_to(PROJECT_ROOT) else str(patches_from),
        "prompt_sha256_16": prompt_hashes(variants),
        "temperature": 0.0,
        "max_tokens": 128,
        "git_commit": git_commit(),
        "started_utc": utc_now(),
        "endpoint": endpoint_meta(client, model),
    }
    write_run_meta(out_dir / "run_meta.json", meta)

    logger.info("Loading val images...")
    images, _ = load_split_arrays("val")

    with tempfile.TemporaryDirectory(prefix="cpath_v3w_") as tmpdir:
        tile = Path(tmpdir) / "patch.png"
        current = {"key": None}

        def call(job: Job):
            # Patch-major job order means the PNG is rewritten once per patch.
            if current["key"] != job.key:
                arr_to_pil(images[int(job.key)]).save(tile, format="PNG")
                current["key"] = job.key
            return client.analyze_tiles(
                tile_paths=[tile], clinical_context="", task="tissue_classification",
                prompt_text=variants[job.variant], model=model, max_tokens=128,
            )

        made, failed = run_jobs(jobs, call, log, logger=logger)
    logger.info("made=%d failed=%d", made, failed)

    rows: list[dict] = []
    missing = 0
    for i, r in patches.iterrows():
        got = collect(log, str(int(r.arr_idx)), K)
        if got is None:
            missing += 1
            if not args.allow_incomplete:
                continue
            labels = [log.records.get(f"{int(r.arr_idx)}|{v}|0", {}).get("label", "unknown") for v in range(K)]
            confs = [float(log.records.get(f"{int(r.arr_idx)}|{v}|0", {}).get("confidence", 0.5)) for v in range(K)]
        else:
            labels, confs = got
        rows.append({
            "patch_index": int(i), "arr_idx": int(r.arr_idx),
            "true_label_idx": int(r.true_label_idx), "true_label_name": r.true_label_name,
            **summarize_k(labels, confs, r.true_label_name, K),
        })

    meta.update({"finished_utc": utc_now(), "call_window": log_time_window(log), "patches_incomplete": missing})
    write_run_meta(out_dir / "run_meta.json", meta)
    if missing and not args.allow_incomplete:
        print(f"{missing} patches still have missing calls; re-run the same command to resume.", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    pred_path = out_dir / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    print(f"Saved: {pred_path}")

    from run_consistency import _compute_and_save_metrics

    _compute_and_save_metrics(df, out_dir, model, "V3W")
    print(f"Saved: {out_dir / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
