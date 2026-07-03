"""
scripts/run_zeroshot.py

Stage 2: Zero-shot tissue classification on a balanced val subset.

Memory-conscious design: loads val images once (~1.5 GB), selects a
balanced subset by index, and processes one patch at a time through the
API. No full-dataset iteration or large in-memory image lists.

Outputs (under results/zeroshot/{model}/):
    predictions.parquet    -- one row per evaluated patch
    metrics.json           -- accuracy, macro_f1, per_class_f1, parse_failure_rate
    confusion_matrix.png   -- normalized confusion matrix

Usage:
    # Estimate token cost (no API calls):
    python scripts/run_zeroshot.py --estimate-only --n-per-class 50

    # Full run with default model (medgemma-27b-it):
    python scripts/run_zeroshot.py --n-per-class 50 --seed 42

    # Ablation with a different model:
    python scripts/run_zeroshot.py --model gemma-3-27b-it --n-per-class 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import (
    LABEL_NAMES,
    LABEL_ABBREV,
    balanced_indices,
    load_labels,
    load_split_arrays,
    arr_to_pil,
)
from src.models.client import Client
from src.models.prompts import get_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]


def _print_estimate(n_samples: int) -> None:
    avg_prompt = 900    # tokens (image + prompt text)
    avg_completion = 80
    print(f"\nCost estimate for {n_samples} samples:")
    print(f"  Prompt tokens:     ~{n_samples * avg_prompt:,}")
    print(f"  Completion tokens: ~{n_samples * avg_completion:,}")
    print()


def run_evaluation(
    model: str,
    n_per_class: int,
    prompt_version: str,
    seed: int,
    out_dir: Path,
    split: str = "val",
    estimate_only: bool = False,
    all_samples: bool = False,
) -> None:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if split == "test":
        logger.warning(
            "Running on TEST split (CRC-VAL-HE-7K). "
            "This is the held-out cross-center set. Touch it only once."
        )

    # Build the sample list first to get the true count for the estimate
    logger.info("Loading %s labels...", split)
    split_labels = load_labels(split)
    if all_samples:
        subset = [(i, int(split_labels[i]), LABEL_NAMES[int(split_labels[i])])
                  for i in range(len(split_labels))]
        logger.info("Full split: %d samples", len(subset))
    else:
        subset = balanced_indices(split_labels, n_per_class=n_per_class, seed=seed)
        logger.info("Balanced subset: %d samples across %d classes", len(subset), len(LABEL_NAMES))

    n_total_est = len(subset)
    _print_estimate(n_total_est)

    if estimate_only:
        print("--estimate-only: no API calls made.")
        return

    if all_samples:
        print(
            f"\nFull-split run: {n_total_est} samples. "
            "Proceeding in 5 seconds... (Ctrl+C to abort)"
        )
        time.sleep(5)
        print("Starting full-scale evaluation...\n")

    # Load images for the chosen split (~1.5 GB val, ~1.1 GB test)
    logger.info("Loading %s images...", split)
    val_images, _ = load_split_arrays(split)
    logger.info("%s images loaded, shape=%s", split, val_images.shape)

    client = Client(model=model)
    prompt_text = get_prompt(task="tissue_classification", version=prompt_version)

    records: list[dict] = []
    total_prompt_tok = 0
    total_completion_tok = 0
    n_parse_fail = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cpath_zs_") as tmpdir:
        tmp = Path(tmpdir)

        for i, (arr_idx, true_idx, true_name) in enumerate(subset):
            # Write a single patch to a temp PNG, call the API, delete it
            patch_path = tmp / "patch.png"
            img = arr_to_pil(val_images[arr_idx])
            img.save(patch_path, format="PNG")

            t0 = time.time()
            try:
                resp = client.analyze_tiles(
                    tile_paths=[patch_path],
                    clinical_context="",
                    task="tissue_classification",
                    prompt_text=prompt_text,
                    model=model,
                    max_tokens=128,
                )
                elapsed = time.time() - t0
                pred_label = resp.prediction
                pred_conf = resp.confidence
                prompt_tok = resp.prompt_tokens
                completion_tok = resp.completion_tokens
                raw_text = resp.raw_text
                rationale = resp.rationale

            except Exception as exc:
                elapsed = time.time() - t0
                logger.warning("API error on sample %d: %s", i, exc)
                pred_label = "unknown"
                pred_conf = 0.5
                prompt_tok = completion_tok = 0
                raw_text = rationale = ""

            if pred_label == "unknown":
                n_parse_fail += 1
            total_prompt_tok += prompt_tok
            total_completion_tok += completion_tok

            records.append({
                "sample_idx": i,
                "arr_idx": arr_idx,
                "true_label_idx": true_idx,
                "true_label_name": true_name,
                "pred_label": pred_label,
                "pred_confidence": float(pred_conf),
                "raw_text": raw_text,
                "rationale": rationale,
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "latency_s": round(elapsed, 3),
            })

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    "%d/%d done | parse_fail=%d | tokens=%d+%d",
                    i + 1, len(subset), n_parse_fail,
                    total_prompt_tok, total_completion_tok,
                )

    del val_images  # free ~1.5 GB

    df = pd.DataFrame(records)
    pred_path = out_dir / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    logger.info("Predictions: %s", pred_path)

    true_names = df["true_label_name"].tolist()
    pred_names = df["pred_label"].tolist()

    accuracy = accuracy_score(true_names, pred_names)
    macro_f1 = f1_score(true_names, pred_names, average="macro", labels=ALL_LABELS, zero_division=0)
    per_class = f1_score(true_names, pred_names, average=None, labels=ALL_LABELS, zero_division=0)

    metrics = {
        "model": model,
        "prompt_version": prompt_version,
        "split": split,
        "seed": seed,
        "n_per_class": n_per_class,
        "n_evaluated": len(records),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {ALL_LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class)},
        "parse_failure_rate": round(n_parse_fail / len(records), 4),
        "n_parse_failures": n_parse_fail,
        "total_prompt_tokens": total_prompt_tok,
        "total_completion_tokens": total_completion_tok,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Model:          {model}")
    print(f"Prompt:         {prompt_version}")
    print(f"N evaluated:    {len(records)}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Parse failures: {n_parse_fail} ({n_parse_fail/len(records):.1%})")
    print(f"Tokens used:    {total_prompt_tok} prompt / {total_completion_tok} completion")
    print(f"{'='*60}\n")
    print(classification_report(true_names, pred_names, labels=ALL_LABELS, zero_division=0))

    _plot_confusion_matrix(true_names, pred_names, out_dir / "confusion_matrix.png")
    print(f"Results: {out_dir}")


def _plot_confusion_matrix(true: list[str], pred: list[str], out_path: Path) -> None:
    abbrevs = [LABEL_ABBREV[i] for i in range(len(LABEL_NAMES))]
    cm = confusion_matrix(true, pred, labels=ALL_LABELS, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(abbrevs)))
    ax.set_yticks(range(len(abbrevs)))
    ax.set_xticklabels(abbrevs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(abbrevs, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Normalized confusion matrix (row = true class)", fontsize=11)

    for r in range(len(abbrevs)):
        for c in range(len(abbrevs)):
            v = cm[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix: %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2/6: zero-shot tissue classification")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--prompt-version", choices=["V1", "V2", "V3", "V4", "V4B", "V5"], default="V2")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Dataset split. Use 'test' only for Stage 6 cross-center eval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help=(
            "Evaluate the full split without balanced sampling. "
            "Outputs go to {prompt_version}_full/ to distinguish from subset runs. "
            "Prints a cost estimate and pauses 5 s before starting."
        ),
    )
    args = parser.parse_args()

    model = args.model or os.environ.get("MEDGEMMA_MODEL", "medgemma-27b-it")
    model_slug = model.replace("/", "_").replace(":", "_")
    # Full-split runs get a _full suffix; test split gets a _test suffix.
    split_tag = "" if args.split == "val" else f"_{args.split}"
    full_tag = "_full" if args.all_samples else ""
    out_dir = (
        PROJECT_ROOT / "results" / "zeroshot" / model_slug
        / f"{args.prompt_version}{full_tag}{split_tag}"
    )

    run_evaluation(
        model=model,
        n_per_class=args.n_per_class,
        prompt_version=args.prompt_version,
        seed=args.seed,
        out_dir=out_dir,
        split=args.split,
        estimate_only=args.estimate_only,
        all_samples=args.all_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
