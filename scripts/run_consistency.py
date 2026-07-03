"""
scripts/run_consistency.py

Task 3: Consistency-based uncertainty estimation for VLM routing.

The textual confidence from MedGemma saturates at T=200 and is not a useful
routing signal. This script tests whether querying the same patch K=5 times
with different prompt phrasings yields a consistency score that IS a useful
signal: patches the model answers consistently tend to be correct; patches
where answers vary across prompts are ambiguous and should be routed.

Protocol:
  1. Select 200 patches per class (1800 total) from the val split, stratified
     so both correct and incorrect predictions are represented when full-scale
     predictions from Task 2 are available.
  2. For each patch, run MedGemma K=5 times using 5 prompt variants.
  3. Compute consistency_score = fraction of K queries that agree with the
     modal label. Route by uncertainty = 1 - consistency_score.
  4. Compare routing AUC of consistency vs random vs textual confidence.

Outputs (under results/consistency/{model}/{prompt_version}/):
    predictions.parquet   -- per-patch: patch_index, true_label, modal_label,
                             consistency_score, uncertainty_score, is_correct,
                             raw_labels (list), mean_textual_conf
    metrics.json          -- accuracy, macro F1, ECE, routing AUCs

Usage:
    # Cost estimate only (no API calls):
    python scripts/run_consistency.py --estimate-only

    # Full run with V3 base prompt (default -- matches the main evaluation prompt):
    python scripts/run_consistency.py \\
        --full-pred-path results/zeroshot/medgemma-27b-it/V3_full/predictions.parquet

    # Run with a different base prompt for comparison:
    python scripts/run_consistency.py --base-prompt-version V5 \\
        --full-pred-path results/zeroshot/medgemma-27b-it/V3_full/predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import (
    LABEL_NAMES,
    balanced_indices,
    load_labels,
    load_split_arrays,
    arr_to_pil,
)
from src.models.client import Client
from src.models.prompts import get_prompt
from src.eval.calibration import ece_score
from src.triage.router import risk_coverage_curve, random_routing_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
K = 5          # number of prompt variants per patch
N_PER_CLASS = 200


# ── Prompt variants ───────────────────────────────────────────────────────────
# Variants differ in framing only; the class list and diagnostic cues are the
# same across all five. At temperature=0.0 the model is deterministic for each
# variant, so variation across variants reflects genuine prompt sensitivity.

def _build_variants(base_version: str) -> list[str]:
    """Return K=5 prompt variant strings for the consistency experiment."""
    base = get_prompt("tissue_classification", version=base_version)

    # Variant 2: prepend an architectural focus instruction
    v2 = (
        "Focus especially on the dominant architectural pattern rather than "
        "individual cell morphology.\n\n" + base
    )

    # Variant 3: class list in alphabetical order.
    # Same diagnostic content as V4 but the nine classes are listed in
    # alphabetical order (ADI, BACK, STR, TUM, DEB, LYM, MUC, NORM, MUS)
    # to test whether listing order biases the prediction.
    v3 = _build_alphabetical_variant()

    # Variant 4: add clinical context
    v4 = get_prompt(
        "tissue_classification",
        version=base_version,
        clinical_context="colorectal surgical specimen",
    )

    # Variant 5: uncertainty reminder appended before the output format block
    # Insert just before the "Respond in exactly this format" line.
    uncertainty_note = (
        "If you are uncertain between two classes, express that uncertainty "
        "by setting CONFIDENCE to a lower value (e.g. 40-60)."
    )
    if "Respond in exactly this format" in base:
        v5 = base.replace(
            "Respond in exactly this format",
            uncertainty_note + "\n\nRespond in exactly this format",
        )
    else:
        v5 = base + "\n\n" + uncertainty_note

    return [base, v2, v3, v4, v5]


def _build_alphabetical_variant() -> str:
    """
    V4-equivalent prompt with the nine tissue classes listed alphabetically.

    Alphabetical order: adipose, background, cancer-associated stroma,
    colorectal adenocarcinoma epithelium, debris, lymphocytes, mucus,
    normal colon mucosa, smooth muscle.
    """
    return (
        "You are an expert gastrointestinal pathologist analyzing an H&E stained "
        "colorectal tissue patch.\n\n"
        "Before choosing a label, reason through these steps. "
        "Do not write the reasoning steps in your response.\n"
        "Step 1: Identify the dominant structural feature visible in the patch "
        "(organized glands or crypts, dense lymphocyte sheets, acellular pools, "
        "or spindle-cell fibers).\n"
        "Step 2: Eliminate classes whose required features are absent.\n"
        "Step 3: Apply the disambiguation rules below to resolve remaining candidates.\n\n"
        "Tissue classes (alphabetical order) and their required diagnostic features:\n"
        "- adipose: large empty rounded vacuoles, thin membranes, nuclei absent\n"
        "- background: white or near-empty glass, no tissue present\n"
        "- cancer-associated stroma: spindle-cell-rich reactive or desmoplastic fibrous "
        "tissue situated adjacent to or infiltrating malignant glands; loose irregular "
        "collagen bundles and reactive fibroblasts; CHOOSE this only when reactive "
        "fibrous tissue is in contact with or surrounds malignant glands.\n"
        "- colorectal adenocarcinoma epithelium: irregular crowded malignant glands, "
        "nuclear pleomorphism, loss of polarity, invasive architecture\n"
        "- debris: amorphous necrotic material, cellular ghosts, no intact cells\n"
        "- lymphocytes: dense sheets of small round dark nuclei, scant cytoplasm\n"
        "- mucus: pale acellular or sparsely cellular pools of extracellular mucin; "
        "the dominant feature is blue-gray homogeneous pools with very few intact cells; "
        "CHOOSE mucus when extracellular mucin pools cover most of the patch.\n"
        "- normal colon mucosa: regular symmetric crypts, columnar epithelium, goblet "
        "cells, preserved crypt architecture, no invasion; "
        "CHOOSE this when well-organized crypts are present with no malignant features.\n"
        "- smooth muscle: organized parallel fascicles of uniformly elongated spindle "
        "cells with blunt cigar-shaped nuclei, abundant eosinophilic cytoplasm; "
        "NO glandular structures are present anywhere in the patch.\n\n"
        "Disambiguation rules:\n"
        "Rule 1 (smooth muscle vs cancer-associated stroma): Both have spindle cells. "
        "Smooth muscle shows uniform parallel fascicles with NO glandular context. "
        "Cancer-associated stroma occurs only adjacent to malignant glands. "
        "If no glands are visible, the label is smooth muscle.\n"
        "Rule 2 (normal colon mucosa vs cancer-associated stroma): Normal colon mucosa "
        "has well-organized regular crypts with columnar epithelium. "
        "Cancer-associated stroma has no organized crypts. "
        "If you see orderly crypts, the label is normal colon mucosa.\n"
        "Rule 3 (mucus vs normal colon mucosa): Mucus patches are dominated by pale "
        "extracellular pools with sparse cells and no organized crypts. "
        "Normal colon mucosa has intact glandular crypts. "
        "If the patch shows pale pooled material without intact crypts, choose mucus.\n\n"
        "\n\n"
        "Respond in exactly this format and nothing else:\n"
        "LABEL: <one tissue type from the list above>\n"
        "CONFIDENCE: <integer 0-100>\n"
        "REASONING: <one sentence naming the dominant histological features you observed>"
    )


# ── Patch selection ───────────────────────────────────────────────────────────

def _select_patches(
    split_labels: np.ndarray,
    n_per_class: int,
    seed: int,
    full_pred_path: Path | None,
) -> list[tuple[int, int, str]]:
    """
    Select up to n_per_class patches per class from the val split.

    If full_pred_path is provided, stratify by correct/incorrect predictions
    so that both easy and hard patches are represented. Otherwise, use
    balanced random sampling.

    Returns a list of (dataset_index, label_idx, label_name) tuples.
    """
    rng = np.random.default_rng(seed)

    if full_pred_path is not None and full_pred_path.exists():
        logger.info("Using full-scale predictions for stratified selection: %s", full_pred_path)
        df = pd.read_parquet(full_pred_path)
        # The full predictions file uses 0-based index into the val split
        result: list[tuple[int, int, str]] = []
        for class_idx in range(len(LABEL_NAMES)):
            class_name = LABEL_NAMES[class_idx]
            # Filter to patches of this class
            mask = df["true_label_name"] == class_name
            class_df = df[mask]
            correct_idx = class_df[class_df["pred_label"] == class_name]["arr_idx"].tolist()
            incorrect_idx = class_df[class_df["pred_label"] != class_name]["arr_idx"].tolist()

            # Take up to half correct and half incorrect
            n_half = n_per_class // 2
            chosen_correct   = rng.choice(correct_idx,   min(n_half, len(correct_idx)),   replace=False).tolist() if correct_idx   else []
            chosen_incorrect = rng.choice(incorrect_idx, min(n_half, len(incorrect_idx)), replace=False).tolist() if incorrect_idx else []

            chosen = chosen_correct + chosen_incorrect
            # Top up from whichever pool has slack if total < n_per_class
            if len(chosen) < n_per_class:
                remaining_pool = [i for i in correct_idx + incorrect_idx if i not in set(chosen)]
                extra = rng.choice(remaining_pool, min(n_per_class - len(chosen), len(remaining_pool)), replace=False).tolist() if remaining_pool else []
                chosen.extend(extra)

            for arr_idx in chosen:
                result.append((int(arr_idx), class_idx, class_name))
        logger.info("Stratified selection: %d patches (%d per class target)", len(result), n_per_class)
        return result
    else:
        if full_pred_path is not None:
            logger.warning("Full predictions not found at %s; falling back to balanced sampling.", full_pred_path)
        return balanced_indices(split_labels, n_per_class=n_per_class, seed=seed)


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_consistency(
    model: str,
    base_version: str,
    seed: int,
    out_dir: Path,
    full_pred_path: Path | None,
    estimate_only: bool,
) -> None:
    split_labels = load_labels("val")
    patches = _select_patches(split_labels, N_PER_CLASS, seed, full_pred_path)
    n_total = len(patches)

    avg_prompt_tok = 550   # V4 is longer than V3
    avg_compl_tok  = 40
    total_calls = n_total * K
    print("\nConsistency experiment cost estimate:")
    print(f"  Patches:          {n_total} ({N_PER_CLASS}/class x {len(LABEL_NAMES)} classes)")
    print(f"  Variants per patch: {K}")
    print(f"  Total API calls:  {total_calls:,}")
    print(f"  Prompt tokens:    ~{total_calls * avg_prompt_tok:,}")
    print(f"  Completion tokens:~{total_calls * avg_compl_tok:,}")
    print()

    if estimate_only:
        print("--estimate-only: no API calls made.")
        return

    print("Starting consistency experiment in 5 seconds... (Ctrl+C to abort)")
    time.sleep(5)
    print()

    variants = _build_variants(base_version)
    assert len(variants) == K

    logger.info("Loading val images...")
    val_images, _ = load_split_arrays("val")
    logger.info("Val images loaded: shape=%s", val_images.shape)

    client = Client(model=model)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    n_parse_fail = 0

    with tempfile.TemporaryDirectory(prefix="cpath_cons_") as tmpdir:
        tmp = Path(tmpdir)

        for patch_i, (arr_idx, true_idx, true_name) in enumerate(patches):
            patch_path = tmp / "patch.png"
            img = arr_to_pil(val_images[arr_idx])
            img.save(patch_path, format="PNG")

            query_labels: list[str] = []
            query_confs:  list[float] = []

            for var_i, variant_prompt in enumerate(variants):
                try:
                    resp = client.analyze_tiles(
                        tile_paths=[patch_path],
                        clinical_context="",
                        task="tissue_classification",
                        prompt_text=variant_prompt,
                        model=model,
                        max_tokens=128,
                    )
                    pred = resp.prediction
                    conf = resp.confidence
                except Exception as exc:
                    logger.warning("API error patch %d variant %d: %s", patch_i, var_i, exc)
                    pred = "unknown"
                    conf = 0.5

                if pred == "unknown":
                    n_parse_fail += 1
                query_labels.append(pred)
                query_confs.append(conf)

            # Compute consistency metrics
            valid_labels = [lbl for lbl in query_labels if lbl != "unknown"]
            if valid_labels:
                counts = Counter(valid_labels)
                modal_label = counts.most_common(1)[0][0]
                modal_count = counts[modal_label]
                consistency_score = modal_count / K
            else:
                modal_label = "unknown"
                consistency_score = 0.0

            is_correct = modal_label == true_name
            mean_conf  = float(np.mean(query_confs))

            records.append({
                "patch_index":        patch_i,
                "arr_idx":            arr_idx,
                "true_label_idx":     true_idx,
                "true_label_name":    true_name,
                "modal_label":        modal_label,
                "consistency_score":  round(consistency_score, 4),
                "uncertainty_score":  round(1.0 - consistency_score, 4),
                "is_correct":         bool(is_correct),
                "mean_textual_conf":  round(mean_conf, 4),
                "raw_labels":         json.dumps(query_labels),
                "raw_confs":          json.dumps([round(c, 3) for c in query_confs]),
            })

            if (patch_i + 1) % 50 == 0 or patch_i == 0:
                done = patch_i + 1
                acc_so_far = sum(r["is_correct"] for r in records) / len(records)
                mean_cs = np.mean([r["consistency_score"] for r in records])
                logger.info(
                    "%d/%d patches | modal_acc=%.3f | mean_consistency=%.3f | "
                    "parse_fail=%d",
                    done, n_total, acc_so_far, mean_cs, n_parse_fail,
                )

    del val_images

    df = pd.DataFrame(records)
    pred_path = out_dir / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    logger.info("Predictions saved: %s", pred_path)

    _compute_and_save_metrics(df, out_dir, model, base_version)


def _compute_and_save_metrics(
    df: pd.DataFrame,
    out_dir: Path,
    model: str,
    base_version: str,
) -> None:
    true_names   = df["true_label_name"].tolist()
    modal_labels = df["modal_label"].tolist()
    consistency  = df["consistency_score"].to_numpy(float)
    is_correct   = df["is_correct"].to_numpy(bool)
    mean_conf    = df["mean_textual_conf"].to_numpy(float)

    modal_acc  = accuracy_score(true_names, modal_labels)
    modal_f1   = f1_score(true_names, modal_labels, average="macro",
                          labels=ALL_LABELS, zero_division=0)
    per_class_f1 = f1_score(true_names, modal_labels, average=None,
                             labels=ALL_LABELS, zero_division=0)

    # ECE: treat consistency_score as P(correct)
    ece_consistency = ece_score(consistency, is_correct.astype(int))

    # Routing AUC: higher consistency = higher confidence = auto-confirm
    curve_consistency = risk_coverage_curve(consistency,   is_correct)
    curve_textual     = risk_coverage_curve(mean_conf,     is_correct)
    curve_random      = random_routing_curve(is_correct, seed=42)

    auc_consistency = curve_consistency["auc"]
    auc_textual     = curve_textual["auc"]
    auc_random      = curve_random["auc"]

    metrics = {
        "model":             model,
        "base_prompt":       base_version,
        "n_patches":         len(df),
        "n_variants":        K,
        "modal_accuracy":    round(float(modal_acc), 4),
        "modal_macro_f1":    round(float(modal_f1),  4),
        "per_class_f1": {ALL_LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class_f1)},
        "ece_consistency_as_confidence": round(float(ece_consistency), 4),
        "routing_auc": {
            "consistency": round(float(auc_consistency), 4),
            "textual_conf": round(float(auc_textual),    4),
            "random":       round(float(auc_random),     4),
        },
        "mean_consistency_score": round(float(consistency.mean()), 4),
        "consistency_beats_random": bool(auc_consistency > auc_random),
        "consistency_beats_textual": bool(auc_consistency > auc_textual),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Risk-coverage figure
    _plot_routing_curves(curve_consistency, curve_textual, curve_random, out_dir, base_version)

    print(f"\n{'='*65}")
    print(f"Consistency experiment results  |  model={model}  prompt={base_version}")
    print(f"{'='*65}")
    print(f"Modal accuracy:      {modal_acc:.4f}")
    print(f"Modal macro F1:      {modal_f1:.4f}")
    print(f"ECE (consistency):   {ece_consistency:.4f}")
    print(f"Routing AUC (consistency): {auc_consistency:.4f}")
    print(f"Routing AUC (textual conf):{auc_textual:.4f}")
    print(f"Routing AUC (random):      {auc_random:.4f}")
    print(f"Consistency beats random?  {'YES' if auc_consistency > auc_random else 'NO'}")
    print(f"Mean consistency score:    {consistency.mean():.3f}")
    print(f"{'='*65}")
    print(f"Results: {out_dir}")


def _plot_routing_curves(
    curve_cons: dict,
    curve_text: dict,
    curve_rand: dict,
    out_dir: Path,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    budgets = np.array(curve_cons["budgets"])

    def _plot(curve: dict, color: str, ls: str, name: str) -> None:
        accs = np.array(curve["auto_confirm_acc"], dtype=float)
        valid = ~np.isnan(accs)
        ax.plot(budgets[valid], accs[valid], ls, color=color, lw=1.8,
                label=f"{name} (AUC={curve['auc']:.3f})")
        if "std_acc" in curve:
            std = np.array(curve["std_acc"])
            ax.fill_between(budgets[valid], (accs - std)[valid], (accs + std)[valid],
                            color=color, alpha=0.12)

    _plot(curve_rand, "#888888", "--", "Random")
    _plot(curve_text, "#4878CF", "-",  "Textual confidence")
    _plot(curve_cons, "#D65F5F", "-",  "Consistency score")

    ax.axvline(0.15, color="black", lw=1, ls=":", alpha=0.7, label="15% routing budget")
    ax.set_xlabel("Fraction routed to specialist", fontsize=11)
    ax.set_ylabel("Accuracy on auto-confirmed set", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Consistency routing vs baselines ({label})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "routing_curves.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Routing figure: %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task 3: consistency-based uncertainty estimation"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--base-prompt-version", choices=["V3", "V4", "V4B", "V5"], default="V3",
        help=(
            "Base prompt version for the 5 variants. Default is V3 because V3 is "
            "the prompt used for the main full-scale evaluation and the one whose "
            "textual confidence is anti-discriminative for routing -- it is the "
            "prompt this experiment needs to test against."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-pred-path", type=Path, default=None,
        help=(
            "Path to full-scale val predictions.parquet from Task 2. "
            "Used to stratify the 1800-patch selection by correct/incorrect. "
            "If omitted, falls back to balanced random sampling."
        ),
    )
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    model = args.model or os.environ.get("MEDGEMMA_MODEL", "medgemma-27b-it")
    model_slug = model.replace("/", "_").replace(":", "_")
    out_dir = (
        PROJECT_ROOT / "results" / "consistency" / model_slug / args.base_prompt_version
    )

    run_consistency(
        model=model,
        base_version=args.base_prompt_version,
        seed=args.seed,
        out_dir=out_dir,
        full_pred_path=args.full_pred_path,
        estimate_only=args.estimate_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
