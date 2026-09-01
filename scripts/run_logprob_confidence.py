"""
scripts/run_logprob_confidence.py

Label-token confidence baseline (revision item C-11, Decision 1 in
docs/DECISIONS_SR_R1.md).

Both reviewers asked whether the confidence failure we document is specific to
the number the model writes in its response, or whether it reaches into the
model's own token distribution. This runs the comparison. The V3 prompt is
byte-identical to the one used everywhere else and logprobs are requested on the
same call, so the verbalized confidence and the label-token confidence come from
one response per patch and are paired by construction rather than by a join.

Design, decided in advance and not to be re-litigated here: read the token
distribution at the first token of the label span, map each candidate token to a
class by prefix, renormalize over the nine classes, and take the probability of
the predicted class as the label-token confidence. Run
scripts/pilot_logprob_design.py first; it tests the three assumptions this
depends on. If its gate A (first-token injectivity) failed, pass
--fallback-span-sum to use length-normalized summed logprobs over the located
label span instead, and state the length confound in Methods.

Sampling, decided in advance:
  validation  the existing 2,001-patch held-out evaluation partition, so the
              numbers are directly comparable to every figure already in Table 5
  test        a fresh unstratified random 2,000-patch sample, seed 42. The
              1,800-patch consistency subset is deliberately not reused: it is
              stratified on correctness and would bias the accuracy base.

Cost: about 4,001 calls per model, 8,002 for both.

Endpoint metadata (provider, any version string, request date range) is captured
into the metrics JSON at run time, which also serves revision item C-13 and the
Editor's reproducibility expectation.

Outputs:
    results/logprob_confidence/{model}/{split}/predictions.parquet
    results/logprob_confidence/{model}/{split}/metrics.json

Usage:
    python scripts/run_logprob_confidence.py --model medgemma-27b-it --split val
    python scripts/run_logprob_confidence.py --model medgemma-27b-it --split test
    python scripts/run_logprob_confidence.py --model gemma-3-27b-it --split val
    python scripts/run_logprob_confidence.py --model gemma-3-27b-it --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot_logprob_design import locate_label_span
from src.data.pathmnist import LABEL_NAMES, arr_to_pil, load_split_arrays
from src.models.client import Client
from src.models.prompts import get_prompt
from src.triage.router import random_routing_curve, risk_coverage_curve, tie_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
N_BINS = 15
SEED = 42


def eval_partition_indices(model: str) -> np.ndarray:
    """
    The exact held-out evaluation partition calibrate.py uses.

    Reproduced rather than re-drawn so the label-token numbers land on the same
    2,001 patches as the published calibration figures.
    """
    df = pd.read_parquet(
        PROJECT_ROOT / "results" / "zeroshot" / model / "V3_full" / "predictions.parquet"
    )
    df = df[df["pred_label"] != "unknown"].copy()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    held = df.iloc[order[int(len(df) * 0.8):]]
    return held["arr_idx"].to_numpy()


def label_token_confidence(
    token_logprobs: list[dict], predicted: str, fallback_span_sum: bool
) -> tuple[float | None, int, dict]:
    """
    Probability of the predicted class from the label-span token distribution.

    Returns (confidence, n_classes_in_topk, per_class_probs). Confidence is None
    when the span cannot be located, which is recorded rather than imputed.
    """
    pos = locate_label_span(token_logprobs)
    if pos is None:
        return None, 0, {}

    if fallback_span_sum:
        # Length-normalized mean logprob over the label span, converted to a
        # pseudo-probability. Comparable across patches only up to the length
        # confound, which Methods states.
        span: list[float] = []
        for entry in token_logprobs[pos:]:
            tok = entry.get("token") or ""
            if "\n" in tok:
                break
            lp = entry.get("logprob")
            if lp is not None:
                span.append(float(lp))
        if not span:
            return None, 0, {}
        return float(math.exp(sum(span) / len(span))), 0, {}

    top = token_logprobs[pos].get("top") or []
    mass: dict[str, float] = {}
    for alt in top:
        tok = (alt.get("token") or "").strip().lower()
        lp = alt.get("logprob")
        if not tok or lp is None:
            continue
        for label in ALL_LABELS:
            if label.lower().startswith(tok):
                # A token may prefix several class names; give each the full
                # mass and renormalize, which is the standard treatment.
                mass[label] = mass.get(label, 0.0) + math.exp(float(lp))

    total = sum(mass.values())
    if total <= 0:
        return None, 0, {}
    probs = {k: v / total for k, v in mass.items()}
    return probs.get(predicted, 0.0), len(mass), probs


def bin_confidence(conf: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = bin_confidence(conf, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def auroc_midrank(scores: np.ndarray, positive: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mi_bits(conf: np.ndarray, correct: np.ndarray) -> float:
    from sklearn.metrics import mutual_info_score

    return float(mutual_info_score(bin_confidence(conf), correct.astype(int)) / np.log(2))


def summarize(conf: np.ndarray, correct: np.ndarray, seed: int) -> dict:
    ties = tie_statistics(conf)
    curve = risk_coverage_curve(conf, correct, tie_break="random", n_repeats=200, seed=seed)
    rnd = random_routing_curve(correct, seed=seed)["auc"]
    return {
        "n": len(conf),
        "accuracy": round(float(correct.mean()), 4),
        "auroc_midrank": round(auroc_midrank(conf, correct), 4),
        "mi_bits": round(mi_bits(conf, correct), 5),
        "ece": round(ece(conf, correct), 4),
        "selective_accuracy_auc": round(float(curve["auc"]), 4),
        "selective_accuracy_auc_sd_over_tie_orders": round(float(curve["auc_sd"]), 4),
        "random_routing_auc": round(float(rnd), 4),
        "gap_vs_random": round(float(curve["auc"] - rnd), 4),
        "tie_fraction": round(float(ties["tie_fraction"]), 4),
        "n_distinct": ties["n_distinct"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--prompt-version", default="V3")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--fallback-span-sum", action="store_true",
        help="Use length-normalized summed logprobs over the label span. Set this "
             "only if the pilot's first-token injectivity gate failed.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap the patch count.")
    args = parser.parse_args()

    pilot = PROJECT_ROOT / "results" / "logprob_confidence" / "design_pilot.json"
    if not pilot.exists():
        print(f"Design pilot not found: {pilot}", file=sys.stderr)
        print("Run scripts/pilot_logprob_design.py first (Decision 1 gate).", file=sys.stderr)
        return 1

    images, labels = load_split_arrays(args.split)

    if args.split == "val":
        idx = eval_partition_indices(args.model)
        sampling = "held-out evaluation partition (random 80/20, seed 42)"
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(labels), size=min(args.n_test, len(labels)), replace=False)
        sampling = f"fresh unstratified random sample, n={len(idx)}, seed {args.seed}"
    if args.limit:
        idx = idx[: args.limit]

    prompt_text = get_prompt(task="tissue_classification", version=args.prompt_version)
    client = Client(model=args.model)
    started = datetime.now(timezone.utc).isoformat()

    records: list[dict] = []
    n_span_missing = 0

    with tempfile.TemporaryDirectory(prefix="cpath_lp_") as tmpdir:
        patch_path = Path(tmpdir) / "patch.png"
        t0 = time.time()

        for i, arr_idx in enumerate(idx):
            arr_to_pil(images[arr_idx]).save(patch_path, format="PNG")
            true_name = LABEL_NAMES[int(labels[arr_idx])]
            try:
                resp, token_logprobs = client.analyze_tiles_with_logprobs(
                    tile_paths=[patch_path],
                    clinical_context="",
                    task="tissue_classification",
                    prompt_text=prompt_text,
                    model=args.model,
                    max_tokens=128,
                    top_logprobs=args.top_logprobs,
                )
            except Exception as exc:
                logger.warning("API error on %d (arr_idx %d): %s", i, arr_idx, exc)
                records.append({
                    "arr_idx": int(arr_idx), "true_label_name": true_name,
                    "pred_label": "unknown", "verbalized_conf": 0.5,
                    "logprob_conf": None, "n_classes_in_topk": 0, "call_failed": True,
                })
                continue

            lp_conf, n_cls, _ = label_token_confidence(
                token_logprobs, resp.prediction, args.fallback_span_sum
            )
            if lp_conf is None:
                n_span_missing += 1

            records.append({
                "arr_idx": int(arr_idx),
                "true_label_name": true_name,
                "pred_label": resp.prediction,
                "verbalized_conf": float(resp.confidence),
                "logprob_conf": lp_conf,
                "n_classes_in_topk": n_cls,
                "call_failed": False,
            })

            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                logger.info("%d/%d  %.1f patches/s  span missing %d",
                            i + 1, len(idx), rate, n_span_missing)

    finished = datetime.now(timezone.utc).isoformat()
    df = pd.DataFrame(records)
    df["correct"] = df["pred_label"] == df["true_label_name"]

    usable = df[df["logprob_conf"].notna() & ~df["call_failed"]].copy()
    metrics: dict = {
        "model": args.model,
        "split": args.split,
        "prompt_version": args.prompt_version,
        "seed": args.seed,
        "sampling": sampling,
        "design": (
            "length-normalized summed logprobs over label span"
            if args.fallback_span_sum
            else "first-token-of-label-span, renormalized over nine classes"
        ),
        "n_attempted": len(df),
        "n_usable": len(usable),
        "n_label_span_missing": int(n_span_missing),
        "n_call_failed": int(df["call_failed"].sum()),
        "endpoint": {
            "provider_base_url": os.environ.get("BASE_URL", "unset"),
            "model_version_string": None,
            "model_version_note": (
                "The endpoint exposes no model snapshot or version identifier; "
                "recorded as absent rather than guessed."
            ),
            "request_window_utc": {"started": started, "finished": finished},
        },
    }

    if len(usable) > 0:
        correct = usable["correct"].to_numpy(bool)
        metrics["label_token_confidence"] = summarize(
            usable["logprob_conf"].to_numpy(float), correct, args.seed
        )
        metrics["verbalized_confidence_same_patches"] = summarize(
            usable["verbalized_conf"].to_numpy(float), correct, args.seed
        )
        metrics["paired_comparison_note"] = (
            "Both rows are measured on the identical patches from the identical "
            "responses, so the difference isolates the confidence source."
        )
        metrics["realized_class_distribution"] = {
            k: int(v) for k, v in usable["true_label_name"].value_counts().items()
        }

    out_dir = PROJECT_ROOT / "results" / "logprob_confidence" / args.model / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.parquet"
    metrics_path = out_dir / "metrics.json"
    df.to_parquet(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n=== Label-token vs verbalized confidence | {args.model} {args.split} ===")
    print(f"  usable {len(usable)}/{len(df)}  (span missing {n_span_missing}, "
          f"call failures {int(df['call_failed'].sum())})")
    if len(usable) > 0:
        for tag in ("label_token_confidence", "verbalized_confidence_same_patches"):
            m = metrics[tag]
            print(f"  {tag:38s} AUROC={m['auroc_midrank']:.4f}  MI={m['mi_bits']:.5f}  "
                  f"ECE={m['ece']:.4f}  AAUC={m['selective_accuracy_auc']:.4f} "
                  f"(rnd {m['random_routing_auc']:.4f})  ties={m['tie_fraction']:.3f}")
    print(f"\nWrote: {pred_path}")
    print(f"Wrote: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
