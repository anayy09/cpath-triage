"""
scripts/pilot_logprob_design.py

Design pilot for the label-token confidence baseline (revision item C-11,
Decision 1 in docs/DECISIONS_SR_R1.md). 50 patches per model, both models.

Why a pilot. The chosen design reads the token distribution at the first token
of the label span in the V3 response, maps each candidate token to a class by
prefix, and renormalizes over the nine classes. That only works if three things
hold on this endpoint's tokenizer, none of which we can assume:

  (a) the first token of the label span is injective across the nine class names,
      so a single token identifies the class;
  (b) enough of the nine classes appear among the top-k alternatives at that
      position for the renormalized distribution to mean anything;
  (c) the label span can be located in the response at all.

This script measures all three and writes the answer to a file. Per Decision 1,
if (a) fails the fallback is to sum token logprobs over the located label span
with length normalization, and the length confound gets stated in Methods. If
more than 5% of responses fail (c), stop and report before spending the full
budget.

The prompt is byte-identical to the V3 prompt used everywhere else, so the pilot
also returns the verbalized confidence from the same call and the two signals are
paired by construction.

Cost: 50 calls per model, 100 total.

Outputs:
    results/logprob_confidence/design_pilot.json

Usage:
    python scripts/pilot_logprob_design.py
    python scripts/pilot_logprob_design.py --n 50 --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import LABEL_NAMES, arr_to_pil, load_split_arrays
from src.models.client import Client
from src.models.prompts import get_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_LABELS = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
MODELS = ("medgemma-27b-it", "gemma-3-27b-it")


def first_token_map(tokenizer_probe: dict[str, str]) -> dict:
    """
    Check whether the first token of each class name identifies the class.

    tokenizer_probe maps class name to the first token the endpoint actually
    emitted for it, gathered from real responses. Injectivity is decided on
    observed tokens rather than on a guess about the vocabulary.
    """
    by_token: dict[str, list[str]] = {}
    for label, tok in tokenizer_probe.items():
        by_token.setdefault(tok, []).append(label)
    collisions = {t: ls for t, ls in by_token.items() if len(ls) > 1}
    return {
        "observed_first_tokens": tokenizer_probe,
        "injective": len(collisions) == 0,
        "collisions": collisions,
        "n_classes_observed": len(tokenizer_probe),
    }


def locate_label_span(token_logprobs: list[dict]) -> int | None:
    """
    Index of the first token after the ``LABEL:`` marker.

    Returns None when the marker cannot be found, which is outcome (c).
    """
    joined = ""
    starts: list[int] = []
    for entry in token_logprobs:
        starts.append(len(joined))
        joined += entry.get("token", "")

    upper = joined.upper()
    marker = upper.find("LABEL:")
    if marker < 0:
        return None
    target = marker + len("LABEL:")
    # First token whose span begins at or after the marker and carries content.
    for i, start in enumerate(starts):
        if start >= target and token_logprobs[i].get("token", "").strip():
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Patches per model.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--prompt-version", default="V3")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images, labels = load_split_arrays(args.split)
    # A plain seeded sample is enough here: the pilot tests tokenizer behaviour,
    # not accuracy, so class balance buys nothing.
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(labels), size=min(args.n, len(labels)), replace=False)
    prompt_text = get_prompt(task="tissue_classification", version=args.prompt_version)

    out: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="cpath_lp_pilot_") as tmpdir:
        tmp = Path(tmpdir)
        patch_path = tmp / "patch.png"

        for model in MODELS:
            client = Client(model=model)
            logger.info("Pilot on %s, %d patches", model, len(idx))

            n_no_logprobs = 0
            n_no_span = 0
            classes_in_topk: list[int] = []
            topk_sizes: list[int] = []
            first_tokens: dict[str, str] = {}
            examples: list[dict] = []
            t_start = time.time()

            for j, arr_idx in enumerate(idx):
                arr_to_pil(images[arr_idx]).save(patch_path, format="PNG")
                try:
                    resp, token_logprobs = client.analyze_tiles_with_logprobs(
                        tile_paths=[patch_path],
                        clinical_context="",
                        task="tissue_classification",
                        prompt_text=prompt_text,
                        model=model,
                        max_tokens=128,
                        top_logprobs=args.top_logprobs,
                    )
                except Exception as exc:
                    logger.warning("call failed on %d: %s", j, exc)
                    n_no_logprobs += 1
                    continue

                if not token_logprobs:
                    n_no_logprobs += 1
                    continue

                pos = locate_label_span(token_logprobs)
                if pos is None:
                    n_no_span += 1
                    continue

                entry = token_logprobs[pos]
                top = entry.get("top") or []
                topk_sizes.append(len(top))

                # Which of the nine class names does each alternative prefix?
                matched = set()
                for alt in top:
                    tok = (alt.get("token") or "").strip().lower()
                    if not tok:
                        continue
                    for label in ALL_LABELS:
                        if label.lower().startswith(tok):
                            matched.add(label)
                classes_in_topk.append(len(matched))

                # Record the emitted first token for the class the model chose,
                # which is what injectivity has to hold over.
                if resp.prediction in ALL_LABELS:
                    first_tokens.setdefault(resp.prediction, (entry.get("token") or "").strip())

                if len(examples) < 5:
                    examples.append({
                        "arr_idx": int(arr_idx),
                        "prediction": resp.prediction,
                        "verbalized_confidence": resp.confidence,
                        "label_span_token": entry.get("token"),
                        "label_span_logprob": entry.get("logprob"),
                        "top_alternatives": top[:10],
                    })

            n_attempted = len(idx)
            n_usable = len(classes_in_topk)
            elapsed = time.time() - t_start

            inj = first_token_map(first_tokens)
            out[model] = {
                "n_attempted": n_attempted,
                "n_usable": n_usable,
                "n_no_logprobs_returned": n_no_logprobs,
                "n_label_span_not_found": n_no_span,
                "frac_label_span_not_found": round(n_no_span / max(n_attempted, 1), 4),
                "gate_c_pass_under_5pct": bool(n_no_span / max(n_attempted, 1) <= 0.05),
                "top_logprobs_requested": args.top_logprobs,
                "top_logprobs_returned_mean": (
                    round(float(np.mean(topk_sizes)), 2) if topk_sizes else None
                ),
                "classes_matched_in_topk": {
                    "mean": round(float(np.mean(classes_in_topk)), 2) if classes_in_topk else None,
                    "min": int(np.min(classes_in_topk)) if classes_in_topk else None,
                    "max": int(np.max(classes_in_topk)) if classes_in_topk else None,
                    "distribution": dict(sorted(Counter(classes_in_topk).items())),
                },
                "gate_a_first_token_injective": inj,
                "elapsed_s": round(elapsed, 1),
                "examples": examples,
            }

            g = out[model]
            logger.info(
                "%s: usable %d/%d, span missing %.1f%%, classes in top-k mean %s, injective %s",
                model, n_usable, n_attempted,
                100 * g["frac_label_span_not_found"],
                g["classes_matched_in_topk"]["mean"],
                inj["injective"],
            )

    out_dir = PROJECT_ROOT / "results" / "logprob_confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_pilot.json"
    out_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "split": args.split,
                "prompt_version": args.prompt_version,
                "design": "first-token-of-label-span, renormalized over nine classes",
                "gates": {
                    "a": "first token of the label span is injective across the nine classes",
                    "b": "how many of the nine classes appear in the top-k at that position",
                    "c": "fraction of responses where the label span cannot be located, must be <= 5%",
                },
                "results": out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Design pilot summary ===")
    for model, g in out.items():
        print(f"\n{model}")
        print(f"  usable responses          : {g['n_usable']}/{g['n_attempted']}")
        print(f"  logprobs absent           : {g['n_no_logprobs_returned']}")
        print(f"  label span not found      : {g['n_label_span_not_found']} "
              f"({100 * g['frac_label_span_not_found']:.1f}%)  "
              f"gate C {'PASS' if g['gate_c_pass_under_5pct'] else 'FAIL'}")
        print(f"  top-k returned (mean)     : {g['top_logprobs_returned_mean']} "
              f"of {g['top_logprobs_requested']} requested")
        print(f"  classes in top-k (mean)   : {g['classes_matched_in_topk']['mean']} of 9")
        inj = g["gate_a_first_token_injective"]
        print(f"  gate A first-token unique : {'PASS' if inj['injective'] else 'FAIL'} "
              f"({inj['n_classes_observed']} classes observed)")
        if inj["collisions"]:
            print(f"    collisions: {inj['collisions']}")

    print("\nIf gate A fails, fall back to length-normalized summed logprobs over the")
    print("label span and state the length confound in Methods (Decision 1).")
    print("If gate C fails on either model, stop and report before the full run.")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
