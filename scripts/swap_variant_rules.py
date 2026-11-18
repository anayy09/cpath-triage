"""
scripts/swap_variant_rules.py

Each variant of the original K=5 family scored under both label rules on the
image-swap control.

The original-family K=5 runs kept no response text, so their labels come from
the earlier whole-text parser and cannot be re-read from the LABEL line. The
swap control sent the same five prompts with 360 real images and kept its text,
so it is the one run where each of those prompts can be scored both ways. The
aggregate version of this comparison (agreement minus random) is in
controls/summary.json; this script breaks it down by variant, which is what is
needed to say whether variant 2's own confidence (Table S9) and its share of
agreement (Table 10, the without-variant-2 row) could depend on the parser.

Every quantity is unweighted on the 360 swapped pairs and scored against the
class of the image the model was shown (the donor patch). At n = 360 the
intervals are wide and the values are indicative only.

Per variant: the number of answers the two rules read differently, accuracy,
unknown answers, the confidence-correctness AUROC (ties at one half) and the
selective-accuracy gap over random routing (ties by their expectation), each
under both rules, with 1,000-resample paired-bootstrap intervals on the
LABEL-line minus earlier-rule difference. Agreement is also scored with all
five variants and with variant 2 left out, under both rules.

Output: results/consistency/{model}/controls/swap/variant_rules.json

Usage:
    python scripts/swap_variant_rules.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.router import risk_coverage_curve, tie_statistics
from src.triage.weighted import simple_bootstrap_indices, summarize_diffs, weighted_auroc

K = 5
N_BOOT = 1000
SUBSETS = {"all_5": (0, 1, 2, 3, 4), "without_2": (0, 1, 3, 4)}


def aauc(sig: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage_curve(sig, correct, tie_break="expected")["auc"])


def gap(sig: np.ndarray, correct: np.ndarray) -> float:
    return aauc(sig, correct) - float(correct.mean())


def auroc(sig: np.ndarray, correct: np.ndarray) -> float:
    return weighted_auroc(sig, correct, np.ones(len(sig)))


def agreement(labels: np.ndarray, keep: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Same rule as src/models/k5.py: unknown answers leave the vote and stay in the denominator."""
    modal, agree = [], []
    for row in labels:
        valid = [row[i] for i in keep if row[i] != "unknown"]
        lab, cnt = Counter(valid).most_common(1)[0] if valid else ("unknown", 0)
        modal.append(lab)
        agree.append(cnt / len(keep))
    return np.array(modal), np.array(agree, dtype=float)


def paired(fn, n: int, idx_matrix: np.ndarray) -> dict:
    """Point estimate and paired-bootstrap interval of fn(index) over the shared resamples."""
    point = fn(np.arange(n))
    diffs = np.array([fn(idx) for idx in idx_matrix])
    return summarize_diffs(diffs, point)


def r4(x: float) -> float:
    return round(float(x), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medgemma-27b-it")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed} n_boot={args.n_boot}")

    run = PROJECT_ROOT / "results" / "consistency" / args.model.replace("/", "_") / "controls" / "swap"
    path = run / "predictions.parquet"
    if not path.exists():
        print(f"Missing required file: {path} (run scripts/run_image_controls.py swap first)", file=sys.stderr)
        return 1
    df = pd.read_parquet(path)
    if "raw_labels_regex" not in df.columns:
        print(f"{path} has no *_regex columns; run scripts/rescore_labels.py first", file=sys.stderr)
        return 1

    donor = df["donor_true_name"].to_numpy()
    lab = {"label_line": np.array([json.loads(s) for s in df["raw_labels"]], dtype=object),
           "earlier_rule": np.array([json.loads(s) for s in df["raw_labels_regex"]], dtype=object)}
    conf = {"label_line": np.array([json.loads(s) for s in df["raw_confs"]], dtype=float),
            "earlier_rule": np.array([json.loads(s) for s in df["raw_confs_regex"]], dtype=float)}
    if lab["label_line"].shape != (len(df), K):
        print("unexpected number of variants", file=sys.stderr)
        return 1
    n = len(df)
    idx_matrix = simple_bootstrap_indices(n, args.n_boot, args.seed)

    per_variant = {}
    for v in range(K):
        new_lab, old_lab = lab["label_line"][:, v], lab["earlier_rule"][:, v]
        new_c, old_c = conf["label_line"][:, v], conf["earlier_rule"][:, v]
        y_new, y_old = new_lab == donor, old_lab == donor
        block = {
            "answers_read_differently": int((new_lab != old_lab).sum()),
            "confidences_read_differently": int((~np.isclose(new_c, old_c)).sum()),
            "n_distinct_confidence": int(tie_statistics(new_c)["n_distinct"]),
            "tie_fraction": r4(tie_statistics(new_c)["tie_fraction"]),
        }
        for rule, lbls, c, y in (("label_line", new_lab, new_c, y_new), ("earlier_rule", old_lab, old_c, y_old)):
            block[rule] = {
                "accuracy": r4(y.mean()),
                "unknown": int((lbls == "unknown").sum()),
                "auroc": r4(auroc(c, y)),
                "gap_vs_random": r4(gap(c, y)),
            }
        block["auroc_label_line_minus_earlier"] = paired(
            lambda i: auroc(new_c[i], y_new[i]) - auroc(old_c[i], y_old[i]), n, idx_matrix)
        block["gap_label_line_minus_earlier"] = paired(
            lambda i: gap(new_c[i], y_new[i]) - gap(old_c[i], y_old[i]), n, idx_matrix)
        block["label_line_auroc_ci"] = paired(lambda i: auroc(new_c[i], y_new[i]), n, idx_matrix)
        block["label_line_gap_ci"] = paired(lambda i: gap(new_c[i], y_new[i]), n, idx_matrix)
        per_variant[str(v)] = block

    agree_out = {}
    for sname, keep in SUBSETS.items():
        sig, out = {}, {}
        for rule in lab:
            modal, a = agreement(lab[rule], keep)
            sig[rule] = (a, modal == donor)
            out[rule] = {"modal_accuracy": r4((modal == donor).mean()), "gap_vs_random": r4(gap(*sig[rule]))}
        (a1, y1), (a0, y0) = sig["label_line"], sig["earlier_rule"]
        out["label_line_minus_earlier"] = paired(lambda i: gap(a1[i], y1[i]) - gap(a0[i], y0[i]), n, idx_matrix)
        out["label_line_gap_ci"] = paired(lambda i: gap(a1[i], y1[i]), n, idx_matrix)
        out["earlier_rule_gap_ci"] = paired(lambda i: gap(a0[i], y0[i]), n, idx_matrix)
        agree_out[sname] = {"variants": list(keep), **out}

    result = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_pairs": n,
        "outcome": "each answer scored against the class of the supplied (donor) image; unweighted",
        "rules": {
            "label_line": "label read from the response's LABEL line (src/models/parsing.py::parse_prediction)",
            "earlier_rule": "whole-text search in fixed class order (parse_prediction_regex), as the stored runs",
        },
        "per_variant": per_variant,
        "agreement": agree_out,
    }
    out_path = run / "variant_rules.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for v, b in per_variant.items():
        print(f"variant {v}: differ {b['answers_read_differently']:3d}  "
              f"AUROC {b['label_line']['auroc']:.3f} / {b['earlier_rule']['auroc']:.3f}  "
              f"gap {b['label_line']['gap_vs_random']:+.3f} / {b['earlier_rule']['gap_vs_random']:+.3f}")
    for s, b in agree_out.items():
        print(f"agreement {s}: {b['label_line']['gap_vs_random']:+.3f} / {b['earlier_rule']['gap_vs_random']:+.3f}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
