"""
scripts/rescore_labels.py

Re-score every saved run that kept the model's response text, reading the label
from the LABEL line (src/models/parsing.py) instead of searching the whole
response.

Why this exists. The v5.0 parser searched the whole response, reasoning
sentence included, so a stated "background" whose reasoning mentions debris was
stored as debris, and a response listing every class was stored as whichever
class the search reached first. The runs that stored their text can be read
again without any API call; the original-family K=5 runs, the V5 family and the
label-token runs stored no text and keep the labels they have.

What it does, for each run:
  - checks that the v5.0 parser, applied to the stored text, reproduces every
    stored label and confidence, and refuses to write if it does not;
  - keeps the stored values beside the new ones (columns suffixed _regex), so
    the v5.0 labels stay available to any comparison against a run that could
    not be re-scored;
  - rewrites the predictions file in place with label-line values, and the
    run's metrics.json (and confusion matrix for zero-shot runs) from them.
The call logs (calls.jsonl) are never modified. Re-running is idempotent.

Runs covered: results/zeroshot/{model}/{version}/ (full-scale and pilots), and
under results/consistency/medgemma-27b-it/: V3W, V3W_test, controls/shuffle,
controls/swap, controls/no_image.

No API calls, no images.

Output:
    results/data/label_rescoring.json   per-run counts and accuracy both ways

Usage:
    python scripts/rescore_labels.py
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.k5 import CallLog, collect, summarize_k
from src.models.parsing import parse_prediction, parse_prediction_regex

TASK = "tissue_classification"
K = 5
ZS = PROJECT_ROOT / "results" / "zeroshot"
CONS = PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it"
OUT = PROJECT_ROOT / "results" / "data" / "label_rescoring.json"
K5_FIELDS = ["modal_label", "consistency_score", "uncertainty_score", "is_correct",
             "mean_textual_conf", "raw_labels", "raw_confs"]


def _check_regex_reproduces(texts: list[str], labels: list[str], confs: list[float], where: str) -> None:
    bad = [i for i, (t, lab, c) in enumerate(zip(texts, labels, confs))
           if parse_prediction_regex(t, TASK) != (lab, float(c))]
    if bad:
        raise SystemExit(f"{where}: the v5.0 parser does not reproduce {len(bad)} stored values "
                         f"(first at row {bad[0]}); refusing to re-score.")


def rescore_zeroshot(pred_path: Path) -> dict:
    from run_zeroshot import _plot_confusion_matrix, metrics_from_predictions

    df = pd.read_parquet(pred_path)
    if "pred_label_regex" not in df.columns:
        df["pred_label_regex"] = df["pred_label"]
        df["pred_confidence_regex"] = df["pred_confidence"]
    texts = df["raw_text"].fillna("").astype(str).tolist()
    _check_regex_reproduces(texts, df["pred_label_regex"].tolist(),
                            df["pred_confidence_regex"].tolist(), str(pred_path))

    parsed = [parse_prediction(t, TASK) for t in texts]
    df["pred_label"] = [p[0] for p in parsed]
    df["pred_confidence"] = [p[1] for p in parsed]
    df.to_parquet(pred_path, index=False)

    true = df["true_label_name"].astype(str).tolist()
    new = metrics_from_predictions(true, df["pred_label"].astype(str).tolist())
    old = metrics_from_predictions(true, df["pred_label_regex"].astype(str).tolist())
    metrics_path = pred_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if "regex_scoring" not in metrics:
        metrics["regex_scoring"] = {k: metrics.get(k) for k in
                                    ("accuracy", "macro_f1", "per_class_f1", "parse_failure_rate", "n_parse_failures")}
    n_unknown = int((df["pred_label"] == "unknown").sum())
    metrics.update(new)
    metrics.update({
        "label_scoring": "LABEL line (scripts/rescore_labels.py); v5.0 values under regex_scoring",
        "n_parse_failures": n_unknown,
        "parse_failure_rate": round(n_unknown / len(df), 4),
    })
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _plot_confusion_matrix(true, df["pred_label"].astype(str).tolist(), pred_path.parent / "confusion_matrix.png")

    changed = df["pred_label"] != df["pred_label_regex"]
    return {
        "n": len(df),
        "n_relabelled": int(changed.sum()),
        "n_confidence_changed": int((df["pred_confidence"] != df["pred_confidence_regex"]).sum()),
        "n_unknown_regex": int((df["pred_label_regex"] == "unknown").sum()),
        "n_unknown_label_line": n_unknown,
        "accuracy_regex": old["accuracy"],
        "accuracy_label_line": new["accuracy"],
        "macro_f1_regex": old["macro_f1"],
        "macro_f1_label_line": new["macro_f1"],
        "changes": {f"{a} -> {b}": n for (a, b), n in
                    Counter(zip(df.loc[changed, "pred_label_regex"], df.loc[changed, "pred_label"])).most_common()},
    }


def _log_check(log: CallLog, where: str) -> dict:
    recs = list(log.records.values())
    _check_regex_reproduces([r["raw_text"] or "" for r in recs], [r["label"] for r in recs],
                            [r["confidence"] for r in recs], where)
    by_variant: dict[str, dict[str, int]] = {}
    for r in recs:
        new = parse_prediction(r["raw_text"] or "", TASK)
        v = by_variant.setdefault(str(r["variant"]), {"n": 0, "relabelled": 0, "unknown_label_line": 0,
                                                       "multi_label": 0})
        v["n"] += 1
        v["relabelled"] += int(new[0] != r["label"])
        v["unknown_label_line"] += int(new[0] == "unknown")
        v["multi_label"] += int((r["raw_text"] or "").upper().count("LABEL:") > 1)
    return by_variant


def rescore_k5(run_dir: Path, key_col: str, truth_cols: dict[str, str]) -> dict:
    """
    Rebuild a K=5 predictions file from its call log. truth_cols maps an
    is_correct column name to the column holding the class it is scored against.
    """
    log = CallLog(run_dir / "calls.jsonl")
    per_variant = _log_check(log, str(run_dir))
    pred_path = run_dir / "predictions.parquet"
    df = pd.read_parquet(pred_path)
    regex_fields = [c for c in K5_FIELDS + list(truth_cols) if c in df.columns]
    if "raw_labels_regex" not in df.columns:
        for c in regex_fields:
            df[f"{c}_regex"] = df[c]
    first_truth = next(iter(truth_cols.values()))
    rows = []
    for _, r in df.iterrows():
        got = collect(log, str(int(r[key_col])), K)
        if got is None:
            raise SystemExit(f"{run_dir}: patch {r[key_col]} has missing calls")
        s = summarize_k(*got, str(r[first_truth]), K)
        for col, truth in truth_cols.items():
            s[col] = bool(s["modal_label"] == r[truth])
        if "is_correct" not in truth_cols:
            s.pop("is_correct")
        rows.append(s)
    new = pd.DataFrame(rows, index=df.index)
    for c in new.columns:
        df[c] = new[c]
    df.to_parquet(pred_path, index=False)

    out = {"per_variant": per_variant}
    for col in truth_cols:
        out[f"modal_{col}_regex"] = round(float(df[f"{col}_regex"].mean()), 4)
        out[f"modal_{col}_label_line"] = round(float(df[col].mean()), 4)
    out["n_patches_modal_changed"] = int((df["modal_label"] != df["modal_label_regex"]).sum())
    out["n_patches_consistency_changed"] = int(
        (df["consistency_score"] != df["consistency_score_regex"]).sum())
    if (run_dir / "metrics.json").exists() and "is_correct" in truth_cols:
        from run_consistency import _compute_and_save_metrics

        meta = json.loads((run_dir / "metrics.json").read_text())
        _compute_and_save_metrics(df, run_dir, meta.get("model", "medgemma-27b-it"),
                                  meta.get("base_prompt", run_dir.name))
    return out


def rescore_no_image(run_dir: Path) -> dict:
    log = CallLog(run_dir / "calls.jsonl")
    per_variant = _log_check(log, str(run_dir))
    path = run_dir / "outputs.parquet"
    df = pd.read_parquet(path)
    if "label_regex" not in df.columns:
        df["label_regex"] = df["label"]
        df["confidence_regex"] = df["confidence"]
    parsed = [parse_prediction(t or "", TASK) for t in df["raw_text"]]
    df["label"] = [p[0] for p in parsed]
    df["confidence"] = [p[1] for p in parsed]
    df.to_parquet(path, index=False)
    return {"per_variant": per_variant, "n_relabelled": int((df["label"] != df["label_regex"]).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Unused; logged for the project convention.")
    args = parser.parse_args()
    print(f"seed={args.seed} (no randomness)")

    summary: dict = {"rule": "label from the single LABEL line; several LABEL lines -> unknown; "
                             "no LABEL line -> whole-text search (v5.0 rule)", "zeroshot": {}, "k5": {}}
    for pred_path in sorted(ZS.glob("*/*/predictions.parquet")):
        if "raw_text" not in pd.read_parquet(pred_path, columns=None).columns:
            continue
        key = f"{pred_path.parent.parent.name}/{pred_path.parent.name}"
        summary["zeroshot"][key] = res = rescore_zeroshot(pred_path)
        print(f"{key}: {res['n_relabelled']} of {res['n']} relabelled; accuracy "
              f"{res['accuracy_regex']:.4f} -> {res['accuracy_label_line']:.4f}")

    k5_runs = {
        "V3W": ("arr_idx", {"is_correct": "true_label_name"}),
        "V3W_test": ("arr_idx", {"is_correct": "true_label_name"}),
        "controls/shuffle": ("arr_idx", {"is_correct": "true_label_name"}),
        "controls/swap": ("recipient_arr_idx", {"is_correct_vs_donor": "donor_true_name",
                                                "is_correct_vs_recipient": "recipient_true_name"}),
    }
    for name, (key_col, truth_cols) in k5_runs.items():
        run_dir = CONS / name
        if not (run_dir / "calls.jsonl").exists():
            print(f"{name}: no call log, skipped")
            continue
        summary["k5"][name] = res = rescore_k5(run_dir, key_col, truth_cols)
        print(f"{name}: modal label changed on {res['n_patches_modal_changed']} patches; "
              f"per-variant relabelled {[v['relabelled'] for v in res['per_variant'].values()]}")
    if (CONS / "controls/no_image/calls.jsonl").exists():
        summary["k5"]["controls/no_image"] = rescore_no_image(CONS / "controls/no_image")

    summary["not_rescored"] = {
        "consistency/medgemma-27b-it/V3, V3_test, V5": "no response text stored",
        "logprob_confidence/*": "no response text stored",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
