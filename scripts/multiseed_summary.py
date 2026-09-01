"""
scripts/multiseed_summary.py

Multi-seed summary for the CNN resolution comparison (revision item C-17,
Decision 3 in docs/DECISIONS_SR_R1.md).

Reviewer 2's objection is that "the higher-resolution 224 px model is the more
brittle" appears in the Abstract and Conclusion but rests on one seed per
resolution with no variance estimate. This aggregates the seeds and applies the
decision rule that was written down before the seeds were looked at:

    Report mean and SD over three seeds for validation accuracy, test accuracy,
    delta-accuracy and test ECE at both resolutions. If the mean plus or minus
    one SD intervals for delta-accuracy overlap between the two resolutions, the
    resolution-brittleness claim is demoted to a preliminary observation in
    Section 5.3 and removed from the Abstract and the Conclusion. If they are
    disjoint, the claim stays and the SDs are reported alongside it.

The rule is evaluated here mechanically and its verdict printed, so the decision
is not made by reading the numbers generously.

Test ECE needs per-seed predictions, so each seed must be evaluated with
scripts/eval_cnn.py or scripts/eval_cnn_224px.py using --run-dir before this
runs. Seeds with no predictions on disk are reported as missing rather than
silently skipped.

Outputs:
    results/cnn/multiseed_summary.json

Usage:
    python scripts/multiseed_summary.py
    python scripts/multiseed_summary.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CNN_ROOT = PROJECT_ROOT / "results" / "cnn"
N_BINS = 15
RESOLUTIONS = ("64px", "224px")


def run_dir(resolution: str, seed: int, base_seed: int) -> Path:
    """Seed 42 lives in the original directory; later seeds carry a suffix."""
    stem = f"resnet18_{resolution}"
    return CNN_ROOT / (stem if seed == base_seed else f"{stem}_seed{seed}")


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def collect(resolution: str, seed: int, base_seed: int) -> dict:
    d = run_dir(resolution, seed, base_seed)
    out: dict = {"seed": seed, "run_dir": str(d.relative_to(PROJECT_ROOT)), "present": False}

    metrics_path = d / "metrics.json"
    if not metrics_path.exists():
        out["missing"] = f"{metrics_path.relative_to(PROJECT_ROOT)} not found"
        return out

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    out.update({
        "present": True,
        "val_accuracy": m.get("val_accuracy"),
        "test_accuracy": m.get("test_accuracy"),
        "val_macro_f1": m.get("val_macro_f1"),
        "test_macro_f1": m.get("test_macro_f1"),
        "epochs_trained": m.get("epochs_trained"),
        "best_epoch": m.get("best_epoch"),
        "seed_recorded_in_metrics": m.get("seed"),
        "train_data_source": m.get("train_data_source"),
    })
    if out["val_accuracy"] is not None and out["test_accuracy"] is not None:
        out["delta_accuracy"] = round(out["test_accuracy"] - out["val_accuracy"], 4)

    # Guard against a mislabelled run: the metrics file records the seed it used.
    if out["seed_recorded_in_metrics"] not in (None, seed):
        out["WARNING"] = (
            f"directory implies seed {seed} but metrics.json records "
            f"{out['seed_recorded_in_metrics']}"
        )

    test_pred = d / "test_predictions.parquet"
    if test_pred.exists():
        df = pd.read_parquet(test_pred)
        out["test_ece"] = round(
            ece(df["confidence"].to_numpy(float), df["correct"].to_numpy(bool)), 4
        )
        out["test_accuracy_from_predictions"] = round(float(df["correct"].mean()), 4)
    else:
        out["test_ece"] = None
        out["test_ece_missing"] = (
            f"{test_pred.relative_to(PROJECT_ROOT)} not found; run the eval script "
            f"with --run-dir {out['run_dir']}"
        )
    return out


def agg(values: list[float]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": len(arr),
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
        "values": [round(float(v), 4) for v in arr],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--base-seed", type=int, default=42,
                        help="The seed whose run lives in the unsuffixed directory.")
    args = parser.parse_args()

    per_res: dict[str, dict] = {}
    for res in RESOLUTIONS:
        runs = [collect(res, s, args.base_seed) for s in args.seeds]
        present = [r for r in runs if r["present"]]
        per_res[res] = {
            "runs": runs,
            "n_present": len(present),
            "n_expected": len(args.seeds),
            "val_accuracy": agg([r.get("val_accuracy") for r in present]),
            "test_accuracy": agg([r.get("test_accuracy") for r in present]),
            "delta_accuracy": agg([r.get("delta_accuracy") for r in present]),
            "test_ece": agg([r.get("test_ece") for r in present]),
        }

    # ── The pre-registered rule ──
    a, b = per_res["64px"]["delta_accuracy"], per_res["224px"]["delta_accuracy"]
    verdict: dict = {
        "rule": (
            "If the mean +/- 1 SD intervals for delta-accuracy overlap between the two "
            "resolutions, demote the resolution-brittleness claim to a preliminary "
            "observation in Section 5.3 and remove it from the Abstract and Conclusion. "
            "If disjoint, keep the claim and report the SDs alongside it."
        ),
        "pre_registered": True,
        "evaluable": bool(a["mean"] is not None and b["mean"] is not None and a["n"] > 1 and b["n"] > 1),
    }
    if verdict["evaluable"]:
        lo_a, hi_a = a["mean"] - a["sd"], a["mean"] + a["sd"]
        lo_b, hi_b = b["mean"] - b["sd"], b["mean"] + b["sd"]
        overlap = not (hi_a < lo_b or hi_b < lo_a)
        verdict.update({
            "interval_64px": [round(lo_a, 4), round(hi_a, 4)],
            "interval_224px": [round(lo_b, 4), round(hi_b, 4)],
            "intervals_overlap": bool(overlap),
            "outcome": "DEMOTE the claim" if overlap else "KEEP the claim, report SDs",
        })
    else:
        need = []
        for res in RESOLUTIONS:
            r = per_res[res]
            if r["n_present"] < r["n_expected"]:
                missing = [x["seed"] for x in r["runs"] if not x["present"]]
                need.append(f"{res}: seeds {missing} not trained")
        verdict["outcome"] = "NOT YET EVALUABLE"
        verdict["blocking"] = need

    out = {
        "seeds": args.seeds,
        "base_seed": args.base_seed,
        "ece_bins": N_BINS,
        "resolutions": per_res,
        "pre_registered_decision": verdict,
    }

    CNN_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CNN_ROOT / "multiseed_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Multi-seed CNN summary ===")
    for res in RESOLUTIONS:
        r = per_res[res]
        print(f"\nResNet-18 {res}  ({r['n_present']}/{r['n_expected']} seeds present)")
        for run in r["runs"]:
            if run["present"]:
                ece_s = f"{run['test_ece']:.4f}" if run.get("test_ece") is not None else "no preds"
                print(f"  seed {run['seed']}: val={run['val_accuracy']:.4f} "
                      f"test={run['test_accuracy']:.4f} "
                      f"delta={run.get('delta_accuracy', float('nan')):+.4f} "
                      f"test ECE={ece_s}")
                if "WARNING" in run:
                    print(f"    WARNING: {run['WARNING']}")
            else:
                print(f"  seed {run['seed']}: MISSING ({run.get('missing')})")
        for key in ("val_accuracy", "test_accuracy", "delta_accuracy", "test_ece"):
            s = r[key]
            if s["mean"] is not None:
                print(f"  {key:<16} mean {s['mean']:+.4f}  sd {s['sd']:.4f}  (n={s['n']})")

    print("\n=== Pre-registered decision ===")
    v = out["pre_registered_decision"]
    if v["evaluable"]:
        print(f"  64px  delta-accuracy mean +/- 1 SD: {v['interval_64px']}")
        print(f"  224px delta-accuracy mean +/- 1 SD: {v['interval_224px']}")
        print(f"  intervals overlap: {v['intervals_overlap']}")
        print(f"  OUTCOME: {v['outcome']}")
    else:
        print(f"  OUTCOME: {v['outcome']}")
        for b_ in v.get("blocking", []):
            print(f"    blocked by {b_}")

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
