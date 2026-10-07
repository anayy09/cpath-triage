"""
scripts/sample_flow.py

Where every validation patch went, and whether the prompt-selection pilot
touched the patches later used for evaluation.

Why this exists. V3 was chosen on macro F1 over a 450-patch balanced pilot. The
manuscript never said which patches those were, so a reader could not check
whether prompt development reused patches that later appear in the held-out
calibration partition, the consistency subset or the label-token sample. The
pilot predictions carry their PathMNIST array indices, so the overlap is a set
intersection rather than a matter of trust. A disjoint selection set cannot be
created after the fact; the equivalent test is to drop the pilot patches from
every validation-side analysis and see whether anything moves. This script does
that for the analyses outside the K=5 consistency family (which has its own
script): routing gap and AUROC on the full validation split, held-out
calibration with T refitted, and the Table S2 label-token contrasts.

Each quantity is computed twice with the same code, on the published sample and
on that sample minus the pilot, so the comparison is like for like. The
published value is carried beside both as a check that the code reproduces it.

Outputs:
    results/data/sample_flow.json
    results/data/pilot_arr_idx.csv
    results/data/pilot_excluded_headlines.json

Usage:
    python scripts/sample_flow.py [--n-boot 1000] [--seed 42]
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from confidence_information import auroc_midrank, bootstrap_auroc
from logprob_comparison import paired_bootstrap
from routing_ci import acc_coverage_auc, bootstrap_gap

from src.eval.calibration import (
    TemperatureScaler,
    adaptive_ece_score,
    brier_binary,
    ece_score,
)
from src.triage.router import risk_coverage_curve

VLMS = ("medgemma-27b-it", "gemma-3-27b-it")
PILOT_VERSIONS = ("V2", "V3", "V4", "V4B", "V5")
ZS = PROJECT_ROOT / "results" / "zeroshot"


def load_pilot() -> pd.DataFrame:
    """The pilot sample, asserted identical across every prompt version that ran."""
    frames = {v: pd.read_parquet(ZS / "medgemma-27b-it" / v / "predictions.parquet")
              for v in PILOT_VERSIONS}
    ref = set(frames["V3"]["arr_idx"])
    for v, f in frames.items():
        if set(f["arr_idx"]) != ref or len(f) != 450:
            raise AssertionError(f"pilot sample for {v} differs from V3")
    return frames["V3"][["arr_idx", "true_label_idx", "true_label_name"]].sort_values("arr_idx")


def full_split(model: str, split: str) -> pd.DataFrame:
    suffix = "V3_full" if split == "val" else "V3_full_test"
    df = pd.read_parquet(ZS / model / suffix / "predictions.parquet")
    return df.assign(correct=df["pred_label"] == df["true_label_name"])


def calibration_partitions(model: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """calibrate.py's split: parse failures dropped, default_rng(seed) permutation, 80/20."""
    df = full_split(model, "val")
    df = df[df["pred_label"] != "unknown"].copy()
    idx = np.random.default_rng(seed).permutation(len(df))
    cut = int(len(df) * 0.8)
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def routing_and_auroc(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    conf = df["pred_confidence"].to_numpy(float)
    correct = df["correct"].to_numpy(bool)
    auc = acc_coverage_auc(conf, correct)
    rnd = float(correct.mean())  # random routing AUC is the base accuracy exactly
    boot = bootstrap_gap(conf, correct, n_boot, seed)
    auroc_boot = bootstrap_auroc(conf, correct, n_boot, seed)
    return {
        "n": len(df),
        "accuracy": round(float(correct.mean()), 4),
        "selective_accuracy_auc": round(float(auc), 4),
        "random_auc": round(float(rnd), 4),
        "gap_point": round(float(auc - rnd), 4),
        "gap_ci": [boot["ci_2.5"], boot["ci_97.5"]],
        "gap_excludes_zero": boot["excludes_zero"],
        "conf_correct_auroc": round(auroc_midrank(conf, correct), 4),
        "auroc_ci": [auroc_boot["ci_2.5"], auroc_boot["ci_97.5"]],
    }


def calibration(cal: pd.DataFrame, ev: pd.DataFrame) -> dict:
    scaler = TemperatureScaler()
    T = scaler.fit_scalar(cal["pred_confidence"].to_numpy(float), cal["correct"].to_numpy(bool))
    conf = ev["pred_confidence"].to_numpy(float)
    corr = ev["correct"].to_numpy(bool)
    return {
        "n_cal": len(cal),
        "n_eval": len(ev),
        "T": round(float(T), 4),
        "ece_raw": round(float(ece_score(conf, corr)), 4),
        "ece_adaptive": round(float(adaptive_ece_score(conf, corr)), 4),
        "brier_raw": round(float(brier_binary(conf, corr)), 4),
        "ece_calibrated": round(float(ece_score(scaler.transform_scalar(conf), corr)), 4),
    }


def label_token(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    correct = df["correct"].to_numpy(bool)
    lt = df["logprob_conf"].to_numpy(float)
    vb = df["verbalized_conf"].to_numpy(float)
    a_lt = risk_coverage_curve(lt, correct, tie_break="expected")["auc"]
    a_vb = risk_coverage_curve(vb, correct, tie_break="expected")["auc"]
    rnd = float(correct.mean())  # random routing AUC is the base accuracy exactly
    out = {"n": len(df), "accuracy": round(float(correct.mean()), 4)}
    for name, point, b in (
        ("label_token_minus_random", a_lt - rnd, paired_bootstrap(lt, None, correct, n_boot, seed)),
        ("label_token_minus_verbalized", a_lt - a_vb, paired_bootstrap(lt, vb, correct, n_boot, seed)),
    ):
        out[name] = {"gap_point": round(float(point), 4), "ci": [b["ci_2.5"], b["ci_97.5"]],
                     "excludes_zero": b["excludes_zero"]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"seed={args.seed} n_boot={args.n_boot}")

    pilot = load_pilot()
    pilot_ids = set(pilot["arr_idx"])
    val_labels = full_split("medgemma-27b-it", "val").set_index("arr_idx")["true_label_name"]
    if not (val_labels.loc[pilot["arr_idx"]].to_numpy() == pilot["true_label_name"].to_numpy()).all():
        raise AssertionError("pilot labels do not match the validation split at those indices")

    cons_val = set(pd.read_parquet(
        PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it" / "V3" / "predictions.parquet"
    )["arr_idx"])

    flow: dict = {
        "pilot": {
            "n": len(pilot_ids),
            "split": "validation (PathMNIST val, NCT-CRC-HE-100K); labels verified against the split",
            "identical_across_versions": list(PILOT_VERSIONS),
            "sampling": "src.data.pathmnist.balanced_indices, 50 per class, seed 42",
            "class_counts": pilot["true_label_name"].value_counts().sort_index().to_dict(),
        },
        "overlaps_with_pilot": {},
        "external_split": "the pilot drew only on validation, so no external patch was used in prompt selection",
    }
    for model in VLMS:
        cal, ev = calibration_partitions(model, args.seed)
        lt_val = set(pd.read_parquet(
            PROJECT_ROOT / "results" / "logprob_confidence" / model / "val" / "predictions.parquet"
        )["arr_idx"])
        flow["overlaps_with_pilot"][model] = {
            "calibration_set": {"n": len(cal), "pilot_overlap": len(pilot_ids & set(cal["arr_idx"]))},
            "held_out_evaluation_partition": {"n": len(ev),
                                              "pilot_overlap": len(pilot_ids & set(ev["arr_idx"]))},
            "label_token_validation_sample": {"n": len(lt_val), "pilot_overlap": len(pilot_ids & lt_val),
                                              "same_as_evaluation_partition": lt_val == set(ev["arr_idx"])},
        }
    flow["overlaps_with_pilot"]["consistency_validation_subset"] = {
        "n": len(cons_val), "pilot_overlap": len(pilot_ids & cons_val)}
    m_ev = set(calibration_partitions("medgemma-27b-it", args.seed)[1]["arr_idx"])
    g_ev = set(calibration_partitions("gemma-3-27b-it", args.seed)[1]["arr_idx"])
    flow["evaluation_partitions_differ_between_models"] = {
        "reason": ("calibrate.py drops parse failures before permuting, and Gemma-3 has two on "
                   "validation, so its 80/20 permutation runs over 10,002 rows rather than 10,004"),
        "shared_patches": len(m_ev & g_ev),
        "n_medgemma": len(m_ev),
        "n_gemma3": len(g_ev),
    }

    out_dir = PROJECT_ROOT / "results" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pilot_arr_idx.csv"
    pilot.to_csv(csv_path, index=False)
    flow_path = out_dir / "sample_flow.json"
    flow_path.write_text(json.dumps(flow, indent=2), encoding="utf-8")
    print(json.dumps(flow["overlaps_with_pilot"], indent=1))

    pub_route = json.loads((PROJECT_ROOT / "results" / "routing" / "routing_auc_ci.json")
                           .read_text(encoding="utf-8"))["results"]
    pub_info = json.loads((PROJECT_ROOT / "results" / "calibration" / "confidence_information.json")
                          .read_text(encoding="utf-8"))["results"]
    pub_extras = json.loads((PROJECT_ROOT / "results" / "calibration" / "calibration_extras.json")
                            .read_text(encoding="utf-8"))
    pub_lp = json.loads((PROJECT_ROOT / "results" / "logprob_confidence" / "comparison.json")
                        .read_text(encoding="utf-8"))["runs"]

    headlines: dict = {}
    for model in VLMS:
        val = full_split(model, "val")
        cal, ev = calibration_partitions(model, args.seed)
        lt = pd.read_parquet(PROJECT_ROOT / "results" / "logprob_confidence" / model / "val"
                             / "predictions.parquet")
        lt = lt[lt["logprob_conf"].notna() & ~lt["call_failed"]]
        r, i = pub_route[f"{model}|val"], pub_info[model]["val_full"]
        pc = json.loads((PROJECT_ROOT / "results" / "calibration" / model / "V3_full"
                         / "calibration_params.json").read_text(encoding="utf-8"))
        b = pub_lp[f"{model}|val"]["bootstrap"]
        headlines[model] = {
            "routing_full_validation": {
                "published": {"gap_point": r["gap_point"],
                              "gap_ci": [r["gap_bootstrap"]["ci_2.5"], r["gap_bootstrap"]["ci_97.5"]],
                              "conf_correct_auroc": i["conf_correct_auroc"],
                              "auroc_ci": [i["conf_correct_auroc_bootstrap"]["ci_2.5"],
                                           i["conf_correct_auroc_bootstrap"]["ci_97.5"]]},
                "recomputed_full": routing_and_auroc(val, args.n_boot, args.seed),
                "pilot_excluded": routing_and_auroc(val[~val["arr_idx"].isin(pilot_ids)],
                                                    args.n_boot, args.seed),
            },
            "calibration_held_out": {
                "published": {"T": pc["T"], "ece_raw": pc["ece_before"],
                              "ece_adaptive": pub_extras["results"][model]["eval_partition"]["ece_adaptive"],
                              "brier_raw": pc["brier_before"], "ece_calibrated": pc["ece_after"]},
                "recomputed_full": calibration(cal, ev),
                "pilot_excluded": calibration(cal[~cal["arr_idx"].isin(pilot_ids)],
                                              ev[~ev["arr_idx"].isin(pilot_ids)]),
            },
            "label_token_validation": {
                "published": {k: {"gap_point": b[k]["gap_point"], "ci": [b[k]["ci_2.5"], b[k]["ci_97.5"]]}
                              for k in ("label_token_minus_random", "label_token_minus_verbalized")},
                "recomputed_full": label_token(lt, args.n_boot, args.seed),
                "pilot_excluded": label_token(lt[~lt["arr_idx"].isin(pilot_ids)], args.n_boot, args.seed),
            },
        }
        print(f"\n{model}")
        for block, v in headlines[model].items():
            print(f"  {block}")
            for tag in ("published", "recomputed_full", "pilot_excluded"):
                print(f"    {tag:16} {json.dumps(v[tag])}")

    out_path = out_dir / "pilot_excluded_headlines.json"
    out_path.write_text(json.dumps({
        "seed": args.seed,
        "n_boot": args.n_boot,
        "pilot_n": len(pilot_ids),
        "note": ("Each block gives the published value, the same quantity recomputed here on the "
                 "published sample, and the quantity with the 450 pilot patches removed. Random "
                 "references are the exact base accuracy; AUCs resolve ties by "
                 "expectation."),
        "models": headlines,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {flow_path}\nSaved: {csv_path}\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
