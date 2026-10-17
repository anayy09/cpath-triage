"""
scripts/audit_manuscript_numbers.py

Cross-check the numbers the manuscript and supplement print against the
artifacts they come from.

Why this exists. Most review passes on this manuscript opened on the same
defect: a number in the text that does not reconcile with the table beside it
or with the file it came from. Editorial care did not catch it; a script that
re-derives every printed value does.

Two kinds of check. Where an artifact stores the unrounded value (the
cohort-weighted K=5 analyses write an "exact" block), the printed string is
built from it, rounded once, in the manuscript's own format, and must appear
verbatim in the LaTeX. Where an artifact stores four decimals only, the printed
three-decimal value must lie within half a unit of the stored one, which is as
much as a four-decimal file can certify. Open [PENDING] markers are counted and
reported; they do not fail the audit until submission.

Usage:
    python scripts/audit_manuscript_numbers.py
    python scripts/audit_manuscript_numbers.py --final    # also fail on [PENDING]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONS = "results/consistency/medgemma-27b-it"
tex = (ROOT / "paper" / "latex" / "main.tex").read_text(encoding="utf-8")
supp = (ROOT / "paper" / "latex" / "supplementary.tex").read_text(encoding="utf-8")
letter_path = ROOT / "paper" / "RESPONSE_LETTER_SR_R3.md"
letter = letter_path.read_text(encoding="utf-8") if letter_path.exists() else ""

fails: list[str] = []
checks = 0


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def num(x: float, signed: bool) -> str:
    """One number as the manuscript prints it inside an interval or as a point."""
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    if s.lstrip("+-") == "0.000" and x < 0:
        s = "-0.000"
    return f"${s}$" if (signed or s.startswith("-")) else s


def gap_string(c: dict) -> str:
    """Point and interval from an artifact's exact block, e.g. $+0.123$ [0.105, 0.140]."""
    e = c["exact"]
    lo, hi = e["ci"]
    return f"{num(e['gap_point'], True)} [{num(lo, False)}, {num(hi, False)}]"


def present(label: str, needle: str, hay: str | None = None, where: str = "main.tex") -> None:
    global checks
    checks += 1
    if needle not in (tex if hay is None else hay):
        fails.append(f"{label}: not found in {where} -> {needle!r}")


def absent(label: str, needle: str, hay: str | None = None, where: str = "main.tex") -> None:
    global checks
    checks += 1
    if needle in (tex if hay is None else hay):
        fails.append(f"{label}: stale text still in {where} -> {needle!r}")


def near(label: str, artifact: float, printed: float, nd: int = 3) -> None:
    """A printed value is consistent with a four-decimal artifact."""
    global checks
    checks += 1
    if abs(float(artifact) - printed) > 0.5 * 10 ** -nd + 1e-9:
        fails.append(f"{label}: artifact {artifact} vs printed {printed}")


def exact(label: str, value: float, printed: float, nd: int = 3) -> None:
    """A printed value equals the unrounded artifact value rounded once."""
    global checks
    checks += 1
    if f"{value:.{nd}f}" != f"{printed:.{nd}f}":
        fails.append(f"{label}: artifact {value:.{nd}f} vs printed {printed:.{nd}f}")


# --- Table 6: routing by confidence -------------------------------------------
rt = load("results/routing/routing_auc_ci.json")["results"]
for key, (cal, rnd, gap, lo, hi) in {
    "medgemma-27b-it|val": (0.396, 0.402, -0.007, -0.012, -0.001),
    "medgemma-27b-it|test": (0.376, 0.342, 0.034, 0.027, 0.041),
    "gemma-3-27b-it|val": (0.470, 0.356, 0.114, 0.108, 0.119),
    "gemma-3-27b-it|test": (0.425, 0.309, 0.115, 0.108, 0.123),
    "resnet18_64px|test": (0.978, 0.911, 0.067, 0.061, 0.073),
}.items():
    r = rt[key]
    near(f"T6 cal {key}", r["cal_auc"], cal)
    near(f"T6 rnd {key}", r["random_auc"], rnd)
    near(f"T6 gap {key}", r["gap_point"], gap)
    near(f"T6 lo {key}", r["gap_bootstrap"]["ci_2.5"], lo)
    near(f"T6 hi {key}", r["gap_bootstrap"]["ci_97.5"], hi)
    checks += 1
    if r["random_auc"] != r["base_accuracy"]:
        fails.append(f"T6 {key}: random reference is not the exact base accuracy")

# --- Tables 8 to 10: cohort-weighted K=5 analyses (exact strings) ---------------
for sub, split in (("V3", "val"), ("V3_test", "test")):
    w = load(f"{CONS}/{sub}/weighted_routing.json")
    fo = load(f"{CONS}/{sub}/fixed_outcome.json")["weighted_cohort"]["contrasts"]
    cv = load(f"{CONS}/{sub}/confusability_valweights.json")
    lvo = load(f"{CONS}/{sub}/leave_variant_out.json")["weighted_cohort"]["contrasts"]
    wc = w["weighted_cohort"]["contrasts"]
    for name, c in (
        ("consistency - random", wc["consistency_minus_random"]),
        ("mean5 - random", wc["mean5_conf_minus_random"]),
        ("single - random", wc["single_query_conf_minus_random"]),
        ("consistency - mean5", wc["consistency_minus_mean5_conf"]),
        ("voting gain", wc["voting_gain_accuracy"]),
        ("fixed modal", fo["modal_outcome:consistency_minus_single_query_conf"]),
        ("fixed single", fo["single_query_outcome:consistency_minus_single_query_conf"]),
        ("confusability - consistency", cv["weighted_cohort"]["contrasts"]["valweights_minus_consistency"]),
    ):
        present(f"T9 {split} {name}", gap_string(c))
    for k, c in lvo.items():
        present(f"T10 {split} {k}", gap_string(c))
    # Table 8 AUCs, cohort and subset, rounded once from the unrounded values.
    order = ("single_query_conf|single_query", "random|single_query", "mean5_conf|modal",
             "consistency|modal", "entropy|modal")
    for blk in ("weighted_cohort", "unweighted_subset"):
        vals = w[blk]["auc_exact"]
        for k in order:
            present(f"T8 {split} {blk} {k}", f"{vals[k]:.3f}")
        present(f"T8 {split} {blk} confusability",
                f"{cv[blk]['auc_exact']['confusability_valweights|modal']:.3f}")
    # Subset point estimates in Table 9.
    uc = w["unweighted_subset"]["contrasts"]
    for k in ("consistency_minus_random", "mean5_conf_minus_random",
              "single_query_conf_minus_random", "consistency_minus_mean5_conf", "voting_gain_accuracy"):
        present(f"T9 subset {split} {k}", num(uc[k]["exact"]["gap_point"], True))
    m4 = load(f"{CONS}/{sub}/accuracy_by_agreement.json")["mean_confidence_without_variant_2"]
    present(f"4.5 mean of four without variant 2 {split}",
            gap_string(m4["mean4_conf_without_v2_minus_random"]))
    present(f"4.5 CNN-test confusability {split}",
            num(cv["weighted_cohort"]["contrasts"]["cnn_test_minus_consistency"]["exact"]["gap_point"], True))

if (ROOT / CONS / "V3W" / "weighted_routing.json").exists():
    v3w = load(f"{CONS}/V3W/weighted_routing.json")["weighted_cohort"]["contrasts"]
    for k in ("consistency_minus_random", "consistency_minus_mean5_conf", "voting_gain_accuracy",
              "mean5_conf_minus_random"):
        present(f"4.5 V3W {k}", gap_string(v3w[k]))
    lw = load(f"{CONS}/V3W/leave_variant_out.json")["weighted_cohort"]["contrasts"]
    drops = [lw[f"drop_{i}:consistency_minus_random"]["exact"]["gap_point"] for i in range(5)]
    present("4.5 V3W leave-one-out range",
            f"between {num(min(drops), True)} and {num(max(drops), True)}")
    s1_v3w = load("results/clustering/design_effect_sensitivity.json")["findings"][
        "Wording-only (V3W) consistency minus random (val)"]
    near("S1 V3W point", s1_v3w["point_estimate"], 0.090)
    checks += 1
    if not s1_v3w["break_even_rho"] > 1:
        fails.append("S1 V3W row: break-even printed as >1 but is not")

v5 = load(f"{CONS}/V5/weighted_routing.json")["contrasts_weighted"]
for k in ("consistency_minus_random", "mean5_conf_minus_random", "consistency_minus_mean5_conf"):
    present(f"4.5 V5 {k}", gap_string(v5[k]))

# --- Holm outcomes printed in Tables 9, 10 and S6 ------------------------------
holm = {r["id"]: r for r in load("results/multiplicity/holm.json")["contrasts"]}
for cid in ("consistency_vs_random", "consistency_vs_mean5", "consistency_vs_single_query_fixed_outcome"):
    for split in ("val", "test"):
        checks += 1
        if not holm[f"{cid}_{split}"]["survives_holm"]:
            fails.append(f"Holm: {cid}_{split} printed as holding but does not")
checks += 1
if holm["v3_edits_only_consistency_vs_random_test"]["survives_holm"] or not \
        holm["v3_edits_only_consistency_vs_random_val"]["survives_holm"]:
    fails.append("Holm: Table 10 caption says V3 edits hold on validation only")

# --- Table S2 and Section 4.6 ---------------------------------------------------
lp = load("results/logprob_confidence/comparison.json")["runs"]
for (run, contrast), printed in {
    ("medgemma-27b-it|val", "label_token_minus_random"): 0.049,
    ("medgemma-27b-it|val", "verbalized_minus_random"): -0.008,
    ("medgemma-27b-it|test", "label_token_minus_random"): -0.056,
    ("medgemma-27b-it|test", "verbalized_minus_random"): 0.021,
    ("gemma-3-27b-it|val", "label_token_minus_random"): 0.106,
    ("gemma-3-27b-it|val", "verbalized_minus_random"): 0.117,
    ("gemma-3-27b-it|test", "label_token_minus_random"): 0.133,
    ("gemma-3-27b-it|test", "verbalized_minus_random"): 0.115,
    ("medgemma-27b-it|val", "label_token_minus_verbalized"): 0.057,
    ("medgemma-27b-it|test", "label_token_minus_verbalized"): -0.077,
}.items():
    c = lp[run]["bootstrap"][contrast]
    near(f"S2 {run} {contrast}", c["gap_point"], printed)
    checks += 1
    if not c["point_inside_ci"]:
        fails.append(f"S2 {run} {contrast}: point outside its own CI")
    checks += 1
    if lp[run]["label_token_confidence"]["random_routing_auc"] != lp[run]["accuracy"]:
        fails.append(f"S2 {run}: random reference is not the exact accuracy")

cs = load("results/logprob_confidence/censoring_sensitivity.json")["runs"]
for run, contrast, printed in (
    ("medgemma-27b-it|val", "label_token_minus_verbalized", -0.076),
    ("medgemma-27b-it|val", "label_token_minus_random", -0.064),
    ("medgemma-27b-it|test", "label_token_minus_verbalized", -0.112),
    ("medgemma-27b-it|test", "label_token_minus_random", -0.099),
    ("gemma-3-27b-it|val", "label_token_minus_random", 0.046),
    ("gemma-3-27b-it|test", "label_token_minus_random", 0.060),
):
    near(f"4.6 all-nine {run} {contrast}", cs[run]["all_nine_classes_present"][contrast]["gap_point"], printed)
near("4.6 eight classes val", cs["medgemma-27b-it|val"]["by_n_classes_stratum"]["8"]
     ["label_token_minus_verbalized"]["gap_point"], 0.084)
for run, (m_val, m_test) in (("medgemma-27b-it", (8.22, 8.25)), ("gemma-3-27b-it", (6.71, 6.75))):
    near(f"3.6 mean classes {run} val", cs[f"{run}|val"]["censoring_profile"]["mean_n_classes"], m_val, nd=2)
    near(f"3.6 mean classes {run} test", cs[f"{run}|test"]["censoring_profile"]["mean_n_classes"], m_test, nd=2)

# --- Table S1 -------------------------------------------------------------------
de = load("results/clustering/design_effect_sensitivity.json")["findings"]
for name, (pt, be) in {
    "MedGemma routing gap vs random (val)": (-0.007, 0.003),
    "MedGemma routing gap vs random (test)": (0.034, 0.073),
    "Gemma-3 routing gap vs random (test)": (0.115, 0.868),
    "CNN-64 routing gap vs random (test)": (0.067, 0.446),
    "Consistency minus random (test)": (0.106, 0.684),
    "Mean-of-5 confidence minus random (val)": (0.026, 0.080),
    "Mean-of-5 confidence minus random (test)": (0.053, 0.128),
    "Consistency minus mean-of-5 confidence (val)": (0.098, 0.842),
    "Consistency minus mean-of-5 confidence (test)": (0.054, 0.068),
    "Consistency minus single-query confidence, fixed outcome (test)": (0.081, 0.186),
    "MedGemma label-token minus random (val)": (0.049, 0.215),
    "MedGemma label-token minus random (test)": (-0.056, 0.091),
    "MedGemma label-token minus verbalized (val)": (0.057, 0.177),
    "MedGemma label-token minus verbalized (test)": (-0.077, 0.108),
}.items():
    near(f"S1 point {name}", de[name]["point_estimate"], pt)
    near(f"S1 rho {name}", de[name]["break_even_rho"], be)
for name in ("Gemma-3 routing gap vs random (val)", "Consistency minus random (val)",
             "Consistency minus single-query confidence, fixed outcome (val)"):
    checks += 1
    if not de[name]["break_even_rho"] > 1:
        fails.append(f"S1: {name} printed as >1 but is not")

# --- Methods facts -------------------------------------------------------------
flow = load("results/data/sample_flow.json")["overlaps_with_pilot"]
for label, value in (("370", flow["medgemma-27b-it"]["calibration_set"]["pilot_overlap"]),
                     ("80", flow["medgemma-27b-it"]["held_out_evaluation_partition"]["pilot_overlap"]),
                     ("367", flow["gemma-3-27b-it"]["calibration_set"]["pilot_overlap"]),
                     ("83", flow["gemma-3-27b-it"]["held_out_evaluation_partition"]["pilot_overlap"]),
                     ("78", flow["consistency_validation_subset"]["pilot_overlap"])):
    checks += 1
    if str(value) != label:
        fails.append(f"3.1 pilot overlap printed {label}, artifact {value}")
nll = load("results/calibration/nll_curve.json")["models"]
near("3.3 MedGemma NLL T=1", nll["medgemma-27b-it"]["nll_at"]["1.0"], 2.117)
near("3.3 MedGemma NLL T=200", nll["medgemma-27b-it"]["nll_at"]["200.0"], 0.695)
near("3.3 Gemma-3 NLL T=1", nll["gemma-3-27b-it"]["nll_at"]["1.0"], 1.492)
near("3.3 Gemma-3 NLL T=200", nll["gemma-3-27b-it"]["nll_at"]["200.0"], 0.694)
near("3.3 log-odds on errors", nll["medgemma-27b-it"]["conditions"]["logodds_mass_on_errors"], 16546, nd=0)
near("3.3 log-odds on correct", nll["medgemma-27b-it"]["conditions"]["logodds_mass_on_correct"], 10737, nd=0)
checks += 1
if nll["gemma-3-27b-it"]["conditions"]["n_logit_negative"] != 29:
    fails.append("3.3: Gemma-3 confidences below 0.5 is not 29")

ctrl = load(f"{CONS}/controls/summary.json")
if "swap" in ctrl:
    sw = ctrl["swap"]
    near("4.5 swap donor match %", 100 * sw["per_call_label_match_donor_stored"], 98.6, nd=1)
    near("4.5 swap recipient match %", 100 * sw["per_call_label_match_recipient_stored"], 24.1, nd=1)
    near("4.5 swap corr donor", sw["consistency_corr_with_donor_stored"], 0.97, nd=2)
    near("4.5 swap corr recipient", sw["consistency_corr_with_recipient_stored"], -0.10, nd=2)
if "shuffle" in ctrl and ctrl["shuffle"].get("weighted_cohort"):
    shw = ctrl["shuffle"]["weighted_cohort"]
    present("4.5 shuffle gap", gap_string(shw["contrasts"]["consistency_minus_random"]))
    present("4.5 real minus shuffled", gap_string(shw["contrasts"]["real_minus_shuffled_gap"]))
    near("4.5 shuffled cohort accuracy %", 100 * shw["auc"]["shuffled_random"], 11.3, nd=1)
    near("4.5 shuffled background share %",
         100 * ctrl["shuffle"]["shuffled"]["modal_label_distribution"]["background"], 89, nd=0)
if "v3w" in ctrl:
    checks += 1
    if ctrl["v3w"]["variant0_disagreements_with_stored_v3_variant0"] != 30:
        fails.append("4.5 V3W drift count is not 30")
if "no_image" in ctrl:
    for cond in ("text_only", "grey"):
        checks += 1
        if not ctrl["no_image"][cond]["all_repeats_identical"]:
            fails.append(f"4.5 no-image {cond}: repeats are not identical")

# --- PLIP baseline (Section 4.3, Table S7) --------------------------------------
if (ROOT / "results/plip/summary.json").exists():
    pl = load("results/plip/summary.json")
    ens, probe = pl["zeroshot_ensemble"], pl["probe"]["test"]
    for split, (acc, auroc, gap, lo, hi) in (("val", (49.7, 0.586, 0.090, 0.082, 0.098)),
                                             ("test", (53.8, 0.593, 0.098, 0.088, 0.107))):
        r = ens[split]
        near(f"4.3 PLIP accuracy {split}", 100 * r["accuracy"], acc, nd=1)
        near(f"4.3 PLIP AUROC {split}", r["auroc"], auroc)
        near(f"4.3 PLIP gap {split}", r["gap_point"], gap)
        near(f"4.3 PLIP gap lo {split}", r["gap_bootstrap"]["ci_2.5"], lo)
        near(f"4.3 PLIP gap hi {split}", r["gap_bootstrap"]["ci_97.5"], hi)
        checks += 1
        if r["random_auc"] != r["accuracy"]:
            fails.append(f"PLIP {split}: random reference is not the exact accuracy")
    near("4.3 PLIP probe accuracy", 100 * probe["accuracy"], 94.7, nd=1)
    near("4.3 PLIP probe gap", probe["gap_point"], 0.044)
    checks += 1
    if probe["leakage_inflated"] or not pl["probe"]["val"]["leakage_inflated"]:
        fails.append("PLIP probe: leakage flags are not val-only")
    present("Table S7 in supplement", "label{tab:plip}", supp, "supplementary.tex")

# --- Round 4: per-variant confidence, calibration maps, K, budgets, clip ---------
vcw = load(f"{CONS}/variant_confidence_weighted.json")["runs"]
for run, split in (("V3", "val"), ("V3_test", "test")):
    v2 = vcw[run]["variants"]["variant_2"]
    present(f"4.5 variant 2 gap {split}", gap_string(v2["cohort_gap_vs_random"]))
    lo, hi = v2["cohort_auroc_ci"]
    present(f"4.5 variant 2 AUROC {split}", f"{v2['cohort_auroc']:.3f} [{lo:.3f}, {hi:.3f}]")
    present(f"4.5 variant 2 accuracy {split}", f"{100 * v2['cohort_accuracy']:.1f}\\%")
present("4.5 V5 single-query gap", gap_string(vcw["V5"]["variants"]["variant_0"]["cohort_gap_vs_random"]))
present("4.5 V3W worst paraphrase", gap_string(vcw["V3W"]["variants"]["variant_3"]["cohort_gap_vs_random"]))
near("4.5 V5 variant 2 AUROC", vcw["V5"]["variants"]["variant_2"]["cohort_auroc"], 0.620)

cal = load("results/calibration/calibration_alternatives.json")["models"]
mg, g3 = cal["medgemma-27b-it"], cal["gemma-3-27b-it"]
near("4.2 Platt ECE", mg["eval"]["platt"]["ece"], 0.006)
near("4.2 isotonic ECE", mg["eval"]["isotonic"]["ece"], 0.005)
near("4.2 MedGemma isotonic Brier", mg["eval"]["isotonic"]["brier"], 0.239)
near("4.2 MedGemma constant Brier", mg["eval"]["constant"]["brier"], 0.240)
near("4.2 Platt slope", mg["params"]["platt_slope"], -0.024)
near("4.2 MedGemma resolution", mg["eval"]["raw"]["brier_decomposition"]["resolution"], 0.0007, nd=4)
near("4.2 MedGemma external Platt Brier", mg["external"]["platt"]["brier"], 0.231)
near("4.2 MedGemma external isotonic Brier", mg["external"]["isotonic"]["brier"], 0.228)
near("4.2 MedGemma external constant Brier", mg["external"]["constant"]["brier"], 0.229)
near("4.2 Gemma-3 isotonic Brier", g3["eval"]["isotonic"]["brier"], 0.200)
near("4.2 Gemma-3 constant Brier", g3["eval"]["constant"]["brier"], 0.230)
near("4.2 Gemma-3 external isotonic Brier", g3["external"]["isotonic"]["brier"], 0.186)
near("4.2 Gemma-3 external constant Brier", g3["external"]["constant"]["brier"], 0.216)
near("4.2 Gemma-3 resolution", g3["eval"]["raw"]["brier_decomposition"]["resolution"], 0.030)

kab = load(f"{CONS}/k_ablation.json")["splits"]["val"]
for k, printed in (("2", 0.070), ("3", 0.090), ("4", 0.102)):
    near(f"4.5 K ablation V3 K={k}", kab["V3"]["by_k"][k]["mean_gap"], printed)
near("4.5 K ablation pooled K=9", kab["pooled"]["by_k"]["9"]["mean_gap"], 0.124)
near("4.5 K ablation pooled K=8 without variant 2", kab["pooled"]["by_k"]["8"]["mean_gap_without_variant_2"], 0.067)
near("4.5 K ablation V3W K=2", kab["V3W"]["by_k"]["2"]["mean_gap"], 0.063)
kab_ext = load(f"{CONS}/k_ablation.json")["splits"]["test"]
near("4.5 K ablation external pooled K=9", kab_ext["pooled"]["by_k"]["9"]["mean_gap"], 0.062)
near("4.5 K ablation external pooled K=8 without variant 2",
     kab_ext["pooled"]["by_k"]["8"]["mean_gap_without_variant_2"], -0.017)
v3w_ext_k = [kab_ext["V3W"]["by_k"][k]["mean_gap"] for k in ("2", "3", "4", "5")]
near("4.5 K ablation external V3W min", min(v3w_ext_k), 0.036)
near("4.5 K ablation external V3W max", max(v3w_ext_k), 0.049)
near("4.5 K ablation without variant 2 K=2", kab["V3"]["by_k"]["2"]["mean_gap_without_variant_2"], 0.023)
near("4.5 K ablation without variant 2 K=4", kab["V3"]["by_k"]["4"]["mean_gap_without_variant_2"], 0.033)

bud = load("results/routing/budget_cost_table.json")["rows"]
for b, printed in (("0.10", 42.5), ("0.20", 44.2), ("0.30", 45.4)):
    near(f"4.5 agreement at budget {b}", 100 * bud["MedGemma V3 agreement|val"]["auto_confirm_accuracy"][b], printed, nd=1)
near("4.5 Gemma-3 budget gain val low", 100 * (bud["gemma-3-27b-it verbalized|val"]["auto_confirm_accuracy"]["0.10"]
                                                - bud["gemma-3-27b-it verbalized|val"]["base_accuracy"]), 3.7, nd=1)
near("4.5 Gemma-3 budget gain val high", 100 * (bud["gemma-3-27b-it verbalized|val"]["auto_confirm_accuracy"]["0.30"]
                                                 - bud["gemma-3-27b-it verbalized|val"]["base_accuracy"]), 9.5, nd=1)

clip = load("results/calibration/nll_clip_sensitivity.json")["models"]["medgemma-27b-it"]
present("3.3 confidences at 100", f"{clip['n_at_100']} of MedGemma's 8,003")
near("3.3 NLL at T=1 with clip 1e-2", clip["by_clip"]["0.01"]["nll_T1"], 1.844)
checks += 1
if not all(r["sufficient_condition_holds"] and r["monotone_decreasing_on_grid"] for r in clip["by_clip"].values()):
    fails.append("3.3 clip sensitivity: condition or monotonicity fails for some clip")

v3w_ext = load(f"{CONS}/V3W_test/weighted_routing.json")["weighted_cohort"]["contrasts"]
present("4.5 V3W external gap", gap_string(v3w_ext["consistency_minus_random"]))
present("4.5 V3W external vs averaged confidence", gap_string(v3w_ext["consistency_minus_mean5_conf"]))
present("4.5 V3W external voting gain", gap_string(v3w_ext["voting_gain_accuracy"]))
present("Table 10 V3W external", "$+0.041$ [0.024, 0.060] \\\\")
lvo_ext = load(f"{CONS}/V3W_test/leave_variant_out.json")["weighted_cohort"]["contrasts"]
drops = [lvo_ext[f"drop_{k}:consistency_minus_random"]["exact"]["gap_point"] for k in range(5)]
exact("4.5 V3W external leave-one-out min", min(drops), 0.025)
exact("4.5 V3W external leave-one-out max", max(drops), 0.059)
holm_fam = load("results/multiplicity/holm.json")
checks += 1
if holm_fam["family_size_present"] != 19 or holm_fam["n_pending"] != 0:
    fails.append("Holm family is not the declared 19 with none pending")
present("Holm family size in Methods", "rest on 19 contrasts")
present("Table S1 V3W external row", "V3W consistency $-$ random (test) & 72 & $+0.041$", supp, "supplementary.tex")

qn = load("results/quiltnet/summary.json")
for split, acc, auroc, gap in (("val", 52.7, 0.672, 0.148), ("test", 50.4, 0.684, 0.165)):
    r = qn["zeroshot_ensemble"][split]
    near(f"4.3 QuiltNet accuracy {split}", 100 * r["accuracy"], acc, nd=1)
    near(f"4.3 QuiltNet AUROC {split}", r["auroc"], auroc)
    near(f"4.3 QuiltNet gap {split}", r["gap_point"], gap)
    near(f"4.3 QuiltNet gap lo {split}", r["gap_bootstrap"]["ci_2.5"], {"val": 0.141, "test": 0.156}[split])
    near(f"4.3 QuiltNet gap hi {split}", r["gap_bootstrap"]["ci_97.5"], {"val": 0.156, "test": 0.173}[split])
near("4.3 QuiltNet probe accuracy", 100 * qn["probe"]["test"]["accuracy"], 95.4, nd=1)
present("4.3 QuiltNet commit", "8ce7728")

pilot = load("results/zeroshot/medgemma-27b-it/pilot_per_class.json")["versions"]
near("5.1 NORM F1 under V2", pilot["V2"]["per_class"]["NORM"]["f1"], 0.61, nd=2)
near("5.1 NORM F1 under V4B", pilot["V4B"]["per_class"]["NORM"]["f1"], 0.53, nd=2)

# --- text that must be present or gone ----------------------------------------
present("endpoint named", "api.ai.it.ufl.edu")
present("code availability version DOI", "10.5281/zenodo.23199902")
present("code availability concept DOI", "10.5281/zenodo.22245989")
absent("superseded version DOI cited as current", "the version described here is v3.0")
cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
present("CITATION.cff version DOI", "doi: 10.5281/zenodo.23199902", cff, "CITATION.cff")
present("gap convention", "computed at full precision and rounded once")
present("A_AUC estimator", r"\frac{1}{0.99} \int_0^{0.99}")
for gone in (
    "Prompt-driven consistency cannot produce",
    "prompt paraphrases routes",
    "8.28 of the nine",
    "averaged over 30",
    "decreases monotonically in T, so the optimizer",
    "our best signal",
    "on validation the failure lies in verbalization",
    "there the failure lies in verbalization",
    "scored as incorrect rather than excluded",
    "lets us isolate what medical fine-tuning does",
    "the signal we end up recommending",
    "$+0.114$ on both splits",
    "by $+0.105$ to $+0.132$",
    "the loss happens in verbalization",
    "label-token signal is the better one",
    "faithful readout",
    "once every class is scored",
    "where all nine were scored",
    "Medical Tuning and the Accuracy Gap",
    "a perfect selective predictor approaches 1",
    "shaded random band",
    "better than both VLMs on both counts",
    "between VLMs and supervised baselines at the level of accuracy",
    "the degeneracy sits in how the model verbalizes",
    "most fragile entry in this block: it",
    "The failure is specific to a prompt",
):
    absent("stale", gone)
    absent("stale", gone, supp, "supplementary.tex")

for label in ("tab:cluster", "tab:logprob", "tab:sampleflow", "tab:pilot", "tab:censoring", "tab:holm"):
    present(f"{label} in supplement", "label{" + label + "}", supp, "supplementary.tex")
present("SI table numbering", r"\renewcommand{\thetable}{S\arabic{table}}", supp, "supplementary.tex")
present("SI figure numbering", r"\renewcommand{\thefigure}{S\arabic{figure}}", supp, "supplementary.tex")
present("SI declaration", "Supplementary information:")

# The manuscript states the current state of the work. Revision history, reviewer
# attributions and self-justification belong in the response letter.
for phrase in ("submitted version", "earlier version", "previous version", "response to reviewers",
               "reviewer", "requested in review", "we withdraw", "was our error", "we no longer"):
    absent(f"narration in main.tex: {phrase}", phrase)
    absent(f"narration in supplementary.tex: {phrase}", phrase, supp, "supplementary.tex")

if letter:
    present("letter answers every Reviewer 3 item", "### R3.15", letter, "letter")

parser = argparse.ArgumentParser()
parser.add_argument("--final", action="store_true")
args = parser.parse_args()
pending = {name: len(re.findall(r"\[PENDING", s)) for name, s in
           (("main.tex", tex), ("supplementary.tex", supp), ("letter", letter))}
print(f"{checks} checks run; open [PENDING] markers: {pending}")
if args.final and any(pending.values()):
    fails.append(f"[PENDING] markers remain: {pending}")
if fails:
    print(f"{len(fails)} FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
