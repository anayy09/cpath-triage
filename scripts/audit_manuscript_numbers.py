"""
scripts/audit_manuscript_numbers.py

Cross-check every AUC, gap and break-even the manuscript prints against the
artifact it comes from, rounded once from full precision.

Why this exists. Three of the four review passes this manuscript has been
through opened on the same defect: a number in the text that does not
reconcile with the table beside it. Editorial care did not catch it; a script
that recomputes every printed value does. Run it before any rebuild of
paper/latex/main.tex.

Usage:
    python scripts/audit_manuscript_numbers.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tex = (ROOT / "paper" / "latex" / "main.tex").read_text(encoding="utf-8")
letter = (ROOT / "paper" / "RESPONSE_LETTER_SR_R1.md").read_text(encoding="utf-8")
supp = (ROOT / "paper" / "latex" / "supplementary.tex").read_text(encoding="utf-8")

fails, checks = [], 0


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def want(label, artifact_value, printed, nd=3):
    global checks
    checks += 1
    got = f"{round(float(artifact_value), nd):+.{nd}f}"
    exp = f"{printed:+.{nd}f}"
    if got != exp:
        fails.append(f"{label}: artifact {got} vs manuscript {exp}")


def present(label, needle, hay=None, where="main.tex"):
    global checks
    checks += 1
    if needle not in (tex if hay is None else hay):
        fails.append(f"{label}: not found in {where} -> {needle!r}")


def absent(label, needle, hay=None, where="main.tex"):
    global checks
    checks += 1
    if needle in (tex if hay is None else hay):
        fails.append(f"{label}: stale text still in {where} -> {needle!r}")


# --- Table 13 gaps and Section 4.6 contrasts -------------------------------
lp = load("results/logprob_confidence/comparison.json")
for (run, contrast), printed in {
    ("medgemma-27b-it|val", "label_token_minus_random"): 0.049,
    ("medgemma-27b-it|val", "verbalized_minus_random"): -0.009,
    ("medgemma-27b-it|test", "label_token_minus_random"): -0.056,
    ("medgemma-27b-it|test", "verbalized_minus_random"): 0.022,
    ("gemma-3-27b-it|val", "label_token_minus_random"): 0.105,
    ("gemma-3-27b-it|val", "verbalized_minus_random"): 0.116,
    ("gemma-3-27b-it|test", "label_token_minus_random"): 0.132,
    ("gemma-3-27b-it|test", "verbalized_minus_random"): 0.115,
    ("medgemma-27b-it|val", "label_token_minus_verbalized"): 0.058,
    ("medgemma-27b-it|test", "label_token_minus_verbalized"): -0.078,
    ("gemma-3-27b-it|val", "label_token_minus_verbalized"): -0.011,
    ("gemma-3-27b-it|test", "label_token_minus_verbalized"): 0.017,
}.items():
    c = lp["runs"][run]["bootstrap"][contrast]
    want(f"T13 {run} {contrast}", c["gap_point"], printed)
    checks += 1
    if not c["point_inside_ci"]:
        fails.append(f"T13 {run} {contrast}: point outside its own CI")

# --- Tables 11 and 12 AUCs --------------------------------------------------
cons = {
    "val": load("results/consistency/medgemma-27b-it/V3/routing_signals.json"),
    "test": load("results/consistency/medgemma-27b-it/V3_test/routing_signals.json"),
    "V5": load("results/consistency/medgemma-27b-it/V5/routing_signals.json"),
}
for split, sig, printed in [
    ("val", "single_query_conf", 0.320),
    ("val", "mean_textual_conf", 0.348),
    ("val", "consistency_score", 0.431),
    ("val", "entropy_over_k", 0.431),
    ("val", "mean_textual_conf_flip", 0.332),
    ("test", "consistency_score", 0.451),
    ("test", "entropy_over_k", 0.451),
    ("test", "mean_textual_conf", 0.372),
    ("test", "mean_textual_conf_flip", 0.344),
    ("V5", "consistency_score", 0.406),
    ("V5", "mean_textual_conf", 0.409),
]:
    want(f"AUC {split} {sig}", cons[split]["signals"][sig]["auc"], printed)
    checks += 1
    if cons[split]["tie_handling"]["tie_break"] != "expected":
        fails.append(f"{split}: tie_break is not 'expected'")

for split, printed in (("val", 0.339), ("test", 0.356), ("V5", 0.354)):
    want(f"random {split}", cons[split]["random_routing_auc"]["modal_vote_outcome"], printed)

# --- Section 4.5 gaps -------------------------------------------------------
for (split, contrast), printed in {
    ("val", "consistency_minus_single_query_conf"): 0.111,
    ("val", "consistency_minus_mean_textual_conf"): 0.083,
    ("val", "mean_textual_conf_minus_random"): 0.010,
    ("val", "mean_textual_conf_flip_minus_random"): -0.007,
    ("val", "consistency_minus_entropy"): -0.000,
    ("test", "consistency_minus_single_query_conf"): 0.096,
    ("test", "consistency_minus_mean_textual_conf"): 0.079,
    ("test", "mean_textual_conf_minus_random"): 0.016,
    ("test", "mean_textual_conf_flip_minus_random"): -0.012,
    ("V5", "mean_textual_conf_minus_random"): 0.056,
    ("V5", "consistency_minus_mean_textual_conf"): -0.004,
}.items():
    c = cons[split]["contrasts"][contrast]
    want(f"gap {split} {contrast}", c["gap_point"], printed)
    checks += 1
    if not c["point_inside_ci"]:
        fails.append(f"gap {split} {contrast}: point outside its own CI")

# --- confusability ----------------------------------------------------------
for split, path, auc_p, gap_p in (
    ("val", "V3", 0.467, 0.036),
    ("test", "V3_test", 0.474, 0.024),
):
    cw = load(f"results/consistency/medgemma-27b-it/{path}/confusability_weighted.json")
    want(f"confus {split} auc", cw["signals"]["confusability_weighted"]["auc"], auc_p)
    want(f"confus {split} gap", cw["contrasts"]["weighted_minus_consistency"]["gap_point"], gap_p)
    checks += 1
    if not cw["uniform_control_reproduces_consistency"]["exact"]:
        fails.append(f"confus {split}: uniform control no longer exact")

# --- Table 9 ----------------------------------------------------------------
for key, printed in (("resnet18_64px", 0.051), ("resnet18_224px", 0.093)):
    t = load(f"results/calibration/{key}/calibration_params.json")["transfer"]
    want(f"T9 {key} change", t["change"], printed)

# --- Table 7 ----------------------------------------------------------------
de = load("results/clustering/design_effect_sensitivity.json")["findings"]
for name, (pt, be) in {
    "MedGemma routing gap vs random (val)": (-0.007, 0.004),
    "MedGemma routing gap vs random (test)": (0.033, 0.068),
    "Gemma-3 routing gap vs random (test)": (0.114, 0.850),
    "CNN-64 routing gap vs random (test)": (0.067, 0.444),
    "Consistency minus mean-of-5 confidence (val)": (0.083, 0.383),
    "Consistency minus mean-of-5 confidence (test)": (0.079, 0.096),
}.items():
    want(f"T7 point {name}", de[name]["point_estimate"], pt)
    want(f"T7 rho {name}", de[name]["break_even_rho"], be)

# --- text that must be present / gone ---------------------------------------
present("endpoint named", "api.ai.it.ufl.edu")
present("request window", "22:07 and 23:53 UTC")
present("file write dates", "25 June to 3 July 2026")
present("gap convention", "computed at full precision and rounded once")
present("tie estimator note", "sampled mean lands within 0.0013")
present("T9 caption", "subtracting the printed cells gives 0.094")
present("within-class ref", r"excluding it (Section~\ref{subsec:routing-results})")
for gone in (
    "the inference date range for every run",
    "difference of the two point estimates in the same table",
    r"(Section~\ref{subsec:calibration-results}). The honest summary",
    "disjoint by 0.021",
    "$+0.112$",
    "$+0.097$",
    "0.475 \\\\",
):
    absent("stale", gone)

# --- the three Supplementary tables and the pointers to them ----------------
# Each was requested in review, so the main text has to say where it went and
# the supplement has to actually contain it.
for label in ("tab:cluster", "tab:logprob", "tab:seeds"):
    absent("moved table still in main", r"\ref{" + label + "}")
    present(f"{label} in supplement", "label{" + label + "}", supp, "supplementary.tex")
for n in ("Table~S1", "Table~S2", "Table~S3"):
    present(f"{n} pointer", n)
present("SI declaration", "Supplementary information:")
present("SI numbering", r"\renewcommand{\thetable}{S\arabic{table}}", supp, "supplementary.tex")

present("letter endpoint", "api.ai.it.ufl.edu", letter, "letter")
present("letter audit section", "Further corrections from our own audit", letter, "letter")

# The manuscript states the current state of the work. Revision history, reviewer
# attributions and self-justification belong in the response letter, not in the paper.
for phrase in (
    "submitted version",
    "earlier version",
    "previous version",
    "response to reviewers",
    "reviewer",
    "requested in review",
    "we withdraw",
    "was our error",
    "we no longer",
    "That objection",
):
    absent(f"narration in main.tex: {phrase}", phrase)
    absent(f"narration in supplementary.tex: {phrase}", phrase, supp, "supplementary.tex")
for gone in (
    "The other two we found while auditing",
    "Both columns order the models the same way",
    "which is 0.409 minus 0.354 to three decimals",
    "0.4307",
    "0.3477",
    "+0.0830",
    "+0.1121",
    "+0.0973",
):
    absent("stale letter", gone, letter, "letter")

# --- build logs -------------------------------------------------------------
# latexdiff adds text, so the marked-up build is allowed the overfull box the
# clean build is not.
for log, allow_overfull in (("paper/latex/main.log", False), ("paper/diff/diff.log", True)):
    checks += 1
    text = (ROOT / log).read_text(encoding="utf-8", errors="ignore")
    if re.search(r"Undefined|undefined (references|citations)", text):
        fails.append(f"{log}: undefined reference")
    if not allow_overfull and "Overfull" in text:
        fails.append(f"{log}: overfull box")

print(f"{checks} checks run")
if fails:
    print(f"{len(fails)} FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
