"""
scripts/supplement_tables.py

LaTeX bodies for the supplementary tables that are generated rather than typed
generated from the result files so no value is transcribed by hand. No API
calls. The output is pasted into paper/latex/supplementary.tex; rerunning after
a result changes shows which cells to update.

Inputs:
    results/calibration/calibration_alternatives.json      (calibrate_alternatives)
    results/consistency/medgemma-27b-it/variant_confidence_weighted.json
    results/consistency/medgemma-27b-it/k_ablation.json
    results/routing/budget_cost_table.json
    results/zeroshot/medgemma-27b-it/pilot_per_class.json

Output: results/supplement/generated_tables.tex

Usage:
    python scripts/supplement_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "results"
OUT = RES / "supplement" / "generated_tables.tex"


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def sgn(x: float, nd: int = 3) -> str:
    return f"$+{x:.{nd}f}$" if x >= 0 else f"$-{abs(x):.{nd}f}$"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if x >= 0 else f"$-{abs(x):.{nd}f}$"


def ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{num(lo, nd)}, {num(hi, nd)}]"


def calibrators() -> str:
    d = load("calibration/calibration_alternatives.json")["models"]
    names = {"raw": "Raw", "temperature": "Temperature", "platt": "Platt (slope, intercept)",
             "isotonic": "Isotonic", "constant": "Constant (calibration accuracy)"}
    rows = []
    for model, label in (("medgemma-27b-it", "MedGemma"), ("gemma-3-27b-it", "Gemma-3")):
        for i, (k, nm) in enumerate(names.items()):
            e, x = d[model]["eval"][k], d[model]["external"][k]
            rows.append(f"{label if i == 0 else ''} & {nm} & {e['ece']:.3f} & {e['brier']:.3f} & "
                        f"{e['nll']:.3f} & {e['auroc']:.3f} & {e['brier_decomposition']['resolution']:.4f} & "
                        f"{x['ece']:.3f} & {x['brier']:.3f} & {x['nll']:.3f} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


def variants() -> str:
    d = load("consistency/medgemma-27b-it/variant_confidence_weighted.json")["runs"]
    labels = {"V3": "V3 family, val", "V3_test": "V3 family, external",
              "V3W": "V3W family, val", "V3W_test": "V3W family, external", "V5": "V5 family, val"}
    rows = []
    for run, label in labels.items():
        if run not in d:
            continue
        for k in range(5):
            v = d[run]["variants"][f"variant_{k}"]
            g = v["cohort_gap_vs_random"]
            a = v["cohort_auroc_ci"]
            rows.append(f"{label if k == 0 else ''} & {k} & {v['cohort_accuracy']:.3f} & "
                        f"{v['cohort_auroc']:.3f} {ci(*a)} & {sgn(g['gap_point'])} {ci(g['ci_2.5'], g['ci_97.5'])} & "
                        f"{v['subset_auroc']:.3f} \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


def k_ablation() -> str:
    d = load("consistency/medgemma-27b-it/k_ablation.json")["splits"]
    fam_label = {"V3": "V3 variants", "V3W": "V3W paraphrases", "pooled": "Pooled (nine)"}
    rows = []
    for split, sl in (("val", "val"), ("test", "external")):
        for fam in ("V3", "V3W", "pooled"):
            if fam not in d.get(split, {}):
                continue
            for i, (k, e) in enumerate(d[split][fam]["by_k"].items()):
                wo = e.get("mean_gap_without_variant_2")
                rows.append(f"{(fam_label[fam] + ', ' + sl) if i == 0 else ''} & {k} & {e['n_subsets']} & "
                            f"{sgn(e['mean_gap'])} {ci(*e['mean_gap_ci'])} & "
                            f"{num(e['min_gap'])} to {num(e['max_gap'])} & "
                            f"{sgn(wo) if wo is not None else '--'} & {e['mean_modal_accuracy']:.3f} \\\\")
            rows.append("\\midrule")
    return "\n".join(rows[:-1])


def budgets() -> str:
    d = load("routing/budget_cost_table.json")
    order = [
        ("medgemma-27b-it verbalized", "MedGemma verbalized", "full"),
        ("gemma-3-27b-it verbalized", "Gemma-3 verbalized", "full"),
        ("PLIP zero-shot", "PLIP zero-shot", "full"),
        ("ResNet-18 64 px", "ResNet-18 64 px", "full"),
        ("MedGemma V3 single-query confidence", "MedGemma single-query confidence", "K=5"),
        ("MedGemma V3 mean-of-5 confidence", "MedGemma mean-of-5 confidence", "K=5"),
        ("MedGemma V3 agreement", "MedGemma agreement, V3 variants", "K=5"),
        ("MedGemma V3W agreement", "MedGemma agreement, V3W", "K=5"),
    ]
    rows = []
    for key, label, sample in order:
        for split, sl in (("val", "val"), ("test", "ext")):
            r = d["rows"].get(f"{key}|{split}")
            if r is None:
                continue
            a = r["auto_confirm_accuracy"]
            rows.append(f"{label} & {sl} & {sample} & {r['queries_per_patch']} & {r['base_accuracy']:.3f} & "
                        f"{a['0.10']:.3f} & {a['0.20']:.3f} & {a['0.30']:.3f} \\\\")
    return "\n".join(rows)


def pilot() -> str:
    d = load("zeroshot/medgemma-27b-it/pilot_per_class.json")["versions"]
    classes = list(d["V3"]["per_class"])
    rows = []
    for v, e in d.items():
        rows.append(f"{v} & {e['macro_f1']:.3f} & "
                    + " & ".join(f"{e['per_class'][c]['f1']:.2f}" for c in classes)
                    + f" & {e['per_class']['MUS']['prediction_share']:.2f} \\\\")
    return "% columns: " + " ".join(classes) + "\n" + "\n".join(rows)


HOLM_LABELS = {
    "medgemma_verbalized_vs_random": "MedGemma verbalized $-$ random",
    "gemma3_verbalized_vs_random": "Gemma-3 verbalized $-$ random",
    "consistency_vs_random": "Consistency $-$ random",
    "consistency_vs_mean5": "Consistency $-$ mean-of-5 conf.",
    "consistency_vs_single_query_fixed_outcome": "Consistency $-$ single query, modal outcome",
    "v3_edits_only_consistency_vs_random": "V3 edits only (0, 1, 3) $-$ random",
    "label_token_vs_random": "MedGemma label-token $-$ random",
    "label_token_vs_verbalized": "MedGemma label-token $-$ verbalized",
    "v3w_consistency_vs_random": "V3W wording-only agreement $-$ random",
    "shuffle_consistency_vs_random": "Pixel-shuffled agreement $-$ random",
}


def pval(p: float) -> str:
    if p == 0.0 or p < 1e-300:
        return "$<10^{-300}$"
    if p >= 0.01:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def holm() -> str:
    d = load("multiplicity/holm.json")
    rows = []
    for r in d["contrasts"]:
        base, split = r["id"].rsplit("_", 1)
        rows.append(f"{HOLM_LABELS[base]} & {split} & {sgn(r['point'], 4)} & {pval(r['p'])} & "
                    f"{pval(r['p_holm'])} & {'yes' if r['survives_holm'] else 'no'} \\\\")
    return "\n".join(rows)


def main() -> int:
    parts = {"calibrators": calibrators(), "variants": variants(), "k_ablation": k_ablation(),
             "budgets": budgets(), "pilot": pilot(), "holm": holm()}
    text = "\n\n".join(f"%% ---- {k} ----\n{v}" for k, v in parts.items()) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
