"""
scripts/cluster_sensitivity.py

Design-effect sensitivity for every headline interval (revision item C-04).

Reviewer 1 is right that all our intervals treat patches as independent when
validation patches nest within 86 slides and external-test patches within 25
slides drawn from 50 patients. We cannot comply with the requested cluster
bootstrap: MedMNIST v2
publishes no index from PathMNIST array position back to a source slide, and
scripts/verify_partitions.py confirms the npz archives store nothing but images
and labels. Rather than drop the concern, this quantifies it.

Under a simple exchangeable-correlation model the variance of a mean over
clustered data is inflated by the design effect

    Deff = 1 + (m - 1) * rho

for average cluster size m and intra-cluster correlation rho. Every interval
half-width is therefore multiplied by sqrt(Deff). This reports each headline
interval at rho in {0.01, 0.05, 0.10, 0.20}, and, more usefully, the break-even
rho at which each conclusion actually changes: the point where an interval that
excludes zero starts to include it. A reader can then judge each claim against
their own prior about how strongly patches from one slide resemble each other.

The correction cuts both ways and both directions are reported. Wider intervals
help the claim that MedGemma's confidence carries no usable signal, and hurt the
claim that the consistency score carries one.

Caveat worth stating in the paper: this treats the AUC gap as if it were a simple
mean. It is a functional of the whole ranking, so Deff is an approximation, not a
correction. It is offered as a sensitivity analysis, not as clustered inference.

Outputs:
    results/clustering/design_effect_sensitivity.json

Usage:
    python scripts/cluster_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cluster counts from Kather et al. NCT-CRC-HE-100K is assembled from 86 tissue
# slides. CRC-VAL-HE-7K is 7,180 patches from 50 patients, but Kather et al. 2019
# state the patches were cut from 25 H&E slides of DACHS-study tissue, so slides
# and patients are not the same unit here and the two counts differ by a factor
# of two. The design effect uses the slide, because the correlation it models is
# staining, scanner and batch, and those are properties of the slide rather than
# of the patient. Using 50 would halve the assumed cluster size and understate
# the widening. Neither number appears in the submitted Methods, which is itself
# part of what Reviewer 1 asked us to fix.
N_SLIDES_TRAIN_VAL = 86
N_SLIDES_TEST = 25
N_PATIENTS_TEST = 50
N_VAL_PATCHES = 10_004
N_TEST_PATCHES = 7_180
N_SUBSET_PATCHES = 1_800

RHO_GRID = (0.01, 0.05, 0.10, 0.20)


def design_effect(m: float, rho: float) -> float:
    return 1.0 + (m - 1.0) * rho


def break_even_rho(point: float, half_width: float, m: float) -> float | None:
    """
    Smallest rho at which the interval stops excluding zero.

    Returns None when the interval already includes zero, so there is nothing to
    break. The interval is treated as symmetric about the point estimate, which
    is close enough for a sensitivity statement.
    """
    if half_width <= 0 or abs(point) <= half_width:
        return None
    deff_star = (point / half_width) ** 2
    rho_star = (deff_star - 1.0) / (m - 1.0)
    return float(rho_star)


def analyse(name: str, point: float, lo: float, hi: float, m: float, n_clusters: int) -> dict:
    half = (hi - lo) / 2.0
    excludes_zero = lo > 0 or hi < 0
    rows = {}
    for rho in RHO_GRID:
        deff = design_effect(m, rho)
        infl = deff**0.5
        new_lo, new_hi = point - half * infl, point + half * infl
        rows[f"rho_{rho:.2f}"] = {
            "design_effect": round(deff, 3),
            "half_width_multiplier": round(infl, 3),
            "ci_2.5": round(new_lo, 4),
            "ci_97.5": round(new_hi, 4),
            "excludes_zero": bool(new_lo > 0 or new_hi < 0),
            "effective_n": round(m * n_clusters / deff, 1),
        }
    rho_star = break_even_rho(point, half, m)
    return {
        "point_estimate": round(point, 4),
        "unclustered_ci": [round(lo, 4), round(hi, 4)],
        "unclustered_excludes_zero": bool(excludes_zero),
        "mean_cluster_size": round(m, 1),
        "n_clusters": n_clusters,
        "break_even_rho": (round(rho_star, 4) if rho_star is not None else None),
        "break_even_note": (
            "interval already includes zero" if rho_star is None
            else f"conclusion survives intra-cluster correlation up to rho = {rho_star:.3f}"
        ),
        "by_rho": rows,
    }


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"Missing required file: {path}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Unused; logged for consistency.")
    args = parser.parse_args()

    routing = _load(PROJECT_ROOT / "results" / "routing" / "routing_auc_ci.json")["results"]
    cons_val = _load(
        PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it" / "V3" / "routing_signals.json"
    )
    cons_test = _load(
        PROJECT_ROOT / "results" / "consistency" / "medgemma-27b-it" / "V3_test"
        / "routing_signals.json"
    )

    m_val_full = N_VAL_PATCHES / N_SLIDES_TRAIN_VAL
    m_test_full = N_TEST_PATCHES / N_SLIDES_TEST
    m_subset_val = N_SUBSET_PATCHES / N_SLIDES_TRAIN_VAL
    m_subset_test = N_SUBSET_PATCHES / N_SLIDES_TEST

    findings: dict[str, dict] = {}

    # Full-scale routing gaps (Table 4).
    for key, label, m, nc in (
        ("medgemma-27b-it|val", "MedGemma routing gap vs random (val)", m_val_full, N_SLIDES_TRAIN_VAL),
        ("medgemma-27b-it|test", "MedGemma routing gap vs random (test)", m_test_full, N_SLIDES_TEST),
        ("gemma-3-27b-it|val", "Gemma-3 routing gap vs random (val)", m_val_full, N_SLIDES_TRAIN_VAL),
        ("gemma-3-27b-it|test", "Gemma-3 routing gap vs random (test)", m_test_full, N_SLIDES_TEST),
        ("resnet18_64px|test", "CNN-64 routing gap vs random (test)", m_test_full, N_SLIDES_TEST),
    ):
        r = routing[key]
        b = r["gap_bootstrap"]
        findings[label] = analyse(label, r["gap_point"], b["ci_2.5"], b["ci_97.5"], m, nc)

    # Consistency contrasts (Tables 8 and 9), on the 1,800-patch subsets.
    for src, split, m, nc in (
        (cons_val, "val", m_subset_val, N_SLIDES_TRAIN_VAL),
        (cons_test, "test", m_subset_test, N_SLIDES_TEST),
    ):
        for contrast, pretty in (
            ("consistency_minus_mean_textual_conf", "Consistency minus mean-of-5 confidence"),
            ("consistency_minus_single_query_conf", "Consistency minus single-query confidence"),
            ("mean_textual_conf_minus_random", "Mean-of-5 confidence minus random"),
        ):
            c = src["contrasts"][contrast]
            label = f"{pretty} ({split})"
            # gap_point is the difference of the two tabulated AUCs, which is what
            # the manuscript prints; the bootstrap mean is a slightly different
            # quantity and using it here would put a break-even value against a
            # point estimate that appears nowhere in the paper.
            findings[label] = analyse(
                label, c.get("gap_point", c["mean_diff"]), c["ci_2.5"], c["ci_97.5"], m, nc
            )

    out = {
        "seed": args.seed,
        "method": "Deff = 1 + (m - 1) * rho; interval half-widths scaled by sqrt(Deff)",
        "cluster_structure": {
            "val_source": "NCT-CRC-HE-100K",
            "n_slides_train_val": N_SLIDES_TRAIN_VAL,
            "test_source": "CRC-VAL-HE-7K",
            "n_slides_test": N_SLIDES_TEST,
            "n_patients_test": N_PATIENTS_TEST,
            "test_cluster_unit": (
                "slide. Kather et al. 2019 cut the 7,180 patches from 25 H&E slides of "
                "DACHS-study tissue banked at NCT Heidelberg; the Zenodo record gives the "
                "50-patient count. Staining, scanner and batch attach to the slide."
            ),
            "mean_cluster_size_val_full": round(m_val_full, 1),
            "mean_cluster_size_test_full": round(m_test_full, 1),
            "mean_cluster_size_val_subset": round(m_subset_val, 1),
            "mean_cluster_size_test_subset": round(m_subset_test, 1),
        },
        "identifiers_available": False,
        "identifier_evidence": "results/data/partition_identity.json",
        "caveat": (
            "The AUC gap is a functional of the full ranking, not a simple mean, so "
            "the design effect is an approximation offered as a sensitivity analysis "
            "rather than as clustered inference."
        ),
        "rho_grid": list(RHO_GRID),
        "findings": findings,
    }

    out_dir = PROJECT_ROOT / "results" / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_effect_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Design-effect sensitivity ===")
    print(f"clusters: {N_SLIDES_TRAIN_VAL} slides (val), {N_SLIDES_TEST} slides (test; 50 patients)")
    print(f"{'finding':<52}{'point':>9}{'rho*':>9}  survives rho =")
    for label, f in findings.items():
        rs = f["break_even_rho"]
        surviving = [r for r in RHO_GRID if f["by_rho"][f"rho_{r:.2f}"]["excludes_zero"]]
        surv = ", ".join(f"{r:g}" for r in surviving) if surviving else "none"
        print(f"{label:<52}{f['point_estimate']:>+9.4f}"
              f"{(f'{rs:.3f}' if rs is not None else 'n/a'):>9}  {surv}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
