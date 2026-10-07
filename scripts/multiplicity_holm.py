"""
scripts/multiplicity_holm.py

Holm adjustment across the declared family of headline contrasts.

The paper reports more than twenty intervals across models, splits, signals and
prompt families. The claims it builds on are fixed here as one family, before
the round-3 inference runs have results, and each is tested with a two-sided
normal-approximation p value from its bootstrap: est / SE, where SE is the
bootstrap standard deviation when the artifact stores it and otherwise the 95%
interval width divided by 2 x 1.96. Holm's step-down procedure then controls the
family-wise error rate at 0.05. Family members whose artifacts do not exist yet
(the V3W and pixel-shuffle runs, done on the machine that holds the images) are
listed as pending and enter the family once their outputs are present, so the
family is declared once and never chosen after the fact.

No API calls. Output: results/multiplicity/holm.json.

Usage:
    python scripts/multiplicity_holm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "results"
CONS = RES / "consistency" / "medgemma-27b-it"
OUT = RES / "multiplicity" / "holm.json"
ALPHA = 0.05


def _routing(model: str, split: str) -> dict:
    r = json.loads((RES / "routing" / "routing_auc_ci.json").read_text())["results"]
    row = r[f"{model}|{split}"]
    g = row["gap_bootstrap"]
    return {"point": row["gap_point"], "lo": g["ci_2.5"], "hi": g["ci_97.5"]}


def _logprob(split: str, key: str) -> dict:
    c = json.loads((RES / "logprob_confidence" / "comparison.json").read_text())
    b = c["runs"][f"medgemma-27b-it|{split}"]["bootstrap"][key]
    return {"point": b["gap_point"], "lo": b["ci_2.5"], "hi": b["ci_97.5"]}


def _cons(split_dir: str, file: str, *keys: str) -> dict:
    d = json.loads((CONS / split_dir / file).read_text())
    for k in keys:
        d = d[k]
    return {"point": d["gap_point"], "lo": d["ci_2.5"], "hi": d["ci_97.5"], "se": d.get("se")}


def _family() -> list[tuple[str, str, Path, object]]:
    """(id, description, artifact the value comes from, loader)."""
    wr, fo, lvo = "weighted_routing.json", "fixed_outcome.json", "leave_variant_out.json"
    fam = []
    for split, sd, name in (("val", "V3", "validation"), ("test", "V3_test", "external")):
        fam += [
            (f"medgemma_verbalized_vs_random_{split}",
             f"MedGemma verbalized confidence routing gap vs random, full {name} split",
             RES / "routing" / "routing_auc_ci.json",
             lambda s=split: _routing("medgemma-27b-it", s)),
            (f"gemma3_verbalized_vs_random_{split}",
             f"Gemma-3 verbalized confidence routing gap vs random, full {name} split",
             RES / "routing" / "routing_auc_ci.json",
             lambda s=split: _routing("gemma-3-27b-it", s)),
            (f"consistency_vs_random_{split}",
             f"Consistency minus random, cohort-weighted, {name}",
             CONS / sd / wr,
             lambda d=sd: _cons(d, wr, "weighted_cohort", "contrasts", "consistency_minus_random")),
            (f"consistency_vs_mean5_{split}",
             f"Consistency minus mean-of-5 confidence, cohort-weighted, {name}",
             CONS / sd / wr,
             lambda d=sd: _cons(d, wr, "weighted_cohort", "contrasts",
                                "consistency_minus_mean5_conf")),
            (f"consistency_vs_single_query_fixed_outcome_{split}",
             (f"Consistency minus single-query confidence on the modal-vote outcome, "
              f"cohort-weighted, {name}"),
             CONS / sd / fo,
             lambda d=sd: _cons(d, fo, "weighted_cohort", "contrasts",
                                "modal_outcome:consistency_minus_single_query_conf")),
            (f"v3_edits_only_consistency_vs_random_{split}",
             f"Agreement among variants 0, 1, 3 only, minus random, cohort-weighted, {name}",
             CONS / sd / lvo,
             lambda d=sd: _cons(d, lvo, "weighted_cohort", "contrasts",
                                "v3_edits_only_0_1_3:consistency_minus_random")),
            (f"label_token_vs_random_{split}",
             f"MedGemma label-token confidence routing gap vs random, {name} sample",
             RES / "logprob_confidence" / "comparison.json",
             lambda s=split: _logprob(s, "label_token_minus_random")),
            (f"label_token_vs_verbalized_{split}",
             f"MedGemma label-token minus verbalized confidence, {name} sample",
             RES / "logprob_confidence" / "comparison.json",
             lambda s=split: _logprob(s, "label_token_minus_verbalized")),
        ]
    fam += [
        ("v3w_consistency_vs_random_val",
         "V3W wording-only paraphrase agreement minus random, cohort-weighted, validation",
         CONS / "V3W" / wr,
         lambda: _cons("V3W", wr, "weighted_cohort", "contrasts", "consistency_minus_random")),
        ("shuffle_consistency_vs_random_val",
         "Agreement on pixel-shuffled patches minus random, cohort-weighted, validation",
         CONS / "controls" / "summary.json",
         lambda: _cons("controls", "summary.json", "shuffle", "weighted_cohort", "contrasts",
                       "consistency_minus_random")),
    ]
    return fam


def main() -> int:
    rows, pending = [], []
    for cid, desc, path, load in _family():
        # A shared artifact can exist before the block a contrast needs is written
        # (controls/summary.json gains its shuffle block last), so a missing key is
        # pending too, not an error.
        try:
            v = load() if path.exists() else None
        except KeyError:
            v = None
        if v is None:
            pending.append({"id": cid, "description": desc, "artifact": str(path.relative_to(
                PROJECT_ROOT))})
            continue
        se = v.get("se") or (v["hi"] - v["lo"]) / (2 * 1.959964)
        p = float(2 * norm.sf(abs(v["point"]) / se))
        rows.append({"id": cid, "description": desc, "point": v["point"],
                     "ci": [v["lo"], v["hi"]], "se": round(se, 5), "p": p})

    # Holm: sort ascending, multiply the k-th smallest by (m - k), enforce monotonicity.
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i]["p"])
        running = max(running, adj)
        rows[i]["p_holm"] = running
        rows[i]["survives_holm"] = running < ALPHA

    out = {"alpha": ALPHA, "family_size_present": m, "n_pending": len(pending),
           "method": "Holm step-down on two-sided normal-approximation p values from "
                     "bootstrap SEs; the family is fixed in this script",
           "contrasts": rows, "pending": pending}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['id']:<52} {r['point']:+.4f}  p={r['p']:.2e}  p_holm={r['p_holm']:.2e}  "
              f"{'holds' if r['survives_holm'] else 'FAILS'}")
    for pnd in pending:
        print(f"{pnd['id']:<52} pending: {pnd['artifact']}")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
