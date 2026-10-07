# cpath-triage

Code and prediction artifacts for a diagnostic study of **verbalized confidence** in an open medical vision-language model, evaluated on colorectal H&E histopathology patches.

The question is narrow: when MedGemma-27b-it writes an integer confidence next to a tissue label, does that number carry enough information about correctness to drive selective prediction? For the prompt we selected, it does not. Its base model Gemma-3-27b-it, given the identical prompt, produces a confidence that does route above random. Agreement across five prompt variants routes where MedGemma's stated number does not, and so does agreement across five paraphrases that change only the wording, though its external margins over confidence do not survive a plausible within-slide correlation.

This is not a proposal for a deployable triage system. Zero-shot accuracy here is 40.2% (MedGemma) and 35.6% (Gemma-3) on nine classes, so neither model is usable as a standalone classifier, and the repository contains no system anyone should deploy. What it contains is the measurement of a confidence signal, the routing analysis that follows from it, and the artifacts needed to check both.

Manuscript: *Diagnosing Verbalized Confidence in a Medical Vision-Language Model for Selective Prediction on Colorectal Histopathology Patches*. Under revision at *Scientific Reports*.

## What is here

- `src/models/client.py` : the only place an API client is constructed. Reads `API_KEY`, `BASE_URL`, `MEDGEMMA_MODEL` from `.env`.
- `src/models/prompts.py` : the prompt registry, including the V3 prompt behind every full-scale run and the wording-only V3W family.
- `src/models/k5.py` : the call log, resume logic and aggregation shared by the K=5 runs.
- `src/triage/router.py` : selective-accuracy curves and the three tie policies. Verbalized confidence is coarse enough that tie handling changes the answer, so it is explicit rather than implicit.
- `src/triage/weighted.py` : inverse-probability-weighted versions of the routing metrics, used to take the K=5 subsets back to their whole split.
- `src/eval/calibration.py` : temperature scaling, equal-width and equal-mass ECE.
- `scripts/` : one script per analysis, each writing to a deterministic path under `results/` and printing it.
- `tests/` : the router permutation-invariance suite, the weighted-metric identities and the prompt-family checks.

## Reproducing the manuscript

Every table and figure below regenerates from `results/`, which holds the saved predictions and call logs. No API calls and no GPU are needed for the analysis scripts; the inference and training scripts that produced `results/` in the first place are marked.

### Main tables

| Table | Content | Script | Artifact |
|---|---|---|---|
| 1 | Per-class counts and shares, both splits | `scripts/cross_center_scalings.py` | `results/stage6/cross_center_scalings.json` |
| 2 | Prompt registry with pilot macro F1 | `scripts/run_zeroshot.py` (API) | `results/zeroshot/medgemma-27b-it/{V2,V3,V4,V4B,V5}/metrics.json` |
| 3 | Validation classification | `scripts/classification_stats.py` | `results/classification_stats.json` |
| 4 | Per-class F1, both VLMs, both splits | `scripts/classification_stats.py` | `results/classification_stats.json` |
| 5 | Calibration: ECE raw and adaptive, Brier, T | `scripts/calibrate.py --full-scale`, `scripts/calibration_extras.py` | `results/calibration/*/calibration_params.json`, `results/calibration/calibration_extras.json` |
| 6 | Routing by confidence, with class-prior baseline | `scripts/routing_ci.py` | `results/routing/routing_auc_ci.json` |
| 7 | Cross-cohort accuracy and F1, VLMs | `scripts/cross_center_scalings.py`, `scripts/classification_stats.py` | `results/stage6/cross_center_scalings.json` |
| 8 | K=5 routing signals, cohort-weighted and subset | `scripts/consistency_population.py --split {val,test}` | `results/consistency/medgemma-27b-it/{V3,V3_test}/{weighted_routing,confusability_valweights}.json` |
| 9 | K=5 paired contrasts, with the Holm outcome | `scripts/consistency_population.py --split {val,test}` | `.../{weighted_routing,fixed_outcome,confusability_valweights}.json` |
| 10 | Agreement with each variant left out; V3W | `scripts/consistency_population.py` (V3W: `--version V3W --selection-from V3 --split val`) | `.../{V3,V3_test,V3W}/leave_variant_out.json`, `.../V3W/weighted_routing.json` |

The inclusion weights behind Tables 8 to 10 come from `scripts/consistency_selection_weights.py`, which reconstructs the stratified selection and checks that it reproduces the stored patch indices. Accuracy by agreement level and the V5 family come from `scripts/consistency_secondary_weighted.py`.

### Supplementary tables and figure

| Item | Content | Script | Artifact |
|---|---|---|---|
| Table S1 | Design-effect sensitivity and break-even rho | `scripts/cluster_sensitivity.py` | `results/clustering/design_effect_sensitivity.json` |
| Table S2 | Label-token against verbalized confidence | `scripts/logprob_comparison.py` | `results/logprob_confidence/comparison.json` |
| Tables S3, S4 | Pilot sample flow; validation headlines without the pilot | `scripts/sample_flow.py`, `scripts/consistency_population.py` | `results/data/{sample_flow,pilot_excluded_headlines}.json`, `.../V3/pilot_excluded.json` |
| Table S5 | Label-token comparisons by number of classes returned | `scripts/logprob_censoring_sensitivity.py` | `results/logprob_confidence/censoring_sensitivity.json` |
| Table S6 | Holm adjustment over the declared family | `scripts/multiplicity_holm.py` | `results/multiplicity/holm.json` |
| Table S7 | PLIP zero-shot and linear-probe baseline | `scripts/run_plip_baseline.py` (local GPU, about 25 min on a GTX 1650; `transformers<4.47`) | `results/plip/summary.json`, `results/plip/predictions_*.parquet` |
| Figure S1 | Binary NLL against temperature | `scripts/temperature_nll_curve.py` | `results/calibration/nll_curve.{json,pdf}` |

The image controls reported in Section 4.5 are summarized by `scripts/analyze_image_controls.py` into `results/consistency/medgemma-27b-it/controls/summary.json`. Supporting analyses that appear only in the prose: `scripts/confidence_information.py` (confidence-correctness AUROC, mutual information with a permutation null, confidence by predicted class), `scripts/verify_calibration_claims.py` (the T = 200 bound identities) and `scripts/parse_failure_check.py` (Gemma-3's eight off-list answers and how each artifact treats them). `scripts/verify_partitions.py` establishes that the 64 px and 224 px releases index the same patches and that no slide or patient identifier exists in the distributed archives, which is why Table S1 is a sensitivity analysis rather than a cluster bootstrap.

### Figures

| Figure | Content | Script | Artifact |
|---|---|---|---|
| 1 | Selective-prediction setting | `scripts/figures/extract_fig1_patch.py` (needs PathMNIST), then `scripts/figures/make_fig1.py --final` | `paper/latex/figures/fig1_pipeline.{pdf,png}`, patch and source record in `scripts/figures/raw/` |
| 2 | MedGemma confusion matrix | `scripts/replot_saved_figures.py` (drawn first by `run_zeroshot.py`) | `results/zeroshot/medgemma-27b-it/V3_full/confusion_matrix.png` |
| 3 | Gemma-3 confusion matrix | `scripts/replot_saved_figures.py` (drawn first by `run_zeroshot.py`) | `results/zeroshot/gemma-3-27b-it/V3_full/confusion_matrix.png` |
| 4 | MedGemma reliability diagrams | `scripts/calibrate.py --full-scale` | `results/calibration/medgemma-27b-it/V3_full/reliability_combined.png` |
| 5 | Gemma-3 reliability diagrams | `scripts/calibrate.py --full-scale --vlm-model gemma-3-27b-it` | `results/calibration/gemma-3-27b-it/V3_full/reliability_combined.png` |
| 6 | Selective-accuracy routing curves | `scripts/plot_paper_figures.py` | `results/routing/risk_coverage_fullscale_comparison.png` |
| 7 | Consistency routing curves, 1,800-patch subset | `scripts/consistency_routing_table.py` | `results/consistency/medgemma-27b-it/V3/routing_curves_tie_expectation.png` |

Figure 1 is drawn in code from a real validation patch (PathMNIST index 6404, CC BY 4.0); no generative tool is involved. Every other manuscript figure file is a copy of its artifact, and none needs a billed call or a GPU to redraw.

### Order of execution

The analysis scripts read saved predictions. `cluster_sensitivity.py` and `multiplicity_holm.py` consume the outputs of the others, so they go last.

```bash
python scripts/verify_partitions.py
python scripts/routing_ci.py
python scripts/logprob_comparison.py
python scripts/logprob_censoring_sensitivity.py
python scripts/confidence_information.py
python scripts/verify_calibration_claims.py
python scripts/temperature_nll_curve.py
python scripts/parse_failure_check.py
python scripts/cross_center_scalings.py
python scripts/classification_stats.py
python scripts/calibration_extras.py
python scripts/calibrate.py --full-scale
python scripts/calibrate.py --full-scale --vlm-model gemma-3-27b-it
python scripts/sample_flow.py
for s in val test; do
  python scripts/consistency_selection_weights.py --split $s
  python scripts/consistency_population.py --split $s
  python scripts/consistency_routing_table.py --split $s
done
python scripts/consistency_population.py --version V3W --selection-from V3 --split val
python scripts/consistency_secondary_weighted.py
python scripts/analyze_image_controls.py
python scripts/cluster_sensitivity.py
python scripts/multiplicity_holm.py
python scripts/plot_paper_figures.py
python scripts/replot_saved_figures.py
```

## Quick start

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in API_KEY and BASE_URL
python -m src.models.client --selftest
bash scripts/check.sh
```

Only the inference scripts need `.env`. The analysis scripts above run without it.

PathMNIST is downloaded by `scripts/download_data.py` into `data/raw/`, or pointed at an existing copy with `PATHMNIST_DATA_ROOT`. The 224 px archive is 12.6 GB, so only the label arrays are materialized where images are not needed.

`scripts/check.sh` runs ruff, the regression tests, and a scan for forbidden phrases, em dashes and placeholder numbers. It should be run before every commit.

## Numpy compatibility

`np.trapz` is deprecated on numpy 2.0 through 2.2 and removed by 2.5. Both trapezoid calls resolve through one shim in `src/triage/router.py`, verified on 1.26.4 and 2.5.2.

## Scope and honesty

The public MedGemma is a patch-level model, not a whole-slide reader, and this project works at the patch level on purpose. Whole-slide inference, TCGA, CAMELYON, and fine-tuning are out of scope.

Four limits are worth knowing before reading any number here.

**The confidence failure is prompt-specific.** V3 was selected on macro F1, with no reference to confidence quality. Under a second variant family built from V5, the stated confidence routes above random. What is documented is a failure of one model and prompt together.

**The K=5 results come from weighted subsets.** Each split's 1,800 patches were drawn to balance correct and incorrect predictions within each class. The tables lead with estimates weighted back to the whole split from the exact inclusion probabilities, and give the unweighted subset values beside them.

**Intervals are unclustered.** Validation patches nest within 86 slides and external test patches within 25 slides from 50 patients, and MedMNIST v2 publishes no mapping from a PathMNIST array index back to its source slide. Table S1 reports how far each conclusion survives an assumed intra-cluster correlation instead. MedGemma's validation routing gap breaks at rho = 0.003, and the external margins of agreement over the two confidence signals break at 0.07 and 0.19, so those external margins are claimed in direction only.

**The CNN is a reference on the external cohort only.** PathMNIST splits at the patch level, so the CNN's validation figures are leakage-inflated and are not interpreted. Its external accuracy, calibration and routing are clean measurements.

## Citation

See `CITATION.cff`.
