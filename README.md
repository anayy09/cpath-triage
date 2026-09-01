# cpath-triage

Code and prediction artifacts for a diagnostic study of **verbalized confidence** in an open medical vision-language model, evaluated on colorectal H&E histopathology patches.

The question is narrow: when MedGemma-27b-it writes an integer confidence next to a tissue label, does that number carry enough information about correctness to drive selective prediction? For the prompt we selected, it does not. Its base model Gemma-3-27b-it, given the identical prompt, produces a confidence that does route above random. Where the stated number fails, agreement across five prompt paraphrases works.

This is not a proposal for a deployable triage system. Zero-shot accuracy here is 40.2% (MedGemma) and 35.6% (Gemma-3) on nine classes, so neither model is usable as a standalone classifier, and the repository contains no system anyone should deploy. What it contains is the measurement of a confidence signal, the routing analysis that follows from it, and the artifacts needed to check both.

Manuscript: *Diagnosing Verbalized Confidence in a Medical Vision-Language Model for Selective Prediction on Colorectal Histopathology Patches: Calibration, Routing, and Robustness to Cohort Shift*. Under revision at *Scientific Reports*.

## What is here

- `src/models/client.py` : the only place an API client is constructed. Reads `API_KEY`, `BASE_URL`, `MEDGEMMA_MODEL` from `.env`.
- `src/models/prompts.py` : the six-version prompt registry. V3 is the prompt behind every full-scale run.
- `src/triage/router.py` : selective-accuracy curves and the three tie policies. Verbalized confidence is coarse enough that tie handling changes the answer, so it is explicit rather than implicit.
- `src/eval/calibration.py` : temperature scaling, equal-width and equal-mass ECE.
- `scripts/` : one script per analysis, each writing to a deterministic path under `results/` and printing it.
- `tests/` : the router permutation-invariance regression suite.

## Reproducing the manuscript

Every table and figure below regenerates from `results/`, which holds the saved predictions. No API calls and no GPU are needed for the analysis scripts; the inference and training scripts that produced `results/` in the first place are marked.

### Tables

| Table | Content | Script | Artifact |
|---|---|---|---|
| 1 | Per-class counts and shares, both splits | `scripts/cross_center_scalings.py` | `results/stage6/cross_center_scalings.json` |
| 2 | Prompt registry with pilot macro F1 | `scripts/run_zeroshot.py` (API) | `results/zeroshot/medgemma-27b-it/{V2,V3,V4,V4B,V5}/metrics.json` |
| 3 | Validation classification, four models | `scripts/classification_stats.py` | `results/classification_stats.json` |
| 4 | Per-class F1, both VLMs, both splits | `scripts/classification_stats.py` | `results/classification_stats.json` |
| 5 | Calibration: ECE raw and adaptive, Brier, T | `scripts/calibrate.py --full-scale`, `scripts/calibration_extras.py` | `results/calibration/*/calibration_params.json`, `results/calibration/calibration_extras.json` |
| 6 | Routing by confidence, with class-prior baseline | `scripts/routing_ci.py` | `results/routing/routing_auc_ci.json` |
| 7 | Design-effect sensitivity and break-even rho | `scripts/cluster_sensitivity.py` | `results/clustering/design_effect_sensitivity.json` |
| 8 | Cross-cohort accuracy and F1, three rescalings | `scripts/cross_center_scalings.py`, `scripts/classification_stats.py` | `results/stage6/cross_center_scalings.json` |
| 9 | Calibration transfer, val-fitted T on test | `scripts/calibrate.py --full-scale` | `results/calibration/*/calibration_params.json`, key `transfer` |
| 10 | K=5 voting against a single query (McNemar) | `scripts/recompute_table7.py` | `results/consistency/*/matched_baseline_v2.json` |
| 11 | Selective-accuracy AUC by routing signal | `scripts/consistency_routing_table.py`, `scripts/confusability_weighted.py` | `results/consistency/*/{routing_signals,confusability_weighted}.json` |
| 12 | Consistency routing, validation against test | `scripts/consistency_routing_table.py --split {val,test}` | `results/consistency/*/routing_signals.json` |

Two supporting analyses appear only in the prose: `scripts/confidence_information.py` (confidence-correctness AUROC, mutual information with a permutation null, confidence by predicted class) and `scripts/verify_calibration_claims.py` (the T = 200 bound identities). `scripts/verify_partitions.py` establishes that the 64 px and 224 px releases index the same patches and that no slide or patient identifier exists in the distributed archives, which is what makes Table 7 a sensitivity analysis rather than a cluster bootstrap.

### Figures

| Figure | Content | Script | Artifact |
|---|---|---|---|
| 1 | Selective-prediction setting | drawn by hand, no data | `paper/latex/figures/fig1_pipeline.png` |
| 2 | ResNet-18 224 px training curve | `scripts/train_cnn_224px.py` (GPU) | `results/cnn/resnet18_224px/training_curve.png` |
| 3 | MedGemma confusion matrix | `scripts/run_zeroshot.py` (API) | `results/zeroshot/medgemma-27b-it/V3_full/confusion_matrix.png` |
| 4 | Gemma-3 confusion matrix | `scripts/run_zeroshot.py` (API) | `results/zeroshot/gemma-3-27b-it/V3_full/confusion_matrix.png` |
| 5 | MedGemma reliability diagrams | `scripts/calibrate.py` | `results/calibration/medgemma-27b-it/V3_full/reliability_combined.png` |
| 6 | Gemma-3 reliability diagrams | `scripts/calibrate.py` | `results/calibration/gemma-3-27b-it/V3_full/reliability_combined.png` |
| 7 | Selective-accuracy routing curves | `scripts/plot_paper_figures.py` | `results/routing/risk_coverage_fullscale_comparison.png` |
| 8 | Validation against test accuracy bars | `scripts/plot_paper_figures.py` | `results/stage6/cross_center_accuracy_bars.png` |
| 9 | Consistency routing curves, 1,800 patches | `scripts/consistency_routing_table.py` | `results/consistency/medgemma-27b-it/V3/routing_curves_tie_expectation.png` |

Each manuscript figure file is a byte-identical copy of its artifact.

### Order of execution

The analysis scripts read saved predictions and can be run in any order except that `cluster_sensitivity.py` consumes the output of `routing_ci.py` and both `consistency_routing_table.py` runs, so it goes last.

```bash
python scripts/verify_partitions.py
python scripts/routing_ci.py
python scripts/confidence_information.py
python scripts/verify_calibration_claims.py
python scripts/cross_center_scalings.py
python scripts/classification_stats.py
python scripts/calibration_extras.py
for s in val test; do
  python scripts/consistency_routing_table.py --split $s
  python scripts/confusability_weighted.py --split $s
  python scripts/per_variant_confidence.py --split $s
  python scripts/recompute_table7.py --split $s
  python scripts/entropy_baseline.py --split $s
done
python scripts/cluster_sensitivity.py
python scripts/plot_paper_figures.py
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

`scripts/check.sh` runs ruff, the router regression tests, and a scan for forbidden phrases, em dashes and placeholder numbers. It should be run before every commit.

## Numpy compatibility

`np.trapz` is deprecated on numpy 2.0 through 2.2 and removed by 2.5. An earlier version of this repository called `np.trapz` in one file and `np.trapezoid` in another, so no single numpy version ran both. Both now resolve through one shim in `src/triage/router.py`, verified on 1.26.4 and 2.5.2.

## Scope and honesty

The public MedGemma is a patch-level model, not a whole-slide reader, and this project works at the patch level on purpose. Whole-slide inference, TCGA, CAMELYON, and fine-tuning are out of scope.

Three limits are worth knowing before reading any number here.

**The confidence failure is prompt-specific.** V3 was selected on macro F1, with no reference to confidence quality. Under a second variant family built from V5, the stated confidence routes above random. What is documented is a failure of one model and prompt together.

**Intervals are unclustered.** Validation patches nest within 86 slides and external test patches within 50 patients, and MedMNIST v2 publishes no mapping from a PathMNIST array index back to its source slide. Table 7 reports how far each conclusion survives an assumed intra-cluster correlation instead. MedGemma's validation routing gap does not survive at rho = 0.004; the consistency result survives across the plausible range.

**The CNN baselines carry slide-level leakage.** PathMNIST splits at the patch level, so the CNN validation accuracies are optimistic, their cross-cohort drops are inflated, and the inflation flatters the VLM comparison. The external test accuracies are unaffected.

## Citation

See `CITATION.cff`.
