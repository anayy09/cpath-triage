# cpath-triage

Uncertainty-aware second-read triage for colorectal H&E histopathology patches, built on an open medical vision-language model.

A patch classifier `medgemma-27b-it` labels colorectal tissue patches and emits a confidence. We calibrate that confidence, then route the least certain patches to a simulated specialist queue while auto-confirming the rest. The research question is whether this routing-and-calibration layer holds up across two clinical centers (NCT-CRC-HE-100K to CRC-VAL-HE-7K), and whether a medical-tuned VLM degrades less under that center shift than a general VLM or a small supervised baseline.

The contribution is the triage policy and its cross-center calibration analysis, not the model.

## Quick start

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in API_KEY and BASE_URL
python -m src.models.client --selftest
python scripts/download_data.py
bash scripts/check.sh
```

## Scope and honesty

The public MedGemma is a patch-level model, not a whole-slide reader. This project works at the patch level on purpose. Whole-slide inference, TCGA, CAMELYON, and fine-tuning are out of scope for this phase because of the model's design.
