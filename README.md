# cpath-triage

Uncertainty-aware second-read triage for colorectal H&E histopathology patches, built on an open medical vision-language model.

A patch classifier (`medgemma-27b-it`, accessed through a remote OpenAI-compatible API) labels colorectal tissue patches and emits a confidence. We calibrate that confidence, then route the least certain patches to a simulated specialist queue while auto-confirming the rest. The research question is whether this routing-and-calibration layer holds up across two clinical centers (NCT-CRC-HE-100K to CRC-VAL-HE-7K), and whether a medical-tuned VLM degrades less under that center shift than a general VLM or a small supervised baseline.

The contribution is the triage policy and its cross-center calibration analysis, not the model. The system runs on modest hardware: a single 4 GB GPU for the local baseline, under 25 GB of storage, and a remote endpoint for the large model.

## Quick start

See `STARTER.md` for full setup. In short:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in API_KEY and BASE_URL
python -m src.models.client --selftest
python scripts/download_data.py
bash scripts/check.sh
```

## Documents

- `CLAUDE.md` - rules the AI agent reads every session.
- `STARTER.md` - human setup guide.
- `docs/PLAN.md` - the staged roadmap, Stage 0 to Stage 8.
- `docs/AGENT.md` - agent operating manual and integrity rules.
- `docs/PROGRESS.md` - running session log and current numbers.

## Scope and honesty

The public MedGemma is a patch-level model, not a whole-slide reader. This project works at the patch level on purpose. Whole-slide inference, TCGA, CAMELYON, and fine-tuning are out of scope for this phase because of the model's design, the storage budget, and the local GPU. They are recorded as future work in `docs/PLAN.md`.

No number in this repository or the eventual paper is a placeholder. If an experiment has not run, its result is blank or marked TODO. See the integrity rules in `docs/AGENT.md`.
