"""
scripts/run_plip_baseline.py

A pathology image encoder baseline for the confidence study: PLIP (Huang et al.,
the vinid/plip checkpoint) zero-shot, and a linear probe on its frozen image
embeddings. Runs locally on the data machine's GTX 1650; no API calls.

Why this exists: the VLMs classify by generating text, so a poor zero-shot
result could mean the model cannot see the tissue or that it sees it and fails
to say so. A contrastive pathology encoder answers part of that. Zero-shot PLIP
classifies by image-text similarity with no generation step, and a linear probe
on the same frozen embeddings measures how much class information the visual
representation holds at all. Both give a continuous confidence, so the same
calibration and routing metrics apply as for the VLMs.

The checkpoint is pinned: the Hugging Face commit hash is resolved before
loading and written to run_meta.json, and --revision accepts a hash so a rerun
loads exactly the same weights. That is the contrast with the remote VLMs,
whose served version cannot be identified.

Design decisions, fixed before any run:
  - Class names are PathMNIST's LABEL_NAMES, the strings the VLM prompts use,
    indexed so position i is label i.
  - The reported zero-shot result is the template ensemble (mean of
    L2-normalised text embeddings over TEMPLATES, renormalised). The single
    template TEMPLATE_SINGLE is reported beside it, not selected after seeing
    results.
  - Confidence is the maximum softmax probability over the nine classes at the
    model's learned logit scale (zero-shot) or the probe's predicted
    probability (probe).
  - The probe trains on a seeded, class-stratified 20,000-patch subsample of
    the PathMNIST training split. PathMNIST splits NCT-CRC-HE-100K at the patch
    level, so validation shares slides with training: the probe's validation
    numbers are written with leakage_inflated=true and only the external
    cohort is reportable. The regularisation strength is chosen by 5-fold
    cross-validation inside the training subsample alone.
  - 20,000 patches keeps embedding the training subsample to minutes on a
    GTX 1650 and the feature matrix to about 40 MB, while giving each class
    more than 1,000 examples for a probe with 4,617 parameters.

Outputs (results/plip/):
    embeddings_{val,test,train_subsample}.npy   cached L2-normalised image embeddings
    train_subsample_idx.npy                     indices into the training split
    predictions_{method}_{split}.parquet        per-patch label, prediction, confidence
    run_meta.json                               checkpoint hash, versions, device, settings
    summary.json                                every metric below, per method and split

Usage:
    python scripts/run_plip_baseline.py --dry-run
    python scripts/run_plip_baseline.py [--batch-size 128] [--seed 42] [--revision main]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pathmnist import EXPECTED_SIZES, LABEL_NAMES

logger = logging.getLogger("plip_baseline")

MODEL_ID = "vinid/plip"
OUT_DIR = PROJECT_ROOT / "results" / "plip"
SEED = 42
N_BOOT = 1000
N_BINS = 15
TRAIN_SUBSAMPLE = 20_000
# ViT-B/32 at 224 px in fp16 needs well under 1 GB of activations at this batch,
# which leaves headroom on a 4 GB card for the model and the CUDA context.
BATCH_SIZE = 128
PROBE_CS = (0.01, 0.1, 1.0, 10.0, 100.0)

TEMPLATE_SINGLE = "an H&E image of {}."
TEMPLATES = (
    "an H&E image of {}.",
    "a histopathology image of {}.",
    "a pathology image showing {}.",
    "an H&E stained patch of colorectal tissue showing {}.",
)

CLASS_NAMES: list[str] = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
EVAL_SPLITS = ("val", "test")


def _cache_path(split: str) -> Path:
    return OUT_DIR / f"embeddings_{split}.npy"


# ---------------------------------------------------------------------------
# Model and embeddings (heavy dependencies are imported here, not at module
# import, so --dry-run works on a machine without torch, transformers or data)
# ---------------------------------------------------------------------------

def load_model(revision: str) -> tuple[object, object, object, dict]:
    """Resolve the checkpoint to a commit hash, then load model and processor from it."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    local_dir = Path(snapshot_download(MODEL_ID, revision=revision))
    # snapshot_download returns .../snapshots/<commit>, so the directory name is the pin.
    commit = local_dir.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(local_dir, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(local_dir)
    meta = {
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_commit": commit,
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dtype": str(dtype).replace("torch.", ""),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    return model, processor, device, meta


def embed_images(model, processor, device, images: np.ndarray, batch_size: int,
                 log_progress: bool = True) -> np.ndarray:
    """L2-normalised image embeddings, computed in batches, returned as float32."""
    import torch

    from src.data.pathmnist import arr_to_pil

    out = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = [arr_to_pil(images[i]) for i in range(start, min(start + batch_size, len(images)))]
            inputs = processor(images=batch, return_tensors="pt")
            pixel = inputs["pixel_values"].to(device, dtype=model.dtype)
            feats = model.get_image_features(pixel_values=pixel).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy())
            done = start + len(batch)
            if log_progress and ((done // batch_size) % 20 == 0 or done == len(images)):
                logger.info("  embedded %d/%d (%.0f img/s)", done, len(images),
                            done / max(time.time() - t0, 1e-6))
    return np.concatenate(out).astype(np.float32)


def embed_texts(model, processor, device, templates: tuple[str, ...]) -> np.ndarray:
    """
    One text embedding per class: each template's embedding is L2-normalised,
    averaged over templates, and renormalised, the usual CLIP prompt ensemble.
    """
    import torch

    per_template = []
    with torch.no_grad():
        for tpl in templates:
            texts = [tpl.format(c) for c in CLASS_NAMES]
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
            feats = model.get_text_features(**inputs).float()
            per_template.append((feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy())
    mean = np.mean(per_template, axis=0)
    return (mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float32)


def cached_split_embeddings(model, processor, device, split: str, batch_size: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings and labels for val or test, computed once and cached."""
    from src.data.pathmnist import load_labels, load_split_arrays

    path = _cache_path(split)
    labels = load_labels(split)
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(labels):
            logger.info("Using cached %s", path)
            return emb, labels
        logger.warning("Cached %s has %d rows, expected %d; recomputing", path, len(emb), len(labels))
    images, labels = load_split_arrays(split)
    logger.info("Embedding %s: %d patches", split, len(images))
    emb = embed_images(model, processor, device, images, batch_size)
    np.save(path, emb)
    return emb, labels


def train_subsample_indices(labels: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """Class-stratified subsample, proportional to class frequency, sorted for mmap reads."""
    rng = np.random.default_rng(seed)
    picked = []
    for c in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(labels == c)
        k = round(len(idx) * n_total / len(labels))
        picked.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(picked))


def cached_train_embeddings(model, processor, device, batch_size: int, n_total: int, seed: int
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embeddings for the training subsample. Images come from load_train_mmap, the
    same path the 224 px CNN used, and are read in index order a batch at a time
    so only the batch is copied out of the archive.
    """
    from src.data.pathmnist import load_labels, load_train_mmap

    labels_all = load_labels("train")
    idx = train_subsample_indices(labels_all, n_total, seed)
    idx_path = OUT_DIR / "train_subsample_idx.npy"
    emb_path = _cache_path("train_subsample")
    if emb_path.exists() and idx_path.exists() and np.array_equal(np.load(idx_path), idx):
        logger.info("Using cached %s", emb_path)
        return np.load(emb_path), labels_all[idx], idx
    train_images = load_train_mmap()
    logger.info("Embedding training subsample: %d of %d patches", len(idx), len(labels_all))
    chunks = []
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        chunks.append(embed_images(model, processor, device, np.asarray(train_images[sel]),
                                   batch_size, log_progress=False))
        if (start // batch_size) % 20 == 0:
            logger.info("  embedded %d/%d training patches", start + len(sel), len(idx))
    emb = np.concatenate(chunks)
    np.save(emb_path, emb)
    np.save(idx_path, idx)
    return emb, labels_all[idx], idx


# ---------------------------------------------------------------------------
# Metrics: the same estimators as the VLM tables
# ---------------------------------------------------------------------------

def evaluate(conf: np.ndarray, pred: np.ndarray, labels: np.ndarray, n_boot: int, seed: int
             ) -> dict:
    """Accuracy, F1, ECE, AUROC with CI, selective-accuracy AUC and its gap over random."""
    from sklearn.metrics import f1_score

    from scripts.confidence_information import auroc_midrank, bootstrap_auroc
    from scripts.routing_ci import acc_coverage_auc, bootstrap_gap
    from src.eval.calibration import ece_score
    from src.triage.router import risk_coverage_curve, tie_statistics

    correct = pred == labels
    router_out = risk_coverage_curve(conf, correct, tie_break="expected")
    auc = acc_coverage_auc(conf, correct)
    assert abs(auc - router_out["auc"]) < 1e-9, "fast AUC disagrees with router"
    mc = risk_coverage_curve(conf, correct, tie_break="random", n_repeats=200, seed=seed)
    base = float(correct.mean())
    ties = tie_statistics(conf)
    per_class = f1_score(labels, pred, labels=list(range(len(CLASS_NAMES))), average=None,
                         zero_division=0)
    return {
        "n": len(labels),
        "accuracy": round(base, 4),
        "macro_f1": round(float(f1_score(labels, pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {CLASS_NAMES[i]: round(float(v), 4) for i, v in enumerate(per_class)},
        "ece_15bin": round(ece_score(conf, correct, n_bins=N_BINS), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "auroc": round(auroc_midrank(conf, correct), 4),
        "auroc_ci": bootstrap_auroc(conf, correct, n_boot, seed),
        "auroc_tie_handling": "mid-rank (Mann-Whitney)",
        "selective_auc": round(float(auc), 4),
        "selective_auc_tie_sd": round(float(mc["auc_sd"]), 4),
        # Random routing integrates to base accuracy in expectation, the same
        # reference the VLM gaps use.
        "random_auc": round(base, 4),
        "gap_point": round(float(auc - base), 4),
        "gap_bootstrap": bootstrap_gap(conf, correct, n_boot, seed),
        "tie_fraction": round(float(ties["tie_fraction"]), 4),
        "n_distinct": int(ties["n_distinct"]),
    }


def zero_shot(img_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float
              ) -> tuple[np.ndarray, np.ndarray]:
    logits = logit_scale * img_emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1), probs.max(axis=1)


def write_predictions(method: str, split: str, labels: np.ndarray, pred: np.ndarray,
                      conf: np.ndarray) -> Path:
    import pandas as pd

    path = OUT_DIR / f"predictions_{method}_{split}.parquet"
    pd.DataFrame({
        "arr_idx": np.arange(len(labels)),
        "true_label_idx": labels.astype(int),
        "true_label_name": [CLASS_NAMES[i] for i in labels],
        "pred_label_idx": pred.astype(int),
        "pred_label_name": [CLASS_NAMES[i] for i in pred],
        "confidence": conf.astype(float),
        "correct": pred == labels,
    }).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------

def dry_run(args: argparse.Namespace) -> int:
    n_eval = sum(EXPECTED_SIZES[s] for s in EVAL_SPLITS)
    print(f"Model: {MODEL_ID} (revision requested: {args.revision}; commit resolved at run time)")
    print(f"Seed: {args.seed}   bootstrap resamples: {args.n_boot}   batch size: {args.batch_size}")
    print(f"Evaluation patches: val {EXPECTED_SIZES['val']:,} + test {EXPECTED_SIZES['test']:,}"
          f" = {n_eval:,}")
    print(f"Probe training subsample: about {args.train_n:,} of {EXPECTED_SIZES['train']:,}"
          " training patches, class-stratified (per-class rounding; run_meta.json records"
          " the exact count)")
    total = n_eval + args.train_n
    print(f"Images to embed on a cold cache: {total:,}"
          f" ({-(-total // args.batch_size):,} forward passes)")
    print(f"Probe C grid (5-fold CV on the training subsample only): {PROBE_CS}")
    print("Cache paths:")
    for split in (*EVAL_SPLITS, "train_subsample"):
        print(f"  {_cache_path(split)}")
    print(f"  {OUT_DIR / 'train_subsample_idx.npy'}")
    print(f"Single template (reported beside the ensemble): {TEMPLATE_SINGLE!r}")
    print("Ensemble templates (the reported zero-shot result):")
    for tpl in TEMPLATES:
        print(f"  {tpl!r}")
    print("Rendered class prompts, first ensemble template:")
    for c in CLASS_NAMES:
        print(f"  {TEMPLATES[0].format(c)}")
    print(f"Outputs: {OUT_DIR / 'summary.json'}, {OUT_DIR / 'run_meta.json'},"
          " predictions_{zeroshot_ensemble,zeroshot_single,probe}_{val,test}.parquet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PLIP zero-shot and linear-probe baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--train-n", type=int, default=TRAIN_SUBSAMPLE)
    parser.add_argument("--revision", default="main",
                        help="Hugging Face revision; pass the commit hash from run_meta.json to rerun")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.dry_run:
        return dry_run(args)

    import torch
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold

    logger.info("Seed: %d", args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    model, processor, device, meta = load_model(args.revision)
    logit_scale = float(model.logit_scale.exp().item())
    text_single = embed_texts(model, processor, device, (TEMPLATE_SINGLE,))
    text_ensemble = embed_texts(model, processor, device, TEMPLATES)

    summary: dict = {"zeroshot_ensemble": {}, "zeroshot_single": {}, "probe": {}}
    eval_emb: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in EVAL_SPLITS:
        emb, labels = cached_split_embeddings(model, processor, device, split, args.batch_size)
        eval_emb[split] = (emb, labels)
        for method, text in (("zeroshot_ensemble", text_ensemble), ("zeroshot_single", text_single)):
            pred, conf = zero_shot(emb, text, logit_scale)
            summary[method][split] = evaluate(conf, pred, labels, args.n_boot, args.seed)
            write_predictions(method, split, labels, pred, conf)
            logger.info("%s %s: acc %.4f  AUROC %.4f  gap %+.4f", method, split,
                        summary[method][split]["accuracy"], summary[method][split]["auroc"],
                        summary[method][split]["gap_point"])

    train_emb, train_labels, train_idx = cached_train_embeddings(
        model, processor, device, args.batch_size, args.train_n, args.seed)
    probe = LogisticRegressionCV(
        Cs=list(PROBE_CS), cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
        max_iter=5000, random_state=args.seed, n_jobs=1,
    ).fit(train_emb, train_labels)
    for split in EVAL_SPLITS:
        emb, labels = eval_emb[split]
        probs = probe.predict_proba(emb)
        pred, conf = probs.argmax(axis=1), probs.max(axis=1)
        result = evaluate(conf, pred, labels, args.n_boot, args.seed)
        # Validation shares slides with the probe's training patches.
        result["leakage_inflated"] = split == "val"
        summary["probe"][split] = result
        write_predictions("probe", split, labels, pred, conf)
        logger.info("probe %s: acc %.4f  AUROC %.4f  gap %+.4f%s", split, result["accuracy"],
                    result["auroc"], result["gap_point"],
                    "  (leakage-inflated)" if split == "val" else "")

    meta.update({
        "seed": args.seed,
        "n_boot": args.n_boot,
        "batch_size": args.batch_size,
        "logit_scale": round(logit_scale, 4),
        "template_single": TEMPLATE_SINGLE,
        "templates_ensemble": list(TEMPLATES),
        "reported_zero_shot": "zeroshot_ensemble",
        "class_names": CLASS_NAMES,
        "probe": {
            "train_split": "train",
            "train_n": len(train_idx),
            "train_class_counts": {CLASS_NAMES[c]: int((train_labels == c).sum())
                                   for c in range(len(CLASS_NAMES))},
            "C_grid": list(PROBE_CS),
            "C_selected": float(np.atleast_1d(probe.C_)[0]),
            "cv": "5-fold stratified, training subsample only",
        },
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    meta_path = OUT_DIR / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({
        "note": ("Zero-shot rows involve no training on either cohort for this task; whether "
                 "PLIP's pretraining images overlap these cohorts is not known. Probe validation "
                 "rows share slides "
                 "with the probe's training patches and are flagged leakage_inflated; only the "
                 "external (test) probe rows are reportable."),
        "model_commit": meta["resolved_commit"],
        **summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {meta_path}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
