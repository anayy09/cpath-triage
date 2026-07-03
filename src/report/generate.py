"""
src/report/generate.py

Preliminary pathology report synthesis from MedGemma patch predictions.

Uses llama-3.3-70b-instruct (text-only, via the shared client) to narrate
an aggregated SlideResult into a short clinical-style summary. The language
is intentionally hedged ("AI analysis suggests...") and never states a
definitive diagnosis.

The SlideResult is a patch-level aggregate, not a whole-slide result.
Claims about the slide as a whole should be framed cautiously and the
report should note that human pathologist review is required.

Usage:
    from src.report.generate import build_slide_result, generate_report

    result = build_slide_result(predictions_df, case_id="CASE-001")
    report_text = generate_report(result)
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.models.client import Client, SlideResult, Response


def build_slide_result(
    predictions_df: pd.DataFrame,
    case_id: str,
    model: str = "medgemma-27b-it",
) -> SlideResult:
    """
    Aggregate a DataFrame of patch predictions into a SlideResult.

    The slide-level prediction is the plurality class across patches.
    Slide-level confidence is the mean of per-patch calibrated confidence
    (or raw confidence if calibrated is not available).

    Args:
        predictions_df: DataFrame with columns pred_label, pred_confidence,
                        true_label_name, raw_text, rationale.
        case_id:        Identifier for the slide / case.
        model:          Model name string for labelling.

    Returns:
        SlideResult ready for generate_report().
    """
    df = predictions_df[predictions_df["pred_label"] != "unknown"].copy()
    if df.empty:
        return SlideResult(
            case_id=case_id,
            prediction="undetermined",
            confidence=0.0,
            n_tiles_analyzed=len(predictions_df),
            error="All predictions were unparseable.",
        )

    # Plurality prediction
    counts = Counter(df["pred_label"].tolist())
    top_label = counts.most_common(1)[0][0]

    # Mean confidence across patches
    conf_col = "cal_confidence" if "cal_confidence" in df.columns else "pred_confidence"
    mean_conf = float(df[conf_col].mean())

    # Build a sample of per-patch Response objects (up to 5 for context)
    tile_responses: list[Response] = []
    for _, row in df.head(5).iterrows():
        tile_responses.append(
            Response(
                prediction=str(row["pred_label"]),
                confidence=float(row.get(conf_col, row["pred_confidence"])),
                rationale=str(row.get("rationale", "")),
                raw_text=str(row.get("raw_text", "")),
                tile_paths=[],
                model=model,
            )
        )

    return SlideResult(
        case_id=case_id,
        prediction=top_label,
        confidence=mean_conf,
        tile_responses=tile_responses,
        n_tiles_analyzed=len(df),
    )


def _report_prompt_extended(result: SlideResult, class_distribution: dict) -> str:
    """Extended prompt that includes class distribution for richer narration."""
    dist_lines = "\n".join(
        f"  - {label}: {n} patches" for label, n in class_distribution.items()
    )
    return (
        "You are an expert computational pathologist writing a preliminary "
        "AI-assisted analysis note.\n\n"
        f"Case ID: {result.case_id}\n"
        f"Total patches analyzed: {result.n_tiles_analyzed}\n"
        f"Predominant tissue type (plurality): {result.prediction}\n"
        f"Mean model confidence: {result.confidence:.0%}\n\n"
        "Patch-level tissue distribution:\n"
        f"{dist_lines}\n\n"
        "Write a 3-4 sentence preliminary report in the style of a computational "
        "pathology note. Requirements:\n"
        "- Begin with 'AI-assisted patch analysis of this colorectal specimen...'\n"
        "- Cite the predominant tissue type and noteworthy secondary findings.\n"
        "- Explicitly note that this is an automated preliminary assessment and "
        "that a qualified pathologist should review the case.\n"
        "- Do not state a definitive diagnosis. Use hedged language such as "
        "'consistent with', 'suggestive of', or 'warrants further review'.\n"
        "- Do not mention confidence percentages in the report text."
    )


def generate_report(
    slide_result: SlideResult,
    class_distribution: dict | None = None,
    report_model: str = "llama-3.3-70b-instruct",
) -> str:
    """
    Generate a preliminary pathology report narrative for a SlideResult.

    Args:
        slide_result:       Aggregated slide-level prediction object.
        class_distribution: Optional dict {label: count} for all patch classes.
                            If provided, the report prompt includes the distribution.
        report_model:       Text model to use for generation.

    Returns:
        Report string. Never states a definitive diagnosis.
    """
    client = Client(model=report_model)

    if class_distribution:
        prompt = _report_prompt_extended(slide_result, class_distribution)
        raw, _, _ = client._call_with_retry(
            model=report_model,
            content=[{"type": "text", "text": prompt}],
            max_tokens=350,
        )
        return raw.strip()

    return client.generate_report(slide_result, model=report_model)
