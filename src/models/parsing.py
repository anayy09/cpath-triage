"""
src/models/parsing.py

Reading a label and a confidence out of a model response.

Every prompt in this project asks for three lines, LABEL, CONFIDENCE and
REASONING. Up to the v5.0 archive the parser ignored that structure: it searched
the whole lowercased response, reasoning sentence included, and returned the
first class name it found in a fixed priority order. "LABEL: background ...
REASONING: no tissue or debris" was therefore stored as debris, and a response
that listed every class was stored as whichever class the search reached first.
About 2% of MedGemma's full-scale V3 labels were affected.

parse_prediction reads the label from the LABEL line. A response with more than
one LABEL line names several classes and is scored "unknown". The whole-text
search survives as parse_prediction_regex, for responses with no LABEL line at
all and so that the stored v5.0 labels can be reproduced exactly
(scripts/rescore_labels.py checks that it does, call by call).

No dependency on the API client, so analysis code can import it without openai.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Order matters for the whole-text search: more specific patterns first so
# "cancer-associated stroma" and "adenocarcinoma" are not shadowed by a bare
# "normal" or "muscle" match.
LABEL_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "egfr_mutation": [
        (r"egfr.mutant|egfr\+|egfr mutation detected|mutant", "EGFR-mutant"),
        (r"egfr.wildtype|egfr-wt|egfr.wild.type|wildtype|wild.type", "EGFR-wildtype"),
    ],
    "kras_mutation": [
        (r"kras.mutant|kras\+|kras mutation", "KRAS-mutant"),
        (r"kras.wildtype|kras-wt|wildtype", "KRAS-wildtype"),
    ],
    "pattern_description": [
        (r"lepidic", "lepidic"),
        (r"acinar", "acinar"),
        (r"papillary", "papillary"),
        (r"solid", "solid"),
        (r"micropapillary", "micropapillary"),
    ],
    # Nine NCT-CRC tissue classes.
    "tissue_classification": [
        (r"colorectal adenocarcinoma|adenocarcinoma|tumou?r epithelium|\btum\b", "colorectal adenocarcinoma epithelium"),
        (r"cancer.associated stroma|\bstroma\b|\bstr\b", "cancer-associated stroma"),
        (r"normal colon mucosa|normal mucosa|\bnorm\b", "normal colon mucosa"),
        (r"smooth muscle|\bmuscle\b|\bmus\b", "smooth muscle"),
        (r"lymphocyte|\blym\b", "lymphocytes"),
        (r"\bmucus\b|\bmuc\b", "mucus"),
        (r"\bdebris\b|\bdeb\b", "debris"),
        (r"adipose|\badi\b|fat tissue", "adipose"),
        (r"background|\bback\b|empty|glass", "background"),
    ],
}

_LABEL_LINE = re.compile(r"^\s*\**\s*label\s*\**\s*:\s*\**\s*(.*?)\s*\**\s*$", re.IGNORECASE | re.MULTILINE)
_CONF_LINE = re.compile(r"^\s*\**\s*confidence\s*\**\s*:\s*\**\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE | re.MULTILINE)


def _search_label(text_lower: str, task: str) -> str:
    for pattern, label in LABEL_PATTERNS.get(task, []):
        if re.search(pattern, text_lower):
            return label
    return "unknown"


def _confidence_regex(text: str) -> float:
    """The v5.0 confidence rule: the first "N%" anywhere, else "confidence: N"."""
    conf_match = re.search(r"(\d{1,3})\s*%", text)
    if conf_match:
        return max(0.0, min(1.0, float(conf_match.group(1)) / 100.0))
    conf_match = re.search(r"confidence[:\s]+([0-9]+(?:\.[0-9]+)?)", text.lower())
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    if confidence > 1.0:
        confidence /= 100.0
    return confidence


def parse_prediction_regex(text: str, task: str) -> tuple[str, float]:
    """
    The v5.0 parser, unchanged: whole-text class search and the v5.0 confidence
    rule. Returns ("unknown", 0.5) when nothing matches.
    """
    prediction = _search_label(text.lower(), task)
    if prediction == "unknown":
        logger.debug("Could not parse prediction from: %s", text[:200])
    return prediction, _confidence_regex(text)


def label_from_line(line: str, task: str) -> str:
    """Map the text after "LABEL:" to a class: exact name first, then the class patterns."""
    s = line.strip().strip("*").strip().rstrip(".").strip().lower()
    names = {label for _, label in LABEL_PATTERNS.get(task, [])}
    if s in names:
        return s
    return _search_label(s, task)


def parse_prediction(text: str, task: str) -> tuple[str, float]:
    """
    Extract a prediction label and confidence from raw model text.

    The label comes from the LABEL line; with no LABEL line the whole-text search
    of parse_prediction_regex is used, and with several the response is scored
    "unknown". The confidence comes from the first CONFIDENCE line, falling back to
    the v5.0 rule. Returns ("unknown", 0.5) if nothing can be read.
    """
    lines = _LABEL_LINE.findall(text)
    if not lines:
        prediction = _search_label(text.lower(), task)
    elif len(lines) > 1:
        prediction = "unknown"
    else:
        prediction = label_from_line(lines[0], task)

    conf = _CONF_LINE.search(text)
    if conf:
        confidence = float(conf.group(1))
        if confidence > 1.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = _confidence_regex(text)

    if prediction == "unknown":
        logger.debug("Could not parse prediction from: %s", text[:200])
    return prediction, confidence


def calibration_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows that enter the calibration analyses: every row the stored v5.0 labels did
    not mark "unknown".

    The 80/20 calibration split is a seeded permutation over exactly these rows, so
    their set is held to its v5.0 definition. Re-scoring marks some list answers
    unknown; letting that change the set would redraw every partition, and the
    label-token validation samples, which were drawn as those partitions, would no
    longer match them. Correctness inside the rows comes from the current labels.
    """
    col = "pred_label_regex" if "pred_label_regex" in df.columns else "pred_label"
    return df[df[col] != "unknown"].copy()
