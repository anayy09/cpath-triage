"""
src/models/client.py

OpenAI-compatible client for inference API.
All model calls in this project go through this module. Other modules must not
instantiate the OpenAI client directly.

Usage:
    from src.models.client import Client, Response

    client = Client()
    result = client.analyze_tiles(
        tile_paths=["path/to/tile.png"],
        clinical_context="68yo female, LUAD",
        task="egfr_mutation",
    )
    print(result.prediction, result.confidence)
"""

import os
import re
import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class Response:
    """Parsed response from a single inference call."""

    prediction: str               # e.g., "EGFR-mutant"
    confidence: float             # 0.0–1.0
    rationale: str                # free-text reasoning from the model
    raw_text: str                 # unparsed model output
    tile_paths: list[str]         # which tiles were sent in this call
    model: str = "medgemma-27b-it"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SlideResult:
    """Aggregated result across all tile batches for a single slide."""

    case_id: str
    prediction: str
    confidence: float             # slide-level confidence (aggregated)
    tile_responses: list[Response] = field(default_factory=list)
    n_tiles_analyzed: int = 0
    error: Optional[str] = None   # set if inference failed for this slide


class Client:
    """
    Client for inference API.

    Reads API_KEY and BASE_URL from environment. Implements exponential
    backoff, tile batching, base64 image encoding, and structured response parsing.

    Args:
        model: Default model name. Override per-call if needed.
        tiles_per_call: Number of tile images per API call. Max 9 for safety.
        max_retries: Maximum retry attempts on retryable errors (429, 5xx).
        base_delay: Initial backoff delay in seconds.
    """

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str = "medgemma-27b-it",
        tiles_per_call: int = 9,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ):
        self.model = model
        self.tiles_per_call = min(tiles_per_call, 16)  # hard cap
        self.max_retries = max_retries
        self.base_delay = base_delay

        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("BASE_URL")

        if not api_key:
            raise ValueError("API_KEY is not set in environment. Check .env.")
        if not base_url:
            raise ValueError("BASE_URL is not set in environment. Check .env.")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("Client initialized: model=%s, base_url=%s", model, base_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_tiles(
        self,
        tile_paths: list[str | Path],
        clinical_context: str,
        task: str,
        prompt_text: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 512,
    ) -> Response:
        """
        Send a batch of tiles to MedGemma for analysis.

        Tiles are base64-encoded and sent as image_url content blocks. The call
        is retried on transient errors. Only sends up to `self.tiles_per_call`
        tiles; callers are responsible for batching.

        Args:
            tile_paths: Paths to PNG tile images. Maximum `tiles_per_call` items.
            clinical_context: Clinical metadata string appended to the prompt.
            task: Task key (e.g., "egfr_mutation"). Used to select default prompt
                if prompt_text is None.
            prompt_text: Override prompt. If None, uses the default for task.
            model: Override model. If None, uses self.model.
            max_tokens: Maximum tokens in model response.

        Returns:
            Response with parsed prediction, confidence, and rationale.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            ValueError: If tile_paths is empty or any path does not exist.
        """
        if not tile_paths:
            raise ValueError("tile_paths is empty.")

        tile_paths = [Path(p) for p in tile_paths]
        for p in tile_paths:
            if not p.exists():
                raise ValueError(f"Tile not found: {p}")

        if len(tile_paths) > self.tiles_per_call:
            logger.warning(
                "analyze_tiles received %d tiles but tiles_per_call=%d. "
                "Truncating to first %d. Use batch_slide() for full-slide inference.",
                len(tile_paths), self.tiles_per_call, self.tiles_per_call,
            )
            tile_paths = tile_paths[: self.tiles_per_call]

        model = model or self.model
        prompt = prompt_text or _default_prompt(task, clinical_context)

        content = [self._encode_tile(p) for p in tile_paths]
        content.append({"type": "text", "text": prompt})

        raw_text, prompt_tokens, completion_tokens = self._call_with_retry(
            model=model,
            content=content,
            max_tokens=max_tokens,
        )

        prediction, confidence = _parse_prediction(raw_text, task)
        rationale = _extract_rationale(raw_text)

        return Response(
            prediction=prediction,
            confidence=confidence,
            rationale=rationale,
            raw_text=raw_text,
            tile_paths=[str(p) for p in tile_paths],
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def embed_text(self, text: str, model: str = "nomic-embed-text-v1.5") -> list[float]:
        """
        Embed a clinical text string using the specified embedding model.

        Args:
            text: The text to embed (e.g., patient clinical notes).
            model: Embedding model name.

        Returns:
            List of floats representing the embedding vector.
        """
        response = self._client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def generate_report(
        self,
        slide_result: "SlideResult",
        model: str = "llama-3.3-70b-instruct",
    ) -> str:
        """
        Generate a clinical report narrative from a SlideResult.

        Uses llama-3.3-70b-instruct (text only) to synthesize the MedGemma
        tile-level predictions into a coherent preliminary pathology report.

        Args:
            slide_result: Aggregated inference result for a slide.
            model: Text model to use for report generation.

        Returns:
            Report string in plain text.
        """
        prompt = _report_prompt(slide_result)
        raw, _, _ = self._call_with_retry(
            model=model,
            content=[{"type": "text", "text": prompt}],
            max_tokens=800,
        )
        return raw

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_tile(self, tile_path: Path) -> dict:
        """Base64-encode a tile PNG and return an image_url content block."""
        with open(tile_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }

    def _call_with_retry(
        self,
        model: str,
        content: list[dict],
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """
        Call the chat completions endpoint with exponential backoff.

        Args:
            model: Model identifier string.
            content: List of content blocks (text + image_url).
            max_tokens: Token budget for response.

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens).

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        messages = [{"role": "user", "content": content}]
        delay = self.base_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,  # deterministic inference
                )
                text = response.choices[0].message.content or ""
                if not text.strip():
                    logger.warning("Empty response on attempt %d", attempt)
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, 60.0)
                        continue
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                return text, prompt_tokens, completion_tokens

            except APIStatusError as e:
                if e.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    logger.warning(
                        "HTTP %d on attempt %d/%d. Retrying in %.0fs.",
                        e.status_code, attempt, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                else:
                    raise RuntimeError(
                        f"API error (HTTP {e.status_code}): {e.message}"
                    ) from e

            except (APIConnectionError, APITimeoutError) as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Connection error on attempt %d/%d: %s. Retrying in %.0fs.",
                        attempt, self.max_retries, str(e)[:80], delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                else:
                    raise RuntimeError(f"API unreachable after {self.max_retries} attempts") from e

        raise RuntimeError(f"All {self.max_retries} retry attempts exhausted.")


# ── Prompt and parsing helpers ────────────────────────────────────────────────

def _default_prompt(task: str, clinical_context: str) -> str:
    """Return a default prompt for the given task. See prompts.py for V1/V2/V3."""
    from src.models.prompts import get_prompt
    return get_prompt(task=task, version="V2", clinical_context=clinical_context)


def _parse_prediction(text: str, task: str) -> tuple[str, float]:
    """
    Extract a prediction label and confidence from raw model text.

    Args:
        text: Raw model response text.
        task: Task key (e.g., "egfr_mutation").

    Returns:
        Tuple of (prediction_label, confidence_float).
        Returns ("unknown", 0.5) if parsing fails.
    """
    text_lower = text.lower()

    # Map task to label patterns
    label_patterns = {
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
        # Nine NCT-CRC tissue classes. Order matters: more specific patterns
        # first so "cancer-associated stroma" and "adenocarcinoma" are not
        # shadowed by a bare "normal" or "muscle" match.
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

    prediction = "unknown"
    for pattern, label in label_patterns.get(task, []):
        if re.search(pattern, text_lower):
            prediction = label
            break

    # Parse confidence percentage
    conf_match = re.search(r"(\d{1,3})\s*%", text)
    if conf_match:
        confidence = float(conf_match.group(1)) / 100.0
        confidence = max(0.0, min(1.0, confidence))
    else:
        # Check for decimal confidence
        conf_match = re.search(r"confidence[:\s]+([0-9]+(?:\.[0-9]+)?)", text_lower)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        if confidence > 1.0:
            confidence /= 100.0

    if prediction == "unknown":
        logger.debug("Could not parse prediction from: %s", text[:200])

    return prediction, confidence


def _extract_rationale(text: str) -> str:
    """
    Extract the reasoning portion from model output.
    Returns the full text if no structured rationale section is found.
    """
    # Look for explicit reasoning markers
    for marker in ["reasoning:", "rationale:", "explanation:", "because"]:
        idx = text.lower().find(marker)
        if idx != -1:
            return text[idx:].strip()[:500]
    # Return first 300 chars as a fallback
    return text.strip()[:300]


def _report_prompt(result: "SlideResult") -> str:
    """Build a report generation prompt from a SlideResult."""
    tile_summary = f"{result.n_tiles_analyzed} tile patches analyzed"
    return (
        f"You are an expert computational pathologist. Generate a concise preliminary "
        f"pathology report based on the following AI analysis results.\n\n"
        f"Case ID: {result.case_id}\n"
        f"Prediction: {result.prediction}\n"
        f"Confidence: {result.confidence:.0%}\n"
        f"Tiles analyzed: {tile_summary}\n\n"
        f"Write 2-3 sentences summarizing the findings in clinical language suitable "
        f"for a pathologist to review. Do not state this as a final diagnosis. "
        f"Use phrases like 'AI analysis suggests...' or 'Preliminary assessment indicates...'."
    )


# ── CLI self-test ─────────────────────────────────────────────────────────────

def _selftest() -> int:
    """
    Minimal connectivity check: one text call and one single-pixel image call.
    Prints latency and token usage. Returns 0 on success, 1 on failure.
    Run with: python -m src.models.client --selftest
    """
    import io
    from PIL import Image

    logging.basicConfig(level=logging.INFO)
    try:
        client = Client()
    except ValueError as e:
        print(f"[selftest] config error: {e}")
        return 1

    # 1. Text-only call through the retry path.
    t0 = time.time()
    try:
        text, ptok, ctok = client._call_with_retry(
            model=client.model,
            content=[{"type": "text", "text": "Reply with the single word: ok"}],
            max_tokens=8,
        )
        print(f"[selftest] text call ok in {time.time() - t0:.2f}s -> {text.strip()[:40]!r} "
              f"(tokens: {ptok}+{ctok})")
    except Exception as e:
        print(f"[selftest] text call FAILED: {e}")
        return 1

    # 2. Single tiny image call. Encode a 1x1 white PNG on the fly.
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    t0 = time.time()
    try:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Reply with the single word: seen"},
        ]
        text, ptok, ctok = client._call_with_retry(model=client.model, content=content, max_tokens=8)
        print(f"[selftest] image call ok in {time.time() - t0:.2f}s -> {text.strip()[:40]!r} "
              f"(tokens: {ptok}+{ctok})")
    except Exception as e:
        print(f"[selftest] image call FAILED: {e}")
        return 1

    print("[selftest] all checks passed")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Usage: python -m src.models.client --selftest")
