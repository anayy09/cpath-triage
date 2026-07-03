"""
scripts/probe_logprobs.py

One-call diagnostic (reviewer M5i): does the inference endpoint expose token
log-probabilities for the VLM / base models? If yes, a logit or
sequence-probability confidence baseline becomes feasible and we can add it; if
no, we document the API constraint honestly in Methods and Limitations. This
makes a single API call through the project Client (no OpenAI client is
instantiated here, per the project rule).

Outputs:
    results/logprobs_probe.json

Usage:
    python scripts/probe_logprobs.py
    python scripts/probe_logprobs.py --model gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.client import Client


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe endpoint for token logprob support")
    ap.add_argument("--model", default=None, help="Model to probe (default: client default).")
    ap.add_argument("--top-logprobs", type=int, default=5)
    args = ap.parse_args()

    client = Client(model=args.model) if args.model else Client()

    # A tiny synthetic patch so the probe exercises the multimodal path the real
    # evaluation uses, not just a text-only call.
    tmp = Path(tempfile.gettempdir()) / "cpath_logprob_probe.png"
    Image.new("RGB", (16, 16), (200, 180, 190)).save(tmp)

    result = client.probe_logprobs(
        prompt_text="Reply with one word naming a tissue type.",
        image_path=tmp,
        top_logprobs=args.top_logprobs,
    )
    result["model"] = args.model or client.model

    out_path = PROJECT_ROOT / "results" / "logprobs_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nLOGPROBS SUPPORTED: {result.get('supported')}")
    if result.get("supported"):
        print("-> A logit/sequence-probability confidence baseline is feasible (M5i).")
    else:
        print("-> Endpoint does not return usable logprobs; document as an API constraint (M5i).")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
