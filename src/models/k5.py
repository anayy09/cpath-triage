"""
src/models/k5.py

Resumable K-query plumbing shared by the round-3 consistency runs
(scripts/run_consistency_v3w.py, scripts/run_image_controls.py).

scripts/run_consistency.py, which produced the archived V3 and V5 results, wrote
nothing until the last patch, so an interrupted run lost every call it had made.
The runs here are 2,000 to 9,000 billed calls each, so every completed call is
appended to a JSONL log and flushed before the next one starts; re-running the
same command skips what is already logged. Failed calls are not logged, so a
re-run retries them.

The per-patch summary reproduces run_consistency.py field for field (modal label
by Counter.most_common over parsed labels, consistency = modal count / K, mean
of all K confidences including unparsed ones), so the outputs can be read by
scripts/consistency_routing_table.py and compared with the stored V3 run.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Job:
    """One billed call: a patch key, a variant index, and a repeat index."""

    key: str
    variant: int
    repeat: int = 0

    @property
    def log_key(self) -> str:
        return f"{self.key}|{self.variant}|{self.repeat}"


class CallLog:
    """Append-only JSONL log of completed calls, keyed by Job.log_key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[rec["log_key"]] = rec

    def done(self, job: Job) -> bool:
        return job.log_key in self.records

    def append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[rec["log_key"]] = rec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_jobs(
    jobs: Iterable[Job],
    call: Callable[[Job], object],
    log: CallLog,
    *,
    extra_fields: Callable[[Job], dict] | None = None,
    progress_every: int = 100,
    logger=None,
) -> tuple[int, int]:
    """
    Execute every job not already in the log.

    `call` returns a client.Response. Exceptions are counted and skipped rather
    than logged, so the next invocation retries them. Returns (made, failed).
    """
    jobs = list(jobs)
    pending = [j for j in jobs if not log.done(j)]
    if logger:
        logger.info("%d jobs, %d already logged, %d to run", len(jobs), len(jobs) - len(pending), len(pending))
    made = failed = 0
    t_start = time.time()
    for i, job in enumerate(pending, 1):
        t0 = time.time()
        try:
            resp = call(job)
        except Exception as exc:
            failed += 1
            if logger:
                logger.warning("call failed %s: %s", job.log_key, str(exc)[:160])
            continue
        rec = {
            "log_key": job.log_key,
            "key": job.key,
            "variant": job.variant,
            "repeat": job.repeat,
            "label": resp.prediction,
            "confidence": resp.confidence,
            "raw_text": resp.raw_text,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": round(time.time() - t0, 3),
            "ts_utc": utc_now(),
        }
        if extra_fields:
            rec.update(extra_fields(job))
        log.append(rec)
        made += 1
        if logger and (i % progress_every == 0 or i == len(pending)):
            rate = i / max(time.time() - t_start, 1e-9) * 60
            logger.info("%d/%d calls | %.0f calls/min | failed=%d", i, len(pending), rate, failed)
    return made, failed


def summarize_k(labels: list[str], confs: list[float], true_name: str, k: int) -> dict:
    """Per-patch K=5 summary, identical in definition to run_consistency.py."""
    valid = [lbl for lbl in labels if lbl != "unknown"]
    if valid:
        counts = Counter(valid)
        modal_label = counts.most_common(1)[0][0]
        consistency = counts[modal_label] / k
    else:
        modal_label = "unknown"
        consistency = 0.0
    return {
        "modal_label": modal_label,
        "consistency_score": round(consistency, 4),
        "uncertainty_score": round(1.0 - consistency, 4),
        "is_correct": bool(modal_label == true_name),
        "mean_textual_conf": round(float(sum(confs) / len(confs)), 4),
        "raw_labels": json.dumps(labels),
        "raw_confs": json.dumps([round(c, 3) for c in confs]),
    }


def collect(log: CallLog, key: str, k: int, repeat: int = 0) -> tuple[list[str], list[float]] | None:
    """Labels and confidences for one patch in variant order, or None if incomplete."""
    labels: list[str] = []
    confs: list[float] = []
    for v in range(k):
        rec = log.records.get(f"{key}|{v}|{repeat}")
        if rec is None:
            return None
        labels.append(rec["label"])
        confs.append(float(rec["confidence"]))
    return labels, confs


def write_run_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def log_time_window(log: CallLog) -> dict:
    """First and last request timestamps in the log, for the Methods provenance line."""
    ts = sorted(r["ts_utc"] for r in log.records.values())
    return {"first_utc": ts[0] if ts else None, "last_utc": ts[-1] if ts else None, "n_calls": len(ts)}

