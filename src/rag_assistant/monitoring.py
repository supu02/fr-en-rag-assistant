"""Monitoring: one JSON line per API request, written to standard output.

Example line (from `POST /ask`):

    {"event": "request", "endpoint": "/ask", "status": 200, "latency_ms": 2140, "lang": "en",
     "retrieval_method": "hybrid", "top_score": 0.031, "n_citations": 2, "no_citation": false,
     "tool_calls": 1, "llm_calls": 3, "input_tokens": 2950, "output_tokens": 120, "est_cost_usd": 0.00052, ...}

Why standard output? Every cloud platform collects it with no extra code (AWS CloudWatch
Logs, Google Cloud Logging, Azure Monitor), and locally you can read it in the terminal.

What is logged: timing, status, language, retrieval method and score, number of
citations, tool calls, tokens and an estimated cost. When the app starts (a cold
start on Lambda), one "startup" line gives the time spent loading the index and the
embedding model.
What is NOT logged: the question and the answer, because users may type personal
details. Only their length is kept.

Live traffic has no "right answer" labels, so these are warning signs, not quality
scores: an answer with no citation is either a refusal or an unsupported answer, and
a low retrieval score often means an out-of-scope question. Quality itself is
measured offline by the evaluation suite (evaluation/).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from rag_assistant.config import ANSWER_MODEL

# Estimated list prices in US dollars per million tokens. They change: check
# https://aws.amazon.com/bedrock/pricing and https://mistral.ai/pricing, and override
# with the environment variables below.
PRICES_PER_M = {
    "mistral.ministral-3-14b-instruct": (0.24, 0.24),  # Bedrock, Europe (Ireland), Nov 2026
    # Mistral Small: third-party price lists, Sept 2026 (not shown on Mistral's own page).
    "mistral-small-latest": (0.15, 0.60),
    "mistral-large-latest": (0.50, 1.50),  # mistral.ai/pricing, Nov 2026
}

logger = logging.getLogger("rag_assistant.requests")


def _setup_logger() -> None:
    """Plain JSON lines on stdout: no timestamp prefix, so log tools can parse each line."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # tests read it with pytest's caplog


_setup_logger()


class TokenCounter(BaseCallbackHandler):
    """Adds up the tokens of every LLM call in one request (translation + answer + tool rounds)."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage = None
        with contextlib.suppress(IndexError, AttributeError):  # some models report no usage
            usage = response.generations[0][0].message.usage_metadata
        with self._lock:
            self.llm_calls += 1
            if usage:
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def as_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


def estimate_cost(input_tokens: int, output_tokens: int, model: str = ANSWER_MODEL) -> float | None:
    """Cost at list prices (on Bedrock, AWS takes it from the account's credits first).
    None if the model's price is unknown."""
    env_in, env_out = os.getenv("PRICE_INPUT_PER_M"), os.getenv("PRICE_OUTPUT_PER_M")
    default = PRICES_PER_M.get(model)
    if default is None and not (env_in and env_out):
        return None
    price_in = float(env_in) if env_in else default[0]
    price_out = float(env_out) if env_out else default[1]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)


def log_event(event: str, **fields: Any) -> dict:
    record = {"event": event, "ts": round(time.time(), 3), **fields}
    logger.info(json.dumps(record, ensure_ascii=False, default=str))
    return record


def log_request(**fields: Any) -> dict:
    return log_event("request", **fields)
