"""Connection to the LLM (Mistral API, through LangChain).

The API key is read from the environment (or the local .env file).
It is never written in the code or committed to git.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter

from rag_assistant.config import ANSWER_MODEL

# The free Mistral plan allows about 1 request per second. One shared limiter
# for every model, so the answer model and the judge never exceed it together.
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("LLM_REQUESTS_PER_SECOND", "1")),
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


class MissingAPIKey(RuntimeError):
    pass


def has_api_key() -> bool:
    return bool(os.getenv("MISTRAL_API_KEY"))


@lru_cache(maxsize=4)
def get_chat_model(model: str = ANSWER_MODEL):
    if not has_api_key():
        raise MissingAPIKey("MISTRAL_API_KEY is not set. Copy .env.example to .env and paste your key.")
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=model,
        temperature=0.0,  # as deterministic as the API allows, for evaluation
        random_seed=0,
        max_retries=6,  # retries with back-off on rate-limit (429) and server errors
        timeout=60,
        rate_limiter=RATE_LIMITER,
    )
