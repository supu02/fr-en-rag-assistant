"""Connection to the LLM, through LangChain.

The models are Mistral's. LLM_PROVIDER (config.py) chooses who runs them:
- bedrock (default): Amazon Bedrock. No API key: AWS credentials come from `aws login`
  on a laptop and from the function's IAM role on AWS Lambda.
- mistral: the Mistral API, with MISTRAL_API_KEY from the environment or the .env file.
Secrets are never written in the code or committed to git.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter

from rag_assistant import config

# Mistral's API and a new AWS account's Bedrock quota both allow few requests per second.
# One shared limiter for every model, so the answer model and the judge stay under it together.
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("LLM_REQUESTS_PER_SECOND", "1")),
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


class LLMNotConfigured(RuntimeError):
    """No credentials for the chosen provider: /ask returns 503, /search still works."""


@lru_cache(maxsize=1)
def _aws_credentials_found() -> bool:
    import boto3
    from botocore.exceptions import BotoCoreError

    try:
        return boto3.Session().get_credentials() is not None
    except BotoCoreError:  # e.g. `aws login` used without the botocore[crt] extra
        return False


def llm_configured() -> bool:
    if config.LLM_PROVIDER == "mistral":
        return bool(os.getenv("MISTRAL_API_KEY"))
    return _aws_credentials_found()


@lru_cache(maxsize=4)
def get_chat_model(model: str = config.ANSWER_MODEL):
    if not llm_configured():
        hint = ("set MISTRAL_API_KEY in .env" if config.LLM_PROVIDER == "mistral"
                else "sign in with `aws login`, or set LLM_PROVIDER=mistral")
        raise LLMNotConfigured(f"No credentials for the LLM ({config.LLM_PROVIDER}): {hint}.")

    if config.LLM_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model,
            temperature=0.0,  # as deterministic as the API allows, for evaluation
            random_seed=0,
            max_retries=6,  # retries with back-off on rate-limit (429) and server errors
            timeout=60,
            rate_limiter=RATE_LIMITER,
        )

    from botocore.config import Config
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(  # Bedrock's Converse API: one format for every model, with tools
        model_id=model,
        provider="mistral",
        region_name=config.BEDROCK_REGION,
        temperature=0.0,
        max_tokens=1024,  # answers are a few sentences; this caps the cost of a runaway reply
        config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=60),
        rate_limiter=RATE_LIMITER,
    )
