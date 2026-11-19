"""All paths and settings in one place. Values can be overridden with
environment variables (or a local .env file, which is never committed)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw" / "vosdroits"
CHUNKS_PATH = ROOT / "data" / "corpus" / "chunks.jsonl"
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
RESULTS_DIR = ROOT / "results"

# Embedding model: small, multilingual (about 100 languages), MIT licence.
# It maps French and English text into the same vector space, so an English
# question can find a French passage.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = 384

# LLMs: Mistral models, run by one of two providers (LLM_PROVIDER):
# - "bedrock" (default): Amazon Bedrock, billed to the AWS account, no API key. Locally the
#   credentials come from `aws login`; on Lambda from the function's IAM role.
#   Bedrock serves the answer model in Ireland (eu-west-1); the app itself can run anywhere.
#   The judge, Mistral Large 3, is only offered in US East (us-east-1). It is used offline by
#   the evaluation and only sees the public test questions and public page excerpts.
# - "mistral": the Mistral API (needs MISTRAL_API_KEY on a plan that includes API access).
# The judge is a larger model than the answering model.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").lower()
DEFAULT_MODELS = {  # provider: (answer model, judge model)
    "bedrock": ("mistral.ministral-3-14b-instruct", "mistral.mistral-large-3-675b-instruct"),
    "mistral": ("mistral-small-latest", "mistral-large-latest"),
}
if LLM_PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, not {LLM_PROVIDER!r}")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", DEFAULT_MODELS[LLM_PROVIDER][0])
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODELS[LLM_PROVIDER][1])
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "eu-west-1")
BEDROCK_JUDGE_REGION = os.getenv("BEDROCK_JUDGE_REGION", "us-east-1")

TOP_K = int(os.getenv("TOP_K", "5"))
