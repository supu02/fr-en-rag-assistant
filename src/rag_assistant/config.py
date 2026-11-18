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

# LLMs (Mistral API). The judge is a larger model than the answering model.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "mistral-small-latest")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "mistral-large-latest")

TOP_K = int(os.getenv("TOP_K", "5"))
