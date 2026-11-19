"""Text -> vector, with a small multilingual model that runs on CPU.

We use fastembed (ONNX runtime) instead of PyTorch: same model quality,
but the Docker image is about 1.5 GB smaller and starts faster.

The e5 models were trained with prefixes: questions start with "query: "
and documents with "passage: ". Forgetting them lowers retrieval quality.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from rag_assistant.config import EMBED_DIM, EMBED_MODEL, ROOT

CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", str(ROOT / ".cache" / "fastembed"))


def _register_e5_small() -> None:
    """multilingual-e5-small is not in fastembed's built-in list, so we
    register its official ONNX export (MIT licence) as a custom model."""
    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if "intfloat/multilingual-e5-small" not in known:
        TextEmbedding.add_custom_model(
            model="intfloat/multilingual-e5-small",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="intfloat/multilingual-e5-small"),
            dim=384,
            model_file="onnx/model.onnx",
        )


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    _register_e5_small()
    return TextEmbedding(model_name=EMBED_MODEL, cache_dir=CACHE_DIR)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    vecs = get_model().embed([f"passage: {t}" for t in texts], batch_size=batch_size)
    return np.asarray(list(vecs), dtype=np.float32).reshape(-1, EMBED_DIM)


def embed_query(text: str) -> np.ndarray:
    return np.asarray(next(iter(get_model().embed([f"query: {text}"]))), dtype=np.float32)
