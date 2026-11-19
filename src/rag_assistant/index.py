"""Step 2: embed every chunk and store the vectors in a Qdrant collection.

Qdrant runs here in "local mode": a library writing to a folder, no server.
The same code works with a Qdrant server or Qdrant Cloud by changing the
client line (QdrantClient(url=...)).

Run:  python -m rag_assistant.index
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_assistant.config import CHUNKS_PATH, EMBED_DIM, INDEX_DIR
from rag_assistant.embeddings import embed_passages

COLLECTION = "chunks"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _copy_writable(src: Path, dst: Path) -> None:
    """Copy file contents only (not the read-only permissions), then rename at the end,
    so a copy cut in the middle is never used."""
    partial = dst.with_name(dst.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    for item in src.rglob("*"):
        target = partial / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    partial.rename(dst)


def get_client() -> QdrantClient:
    path = INDEX_DIR / "qdrant"
    if path.exists() and not os.access(path, os.W_OK):
        # Read-only disk (AWS Lambda): Qdrant local mode must write a lock file,
        # so it opens a copy in the temp folder, the only writable place there (~40 MB).
        copy = Path(tempfile.gettempdir()) / "qdrant-index"
        if not copy.exists():
            _copy_writable(path, copy)
        path = copy
    return QdrantClient(path=str(path))


def build_index(batch: int = 256) -> None:
    chunks = load_chunks()
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    # Cosine similarity: the vectors are normalised, so this is a dot product.
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))

    start = time.time()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vectors = embed_passages([c["text"] for c in part])
        client.upsert(COLLECTION, points=[
            PointStruct(id=i + j, vector=vectors[j].tolist(), payload=c) for j, c in enumerate(part)
        ])
        print(f"  {min(i + batch, len(chunks))}/{len(chunks)} chunks embedded ({time.time() - start:.0f}s)")
    client.close()
    print(f"Index ready in {INDEX_DIR / 'qdrant'}")


if __name__ == "__main__":
    build_index()
