"""Step 2: embed every chunk and store the vectors in a Qdrant collection.

Qdrant runs here in "local mode": a library writing to a folder, no server.
The same code works with a Qdrant server or Qdrant Cloud by changing the
client line (QdrantClient(url=...)).

Run:  python -m rag_assistant.index
"""

from __future__ import annotations

import json
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_assistant.config import CHUNKS_PATH, EMBED_DIM, INDEX_DIR
from rag_assistant.embeddings import embed_passages

COLLECTION = "chunks"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def get_client() -> QdrantClient:
    return QdrantClient(path=str(INDEX_DIR / "qdrant"))


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
