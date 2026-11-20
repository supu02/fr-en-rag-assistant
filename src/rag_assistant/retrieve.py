"""Step 3: find the chunks most relevant to a question.

Retrieval methods (the evaluation compares them):
- dense:   cosine similarity between e5 vectors (works across languages)
- lexical: BM25 keyword matching (strong on exact words, but an English
           question shares almost no words with a French page)
- hybrid:  both lists merged with Reciprocal Rank Fusion (RRF)
- dense_diverse: dense, with at most 2 chunks per page in the results
- auto:    hybrid for French questions, dense_diverse for English ones
           (chosen from the retrieval evaluation, see results/retrieval.json)
"""

from __future__ import annotations

import re
import threading
import unicodedata
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag_assistant.config import TOP_K
from rag_assistant.embeddings import embed_query
from rag_assistant.index import COLLECTION, get_client, load_chunks
from rag_assistant.lang import detect_language

METHODS = ("dense", "dense_diverse", "lexical", "hybrid", "auto")


def resolve_method(query: str, method: str = "auto") -> str:
    """The method actually used: 'auto' becomes 'hybrid' (French) or 'dense_diverse' (English)."""
    if method != "auto":
        return method
    return "hybrid" if detect_language(query) == "fr" else "dense_diverse"


def tokenize(text: str) -> list[str]:
    """Lowercase, remove accents, keep words of 2+ letters (for BM25)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 1]


class Retriever:
    def __init__(self) -> None:
        self.client = get_client()
        self.chunks = load_chunks()  # point id i in Qdrant == self.chunks[i]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._position = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self._lock = threading.Lock()  # Qdrant local mode is not made for parallel calls

    def dense(self, query: str, k: int = TOP_K) -> list[dict]:
        hits = self.client.query_points(COLLECTION, query=embed_query(query).tolist(), limit=k).points
        return [{**h.payload, "score": float(h.score)} for h in hits]

    def dense_diverse(self, query: str, k: int = TOP_K, per_page: int = 2, pool: int = 50) -> list[dict]:
        """Dense search, but at most `per_page` chunks from the same page.
        Many pages repeat one text for several cases, so plain top-k is often
        filled by one page; this keeps room for other candidate pages."""
        out, count = [], {}
        for chunk in self.dense(query, pool):
            if count.get(chunk["page_id"], 0) < per_page:
                out.append(chunk)
                count[chunk["page_id"]] = count.get(chunk["page_id"], 0) + 1
            if len(out) == k:
                break
        return out

    def lexical(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def hybrid(self, query: str, k: int = TOP_K, pool: int = 30, rrf_k: int = 60) -> list[dict]:
        """RRF: each chunk gets sum(1 / (rrf_k + rank)) over the two lists.
        It only uses ranks, so we never compare a cosine score with a BM25 score."""
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for results in (self.dense_diverse(query, pool), self.lexical(query, pool)):
            for rank, chunk in enumerate(results, start=1):
                cid = chunk["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                by_id[cid] = chunk
        best = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [{**by_id[cid], "score": fused[cid]} for cid in best]

    def auto(self, query: str, k: int = TOP_K) -> list[dict]:
        """BM25 only helps when the question shares words with the French pages."""
        return getattr(self, resolve_method(query))(query, k)

    def close(self) -> None:
        self.client.close()

    def search(self, query: str, k: int = TOP_K, method: str = "auto") -> list[dict]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        with self._lock:
            return getattr(self, method)(query, k)

    def add_context(self, hits: list[dict]) -> list[dict]:
        return with_neighbours(hits, self.chunks, self._position)


def with_neighbours(hits: list[dict], chunks: list[dict], position: dict[str, int]) -> list[dict]:
    """Give each hit the chunk just before and just after it on the same page.

    Pages are cut into small chunks (median 690 characters), so the chunk that matches
    the question often names the case while the number sits in the next one. On the 33
    French test questions whose reference answer has a number, all of its numbers were in
    the 5 retrieved chunks for 25; with the neighbours, for 29 (10 chunks: 27).
    Measured by evaluation/context_coverage.py.
    A hit already shown as an earlier hit's neighbour is dropped, so no text is repeated.
    `chunks` is in page order and `position` maps a chunk_id to its index in it.
    """
    used: set[int] = set()
    out = []
    for hit in hits:
        i = position[hit["chunk_id"]]
        if i in used:
            continue
        window = [j for j in (i - 1, i, i + 1)
                  if 0 <= j < len(chunks) and chunks[j]["page_id"] == hit["page_id"] and j not in used]
        used.update(window)
        heading, _, body = chunks[i]["text"].partition("\n")
        parts = []
        for j in window:
            text = body if j == i else chunks[j]["text"].partition("\n")[2]
            if j != i and chunks[j]["section"] != hit["section"]:
                text = f"({chunks[j]['section']})\n{text}"  # a neighbour can be about another case
            parts.append(text)
        out.append({**hit, "text": heading + "\n" + "\n\n".join(parts),
                    "chunk_ids": [chunks[j]["chunk_id"] for j in window]})
    return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """One shared instance: Qdrant local mode allows one client per folder."""
    return Retriever()
