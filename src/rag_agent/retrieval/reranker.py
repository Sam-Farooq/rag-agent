"""Cross-encoder re-ranking.

Bi-encoder recall is cheap and imprecise: it scores the query and the passage
apart. A cross-encoder reads them together and is far better at ordering, which
is why the pipeline pulls 24 and keeps 5 rather than asking Qdrant for 5.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag_agent.config import get_settings
from rag_agent.graph.state import Document


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(get_settings().rerank_model)


class CrossEncoderReranker:
    async def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        def _run() -> list[float]:
            pairs = [(query, d.text) for d in docs]
            return _model().predict(pairs).tolist()

        scores = await asyncio.to_thread(_run)
        for doc, score in zip(docs, scores, strict=True):
            doc.rerank_score = float(score)
        return sorted(docs, key=lambda d: d.rerank_score, reverse=True)
