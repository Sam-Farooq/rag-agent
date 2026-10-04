"""Embeddings. One process-wide model, loaded lazily.

sentence-transformers holds the weights in memory, so constructing this per
request is the easiest way to turn a 40ms call into a 4s one.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag_agent.config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embed_model)


async def embed(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings keeps cosine and dot product interchangeable, which
    # matters because Qdrant is configured for cosine below.
    def _run() -> list[list[float]]:
        return _model().encode(texts, normalize_embeddings=True).tolist()

    return await asyncio.to_thread(_run)


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
