"""Qdrant wrapper. Dense search only; the sparse half lives in hybrid-search."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from rag_agent.config import get_settings
from rag_agent.graph.state import Document
from rag_agent.retrieval.embedder import embed, embed_one


class QdrantStore:
    def __init__(self, client: AsyncQdrantClient | None = None):
        self.cfg = get_settings()
        self.client = client or AsyncQdrantClient(url=self.cfg.qdrant_url)

    async def ensure_collection(self, dim: int = 768) -> None:
        existing = {c.name for c in (await self.client.get_collections()).collections}
        if self.cfg.collection in existing:
            return
        await self.client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            # Regulatory text is read far more than it is written, so pay the
            # indexing cost up front rather than on the first query.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    async def upsert(self, docs: list[Document]) -> int:
        vectors = await embed([d.text for d in docs])
        await self.client.upsert(
            collection_name=self.cfg.collection,
            points=[
                models.PointStruct(
                    id=d.id,
                    vector=v,
                    payload={"text": d.text, "source": d.source, "page": d.page},
                )
                for d, v in zip(docs, vectors, strict=True)
            ],
        )
        return len(docs)

    async def search(self, query: str, limit: int) -> list[Document]:
        vector = await embed_one(query)
        hits = await self.client.query_points(
            collection_name=self.cfg.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            Document(
                id=str(h.id),
                text=h.payload["text"],
                source=h.payload["source"],
                page=h.payload.get("page"),
                dense_score=h.score,
            )
            for h in hits.points
        ]
