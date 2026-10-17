"""Ingest batching.

Three things are asserted here. That `upsert` never embeds or writes more than
`max_batch_size` documents in one go, at the store and through the route, which
is what issue #5 asked for. That a batch size below one is refused where the
setting is read. And that the event loop stays free while a batch is embedding,
which is the part of #5 that was wrong: `embed` already hands the model to a
worker thread, so a long ingest never stopped /health from answering.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_agent.api import main
from rag_agent.config import Settings
from rag_agent.graph.state import Document
from rag_agent.retrieval import embedder as emb
from rag_agent.retrieval import qdrant_store as qs


class FakeClient:
    """Records the size of every upsert so batching is observable."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    async def upsert(self, collection_name: str, points: list) -> None:
        self.batches.append(len(points))


def _docs(n: int) -> list[Document]:
    return [
        Document(id=str(uuid.uuid4()), text=f"Article {i} of the regulation.", source="crr", page=i)
        for i in range(n)
    ]


@pytest.fixture
def store(monkeypatch) -> qs.QdrantStore:
    embedded: list[int] = []

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        embedded.append(len(texts))
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(qs, "embed", fake_embed)
    store = qs.QdrantStore(client=FakeClient())
    store.embedded = embedded  # type: ignore[attr-defined]
    return store


async def test_upsert_chunks_at_max_batch_size(store):
    store.cfg = store.cfg.model_copy(update={"max_batch_size": 10})

    assert await store.upsert(_docs(25)) == 25
    # Three embed calls and three writes, not one of 25. The last batch is the
    # remainder, so a size that does not divide evenly must not be dropped.
    assert store.embedded == [10, 10, 5]
    assert store.client.batches == [10, 10, 5]


async def test_upsert_under_the_limit_stays_one_call(store):
    store.cfg = store.cfg.model_copy(update={"max_batch_size": 10})

    assert await store.upsert(_docs(3)) == 3
    assert store.embedded == [3]
    assert store.client.batches == [3]


async def test_upsert_of_nothing_writes_nothing(store):
    assert await store.upsert([]) == 0
    assert store.embedded == []
    assert store.client.batches == []


def test_a_batch_size_below_one_is_rejected():
    # range(0, n, 0) raises, so a zero would turn every ingest into a 500. The
    # floor has to be refused where the setting is read, not at the call site.
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(max_batch_size=-5)


async def test_the_event_loop_is_free_while_a_batch_embeds(monkeypatch):
    """Issue #5 claimed a large upsert runs the embedder on the event loop and
    the health check starts failing partway through. It does not. This fails if
    anyone takes the thread hop out of `embed`."""

    class SlowModel:
        def encode(self, texts, normalize_embeddings=False):
            time.sleep(0.2)
            return np.zeros((len(texts), 2))

    monkeypatch.setattr(emb, "_model", lambda: SlowModel())

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    beat = asyncio.create_task(heartbeat())
    await emb.embed([f"doc {i}" for i in range(500)])
    beat.cancel()

    # 0.2s of encoding against a 5ms heartbeat measures 37 ticks here, so the
    # floor leaves room for a slower machine. Dropping the to_thread hop and
    # calling _run() inline leaves this at 0.
    assert ticks > 10


def test_the_ingest_route_writes_in_batches(store, monkeypatch):
    # The route is what issue #5 named, so it is driven over HTTP rather than
    # by calling upsert again. Lifespan is skipped: TestClient only runs it as
    # a context manager, so the store is injected instead of being built.
    store.cfg = store.cfg.model_copy(update={"max_batch_size": 10})
    monkeypatch.setitem(main.state, "store", store)
    client = TestClient(main.app)

    body = {"documents": [d.model_dump() for d in _docs(25)]}
    response = client.post("/ingest", json=body)

    assert response.status_code == 200
    assert response.json() == {"upserted": 25}
    assert store.client.batches == [10, 10, 5]


def test_the_ingest_route_still_refuses_an_empty_list(store, monkeypatch):
    monkeypatch.setitem(main.state, "store", store)
    client = TestClient(main.app)

    response = client.post("/ingest", json={"documents": []})

    assert response.status_code == 400
    assert store.client.batches == []
