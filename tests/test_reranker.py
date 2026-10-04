import pytest

from rag_agent.retrieval import reranker as rr
from tests.conftest import FakeCrossEncoder


@pytest.fixture(autouse=True)
def _clear_cache():
    rr._model.cache_clear()
    yield
    rr._model.cache_clear()


async def test_rerank_reorders_against_dense_order(docs, monkeypatch):
    # Dense put doc 1 first. The cross-encoder disagrees and must win.
    monkeypatch.setattr(rr, "_model", lambda: FakeCrossEncoder([0.1, 0.9, 0.4]))
    out = await rr.CrossEncoderReranker().rerank("leverage ratio", docs)
    assert [d.id for d in out] == ["2", "3", "1"]
    assert out[0].rerank_score == pytest.approx(0.9)


async def test_rerank_empty_is_not_an_error(monkeypatch):
    monkeypatch.setattr(rr, "_model", lambda: FakeCrossEncoder([]))
    assert await rr.CrossEncoderReranker().rerank("anything", []) == []


async def test_dense_score_is_preserved(docs, monkeypatch):
    monkeypatch.setattr(rr, "_model", lambda: FakeCrossEncoder([0.1, 0.9, 0.4]))
    out = await rr.CrossEncoderReranker().rerank("q", docs)
    assert {d.id: d.dense_score for d in out}["1"] == pytest.approx(0.81)
