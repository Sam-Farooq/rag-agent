"""HTTP surface. Three routes: ask, ingest, health."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from rag_agent.cache import AnswerCache
from rag_agent.config import get_settings
from rag_agent.graph.builder import build_graph
from rag_agent.graph.nodes import Nodes
from rag_agent.graph.state import Document
from rag_agent.observability.langfuse_tracer import Tracer
from rag_agent.retrieval.qdrant_store import QdrantStore
from rag_agent.retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = get_settings()
    store = QdrantStore()
    await store.ensure_collection()
    llm = ChatAnthropic(
        model=cfg.primary_model, temperature=cfg.temperature, max_tokens=cfg.max_tokens
    )
    state["store"] = store
    state["graph"] = build_graph(Nodes(llm, store, CrossEncoderReranker()))
    state["cache"] = AnswerCache()
    state["tracer"] = Tracer()
    log.info("ready: model=%s collection=%s", cfg.primary_model, cfg.collection)
    yield


app = FastAPI(title="rag-agent", version="0.4.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    retries: int
    retrieval_score: float | None = None
    grounding_score: float | None = None
    cached: bool = False
    trace_id: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    cache, tracer = state["cache"], state["tracer"]

    if req.use_cache and (hit := await cache.get(req.question)):
        return AskResponse(**hit, cached=True)

    trace_id = str(uuid.uuid4())
    with tracer.trace("ask", question=req.question, trace_id=trace_id) as span:
        result = await state["graph"].ainvoke(
            {"question": req.question, "retries": 0, "trace_id": trace_id}
        )
        span.update(output={"answer": result.get("answer", "")[:500]})

    retrieval = result.get("retrieval_grade")
    grounding = result.get("grounding_grade")
    if retrieval:
        tracer.score(trace_id, "retrieval", retrieval.score, retrieval.reason)
    if grounding:
        tracer.score(trace_id, "grounding", grounding.score, grounding.reason)

    payload = {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "retries": result.get("retries", 0),
        "retrieval_score": retrieval.score if retrieval else None,
        "grounding_score": grounding.score if grounding else None,
        "trace_id": trace_id,
    }
    # Only cache answers that cleared the grounding gate. Caching a refusal
    # means a transient Qdrant outage poisons that question for an hour.
    if grounding and grounding.score >= get_settings().min_grounding_score:
        await cache.set(req.question, payload)
    return AskResponse(**payload, cached=False)


class IngestRequest(BaseModel):
    documents: list[Document]


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    if not req.documents:
        raise HTTPException(status_code=400, detail="no documents")
    # TODO: chunk this. A 10k-document upsert blocks the loop for ~40s and the
    # health check starts failing halfway through.
    return {"upserted": await state["store"].upsert(req.documents)}
