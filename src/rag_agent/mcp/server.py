"""MCP server.

Exposes retrieval as two tools so an external client (Claude Desktop, an IDE,
another agent) can query the corpus without going through the HTTP API.

The graph is deliberately not exposed. A caller that wants the self-correction
loop should call /ask and pay for it. A caller that wants raw passages to
reason over itself should not have to.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from rag_agent.config import get_settings
from rag_agent.retrieval.qdrant_store import QdrantStore
from rag_agent.retrieval.reranker import CrossEncoderReranker

server = MCPServer(name="rag-agent")
_store = QdrantStore()
_reranker = CrossEncoderReranker()

MAX_RESULTS = 20


@server.tool()
async def search_regulations(query: str, k: int = 5) -> str:
    """Search the indexed regulatory corpus.

    Dense retrieval followed by cross-encoder re-ranking. Returns passages with
    their source and page.

    Args:
        query: What to search for.
        k: How many passages to return, 1 to 20.
    """
    cfg = get_settings()
    k = max(1, min(k, MAX_RESULTS))
    candidates = await _store.search(query, limit=cfg.candidate_k)
    ranked = await _reranker.rerank(query, candidates)
    body = "\n\n".join(
        f"[{d.source}:{d.page}] (score {d.rerank_score:.3f})\n{d.text}"
        for d in ranked[:k]
    )
    return body or "No matching passages."


@server.tool()
async def get_collection_info() -> str:
    """Report how many points are indexed in the active collection."""
    cfg = get_settings()
    info = await _store.client.get_collection(cfg.collection)
    return f"collection={cfg.collection} points={info.points_count}"


if __name__ == "__main__":
    server.run(transport="stdio")
