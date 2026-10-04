"""MCP server.

Exposes retrieval as two tools so an external client (Claude Desktop, an IDE,
another agent) can search the corpus without going through the HTTP API. The
graph is deliberately not exposed: a caller that wants the self-correction loop
should call /ask, and a caller that wants raw passages should not pay for it.
"""
from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from rag_agent.config import get_settings
from rag_agent.retrieval.qdrant_store import QdrantStore
from rag_agent.retrieval.reranker import CrossEncoderReranker

server = Server("rag-agent")
_store = QdrantStore()
_reranker = CrossEncoderReranker()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_regulations",
            description=(
                "Dense search over the indexed regulatory corpus, cross-encoder "
                "re-ranked. Returns passages with source and page."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_collection_info",
            description="Point count and vector config of the active collection.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    cfg = get_settings()

    if name == "search_regulations":
        k = int(arguments.get("k", 5))
        candidates = await _store.search(arguments["query"], limit=cfg.candidate_k)
        ranked = await _reranker.rerank(arguments["query"], candidates)
        body = "\n\n".join(
            f"[{d.source}:{d.page}] (score {d.rerank_score:.3f})\n{d.text}"
            for d in ranked[:k]
        )
        return [TextContent(type="text", text=body or "No matching passages.")]

    if name == "get_collection_info":
        info = await _store.client.get_collection(cfg.collection)
        return [TextContent(type="text", text=f"points={info.points_count}")]

    raise ValueError(f"unknown tool: {name}")


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
