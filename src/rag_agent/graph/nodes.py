"""Graph nodes.

Each node takes the state and returns only the keys it changed. Keeping the
returns narrow is what makes the LangGraph reducer predictable when branches
rejoin after a retry.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from rag_agent.config import get_settings
from rag_agent.graph.state import AgentState, Document, Grade
from rag_agent.retrieval.qdrant_store import QdrantStore
from rag_agent.retrieval.reranker import CrossEncoderReranker

log = logging.getLogger(__name__)

REWRITE = ChatPromptTemplate.from_messages([
    ("system",
     ("Rewrite the user question as a standalone search query for a regulatory "
      "corpus. Keep every proper noun, statute number and date. Return the query "
      "and nothing else.")),
    ("human", "{question}"),
])

GRADE_DOCS = ChatPromptTemplate.from_messages([
    ("system",
     ("Score 0 to 1 how well these passages let you answer the question. "
      "0.0 means nothing relevant, 1.0 means fully answerable. Be strict: a "
      "passage about the right statute but the wrong subsection is not relevant.")),
    ("human", "Question: {question}\n\nPassages:\n{context}"),
])

ANSWER = ChatPromptTemplate.from_messages([
    ("system",
     ("Answer using only the passages. Cite every claim as [source:page]. If the "
      "passages do not contain the answer, say so plainly rather than reasoning "
      "from general knowledge.")),
    ("human", "Question: {question}\n\nPassages:\n{context}"),
])

GROUNDING = ChatPromptTemplate.from_messages([
    ("system",
     ("Score 0 to 1 how fully the answer is supported by the passages. Any "
      "sentence asserting a fact absent from the passages drives the score down.")),
    ("human", "Passages:\n{context}\n\nAnswer:\n{answer}"),
])


def _fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.source}:{d.page}] {d.text}" for d in docs)


class Nodes:
    def __init__(self, llm, store: QdrantStore, reranker: CrossEncoderReranker):
        self.llm = llm
        self.store = store
        self.reranker = reranker
        self.cfg = get_settings()

    async def rewrite(self, state: AgentState) -> AgentState:
        # On the first pass the raw question is usually fine. Rewriting only
        # earns its latency once retrieval has already come back weak.
        if state.get("retries", 0) == 0:
            return {"rewritten_question": state["question"]}
        out = await (REWRITE | self.llm).ainvoke({"question": state["question"]})
        return {"rewritten_question": out.content.strip()}

    async def retrieve(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_question") or state["question"]
        candidates = await self.store.search(query, limit=self.cfg.candidate_k)
        ranked = await self.reranker.rerank(query, candidates)
        kept = [d for d in ranked if d.rerank_score >= self.cfg.min_rerank_score]
        log.info("retrieved=%d kept=%d", len(candidates), len(kept))
        return {"documents": kept[: self.cfg.final_k]}

    async def grade_documents(self, state: AgentState) -> AgentState:
        docs = state.get("documents") or []
        if not docs:
            return {"retrieval_grade": Grade(score=0.0, reason="no documents above threshold")}
        grader = GRADE_DOCS | self.llm.with_structured_output(Grade)
        grade = await grader.ainvoke(
            {"question": state["question"], "context": _fmt(docs)}
        )
        return {"retrieval_grade": grade}

    async def generate(self, state: AgentState) -> AgentState:
        docs = state["documents"]
        out = await (ANSWER | self.llm).ainvoke(
            {"question": state["question"], "context": _fmt(docs)}
        )
        return {
            "answer": out.content,
            "citations": [f"{d.source}:{d.page}" for d in docs],
        }

    async def grade_grounding(self, state: AgentState) -> AgentState:
        grader = GROUNDING | self.llm.with_structured_output(Grade)
        grade = await grader.ainvoke(
            {"context": _fmt(state["documents"]), "answer": state["answer"]}
        )
        return {"grounding_grade": grade}

    async def refuse(self, state: AgentState) -> AgentState:
        return {
            "answer": (
                "The indexed documents do not cover this. Rather than answer from "
                "general knowledge, here is what was searched: "
                f"{state.get('rewritten_question') or state['question']}"
            ),
            "citations": [],
        }
