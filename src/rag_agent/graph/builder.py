"""Graph wiring.

    rewrite -> retrieve -> grade_documents -> {generate | rewrite(retry) | refuse}
    generate -> grade_grounding -> {END | rewrite(retry) | refuse}

Two separate gates. The first catches bad retrieval, the second catches a model
that answered confidently from thin context. They fail differently and a single
gate misses one of them.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_agent.config import get_settings
from rag_agent.graph.nodes import Nodes
from rag_agent.graph.state import AgentState


def _after_docs(state: AgentState) -> str:
    cfg = get_settings()
    if state["retrieval_grade"].score >= cfg.min_grounding_score:
        return "generate"
    if state.get("retries", 0) < cfg.max_retrieval_retries:
        return "retry"
    return "refuse"


def _after_grounding(state: AgentState) -> str:
    cfg = get_settings()
    if state["grounding_grade"].score >= cfg.min_grounding_score:
        return "done"
    if state.get("retries", 0) < cfg.max_retrieval_retries:
        return "retry"
    return "refuse"


def _bump(state: AgentState) -> AgentState:
    return {"retries": state.get("retries", 0) + 1}


def build_graph(nodes: Nodes):
    g = StateGraph(AgentState)

    g.add_node("rewrite", nodes.rewrite)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("generate", nodes.generate)
    g.add_node("grade_grounding", nodes.grade_grounding)
    g.add_node("refuse", nodes.refuse)
    g.add_node("bump_retries", _bump)

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges("grade_documents", _after_docs, {
        "generate": "generate",
        "retry": "bump_retries",
        "refuse": "refuse",
    })
    g.add_edge("generate", "grade_grounding")
    g.add_conditional_edges("grade_grounding", _after_grounding, {
        "done": END,
        "retry": "bump_retries",
        "refuse": "refuse",
    })
    g.add_edge("bump_retries", "rewrite")
    g.add_edge("refuse", END)

    return g.compile()
