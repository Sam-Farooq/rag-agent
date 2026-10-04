"""The state object threaded through every node in the graph."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    source: str
    page: int | None = None
    dense_score: float = 0.0
    rerank_score: float = 0.0


class Grade(BaseModel):
    """Structured output from the grader nodes. Kept small on purpose: a grader
    that writes prose is a grader you cannot threshold."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=280)


def keep_last(_: list, new: list) -> list:
    return new


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    documents: Annotated[list[Document], keep_last]
    answer: str
    citations: list[str]

    retrieval_grade: Grade
    grounding_grade: Grade
    retries: int
    route: Literal["answer", "retry", "refuse"]
    trace_id: str
