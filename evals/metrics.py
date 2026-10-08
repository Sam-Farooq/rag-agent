"""Eval metrics.

Deliberately not a single aggregate score. Retrieval and grounding fail for
different reasons and averaging them hides which one regressed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseResult:
    id: str
    answered: bool
    expected_refusal: bool
    citation_hit: bool
    contains_ok: bool
    forbidden_ok: bool
    retrieval_score: float
    grounding_score: float
    latency_ms: float


def refusal_accuracy(results: list[CaseResult]) -> float:
    """Did it refuse exactly when it should have?

    The metric that matters most here. A RAG system that answers out-of-corpus
    questions from parametric memory is worse than useless in a regulatory
    setting, and no retrieval metric catches it.
    """
    relevant = [r for r in results if r.expected_refusal or not r.answered]
    if not relevant:
        return 1.0
    correct = sum(1 for r in relevant if r.expected_refusal != r.answered)
    return correct / len(relevant)


def citation_precision(results: list[CaseResult]) -> float:
    answerable = [r for r in results if not r.expected_refusal]
    if not answerable:
        return 1.0
    return sum(1 for r in answerable if r.citation_hit) / len(answerable)


def content_recall(results: list[CaseResult]) -> float:
    answerable = [r for r in results if not r.expected_refusal]
    if not answerable:
        return 1.0
    return sum(1 for r in answerable if r.contains_ok) / len(answerable)


def mean_grounding(results: list[CaseResult]) -> float:
    scored = [r.grounding_score for r in results if r.grounding_score > 0]
    return sum(scored) / len(scored) if scored else 0.0


def slowest_latency(results: list[CaseResult]) -> float:
    """The slowest case, not a percentile.

    The held-out set is seven cases, and a p95 over seven samples is the
    maximum with extra arithmetic in front of it. Naming it p95 would claim a
    tail estimate the sample size cannot support.
    """
    return max((r.latency_ms for r in results), default=0.0)


def summarise(results: list[CaseResult]) -> dict[str, float]:
    return {
        "refusal_accuracy": round(refusal_accuracy(results), 3),
        "citation_precision": round(citation_precision(results), 3),
        "content_recall": round(content_recall(results), 3),
        "mean_grounding": round(mean_grounding(results), 3),
        "slowest_ms": round(slowest_latency(results), 1),
        "n": len(results),
    }
