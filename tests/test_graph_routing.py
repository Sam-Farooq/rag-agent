"""Routing tests.

The graph's value is entirely in where it branches, so the branch predicates
are tested directly rather than through a mocked LLM. A test that stubs the
model and asserts the stub was called proves nothing about the routing.
"""
from rag_agent.graph.builder import _after_docs, _after_grounding
from rag_agent.graph.state import Grade


def test_good_retrieval_generates():
    state = {"retrieval_grade": Grade(score=0.9, reason="ok"), "retries": 0}
    assert _after_docs(state) == "generate"


def test_weak_retrieval_retries_while_budget_remains():
    state = {"retrieval_grade": Grade(score=0.2, reason="thin"), "retries": 0}
    assert _after_docs(state) == "retry"


def test_weak_retrieval_refuses_once_budget_is_spent():
    state = {"retrieval_grade": Grade(score=0.2, reason="thin"), "retries": 2}
    assert _after_docs(state) == "refuse"


def test_ungrounded_answer_does_not_reach_the_user():
    state = {"grounding_grade": Grade(score=0.1, reason="invented"), "retries": 2}
    assert _after_grounding(state) == "refuse"


def test_grounded_answer_ends():
    state = {"grounding_grade": Grade(score=0.95, reason="supported"), "retries": 1}
    assert _after_grounding(state) == "done"


def test_threshold_is_inclusive():
    # 0.6 is the configured floor. Exactly 0.6 must pass, or the gate is
    # off by one against every score the grader rounds to the boundary.
    state = {"retrieval_grade": Grade(score=0.6, reason="borderline"), "retries": 0}
    assert _after_docs(state) == "generate"
