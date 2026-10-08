from evals.metrics import CaseResult, citation_precision, refusal_accuracy, summarise


def _case(**kw):
    base = {"id": "x", "answered": True, "expected_refusal": False,
            "citation_hit": True, "contains_ok": True, "forbidden_ok": True,
            "retrieval_score": 0.8, "grounding_score": 0.9, "latency_ms": 100.0}
    return CaseResult(**{**base, **kw})


def test_refusal_accuracy_penalises_answering_an_out_of_corpus_question():
    results = [_case(expected_refusal=True, answered=True)]
    assert refusal_accuracy(results) == 0.0


def test_refusal_accuracy_rewards_a_correct_refusal():
    results = [_case(expected_refusal=True, answered=False)]
    assert refusal_accuracy(results) == 1.0


def test_citation_precision_ignores_refusal_cases():
    results = [_case(expected_refusal=True, answered=False, citation_hit=False), _case()]
    assert citation_precision(results) == 1.0


def test_summarise_reports_every_gate():
    report = summarise([_case()])
    assert {"refusal_accuracy", "citation_precision", "mean_grounding"} <= report.keys()
