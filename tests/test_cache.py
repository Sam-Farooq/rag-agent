from rag_agent.cache import _key


def test_key_is_whitespace_and_case_insensitive():
    assert _key("What is the  capital RATIO?") == _key("what is the capital ratio?")


def test_different_questions_do_not_collide():
    assert _key("capital ratio") != _key("leverage ratio")


def test_key_is_prefixed_and_bounded():
    k = _key("anything")
    assert k.startswith("rag:answer:") and len(k) == len("rag:answer:") + 32
