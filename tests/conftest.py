import pytest

from rag_agent.graph.state import Document


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(id="1", text="Institutions shall maintain a total capital ratio of 8%.",
                 source="crr-575-2013", page=112, dense_score=0.81),
        Document(id="2", text="The leverage ratio shall be at least 3%.",
                 source="crr-575-2013", page=240, dense_score=0.74),
        Document(id="3", text="Annex II sets out the mapping of exposure classes.",
                 source="crr-575-2013", page=501, dense_score=0.52),
    ]


class FakeCrossEncoder:
    """Scores by position so ordering is deterministic under test."""

    def __init__(self, scores: list[float]):
        self.scores = scores

    def predict(self, pairs):
        import numpy as np
        return np.array(self.scores[: len(pairs)])
