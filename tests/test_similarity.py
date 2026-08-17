# tests/test_similarity.py

from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer


def test_identical_text_scores_near_one():
    vectorizer = UnifiedVectorizer()
    scorer = UnifiedScorer()

    vectors = vectorizer.transform(
        ["this is a test sentence", "this is a test sentence"]
    )
    score = scorer.compute_score(vectors[0:1], vectors[1:2])

    assert score > 0.95


def test_unrelated_text_scores_low():
    vectorizer = UnifiedVectorizer()
    scorer = UnifiedScorer()

    vectors = vectorizer.transform(
        [
            "The cell membrane controls what enters and exits the cell.",
            "Quantum computing relies on qubits and quantum superposition.",
        ]
    )
    score = scorer.compute_score(vectors[0:1], vectors[1:2])

    assert score < 0.60


def test_match_level_thresholds():
    scorer = UnifiedScorer()
    assert scorer.match_level(0.90) == "Excellent"
    assert scorer.match_level(0.70) == "Good"
    assert scorer.match_level(0.20) == "Poor"
