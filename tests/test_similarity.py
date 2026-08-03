# tests/test_similarity.py

from src.similarity.vectorizer import TfidfSimilarityVectorizer
from src.similarity.similarity_scorer import SimilarityScorer


def test_identical_text_scores_near_one():
    vectorizer = TfidfSimilarityVectorizer()
    scorer = SimilarityScorer()

    matrix = vectorizer.fit_transform(["this is a test", "this is a test"])
    score = scorer.compute_score(matrix[0], matrix[1])

    assert score > 0.99


def test_match_level_thresholds():
    scorer = SimilarityScorer()
    assert scorer.match_level(0.9) == "Excellent"
    assert scorer.match_level(0.6) == "Good"
    assert scorer.match_level(0.1) == "Poor"
