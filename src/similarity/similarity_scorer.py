# src/similarity/similarity_scorer.py

from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class SimilarityScorer:
    def __init__(self):
        config = load_config()
        self.thresholds = config["similarity"]["thresholds"]

    def compute_score(self, vector_a, vector_b) -> float:
        score = cosine_similarity(vector_a, vector_b)[0][0]
        return float(score)

    def match_level(self, score: float) -> str:
        if score >= self.thresholds["excellent"]:
            return "Excellent"
        elif score >= self.thresholds["good"]:
            return "Good"
        else:
            return "Poor"
