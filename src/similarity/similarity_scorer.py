# src/similarity/similarity_scorer.py

import requests
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class UnifiedScorer:
    def __init__(self):
        self.config = load_config()
        self.method = self.config["similarity"].get("method", "semantic")
        self.thresholds = self.config["similarity"]["thresholds"]
        self.llm_config = self.config.get("llm_reasoning", {})

    def compute_score(self, vector_a, vector_b) -> float:
        if self.method == "semantic":
            score = util.cos_sim(vector_a, vector_b)[0][0].item()
        else:
            score = cosine_similarity(vector_a, vector_b)[0][0]
        return float(score)

    def match_level(self, score: float) -> str:
        if score >= self.thresholds["excellent"]:
            return "Excellent"
        elif score >= self.thresholds["good"]:
            return "Good"
        else:
            return "Poor"

    def generate_feedback(
        self, master_text: str, student_text: str, score: float
    ) -> str:
        if not self.llm_config.get("enabled", False):
            return "LLM reasoning disabled."

        model = self.llm_config.get("local_model", "llama3")
        prompt = (
            f"You are an AI teaching assistant. The student scored {score:.2f}.\n\n"
            f"Master Key: {master_text}\n"
            f"Student Answer: {student_text}\n\n"
            f"In 2 short sentences, explain exactly what key concepts the student missed."
        )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "No feedback generated.")
        except Exception as e:
            logger.error(f"Failed to reach local LLM: {e}")
            return "Error generating LLM feedback."
