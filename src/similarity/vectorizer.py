# src/similarity/vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class TfidfSimilarityVectorizer:
    def __init__(self):
        config = load_config()
        sim_config = config["similarity"]
        ngram_range = tuple(sim_config.get("tfidf_ngram_range", [1, 1]))

        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        logger.info(f"TF-IDF vectorizer initialized with ngram_range={ngram_range}")

    def fit_transform(self, documents: list[str]):
        logger.debug(f"Fitting TF-IDF on {len(documents)} documents")
        return self.vectorizer.fit_transform(documents)
