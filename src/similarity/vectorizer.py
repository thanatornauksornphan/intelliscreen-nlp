# src/similarity/vectorizer.py

import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class UnifiedVectorizer:
    def __init__(self):
        config = load_config()
        self.method = config["similarity"].get("method", "semantic")

        if self.method == "semantic":
            model_name = config["similarity"]["semantic_model"]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Initialized Semantic Vectorizer with '{model_name}' on {self.device.upper()}"
            )
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            ngram_range = tuple(config["similarity"].get("tfidf_ngram_range", [1, 2]))
            logger.info(f"Initialized TF-IDF Vectorizer with ngram_range={ngram_range}")
            self.model = TfidfVectorizer(ngram_range=ngram_range)

    def transform(self, documents: list[str]):
        if self.method == "semantic":
            return self.model.encode(documents, convert_to_tensor=True)
        else:
            return self.model.fit_transform(documents)
