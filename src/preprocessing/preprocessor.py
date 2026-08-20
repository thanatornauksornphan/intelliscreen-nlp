# src/preprocessing/preprocessor.py

from pathlib import Path

import spacy

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TextPreprocessor:
    def __init__(self):
        config = load_config()
        prep_config = config["preprocessing"]
        self.method = config["similarity"].get("method", "semantic")

        logger.info(f"Loading spaCy model: {prep_config['spacy_model']}")
        self.nlp = spacy.load(prep_config["spacy_model"])

        self.min_token_length = prep_config["min_token_length"]
        self.use_lemmatization = prep_config["use_lemmatization"]
        self.preserve_numbers = prep_config["preserve_numbers"]

        self.stopwords = self.nlp.Defaults.stop_words.union(
            set(prep_config["custom_stopwords"])
        )
        self.protected_terms = set(prep_config.get("protected_terms", []))

    def preprocess(self, text: str) -> list[str]:
        if not text or not text.strip():
            logger.warning("Empty text passed to preprocess()")
            return []

        # If using Semantic Embeddings, bypass heavy NLP filtering
        if self.method == "semantic":
            doc = self.nlp(text.strip())
            return [token.text for token in doc if not token.is_space]

        # TF-IDF Pipeline: Tokenization, Stopword Removal, Lemmatization
        doc = self.nlp(text.lower())
        tokens = []

        for token in doc:
            if token.text in self.protected_terms:
                tokens.append(token.text)
                continue
            if token.text in self.stopwords:
                continue
            if token.is_punct:
                continue
            if token.is_space:
                continue
            if token.like_num and not self.preserve_numbers:
                continue
            if len(token.text) < self.min_token_length and not token.like_num:
                continue

            processed = token.lemma_ if self.use_lemmatization else token.text
            tokens.append(processed)

        logger.debug(
            f"Preprocessed text: {len(tokens)} tokens retained from {len(doc)} original tokens"
        )
        return tokens

    def preprocess_to_string(self, text: str) -> str:
        if self.method == "semantic":
            # Return cleaned, natural text with full sentence grammar intact
            return " ".join(text.split())
        return " ".join(self.preprocess(text))

    def preprocess_for_display(self, text: str) -> str:
        """Always applies full stopword/lemmatization filtering, regardless of
        the active similarity.method. Intended for human-facing outputs like
        word clouds — semantic mode intentionally skips filtering for
        embedding quality, but that's the wrong choice when the goal is a
        readable visual summary, not a vector."""
        if not text or not text.strip():
            return ""

        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if token.text in self.protected_terms:
                tokens.append(token.text)
                continue
            if token.text in self.stopwords:
                continue
            if token.is_punct or token.is_space:
                continue
            if token.like_num and not self.preserve_numbers:
                continue
            if len(token.text) < self.min_token_length and not token.like_num:
                continue
            tokens.append(token.lemma_ if self.use_lemmatization else token.text)
        return " ".join(tokens)

    def generate_ngrams(self, tokens: list[str], n: int = 2) -> list[str]:
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def save_preprocessed(
        self,
        tokens: list[str],
        original_filename: str,
        output_dir: str | None = None,
    ) -> None:
        output_dir = output_dir or PROJECT_ROOT / "data" / "processed"
        filename = Path(original_filename).stem + "_preprocessed.txt"
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(" ".join(tokens), encoding="utf-8")
        logger.debug(f"Saved preprocessed tokens to {output_path}")
