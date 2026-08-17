# src/visualization/report_generator.py

from pathlib import Path
import re
import pandas as pd
import spacy
from src.similarity.similarity_scorer import UnifiedScorer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load spaCy sentence segmenter once
try:
    nlp_sentencizer = spacy.blank("en")
    nlp_sentencizer.add_pipe("sentencizer")
except Exception:
    nlp_sentencizer = None


def split_into_sentences(text: str) -> list[str]:
    """Segment text into clean sentences using spaCy with regex fallback."""
    if not text or not text.strip():
        return []

    if nlp_sentencizer:
        doc = nlp_sentencizer(text.strip())
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    return sentences


def top_matching_sentences(
    student_text: str, master_text: str, vectorizer, top_n: int = None
) -> list[str]:
    """Find the top N student sentences semantically most relevant to the master text."""
    config = load_config()
    if top_n is None:
        top_n = config["reporting"].get("top_n_matching_sentences", 3)

    student_sentences = split_into_sentences(student_text)
    if not student_sentences:
        logger.warning("No valid sentences found in student text.")
        return []

    scorer = UnifiedScorer()
    corpus = [master_text] + student_sentences

    # Vectorize corpus using the active engine (TF-IDF sparse matrix or GPU tensor)
    vectors = vectorizer.transform(corpus)
    master_vec = vectors[0:1]

    scores = []
    for i, sentence in enumerate(student_sentences):
        sent_vec = vectors[i + 1 : i + 2]
        score = scorer.compute_score(master_vec, sent_vec)
        scores.append((sentence, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Computed similarity across {len(student_sentences)} sentences.")
    return [sentence for sentence, score in scores[:top_n]]


def export_report_csv(
    df: pd.DataFrame, filename: str = "similarity_report.csv"
) -> Path:
    config = load_config()
    output_dir = PROJECT_ROOT / config["reporting"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Exported similarity report to {path}")
    return path
