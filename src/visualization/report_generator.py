# src/visualization/report_generator.py

import re
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s) > 10]  # filter noise fragments


def top_matching_sentences(
    student_text: str, master_text: str, vectorizer, top_n: int = None
) -> list[str]:
    """Find the N student sentences most similar to the master text.

    If top_n is not explicitly provided, falls back to the value defined
    in config.yaml under reporting.top_n_matching_sentences.
    """
    if top_n is None:
        from src.utils.config_loader import load_config

        config = load_config()
        top_n = config["reporting"]["top_n_matching_sentences"]

    student_sentences = split_into_sentences(student_text)
    if not student_sentences:
        logger.warning(
            "No valid sentences found in student text; skipping top-matching-sentence extraction"
        )
        return []

    corpus = [master_text] + student_sentences
    matrix = vectorizer.fit_transform(corpus)
    master_vec = matrix[0]

    scores = [
        (sentence, cosine_similarity(master_vec, matrix[i + 1])[0][0])
        for i, sentence in enumerate(student_sentences)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"Compared {len(student_sentences)} sentences, returning top {top_n} matches"
    )
    return [sentence for sentence, score in scores[:top_n]]


def export_report_csv(
    df: pd.DataFrame, filename: str = "similarity_report.csv"
) -> None:
    from src.utils.config_loader import load_config

    config = load_config()
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    logger.info(f"Exported report to {path}")
