# src/similarity/question_comparator.py

"""Functions for comparing two texts (master vs. student) on a per-question basis,
if question-wise comparison is enabled in the config.yaml."""

from pathlib import Path

import pandas as pd

from src.extraction.extractor import extract_text
from src.preprocessing.preprocessor import TextPreprocessor
from src.similarity.question_splitter import split_by_question
from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sort_key(question_id: str):
    """Return a numeric sort key for question IDs like "Q1", "Q2", etc."""

    digits = question_id[1:]
    return int(digits) if digits.isdigit() else question_id


def compare_students_to_master_by_question(
    student_file_paths: list[str], master_file_path: str
) -> pd.DataFrame:
    """Compare each student file to the master answer key on a per-question basis, returning a DataFrame of results."""

    preprocessor = TextPreprocessor()
    vectorizer = UnifiedVectorizer()
    scorer = UnifiedScorer()

    master_raw = extract_text(master_file_path)["text"]
    master_questions = split_by_question(master_raw)

    if not master_questions:
        logger.error(
            "No question markers found in the master answer key — "
            "question-wise comparison requires a structured master key "
            "(e.g. 'Q1:', 'Q2:', ...)."
        )
        return pd.DataFrame()

    rows = []
    for path in student_file_paths:
        result = extract_text(path)
        if "error" in result["metadata"]:
            logger.warning(f"Skipping {path} due to extraction error")
            continue

        student_raw = result["text"]
        student_questions = split_by_question(student_raw)

        if not student_questions:
            logger.warning(
                f"{path}: no question markers detected, falling back to whole-document comparison"
            )
            master_clean = preprocessor.preprocess_to_string(master_raw)
            student_clean = preprocessor.preprocess_to_string(student_raw)
            vectors = vectorizer.transform([master_clean, student_clean])
            score = scorer.compute_score(vectors[0:1], vectors[1:2])
            rows.append(
                {
                    "filename": path,
                    "question_id": "OVERALL",
                    "similarity_score": round(score, 4),
                    "match_level": scorer.match_level(score),
                }
            )
            continue

        shared_ids = set(master_questions) & set(student_questions)
        missing_ids = set(master_questions) - set(student_questions)
        if missing_ids:
            logger.warning(
                f"{path}: no answer detected for {sorted(missing_ids, key=_sort_key)}"
            )

        for qid in sorted(shared_ids, key=_sort_key):
            master_clean = preprocessor.preprocess_to_string(master_questions[qid])
            student_clean = preprocessor.preprocess_to_string(student_questions[qid])
            vectors = vectorizer.transform([master_clean, student_clean])
            score = scorer.compute_score(vectors[0:1], vectors[1:2])
            rows.append(
                {
                    "filename": path,
                    "question_id": qid,
                    "similarity_score": round(score, 4),
                    "match_level": scorer.match_level(score),
                }
            )
            logger.info(
                f"{path} [{qid}]: score={score:.4f}, level={scorer.match_level(score)}"
            )

        for qid in sorted(missing_ids, key=_sort_key):
            rows.append(
                {
                    "filename": path,
                    "question_id": qid,
                    "similarity_score": 0.0,
                    "match_level": "Poor",
                }
            )

    return pd.DataFrame(rows)


def to_score_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of question-wise similarity scores into a pivoted matrix format."""

    if df.empty:
        return df
    pivot_df = df.copy()
    pivot_df["display_name"] = pivot_df["filename"].apply(lambda p: Path(p).name)
    return pivot_df.pivot(
        index="display_name", columns="question_id", values="similarity_score"
    )
