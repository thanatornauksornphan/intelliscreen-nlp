# src/similarity/question_comparator.py

"""Functions for comparing two texts (master vs. student) on a per-question basis,
if question-wise comparison is enabled in the config.yaml."""

import pandas as pd

from src.extraction.extractor import extract_text
from src.preprocessing.preprocessor import TextPreprocessor
from src.similarity.question_splitter import split_by_question
from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sort_key(question_id: str):
    """Sort question IDs in natural order (Q1, Q2, Q10, etc.)"""

    digits = question_id[1:]
    return int(digits) if digits.isdigit() else question_id


def _score_fallback_whole_document(
    path: str, raw_text: str, master_raw: str, preprocessor, vectorizer, scorer
) -> dict:
    """Score a student submission that has no detectable question markers at all
    against the master answer key as a whole-document comparison. Returns a dict."""

    master_clean = preprocessor.preprocess_to_string(master_raw)
    student_clean = preprocessor.preprocess_to_string(raw_text)
    vectors = vectorizer.transform([master_clean, student_clean])
    score = scorer.compute_score(vectors[0:1], vectors[1:2])
    return {
        "filename": path,
        "question_id": "OVERALL",
        "similarity_score": round(score, 4),
        "match_level": scorer.match_level(score),
    }


def compare_students_to_master_by_question(
    student_file_paths: list[str], master_file_path: str
) -> pd.DataFrame:
    """Compare each student submission to the master answer key on a per-question basis.
    Returns a long-format DataFrame with columns: filename, question_id, similarity_score, match_level. Students with no detectable question markers at all are scored once under "OVERALL" instead.
    """

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

    # Pass 1: extract + split every student once, up front. Students with no
    # detectable markers at all are set aside for whole-document fallback.
    student_questions_by_path: dict[str, dict[str, str]] = {}
    fallback_students: list[tuple[str, str]] = []

    for path in student_file_paths:
        result = extract_text(path)
        if "error" in result["metadata"]:
            logger.warning(f"Skipping {path} due to extraction error")
            continue

        raw = result["text"]
        questions = split_by_question(raw)
        if not questions:
            fallback_students.append((path, raw))
        else:
            student_questions_by_path[path] = questions

    rows = []

    # Handle whole-document fallback students (rare path, not worth batching)
    for path, raw in fallback_students:
        logger.warning(
            f"{path}: no question markers detected, falling back to whole-document comparison"
        )
        rows.append(
            _score_fallback_whole_document(
                path, raw, master_raw, preprocessor, vectorizer, scorer
            )
        )

    # Pass 2: for each question, batch every student who answered it into one
    # vectorizer call, instead of one call per (student, question) pair.
    for qid in sorted(master_questions.keys(), key=_sort_key):
        master_clean = preprocessor.preprocess_to_string(master_questions[qid])

        answering_paths = [
            p for p, qs in student_questions_by_path.items() if qid in qs
        ]
        missing_paths = [
            p for p, qs in student_questions_by_path.items() if qid not in qs
        ]

        if answering_paths:
            student_clean_texts = [
                preprocessor.preprocess_to_string(student_questions_by_path[p][qid])
                for p in answering_paths
            ]
            corpus = [master_clean] + student_clean_texts
            vectors = vectorizer.transform(corpus)
            master_vec = vectors[0:1]

            for i, path in enumerate(answering_paths):
                student_vec = vectors[i + 1 : i + 2]
                score = scorer.compute_score(master_vec, student_vec)
                level = scorer.match_level(score)
                rows.append(
                    {
                        "filename": path,
                        "question_id": qid,
                        "similarity_score": round(score, 4),
                        "match_level": level,
                    }
                )
                logger.info(f"{path} [{qid}]: score={score:.4f}, level={level}")

        for path in missing_paths:
            logger.warning(f"{path}: no answer detected for {qid}")
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
    """Convert a long-format DataFrame of question-wise scores into a wide-format
    matrix with one row per student and one column per question. Students with no detectable question markers at all will have a single column "OVERALL" instead.
    """

    if df.empty:
        return df
    from pathlib import Path

    pivot_df = df.copy()
    pivot_df["display_name"] = pivot_df["filename"].apply(lambda p: Path(p).name)
    return pivot_df.pivot(
        index="display_name", columns="question_id", values="similarity_score"
    )
