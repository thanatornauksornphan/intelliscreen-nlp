# src/similarity/batch_comparator.py

import pandas as pd
from src.extraction.extractor import extract_text
from src.preprocessing.preprocessor import TextPreprocessor
from src.similarity.vectorizer import TfidfSimilarityVectorizer
from src.similarity.similarity_scorer import SimilarityScorer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compare_students_to_master(
    student_file_paths: list[str], master_file_path: str
) -> pd.DataFrame:
    preprocessor = TextPreprocessor()
    vectorizer = TfidfSimilarityVectorizer()
    scorer = SimilarityScorer()

    # 1. Extract + preprocess master
    master_raw = extract_text(master_file_path)["text"]
    master_clean = preprocessor.preprocess_to_string(master_raw)

    # 2. Extract + preprocess all students
    student_texts = []
    student_filenames = []
    for path in student_file_paths:
        result = extract_text(path)
        if "error" in result["metadata"]:
            logger.warning(f"Skipping {path} due to extraction error")
            continue
        cleaned = preprocessor.preprocess_to_string(result["text"])
        student_texts.append(cleaned)
        student_filenames.append(path)

    # 3. Fit TF-IDF on the whole corpus (master + all students together)
    corpus = [master_clean] + student_texts
    tfidf_matrix = vectorizer.fit_transform(corpus)

    master_vector = tfidf_matrix[0]
    student_vectors = tfidf_matrix[1:]

    # 4. Score each student against master
    rows = []
    for i, filename in enumerate(student_filenames):
        score = scorer.compute_score(master_vector, student_vectors[i])
        level = scorer.match_level(score)
        rows.append(
            {
                "filename": filename,
                "similarity_score": round(score, 4),
                "match_level": level,
            }
        )
        logger.info(f"{filename}: score={score:.4f}, level={level}")

    return pd.DataFrame(rows)
