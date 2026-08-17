# src/similarity/batch_comparator.py

import pandas as pd
from src.extraction.extractor import extract_text
from src.preprocessing.preprocessor import TextPreprocessor
from src.similarity.vectorizer import UnifiedVectorizer
from src.similarity.similarity_scorer import UnifiedScorer
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def compare_students_to_master(
    student_file_paths: list[str], master_file_path: str
) -> pd.DataFrame:
    config = load_config()
    preprocessor = TextPreprocessor()
    vectorizer = UnifiedVectorizer()
    scorer = UnifiedScorer()

    method = config["similarity"].get("method", "semantic")
    llm_enabled = config.get("llm_reasoning", {}).get("enabled", False)

    # 1. Extract and preprocess master answer key
    master_result = extract_text(master_file_path)
    master_raw = master_result["text"]
    master_clean = preprocessor.preprocess_to_string(master_raw)

    # 2. Extract and preprocess student submissions
    student_clean_texts = []
    student_raw_texts = []
    student_filenames = []

    for path in student_file_paths:
        result = extract_text(path)
        if "error" in result["metadata"]:
            logger.warning(f"Skipping {path} due to extraction error")
            continue

        student_raw = result["text"]
        student_clean = preprocessor.preprocess_to_string(student_raw)

        student_raw_texts.append(student_raw)
        student_clean_texts.append(student_clean)
        student_filenames.append(path)

    if not student_filenames:
        logger.error("No valid student files to process.")
        return pd.DataFrame()

    # 3. Vectorization (Dual-Engine Handling)
    corpus = [master_clean] + student_clean_texts
    vectors = vectorizer.transform(corpus)

    master_vector = vectors[0:1]
    student_vectors = vectors[1:]

    # 4. Score each student and generate feedback
    rows = []
    for i, filename in enumerate(student_filenames):
        student_vec = student_vectors[i : i + 1]
        score = scorer.compute_score(master_vector, student_vec)
        level = scorer.match_level(score)

        row_data = {
            "filename": filename,
            "similarity_score": round(score, 4),
            "match_level": level,
            "method_used": method,
        }

        # Optional LLM Reasoning
        if llm_enabled:
            feedback = scorer.generate_feedback(
                master_text=master_raw,
                student_text=student_raw_texts[i],
                score=score,
            )
            row_data["ai_feedback"] = feedback

        rows.append(row_data)
        logger.info(f"{filename}: score={score:.4f}, level={level}")

    return pd.DataFrame(rows)
