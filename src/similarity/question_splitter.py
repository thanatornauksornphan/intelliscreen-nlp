# src/similarity/question_splitter.py
"""Functions for splitting text into question-answer segments."""

import re

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_by_question(text: str) -> dict[str, str]:
    """Split the input text into segments based on question markers (e.g., "Q1:", "Q2:", etc.) and return a dictionary mapping question IDs to their corresponding text segments."""

    if not text or not text.strip():
        return {}

    config = load_config()
    pattern = config.get("question_wise", {}).get("marker_pattern", r"Q(\d+)[:.)]")

    matches = list(re.finditer(pattern, text))
    if not matches:
        logger.debug("No question markers found; question-wise split not applicable.")
        return {}

    segments: dict[str, str] = {}
    for i, match in enumerate(matches):
        question_id = f"Q{match.group(1)}"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment_text = text[start:end].strip()
        if segment_text:
            segments[question_id] = segment_text

    logger.debug(
        f"Split text into {len(segments)} question segments: {list(segments.keys())}"
    )
    return segments
