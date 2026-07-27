# src/extraction/txt_extractor.py

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_txt(file_path: str) -> dict:
    logger.info(f"Attempting to extract text from TXT: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        logger.debug(f"Successfully extracted {len(text)} characters from {file_path}")
    except Exception as e:
        logger.error(f"Failed to read TXT {file_path}: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {"filename": file_path, "char_count": 0, "error": str(e)},
        }
    return {"text": text, "metadata": {"filename": file_path, "char_count": len(text)}}
