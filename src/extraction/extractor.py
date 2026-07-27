# src/extraction/extractor.py

from pathlib import Path
from src.extraction.pdf_extractor import extract_text_from_pdf
from src.extraction.docx_extractor import extract_text_from_docx
from src.extraction.txt_extractor import extract_text_from_txt
from src.extraction.ocr_extractor import (
    extract_text_from_image,
    extract_text_from_scanned_pdf,
)
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

config = load_config()
logger = get_logger(__name__)
MIN_TEXT_THRESHOLD = config["extraction"]["min_text_length_for_native_pdf"]


def extract_text(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            result = extract_text_from_pdf(file_path)
            if result["metadata"]["char_count"] < MIN_TEXT_THRESHOLD:
                logger.info(f"{file_path} looks scanned, falling back to OCR")
                result = extract_text_from_scanned_pdf(file_path)
        elif ext == ".docx":
            result = extract_text_from_docx(file_path)
        elif ext == ".txt":
            result = extract_text_from_txt(file_path)
        elif ext in (".jpg", ".jpeg", ".png"):
            result = extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return result
    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}", exc_info=True)
        return {"text": "", "metadata": {"filename": file_path, "error": str(e)}}
