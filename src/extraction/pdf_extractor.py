# src/extraction/pdf_extractor.py

import fitz
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(file_path: str) -> dict:
    logger.info(f"Attempting to extract text from PDF: {file_path}")
    try:
        with fitz.open(file_path) as doc:
            text_per_page = []

            for page_num, page in enumerate(doc):
                text_per_page.append(page.get_text())

            full_text = "\n".join(text_per_page)

            metadata = {
                "filename": file_path,
                "page_count": doc.page_count,
                "char_count": len(full_text),
            }

            logger.debug(
                f"Successfully extracted {len(full_text)} characters from {file_path}"
            )
            return {"text": full_text, "metadata": metadata}

    except Exception as e:
        logger.error(f"Failed to open or read PDF {file_path}: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {"filename": file_path, "char_count": 0, "error": str(e)},
        }
