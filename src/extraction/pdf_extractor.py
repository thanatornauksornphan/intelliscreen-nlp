# src/extraction/pdf_extractor.py

import fitz
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(file_path: str) -> dict:
    """Extract text from a native (non-scanned) PDF."""
    doc = fitz.open(file_path)
    text_per_page = []
    for page_nnum, page in enumerate(doc):
        text_per_page.append(page.get_text())
    full_text = "\n".join(text_per_page)
    metadata = {
        "filename": file_path,
        "page_count": doc.page_count,
        "char_count": len(full_text),
    }
    doc.close()
    return {"text": full_text, "metadata": metadata}
