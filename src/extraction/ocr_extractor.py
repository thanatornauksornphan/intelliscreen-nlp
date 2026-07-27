# src/extraction/ocr_extractor.py

import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()
pytesseract.pytesseract.tesseract_cmd = config["paths"]["tesseract_cmd"]


def preprocess_image_for_ocr(image_path: str):
    logger.info(f"Preprocessed image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"cv2 could not read image (corrupt or unsupported): {image_path}")
        raise ValueError(f"Unreadable image file: {image_path}")
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text_from_image(file_path: str) -> dict:
    logger.info(f"Extracting text via OCR from: {file_path}")
    processed = preprocess_image_for_ocr(file_path)
    text = pytesseract.image_to_string(processed)
    return {"text": text, "metadata": {"filename": file_path, "char_count": len(text)}}


def extract_text_from_scanned_pdf(file_path: str) -> dict:
    logger.info(f"Converting scanned PDF to images: {file_path}")

    dpi = config["extraction"]["ocr_dpi"]
    pages = convert_from_path(file_path, dpi=dpi)

    text_per_page = []

    for page_num, pil_img in enumerate(pages):
        logger.debug(f"Running OCR on page {page_num + 1} of {file_path}")

        # 1. Convert PIL Image to OpenCV Format
        open_cv_image = np.array(pil_img)

        # 2. Convert RGB to Grayscale
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

        # 3. Apply Otsu's Thresholding
        _, thresh = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. Extract Text
        text = pytesseract.image_to_string(thresh)
        text_per_page.append(text)

    full_text = "\n".join(text_per_page)

    metadata = {
        "filename": file_path,
        "page_count": len(pages),
        "char_count": len(full_text),
    }

    return {"text": full_text, "metadata": metadata}
