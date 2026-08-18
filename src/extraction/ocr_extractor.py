# src/extraction/ocr_extractor.py

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForCausalLM


from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

# --- Initialization ---
OCR_ENGINE = config["extraction"].get("ocr_engine", "tesseract")
logger.info(f"OCR Extractor configured with engine: {OCR_ENGINE.upper()}")

if OCR_ENGINE == "tesseract":
    pytesseract.pytesseract.tesseract_cmd = config["paths"]["tesseract_cmd"]
elif OCR_ENGINE != "florence2":
    logger.warning(
        f"Unrecognized ocr_engine '{OCR_ENGINE}' in config.yaml — falling back to VLM code path."
    )

_florence_processor = None
_florence_model = None
_florence_device = None


def _get_florence_model():
    global _florence_processor, _florence_model, _florence_device

    if _florence_model is None:
        _florence_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "microsoft/Florence-2-large"
        logger.info(
            f"Loading Florence-2 model on {_florence_device.upper()} (first use)..."
        )
        try:
            _florence_processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            _florence_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(_florence_device)
            logger.info("Florence-2 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 VLM: {e}", exc_info=True)
            raise RuntimeError(
                "Florence-2 model failed to load. Check network access, "
                "transformers/timm versions, and available VRAM."
            ) from e

    return _florence_processor, _florence_model, _florence_device


# --- Internal Extraction Logic ---
def _run_tesseract(image: Image.Image) -> str:
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image.convert("RGB"))
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    # Otsu's thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh)


def _run_vlm(image: Image.Image) -> str:
    processor, model, device = _get_florence_model()

    prompt = "<OCR>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    return parsed.get(prompt, "")


# --- Public API ---
def extract_text_from_image(file_path: str) -> dict:
    logger.info(f"Extracting text via {OCR_ENGINE.upper()} from: {file_path}")
    image = Image.open(file_path).convert("RGB")

    if OCR_ENGINE == "tesseract":
        text = _run_tesseract(image)
    else:
        text = _run_vlm(image)

    return {
        "text": text,
        "metadata": {
            "filename": file_path,
            "char_count": len(text),
            "engine": OCR_ENGINE,
        },
    }


def extract_text_from_scanned_pdf(file_path: str) -> dict:
    logger.info(
        f"Converting scanned PDF for {OCR_ENGINE.upper()} extraction: {file_path}"
    )
    dpi = config["extraction"]["ocr_dpi"]
    pages = convert_from_path(file_path, dpi=dpi)

    text_per_page = []
    for page_num, pil_img in enumerate(pages):
        logger.debug(
            f"Running {OCR_ENGINE.upper()} on page {page_num + 1} of {file_path}"
        )

        if OCR_ENGINE == "tesseract":
            text = _run_tesseract(pil_img)
        else:
            text = _run_vlm(pil_img.convert("RGB"))

        text_per_page.append(text)

    full_text = "\n".join(text_per_page)

    metadata = {
        "filename": file_path,
        "page_count": len(pages),
        "char_count": len(full_text),
        "engine": OCR_ENGINE,
    }
    return {"text": full_text, "metadata": metadata}
