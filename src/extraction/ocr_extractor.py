# src/extraction/ocr_extractor.py

import cv2
import pytesseract
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

# --- Initialization ---
OCR_ENGINE = config["extraction"].get("ocr_engine", "tesseract")
logger.info(f"Initializing OCR Extractor with engine: {OCR_ENGINE.upper()}")

if OCR_ENGINE == "tesseract":
    pytesseract.pytesseract.tesseract_cmd = config["paths"]["tesseract_cmd"]
    processor, model, device = None, None, None
elif OCR_ENGINE == "florence2":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/Florence-2-large"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        processor, model = None, None


# --- Internal Extraction Logic ---
def _run_tesseract(image_path: str) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh)


def _run_vlm(image: Image.Image) -> str:
    if model is None:
        raise RuntimeError("VLM Model is not loaded.")
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
    if OCR_ENGINE == "tesseract":
        text = _run_tesseract(file_path)
    else:
        text = _run_vlm(Image.open(file_path).convert("RGB"))

    return {
        "text": text,
        "metadata": {
            "filename": file_path,
            "char_count": len(text),
            "engine": OCR_ENGINE,
        },
    }
