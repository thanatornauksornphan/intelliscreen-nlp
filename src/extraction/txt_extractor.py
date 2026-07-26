def extract_text_from_txt(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return {"text": text, "metadata": {"filename": file_path, "char_count": len(text)}}
