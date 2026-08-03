# tests/test_extraction.py

from src.extraction.txt_extractor import extract_text_from_txt
from src.extraction.extractor import extract_text


def test_txt_extraction_success(tmp_path):
    """A valid txt file should return non-empty text and no error."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("This is a test answer.", encoding="utf-8")

    result = extract_text_from_txt(str(sample_file))

    assert result["text"] == "This is a test answer."
    assert "error" not in result["metadata"]


def test_extract_text_missing_file_returns_error_dict():
    """A non-existent file should fail gracefully, not raise an exception."""
    result = extract_text("does_not_exist.pdf")

    assert result["text"] == ""
    assert "error" in result["metadata"]


def test_extract_text_unsupported_format(tmp_path):
    """An unsupported extension should be handled as an error, not crash."""
    bad_file = tmp_path / "sample.xyz"
    bad_file.write_text("irrelevant")

    result = extract_text(str(bad_file))

    assert "error" in result["metadata"]
