import pytest

from scripts import compare_engines

SAMPLE_CONFIG_TEXT = """\
paths:
  tesseract_cmd: "C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe"
extraction:
  ocr_engine: "tesseract"
similarity:
  method: "semantic"
  thresholds:
    excellent: 0.85
    good: 0.65
llm_reasoning:
  enabled: false
"""


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point compare_engines.CONFIG_PATH at a throwaway file instead of the
    real project config.yaml, so these tests can never touch or corrupt the
    actual configuration."""
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text(SAMPLE_CONFIG_TEXT, encoding="utf-8")
    monkeypatch.setattr(compare_engines, "CONFIG_PATH", fake_config)
    return fake_config


def test_write_config_with_method_only_changes_method(isolated_config):
    original_text = compare_engines._load_raw_config_text()

    compare_engines._write_config_with_method(original_text, "tfidf")

    updated_text = isolated_config.read_text(encoding="utf-8")
    assert "method: tfidf" in updated_text
    # Everything else should be untouched
    assert "excellent: 0.85" in updated_text
    assert "ocr_engine: tesseract" in updated_text


def test_run_comparison_restores_config_after_success(isolated_config, monkeypatch):
    original_text = isolated_config.read_text(encoding="utf-8")

    # Stub out the expensive part — we're testing config safety, not scoring
    monkeypatch.setattr(
        compare_engines,
        "run_pass_for_method",
        lambda method, master_raw, student_raw_texts: [
            {
                "filename": "dummy.txt",
                "method": method,
                "similarity_score": 0.5,
                "match_level": "Good",
            }
        ],
    )

    rows = compare_engines.run_comparison("master text", {"dummy.txt": "student text"})

    # Both passes should have produced results
    methods_seen = {row["method"] for row in rows}
    assert methods_seen == {"tfidf", "semantic"}

    # Config must be back to exactly what it was before the run
    assert isolated_config.read_text(encoding="utf-8") == original_text


def test_run_comparison_restores_config_even_if_a_pass_raises(
    isolated_config, monkeypatch
):
    original_text = isolated_config.read_text(encoding="utf-8")

    def failing_pass(method, master_raw, student_raw_texts):
        raise RuntimeError("Simulated failure mid-pass")

    monkeypatch.setattr(compare_engines, "run_pass_for_method", failing_pass)

    # The exception should still propagate — the finally block must not swallow it
    with pytest.raises(RuntimeError, match="Simulated failure mid-pass"):
        compare_engines.run_comparison("master text", {"dummy.txt": "student text"})

    # But config.yaml must still be restored correctly despite the failure
    assert isolated_config.read_text(encoding="utf-8") == original_text
