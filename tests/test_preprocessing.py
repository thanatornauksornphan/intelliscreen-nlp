# tests/test_preprocessing.py

from src.preprocessing.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()


def test_empty_input_returns_empty():
    assert preprocessor.preprocess("") == []
    assert preprocessor.preprocess("   ") == []
    assert preprocessor.preprocess_to_string("") == ""


def test_protected_terms_preserved():
    text = "The solution contains H2O and CO2 molecules."
    tokens = preprocessor.preprocess(text)
    # Protected terms survive in both tfidf and semantic modes
    assert any("h2o" in token.lower() for token in tokens)
    assert any("co2" in token.lower() for token in tokens)


def test_preprocess_to_string_structure():
    text = "Photosynthesis converts carbon dioxide into glucose."
    cleaned = preprocessor.preprocess_to_string(text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
