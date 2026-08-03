# tests/test_preprocessing.py

from src.preprocessing.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()  # loaded once, shared across tests in this file


def test_stopwords_are_removed():
    tokens = preprocessor.preprocess("The cat is on the mat")
    assert "the" not in tokens
    assert "is" not in tokens


def test_protected_terms_survive():
    tokens = preprocessor.preprocess("The solution contains H2O and CO2")
    assert "h2o" in tokens
    assert "co2" in tokens


def test_empty_input_returns_empty_list():
    assert preprocessor.preprocess("") == []
    assert preprocessor.preprocess("   ") == []
