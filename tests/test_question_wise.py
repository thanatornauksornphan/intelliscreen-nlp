# tests/test_question_wise.py

import pandas as pd

from src.similarity.question_comparator import to_score_matrix
from src.similarity.question_splitter import split_by_question


def test_split_by_question_basic():
    text = (
        "Q1: The mitochondria is the powerhouse of the cell. "
        "Q2: DNA carries genetic information. "
        "Q3: Photosynthesis converts light into chemical energy."
    )
    result = split_by_question(text)

    assert set(result.keys()) == {"Q1", "Q2", "Q3"}
    assert "mitochondria" in result["Q1"]
    assert "DNA" in result["Q2"]
    assert "Photosynthesis" in result["Q3"]


def test_split_by_question_no_markers_returns_empty_dict():
    text = "This is a plain answer with no question numbering at all."
    result = split_by_question(text)
    assert result == {}


def test_split_by_question_empty_input():
    assert split_by_question("") == {}
    assert split_by_question("   ") == {}


def test_split_by_question_alternate_punctuation():
    text = "Q1) First answer here. Q2. Second answer here."
    result = split_by_question(text)
    assert "First answer" in result["Q1"]
    assert "Second answer" in result["Q2"]


def test_to_score_matrix_pivots_correctly():
    df = pd.DataFrame(
        [
            {
                "filename": "data/samples/student_a.txt",
                "question_id": "Q1",
                "similarity_score": 0.9,
                "match_level": "Excellent",
            },
            {
                "filename": "data/samples/student_a.txt",
                "question_id": "Q2",
                "similarity_score": 0.4,
                "match_level": "Poor",
            },
            {
                "filename": "data/samples/student_b.txt",
                "question_id": "Q1",
                "similarity_score": 0.6,
                "match_level": "Good",
            },
            {
                "filename": "data/samples/student_b.txt",
                "question_id": "Q2",
                "similarity_score": 0.7,
                "match_level": "Good",
            },
        ]
    )

    matrix = to_score_matrix(df)

    assert list(matrix.index) == ["student_a.txt", "student_b.txt"]
    assert list(matrix.columns) == ["Q1", "Q2"]
    assert matrix.loc["student_a.txt", "Q1"] == 0.9
    assert matrix.loc["student_b.txt", "Q2"] == 0.7


def test_to_score_matrix_empty_input_returns_empty():
    result = to_score_matrix(pd.DataFrame())
    assert result.empty
