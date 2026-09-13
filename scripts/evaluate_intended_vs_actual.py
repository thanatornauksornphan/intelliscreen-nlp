# scripts/evaluate_intended_vs_actual.py

"""Compare the intended match levels (as authored in the evaluation batch) against the actual match levels produced by the IntelliScreen scoring pipeline. This script is used for internal validation and quality assurance of the scoring system.
The intended match levels are defined in the WD_INTENDED and QWB_INTENDED dictionaries, which map student submission filenames (and question IDs for the question-wise batch) to their expected match levels ("Excellent", "Good", "Poor"). The script runs the scoring pipeline on the evaluation batch and compares the actual match levels against these intended values, reporting any discrepancies.
Note: A mismatch does not necessarily indicate an error in the scoring system. Some mismatches may be due to subjective differences in interpretation or the inherent variability of natural language. The purpose of this script is to highlight areas for review and potential improvement in the scoring logic or question design.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.similarity.batch_comparator import compare_students_to_master
from src.similarity.question_comparator import compare_students_to_master_by_question
from src.utils.logger import get_logger

logger = get_logger("IntendedVsActual")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Ground truth: intended quality tier, fixed at authoring time ---

WD_INTENDED = {
    "WD_S01.txt": "Excellent",
    "WD_S02.pdf": "Excellent",
    "WD_S03.txt": "Good",
    "WD_S04.docx": "Good",
    "WD_S05.txt": "Good",
    "WD_S06.txt": "Poor",
    "WD_S07.txt": "Poor",
    "WD_S08.txt": "Poor",
    "WD_S09.txt": "Poor",
    "WD_S10.pdf": "Excellent",
}

# (filename, question_id) -> intended tier. Questions the student deliberately
# skipped are labeled "Poor", matching the system's own defined behavior for
# missing answers (a 0.0 score) — this is the system working as designed,
# not a gap being papered over.
QWB_INTENDED = {
    ("QWB_S01.txt", "Q1"): "Excellent",
    ("QWB_S01.txt", "Q2"): "Excellent",
    ("QWB_S01.txt", "Q3"): "Excellent",
    ("QWB_S02.txt", "Q1"): "Excellent",
    ("QWB_S02.txt", "Q2"): "Excellent",
    ("QWB_S02.txt", "Q3"): "Poor",
    ("QWB_S03.txt", "Q1"): "Poor",
    ("QWB_S03.txt", "Q2"): "Excellent",
    ("QWB_S03.txt", "Q3"): "Excellent",
    ("QWB_S04.docx", "Q1"): "Excellent",
    ("QWB_S04.docx", "Q2"): "Poor",
    ("QWB_S04.docx", "Q3"): "Excellent",
    ("QWB_S05.txt", "Q1"): "Good",
    ("QWB_S05.txt", "Q2"): "Good",
    ("QWB_S05.txt", "Q3"): "Good",
    ("QWB_S06.txt", "Q1"): "Poor",
    ("QWB_S06.txt", "Q2"): "Poor",
    ("QWB_S06.txt", "Q3"): "Excellent",
    ("QWB_S07.pdf", "Q1"): "Excellent",
    ("QWB_S07.pdf", "Q2"): "Poor",
    ("QWB_S07.pdf", "Q3"): "Good",
    ("QWB_S08.txt", "Q1"): "Poor",
    ("QWB_S08.txt", "Q2"): "Poor",
    ("QWB_S08.txt", "Q3"): "Poor",
    ("QWB_S09.txt", "Q1"): "Poor",
    ("QWB_S09.txt", "Q2"): "Excellent",
    ("QWB_S09.txt", "Q3"): "Poor",
    ("QWB_S10.txt", "Q1"): "Excellent",
    ("QWB_S10.txt", "Q2"): "Excellent",
    ("QWB_S10.txt", "Q3"): "Poor",
}


def evaluate_whole_document(master_path: str, samples_dir: str) -> pd.DataFrame:
    student_paths = [str(Path(samples_dir) / fname) for fname in WD_INTENDED]
    df = compare_students_to_master(student_paths, master_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name)
    df["intended"] = df["basename"].map(WD_INTENDED)
    df["match"] = df["intended"] == df["match_level"]
    return df


def evaluate_question_wise(master_path: str, samples_dir: str) -> pd.DataFrame:
    filenames = sorted({fname for fname, _ in QWB_INTENDED})
    student_paths = [str(Path(samples_dir) / fname) for fname in filenames]
    df = compare_students_to_master_by_question(student_paths, master_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name)
    df["intended"] = df.apply(
        lambda row: QWB_INTENDED.get((row["basename"], row["question_id"])), axis=1
    )
    df["match"] = df["intended"] == df["match_level"]
    return df


def print_summary(df: pd.DataFrame, label: str) -> None:
    total = len(df)
    correct = int(df["match"].sum())
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 70)
    print(f" {label} — Intended vs Actual Match-Level Agreement")
    print("=" * 70)
    print(f"Agreement: {correct}/{total} ({accuracy:.1%})")
    print()
    print("Confusion matrix (rows = intended, columns = actual):")
    print(pd.crosstab(df["intended"], df["match_level"]))

    mismatches = df[~df["match"]]
    if not mismatches.empty:
        print("\nMismatches (not necessarily errors — see script docstring):")
        cols = [
            c
            for c in [
                "basename",
                "question_id",
                "intended",
                "match_level",
                "similarity_score",
            ]
            if c in mismatches.columns
        ]
        print(mismatches[cols].to_string(index=False))
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare intended vs actual match levels on the authored evaluation batch."
    )
    parser.add_argument("--samples-dir", default="data/samples")
    parser.add_argument("--wd-master", default="data/samples/ANS_DOCX.docx")
    parser.add_argument("--qw-master", default="data/samples/QW_Master.docx")
    args = parser.parse_args()

    wd_df = evaluate_whole_document(args.wd_master, args.samples_dir)
    print_summary(wd_df, "WHOLE-DOCUMENT BATCH")

    qw_df = evaluate_question_wise(args.qw_master, args.samples_dir)
    print_summary(qw_df, "QUESTION-WISE BATCH")

    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wd_df.to_csv(output_dir / "eval_whole_document.csv", index=False)
    qw_df.to_csv(output_dir / "eval_question_wise.csv", index=False)
    logger.info(f"Evaluation CSVs saved to {output_dir}")


if __name__ == "__main__":
    main()
