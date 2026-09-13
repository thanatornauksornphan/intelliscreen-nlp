# main.py

import argparse

from src.extraction.extractor import extract_text
from src.similarity.batch_comparator import compare_students_to_master
from src.similarity.question_comparator import (
    compare_students_to_master_by_question,
    to_score_matrix,
)
from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.visualization.charts import (
    plot_match_level_pie,
    plot_similarity_bar_chart,
    plot_similarity_heatmap,
)
from src.visualization.report_generator import export_report_csv, top_matching_sentences
from src.visualization.wordcloud_gen import generate_wordcloud

logger = get_logger("IntelliScreen-CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IntelliScreen: Advanced NLP Exam Screening Pipeline"
    )
    parser.add_argument(
        "--master",
        required=True,
        help="Path to the master answer key file (PDF, DOCX, TXT, Image)",
    )
    parser.add_argument(
        "--students",
        required=True,
        nargs="+",
        help="Paths to student answer files",
    )
    parser.add_argument(
        "--output-csv",
        default="similarity_report.csv",
        help="Filename for the exported CSV report",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate and save visualization charts (bar chart, pie chart, word cloud)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the top-matching sentences for each student against the master key "
        "(ignored in --by-question mode)",
    )
    parser.add_argument(
        "--by-question",
        action="store_true",
        help="Score each question independently instead of the whole document at once. "
        "Requires the master key and student submissions to use consistent question "
        "markers (e.g. 'Q1:', 'Q2:'). Renders a heatmap instead of the bar/pie charts.",
    )
    return parser.parse_args()


def run_whole_document_mode(args, config, method):
    df = compare_students_to_master(args.students, args.master)

    if df.empty:
        logger.error("Screening failed. No valid results generated.")
        return

    print("\n" + "=" * 80)
    print(f" INTELLISCREEN SCREENING RESULTS [{method} MODE]")
    print("=" * 80)

    display_cols = ["filename", "similarity_score", "match_level"]
    if "ai_feedback" in df.columns:
        display_cols.append("ai_feedback")

    print(df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")

    if args.explain:
        logger.info("Generating top-matching-sentence explanations...")
        master_raw = extract_text(args.master)["text"]
        vectorizer = UnifiedVectorizer()

        for _, row in df.iterrows():
            student_path = row["filename"]
            student_raw = extract_text(student_path)["text"]
            matches = top_matching_sentences(student_raw, master_raw, vectorizer)

            print(
                f"--- {student_path} (score: {row['similarity_score']:.4f}, {row['match_level']}) ---"
            )
            if matches:
                for i, sentence in enumerate(matches, 1):
                    print(f"  {i}. {sentence}")
            else:
                print("  No matching sentences found.")
            print()

    report_path = export_report_csv(df, filename=args.output_csv)
    logger.info(f"CSV Report exported to {report_path}")

    if args.charts:
        logger.info("Generating report charts...")
        plot_similarity_bar_chart(df, save=True, show=False)
        plot_match_level_pie(df, save=True, show=False)

        master_data = extract_text(args.master)
        if master_data["text"]:
            from src.preprocessing.preprocessor import TextPreprocessor

            preprocessor = TextPreprocessor()
            wordcloud_text = preprocessor.preprocess_for_display(master_data["text"])
            generate_wordcloud(
                wordcloud_text,
                title="Master Answer Key Word Cloud",
                save=True,
                show=False,
            )
        logger.info("Charts successfully generated in output directory.")


def run_question_wise_mode(args):
    if args.explain:
        logger.warning("--explain is not supported in --by-question mode; ignoring.")

    df = compare_students_to_master_by_question(args.students, args.master)

    if df.empty:
        logger.error(
            "Question-wise screening failed. This usually means the master answer key "
            "has no detectable question markers (e.g. 'Q1:', 'Q2:'). Check "
            "configs/config.yaml's question_wise.marker_pattern, or drop --by-question "
            "to run whole-document comparison instead."
        )
        return

    print("\n" + "=" * 80)
    print(" INTELLISCREEN SCREENING RESULTS [QUESTION-WISE MODE]")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    report_path = export_report_csv(df, filename=args.output_csv)
    logger.info(f"CSV Report exported to {report_path}")

    if args.charts:
        logger.info("Generating question-wise heatmap...")
        score_matrix = to_score_matrix(df)
        plot_similarity_heatmap(score_matrix, save=True, show=False)
        logger.info("Heatmap successfully generated in output directory.")


def main():
    args = parse_args()
    config = load_config()
    method = config["similarity"].get("method", "semantic").upper()

    logger.info(f"Starting IntelliScreen Run [Engine: {method}]")
    logger.info(f"Master Answer Key: {args.master}")
    logger.info(f"Student Submissions: {len(args.students)} files")
    if args.by_question:
        logger.info("Mode: QUESTION-WISE")

    # Fail fast if LLM reasoning is requested but Ollama isn't actually reachable —
    # better to know now than to silently get "Error generating LLM feedback." on every row.
    if config.get("llm_reasoning", {}).get("enabled", False):
        scorer_check = UnifiedScorer()
        if not scorer_check.check_llm_availability():
            logger.error(
                "LLM reasoning is enabled but Ollama is not ready. Aborting run."
            )
            return

    if args.by_question:
        run_question_wise_mode(args)
    else:
        run_whole_document_mode(args, config, method)

    logger.info("IntelliScreen pipeline completed successfully.")


if __name__ == "__main__":
    main()
