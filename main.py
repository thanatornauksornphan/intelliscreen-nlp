# main.py

import argparse

from src.extraction.extractor import extract_text
from src.similarity.batch_comparator import compare_students_to_master
from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.visualization.charts import (
    plot_match_level_pie,
    plot_similarity_bar_chart,
)
from src.visualization.report_generator import export_report_csv, top_matching_sentences
from src.visualization.wordcloud_gen import generate_wordcloud

logger = get_logger("IntelliScreen-CLI")


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
        help="Print the top-matching sentences for each student against the master key",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    method = config["similarity"].get("method", "semantic").upper()

    logger.info(f"Starting IntelliScreen Run [Engine: {method}]")
    logger.info(f"Master Answer Key: {args.master}")
    logger.info(f"Student Submissions: {len(args.students)} files")

    # Fail fast if LLM reasoning is requested but Ollama isn't actually reachable —
    # better to know now than to silently get "Error generating LLM feedback." on every row.
    if config.get("llm_reasoning", {}).get("enabled", False):
        scorer_check = UnifiedScorer()
        if not scorer_check.check_llm_availability():
            logger.error(
                "LLM reasoning is enabled but Ollama is not ready. Aborting run."
            )
            return

    # Run screening comparator
    df = compare_students_to_master(args.students, args.master)

    if df.empty:
        logger.error("Screening failed. No valid results generated.")
        return

    # Print clean formatted summary table to console
    print("\n" + "=" * 80)
    print(f" INTELLISCREEN SCREENING RESULTS [{method} MODE]")
    print("=" * 80)

    display_cols = ["filename", "similarity_score", "match_level"]
    if "ai_feedback" in df.columns:
        display_cols.append("ai_feedback")

    print(df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")

    # Optional explainability: top-matching sentences per student
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

    # Export CSV Report
    report_path = export_report_csv(df, filename=args.output_csv)
    logger.info(f"CSV Report exported to {report_path}")

    # Generate Visualizations
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

    logger.info("IntelliScreen pipeline completed successfully.")


if __name__ == "__main__":
    main()
