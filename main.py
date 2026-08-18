# main.py

import argparse

from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.batch_comparator import compare_students_to_master
from src.visualization.report_generator import export_report_csv
from src.visualization.charts import (
    plot_similarity_bar_chart,
    plot_match_level_pie,
)
from src.visualization.wordcloud_gen import generate_wordcloud
from src.extraction.extractor import extract_text
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

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
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    method = config["similarity"].get("method", "semantic").upper()

    logger.info(f"Starting IntelliScreen Run [Engine: {method}]")
    logger.info(f"Master Answer Key: {args.master}")
    logger.info(f"Student Submissions: {len(args.students)} files")

    config = load_config()

    if config.get("llm_reasoning", {}).get("enabled", False):
        scorer = UnifiedScorer()
    if not scorer.check_llm_availability():
        logger.error("LLM reasoning is enabled but Ollama is not ready. Aborting run.")
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

    # Export CSV Report
    report_path = export_report_csv(df, filename=args.output_csv)
    logger.info(f"CSV Report exported to {report_path}")

    # Generate Visualizations
    if args.charts:
        logger.info("Generating report charts...")
        plot_similarity_bar_chart(df, save=True, show=False)
        plot_match_level_pie(df, save=True, show=False)

        # Generate word cloud for master key
        master_data = extract_text(args.master)
        if master_data["text"]:
            generate_wordcloud(
                master_data["text"],
                title="Master Answer Key Word Cloud",
                save=True,
                show=False,
            )
        logger.info("Charts successfully generated in output directory.")

    logger.info("IntelliScreen pipeline completed successfully.")


if __name__ == "__main__":
    main()
