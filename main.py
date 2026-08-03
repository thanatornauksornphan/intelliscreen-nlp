# main.py

import argparse
from pathlib import Path
from src.similarity.batch_comparator import compare_students_to_master
from src.visualization.report_generator import export_report_csv
from src.visualization.charts import plot_similarity_bar_chart, plot_match_level_pie
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NLP Exam Paper Screening System")
    parser.add_argument(
        "--master", required=True, help="Path to master answer key file"
    )
    parser.add_argument(
        "--students", required=True, nargs="+", help="Paths to student answer files"
    )
    parser.add_argument(
        "--output-csv", default="similarity_report.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--charts", action="store_true", help="Generate and save charts"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        f"Starting screening run: master={args.master}, students={len(args.students)} files"
    )

    df = compare_students_to_master(args.students, args.master)
    print(df.to_string(index=False))

    export_report_csv(df, filename=args.output_csv)

    if args.charts:
        plot_similarity_bar_chart(df)
        plot_match_level_pie(df)

    logger.info("Screening run complete")


if __name__ == "__main__":
    main()
