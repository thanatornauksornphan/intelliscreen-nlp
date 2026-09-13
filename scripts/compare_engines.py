# scripts/compare_engines.py

"""
Evaluation script: runs the same student submissions against the same master
answer key through BOTH similarity engines (TF-IDF and Semantic embeddings),
and produces a side-by-side comparison table + chart.

This is intended for the thesis evaluation chapter — it answers the question
"does semantic similarity actually catch paraphrasing better than lexical
TF-IDF overlap, on the same inputs?" with real numbers instead of just an
architecture description.

Text extraction (PDF/DOCX/TXT/OCR) is run ONCE and shared across both engine
passes, since extraction is independent of which similarity method is used —
only preprocessing and vectorization differ between TF-IDF and semantic mode.

IMPORTANT — run this as a module, from the project root, not as a bare script:
    python -m scripts.compare_engines --master data/samples/Answer-Variant-A.docx \
        --students data/samples/Answer-Variant-B.pdf data/samples/Answer-Variant-C.txt

Running it with `-m` from the project root is what lets Python resolve the
`src` package without any manual sys.path manipulation — running it as
`python scripts/compare_engines.py` directly will NOT work, since Python
would only add the scripts/ folder itself to sys.path, not the project root.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from src.extraction.extractor import extract_text
from src.utils.logger import get_logger

logger = get_logger("EngineComparison")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_raw_config_text() -> str:
    """Read config.yaml as raw text (not through the app's cached loader),
    so we have an exact, byte-for-byte copy to restore afterward."""
    return CONFIG_PATH.read_text(encoding="utf-8")


def _write_config_with_method(original_text: str, method: str) -> None:
    """Temporarily rewrite config.yaml with a different similarity.method,
    leaving every other setting untouched."""
    config_dict = yaml.safe_load(original_text)
    config_dict["similarity"]["method"] = method
    CONFIG_PATH.write_text(
        yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8"
    )


def run_pass_for_method(
    method: str, master_raw: str, student_raw_texts: dict[str, str]
) -> list[dict]:
    """Instantiate fresh preprocessing/vectorization/scoring components
    configured for the given method (picked up from the just-rewritten
    config.yaml), and score every student against the master."""

    # Imported here, not at module top, so each pass re-reads the config
    # that was just written to disk immediately before this call.
    from src.preprocessing.preprocessor import TextPreprocessor
    from src.similarity.similarity_scorer import UnifiedScorer
    from src.similarity.vectorizer import UnifiedVectorizer

    preprocessor = TextPreprocessor()
    vectorizer = UnifiedVectorizer()
    scorer = UnifiedScorer()

    master_clean = preprocessor.preprocess_to_string(master_raw)
    student_filenames = list(student_raw_texts.keys())
    student_clean = [
        preprocessor.preprocess_to_string(t) for t in student_raw_texts.values()
    ]

    corpus = [master_clean] + student_clean
    vectors = vectorizer.transform(corpus)
    master_vec = vectors[0:1]

    rows = []
    for i, filename in enumerate(student_filenames):
        student_vec = vectors[i + 1 : i + 2]
        score = scorer.compute_score(master_vec, student_vec)
        level = scorer.match_level(score)
        rows.append(
            {
                "filename": filename,
                "method": method,
                "similarity_score": round(score, 4),
                "match_level": level,
            }
        )
        logger.info(f"[{method.upper()}] {filename}: score={score:.4f} ({level})")

    return rows


def run_comparison(master_raw: str, student_raw_texts: dict[str, str]) -> list[dict]:
    """Run both TF-IDF and semantic passes, temporarily swapping
    config.yaml's similarity.method between them.

    The try/finally guarantees the original config.yaml is restored exactly,
    even if a pass raises partway through.
    """
    original_config_text = _load_raw_config_text()
    all_rows = []
    try:
        for method in ["tfidf", "semantic"]:
            logger.info(f"--- Running {method.upper()} pass ---")
            _write_config_with_method(original_config_text, method)
            all_rows.extend(run_pass_for_method(method, master_raw, student_raw_texts))
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")
        logger.info("Restored original config.yaml")
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF vs Semantic similarity engines on the same inputs."
    )
    parser.add_argument(
        "--master", required=True, help="Path to master answer key file"
    )
    parser.add_argument(
        "--students", required=True, nargs="+", help="Paths to student answer files"
    )
    parser.add_argument("--output-csv", default="engine_comparison.csv")
    args = parser.parse_args()

    # 1. Extract text ONCE — extraction doesn't depend on the similarity
    #    method, so there's no reason to re-run PDF/OCR parsing per engine pass.
    logger.info("Extracting text (shared across both engine passes)...")
    master_result = extract_text(args.master)
    if "error" in master_result["metadata"]:
        logger.error(
            f"Failed to extract master file: {master_result['metadata']['error']}"
        )
        return

    student_raw_texts = {}
    for path in args.students:
        result = extract_text(path)
        if "error" in result["metadata"]:
            logger.warning(f"Skipping {path} due to extraction error")
            continue
        student_raw_texts[path] = result["text"]

    if not student_raw_texts:
        logger.error("No valid student files to compare.")
        return

    master_raw = master_result["text"]

    # 2. Run both engine passes
    all_rows = run_comparison(master_raw, student_raw_texts)

    # 3. Combine results and export
    df = pd.DataFrame(all_rows)
    df["display_name"] = df["filename"].apply(lambda p: Path(p).name)

    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.output_csv
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison results saved to {csv_path}")

    print("\n" + "=" * 80)
    print(" ENGINE COMPARISON: TF-IDF vs SEMANTIC")
    print("=" * 80)
    print(
        df[["display_name", "method", "similarity_score", "match_level"]].to_string(
            index=False
        )
    )
    print("=" * 80 + "\n")

    # 4. Grouped bar chart — side by side, per file, per engine
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="display_name", y="similarity_score", hue="method")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Similarity Score")
    plt.xlabel("Submission")
    plt.title("TF-IDF vs Semantic Similarity — Same Inputs")
    plt.legend(title="Engine")
    plt.tight_layout()

    chart_path = output_dir / "engine_comparison_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    main()
