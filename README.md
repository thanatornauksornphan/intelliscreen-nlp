# IntelliScreen-NLP: Exam Paper Screening using NLP

## Overview
IntelliScreen-NLP is an NLP-based system for screening and evaluating exam answer scripts against a master answer key. It accepts student submissions in multiple formats (PDF, DOCX, TXT, and scanned images), extracts and preprocesses the text, and computes similarity scores using TF-IDF and cosine similarity to flag how closely each answer matches the reference key. The system outputs detailed reports, visualizations, and exportable CSV results to support human graders — not replace them.

## Motivation / Problem Statement
Manually screening large batches of exam answers for content coverage or potential plagiarism is time-consuming and inconsistent between graders. This project explores whether a lightweight, CPU-friendly NLP pipeline can provide a fast, explainable first-pass screening layer — surfacing similarity scores and supporting evidence (e.g., top-matching sentences) that a human grader can then review, rather than attempting to fully automate grading.

## Features
- Multi-format text extraction: native PDF, DOCX (including tables), TXT, and image/scanned-PDF via OCR
- Automatic fallback to OCR when a PDF has no embedded text layer
- Configurable NLP preprocessing: stopword removal, lemmatization, domain-specific stopwords, and a protected-terms list for technical vocabulary (e.g., chemical formulas)
- TF-IDF vectorization with cosine similarity scoring against a master answer key
- Configurable match-level thresholds (Excellent / Good / Poor)
- Top-matching-sentence extraction for explainability
- Visual reporting: bar charts, pie charts, word clouds
- CSV export of full similarity reports
- CLI entry point (`main.py`) for running the full pipeline without a notebook
- Centralized YAML configuration, structured logging, and an automated test suite

## Tech Stack
- **Language:** Python 3.12
- **Text extraction:** PyMuPDF (fitz), python-docx, pytesseract, OpenCV, Pillow, pdf2image
- **NLP:** spaCy (`en_core_web_sm`)
- **Vectorization & similarity:** scikit-learn (TF-IDF, cosine similarity)
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Testing & quality:** pytest, ruff, pre-commit
- **Config & logging:** PyYAML, Python's built-in `logging` (rotating file handler)

## Project Structure
intelliscreen-nlp/
├── src/
│ ├── extraction/ # PDF, DOCX, TXT, and OCR extractors + unified dispatcher
│ ├── preprocessing/ # spaCy-based text cleaning and tokenization
│ ├── similarity/ # TF-IDF vectorization, cosine scoring, batch comparison
│ ├── visualization/ # charts, word clouds, report generation
│ └── utils/ # logger and config loader
├── data/
│ ├── raw/ # input files (gitignored)
│ ├── samples/ # small anonymized example files
│ ├── processed/ # extracted/preprocessed text (gitignored)
│ └── outputs/ # generated reports and charts (gitignored)
├── notebooks/ # phase-by-phase development/testing notebooks
├── tests/ # pytest suite
├── configs/
│ └── config.yaml # central configuration (paths, thresholds, model settings)
├── logs/ # rotating application logs (gitignored)
├── main.py # CLI entry point
├── requirements.txt
├── requirements-dev.txt
├── .pre-commit-config.yaml
└── README.md

## Setup & Installation

**Prerequisites:** Python 3.12, [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows build), and Poppler (required by `pdf2image`).

```powershell
# Clone the repository
git clone https://github.com/thanatornauksornphan/intelliscreen-nlp.git
cd intelliscreen-nlp

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm
```

Update `configs/config.yaml` with the correct path to your local Tesseract installation:
```yaml
paths:
  tesseract_cmd: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

## Usage

Run the full screening pipeline from the command line:

```powershell
python main.py --master data/samples/Answer-Variant-A.docx --students data/samples/Answer-Variant-B.pdf data/samples/Answer-Variant-C.txt --charts
```

This extracts and preprocesses all files, computes similarity scores against the master answer key, prints a results table, exports a CSV report to `data/outputs/`, and (with `--charts`) generates and saves visualizations.

Run the automated test suite:
```powershell
python -m pytest tests/ -v
```

## Methodology
The pipeline follows four stages:
1. **Extraction** — format-specific extractors return raw text and metadata; PDFs with little to no embedded text automatically fall back to an OCR pipeline (image preprocessing via OpenCV, text recognition via Tesseract).
2. **Preprocessing** — spaCy tokenizes and lemmatizes text, removes stopwords (including domain-specific terms like "exam" and "marks"), and preserves a configurable list of protected technical terms that would otherwise be lost to generic filtering.
3. **Vectorization & Similarity** — all documents in a comparison batch are vectorized together using TF-IDF (so IDF weighting reflects the actual corpus), and each student document's cosine similarity to the master answer key is computed and mapped to a qualitative match level using configurable thresholds.
4. **Reporting** — results are compiled into a structured DataFrame, exportable as CSV, alongside bar charts, pie charts, and word clouds for visual review, and top-matching-sentence extraction for explainability.

## Results
*(To be completed once evaluated against a labeled dataset — include similarity score distributions, comparison against human-assigned grades or match judgments if available, and OCR accuracy observations on scanned samples.)*

## Roadmap / Status
**Current status:** core pipeline complete (extraction, preprocessing, similarity scoring, reporting) with an automated test suite and CLI entry point.

**Planned / potential extensions:**
- Semantic similarity via sentence-transformers as an alternative to TF-IDF, for catching paraphrased answers that share little lexical overlap
- Question-wise comparison (splitting master key and student answers by question rather than scoring the whole document at once)
- Streamlit-based demo interface for interactive use
- GitHub Actions CI to run tests and linting automatically on push

## Ethics & Limitations
- This system is designed as a **screening aid**, not an automated grading replacement — similarity scores should support, not replace, human review.
- TF-IDF measures lexical overlap; a correctly-paraphrased answer with little word overlap may score lower than its actual quality warrants.
- OCR accuracy is dependent on scan/image quality and has not been benchmarked against a formal accuracy metric.
- Sentence-level matching uses a simple regex-based sentence splitter, which can mis-split text containing abbreviations or decimal numbers.
- No real student data is included in this repository; sample files used for development and testing are synthetic.

## Author & Acknowledgements
Thanatorn Auksornphan, Dr. Nasith Laosen, Phuket Rajabhat University.

## License
MIT License.
