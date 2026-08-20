# IntelliScreen-NLP: Exam Paper Screening using NLP

## Overview
IntelliScreen-NLP is an NLP-based system for screening and evaluating exam answer scripts against a master answer key. It accepts student submissions in multiple formats (PDF, DOCX, TXT, and images/scanned PDFs), extracts and preprocesses the text, and computes similarity scores against a reference answer key to flag how closely each submission matches. The system supports two interchangeable extraction/scoring engines — a lightweight lexical pipeline and a GPU-accelerated semantic pipeline — and outputs detailed reports, visualizations, and exportable CSV results to support human graders, not replace them.

## Motivation / Problem Statement
Manually screening large batches of exam answers for content coverage or potential plagiarism is time-consuming and inconsistent between graders. This project explores whether an NLP pipeline can provide a fast, explainable first-pass screening layer — surfacing similarity scores and supporting evidence (e.g., top-matching sentences, optional AI-generated feedback) that a human grader can then review, rather than attempting to fully automate grading.

## Features
- Multi-format text extraction: native PDF, DOCX (including tables), TXT, and image/scanned-PDF
- **Dual OCR engines**, switchable via config: Tesseract (fast, CPU-only, no model download) or Florence-2 (GPU-accelerated vision-language model, generally higher accuracy on difficult scans)
- Automatic fallback to OCR when a PDF has no embedded text layer
- Configurable NLP preprocessing: stopword removal, lemmatization, domain-specific stopwords, and a protected-terms list for technical vocabulary (e.g., chemical formulas, acronyms)
- **Dual similarity engines**, switchable via config: TF-IDF + cosine similarity (lexical overlap) or sentence-embedding semantic similarity via `BAAI/bge-large-en-v1.5` (GPU-accelerated, catches paraphrased answers with little word overlap)
- Configurable match-level thresholds (Excellent / Good / Poor)
- Optional AI-generated feedback per submission via a local LLM (Ollama + Llama 3) — explains what a student's answer missed, entirely offline
- Top-matching-sentence extraction for explainability
- Visual reporting: bar charts (with live threshold reference lines), pie charts, word clouds
- CSV export of full similarity reports
- CLI entry point (`main.py`) for running the full pipeline without a notebook
- Centralized YAML configuration, structured logging, and an automated test suite

## Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Language & Testing** | Python 3.12, `pytest`, `ruff`, `pre-commit` |
| **Text & OCR Extraction** | PyMuPDF, `python-docx`, Tesseract-OCR, Florence-2 Large (`microsoft/Florence-2-large`) |
| **NLP & Preprocessing** | spaCy (`en_core_web_sm`) |
| **Vectorization & Models** | scikit-learn (TF-IDF), Sentence-Transformers (`BAAI/bge-large-en-v1.5`), Ollama (Llama 3) |
| **Data & Visualization** | pandas, NumPy, matplotlib, seaborn, wordcloud |
| **Acceleration** | PyTorch (CUDA 12.8) |

## Project Structure
```text
intelliscreen-nlp/
├── src/
│   ├── extraction/      # PDF, DOCX, TXT extractors + dual-engine OCR (Tesseract / Florence-2)
│   ├── preprocessing/   # spaCy-based text cleaning (mode-aware: lexical vs. semantic)
│   ├── similarity/      # Dual-engine vectorization (TF-IDF / semantic) + scoring + LLM feedback
│   ├── visualization/   # charts, word clouds, report generation
│   └── utils/           # logger and config loader
├── data/
│   ├── raw/             # input files (gitignored)
│   ├── samples/         # small anonymized example files (tracked)
│   ├── processed/       # extracted/preprocessed text (gitignored)
│   └── outputs/         # generated reports and charts (gitignored)
├── notebooks/           # phase-by-phase development/testing notebooks
├── tests/               # pytest suite
├── configs/
│   └── config.yaml      # engine selection, thresholds, model settings
├── logs/                # rotating application logs (gitignored)
├── main.py              # CLI entry point
├── requirements.txt
├── requirements-dev.txt
├── .pre-commit-config.yaml
└── README.md
```


## Setup & Installation

**Prerequisites:**
- Python 3.12
- An NVIDIA GPU with a recent driver, if you want GPU-accelerated semantic similarity / Florence-2 OCR (the system runs on CPU-only hardware too, just slower — set `ocr_engine: "tesseract"` and `similarity.method: "tfidf"` in that case)
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows build)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) (required by `pdf2image`; add its `Library\bin` folder to PATH)
- [Ollama](https://ollama.com) with the `llama3` model pulled, **only** if you want AI-generated feedback (`llm_reasoning.enabled: true`)

```powershell
# Clone the repository
git clone https://github.com/thanatornauksornphan/intelliscreen-nlp.git
cd intelliscreen-nlp

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch FIRST, with the CUDA build matching your GPU
# (check https://pytorch.org/get-started/locally/ for the current recommended index)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Then install the rest of the dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm
```

If using LLM feedback, also:
```powershell
ollama pull llama3
```

Update `configs/config.yaml` with the correct path to your local Tesseract installation, and choose your active engines:
```yaml
paths:
  tesseract_cmd: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

extraction:
  ocr_engine: "tesseract"   # or "florence2" for GPU-accelerated VLM OCR

similarity:
  method: "semantic"        # or "tfidf" for lightweight lexical scoring

llm_reasoning:
  enabled: false             # set true only if Ollama is installed and running
```

> **Note on `transformers`:** this project pins `transformers==4.51.3` exactly rather than allowing newer versions. Florence-2's model code has a known incompatibility with `transformers` 5.0+ (`AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`) that has not yet been patched upstream by Microsoft. Do not upgrade this dependency without first re-testing the Florence-2 OCR path.

## Usage

Run the full screening pipeline from the command line:

```powershell
python main.py --master data/samples/Answer-Variant-A.docx --students data/samples/Answer-Variant-B.pdf data/samples/Answer-Variant-C.txt --charts
```

This extracts and preprocesses all files, computes similarity scores against the master answer key using whichever engine is active in `config.yaml`, prints a results table, exports a CSV report to `data/outputs/`, and (with `--charts`) generates and saves visualizations.

Run the automated test suite:
```powershell
python -m pytest tests/ -v
```

## Methodology
The pipeline follows four stages, with two swappable engine choices at the extraction and scoring stages:

1. **Extraction** — format-specific extractors return raw text and metadata. PDFs with little to no embedded text automatically fall back to OCR, using either Tesseract (fast, deterministic, CPU-only) or Florence-2 (a GPU-accelerated vision-language model, loaded lazily on first use to avoid unnecessary startup cost when OCR isn't needed).
2. **Preprocessing** — spaCy handles tokenization. In `tfidf` mode, it also performs lemmatization, stopword removal (including domain-specific terms), and preserves a configurable protected-terms list. In `semantic` mode, preprocessing is intentionally minimal — sentence-embedding models perform better on natural, grammatical text than on aggressively stripped tokens.
3. **Vectorization & Similarity** — all documents in a comparison batch are vectorized together, either via TF-IDF (fit fresh on each batch's corpus) or via `BAAI/bge-large-en-v1.5` sentence embeddings (GPU-accelerated, context-independent per document). Each student document's similarity to the master answer key is computed and mapped to a qualitative match level using configurable thresholds. Optionally, a locally-run LLM (Llama 3 via Ollama) generates short natural-language feedback explaining what a student's answer missed.
4. **Reporting** — results are compiled into a structured DataFrame, exportable as CSV, alongside bar charts (with live threshold reference lines), pie charts, and word clouds, plus top-matching-sentence extraction for explainability.

## Results
*(To be completed once evaluated against a labeled dataset — include a comparison of TF-IDF vs. semantic scoring on the same submissions, Tesseract vs. Florence-2 OCR accuracy on scanned samples, and, if human-graded reference scores are available, correlation between system output and human judgment.)*

## Roadmap / Status
**Current status:** dual-engine pipeline complete and verified end-to-end — extraction (Tesseract / Florence-2), preprocessing, similarity scoring (TF-IDF / semantic embeddings), optional local LLM feedback, reporting, an automated test suite, and a CLI entry point, all running on GPU-accelerated hardware.

**Planned / potential extensions:**
- A dedicated engine-comparison script to systematically evaluate TF-IDF vs. semantic and Tesseract vs. Florence-2 on the same inputs, for the thesis evaluation chapter
- Question-wise comparison (splitting master key and student answers by question rather than scoring the whole document at once)
- Wiring `top_matching_sentences` into the CLI output directly (currently notebook-only)
- Streamlit-based demo interface for interactive use
- GitHub Actions CI to run tests and linting automatically on push

## Ethics & Limitations
- This system is designed as a **screening aid**, not an automated grading replacement — similarity scores and AI-generated feedback should support, not replace, human review.
- TF-IDF measures lexical overlap; semantic embeddings capture meaning more broadly but are not immune to false positives — a superficially similar but substantively wrong answer can still score well under either method.
- AI-generated feedback (via the local LLM) is not guaranteed to be accurate; it should be treated as a starting point for a human grader, not a final judgment.
- OCR accuracy depends on scan/image quality and has not been benchmarked against a formal accuracy metric for either engine.
- The GPU-accelerated engines (Florence-2, semantic embeddings, LLM feedback) require a CUDA-capable GPU with sufficient VRAM to run at practical speed; the system falls back to functional but slower operation on CPU-only hardware.
- Sentence-level matching uses a simple sentence splitter, which can mis-split text containing abbreviations or decimal numbers.
- No real student data is included in this repository; sample files used for development and testing are synthetic.

## Author & Acknowledgements
Thanatorn Auksornphan, Dr. Nasith Laosen, Phuket Rajabhat University.

## License
MIT License.
