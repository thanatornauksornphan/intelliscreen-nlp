# IntelliScreen-NLP: Exam Paper Screening using NLP

![CI](https://github.com/thanatornauksornphan/intelliscreen-nlp/actions/workflows/ci.yml/badge.svg)

## Overview
IntelliScreen-NLP is an NLP-based system for screening and evaluating exam answer scripts against a master answer key. It accepts student submissions in multiple formats (PDF, DOCX, TXT, and images/scanned PDFs), extracts and preprocesses the text, and computes similarity scores against a reference answer key to flag how closely each submission matches. The system supports two interchangeable extraction/scoring engines — a lightweight lexical pipeline and a GPU-accelerated semantic pipeline — plus an optional question-wise scoring mode, and outputs detailed reports, visualizations, and exportable CSV results to support human graders, not replace them.

## Motivation / Problem Statement
Manually screening large batches of exam answers for content coverage or potential plagiarism is time-consuming and inconsistent between graders. This project explores whether an NLP pipeline can provide a fast, explainable first-pass screening layer — surfacing similarity scores and supporting evidence (e.g., top-matching sentences, optional AI-generated feedback) that a human grader can then review, rather than attempting to fully automate grading.

## Features
- Multi-format text extraction: native PDF, DOCX (including tables), TXT, and image/scanned-PDF
- **Dual OCR engines**, switchable via config: Tesseract (fast, CPU-only) or Florence-2 (GPU-accelerated vision-language model), lazily loaded so switching costs nothing when unused
- Automatic fallback to OCR when a PDF has no embedded text layer
- Configurable NLP preprocessing: stopword removal, lemmatization, domain-specific stopwords, and a protected-terms list for technical vocabulary
- **Dual similarity engines**, switchable via config: TF-IDF + cosine similarity, or semantic sentence-embedding similarity via `BAAI/bge-large-en-v1.5`
- **Question-wise comparison mode** (`--by-question`): splits both the master key and student submissions by question marker (e.g. "Q1:") and scores each question independently, rather than the whole document at once — includes graceful fallback to whole-document scoring if a submission has no detectable markers, and explicit handling of skipped questions
- Configurable match-level thresholds (Excellent / Good / Poor)
- Optional AI-generated feedback per submission via a local LLM (Ollama + Llama 3), with a pre-flight availability check that fails fast with a clear message if Ollama isn't reachable
- Explainability: top-matching-sentence extraction per submission (`--explain`)
- Visual reporting: bar charts (with live threshold reference lines), pie charts, word clouds, and heatmaps (student × question)
- CSV export of full similarity reports
- CLI entry point (`main.py`) and a web-based demo (`app.py`, Streamlit)
- A dedicated evaluation script (`scripts/compare_engines.py`) comparing TF-IDF vs. semantic scoring head-to-head on the same inputs
- A face-validity evaluation script (`scripts/evaluate_intended_vs_actual.py`) checking system output against a deliberately-authored evaluation batch with known intended quality tiers
- Centralized YAML configuration, structured logging, an automated test suite, and CI (GitHub Actions) running lint + tests on every push

## Tech Stack
- **Language:** Python 3.12
- **Text extraction:** PyMuPDF (fitz), python-docx, pytesseract, OpenCV, Pillow, pdf2image (requires Poppler)
- **OCR (VLM path):** transformers (pinned to 4.51.3 — see note below), Florence-2-large, timm, torchvision
- **NLP preprocessing:** spaCy (`en_core_web_sm`)
- **Vectorization & similarity:** scikit-learn (TF-IDF, cosine similarity) *or* sentence-transformers (`BAAI/bge-large-en-v1.5`)
- **GPU acceleration:** PyTorch (CUDA 12.8 build)
- **LLM reasoning (optional):** Ollama, running Llama 3 locally
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn, wordcloud
- **Web demo:** Streamlit
- **Testing & quality:** pytest, ruff, pre-commit, GitHub Actions
- **Config & logging:** PyYAML, Python's built-in `logging`

## Project Structure
```text
intelliscreen-nlp/
├── .github/workflows/
│   └── ci.yml                           # lint + test on every push/PR
├── src/
│   ├── extraction/                      # PDF, DOCX, TXT extractors + dual-engine OCR
│   ├── preprocessing/                   # spaCy-based text cleaning (mode-aware)
│   ├── similarity/                      # vectorization, scoring, batch + question-wise comparators
│   ├── visualization/                   # charts, word clouds, report generation
│   └── utils/                           # logger and config loader
├── scripts/
│   ├── compare_engines.py               # TF-IDF vs semantic evaluation
│   └── evaluate_intended_vs_actual.py   # face-validity check on the authored batch
├── data/
│   ├── raw/                             # input files (gitignored)
│   ├── samples/                         # example files, incl. evaluation batch (tracked)
│   ├── processed/                       # extracted/preprocessed text (gitignored)
│   └── outputs/                         # generated reports and charts (gitignored)
├── notebooks/                           # phase-by-phase development notebooks
├── tests/                               # pytest suite
├── configs/
│   └── config.yaml                      # engine selection, thresholds, question markers
├── logs/                                # rotating application logs (gitignored)
├── app.py                               # Streamlit demo
├── main.py                              # CLI entry point
├── requirements.txt
├── requirements-dev.txt
├── .pre-commit-config.yaml
└── README.md
```



## Setup & Installation

**Prerequisites:**
- Python 3.12
- An NVIDIA GPU with a recent driver, for GPU-accelerated semantic similarity / Florence-2 OCR (runs on CPU-only hardware too, just slower)
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) (`Library\bin` on PATH)
- [Ollama](https://ollama.com) with `llama3` pulled, only if using AI-generated feedback

```powershell
git clone https://github.com/thanatornauksornphan/intelliscreen-nlp.git
cd intelliscreen-nlp

python -m venv .venv
.venv\Scripts\activate

# Install PyTorch FIRST, from the CUDA-specific index — DO NOT rely on
# requirements.txt alone for this, or you may silently get a CPU-only or
# incompatible build (see note below)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
```

> **`transformers` is pinned exactly to `4.51.3`.** Florence-2's model code is incompatible with `transformers` 5.0+ (`AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`), unpatched upstream as of this writing. Do not upgrade this dependency without re-testing the Florence-2 OCR path.

## Usage

**Whole-document comparison:**
```powershell
python main.py --master data/samples/ANS_DOCX.docx --students data/samples/ANS_PDF.pdf data/samples/ANS_TXT.txt --explain --charts
```

**Question-wise comparison** (master key and submissions must use consistent `Q1:`, `Q2:`, ... markers):
```powershell
python main.py --master data/samples/QW_Master.docx --students data/samples/QW_Student_A.pdf data/samples/QW_Student_B.txt --by-question --charts
```

**Web demo:**
```powershell
streamlit run app.py
```

**Engine comparison (evaluation):**
```powershell
python -m scripts.compare_engines --master data/samples/ANS_DOCX.docx --students data/samples/ANS_PDF.pdf data/samples/ANS_TXT.txt
```

**Face-validity check against the authored evaluation batch:**
```powershell
python -m scripts.evaluate_intended_vs_actual
```

**Tests:**
```powershell
python -m pytest tests/ -v
```

## Methodology
1. **Extraction** — Tesseract or Florence-2 OCR fallback, lazily loaded.
2. **Preprocessing** — mode-aware: full lexical filtering for TF-IDF, minimal filtering for semantic (embedding models perform better on natural text).
3. **Vectorization & Similarity** — TF-IDF or semantic embeddings, computed per whole document or per question. Question-wise mode batches all students per question into a single vectorizer call for both performance and (for TF-IDF) more statistically meaningful IDF weighting than a 2-document comparison would give.
4. **Reporting** — CSV export, bar/pie/heatmap charts, word clouds, top-matching-sentence explainability, and optional local-LLM feedback.

## Results
Evaluated against a 20-submission authored batch spanning the full intended quality spectrum (Excellent/Good/Poor, including off-topic and wrong-topic distractors) across both whole-document and question-wise modes. See `scripts/evaluate_intended_vs_actual.py` output and `data/outputs/eval_*.csv` for the current agreement rate and confusion matrix between intended and actual classification. TF-IDF vs. semantic comparison (`scripts/compare_engines.py`) shows semantic scoring substantially outperforms TF-IDF on paraphrased answers with low lexical overlap (e.g. one heavily-paraphrased submission scored 0.03 under TF-IDF vs. 0.76 under semantic), while also showing semantic scoring can reward topical closeness over precision on weaker answers — both findings are discussed in the Ethics & Limitations section below.

## Roadmap / Status
**Current status:** feature-complete — dual-engine extraction and similarity, question-wise comparison, optional local LLM feedback, explainability, a working CLI and web demo, a full test suite, CI, and two dedicated evaluation scripts.

**Potential future extensions:**
- GPU-accelerated batch OCR comparison (Tesseract vs. Florence-2) as a single automated run, rather than two manual passes
- Evaluation against real, independently human-graded submissions (current evaluation uses synthetic, author-authored samples for privacy/ethics reasons)
- LMS integration

## Ethics & Limitations
- This system is a **screening aid**, not an automated grading replacement.
- TF-IDF measures lexical overlap; semantic embeddings capture meaning more broadly but can reward topical closeness over precision — observed directly in this project's own evaluation batch.
- AI-generated feedback (local LLM) is not guaranteed accurate; treat as a starting point for human review.
- OCR accuracy depends on scan/image quality and hasn't been benchmarked against a formal accuracy metric.
- Question-wise mode requires the master key and submissions to use a consistent, detectable question-marker format (e.g. `Q1:`) — this is a real constraint on input structure, not a general document-parsing capability.
- GPU-accelerated engines require a CUDA-capable GPU with sufficient VRAM for practical speed; the system remains functional but slower on CPU-only hardware.
- No real student data is used anywhere in this repository or its evaluation; all sample and evaluation-batch files are synthetic and author-created.

## Author & Acknowledgements
Thanatorn Auksornphan, Dr. Nasith Laosen, Phuket Rajabhat University.

## License
MIT License.
