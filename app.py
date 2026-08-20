# app.py

"""
IntelliScreen-NLP — Streamlit demo interface.

A thin UI wrapper around the existing pipeline (src/). No screening logic
lives here — this file only handles file upload, calling the pipeline, and
displaying results. All extraction / preprocessing / scoring / charting
logic stays in src/, unchanged.

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.extraction.extractor import extract_text
from src.preprocessing.preprocessor import TextPreprocessor
from src.similarity.batch_comparator import compare_students_to_master
from src.similarity.similarity_scorer import UnifiedScorer
from src.similarity.vectorizer import UnifiedVectorizer
from src.utils.config_loader import load_config
from src.visualization.charts import plot_match_level_pie, plot_similarity_bar_chart
from src.visualization.report_generator import top_matching_sentences
from src.visualization.wordcloud_gen import generate_wordcloud

st.set_page_config(page_title="IntelliScreen-NLP", layout="wide")

config = load_config()
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, target_dir: Path) -> str:
    """Write a Streamlit UploadedFile to disk so the existing extract_text()
    pipeline (which expects file paths, not in-memory buffers) can use it."""
    target_path = target_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path)


st.title("📝 IntelliScreen-NLP")
st.caption(
    "NLP-based exam paper screening — upload a master answer key and student submissions to compare."
)

with st.sidebar:
    st.subheader("Active Configuration")
    st.write(
        f"**OCR Engine:** {config['extraction'].get('ocr_engine', 'tesseract').upper()}"
    )
    st.write(
        f"**Similarity Method:** {config['similarity'].get('method', 'semantic').upper()}"
    )
    llm_enabled = config.get("llm_reasoning", {}).get("enabled", False)
    st.write(f"**AI Feedback (LLM):** {'Enabled' if llm_enabled else 'Disabled'}")
    st.caption("Change these in configs/config.yaml, then restart the app.")

col1, col2 = st.columns(2)
with col1:
    master_file = st.file_uploader(
        "Master Answer Key", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"]
    )
with col2:
    student_files = st.file_uploader(
        "Student Submissions",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

run_clicked = st.button(
    "Run Screening", type="primary", disabled=not (master_file and student_files)
)

if run_clicked:
    # LLM availability pre-check — same pattern as main.py, fail fast with a clear message
    if llm_enabled:
        scorer_check = UnifiedScorer()
        if not scorer_check.check_llm_availability():
            st.error(
                "AI feedback is enabled in config.yaml but Ollama is not reachable. "
                "Start Ollama (and confirm `ollama pull llama3` has been run), or set "
                "`llm_reasoning.enabled: false` and re-run."
            )
            st.stop()

        with (
            st.spinner("Extracting, preprocessing, and scoring submissions..."),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            tmp_path = Path(tmp_dir)
            master_path = save_uploaded_file(master_file, tmp_path)
            student_paths = [save_uploaded_file(f, tmp_path) for f in student_files]

            df = compare_students_to_master(student_paths, master_path)

            if df.empty:
                st.error(
                    "Screening failed — no valid results were generated. Check that your files extracted correctly."
                )
                st.stop()

            # Charts are generated and saved to disk by the existing functions;
            # read them back as images rather than modifying those functions
            # just to support inline rendering.
            plot_similarity_bar_chart(df, save=True, show=False)
            plot_match_level_pie(df, save=True, show=False)

            master_raw = extract_text(master_path)["text"]
            if master_raw:
                preprocessor = TextPreprocessor()
                wordcloud_text = preprocessor.preprocess_for_display(master_raw)
                generate_wordcloud(
                    wordcloud_text,
                    title="Master Answer Key Word Cloud",
                    save=True,
                    show=False,
                )

            # Explainability — same top_matching_sentences function used by main.py --explain
            vectorizer = UnifiedVectorizer()
            explanations = {}
            for path in student_paths:
                student_raw = extract_text(path)["text"]
                explanations[path] = top_matching_sentences(
                    student_raw, master_raw, vectorizer
                )

    st.success(f"Screening complete — {len(df)} submission(s) scored.")

    st.subheader("Results")
    display_cols = ["filename", "similarity_score", "match_level"]
    if "ai_feedback" in df.columns:
        display_cols.append("ai_feedback")
    st.dataframe(df[display_cols], use_container_width=True)

    st.download_button(
        "Download CSV Report",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="similarity_report.csv",
        mime="text/csv",
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        bar_path = OUTPUT_DIR / "similarity_bar_chart.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Similarity Scores")
    with chart_col2:
        pie_path = OUTPUT_DIR / "match_level_pie.png"
        if pie_path.exists():
            st.image(str(pie_path), caption="Match Level Distribution")

    wc_path = OUTPUT_DIR / "master_answer_key_word_cloud.png"
    if wc_path.exists():
        st.image(str(wc_path), caption="Master Answer Key Word Cloud")

    st.subheader("Explainability — Top Matching Sentences")
    for _, row in df.iterrows():
        with st.expander(
            f"{Path(row['filename']).name} — {row['similarity_score']:.4f} ({row['match_level']})"
        ):
            matches = explanations.get(row["filename"], [])
            if matches:
                for i, sentence in enumerate(matches, 1):
                    st.write(f"{i}. {sentence}")
            else:
                st.write("No matching sentences found.")
