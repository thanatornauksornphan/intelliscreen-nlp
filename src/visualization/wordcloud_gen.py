# src/visualization/wordcloud_gen.py

from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_wordcloud(
    text: str, title: str = "Word Cloud", save: bool = True, show: bool = False
) -> None:
    if not text or not text.strip():
        logger.warning(
            f"Empty text provided for wordcloud '{title}'. Skipping generation."
        )
        return

    config = load_config()
    output_dir = PROJECT_ROOT / config["reporting"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    max_words = config["reporting"].get("wordcloud_max_words", 100)
    dpi = config["reporting"].get("chart_dpi", 150)

    wc = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color="white",
        colormap="viridis",
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()

    if save:
        safe_title = title.lower().replace(" ", "_")
        path = output_dir / f"{safe_title}.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved word cloud to {path}")

    if show:
        plt.show()
    plt.close()
