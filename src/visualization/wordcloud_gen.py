# src/visualization/wordcloud_gen.py

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def generate_wordcloud(text: str, title: str = "Word Cloud", save: bool = True) -> None:
    config = load_config()
    output_dir = Path(config["reporting"]["output_dir"])
    max_words = config["reporting"]["wordcloud_max_words"]

    wc = WordCloud(
        width=800, height=400, max_words=max_words, background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.lower().replace(" ", "_")
        path = output_dir / f"{safe_title}.png"
        plt.savefig(path, dpi=config["reporting"]["chart_dpi"])
        logger.info(f"Saved word cloud to {path}")
    plt.show()
