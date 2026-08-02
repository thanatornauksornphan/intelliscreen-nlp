# src/visualization/charts.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def plot_similarity_bar_chart(df: pd.DataFrame, save: bool = True) -> None:
    config = load_config()
    output_dir = Path(config["reporting"]["output_dir"])
    dpi = config["reporting"]["chart_dpi"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="filename", y="similarity_score", hue="match_level")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Scores by File")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "similarity_bar_chart.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved bar chart to {path}")
    plt.show()


def plot_match_level_pie(df: pd.DataFrame, save: bool = True) -> None:
    config = load_config()
    output_dir = Path(config["reporting"]["output_dir"])
    dpi = config["reporting"]["chart_dpi"]

    counts = df["match_level"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Match Level Distribution")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "match_level_pie.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved pie chart to {path}")
    plt.show()


def plot_similarity_heatmap(score_matrix_df: pd.DataFrame, save: bool = True) -> None:
    """score_matrix_df: rows = students, columns = master keys/questions, values = similarity scores."""
    config = load_config()
    output_dir = Path(config["reporting"]["output_dir"])
    dpi = config["reporting"]["chart_dpi"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(score_matrix_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Similarity Heatmap")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "similarity_heatmap.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved heatmap to {path}")
    plt.show()
