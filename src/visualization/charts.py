# src/visualization/charts.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_output_dir() -> tuple[Path, int]:
    config = load_config()
    output_dir = PROJECT_ROOT / config["reporting"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = config["reporting"].get("chart_dpi", 150)
    return output_dir, dpi


def plot_similarity_bar_chart(
    df: pd.DataFrame, save: bool = True, show: bool = False
) -> None:
    if df.empty:
        logger.warning("Empty DataFrame passed to plot_similarity_bar_chart.")
        return

    output_dir, dpi = _get_output_dir()
    config = load_config()
    thresholds = config["similarity"]["thresholds"]

    plot_df = df.copy()
    plot_df["display_name"] = plot_df["filename"].apply(lambda p: Path(p).name)

    plt.figure(figsize=(10, 6))
    palette = {"Excellent": "#2ecc71", "Good": "#f39c12", "Poor": "#e74c3c"}
    sns.barplot(
        data=plot_df,
        x="display_name",
        y="similarity_score",
        hue="match_level",
        palette=palette,
    )
    plt.axhline(
        thresholds["excellent"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Excellent Threshold ({thresholds['excellent']:.2f})",
    )
    plt.axhline(
        thresholds["good"],
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Good Threshold ({thresholds['good']:.2f})",
    )
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Similarity Score")
    plt.xlabel("Submission")
    plt.title("Exam Paper Similarity Scores against Master Key")
    plt.legend()
    plt.tight_layout()

    if save:
        path = output_dir / "similarity_bar_chart.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved bar chart to {path}")

    if show:
        plt.show()
    plt.close()


def plot_match_level_pie(
    df: pd.DataFrame, save: bool = True, show: bool = False
) -> None:
    if df.empty or "match_level" not in df.columns:
        logger.warning("Invalid DataFrame passed to plot_match_level_pie.")
        return

    output_dir, dpi = _get_output_dir()
    counts = df["match_level"].value_counts()
    colors = {"Excellent": "#2ecc71", "Good": "#f39c12", "Poor": "#e74c3c"}
    pie_colors = [colors.get(level, "#95a5a6") for level in counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.title("Match Level Distribution")
    plt.tight_layout()

    if save:
        path = output_dir / "match_level_pie.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved pie chart to {path}")

    if show:
        plt.show()
    plt.close()


def plot_similarity_heatmap(
    score_matrix_df: pd.DataFrame, save: bool = True, show: bool = False
) -> None:
    if score_matrix_df.empty:
        logger.warning("Empty matrix passed to plot_similarity_heatmap.")
        return

    output_dir, dpi = _get_output_dir()

    plt.figure(figsize=(10, 8))
    sns.heatmap(score_matrix_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
    plt.title("Student-to-Question Similarity Heatmap")
    plt.xlabel("Evaluation Rubric / Questions")
    plt.ylabel("Student Submissions")
    plt.tight_layout()

    if save:
        path = output_dir / "similarity_heatmap.png"
        plt.savefig(path, dpi=dpi)
        logger.info(f"Saved heatmap to {path}")

    if show:
        plt.show()
    plt.close()
