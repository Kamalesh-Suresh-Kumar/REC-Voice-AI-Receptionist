from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# visualize_results.py
#     ↓
# evaluation
#     ↓
# app
#     ↓
# backend
#     ↓
# PROJECT ROOT
#
# parents[3] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Evaluation results are stored at:
# REC-Voice-AI-Receptionist/evaluation_results

EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"

CSV_FILE = EVALUATION_DIR / "rag_evaluation_results.csv"

OUTPUT_DIR = EVALUATION_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CSV
# ============================================================

def load_results() -> pd.DataFrame:
    """Load the RAG evaluation results CSV."""

    print()
    print("=" * 70)
    print("REC AI RECEPTIONIST - VISUAL EVALUATION")
    print("=" * 70)

    print()
    print("[CSV] Reading:")
    print(CSV_FILE)

    if not CSV_FILE.exists():
        print()
        print("[ERROR] Evaluation CSV was not found.")
        print()
        print("Expected:")
        print(CSV_FILE)
        print()

        raise FileNotFoundError(
            f"Evaluation CSV not found: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    if df.empty:
        raise ValueError(
            "The evaluation CSV is empty."
        )

    print()
    print(
        f"[CSV] Records loaded: {len(df)}"
    )

    print()
    print("[CSV] Columns:")

    for column in df.columns:
        print(f"  - {column}")

    return df


# ============================================================
# CONVERT YES / NO TO 1 / 0
# ============================================================

def yes_to_int(
    series: pd.Series,
) -> pd.Series:
    """
    Convert YES / NO values into 1 / 0.

    YES -> 1
    NO  -> 0
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("YES")
        .astype(int)
    )


# ============================================================
# CALCULATE OVERALL METRICS
# ============================================================

def calculate_metrics(
    df: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Calculate retrieval, answer and refusal accuracy."""

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "category",
        "retrieval_correct",
        "answer_correct",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Normalize category
    # --------------------------------------------------------

    category_series = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Normal questions
    # --------------------------------------------------------

    normal_df = df[
        category_series != "negative"
    ].copy()

    # --------------------------------------------------------
    # Negative / out-of-domain questions
    # --------------------------------------------------------

    negative_df = df[
        category_series == "negative"
    ].copy()

    # --------------------------------------------------------
    # Retrieval accuracy
    # --------------------------------------------------------

    if not normal_df.empty:

        retrieval_values = yes_to_int(
            normal_df["retrieval_correct"]
        )

        retrieval_accuracy = float(
            retrieval_values.mean() * 100.0
        )

    else:

        retrieval_accuracy = 0.0

    # --------------------------------------------------------
    # Answer accuracy
    # --------------------------------------------------------

    if not normal_df.empty:

        answer_values = yes_to_int(
            normal_df["answer_correct"]
        )

        answer_accuracy = float(
            answer_values.mean() * 100.0
        )

    else:

        answer_accuracy = 0.0

    # --------------------------------------------------------
    # Refusal accuracy
    # --------------------------------------------------------

    if not negative_df.empty:

        refusal_values = yes_to_int(
            negative_df["answer_correct"]
        )

        refusal_accuracy = float(
            refusal_values.mean() * 100.0
        )

    else:

        refusal_accuracy = 0.0

    return (
        retrieval_accuracy,
        answer_accuracy,
        refusal_accuracy,
        normal_df,
        negative_df,
    )


# ============================================================
# CHART 1
# OVERALL ACCURACY
# ============================================================

def create_overall_chart(
    retrieval_accuracy: float,
    answer_accuracy: float,
    refusal_accuracy: float,
) -> Path:
    """Create and save overall RAG performance chart."""

    labels: list[str] = [
        "Retrieval\nHit@5",
        "Answer\nAccuracy",
        "Refusal\nAccuracy",
    ]

    values: list[float] = [
        float(retrieval_accuracy),
        float(answer_accuracy),
        float(refusal_accuracy),
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "REC AI Receptionist - Overall RAG Performance",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # --------------------------------------------------------
    # Percentage labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values,
    ):
        bar_x = float(bar.get_x())
        bar_width = float(bar.get_width())
        numeric_value = float(value)

        label_y = min(
            numeric_value + 2.0,
            98.0,
        )

        ax.text(
            bar_x + (bar_width / 2.0),
            label_y,
            f"{numeric_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "overall_accuracy.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output_file}"
    )

    return output_file


# ============================================================
# CHART 2
# QUESTION-WISE PERFORMANCE
# ============================================================

def create_question_chart(
    normal_df: pd.DataFrame,
) -> Path | None:
    """Create and save question-wise performance chart."""

    if normal_df.empty:

        print(
            "[INFO] No normal questions available "
            "for question-wise chart."
        )

        return None

    # --------------------------------------------------------
    # Question IDs
    # --------------------------------------------------------

    if "id" in normal_df.columns:

        question_ids: list[str] = [
            str(value)
            for value in normal_df["id"].tolist()
        ]

    else:

        question_ids = [
            str(index + 1)
            for index in range(
                len(normal_df)
            )
        ]

    # --------------------------------------------------------
    # Retrieval values
    # --------------------------------------------------------

    retrieval_series = yes_to_int(
        normal_df["retrieval_correct"]
    )

    retrieval_values: list[float] = [
        float(value) * 100.0
        for value in retrieval_series.tolist()
    ]

    # --------------------------------------------------------
    # Answer values
    # --------------------------------------------------------

    answer_series = yes_to_int(
        normal_df["answer_correct"]
    )

    answer_values: list[float] = [
        float(value) * 100.0
        for value in answer_series.tolist()
    ]

    # --------------------------------------------------------
    # X positions
    # --------------------------------------------------------

    positions: list[int] = list(
        range(
            len(question_ids)
        )
    )

    width = 0.35

    retrieval_positions: list[float] = [
        float(position) - (width / 2.0)
        for position in positions
    ]

    answer_positions: list[float] = [
        float(position) + (width / 2.0)
        for position in positions
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.bar(
        retrieval_positions,
        retrieval_values,
        width=width,
        label="Retrieval",
    )

    ax.bar(
        answer_positions,
        answer_values,
        width=width,
        label="Answer",
    )

    ax.set_title(
        "Normal Query Evaluation Performance",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Test Question ID",
        fontsize=12,
    )

    ax.set_ylabel(
        "Correctness (%)",
        fontsize=12,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        question_ids
    )

    ax.set_ylim(
        0.0,
        110.0,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "question_wise_accuracy.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output_file}"
    )

    return output_file


# ============================================================
# CHART 3
# CATEGORY-WISE ACCURACY
# ============================================================

def create_category_chart(
    normal_df: pd.DataFrame,
) -> Path | None:
    """Create and save knowledge-category accuracy chart."""

    if normal_df.empty:

        print(
            "[INFO] No normal questions available "
            "for category chart."
        )

        return None

    # --------------------------------------------------------
    # Copy dataframe
    # --------------------------------------------------------

    category_df = normal_df.copy()

    # --------------------------------------------------------
    # Convert answer correctness to numeric
    # --------------------------------------------------------

    category_df["correct_numeric"] = yes_to_int(
        category_df["answer_correct"]
    )

    # --------------------------------------------------------
    # Calculate category accuracy
    # --------------------------------------------------------

    category_accuracy = (
        category_df
        .groupby("category")["correct_numeric"]
        .mean()
        .mul(100.0)
    )

    # --------------------------------------------------------
    # Explicitly convert pandas values to Python types
    #
    # This avoids the Pylance object/float warnings.
    # --------------------------------------------------------

    categories: list[str] = [
        str(value)
        for value in category_accuracy.index.tolist()
    ]

    values: list[float] = [
        float(value)
        for value in category_accuracy.tolist()
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        categories,
        values,
    )

    ax.set_title(
        "Answer Accuracy by Knowledge Category",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Knowledge Category",
        fontsize=12,
    )

    ax.set_ylabel(
        "Answer Accuracy (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # --------------------------------------------------------
    # Percentage labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values,
    ):
        bar_x = float(bar.get_x())
        bar_width = float(bar.get_width())
        numeric_value = float(value)

        label_y = min(
            numeric_value + 2.0,
            98.0,
        )

        ax.text(
            bar_x + (bar_width / 2.0),
            label_y,
            f"{numeric_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Rotate category names if necessary
    # --------------------------------------------------------

    plt.xticks(
        rotation=20,
        ha="right",
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "category_accuracy.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output_file}"
    )

    return output_file


# ============================================================
# SAVE METRICS CSV
# ============================================================

def save_metrics_csv(
    df: pd.DataFrame,
    normal_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    retrieval_accuracy: float,
    answer_accuracy: float,
    refusal_accuracy: float,
) -> Path:
    """
    Save the final evaluation metrics as a small CSV.

    This gives you a clean numerical record in addition
    to the graphical output.
    """

    output_file = (
        OUTPUT_DIR
        / "evaluation_metrics.csv"
    )

    metrics = pd.DataFrame(
        [
            {
                "metric": "Retrieval Hit@5",
                "accuracy_percent": float(
                    retrieval_accuracy
                ),
                "sample_count": len(normal_df),
            },
            {
                "metric": "Answer Accuracy",
                "accuracy_percent": float(
                    answer_accuracy
                ),
                "sample_count": len(normal_df),
            },
            {
                "metric": "Refusal Accuracy",
                "accuracy_percent": float(
                    refusal_accuracy
                ),
                "sample_count": len(negative_df),
            },
            {
                "metric": "Overall Test Questions",
                "accuracy_percent": None,
                "sample_count": len(df),
            },
        ]
    )

    metrics.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[SAVED] {output_file}"
    )

    return output_file


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_summary(
    df: pd.DataFrame,
    retrieval_accuracy: float,
    answer_accuracy: float,
    refusal_accuracy: float,
) -> Path:
    """Save a human-readable evaluation summary."""

    output_file = (
        OUTPUT_DIR
        / "evaluation_summary.txt"
    )

    category_series = (
        df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    normal_count = int(
        (
            category_series != "negative"
        ).sum()
    )

    negative_count = int(
        (
            category_series == "negative"
        ).sum()
    )

    lines: list[str] = [
        "REC AI RECEPTIONIST",
        "RAG EVALUATION SUMMARY",
        "=" * 50,
        "",
        f"Total test questions : {len(df)}",
        f"Normal questions     : {normal_count}",
        f"Negative questions   : {negative_count}",
        "",
        f"Retrieval Hit@5      : {retrieval_accuracy:.2f}%",
        f"Answer Accuracy      : {answer_accuracy:.2f}%",
        f"Refusal Accuracy     : {refusal_accuracy:.2f}%",
        "",
        "Evaluation source:",
        str(CSV_FILE),
        "",
        "Generated visualizations:",
        "1. overall_accuracy.png",
        "2. question_wise_accuracy.png",
        "3. category_accuracy.png",
        "",
        "Generated numerical result:",
        "4. evaluation_metrics.csv",
    ]

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[SAVED] {output_file}"
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete visual evaluation."""

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    df = load_results()

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    (
        retrieval_accuracy,
        answer_accuracy,
        refusal_accuracy,
        normal_df,
        negative_df,
    ) = calculate_metrics(df)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Total Questions    : {len(df)}"
    )

    print(
        f"Normal Questions   : {len(normal_df)}"
    )

    print(
        f"Negative Questions : {len(negative_df)}"
    )

    print()

    print(
        f"Retrieval Hit@5    : "
        f"{retrieval_accuracy:.2f}%"
    )

    print(
        f"Answer Accuracy    : "
        f"{answer_accuracy:.2f}%"
    )

    print(
        f"Refusal Accuracy   : "
        f"{refusal_accuracy:.2f}%"
    )

    # ========================================================
    # GENERATE CHARTS
    # ========================================================

    print()
    print(
        "[VISUALIZATION] Creating charts..."
    )

    create_overall_chart(
        retrieval_accuracy,
        answer_accuracy,
        refusal_accuracy,
    )

    create_question_chart(
        normal_df
    )

    create_category_chart(
        normal_df
    )

    # ========================================================
    # SAVE METRICS CSV
    # ========================================================

    save_metrics_csv(
        df,
        normal_df,
        negative_df,
        retrieval_accuracy,
        answer_accuracy,
        refusal_accuracy,
    )

    # ========================================================
    # SAVE TEXT SUMMARY
    # ========================================================

    save_summary(
        df,
        retrieval_accuracy,
        answer_accuracy,
        refusal_accuracy,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("VISUAL EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Saved inside:")

    print(
        OUTPUT_DIR
    )

    print()
    print("Files:")

    print(
        "  ✓ overall_accuracy.png"
    )

    print(
        "  ✓ question_wise_accuracy.png"
    )

    print(
        "  ✓ category_accuracy.png"
    )

    print(
        "  ✓ evaluation_metrics.csv"
    )

    print(
        "  ✓ evaluation_summary.txt"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()