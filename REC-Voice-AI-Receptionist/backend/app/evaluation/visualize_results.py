from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# visualize_results.py
#       ↓
# evaluation
#       ↓
# app
#       ↓
# backend
#       ↓
# project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = (
    PROJECT_ROOT / "evaluation_results"
)

# ------------------------------------------------------------
# CSV FILES
# ------------------------------------------------------------

RAG_CSV = (
    EVALUATION_DIR
    / "evaluation_metrics.csv"
)

RAGAS_CSV = (
    EVALUATION_DIR
    / "ragas_evaluation_results.csv"
)

ARES_CSV = (
    EVALUATION_DIR
    / "ares_evaluation_results.csv"
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = EVALUATION_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DIRECT RAG RESULTS
# ============================================================

def load_rag_results() -> pd.DataFrame:
    """Load direct RAG evaluation metrics."""

    print()
    print("=" * 70)
    print("LOADING DIRECT RAG RESULTS")
    print("=" * 70)

    print()
    print("[CSV]")
    print(RAG_CSV)

    if not RAG_CSV.exists():

        raise FileNotFoundError(
            f"Direct RAG CSV not found:\n{RAG_CSV}"
        )

    df = pd.read_csv(
        RAG_CSV
    )

    if df.empty:

        raise ValueError(
            "Direct RAG CSV is empty."
        )

    print()
    print(
        f"[OK] Records loaded: {len(df)}"
    )

    return df


# ============================================================
# LOAD RAGAS RESULTS
# ============================================================

def load_ragas_results() -> pd.DataFrame:
    """Load RAGAS evaluation results."""

    print()
    print("=" * 70)
    print("LOADING RAGAS RESULTS")
    print("=" * 70)

    print()
    print("[CSV]")
    print(RAGAS_CSV)

    if not RAGAS_CSV.exists():

        raise FileNotFoundError(
            f"RAGAS CSV not found:\n{RAGAS_CSV}"
        )

    df = pd.read_csv(
        RAGAS_CSV
    )

    if df.empty:

        raise ValueError(
            "RAGAS CSV is empty."
        )

    print()
    print(
        f"[OK] Records loaded: {len(df)}"
    )

    return df


# ============================================================
# LOAD ARES RESULTS
# ============================================================

def load_ares_results() -> pd.DataFrame:
    """Load ARES-inspired evaluation results."""

    print()
    print("=" * 70)
    print("LOADING ARES-INSPIRED RESULTS")
    print("=" * 70)

    print()
    print("[CSV]")
    print(ARES_CSV)

    if not ARES_CSV.exists():

        raise FileNotFoundError(
            f"ARES CSV not found:\n{ARES_CSV}"
        )

    df = pd.read_csv(
        ARES_CSV
    )

    if df.empty:

        raise ValueError(
            "ARES CSV is empty."
        )

    print()
    print(
        f"[OK] Records loaded: {len(df)}"
    )

    return df


# ============================================================
# GET VALUE FROM DIRECT RAG CSV
# ============================================================

def get_rag_value(
    df: pd.DataFrame,
    metric_name: str,
) -> float:
    """Get one metric from evaluation_metrics.csv."""

    row = df[
        df["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
        == metric_name.lower()
    ]

    if row.empty:

        raise ValueError(
            f"Metric not found in RAG CSV: "
            f"{metric_name}"
        )

    return float(
        row.iloc[0]["accuracy_percent"]
    )


# ============================================================
# CREATE DIRECT RAG GRAPH
# ============================================================

def create_direct_rag_graph(
    rag_df: pd.DataFrame,
) -> Path:
    """Create the final direct RAG evaluation graph."""

    # --------------------------------------------------------
    # Read values directly from CSV
    # --------------------------------------------------------

    retrieval = get_rag_value(
        rag_df,
        "Retrieval Hit@5",
    )

    answer = get_rag_value(
        rag_df,
        "Answer Accuracy",
    )

    refusal = get_rag_value(
        rag_df,
        "Refusal Accuracy",
    )

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    print()
    print(
        "[DIRECT RAG RESULTS]"
    )

    print(
        f"Retrieval Hit@5       : "
        f"{retrieval:.2f}%"
    )

    print(
        f"Answer Accuracy       : "
        f"{answer:.2f}%"
    )

    print(
        f"Negative Refusal     : "
        f"{refusal:.2f}%"
    )

    # --------------------------------------------------------
    # Chart data
    # --------------------------------------------------------

    labels = [
        "Retrieval\nHit@5",
        "Answer\nAccuracy",
        "Negative Refusal\nAccuracy",
    ]

    values = [
        retrieval,
        answer,
        refusal,
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        values,
        color=[
            "#4361EE",
            "#06D6A0",
            "#F72585",
        ],
        width=0.60,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "Direct RAG Evaluation Performance",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        110,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Remove unnecessary borders
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        OUTPUT_DIR
        / "overall_accuracy.png"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output}"
    )

    return output


# ============================================================
# CREATE RAGAS GRAPH
# ============================================================

def create_ragas_graph(
    ragas_df: pd.DataFrame,
) -> Path:
    """Create final RAGAS evaluation graph."""

    # --------------------------------------------------------
    # Calculate directly from CSV
    # --------------------------------------------------------

    faithfulness = (
        ragas_df[
            "faithfulness"
        ].mean()
        * 100
    )

    relevancy = (
        ragas_df[
            "answer_relevancy"
        ].mean()
        * 100
    )

    precision = (
        ragas_df[
            "context_precision"
        ].mean()
        * 100
    )

    recall = (
        ragas_df[
            "context_recall"
        ].mean()
        * 100
    )

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    print()
    print(
        "[RAGAS RESULTS]"
    )

    print(
        f"Faithfulness          : "
        f"{faithfulness:.2f}%"
    )

    print(
        f"Answer Relevancy      : "
        f"{relevancy:.2f}%"
    )

    print(
        f"Context Precision     : "
        f"{precision:.2f}%"
    )

    print(
        f"Context Recall        : "
        f"{recall:.2f}%"
    )

    # --------------------------------------------------------
    # Chart data
    # --------------------------------------------------------

    labels = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
    ]

    values = [
        faithfulness,
        relevancy,
        precision,
        recall,
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars = ax.bar(
        labels,
        values,
        color=[
            "#7B2CBF",
            "#F72585",
            "#4361EE",
            "#06D6A0",
        ],
        width=0.60,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "RAGAS Evaluation Performance",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_ylabel(
        "Score (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        110,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Clean borders
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        OUTPUT_DIR
        / "ragas_overall_metrics.png"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output}"
    )

    return output


# ============================================================
# CREATE ARES GRAPH
# ============================================================

def create_ares_graph(
    ares_df: pd.DataFrame,
) -> Path:
    """Create final ARES-inspired evaluation graph."""

    # --------------------------------------------------------
    # Calculate directly from CSV
    # --------------------------------------------------------

    context_relevance = (
        ares_df[
            "context_relevance"
        ].mean()
        * 100
    )

    answer_faithfulness = (
        ares_df[
            "answer_faithfulness"
        ].mean()
        * 100
    )

    answer_relevance = (
        ares_df[
            "answer_relevance"
        ].mean()
        * 100
    )

    # --------------------------------------------------------
    # Display values
    # --------------------------------------------------------

    print()
    print(
        "[ARES-INSPIRED RESULTS]"
    )

    print(
        f"Context Relevance     : "
        f"{context_relevance:.2f}%"
    )

    print(
        f"Answer Faithfulness   : "
        f"{answer_faithfulness:.2f}%"
    )

    print(
        f"Answer Relevance      : "
        f"{answer_relevance:.2f}%"
    )

    # --------------------------------------------------------
    # Chart data
    # --------------------------------------------------------

    labels = [
        "Context\nRelevance",
        "Answer\nFaithfulness",
        "Answer\nRelevance",
    ]

    values = [
        context_relevance,
        answer_faithfulness,
        answer_relevance,
    ]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        values,
        color=[
            "#118AB2",
            "#8338EC",
            "#FB5607",
        ],
        width=0.60,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "ARES-Inspired Evaluation Performance",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_ylabel(
        "Score (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        110,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # Clean borders
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        OUTPUT_DIR
        / "ares_overall_metrics.png"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {output}"
    )

    return output


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

def save_summary(
    retrieval: float,
    answer: float,
    refusal: float,
    faithfulness: float,
    relevancy: float,
    precision: float,
    recall: float,
    context_relevance: float,
    answer_faithfulness: float,
    answer_relevance: float,
):
    """Save final visualization summary."""

    output = (
        OUTPUT_DIR
        / "visualization_summary.txt"
    )

    lines = [
        "REC VOICE AI RECEPTIONIST",
        "FINAL EVALUATION VISUALIZATION SUMMARY",
        "=" * 60,
        "",
        "FINAL MODEL",
        "-" * 60,
        "Gemma 3:4B",
        "",
        "DIRECT RAG EVALUATION",
        "-" * 60,
        f"Retrieval Hit@5       : {retrieval:.2f}%",
        f"Answer Accuracy       : {answer:.2f}%",
        f"Negative Refusal      : {refusal:.2f}%",
        "",
        "RAGAS EVALUATION",
        "-" * 60,
        f"Faithfulness          : {faithfulness:.2f}%",
        f"Answer Relevancy      : {relevancy:.2f}%",
        f"Context Precision     : {precision:.2f}%",
        f"Context Recall        : {recall:.2f}%",
        "",
        "ARES-INSPIRED EVALUATION",
        "-" * 60,
        f"Context Relevance     : {context_relevance:.2f}%",
        f"Answer Faithfulness   : {answer_faithfulness:.2f}%",
        f"Answer Relevance      : {answer_relevance:.2f}%",
        "",
        "GENERATED GRAPHS",
        "-" * 60,
        "1. overall_accuracy.png",
        "2. ragas_overall_metrics.png",
        "3. ares_overall_metrics.png",
    ]

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[SAVED] {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Generate final evaluation graphs."""

    print()
    print("=" * 70)
    print("REC VOICE AI RECEPTIONIST")
    print("FINAL EVALUATION VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------

    rag_df = load_rag_results()

    ragas_df = load_ragas_results()

    ares_df = load_ares_results()

    # --------------------------------------------------------
    # Calculate values
    # --------------------------------------------------------

    retrieval = get_rag_value(
        rag_df,
        "Retrieval Hit@5",
    )

    answer = get_rag_value(
        rag_df,
        "Answer Accuracy",
    )

    refusal = get_rag_value(
        rag_df,
        "Refusal Accuracy",
    )

    faithfulness = (
        ragas_df[
            "faithfulness"
        ].mean()
        * 100
    )

    relevancy = (
        ragas_df[
            "answer_relevancy"
        ].mean()
        * 100
    )

    precision = (
        ragas_df[
            "context_precision"
        ].mean()
        * 100
    )

    recall = (
        ragas_df[
            "context_recall"
        ].mean()
        * 100
    )

    context_relevance = (
        ares_df[
            "context_relevance"
        ].mean()
        * 100
    )

    answer_faithfulness = (
        ares_df[
            "answer_faithfulness"
        ].mean()
        * 100
    )

    answer_relevance = (
        ares_df[
            "answer_relevance"
        ].mean()
        * 100
    )

    # --------------------------------------------------------
    # Create graphs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING FINAL GRAPHS")
    print("=" * 70)

    create_direct_rag_graph(
        rag_df
    )

    create_ragas_graph(
        ragas_df
    )

    create_ares_graph(
        ares_df
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        retrieval,
        answer,
        refusal,
        faithfulness,
        relevancy,
        precision,
        recall,
        context_relevance,
        answer_faithfulness,
        answer_relevance,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Final graphs:"
    )

    print(
        "  ✓ overall_accuracy.png"
    )

    print(
        "  ✓ ragas_overall_metrics.png"
    )

    print(
        "  ✓ ares_overall_metrics.png"
    )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()