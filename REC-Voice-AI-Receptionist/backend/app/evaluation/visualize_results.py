# from __future__ import annotations

# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# # backend/app/evaluation/visualize_results.py
# #                         |
# #                         v
# #                    Project root

# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
# OUTPUT_DIR = EVALUATION_DIR

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # CSV FILES
# # ============================================================

# RAG_CSV = EVALUATION_DIR / "evaluation_metrics.csv"

# RAGAS_CSV = (
#     EVALUATION_DIR
#     / "ragas_evaluation_results.csv"
# )

# ARES_CSV = EVALUATION_DIR / "ares_evaluation_results.csv"

# STT_CSV = EVALUATION_DIR / "stt_quality_summary.csv"


# # ============================================================
# # COMMON HELPERS
# # ============================================================

# def load_csv(path: Path, description: str) -> pd.DataFrame:
#     """Load a CSV and validate that it exists and is not empty."""

#     print(f"\n[LOAD] {description}")
#     print(f"[CSV]  {path}")

#     if not path.exists():
#         raise FileNotFoundError(
#             f"{description} CSV not found:\n{path}"
#         )

#     df = pd.read_csv(path)

#     if df.empty:
#         raise ValueError(
#             f"{description} CSV is empty:\n{path}"
#         )

#     print(f"[OK] Rows loaded: {len(df)}")
#     print(f"[OK] Columns: {df.columns.tolist()}")

#     return df


# def require_columns(
#     df: pd.DataFrame,
#     required: list[str],
#     description: str,
# ) -> None:
#     """Raise a clear error if required columns are missing."""

#     missing = [
#         column for column in required
#         if column not in df.columns
#     ]

#     if missing:
#         raise ValueError(
#             f"{description} CSV is missing columns: {missing}\n"
#             f"Available columns: {df.columns.tolist()}"
#         )


# def numeric_series(
#     df: pd.DataFrame,
#     column: str,
#     description: str,
# ) -> pd.Series:
#     """Convert a column to numeric values and reject empty data."""

#     require_columns(df, [column], description)

#     values = pd.to_numeric(
#         df[column], errors="coerce"
#     ).dropna()

#     if values.empty:
#         raise ValueError(
#             f"No valid numeric values found in '{column}' "
#             f"for {description}."
#         )

#     return values


# def metric_mean_percent(
#     df: pd.DataFrame,
#     column: str,
#     description: str,
# ) -> float:
#     """Calculate a mean score stored on a 0-to-1 scale."""

#     values = numeric_series(df, column, description)

#     if ((values < 0) | (values > 1)).any():
#         raise ValueError(
#             f"Expected '{column}' in {description} to contain "
#             "scores between 0 and 1."
#         )

#     return float(values.mean() * 100)


# def get_rag_value(
#     df: pd.DataFrame,
#     metric_name: str,
# ) -> float:
#     """Read a direct RAG metric from evaluation_metrics.csv."""

#     require_columns(
#         df,
#         ["metric", "accuracy_percent"],
#         "Direct RAG",
#     )

#     normalized = (
#         df["metric"].astype(str).str.strip().str.casefold()
#     )

#     aliases = {
#         "refusal accuracy": {
#             "refusal accuracy",
#             "negative refusal accuracy",
#             "negative refusal",
#         }
#     }

#     requested = metric_name.strip().casefold()
#     accepted_names = aliases.get(requested, {requested})

#     rows = df[normalized.isin(accepted_names)]

#     if rows.empty:
#         raise ValueError(
#             f"Metric '{metric_name}' not found in {RAG_CSV.name}.\n"
#             f"Available metrics: {df['metric'].tolist()}"
#         )

#     values = pd.to_numeric(
#         rows["accuracy_percent"],
#         errors="coerce",
#     ).dropna()

#     if values.empty:
#         raise ValueError(
#             f"Metric '{metric_name}' has no valid numeric score."
#         )

#     return float(values.iloc[0])


# def style_axes(ax) -> None:
#     """Apply consistent styling to charts."""

#     ax.grid(axis="y", alpha=0.25, linestyle="--")
#     ax.set_axisbelow(True)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)


# def add_bar_labels(
#     ax,
#     bars,
#     suffix: str = "%",
#     decimals: int = 2,
#     offset: float = 1.0,
# ) -> None:
#     """Add values above each bar."""

#     for bar in bars:
#         value = bar.get_height()

#         if not np.isfinite(value):
#             continue

#         ax.annotate(
#             f"{value:.{decimals}f}{suffix}",
#             xy=(
#                 bar.get_x() + bar.get_width() / 2,
#                 value,
#             ),
#             xytext=(0, 4),
#             textcoords="offset points",
#             ha="center",
#             va="bottom",
#             fontsize=9,
#         )


# def save_figure(fig, filename: str) -> Path:
#     """Save a figure at publication-quality resolution."""

#     output = OUTPUT_DIR / filename

#     fig.tight_layout()
#     fig.savefig(
#         output,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor="white",
#     )
#     plt.close(fig)

#     print(f"[SAVED] {output}")

#     return output


# # ============================================================
# # LOAD EVALUATION DATASETS
# # ============================================================

# def load_rag_results() -> pd.DataFrame:
#     df = load_csv(RAG_CSV, "Direct RAG")

#     require_columns(
#         df,
#         ["metric", "accuracy_percent", "sample_count"],
#         "Direct RAG",
#     )

#     return df


# def load_ragas_results() -> pd.DataFrame:
#     df = load_csv(RAGAS_CSV, "RAGAS")

#     require_columns(
#         df,
#         [
#             "faithfulness",
#             "answer_relevancy",
#             "context_precision",
#             "context_recall",
#         ],
#         "RAGAS",
#     )

#     return df


# def load_ares_results() -> pd.DataFrame:
#     df = load_csv(ARES_CSV, "ARES-inspired")

#     require_columns(
#         df,
#         [
#             "context_relevance",
#             "answer_faithfulness",
#             "answer_relevance",
#         ],
#         "ARES-inspired",
#     )

#     return df


# def load_stt_results() -> pd.DataFrame:
#     df = load_csv(STT_CSV, "Speech recognition")

#     require_columns(
#         df,
#         [
#             "condition",
#             "test_count",
#             "average_wer_percent",
#             "average_cer_percent",
#         ],
#         "Speech recognition",
#     )

#     return df


# # ============================================================
# # 1. DIRECT RAG GRAPH
# # ============================================================

# def create_direct_rag_graph(
#     rag_df: pd.DataFrame,
# ) -> dict:
#     """Create the direct RAG evaluation chart."""

#     metrics = {
#         "Retrieval Hit@5": get_rag_value(
#             rag_df, "Retrieval Hit@5"
#         ),
#         "Answer Accuracy": get_rag_value(
#             rag_df, "Answer Accuracy"
#         ),
#         "Negative Refusal Accuracy": get_rag_value(
#             rag_df, "Refusal Accuracy"
#         ),
#     }

#     print("\n[DIRECT RAG RESULTS]")

#     for name, value in metrics.items():
#         print(f"{name:<28}: {value:.2f}%")

#     labels = [
#         "Retrieval\nHit@5",
#         "Answer\nAccuracy",
#         "Negative Refusal\nAccuracy",
#     ]

#     fig, ax = plt.subplots(figsize=(9, 5.5))

#     bars = ax.bar(
#         labels,
#         list(metrics.values()),
#         color=["#4361EE", "#06A77D", "#D94F9D"],
#         width=0.60,
#     )

#     ax.set_title(
#         "Direct RAG Evaluation Performance",
#         fontsize=15,
#         fontweight="bold",
#         pad=14,
#     )
#     ax.set_ylabel("Accuracy (%)")
#     ax.set_ylim(0, 110)

#     style_axes(ax)
#     add_bar_labels(ax, bars)

#     save_figure(fig, "rag_overall_accuracy.png")

#     return metrics


# # ============================================================
# # 2. RAGAS GRAPH
# # ============================================================

# def create_ragas_graph(
#     ragas_df: pd.DataFrame,
# ) -> dict:
#     """Create the RAGAS evaluation chart."""

#     columns = {
#         "Faithfulness": "faithfulness",
#         "Answer Relevancy": "answer_relevancy",
#         "Context Precision": "context_precision",
#         "Context Recall": "context_recall",
#     }

#     metrics = {
#         label: metric_mean_percent(
#             ragas_df, column, "RAGAS"
#         )
#         for label, column in columns.items()
#     }

#     print("\n[RAGAS RESULTS]")

#     for name, value in metrics.items():
#         print(f"{name:<25}: {value:.2f}%")

#     fig, ax = plt.subplots(figsize=(10, 5.5))

#     bars = ax.bar(
#         list(metrics.keys()),
#         list(metrics.values()),
#         color=["#9B5DE5", "#D65DB1", "#5773D8", "#0B8F69"],
#         width=0.62,
#     )

#     ax.set_title(
#         "RAGAS Evaluation Performance",
#         fontsize=15,
#         fontweight="bold",
#         pad=14,
#     )
#     ax.set_ylabel("Score (%)")
#     ax.set_ylim(0, 110)

#     style_axes(ax)
#     add_bar_labels(ax, bars)

#     save_figure(fig, "ragas_overall_metrics.png")

#     return metrics


# # ============================================================
# # 3. ARES-INSPIRED GRAPH
# # ============================================================

# def create_ares_graph(
#     ares_df: pd.DataFrame,
# ) -> dict:
#     """Create the ARES-inspired evaluation chart."""

#     columns = {
#         "Context Relevance": "context_relevance",
#         "Answer Faithfulness": "answer_faithfulness",
#         "Answer Relevance": "answer_relevance",
#     }

#     metrics = {
#         label: metric_mean_percent(
#             ares_df, column, "ARES-inspired"
#         )
#         for label, column in columns.items()
#     }

#     print("\n[ARES-INSPIRED RESULTS]")

#     for name, value in metrics.items():
#         print(f"{name:<25}: {value:.2f}%")

#     fig, ax = plt.subplots(figsize=(9, 5.5))

#     bars = ax.bar(
#         list(metrics.keys()),
#         list(metrics.values()),
#         color=["#3288A8", "#9C70D1", "#E66A32"],
#         width=0.60,
#     )

#     ax.set_title(
#         "ARES-Inspired Evaluation Performance",
#         fontsize=15,
#         fontweight="bold",
#         pad=14,
#     )
#     ax.set_ylabel("Score (%)")
#     ax.set_ylim(0, 110)

#     style_axes(ax)
#     add_bar_labels(ax, bars)

#     save_figure(fig, "ares_overall_metrics.png")

#     return metrics


# # ============================================================
# # 4. SPEECH RECOGNITION WER/CER GRAPH
# # ============================================================

# def create_speech_recognition_graph(
#     stt_df: pd.DataFrame,
# ) -> dict:
#     """Create a WER/CER chart using stt_quality_summary.csv."""

#     df = stt_df.copy()

#     df["condition"] = (
#         df["condition"].astype(str).str.strip().str.casefold()
#     )

#     df["average_wer_percent"] = pd.to_numeric(
#         df["average_wer_percent"], errors="coerce"
#     )

#     df["average_cer_percent"] = pd.to_numeric(
#         df["average_cer_percent"], errors="coerce"
#     )

#     df["test_count"] = pd.to_numeric(
#         df["test_count"], errors="coerce"
#     )

#     df = df.dropna(
#         subset=[
#             "condition",
#             "average_wer_percent",
#             "average_cer_percent",
#         ]
#     )

#     preferred_order = [
#         "clean",
#         "noise",
#         "noisy",
#         "telephone",
#         "telephone-filtered",
#         "accent",
#     ]

#     available = df["condition"].tolist()

#     ordered_conditions = [
#         condition
#         for condition in preferred_order
#         if condition in available
#     ]

#     ordered_conditions.extend(
#         condition
#         for condition in available
#         if condition not in ordered_conditions
#     )

#     df = (
#         df.drop_duplicates(
#             subset=["condition"], keep="last"
#         )
#         .set_index("condition")
#         .reindex(ordered_conditions)
#         .dropna(
#             subset=[
#                 "average_wer_percent",
#                 "average_cer_percent",
#             ]
#         )
#         .reset_index()
#     )

#     if df.empty:
#         raise ValueError(
#             f"No valid speech-recognition rows found in "
#             f"{STT_CSV.name}."
#         )

#     label_map = {
#         "clean": "Clean",
#         "noise": "Noisy",
#         "noisy": "Noisy",
#         "telephone": "Telephone-filtered",
#         "telephone-filtered": "Telephone-filtered",
#         "accent": "Accent",
#     }

#     labels = [
#         label_map.get(condition, condition.title())
#         for condition in df["condition"]
#     ]

#     wer = df["average_wer_percent"].astype(float).to_numpy()
#     cer = df["average_cer_percent"].astype(float).to_numpy()

#     print("\n[SPEECH RECOGNITION RESULTS]")

#     for label, w, c in zip(labels, wer, cer):
#         print(
#             f"{label:<24}: WER {w:.2f}% | CER {c:.2f}%"
#         )

#     x = np.arange(len(labels))
#     width = 0.34

#     fig, ax = plt.subplots(figsize=(9, 5.5))

#     wer_bars = ax.bar(
#         x - width / 2,
#         wer,
#         width,
#         label="WER",
#         color="#4361EE",
#     )

#     cer_bars = ax.bar(
#         x + width / 2,
#         cer,
#         width,
#         label="CER",
#         color="#06A77D",
#     )

#     ax.set_title(
#         "Speech Recognition Performance",
#         fontsize=15,
#         fontweight="bold",
#         pad=14,
#     )
#     ax.set_ylabel("Error Rate (%)")
#     ax.set_xlabel("Audio Condition")
#     ax.set_xticks(x)
#     ax.set_xticklabels([str(label) if label is not None else "" for label in labels])
#     ax.legend(frameon=False)

#     max_value = float(max(np.max(wer), np.max(cer)))
#     ax.set_ylim(0, max(1.0, max_value * 1.35))

#     style_axes(ax)
#     add_bar_labels(ax, wer_bars, decimals=2)
#     add_bar_labels(ax, cer_bars, decimals=2)

#     save_figure(fig, "speech_recognition_metrics.png")

#     return {
#         label: {
#             "WER": float(w),
#             "CER": float(c),
#             "samples": (
#                 int(sample_count)
#                 if pd.notna(sample_count)
#                 else 0
#             ),
#         }
#         for label, w, c, sample_count in zip(
#             labels,
#             wer,
#             cer,
#             df["test_count"],
#         )
#     }


# # ============================================================
# # SAVE FINAL SUMMARY
# # ============================================================

# def save_summary(
#     direct_metrics: dict,
#     ragas_metrics: dict,
#     ares_metrics: dict,
#     stt_metrics: dict,
#     rag_df: pd.DataFrame,
#     ragas_df: pd.DataFrame,
#     ares_df: pd.DataFrame,
# ) -> Path:
#     """Save a summary derived from the evaluation CSV files."""

#     output = OUTPUT_DIR / "visualization_summary.txt"

#     lines = [
#         "REC VOICE AI RECEPTIONIST",
#         "FINAL EVALUATION VISUALIZATION SUMMARY",
#         "=" * 65,
#         "",
#         "DIRECT RAG EVALUATION",
#         "-" * 65,
#     ]

#     for metric, score in direct_metrics.items():
#         lines.append(f"{metric:<32}: {score:.2f}%")

#     if "sample_count" in rag_df.columns:
#         lines.extend(["", "Direct RAG metric sample counts:"])
#         for _, row in rag_df.iterrows():
#             lines.append(
#                 f"{str(row['metric']):<32}: {row['sample_count']}"
#             )

#     lines.extend(["", "RAGAS EVALUATION", "-" * 65])
#     for metric, score in ragas_metrics.items():
#         lines.append(f"{metric:<32}: {score:.2f}%")
#     lines.append(f"Evaluated rows: {len(ragas_df)}")

#     lines.extend(["", "ARES-INSPIRED EVALUATION", "-" * 65])
#     for metric, score in ares_metrics.items():
#         lines.append(f"{metric:<32}: {score:.2f}%")
#     lines.append(f"Evaluated rows: {len(ares_df)}")

#     lines.extend(["", "SPEECH RECOGNITION EVALUATION", "-" * 65])
#     for condition, scores in stt_metrics.items():
#         lines.append(
#             f"{condition:<24}: "
#             f"WER {scores['WER']:.2f}%, "
#             f"CER {scores['CER']:.2f}%, "
#             f"samples {scores['samples']}"
#         )

#     lines.extend([
#         "",
#         "GENERATED GRAPHS",
#         "-" * 65,
#         "1. rag_overall_accuracy.png",
#         "2. ragas_overall_metrics.png",
#         "3. ares_overall_metrics.png",
#         "4. speech_recognition_metrics.png",
#     ])

#     output.write_text("\n".join(lines), encoding="utf-8")
#     print(f"\n[SAVED SUMMARY] {output}")
#     return output



# # ============================================================
# # MAIN
# # ============================================================

# def main() -> None:
#     """Load evaluation CSVs, generate the four retained charts, and save summary."""

#     print("\n" + "=" * 70)
#     print("REC VOICE AI RECEPTIONIST")
#     print("FINAL EVALUATION VISUALIZATION")
#     print("=" * 70)

#     rag_df = load_rag_results()
#     ragas_df = load_ragas_results()
#     ares_df = load_ares_results()
#     stt_df = load_stt_results()

#     direct_metrics = create_direct_rag_graph(rag_df)
#     ragas_metrics = create_ragas_graph(ragas_df)
#     ares_metrics = create_ares_graph(ares_df)
#     stt_metrics = create_speech_recognition_graph(stt_df)

#     save_summary(
#         direct_metrics=direct_metrics,
#         ragas_metrics=ragas_metrics,
#         ares_metrics=ares_metrics,
#         stt_metrics=stt_metrics,
#         rag_df=rag_df,
#         ragas_df=ragas_df,
#         ares_df=ares_df,
#     )

#     print("\n" + "=" * 70)
#     print("VISUALIZATION COMPLETE")
#     print("=" * 70)

#     output_files = [
#         "rag_overall_accuracy.png",
#         "ragas_overall_metrics.png",
#         "ares_overall_metrics.png",
#         "speech_recognition_metrics.png",
#         "visualization_summary.txt",
#     ]

#     for filename in output_files:
#         print(f"[OUTPUT] {OUTPUT_DIR / filename}")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# backend/app/evaluation/visualize_results.py
#                         |
#                         v
#                    Project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR = EVALUATION_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CSV FILES
# ============================================================

RAG_CSV = EVALUATION_DIR / "evaluation_metrics.csv"

RAGAS_CSV = (
    EVALUATION_DIR
    / "ragas_evaluation_results.csv"
)

ARES_CSV = EVALUATION_DIR / "ares_evaluation_results.csv"

STT_CSV = EVALUATION_DIR / "stt_quality_summary.csv"


# ============================================================
# COMMON HELPERS
# ============================================================

def load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV and validate that it exists and is not empty."""

    print(f"\n[LOAD] {description}")
    print(f"[CSV]  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"{description} CSV not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{description} CSV is empty:\n{path}"
        )

    print(f"[OK] Rows loaded: {len(df)}")
    print(f"[OK] Columns: {df.columns.tolist()}")

    return df


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    description: str,
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{description} CSV is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def numeric_series(
    df: pd.DataFrame,
    column: str,
    description: str,
) -> pd.Series:
    """Convert a column to numeric values and reject empty data."""

    require_columns(df, [column], description)

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"No valid numeric values found in '{column}' "
            f"for {description}."
        )

    return values


def metric_mean_percent(
    df: pd.DataFrame,
    column: str,
    description: str,
) -> float:
    """Calculate a mean score stored on a 0-to-1 scale."""

    values = numeric_series(df, column, description)

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"Expected '{column}' in {description} to contain "
            "scores between 0 and 1."
        )

    return float(values.mean() * 100)


def get_rag_value(
    df: pd.DataFrame,
    metric_name: str,
) -> float:
    """Read a precomputed Direct RAG metric from evaluation_metrics.csv."""

    require_columns(
        df,
        ["metric", "accuracy_percent"],
        "Direct RAG",
    )

    normalized = (
        df["metric"].astype(str).str.strip().str.casefold()
    )

    aliases = {
        "refusal accuracy": {
            "refusal accuracy",
            "negative refusal accuracy",
            "negative refusal",
        }
    }

    requested = metric_name.strip().casefold()
    accepted_names = aliases.get(requested, {requested})

    rows = df[normalized.isin(accepted_names)]

    if rows.empty:
        raise ValueError(
            f"Metric '{metric_name}' not found in {RAG_CSV.name}.\n"
            f"Available metrics: {df['metric'].tolist()}"
        )

    values = pd.to_numeric(
        rows["accuracy_percent"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(
            f"Metric '{metric_name}' has no valid numeric score."
        )

    return float(values.iloc[0])


def style_axes(ax) -> None:
    """Apply consistent styling to charts."""

    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(
    ax,
    bars,
    suffix: str = "%",
    decimals: int = 2,
    offset: float = 1.0,
) -> None:
    """Add values above each bar."""

    for bar in bars:
        value = bar.get_height()

        if not np.isfinite(value):
            continue

        ax.annotate(
            f"{value:.{decimals}f}{suffix}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, filename: str) -> Path:
    """Save a figure at publication-quality resolution."""

    output = OUTPUT_DIR / filename

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"[SAVED] {output}")

    return output


# ============================================================
# LOAD EVALUATION DATASETS
# ============================================================

def load_rag_results() -> pd.DataFrame:
    df = load_csv(RAG_CSV, "Direct RAG")

    require_columns(
        df,
        ["metric", "accuracy_percent", "sample_count"],
        "Direct RAG",
    )

    return df


def load_ragas_results() -> pd.DataFrame:
    df = load_csv(RAGAS_CSV, "RAGAS")

    require_columns(
        df,
        [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ],
        "RAGAS",
    )

    return df


def load_ares_results() -> pd.DataFrame:
    df = load_csv(ARES_CSV, "ARES-inspired")

    require_columns(
        df,
        [
            "context_relevance",
            "answer_faithfulness",
            "answer_relevance",
        ],
        "ARES-inspired",
    )

    return df


def load_stt_results() -> pd.DataFrame:
    df = load_csv(STT_CSV, "Speech recognition")

    require_columns(
        df,
        [
            "condition",
            "test_count",
            "average_wer_percent",
            "average_cer_percent",
        ],
        "Speech recognition",
    )

    return df


# ============================================================
# 1. DIRECT RAG GRAPH
# ============================================================

def create_direct_rag_graph(
    rag_df: pd.DataFrame,
) -> dict:
    """Plot precomputed Direct RAG metrics saved by evaluate_rag.py."""

    metrics = {
        "Retrieval Hit@5": get_rag_value(
            rag_df, "Retrieval Hit@5"
        ),
        "Answer Accuracy": get_rag_value(
            rag_df, "Answer Accuracy"
        ),
        "Negative Refusal Accuracy": get_rag_value(
            rag_df, "Refusal Accuracy"
        ),
    }

    print("\n[DIRECT RAG RESULTS]")

    for name, value in metrics.items():
        print(f"{name:<28}: {value:.2f}%")

    labels = [
        "Retrieval\nHit@5",
        "Answer\nAccuracy",
        "Negative Refusal\nAccuracy",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        labels,
        list(metrics.values()),
        color=["#4361EE", "#06A77D", "#D94F9D"],
        width=0.60,
    )

    ax.set_title(
        "Direct RAG Evaluation Performance",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)

    style_axes(ax)
    add_bar_labels(ax, bars)

    save_figure(fig, "rag_overall_accuracy.png")

    return metrics


# ============================================================
# 2. RAGAS GRAPH
# ============================================================

def create_ragas_graph(
    ragas_df: pd.DataFrame,
) -> dict:
    """Create the RAGAS evaluation chart."""

    columns = {
        "Faithfulness": "faithfulness",
        "Answer Relevancy": "answer_relevancy",
        "Context Precision": "context_precision",
        "Context Recall": "context_recall",
    }

    metrics = {
        label: metric_mean_percent(
            ragas_df, column, "RAGAS"
        )
        for label, column in columns.items()
    }

    print("\n[RAGAS RESULTS]")

    for name, value in metrics.items():
        print(f"{name:<25}: {value:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=["#9B5DE5", "#D65DB1", "#5773D8", "#0B8F69"],
        width=0.62,
    )

    ax.set_title(
        "RAGAS Evaluation Performance",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)

    style_axes(ax)
    add_bar_labels(ax, bars)

    save_figure(fig, "ragas_overall_metrics.png")

    return metrics


# ============================================================
# 3. ARES-INSPIRED GRAPH
# ============================================================

def create_ares_graph(
    ares_df: pd.DataFrame,
) -> dict:
    """Create the ARES-inspired evaluation chart."""

    columns = {
        "Context Relevance": "context_relevance",
        "Answer Faithfulness": "answer_faithfulness",
        "Answer Relevance": "answer_relevance",
    }

    metrics = {
        label: metric_mean_percent(
            ares_df, column, "ARES-inspired"
        )
        for label, column in columns.items()
    }

    print("\n[ARES-INSPIRED RESULTS]")

    for name, value in metrics.items():
        print(f"{name:<25}: {value:.2f}%")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=["#3288A8", "#9C70D1", "#E66A32"],
        width=0.60,
    )

    ax.set_title(
        "ARES-Inspired Evaluation Performance",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)

    style_axes(ax)
    add_bar_labels(ax, bars)

    save_figure(fig, "ares_overall_metrics.png")

    return metrics


# ============================================================
# 4. SPEECH RECOGNITION WER/CER GRAPH
# ============================================================

def create_speech_recognition_graph(
    stt_df: pd.DataFrame,
) -> dict:
    """Create a WER/CER chart using stt_quality_summary.csv."""

    df = stt_df.copy()

    df["condition"] = (
        df["condition"].astype(str).str.strip().str.casefold()
    )

    df["average_wer_percent"] = pd.to_numeric(
        df["average_wer_percent"], errors="coerce"
    )

    df["average_cer_percent"] = pd.to_numeric(
        df["average_cer_percent"], errors="coerce"
    )

    df["test_count"] = pd.to_numeric(
        df["test_count"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "condition",
            "average_wer_percent",
            "average_cer_percent",
        ]
    )

    preferred_order = [
        "clean",
        "noise",
        "noisy",
        "telephone",
        "telephone-filtered",
        "accent",
    ]

    available = df["condition"].tolist()

    ordered_conditions = [
        condition
        for condition in preferred_order
        if condition in available
    ]

    ordered_conditions.extend(
        condition
        for condition in available
        if condition not in ordered_conditions
    )

    df = (
        df.drop_duplicates(
            subset=["condition"], keep="last"
        )
        .set_index("condition")
        .reindex(ordered_conditions)
        .dropna(
            subset=[
                "average_wer_percent",
                "average_cer_percent",
            ]
        )
        .reset_index()
    )

    if df.empty:
        raise ValueError(
            f"No valid speech-recognition rows found in "
            f"{STT_CSV.name}."
        )

    label_map = {
        "clean": "Clean",
        "noise": "Noisy",
        "noisy": "Noisy",
        "telephone": "Telephone-filtered",
        "telephone-filtered": "Telephone-filtered",
        "accent": "Accent",
    }

    labels = [
        label_map.get(condition, condition.title())
        for condition in df["condition"]
    ]

    wer = df["average_wer_percent"].astype(float).to_numpy()
    cer = df["average_cer_percent"].astype(float).to_numpy()

    print("\n[SPEECH RECOGNITION RESULTS]")

    for label, w, c in zip(labels, wer, cer):
        print(
            f"{label:<24}: WER {w:.2f}% | CER {c:.2f}%"
        )

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))

    wer_bars = ax.bar(
        x - width / 2,
        wer,
        width,
        label="WER",
        color="#4361EE",
    )

    cer_bars = ax.bar(
        x + width / 2,
        cer,
        width,
        label="CER",
        color="#06A77D",
    )

    ax.set_title(
        "Speech Recognition Performance",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Error Rate (%)")
    ax.set_xlabel("Audio Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) if label is not None else "" for label in labels])
    ax.legend(frameon=False)

    max_value = float(max(np.max(wer), np.max(cer)))
    ax.set_ylim(0, max(1.0, max_value * 1.35))

    style_axes(ax)
    add_bar_labels(ax, wer_bars, decimals=2)
    add_bar_labels(ax, cer_bars, decimals=2)

    save_figure(fig, "speech_recognition_metrics.png")

    return {
        label: {
            "WER": float(w),
            "CER": float(c),
            "samples": (
                int(sample_count)
                if pd.notna(sample_count)
                else 0
            ),
        }
        for label, w, c, sample_count in zip(
            labels,
            wer,
            cer,
            df["test_count"],
        )
    }


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

def save_summary(
    direct_metrics: dict,
    ragas_metrics: dict,
    ares_metrics: dict,
    stt_metrics: dict,
    rag_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    ares_df: pd.DataFrame,
) -> Path:
    """Save a summary derived from the evaluation CSV files."""

    output = OUTPUT_DIR / "visualization_summary.txt"

    lines = [
        "REC VOICE AI RECEPTIONIST",
        "FINAL EVALUATION VISUALIZATION SUMMARY",
        "=" * 65,
        "",
        "DIRECT RAG EVALUATION",
        "-" * 65,
    ]

    for metric, score in direct_metrics.items():
        lines.append(f"{metric:<32}: {score:.2f}%")

    if "sample_count" in rag_df.columns:
        lines.extend(["", "Direct RAG metric sample counts:"])
        for _, row in rag_df.iterrows():
            lines.append(
                f"{str(row['metric']):<32}: {row['sample_count']}"
            )

    lines.extend(["", "RAGAS EVALUATION", "-" * 65])
    for metric, score in ragas_metrics.items():
        lines.append(f"{metric:<32}: {score:.2f}%")
    lines.append(f"Evaluated rows: {len(ragas_df)}")

    lines.extend(["", "ARES-INSPIRED EVALUATION", "-" * 65])
    for metric, score in ares_metrics.items():
        lines.append(f"{metric:<32}: {score:.2f}%")
    lines.append(f"Evaluated rows: {len(ares_df)}")

    lines.extend(["", "SPEECH RECOGNITION EVALUATION", "-" * 65])
    for condition, scores in stt_metrics.items():
        lines.append(
            f"{condition:<24}: "
            f"WER {scores['WER']:.2f}%, "
            f"CER {scores['CER']:.2f}%, "
            f"samples {scores['samples']}"
        )

    lines.extend([
        "",
        "GENERATED GRAPHS",
        "-" * 65,
        "1. rag_overall_accuracy.png",
        "2. ragas_overall_metrics.png",
        "3. ares_overall_metrics.png",
        "4. speech_recognition_metrics.png",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SAVED SUMMARY] {output}")
    return output



# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Load evaluation CSVs, generate the four retained charts, and save summary."""

    print("\n" + "=" * 70)
    print("REC VOICE AI RECEPTIONIST")
    print("FINAL EVALUATION VISUALIZATION")
    print("=" * 70)

    rag_df = load_rag_results()
    ragas_df = load_ragas_results()
    ares_df = load_ares_results()
    stt_df = load_stt_results()

    direct_metrics = create_direct_rag_graph(rag_df)
    ragas_metrics = create_ragas_graph(ragas_df)
    ares_metrics = create_ares_graph(ares_df)
    stt_metrics = create_speech_recognition_graph(stt_df)

    save_summary(
        direct_metrics=direct_metrics,
        ragas_metrics=ragas_metrics,
        ares_metrics=ares_metrics,
        stt_metrics=stt_metrics,
        rag_df=rag_df,
        ragas_df=ragas_df,
        ares_df=ares_df,
    )

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    output_files = [
        "rag_overall_accuracy.png",
        "ragas_overall_metrics.png",
        "ares_overall_metrics.png",
        "speech_recognition_metrics.png",
        "visualization_summary.txt",
    ]

    for filename in output_files:
        print(f"[OUTPUT] {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()