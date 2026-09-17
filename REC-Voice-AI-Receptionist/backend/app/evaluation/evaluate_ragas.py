from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from openai import AsyncOpenAI

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = PROJECT_ROOT / "evaluation_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORT_DIR / "ragas_evaluation_results.csv"


# ============================================================
# IMPORT PROJECT SERVICES
# ============================================================

try:

    from app.rag.rag_service import RAGService

except Exception as exc:

    print()
    print("=" * 70)
    print("[IMPORT ERROR]")
    print(exc)
    print("=" * 70)
    print()
    print("Run this program from the backend directory:")
    print()
    print("python -m app.evaluation.evaluate_ragas")
    print()

    sys.exit(1)


# ============================================================
# TEST DATA
#
# Same 18 answerable questions from evaluate_rag.py.
# Negative/refusal questions are intentionally excluded.
# ============================================================

TEST_CASES = [

    {
        "id": 1,
        "question": "What is the subject code for Database Management Systems?",
        "reference": (
            "The subject code for Database Management Systems "
            "is CS23332."
        ),
    },

    {
        "id": 2,
        "question": "What is the subject code for Computer Networks?",
        "reference": (
            "The subject code for Computer Networks "
            "is CS23532."
        ),
    },

    {
        "id": 3,
        "question": "What concepts are covered in Database Management Systems?",
        "reference": (
            "Database Management Systems covers database concepts, "
            "SQL, normalization, and transaction management."
        ),
    },

    {
        "id": 4,
        "question": "What is the duration of the B.E. or B.Tech. programme?",
        "reference": (
            "The B.E. or B.Tech. programme normally has a duration "
            "of 8 semesters or four academic years."
        ),
    },

    {
        "id": 5,
        "question": "How many working days does each semester normally have?",
        "reference": (
            "Each semester normally has 90 working days."
        ),
    },

    {
        "id": 6,
        "question": "What is the minimum attendance required for a course?",
        "reference": (
            "The minimum attendance required is 75 percent."
        ),
    },

    {
        "id": 7,
        "question": "What happens when attendance is between 65% and 74%?",
        "reference": (
            "Students with attendance between 65 percent and 74 percent "
            "may be eligible for attendance concession on approved "
            "medical or sports grounds, subject to the regulations."
        ),
    },

    {
        "id": 8,
        "question": "How many semesters can attendance concession be availed?",
        "reference": (
            "Attendance concession can be availed for a maximum of "
            "two semesters."
        ),
    },

    {
        "id": 9,
        "question": "What happens if a student does not satisfy the attendance requirements?",
        "reference": (
            "A student who does not satisfy the attendance requirements "
            "may have to redo the course."
        ),
    },

    {
        "id": 10,
        "question": "What is the maximum number of courses that can be added or dropped in a semester?",
        "reference": (
            "A maximum of two courses can be added or dropped "
            "in a semester."
        ),
    },

    {
        "id": 11,
        "question": "What is the medium of instruction at Rajalakshmi Engineering College?",
        "reference": (
            "The medium of instruction at Rajalakshmi Engineering "
            "College is English."
        ),
    },

    {
        "id": 12,
        "question": "What is the prescribed range of total credits for each B.E./B.Tech. degree programme?",
        "reference": (
            "The prescribed total credit requirement for a B.E./B.Tech. "
            "degree programme is in the range of 160 to 165 credits."
        ),
    },

    {
        "id": 13,
        "question": "What is the maximum duration for completing the B.E. or B.Tech programme for HSC students?",
        "reference": (
            "For HSC students, the maximum duration for completing "
            "the B.E. or B.Tech programme is 14 semesters."
        ),
    },

    {
        "id": 14,
        "question": "How many semesters is a B.E. or B.Tech programme normally completed in?",
        "reference": (
            "A B.E. or B.Tech programme is normally completed in "
            "8 semesters."
        ),
    },

    {
        "id": 15,
        "question": "What courses can a student register for from the second semester onwards?",
        "reference": (
            "From the second semester onwards, students can register "
            "for current semester courses and permitted reappearance courses."
        ),
    },

    {
        "id": 16,
        "question": "When does course enrollment for semesters II to VIII commence?",
        "reference": (
            "Course enrollment for semesters II to VIII commences "
            "within the prescribed period before the semester, including "
            "the specified 10 working day period."
        ),
    },

    {
        "id": 17,
        "question": "What subject is associated with CS23332?",
        "reference": (
            "CS23332 is the course code for Database Management Systems."
        ),
    },

    {
        "id": 18,
        "question": "What does Database Management Systems cover regarding database design?",
        "reference": (
            "Database Management Systems covers database design concepts "
            "including E-R diagrams and relational concepts."
        ),
    },

]


# ============================================================
# OLLAMA RAGAS EVALUATOR
# ============================================================

def create_evaluator():

    print()
    print("[STARTUP] Connecting RAGAS to local Ollama...")

    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        timeout=120.0,
        max_retries=0,
    )

    llm = llm_factory(
        "gemma3:4b",
        provider="openai",
        client=client,
        max_tokens=32768,
    )

    print("[STARTUP] RAGAS evaluator ready.")

    return llm, client


# ============================================================
# RUN SINGLE RAGAS METRIC
# ============================================================

async def run_metric(
    metric: Any,
    inputs: list[dict[str, Any]],
) -> list[float | None]:

    scores: list[float | None] = []

    for index, item in enumerate(
        inputs,
        start=1,
    ):

        try:

            result = await metric.ascore(
                **item
            )

            score_value = float(
                result.value
            )

            scores.append(
                score_value
            )

            print(
                f"  [{index}/{len(inputs)}] "
                f"{score_value:.4f}"
            )

        except Exception as exc:

            print(
                f"  [{index}/{len(inputs)}] "
                f"ERROR: {exc}"
            )

            scores.append(None)

    return scores


# ============================================================
# RUN ALL RAGAS METRICS
# ============================================================

async def run_all_metrics(
    faithfulness_metric: Any,
    faithfulness_inputs: list[dict[str, Any]],
    answer_relevancy_metric: Any,
    answer_relevancy_inputs: list[dict[str, Any]],
    context_precision_metric: Any,
    context_precision_inputs: list[dict[str, Any]],
    context_recall_metric: Any,
    context_recall_inputs: list[dict[str, Any]],
):

    print()
    print("[METRIC 1/4] Faithfulness")

    faithfulness_scores = await run_metric(
        faithfulness_metric,
        faithfulness_inputs,
    )

    print(
        "[OK] Faithfulness completed."
    )

    print()
    print("[METRIC 2/4] Answer Relevancy")

    answer_relevancy_scores = await run_metric(
        answer_relevancy_metric,
        answer_relevancy_inputs,
    )

    print(
        "[OK] Answer Relevancy completed."
    )

    print()
    print("[METRIC 3/4] Context Precision")

    context_precision_scores = await run_metric(
        context_precision_metric,
        context_precision_inputs,
    )

    print(
        "[OK] Context Precision completed."
    )

    print()
    print("[METRIC 4/4] Context Recall")

    context_recall_scores = await run_metric(
        context_recall_metric,
        context_recall_inputs,
    )

    print(
        "[OK] Context Recall completed."
    )

    return (
        faithfulness_scores,
        answer_relevancy_scores,
        context_precision_scores,
        context_recall_scores,
    )


# ============================================================
# MAIN
# ============================================================

async def main() -> None:

    print()
    print("=" * 75)
    print("REC AI RECEPTIONIST - RAGAS EVALUATION")
    print("=" * 75)

    print()
    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print()
    print(
        f"[INFO] Answerable test questions: "
        f"{len(TEST_CASES)}"
    )

    # --------------------------------------------------------
    # Load RAG service
    # --------------------------------------------------------

    print()
    print("[STARTUP] Loading RAG service...")

    rag = RAGService()

    print("[STARTUP] RAG service ready.")

    # --------------------------------------------------------
    # RAGAS evaluator
    # --------------------------------------------------------

    evaluator_llm, evaluator_client = create_evaluator()

    print()
    print("[STARTUP] Loading RAGAS embedding model...")

    evaluator_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    )

    print("[STARTUP] RAGAS embedding model ready.")

    # --------------------------------------------------------
    # Prepare evaluation data
    # --------------------------------------------------------

    questions: list[str] = []
    contexts: list[list[str]] = []
    answers: list[str] = []
    references: list[str] = []

    print()
    print("=" * 75)
    print("COLLECTING RAG RESPONSES")
    print("=" * 75)

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        question = test["question"]

        print()
        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"{question}"
        )

        start = time.perf_counter()

        try:

            result = rag.ask(question)

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            retrieved = result.get(
                "retrieved",
                [],
            )

            context_list: list[str] = []

            for item in retrieved:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                text = str(
                    item.get(
                        "text",
                        "",
                    )
                ).strip()

                if text:

                    context_list.append(
                        text
                    )

            if not context_list:

                context_list = [
                    "No relevant context was retrieved."
                ]

            questions.append(
                question
            )

            contexts.append(
                context_list
            )

            answers.append(
                answer
            )

            references.append(
                test["reference"]
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"[OK] Retrieved contexts: "
                f"{len(context_list)}"
            )

            print(
                f"[TIME] {elapsed:.2f}s"
            )

        except Exception as exc:

            print(
                f"[ERROR] {exc}"
            )

            questions.append(
                question
            )

            contexts.append(
                [
                    "No relevant context was retrieved."
                ]
            )

            answers.append(
                "No answer was generated."
            )

            references.append(
                test["reference"]
            )

    # --------------------------------------------------------
    # Initialize RAGAS metrics
    # --------------------------------------------------------

    print()
    print(
        "[STARTUP] Initializing RAGAS metrics..."
    )

    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    context_precision_metric = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall_metric = ContextRecall(
        llm=evaluator_llm
    )

    print(
        "[STARTUP] RAGAS metrics ready."
    )

    # --------------------------------------------------------
    # Prepare metric inputs
    # --------------------------------------------------------

    faithfulness_inputs = [
    {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": context[:3],
    }
    for question, answer, context in zip(
        questions, answers, contexts
    )
]

    answer_relevancy_inputs = [

        {
            "user_input": question,
            "response": answer,
        }

        for question, answer

        in zip(
            questions,
            answers,
        )
    ]

    context_precision_inputs = [

        {
            "user_input": question,
            "retrieved_contexts": context,
            "reference": reference,
        }

        for question, context, reference

        in zip(
            questions,
            contexts,
            references,
        )
    ]

    context_recall_inputs = [

        {
            "user_input": question,
            "retrieved_contexts": context,
            "reference": reference,
        }

        for question, context, reference

        in zip(
            questions,
            contexts,
            references,
        )
    ]

    # --------------------------------------------------------
    # Run RAGAS
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("RUNNING RAGAS METRICS")
    print("=" * 75)

    print()
    print(
        "This may take several minutes because "
        "gemma3:4b is being used as the evaluator."
    )

    start_time = time.perf_counter()

    try:

        (
            faithfulness_scores,
            answer_relevancy_scores,
            context_precision_scores,
            context_recall_scores,
        ) = await run_all_metrics(
            faithfulness_metric,
            faithfulness_inputs,
            answer_relevancy_metric,
            answer_relevancy_inputs,
            context_precision_metric,
            context_precision_inputs,
            context_recall_metric,
            context_recall_inputs,
        )

    finally:

        await evaluator_client.close()

    total_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        {
            "user_input": questions,
            "response": answers,
            "reference": references,
            "retrieved_contexts": contexts,
            "faithfulness": faithfulness_scores,
            "answer_relevancy": answer_relevancy_scores,
            "context_precision": context_precision_scores,
            "context_recall": context_recall_scores,
        }
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    dataframe.to_csv(
        CSV_REPORT,
        index=False,
    )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    print()
    print("=" * 75)
    print("FINAL RAGAS RESULT")
    print("=" * 75)

    print()

    for metric_name in metric_names:

        values = dataframe[
            metric_name
        ].dropna()

        if len(values) > 0:

            average = values.mean()

            print(
                f"{metric_name.replace('_', ' ').title():25s}"
                f": {average:.4f}"
            )

        else:

            print(
                f"{metric_name.replace('_', ' ').title():25s}"
                f": N/A"
            )

    print()

    print(
        f"Evaluation time           : "
        f"{total_time:.2f} seconds"
    )

    print()

    print(
        f"[REPORT] {CSV_REPORT}"
    )

    print()

    print("=" * 75)
    print("RAGAS EVALUATION COMPLETE")
    print("=" * 75)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())