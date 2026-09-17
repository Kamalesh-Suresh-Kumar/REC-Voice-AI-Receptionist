from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.rag.rag_service import RAGService


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "ares_evaluation_results.csv"


# ============================================================
# ARES-INSPIRED TEST SET
# Same 18 answerable questions used for RAGAS evaluation
# ============================================================

TEST_CASES = [
    {
        "question": "What is the subject code for Database Management Systems?"
    },
    {
        "question": "What is the subject code for Computer Networks?"
    },
    {
        "question": "What concepts are covered in Database Management Systems?"
    },
    {
        "question": "What is the duration of the B.E. or B.Tech. programme?"
    },
    {
        "question": "How many working days does each semester normally have?"
    },
    {
        "question": "What is the minimum attendance required for a course?"
    },
    {
        "question": "What happens when attendance is between 65% and 74%?"
    },
    {
        "question": "How many semesters can attendance concession be availed?"
    },
    {
        "question": "What happens if a student does not satisfy the attendance requirements?"
    },
    {
        "question": "What is the maximum number of courses that can be added or dropped in a semester?"
    },
    {
        "question": "What is the medium of instruction at Rajalakshmi Engineering College?"
    },
    {
        "question": "What is the prescribed range of total credits for each B.E./B.Tech. degree programme?"
    },
    {
        "question": "What is the maximum duration for completing the B.E. or B.Tech programme for HSC students?"
    },
    {
        "question": "How many semesters is a B.E. or B.Tech programme normally completed in?"
    },
    {
        "question": "What courses can a student register for from the second semester onwards?"
    },
    {
        "question": "When does course enrollment for semesters II to VIII commence?"
    },
    {
        "question": "What subject is associated with CS23332?"
    },
    {
        "question": "What does Database Management Systems cover regarding database design?"
    },
]


# ============================================================
# LOCAL LLM EVALUATOR
# Gemma 3:4B
# Used ONLY for ARES-inspired evaluation
# ============================================================

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

EVALUATOR_MODEL = "gemma3:4b"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_json_text(text: str) -> str:
    """
    Extract a JSON object from the model response.
    Handles markdown code fences and extra text.
    """

    text = text.strip()

    # Remove markdown code fences if present.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find first JSON object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        return match.group(0)

    return text


def call_judge(
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:

    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )

    content = response.choices[0].message.content or ""

    cleaned = clean_json_text(content)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "reason": "Evaluator returned an invalid JSON response.",
            "raw": content,
        }

    score = result.get("score", 0.0)

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "reason": str(result.get("reason", "")),
        "raw": content,
    }


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def evaluate_context_relevance(
    question: str,
    contexts: list[str],
) -> dict[str, Any]:

    context_text = "\n\n".join(
        f"PASSAGE {index + 1}:\n{context}"
        for index, context in enumerate(contexts)
    )

    system_prompt = """
You are an objective evaluator for a Retrieval-Augmented Generation system.

Evaluate CONTEXT RELEVANCE.

Determine whether the retrieved passages are relevant for answering the
user's question.

Score:
1.0 = the retrieved passages are directly relevant and provide useful
      information for answering the question.
0.75 = most passages are relevant, with minor irrelevant information.
0.50 = partially relevant information is present.
0.25 = mostly irrelevant information.
0.0 = the passages do not help answer the question.

Return ONLY valid JSON:

{
  "score": 0.0,
  "reason": "brief explanation"
}
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PASSAGES:
{context_text}

Evaluate the relevance of the retrieved passages to the question.
"""

    return call_judge(system_prompt, user_prompt)


# ============================================================
# ANSWER FAITHFULNESS
# ============================================================

def evaluate_answer_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:

    context_text = "\n\n".join(
        f"PASSAGE {index + 1}:\n{context}"
        for index, context in enumerate(contexts)
    )

    system_prompt = """
You are an objective evaluator for a Retrieval-Augmented Generation system.

Evaluate ANSWER FAITHFULNESS.

Determine whether the answer is supported by the retrieved passages.

Do not judge whether the answer is stylistically good.
Judge only whether the claims in the answer are supported by the supplied
retrieved context.

Score:
1.0 = all important claims are supported by the retrieved context.
0.75 = almost all claims are supported, with only a minor unsupported detail.
0.50 = some claims are supported and some are unsupported.
0.25 = most claims are unsupported.
0.0 = the answer is not supported by the retrieved context.

Return ONLY valid JSON:

{
  "score": 0.0,
  "reason": "brief explanation"
}
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PASSAGES:
{context_text}

GENERATED ANSWER:
{answer}

Evaluate whether the generated answer is faithful to the retrieved
passages.
"""

    return call_judge(system_prompt, user_prompt)


# ============================================================
# ANSWER RELEVANCE
# ============================================================

def evaluate_answer_relevance(
    question: str,
    answer: str,
) -> dict[str, Any]:

    system_prompt = """
You are an objective evaluator for a Retrieval-Augmented Generation system.

Evaluate ANSWER RELEVANCE.

Determine whether the generated answer directly addresses the user's
question.

Do not primarily judge factual correctness.
Judge whether the answer responds to what was asked.

Score:
1.0 = directly and completely addresses the question.
0.75 = relevant and mostly addresses the question.
0.50 = partially addresses the question.
0.25 = mostly unrelated or incomplete.
0.0 = does not answer the question.

Return ONLY valid JSON:

{
  "score": 0.0,
  "reason": "brief explanation"
}
"""

    user_prompt = f"""
QUESTION:
{question}

GENERATED ANSWER:
{answer}

Evaluate the relevance of the answer to the question.
"""

    return call_judge(system_prompt, user_prompt)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print("=" * 75)
    print("REC AI RECEPTIONIST - ARES-INSPIRED EVALUATION")
    print("=" * 75)

    print()
    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print()
    print("[INFO] ARES-inspired evaluator:")
    print(EVALUATOR_MODEL)

    print()
    print("[INFO] Test questions:")
    print(len(TEST_CASES))

    print()
    print("[STARTUP] Loading RAG service...")

    rag = RAGService()

    print("[STARTUP] RAG service ready.")

    print()
    print("=" * 75)
    print("COLLECTING RAG RESPONSES")
    print("=" * 75)

    evaluation_rows: list[dict[str, Any]] = []

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]

        print()
        print(f"[{index}/{len(TEST_CASES)}] {question}")

        start_time = time.time()

        try:
            result = rag.ask(question)

            answer = result.get("answer", "")

            retrieved = result.get("retrieved", [])

            contexts = [
                item.get("text", "")
                for item in retrieved
                if item.get("text")
            ]

            elapsed = time.time() - start_time

            print(f"[OK] Retrieved contexts: {len(contexts)}")
            print(f"[TIME] {elapsed:.2f}s")

            evaluation_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "rag_time": elapsed,
                }
            )

        except Exception as exc:

            print(f"[ERROR] RAG failed: {exc}")

            evaluation_rows.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "rag_time": time.time() - start_time,
                }
            )

    print()
    print("=" * 75)
    print("RUNNING ARES-INSPIRED METRICS")
    print("=" * 75)

    print()
    print(
        "Evaluating Context Relevance, Answer Faithfulness, "
        "and Answer Relevance using Gemma 3:4B."
    )

    context_scores: list[float] = []
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []

    final_rows: list[dict[str, Any]] = []

    evaluation_start = time.time()

    for index, row in enumerate(evaluation_rows, start=1):

        question = row["question"]
        answer = row["answer"]
        contexts = row["contexts"]

        print()
        print(f"[{index}/{len(evaluation_rows)}] Evaluating:")
        print(question)

        # ----------------------------------------------------
        # Context Relevance
        # ----------------------------------------------------

        print("  → Context Relevance")

        context_result = evaluate_context_relevance(
            question=question,
            contexts=contexts,
        )

        context_score = context_result["score"]

        # ----------------------------------------------------
        # Answer Faithfulness
        # ----------------------------------------------------

        print("  → Answer Faithfulness")

        faithfulness_result = evaluate_answer_faithfulness(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        faithfulness_score = faithfulness_result["score"]

        # ----------------------------------------------------
        # Answer Relevance
        # ----------------------------------------------------

        print("  → Answer Relevance")

        relevance_result = evaluate_answer_relevance(
            question=question,
            answer=answer,
        )

        relevance_score = relevance_result["score"]

        print(
            f"  Scores: "
            f"Context={context_score:.4f}, "
            f"Faithfulness={faithfulness_score:.4f}, "
            f"Relevance={relevance_score:.4f}"
        )

        context_scores.append(context_score)
        faithfulness_scores.append(faithfulness_score)
        relevance_scores.append(relevance_score)

        final_rows.append(
            {
                "question": question,
                "answer": answer,
                "context_relevance": context_score,
                "answer_faithfulness": faithfulness_score,
                "answer_relevance": relevance_score,
                "context_relevance_reason": context_result["reason"],
                "faithfulness_reason": faithfulness_result["reason"],
                "answer_relevance_reason": relevance_result["reason"],
                "rag_processing_time": row["rag_time"],
            }
        )

    evaluation_time = time.time() - evaluation_start

    # ========================================================
    # AVERAGES
    # ========================================================

    def average(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    avg_context = average(context_scores)
    avg_faithfulness = average(faithfulness_scores)
    avg_relevance = average(relevance_scores)

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "question",
        "answer",
        "context_relevance",
        "answer_faithfulness",
        "answer_relevance",
        "context_relevance_reason",
        "faithfulness_reason",
        "answer_relevance_reason",
        "rag_processing_time",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(final_rows)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL ARES-INSPIRED RESULT")
    print("=" * 75)

    print()
    print(
        f"Context Relevance       : {avg_context:.4f}"
    )

    print(
        f"Answer Faithfulness     : {avg_faithfulness:.4f}"
    )

    print(
        f"Answer Relevance        : {avg_relevance:.4f}"
    )

    print()
    print(
        f"Evaluation time         : {evaluation_time:.2f} seconds"
    )

    print()
    print(f"[REPORT] {OUTPUT_CSV}")

    print()
    print("=" * 75)
    print("ARES-INSPIRED EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()