
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.rag.rag_service import RAGService
from app.evaluation.evaluate_rag import TEST_CASES as ALL_TEST_CASES


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "ares_evaluation_results.csv"
SUMMARY_CSV = RESULTS_DIR / "ares_evaluation_summary.csv"

EXPECTED_POSITIVE_CASES = 95


# ============================================================
# POSITIVE TEST SET
# Reuse the same 100-case dataset as evaluate_rag.py.
# Exclude the five unsupported/out-of-scope negative cases.
# ============================================================

TEST_CASES = [
    case for case in ALL_TEST_CASES
    if not case.get("negative", False)
]

if len(ALL_TEST_CASES) != 100:
    raise ValueError(
        f"Expected 100 shared test cases, found {len(ALL_TEST_CASES)}."
    )

if len(TEST_CASES) != EXPECTED_POSITIVE_CASES:
    raise ValueError(
        f"Expected {EXPECTED_POSITIVE_CASES} positive cases, "
        f"found {len(TEST_CASES)}. Check the negative flags "
        "in evaluate_rag.py."
    )


# ============================================================
# LOCAL LLM EVALUATOR
# Gemma 3:12B via Ollama
# Used only to judge ARES-inspired metrics.
# ============================================================

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

EVALUATOR_MODEL = "gemma3:12b"


# ============================================================
# JSON AND JUDGE HELPERS
# ============================================================

def clean_json_text(text: str) -> str:
    """Extract one JSON object, including when wrapped in Markdown."""
    text = text.strip()

    # Remove Markdown code fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    # Locate the first JSON object and respect braces inside strings.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. Raw response: {text!r}")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    raise ValueError(f"Incomplete JSON object. Raw response: {text!r}")


def call_judge(
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Call the local LLM, retry malformed JSON once, and validate scores."""

    last_error = ""

    for attempt in range(2):
        try:
            current_user_prompt = user_prompt

            if attempt == 1:
                current_user_prompt += """

IMPORTANT: Your previous response could not be parsed.
Return exactly one valid JSON object with this structure:
{"score": 0.75, "reason": "brief explanation"}

Use a numeric score from 0.0 to 1.0.
Do not use Markdown fences or add text outside the JSON object.
Escape quotation marks inside JSON strings correctly.
"""

            response = client.chat.completions.create(
                model=EVALUATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )

            content = response.choices[0].message.content or ""
            result = json.loads(clean_json_text(content))

            if not isinstance(result, dict) or "score" not in result:
                raise ValueError("JSON response must contain a score field.")

            score = float(result["score"])

            if not 0.0 <= score <= 1.0:
                raise ValueError("Score is outside the range 0 to 1.")

            return {
                "score": score,
                "reason": str(result.get("reason", "")),
                "error": "",
            }

        except Exception as exc:
            last_error = f"Attempt {attempt + 1}: {exc}"
            print(f"  [JUDGE WARNING] {last_error}")

    return {
        "score": None,
        "reason": "",
        "error": last_error,
    }


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def evaluate_context_relevance(
    question: str,
    contexts: list[str],
) -> dict[str, Any]:

    system_prompt = """
You are an objective evaluator for a retrieval-augmented generation
(RAG) system.

Evaluate CONTEXT RELEVANCE: determine whether the supplied retrieved
passages contain information useful for answering the question.

Score:
1.00 = directly relevant and useful.
0.75 = mostly relevant, with minor irrelevant content.
0.50 = partially relevant.
0.25 = mostly irrelevant.
0.00 = does not help answer the question.

Judge only the supplied passages and question. Do not assume facts
that are not present in the passages.

Return only valid JSON:
{"score": 0.0, "reason": "brief explanation"}
"""

    context_text = "\n\n".join(
        f"PASSAGE {i + 1}:\n{context}"
        for i, context in enumerate(contexts)
    )

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PASSAGES:
{context_text if context_text else "[No passages retrieved]"}

Evaluate context relevance.
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

    system_prompt = """
You are an objective evaluator for a retrieval-augmented generation
(RAG) system.

Evaluate ANSWER FAITHFULNESS: determine whether the important claims
in the answer are supported by the retrieved passages.

Do not primarily judge writing style or whether the answer is relevant.
Judge support from the supplied context. Do not use outside knowledge.

Score:
1.00 = all important claims are supported.
0.75 = nearly all claims are supported.
0.50 = some claims are supported and some are unsupported.
0.25 = most claims are unsupported.
0.00 = the answer is not supported by the supplied context.

Return only valid JSON:
{"score": 0.0, "reason": "brief explanation"}
"""

    context_text = "\n\n".join(
        f"PASSAGE {i + 1}:\n{context}"
        for i, context in enumerate(contexts)
    )

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PASSAGES:
{context_text if context_text else "[No passages retrieved]"}

GENERATED ANSWER:
{answer if answer else "[No answer generated]"}

Evaluate answer faithfulness.
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
You are an objective evaluator for a retrieval-augmented generation
(RAG) system.

Evaluate ANSWER RELEVANCE: determine whether the answer directly
addresses the user's question.

Judge responsiveness and completeness, not primarily factual accuracy.

Score:
1.00 = directly and completely addresses the question.
0.75 = relevant and mostly addresses the question.
0.50 = partially addresses the question.
0.25 = mostly unrelated or incomplete.
0.00 = does not address the question.

Return only valid JSON:
{"score": 0.0, "reason": "brief explanation"}
"""

    user_prompt = f"""
QUESTION:
{question}

GENERATED ANSWER:
{answer if answer else "[No answer generated]"}

Evaluate answer relevance.
"""

    return call_judge(system_prompt, user_prompt)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print("=" * 75)
    print("REC AI RECEPTIONIST - ARES-INSPIRED EVALUATION")
    print("=" * 75)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Evaluator model    : {EVALUATOR_MODEL}")
    print(f"Shared test cases  : {len(ALL_TEST_CASES)}")
    print(f"Positive test cases: {len(TEST_CASES)}")
    print(f"Output CSV         : {OUTPUT_CSV}")

    print("\n[STARTUP] Loading RAG service...")
    rag = RAGService()
    print("[STARTUP] RAG service ready.")

    final_rows: list[dict[str, Any]] = []
    evaluation_start = time.perf_counter()

    for index, case in enumerate(TEST_CASES, start=1):

        case_id = case.get("id", index)
        category = case.get("category", "")
        question = case["question"]

        print(f"\n[{index}/{len(TEST_CASES)}] "
              f"Case {case_id}: {question}")

        rag_start = time.perf_counter()

        try:
            result = rag.ask(question)
            answer = result.get("answer", "") or ""
            retrieved = result.get("retrieved", []) or []

            contexts = [
                item.get("text", "")
                for item in retrieved
                if isinstance(item, dict) and item.get("text")
            ]

            rag_time = time.perf_counter() - rag_start
            rag_error = ""

        except Exception as exc:
            answer = ""
            contexts = []
            rag_time = time.perf_counter() - rag_start
            rag_error = str(exc)

            print(f"[ERROR] RAG failed: {rag_error}")

        print(f"  Retrieved passages: {len(contexts)}")
        print(f"  RAG processing time: {rag_time:.2f}s")

        # Evaluate each metric separately.
        context_result = evaluate_context_relevance(
            question, contexts
        )
        faithfulness_result = evaluate_answer_faithfulness(
            question, answer, contexts
        )
        relevance_result = evaluate_answer_relevance(
            question, answer
        )

        context_score = context_result["score"]
        faithfulness_score = faithfulness_result["score"]
        relevance_score = relevance_result["score"]

        print(
            "  Scores: "
            f"Context={context_score}, "
            f"Faithfulness={faithfulness_score}, "
            f"Relevance={relevance_score}"
        )

        final_rows.append({
            "id": case_id,
            "category": category,
            "question": question,
            "answer": answer,
            "retrieved_context_count": len(contexts),
            "context_relevance": context_score,
            "answer_faithfulness": faithfulness_score,
            "answer_relevance": relevance_score,
            "context_relevance_reason": context_result["reason"],
            "faithfulness_reason": faithfulness_result["reason"],
            "answer_relevance_reason": relevance_result["reason"],
            "context_evaluation_error": context_result["error"],
            "faithfulness_evaluation_error": faithfulness_result["error"],
            "relevance_evaluation_error": relevance_result["error"],
            "rag_processing_time_seconds": round(rag_time, 4),
            "rag_error": rag_error,
        })

    total_evaluation_time = time.perf_counter() - evaluation_start

    # ========================================================
    # SAVE PER-QUESTION RESULTS
    # ========================================================

    fieldnames = list(final_rows[0].keys()) if final_rows else []

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    # ========================================================
    # CALCULATE METRIC MEANS
    # Failed evaluator calls remain missing, not zero.
    # ========================================================

    metric_names = [
        "context_relevance",
        "answer_faithfulness",
        "answer_relevance",
    ]

    summary_rows: list[dict[str, Any]] = []

    for metric in metric_names:
        valid_scores = [
            row[metric]
            for row in final_rows
            if row[metric] is not None
        ]

        average = (
            sum(valid_scores) / len(valid_scores)
            if valid_scores else None
        )

        summary_rows.append({
            "metric": metric,
            "mean_score": (
                round(average, 6)
                if average is not None else ""
            ),
            "mean_percentage": (
                round(average * 100, 2)
                if average is not None else ""
            ),
            "successful_evaluations": len(valid_scores),
            "expected_evaluations": len(final_rows),
        })

    with SUMMARY_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "mean_score",
                "mean_percentage",
                "successful_evaluations",
                "expected_evaluations",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL ARES-INSPIRED EVALUATION")
    print("=" * 75)

    for row in summary_rows:
        if row["mean_percentage"] == "":
            value = "N/A"
        else:
            value = f'{row["mean_percentage"]:.2f}%'

        print(
            f'{row["metric"]:25}: {value} '
            f'({row["successful_evaluations"]}/'
            f'{row["expected_evaluations"]} successful)'
        )

    failed_judgements = sum(
        row[metric] is None
        for row in final_rows
        for metric in metric_names
    )

    print(f"\nPositive test cases: {len(TEST_CASES)}")
    print(f"Failed metric judgments: {failed_judgements}")
    print(f"Total evaluation time: {total_evaluation_time:.2f}s")
    print(f"Detailed results: {OUTPUT_CSV}")
    print(f"Metric summary: {SUMMARY_CSV}")
    print("\nARES-inspired evaluation complete.")


if __name__ == "__main__":
    main()
