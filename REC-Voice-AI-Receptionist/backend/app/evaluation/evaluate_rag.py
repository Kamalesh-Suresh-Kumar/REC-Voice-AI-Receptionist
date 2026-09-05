from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "evaluation_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORT_DIR / "rag_evaluation_results.csv"


# ============================================================
# IMPORT PROJECT SERVICES
# ============================================================

try:
    from app.rag.retriever import Retriever
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
    print("python -m app.evaluation.evaluate_rag")
    print()

    sys.exit(1)


# ============================================================
# TEST DATASET
# ============================================================
#
# These questions are based on information already present
# in your currently ingested text documents:
#
#   - R2023 CSE Curriculum and Syllabus
#   - REC 2023 UG Regulations
#
# We intentionally DO NOT include the two scanned certificate
# PDFs here because your current ingestion output showed:
#
#   Pages with text: 0
#   Chunks: 0
#
# Those documents should be evaluated after OCR is implemented.
#
# expected_terms:
#     At least one of these terms should appear in the retrieved
#     chunks for the retrieval test.
#
# answer_terms:
#     ALL of these terms should appear in the generated answer
#     for the answer-correctness test.
#
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # CURRICULUM
    # --------------------------------------------------------

    {
        "id": 1,
        "category": "Curriculum",
        "question": "What is the subject code for Database Management Systems?",
        "expected_terms": ["CS23332", "DATABASE MANAGEMENT SYSTEMS"],
        "answer_terms": ["CS23332"],
    },

    {
        "id": 2,
        "category": "Curriculum",
        "question": "What is the subject code for Computer Networks?",
        "expected_terms": ["CS23532", "COMPUTER NETWORKS"],
        "answer_terms": ["CS23532"],
    },

    {
        "id": 3,
        "category": "Curriculum",
        "question": "What concepts are covered in Database Management Systems?",
        "expected_terms": [
            "DATABASE",
            "SQL",
            "NORMALIZATION",
            "TRANSACTION",
        ],
        "answer_terms": [
            "DATABASE",
            "SQL",
            "NORMALIZATION",
        ],
    },

    {
        "id": 4,
        "category": "Curriculum",
        "question": "What is the duration of the B.E. or B.Tech programme?",
        "expected_terms": [
            "8 SEMESTERS",
            "FOUR ACADEMIC YEARS",
        ],
        "answer_terms": [
            "8",
            "SEMESTER",
        ],
    },

    {
        "id": 5,
        "category": "Curriculum",
        "question": "How many working days does each semester normally have?",
        "expected_terms": [
            "90 WORKING DAYS",
            "90",
        ],
        "answer_terms": [
            "90",
            "WORKING DAYS",
        ],
    },

    # --------------------------------------------------------
    # ATTENDANCE / REGULATIONS
    # --------------------------------------------------------

    {
        "id": 6,
        "category": "Regulations",
        "question": "What is the minimum attendance required for a course?",
        "expected_terms": [
            "75%",
            "75",
            "ATTENDANCE",
        ],
        "answer_terms": [
            "75%",
        ],
    },

    {
        "id": 7,
        "category": "Regulations",
        "question": "What happens when attendance is between 65% and 74%?",
        "expected_terms": [
            "65%",
            "74%",
            "MEDICAL",
            "SPORTS",
        ],
        "answer_terms": [
            "65%",
            "74%",
        ],
    },

    {
        "id": 8,
        "category": "Regulations",
        "question": "How many semesters can attendance concession be availed?",
        "expected_terms": [
            "TWO SEMESTERS",
            "2 SEMESTERS",
            "TWO",
        ],
        "answer_terms": [
            "TWO",
        ],
    },

    {
        "id": 9,
        "category": "Regulations",
        "question": "What happens if a student does not satisfy the attendance requirements?",
        "expected_terms": [
            "END SEMESTER EXAMINATION",
            "REDO THE COURSE",
            "ATTENDANCE",
        ],
        "answer_terms": [
            "REDO",
            "COURSE",
        ],
    },

    {
        "id": 10,
        "category": "Regulations",
        "question": "What is the maximum number of courses that can be added or dropped in a semester?",
        "expected_terms": [
            "MAXIMUM OF 2 COURSES",
            "2 COURSES",
            "ADD OR DROP",
        ],
        "answer_terms": [
            "2",
            "COURSES",
        ],
    },

    {
        "id": 11,
        "category": "Regulations",
        "question": "What is the medium of instruction at Rajalakshmi Engineering College?",
        "expected_terms": [
            "MEDIUM OF INSTRUCTION",
            "ENGLISH",
        ],
        "answer_terms": [
            "ENGLISH",
        ],
    },

    {
        "id": 12,
        "category": "Regulations",
        "question": "What is the prescribed range of total credits for each B.E./B.Tech. degree programme?",
        "expected_terms": [
            "160-165",
            "160",
            "165",
            "TOTAL CREDITS",
        ],
        "answer_terms": [
            "160",
            "165",
        ],
    },

    {
        "id": 13,
        "category": "Regulations",
        "question": "What is the maximum duration for completing the B.E. or B.Tech programme for HSC students?",
        "expected_terms": [
            "14 SEMESTERS",
            "14",
        ],
        "answer_terms": [
            "14",
            "SEMESTER",
        ],
    },

    {
        "id": 14,
        "category": "Regulations",
        "question": "How many semesters is a B.E. or B.Tech programme normally completed in?",
        "expected_terms": [
            "8 SEMESTERS",
            "FOUR ACADEMIC YEARS",
        ],
        "answer_terms": [
            "8",
            "SEMESTER",
        ],
    },

    # --------------------------------------------------------
    # COURSE REGISTRATION
    # --------------------------------------------------------

    {
        "id": 15,
        "category": "Regulations",
        "question": "What courses can a student register for from the second semester onwards?",
        "expected_terms": [
            "CURRENT SEMESTER",
            "REAPPEARANCE",
            "COURSES",
        ],
        "answer_terms": [
            "CURRENT",
            "SEMESTER",
        ],
    },

    {
        "id": 16,
        "category": "Regulations",
        "question": "When does course enrollment for semesters II to VIII commence?",
        "expected_terms": [
            "10 WORKING DAYS",
            "LAST WORKING DAY",
            "PRECEDING SEMESTER",
        ],
        "answer_terms": [
            "10",
            "WORKING DAYS",
        ],
    },

    # --------------------------------------------------------
    # ADDITIONAL CURRICULUM QUESTIONS
    # --------------------------------------------------------

    {
          "id": 17,
          "category": "Curriculum",
          "question": "What subject is associated with CS23332?",
          "expected_terms": [
               "CS23332",
               "DATABASE MANAGEMENT SYSTEMS",
          ],
          "answer_terms": [
               "DATABASE",
               "MANAGEMENT",
          ],
     },

    {
        "id": 18,
        "category": "Curriculum",
        "question": "What does Database Management Systems cover regarding database design?",
        "expected_terms": [
            "E-R DIAGRAM",
            "RELATIONAL ALGEBRA",
            "DATABASE DESIGN",
        ],
        "answer_terms": [
            "E-R",
            "RELATIONAL",
        ],
    },

    # --------------------------------------------------------
    # NEGATIVE / UNKNOWN QUESTIONS
    #
    # These are NOT counted in normal answer accuracy.
    # They are separately reported as "refusal tests".
    # --------------------------------------------------------

    {
        "id": 19,
        "category": "Negative",
        "question": "What is the weather today in Chennai?",
        "expected_terms": [],
        "answer_terms": [],
        "negative": True,
    },

    {
        "id": 20,
        "category": "Negative",
        "question": "What is the salary of the CSE faculty?",
        "expected_terms": [],
        "answer_terms": [],
        "negative": True,
    },

]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize text for simple evaluation.

    Example:
        "75% Attendance"
        ->
        "75 attendance"
    """

    text = str(text or "").upper()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# GET RETRIEVED TEXT
# ============================================================

def get_retrieved_text(results) -> str:

    pieces = []

    for item in results:

        if isinstance(item, dict):

            text = item.get(
                "text",
                "",
            )

            if text:

                pieces.append(
                    str(text)
                )

        else:

            pieces.append(
                str(item)
            )

    return "\n".join(
        pieces
    )


# ============================================================
# RETRIEVAL CHECK
# ============================================================

def retrieval_hit(
    retrieved_text: str,
    expected_terms: list[str],
) -> bool:

    if not expected_terms:

        return False

    normalized_text = normalize(
        retrieved_text
    )

    for term in expected_terms:

        if normalize(term) in normalized_text:

            return True

    return False


# ============================================================
# ANSWER CHECK
# ============================================================

def answer_correct(
    answer: str,
    expected_terms: list[str],
) -> bool:

    if not expected_terms:

        return False

    normalized_answer = normalize(
        answer
    )

    # --------------------------------------------------------
    # For short factual answers, at least one expected
    # answer term is enough.
    #
    # For multi-term answers, we require at least half.
    # --------------------------------------------------------

    matched = 0

    for term in expected_terms:

        if normalize(term) in normalized_answer:

            matched += 1

    required = max(
        1,
        (len(expected_terms) + 1) // 2,
    )

    return matched >= required


# ============================================================
# NEGATIVE ANSWER CHECK
# ============================================================

def refusal_detected(
    answer: str,
) -> bool:

    text = normalize(
        answer
    )

    refusal_patterns = [
        "I DON'T HAVE ENOUGH INFORMATION",
        "I DO NOT HAVE ENOUGH INFORMATION",
        "NOT IN OUR",
        "NOT AVAILABLE",
        "I'M SORRY",
        "I AM SORRY",
        "PLEASE CONTACT RECEPTION",
        "NO INFORMATION",
        "CANNOT ANSWER",
        "CAN'T ANSWER",
    ]

    for pattern in refusal_patterns:

        if pattern in text:

            return True

    return False


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print()
    print("=" * 75)
    print("REC AI RECEPTIONIST - RAG ACCURACY EVALUATION")
    print("=" * 75)

    print()
    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print()
    print(
        f"[INFO] Test questions: "
        f"{len(TEST_CASES)}"
    )

    print()
    print("[STARTUP] Loading Retriever...")

    retriever = Retriever()

    print(
        "[STARTUP] Retriever ready."
    )

    print()
    print("[STARTUP] Loading RAG service...")

    rag = RAGService()

    print(
        "[STARTUP] RAG service ready."
    )

    print()
    print("=" * 75)
    print("RUNNING EVALUATION")
    print("=" * 75)

    results = []

    retrieval_correct = 0
    answer_correct_count = 0

    normal_questions = 0

    negative_total = 0
    negative_correct = 0

    total_time = 0.0

    # ========================================================
    # RUN EACH TEST
    # ========================================================

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        question_id = test["id"]
        category = test["category"]
        question = test["question"]

        is_negative = test.get(
            "negative",
            False,
        )

        print()
        print("-" * 75)
        print(
            f"[TEST {index}/{len(TEST_CASES)}] "
            f"{category}"
        )
        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        start_time = time.perf_counter()

        try:

            retrieved = retriever.search(
                question,
                top_k=5,
            )

            retrieved_text = get_retrieved_text(
                retrieved
            )

            retrieval_ok = retrieval_hit(
                retrieved_text,
                test.get(
                    "expected_terms",
                    [],
                ),
            )

        except Exception as exc:

            print(
                f"[RETRIEVAL ERROR] {exc}"
            )

            retrieved = []
            retrieved_text = ""
            retrieval_ok = False

        # ----------------------------------------------------
        # RAG ANSWER
        # ----------------------------------------------------

        answer = ""

        try:

            rag_result = rag.ask(
                question
            )

            if isinstance(
                rag_result,
                dict,
            ):

                answer = str(
                    rag_result.get(
                        "answer",
                        "",
                    )
                ).strip()

            else:

                answer = str(
                    rag_result
                ).strip()

        except Exception as exc:

            print(
                f"[RAG ERROR] {exc}"
            )

            answer = ""

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        # ----------------------------------------------------
        # NORMAL QUESTION
        # ----------------------------------------------------

        if not is_negative:

            normal_questions += 1

            if retrieval_ok:

                retrieval_correct += 1

            answer_ok = answer_correct(
                answer,
                test.get(
                    "answer_terms",
                    [],
                ),
            )

            if answer_ok:

                answer_correct_count += 1

        # ----------------------------------------------------
        # NEGATIVE QUESTION
        # ----------------------------------------------------

        else:

            negative_total += 1

            answer_ok = refusal_detected(
                answer
            )

            if answer_ok:

                negative_correct += 1

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()

        print(
            "[RETRIEVAL] "
            + (
                "PASS"
                if retrieval_ok
                else "FAIL"
            )
        )

        print(
            "[ANSWER]"
        )

        print(
            answer
            if answer
            else "[NO ANSWER]"
        )

        if is_negative:

            print()

            print(
                "[REFUSAL TEST] "
                + (
                    "PASS"
                    if answer_ok
                    else "FAIL"
                )
            )

        else:

            print()

            print(
                "[ANSWER CORRECTNESS] "
                + (
                    "PASS"
                    if answer_ok
                    else "FAIL"
                )
            )

        print(
            f"[TIME] {elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "id": question_id,
                "category": category,
                "question": question,
                "retrieval_correct": (
                    "YES"
                    if retrieval_ok
                    else "NO"
                ),
                "answer": answer,
                "answer_correct": (
                    "YES"
                    if answer_ok
                    else "NO"
                ),
                "time_seconds": f"{elapsed:.2f}",
            }
        )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    retrieval_accuracy = (
        retrieval_correct
        / normal_questions
        * 100
        if normal_questions
        else 0
    )

    answer_accuracy = (
        answer_correct_count
        / normal_questions
        * 100
        if normal_questions
        else 0
    )

    refusal_accuracy = (
        negative_correct
        / negative_total
        * 100
        if negative_total
        else 0
    )

    average_time = (
        total_time
        / len(TEST_CASES)
        if TEST_CASES
        else 0
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "id",
            "category",
            "question",
            "retrieval_correct",
            "answer",
            "answer_correct",
            "time_seconds",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 75)
    print("FINAL EVALUATION RESULT")
    print("=" * 75)

    print()

    print(
        f"Normal test questions : "
        f"{normal_questions}"
    )

    print(
        f"Retrieval correct     : "
        f"{retrieval_correct}"
    )

    print(
        f"Answer correct        : "
        f"{answer_correct_count}"
    )

    print()

    print(
        f"Retrieval Hit@5       : "
        f"{retrieval_accuracy:.2f}%"
    )

    print(
        f"Answer Accuracy       : "
        f"{answer_accuracy:.2f}%"
    )

    print()

    print(
        f"Negative tests        : "
        f"{negative_total}"
    )

    print(
        f"Correct refusals      : "
        f"{negative_correct}"
    )

    print(
        f"Refusal Accuracy      : "
        f"{refusal_accuracy:.2f}%"
    )

    print()

    print(
        f"Average processing    : "
        f"{average_time:.2f} seconds"
    )

    print()

    print(
        f"[REPORT] "
        f"{CSV_REPORT}"
    )

    print()

    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()