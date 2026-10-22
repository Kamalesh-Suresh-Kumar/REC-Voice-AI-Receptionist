from __future__ import annotations

import asyncio
import importlib.util
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

REPORT_DIR = (
    PROJECT_ROOT / "evaluation_results"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_REPORT = (
    REPORT_DIR
    / "ragas_100_evaluation_results_final_v2.csv"
)


# ============================================================
# PROJECT SERVICE
# ============================================================

try:

    from app.rag.rag_service import RAGService

except Exception as exc:

    print()
    print("=" * 75)
    print("[IMPORT ERROR]")
    print(exc)
    print("=" * 75)
    print()
    print(
        "Run from the backend directory:"
    )
    print(
        "python -m app.evaluation.evaluate_ragas"
    )
    print()

    sys.exit(1)


# ============================================================
# LOAD TEST DATA
# ============================================================

TEST_FILE = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "evaluation"
    / "evaluate_rag.py"
)

if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n{TEST_FILE}"
    )


print()
print("[TEST DATA] Loading:")
print(TEST_FILE)


spec = importlib.util.spec_from_file_location(
    "evaluate_rag_final",
    TEST_FILE,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not load evaluate_rag.py"
    )


test_module = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    test_module
)


ALL_TEST_CASES = (
    test_module.TEST_CASES
)


print(
    f"[TEST DATA] Total cases: "
    f"{len(ALL_TEST_CASES)}"
)


# ============================================================
# NATURAL REFERENCES
#
# These are evaluation references.
# They are NOT used by the production RAG.
# ============================================================

REFERENCE_ANSWERS = {

    1:
        "B.E. Computer Science and Engineering is offered by Rajalakshmi Engineering College.",

    2:
        "M.E. Computer Science and Engineering is offered by the Computer Science and Engineering department.",

    3:
        "The MBA programme has a duration of 2 years.",

    4:
        "Yes, Rajalakshmi Engineering College offers a Ph.D. programme in Computer Science and Engineering.",

    5:
        "The opening cutoff for B.E. Computer Science and Engineering is 198.",

    6:
        "The closing cutoff for B.E. Computer Science and Engineering is 192.",

    7:
        "The opening cutoff for B.Tech. Artificial Intelligence and Data Science is 197.",

    8:
        "The closing cutoff for B.Tech. Information Technology is 190.",

    9:
        "The closing cutoff for B.E. Robotics and Automation is 181.",

    10:
        "The closing cutoff for B.Tech. Food Technology is 173.",

    11:
        "The subject code for Database Management Systems is CS23332.",

    12:
        "The subject code for Computer Networks is CS23532.",

    13:
        "Database Management Systems covers database concepts, SQL and normalization.",

    14:
        "The normal duration of the B.E. or B.Tech. programme is four years or eight semesters.",

    15:
        "Each semester normally has 90 working days.",

    16:
        "The prescribed total credit range for a B.E./B.Tech. degree is 160 to 165 credits.",

    17:
        "A B.E. or B.Tech. programme is normally completed in eight semesters.",

    18:
        "The maximum duration for completing the B.E. or B.Tech. programme for HSC students is 14 semesters.",

    19:
        "From the second semester onwards, students can register for courses according to the current semester and reappearance requirements.",

    20:
        "Course enrollment for Semesters II to VIII will commence 10 working days prior to the last working day of the preceding semester.",

    21:
        "The Information Technology department offers B.Tech. Information Technology.",

    22:
        "The Information Technology department offers M.Tech. Data Science.",

    23:
        "The Electronics and Communication Engineering department offers B.E. Electronics and Communication Engineering, M.E. Communication Systems, and Ph.D. programmes.",

    24:
        "The Artificial Intelligence and Machine Learning department offers B.Tech. Artificial Intelligence and Machine Learning.",

    25:
        "Rajalakshmi Engineering College offers a Minor Degree Programme in Data Science.",

    26:
        "Dr. P. Kumar is the Professor and Head of the Computer Science and Engineering department.",

    27:
        "Dr. M. Palanivelan is the Professor and Head of the Electronics and Communication Engineering department.",

    28:
        "Dr. Priya Vijay is the Professor and Head of the Information Technology department.",

    29:
        "Dr. S. Baghavathi Priya is the Professor and Head of the Artificial Intelligence and Machine Learning department.",

    30:
        "Dr. P. Revathy is the Professor and Head of the Computer Science and Design department.",

    31:
        "Dr. S.P. Srinivasan is the Professor and Head of the Mechanical Engineering department.",

    32:
        "Dr. R. Giri is the Professor and Head of the Robotics and Automation department.",

    33:
        "Dr. M. Uma Raman is the Professor and Head of Management Studies.",

    34:
        "Dr. M. Muthusamy is the Professor and Head of Humanities and Sciences.",

    35:
        "R. Karthikeyan is the Controller of Examinations.",

    36:
        "The minimum attendance required for a course is 75%.",

    37:
        "Attendance between 65% and 74% is handled according to the attendance concession rules specified by the institution.",

    38:
        "A student can avail the attendance concession for any two semesters during the entire course of the study.",

    39:
        "A student who does not satisfy the attendance requirements is not permitted to appear for the corresponding semester examination.",

    40:
        "A maximum of two courses can be added or dropped in a semester.",

    41:
        "The medium of instruction is English.",

    42:
        "The normal duration of the B.E./B.Tech. programme is eight semesters.",

    43:
        "The maximum programme duration allowed for HSC or equivalent students is 14 semesters.",

    44:
        "A semester normally has 90 working days.",

    45:
        "A maximum of two courses can be added or dropped in one semester.",

    46:
        "Performance in each course is evaluated through continuous assessment and the End Semester Examination.",

    47:
        "The maximum mark for each course is 100 marks.",

    48:
        "The normal End Semester Examination duration for theory courses is three hours.",

    49:
        "Odd-semester End Semester Examinations are normally conducted in December or January.",

    50:
        "Even-semester End Semester Examinations are normally conducted in May or June.",

    51:
        "The Controller of Examinations appoints the internal and external examiners.",

    52:
        "A minimum of 50% of the total marks is required to pass a course.",

    53:
        "A minimum of 45% in the End Semester Examination is required for passing a course.",

    54:
        "A student can submit a revaluation application through the prescribed examination process.",

    55:
        "The normal withdrawal application deadline is 10 working days before the End Semester Examination.",

    56:
        "The Administrative Department is headed by the Administrative Officer.",

    57:
        "The Administrative Office handles student service, HR, purchase and dispatch.",

    58:
        "Yes, students can contact the Administrative Office for fee-related queries.",

    59:
        "Yes, students can contact the Administrative Office for identity cards.",

    60:
        "Yes, students can contact the Administrative Office for bonafide certificates.",

    61:
        "Yes, students can contact the Administrative Office for bus passes.",

    62:
        "Yes, students can contact the Administrative Office for hall tickets.",

    63:
        "The ERP software used by Rajalakshmi Engineering College for college administration is DigiCampus.",

    64:
        "The ERP modules include attendance, library, transport, examinations and hostel.",

    65:
        "Yes, the REC Unified Portal can generate fee certificates.",

    66:
        "The college hostels can accommodate 2000 boys.",

    67:
        "The college hostels can accommodate 1200 girls.",

    68:
        "Yes, separate hostels are available for boys and girls.",

    69:
        "The warden is responsible for allocating hostel rooms.",

    70:
        "The hostel mess serves vegetarian food.",

    71:
        "Hostel visitors are allowed from 11 AM to 6 PM on holidays.",

    72:
        "The Central Library holds more than 61,000 books.",

    73:
        "Yes, the Central Library provides remote access to e-resources.",

    74:
        "The library provides a Digital Library facility for digital resources.",

    75:
        "The college sports complex maintains play fields, courts and tracks.",

    76:
        "Yes, Rajalakshmi Engineering College has a dedicated Training and Placement Cell.",

    77:
        "The Training and Placement Cell provides training and employment-related support to students.",

    78:
        "Yes, an eligible student can apply for a company visiting the campus.",

    79:
        "The 2021-2025 placement batch had 1941+ offers as of 12.09.2025.",

    80:
        "Accenture is one of the companies listed among the placement recruiters.",

    81:
        "A Super Dream offer is 6.5 LPA or above.",

    82:
        "A Dream offer is 4.5 LPA.",

    83:
        "A General offer is less than 4.5 LPA.",

    84:
        "Yes, a student with a General offer can apply for another Dream, Super Dream or Core offer.",

    85:
        "The Training and Placement Cell provides training and value-added programmes to improve employability.",

    86:
        "Rajalakshmi Engineering College publishes its latest events and updates through its official communication channels.",

    87:
        "Athletics is one of the student-life sections.",

    88:
        "Health Care is one of the student-life sections.",

    89:
        "Technical Societies is one of the student-life sections.",

    90:
        "Dining is one of the student-life sections.",

    91:
        "Rajalakshmi Engineering College was established in 1997.",

    92:
        "Yes, Rajalakshmi Engineering College is an autonomous institution affiliated with Anna University.",

    93:
        "The college campus is located at Rajalakshmi Nagar, Thandalam, Tamil Nadu 602105.",

    94:
        "Rajalakshmi Engineering College currently lists 19 undergraduate programmes.",

    95:
        "Rajalakshmi Engineering College has an A++ NAAC accreditation grade.",
}


# ============================================================
# BUILD ANSWERABLE TESTS
# ============================================================

TEST_CASES = []

for test in ALL_TEST_CASES:

    test_id = int(
        test.get(
            "id",
            0,
        )
    )

    question = str(
        test.get(
            "question",
            "",
        )
    ).strip()

    answer_terms = test.get(
        "answer_terms",
        [],
    )

    expected_terms = test.get(
        "expected_terms",
        [],
    )

    # Exclude unsupported/negative tests.
    if not answer_terms or not expected_terms:
        continue

    reference = REFERENCE_ANSWERS.get(
        test_id
    )

    if not reference:

        raise ValueError(
            f"Missing reference for test {test_id}"
        )

    TEST_CASES.append(
        {
            "id": test_id,
            "category": test.get(
                "category",
                "",
            ),
            "question": question,
            "reference": reference,
        }
    )


print()
print(
    f"[TEST DATA] Answerable cases: "
    f"{len(TEST_CASES)}"
)

if len(TEST_CASES) != 95:

    raise RuntimeError(
        "Expected exactly 95 answerable tests."
    )


# ============================================================
# OLLAMA EVALUATOR
# ============================================================

def create_evaluator():

    print()
    print(
        "[STARTUP] Connecting RAGAS to Ollama..."
    )

    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        timeout=1200.0,
        max_retries=2,
    )

    llm = llm_factory(
        "gemma3:12b",
        provider="openai",
        client=client,
        max_tokens=8192,
    )

    print(
        "[STARTUP] Gemma 3:12B "
        "RAGAS evaluator ready."
    )

    return llm


# ============================================================
# RUN METRIC
# ============================================================

async def run_metric(
    metric: Any,
    inputs: list[dict[str, Any]],
    metric_name: str,
) -> list[float | None]:

    scores = []

    total = len(inputs)

    for index, item in enumerate(
        inputs,
        start=1,
    ):

        print()
        print(
            f"[{metric_name}] "
            f"{index}/{total}"
        )

        try:

            result = await asyncio.wait_for(
                metric.ascore(
                    **item
                ),
                timeout=1500.0,
            )

            score = float(
                result.value
            )

            scores.append(
                score
            )

            print(
                f"[{metric_name}] "
                f"{index}/{total} -> "
                f"{score:.4f}"
            )

        except asyncio.TimeoutError:

            print(
                f"[{metric_name}] "
                f"{index}/{total} -> TIMEOUT"
            )

            scores.append(None)

        except Exception as exc:

            print(
                f"[{metric_name}] "
                f"{index}/{total} -> "
                f"ERROR: {exc}"
            )

            scores.append(None)

    return scores


# ============================================================
# MAIN
# ============================================================

async def main():

    print()
    print("=" * 75)
    print(
        "REC AI RECEPTIONIST - RAGAS FINAL V2"
    )
    print("=" * 75)

    print()
    print(
        f"[INFO] Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"[INFO] Answerable tests: "
        f"{len(TEST_CASES)}"
    )

    # ========================================================
    # LOAD RAG
    # ========================================================

    print()
    print(
        "[STARTUP] Loading RAG service..."
    )

    rag = RAGService()

    print(
        "[STARTUP] RAG service ready."
    )

    # ========================================================
    # LOAD 12B EVALUATOR
    # ========================================================

    evaluator_llm = create_evaluator()

    # ========================================================
    # RAGAS EMBEDDINGS
    # ========================================================

    print()
    print(
        "[STARTUP] Loading RAGAS embeddings..."
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        device="cuda",
    )

    print(
        "[STARTUP] RAGAS embeddings ready."
    )

    # ========================================================
    # COLLECT RESPONSES
    # ========================================================

    questions = []
    answers = []
    contexts = []
    references = []

    print()
    print("=" * 75)
    print(
        "COLLECTING RAG RESPONSES"
    )
    print("=" * 75)

    collection_start = time.perf_counter()

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

            result = rag.ask(
                question
            )

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

            actual_contexts = []

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

                    actual_contexts.append(
                        text
                    )

            # Keep the actual production top-5.
            actual_contexts = (
                actual_contexts[:5]
            )

            if not actual_contexts:

                actual_contexts = [
                    "No relevant context was retrieved."
                ]

            questions.append(
                question
            )

            answers.append(
                answer
            )

            contexts.append(
                actual_contexts
            )

            references.append(
                test["reference"]
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"[OK] Contexts: "
                f"{len(actual_contexts)}"
            )

            print(
                f"[TIME] "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:

            print(
                f"[ERROR] {exc}"
            )

            questions.append(
                question
            )

            answers.append(
                "No answer was generated."
            )

            contexts.append(
                [
                    "No relevant context was retrieved."
                ]
            )

            references.append(
                test["reference"]
            )

    collection_time = (
        time.perf_counter()
        - collection_start
    )

    print()
    print(
        f"[INFO] Response collection time: "
        f"{collection_time:.2f}s"
    )

    # ========================================================
    # INITIALIZE METRICS
    # ========================================================

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

    # ========================================================
    # INPUTS
    # ========================================================

    faithfulness_inputs = [

        {
            "user_input": q,
            "response": a,
            "retrieved_contexts": c,
        }

        for q, a, c in zip(
            questions,
            answers,
            contexts,
        )
    ]

    answer_relevancy_inputs = [

        {
            "user_input": q,
            "response": a,
        }

        for q, a in zip(
            questions,
            answers,
        )
    ]

    context_precision_inputs = [

        {
            "user_input": q,
            "retrieved_contexts": c,
            "reference": r,
        }

        for q, c, r in zip(
            questions,
            contexts,
            references,
        )
    ]

    context_recall_inputs = [

        {
            "user_input": q,
            "retrieved_contexts": c,
            "reference": r,
        }

        for q, c, r in zip(
            questions,
            contexts,
            references,
        )
    ]

    # ========================================================
    # RUN
    # ========================================================

    print()
    print("=" * 75)
    print(
        "RUNNING RAGAS FINAL V2"
    )
    print("=" * 75)

    start_time = time.perf_counter()

    # ========================================================
    # FAITHFULNESS
    # ========================================================

    print()
    print("[1/4] Faithfulness")

    faithfulness_scores = await run_metric(
        faithfulness_metric,
        faithfulness_inputs,
        "Faithfulness",
    )

    print(
        "[CHECKPOINT] Faithfulness completed."
    )

    # ========================================================
    # ANSWER RELEVANCY
    # ========================================================

    print()
    print("[2/4] Answer Relevancy")

    answer_relevancy_scores = await run_metric(
        answer_relevancy_metric,
        answer_relevancy_inputs,
        "Answer Relevancy",
    )

    print(
        "[CHECKPOINT] Answer Relevancy completed."
    )

    # ========================================================
    # CONTEXT PRECISION
    # ========================================================

    print()
    print("[3/4] Context Precision")

    context_precision_scores = await run_metric(
        context_precision_metric,
        context_precision_inputs,
        "Context Precision",
    )

    print(
        "[CHECKPOINT] Context Precision completed."
    )

    # ========================================================
    # CONTEXT RECALL
    # ========================================================

    print()
    print("[4/4] Context Recall")

    context_recall_scores = await run_metric(
        context_recall_metric,
        context_recall_inputs,
        "Context Recall",
    )

    print(
        "[CHECKPOINT] Context Recall completed."
    )

    elapsed_total = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    rows = []

    for index in range(
        len(questions)
    ):

        rows.append(
            {
                "test_id":
                    TEST_CASES[index]["id"],

                "category":
                    TEST_CASES[index]["category"],

                "user_input":
                    questions[index],

                "response":
                    answers[index],

                "reference":
                    references[index],

                "retrieved_contexts":
                    contexts[index],

                "faithfulness":
                    faithfulness_scores[index],

                "answer_relevancy":
                    answer_relevancy_scores[index],

                "context_precision":
                    context_precision_scores[index],

                "context_recall":
                    context_recall_scores[index],
            }
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        CSV_REPORT,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # MEAN
    # ========================================================

    def mean_score(
        values: list[float | None],
    ) -> float:

        valid = [
            float(value)
            for value in values
            if value is not None
        ]

        if not valid:
            return 0.0

        return (
            sum(valid)
            / len(valid)
        )

    faithfulness_avg = mean_score(
        faithfulness_scores
    )

    answer_relevancy_avg = mean_score(
        answer_relevancy_scores
    )

    context_precision_avg = mean_score(
        context_precision_scores
    )

    context_recall_avg = mean_score(
        context_recall_scores
    )

    # ========================================================
    # VALID COUNTS
    # ========================================================

    faithfulness_valid = sum(
        score is not None
        for score in faithfulness_scores
    )

    answer_relevancy_valid = sum(
        score is not None
        for score in answer_relevancy_scores
    )

    context_precision_valid = sum(
        score is not None
        for score in context_precision_scores
    )

    context_recall_valid = sum(
        score is not None
        for score in context_recall_scores
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print(
        "RAGAS FINAL V2 RESULTS"
    )
    print("=" * 75)

    print()

    print(
        f"Faithfulness       : "
        f"{faithfulness_avg * 100:.2f}%"
    )

    print(
        f"Answer Relevancy   : "
        f"{answer_relevancy_avg * 100:.2f}%"
    )

    print(
        f"Context Precision  : "
        f"{context_precision_avg * 100:.2f}%"
    )

    print(
        f"Context Recall     : "
        f"{context_recall_avg * 100:.2f}%"
    )

    print()

    print(
        f"Faithfulness valid : "
        f"{faithfulness_valid}/{len(questions)}"
    )

    print(
        f"Answer Rel. valid  : "
        f"{answer_relevancy_valid}/{len(questions)}"
    )

    print(
        f"Context Prec. valid: "
        f"{context_precision_valid}/{len(questions)}"
    )

    print(
        f"Context Recall valid: "
        f"{context_recall_valid}/{len(questions)}"
    )

    print()

    print(
        f"Questions          : "
        f"{len(questions)}"
    )

    print(
        f"Evaluation time    : "
        f"{elapsed_total:.2f} seconds"
    )

    print()

    print(
        f"[CSV SAVED] "
        f"{CSV_REPORT}"
    )

    print()
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )