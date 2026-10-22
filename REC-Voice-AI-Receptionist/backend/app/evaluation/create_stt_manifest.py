from __future__ import annotations

import csv
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
AUDIO_DIR = EVALUATION_DIR / "stt_audio"

MANIFEST_FILE = EVALUATION_DIR / "stt_test_manifest.csv"


# ============================================================
# IMPORT THE SAME 100 TEST QUESTIONS
# ============================================================

try:
    from app.evaluation.evaluate_rag import TEST_CASES

except Exception as exc:

    print()
    print("=" * 70)
    print("[IMPORT ERROR]")
    print(exc)
    print("=" * 70)
    print()
    print("Run this program from the backend directory:")
    print()
    print("python -m app.evaluation.create_stt_manifest")
    print()

    sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 75)
    print("REC AI RECEPTIONIST - STT TEST MANIFEST CREATOR")
    print("=" * 75)

    print()

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = [
        "clean",
        "noise",
        "telephone",
        "accent",
    ]

    for condition in conditions:

        (
            AUDIO_DIR / condition
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Create CSV
    # --------------------------------------------------------

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "id",
            "audio_file",
            "reference",
            "condition",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for test in TEST_CASES:

            question_id = test["id"]
            question = test["question"]

            for condition in conditions:

                audio_file = (
                    AUDIO_DIR
                    / condition
                    / f"q{question_id:03d}.wav"
                )

                writer.writerow(
                    {
                        "id": question_id,
                        "audio_file": str(
                            audio_file
                        ),
                        "reference": question,
                        "condition": condition,
                    }
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"[INFO] Total test cases: "
        f"{len(TEST_CASES)}"
    )

    print(
        "[INFO] Conditions: "
        "clean, noise, telephone, accent"
    )

    print(
        f"[INFO] Total manifest rows: "
        f"{len(TEST_CASES) * len(conditions)}"
    )

    print()

    print(
        f"[MANIFEST] {MANIFEST_FILE}"
    )

    print()

    print("=" * 75)
    print("MANIFEST CREATED")
    print("=" * 75)

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()