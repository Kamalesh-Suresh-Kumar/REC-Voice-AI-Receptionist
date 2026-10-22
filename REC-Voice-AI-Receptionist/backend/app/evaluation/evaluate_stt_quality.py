from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path
from typing import Iterable

from app.voice.stt_service import STTService


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
AUDIO_DIR = EVALUATION_DIR / "stt_quality_audio"
MANIFEST_FILE = EVALUATION_DIR / "stt_test_manifest.csv"

RESULTS_FILE = EVALUATION_DIR / "stt_quality_results.csv"
SUMMARY_FILE = EVALUATION_DIR / "stt_quality_summary.csv"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> list[str]:
    """
    Normalize a transcript for WER/CER evaluation.

    - Converts to lowercase
    - Removes punctuation
    - Collapses whitespace
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.split() if text else []


def normalized_string(text: str) -> str:
    return " ".join(normalize_text(text))


# ============================================================
# EDIT DISTANCE
# ============================================================

def levenshtein(reference: Iterable, hypothesis: Iterable) -> int:
    """
    Calculate Levenshtein edit distance.
    """
    ref = list(reference)
    hyp = list(hypothesis)

    if not ref:
        return len(hyp)

    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))

    for i, ref_item in enumerate(ref, start=1):
        current = [i]

        for j, hyp_item in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (
                0 if ref_item == hyp_item else 1
            )
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            current.append(
                min(
                    substitution,
                    insertion,
                    deletion,
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# WER / CER
# ============================================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate.

    WER = word edit distance / number of reference words
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return levenshtein(ref_words, hyp_words) / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate.

    CER = character edit distance / number of reference characters
    """
    ref_chars = list(normalized_string(reference).replace(" ", ""))
    hyp_chars = list(normalized_string(hypothesis).replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


# ============================================================
# MANIFEST
# ============================================================

REQUIRED_COLUMNS = {
    "audio_file",
    "reference",
    "condition",
}


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_FILE}\n\n"
            "Create stt_test_manifest.csv with columns:\n"
            "audio_file,reference,condition"
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "The STT manifest has no header."
            )

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "STT manifest is missing columns: "
                + ", ".join(sorted(missing))
            )

        rows = []

        for row in reader:
            audio_file = str(
                row.get("audio_file", "")
            ).strip()

            reference = str(
                row.get("reference", "")
            ).strip()

            condition = str(
                row.get("condition", "")
            ).strip().lower()

            if not audio_file:
                continue

            if not reference:
                raise ValueError(
                    f"Missing reference transcript for: "
                    f"{audio_file}"
                )

            if not condition:
                raise ValueError(
                    f"Missing condition for: "
                    f"{audio_file}"
                )

            rows.append(
                {
                    "audio_file": audio_file,
                    "reference": reference,
                    "condition": condition,
                }
            )

    if not rows:
        raise ValueError(
            "The STT manifest contains no test cases."
        )

    return rows


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:
    print("=" * 75)
    print("REC VOICE AI RECEPTIONIST - STT QUALITY EVALUATION")
    print("=" * 75)

    print(f"\nManifest: {MANIFEST_FILE}")

    rows = load_manifest()

    print(
        f"[INFO] Test recordings: {len(rows)}"
    )

    print("\nLoading STT service...")

    stt = STTService()

    print("[STARTUP] STT service ready.")

    results: list[dict] = []

    for index, row in enumerate(rows, start=1):

        audio_path = (
            EVALUATION_DIR
            / row["audio_file"]
        )

        if not audio_path.exists():
            print(
                f"\n[SKIP] Missing audio: "
                f"{audio_path}"
            )
            continue

        reference = row["reference"]
        condition = row["condition"]

        print()
        print("-" * 75)
        print(
            f"[{index}/{len(rows)}] "
            f"{audio_path.name}"
        )
        print(f"Condition : {condition}")
        print(f"Reference : {reference}")

        try:
            hypothesis = stt.transcribe(
                str(audio_path)
            ).strip()

        except Exception as exc:
            print(f"[STT ERROR] {exc}")
            hypothesis = ""

        wer = calculate_wer(
            reference,
            hypothesis,
        )

        cer = calculate_cer(
            reference,
            hypothesis,
        )

        print(f"Hypothesis: {hypothesis}")
        print(
            f"WER       : {wer:.4f} "
            f"({wer * 100:.2f}%)"
        )
        print(
            f"CER       : {cer:.4f} "
            f"({cer * 100:.2f}%)"
        )

        results.append(
            {
                "audio_file": row["audio_file"],
                "condition": condition,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
            }
        )

    if not results:
        raise RuntimeError(
            "No valid audio recordings were evaluated."
        )

    # ========================================================
    # SAVE PER-RECORDING RESULTS
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "audio_file",
            "condition",
            "reference",
            "hypothesis",
            "wer",
            "cer",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # SUMMARY BY CONDITION
    # ========================================================

    conditions = sorted(
        {
            row["condition"]
            for row in results
        }
    )

    summary_rows = []

    print()
    print("=" * 75)
    print("STT QUALITY SUMMARY")
    print("=" * 75)

    for condition in conditions:

        condition_results = [
            row
            for row in results
            if row["condition"] == condition
        ]

        wer_values = [
            float(row["wer"])
            for row in condition_results
        ]

        cer_values = [
            float(row["cer"])
            for row in condition_results
        ]

        average_wer = statistics.mean(
            wer_values
        )

        average_cer = statistics.mean(
            cer_values
        )

        summary_rows.append(
            {
                "condition": condition,
                "test_count": len(
                    condition_results
                ),
                "average_wer": round(
                    average_wer,
                    4,
                ),
                "average_wer_percent": round(
                    average_wer * 100,
                    2,
                ),
                "average_cer": round(
                    average_cer,
                    4,
                ),
                "average_cer_percent": round(
                    average_cer * 100,
                    2,
                ),
            }
        )

        print(
            f"{condition:15s} "
            f"Tests: {len(condition_results):3d}   "
            f"WER: {average_wer * 100:7.2f}%   "
            f"CER: {average_cer * 100:7.2f}%"
        )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "condition",
            "test_count",
            "average_wer",
            "average_wer_percent",
            "average_cer",
            "average_cer_percent",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 75)
    print("STT QUALITY EVALUATION COMPLETED")
    print("=" * 75)
    print(f"\nDetailed results:")
    print(RESULTS_FILE)
    print(f"\nCondition summary:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
