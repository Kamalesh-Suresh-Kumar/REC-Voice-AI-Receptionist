
from pathlib import Path
import csv
import re
import string
from collections import defaultdict

from app.voice.stt_service import STTService


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation_results" / "stt_test_manifest.csv"
)
AUDIO_ROOT = PROJECT_ROOT / "evaluation_results" / "stt_audio"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

CONDITIONS = ("clean", "noise", "telephone")


def normalize_text(text):
    """Lowercase, remove punctuation, and normalize whitespace."""
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_transcript(result):
    """Handle common string, dictionary, and tuple return formats."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("text", "transcript", "transcription"):
            if key in result:
                return extract_transcript(result[key])
        raise TypeError(
            f"Could not identify transcript in dictionary keys: "
            f"{list(result.keys())}"
        )

    if isinstance(result, (tuple, list)):
        if not result:
            return ""
        return extract_transcript(result[0])

    if hasattr(result, "text"):
        return extract_transcript(result.text)

    raise TypeError(
        f"Unsupported STT return type: {type(result).__name__}"
    )


def edit_distance(reference, hypothesis):
    """Levenshtein edit distance using insertion, deletion, substitution."""
    previous = list(range(len(hypothesis) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1

            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def error_counts(reference_text, hypothesis_text):
    """Calculate word and character edit counts."""
    reference = normalize_text(reference_text)
    hypothesis = normalize_text(hypothesis_text)

    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []

    word_edits = edit_distance(ref_words, hyp_words)
    char_edits = edit_distance(reference, hypothesis)

    return {
        "reference_words": len(ref_words),
        "reference_characters": len(reference),
        "word_edits": word_edits,
        "character_edits": char_edits,
        "wer": word_edits / len(ref_words) if ref_words else 0.0,
        "cer": char_edits / len(reference) if reference else 0.0,
    }


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    rows = []

    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required = {"id", "reference", "condition"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        for row in reader:
            condition = row["condition"].strip().lower()

            # Accent evaluation is excluded until genuine accent-varied
            # speech recordings are available.
            if condition not in CONDITIONS:
                continue

            rows.append({
                "id": int(row["id"]),
                "reference": row["reference"],
                "condition": condition,
            })

    if not rows:
        raise ValueError(
            "No clean, noise, or telephone rows found in manifest."
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest()
    print(f"Manifest samples selected: {len(manifest_rows)}")
    print("Conditions:", ", ".join(CONDITIONS))
    print("Loading Faster-Whisper STT service...")

    # Use the existing production STT implementation unchanged.
    stt = STTService()

    detailed_results = []
    totals = defaultdict(
        lambda: {
            "samples": 0,
            "successful": 0,
            "failed": 0,
            "word_edits": 0,
            "reference_words": 0,
            "character_edits": 0,
            "reference_characters": 0,
        }
    )

    for index, row in enumerate(manifest_rows, start=1):
        question_id = row["id"]
        condition = row["condition"]
        reference = row["reference"]

        audio_path = (
            AUDIO_ROOT
            / condition
            / f"q{question_id:03d}.wav"
        )

        hypothesis = ""
        status = "ok"
        error_message = ""

        try:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            result = stt.transcribe(str(audio_path))
            hypothesis = extract_transcript(result)

            if not hypothesis.strip():
                status = "empty_transcription"

        except Exception as exc:
            # Do not replace a failed transcription with the reference.
            # An empty hypothesis is scored as deletion errors.
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"\nFAILED {condition} q{question_id:03d}: "
                f"{error_message}"
            )

        metrics = error_counts(reference, hypothesis)

        if status == "ok":
            pass
        elif status == "empty_transcription":
            # Keep empty output visible as a failure.
            pass

        record = {
            "id": question_id,
            "condition": condition,
            "audio_file": str(audio_path),
            "reference": reference,
            "hypothesis": hypothesis,
            "status": status,
            "error": error_message,
            "reference_words": metrics["reference_words"],
            "reference_characters": metrics["reference_characters"],
            "word_edits": metrics["word_edits"],
            "character_edits": metrics["character_edits"],
            "WER": metrics["wer"],
            "CER": metrics["cer"],
        }
        detailed_results.append(record)

        summary = totals[condition]
        summary["samples"] += 1
        summary["word_edits"] += metrics["word_edits"]
        summary["reference_words"] += metrics["reference_words"]
        summary["character_edits"] += metrics["character_edits"]
        summary["reference_characters"] += metrics["reference_characters"]

        if status == "ok":
            summary["successful"] += 1
        else:
            summary["failed"] += 1

        if index % 10 == 0 or index == len(manifest_rows):
            print(
                f"\rEvaluated {index}/{len(manifest_rows)} samples",
                end="",
                flush=True,
            )

    print("\n")

    # Save all per-sample transcripts and errors.
    detail_path = OUTPUT_DIR / "stt_detailed_results.csv"

    with detail_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(detailed_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(detailed_results)

    # Aggregate edit counts across all references in each condition.
    summary_rows = []

    for condition in CONDITIONS:
        item = totals[condition]

        if item["samples"] == 0:
            continue

        total_words = item["reference_words"]
        total_chars = item["reference_characters"]

        wer = (
            item["word_edits"] / total_words
            if total_words else 0.0
        )
        cer = (
            item["character_edits"] / total_chars
            if total_chars else 0.0
        )

        summary_rows.append({
            "condition": condition,
            "samples": item["samples"],
            "successful": item["successful"],
            "failed_or_empty": item["failed"],
            "total_reference_words": total_words,
            "total_word_edits": item["word_edits"],
            "WER": wer,
            "WER_percent": wer * 100,
            "total_reference_characters": total_chars,
            "total_character_edits": item["character_edits"],
            "CER": cer,
            "CER_percent": cer * 100,
        })

    summary_path = OUTPUT_DIR / "stt_condition_summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("========== STT EVALUATION SUMMARY ==========")

    for item in summary_rows:
        print(
            f"{item['condition']:10s} | "
            f"samples={item['samples']:3d} | "
            f"failed/empty={item['failed_or_empty']:3d} | "
            f"WER={item['WER_percent']:.2f}% | "
            f"CER={item['CER_percent']:.2f}%"
        )

    print("\nDetailed results:", detail_path)
    print("Condition summary:", summary_path)
    print(
        "\nNote: WER/CER are aggregate edit counts divided by "
        "total reference word/character counts. "
        "WER/CER may exceed 100%."
    )


if __name__ == "__main__":
    main()
