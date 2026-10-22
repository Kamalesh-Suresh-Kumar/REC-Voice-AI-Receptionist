
import csv
import wave
from pathlib import Path

from app.voice.tts_service import TTSService


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_FILE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "stt_test_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation_results"
    / "stt_audio"
    / "clean"
)

TTS_MODEL = (
    PROJECT_ROOT
    / "models"
    / "tts"
    / "en_US-lessac-medium.onnx"
)


def main():
    print("=" * 65)
    print("PIPER TTS - STT EVALUATION AUDIO GENERATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"STT manifest not found: {MANIFEST_FILE}"
        )

    if not TTS_MODEL.exists():
        raise FileNotFoundError(
            f"Piper model not found: {TTS_MODEL}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load clean questions from the existing manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_FILE,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "id",
            "reference",
            "condition",
        }

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Manifest must contain columns: "
                "id, reference, condition"
            )

        questions = [
            row
            for row in reader
            if row["condition"].strip().lower() == "clean"
        ]

    if not questions:
        raise ValueError(
            "No clean questions found in the manifest."
        )

    # --------------------------------------------------------
    # Load the existing Piper model once
    # --------------------------------------------------------

    print(f"Questions: {len(questions)}")
    print(f"Model: {TTS_MODEL}")

    tts = TTSService(model_path=TTS_MODEL)

    # --------------------------------------------------------
    # Generate WAV files
    # --------------------------------------------------------

    successful = 0
    failures = []

    for index, row in enumerate(questions, start=1):
        try:
            question_id = int(row["id"])
            reference = row["reference"].strip()

            if not reference:
                raise ValueError("Reference question is empty.")

            output_wav = (
                OUTPUT_DIR / f"q{question_id:03d}.wav"
            )

            print(
                f"\n[{index}/{len(questions)}] "
                f"Question ID: {question_id}"
            )
            print(f"Text: {reference}")

            tts.synthesize(
                reference,
                output_wav,
            )

            # Verify the generated WAV is readable and nonempty.
            with wave.open(str(output_wav), "rb") as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()

                if frames <= 0:
                    raise ValueError(
                        "Generated WAV contains no audio frames."
                    )

            print(
                f"Saved: {output_wav.name} | "
                f"Duration: {frames / sample_rate:.2f}s | "
                f"Rate: {sample_rate} Hz | "
                f"Channels: {channels}"
            )

            successful += 1

        except Exception as exc:
            print(f"[ERROR] Question ID {row.get('id')}: {exc}")
            failures.append(
                {
                    "id": row.get("id", ""),
                    "error": str(exc),
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AUDIO GENERATION SUMMARY")
    print("=" * 65)
    print(f"Total clean questions: {len(questions)}")
    print(f"Successful WAV files: {successful}")
    print(f"Failed questions: {len(failures)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if failures:
        failure_file = (
            PROJECT_ROOT
            / "evaluation_results"
            / "stt_tts_generation_errors.csv"
        )

        with open(
            failure_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["id", "error"],
            )
            writer.writeheader()
            writer.writerows(failures)

        print(f"Failure report: {failure_file}")

    if failures:
        raise SystemExit(
            "Some WAV files failed. Resolve the errors "
            "before starting WER/CER evaluation."
        )

    print("\nAll clean WAV files generated successfully.")


if __name__ == "__main__":
    main()
