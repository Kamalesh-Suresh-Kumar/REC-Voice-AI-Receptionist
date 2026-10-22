
from __future__ import annotations

import csv
import statistics
import time
import wave
from pathlib import Path

from app.rag.rag_service import RAGService
from app.voice.stt_service import STTService
from app.voice.tts_service import TTSService


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
AUDIO_DIR = EVALUATION_DIR / "stt_audio" / "clean"
MANIFEST_FILE = EVALUATION_DIR / "stt_test_manifest.csv"

CSV_FILE = EVALUATION_DIR / "voice_latency_results.csv"
SUMMARY_FILE = EVALUATION_DIR / "voice_latency_summary.txt"
GRAPH_FILE = EVALUATION_DIR / "voice_latency_average_graph.png"

LATENCY_AUDIO_DIR = EVALUATION_DIR / "latency_test_audio"
TTS_MODEL = PROJECT_ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"

# Evaluate up to 100 real clean synthetic-speech WAV samples.
MAX_SAMPLES = 100

# Keep more precision for short stages and near-zero rule-based responses.
DECIMAL_PLACES = 4

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
LATENCY_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load test samples
# ---------------------------------------------------------------------

def load_samples() -> list[dict]:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"STT manifest not found: {MANIFEST_FILE}"
        )

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Clean STT audio directory not found: {AUDIO_DIR}"
        )

    with MANIFEST_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    clean_rows = [
        row
        for row in rows
        if row.get("condition", "").strip().lower() == "clean"
    ]

    samples = []

    for row in clean_rows:
        try:
            sample_id = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue

        audio_path = AUDIO_DIR / f"q{sample_id:03d}.wav"

        if not audio_path.exists():
            print(f"[SKIP] Missing audio file: {audio_path}")
            continue

        samples.append(
            {
                "test_id": sample_id,
                "reference": row.get("reference", "").strip(),
                "audio_path": audio_path,
            }
        )

    samples.sort(key=lambda item: item["test_id"])

    if not samples:
        raise RuntimeError(
            "No clean audio samples were found. "
            "Check the manifest and audio directory."
        )

    selected_samples = samples[:MAX_SAMPLES]

    if len(selected_samples) < MAX_SAMPLES:
        print(
            f"[WARNING] Requested {MAX_SAMPLES} samples, "
            f"but only {len(selected_samples)} valid clean audio "
            "samples are available."
        )

    return selected_samples


# ---------------------------------------------------------------------
# Convert TTS output to 8 kHz mono 16-bit PCM
# ---------------------------------------------------------------------

def convert_wav_to_8khz_pcm(input_wav: Path) -> bytes:
    """Prepare synthesized WAV audio as 8 kHz mono 16-bit PCM."""

    try:
        import audioop
    except ImportError as exc:
        raise RuntimeError(
            "The audioop module is unavailable in this Python version. "
            "Use a compatible Python version or replace this conversion "
            "with an installed audio-processing library."
        ) from exc

    with wave.open(str(input_wav), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM audio, got sample width "
            f"{sample_width} bytes."
        )

    if channels == 2:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(
            f"Unsupported TTS channel count: {channels}"
        )

    if sample_rate != 8000:
        pcm, _ = audioop.ratecv(
            pcm,
            sample_width,
            1,
            sample_rate,
            8000,
            None,
        )

    return pcm


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_csv(results: list[dict]) -> None:
    fieldnames = [
        "test_id",
        "reference",
        "transcript",
        "status",
        "error",
        "stt_seconds",
        "rag_retrieval_seconds",
        "llm_seconds",
        "tts_seconds",
        "total_seconds",
    ]

    with CSV_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def write_summary(results: list[dict]) -> None:
    successful = [
        row for row in results if row["status"] == "ok"
    ]

    failed = [
        row for row in results if row["status"] != "ok"
    ]

    metrics = [
        ("stt_seconds", "STT transcription"),
        ("rag_retrieval_seconds", "RAG retrieval/context"),
        ("llm_seconds", "LLM/service response stage"),
        ("tts_seconds", "TTS synthesis + audio conversion"),
        ("total_seconds", "Total offline pipeline"),
    ]

    lines = [
        "REC Voice AI Receptionist - Offline Latency Evaluation",
        "=" * 60,
        f"Samples attempted: {len(results)}",
        f"Successful end-to-end samples: {len(successful)}",
        f"Failed samples: {len(failed)}",
        "",
        "Evaluation notes:",
        "- Uses actual clean synthetic-speech WAV files.",
        "- STT uses the recognized transcript, not the reference text.",
        "- Failed samples are excluded from successful-sample averages.",
        "- LLM/service timing may include deterministic rule-based answers.",
        "- Total time measures the offline pipeline stages, not a complete "
        "live telephone call.",
        "",
        "Averages below use successful samples only.",
        "",
    ]

    for field, label in metrics:
        values = [
            float(row[field])
            for row in successful
            if row[field] != ""
        ]

        if values:
            lines.append(
                f"{label}: "
                f"average={statistics.mean(values):.3f}s, "
                f"min={min(values):.3f}s, "
                f"max={max(values):.3f}s"
            )
        else:
            lines.append(
                f"{label}: no successful measurements"
            )

    text = "\n".join(lines)

    SUMMARY_FILE.write_text(text, encoding="utf-8")

    print("\n" + text)


# ---------------------------------------------------------------------
# Average latency graph
# ---------------------------------------------------------------------

def make_average_graph(results: list[dict]) -> None:
    successful = [
        row for row in results if row["status"] == "ok"
    ]

    if not successful:
        print(
            "[WARNING] No successful samples; graph not generated."
        )
        return

    import matplotlib.pyplot as plt

    fields = [
        ("stt_seconds", "STT"),
        ("rag_retrieval_seconds", "RAG retrieval"),
        ("llm_seconds", "LLM/service"),
        ("tts_seconds", "TTS"),
        ("total_seconds", "Total pipeline"),
    ]

    labels = []
    averages = []

    for field, label in fields:
        values = [
            float(row[field])
            for row in successful
            if row[field] != ""
        ]

        if values:
            labels.append(label)
            averages.append(statistics.mean(values))

    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, averages)

    plt.xlabel("Pipeline stage")
    plt.ylabel("Average time (seconds)")
    plt.title(
        "Average Offline Voice AI Receptionist Pipeline Latency"
    )

    for bar, value in zip(bars, averages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)
    plt.close()

    print(f"Average latency graph: {GRAPH_FILE}")


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("REC VOICE AI RECEPTIONIST - OFFLINE LATENCY EVALUATION")
    print("=" * 70)

    print(
        "Using actual clean synthetic-speech WAV files; "
        "no silent audio or transcript fallback."
    )

    samples = load_samples()

    print(f"Samples selected: {len(samples)}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Output directory: {EVALUATION_DIR}")

    print("\nLoading STT model...")
    stt = STTService()

    print("Loading RAG service...")
    rag = RAGService()

    print("Loading TTS service...")
    tts = TTSService(model_path=str(TTS_MODEL))

    print("All services loaded.")

    results = []

    for index, sample in enumerate(samples, start=1):
        test_id = sample["test_id"]
        audio_path = sample["audio_path"]
        reference = sample["reference"]

        print("\n" + "-" * 70)
        print(
            f"Sample {index}/{len(samples)} | "
            f"ID {test_id} | {audio_path.name}"
        )

        pipeline_start = time.perf_counter()

        row = {
            "test_id": test_id,
            "reference": reference,
            "transcript": "",
            "status": "failed",
            "error": "",
            "stt_seconds": "",
            "rag_retrieval_seconds": "",
            "llm_seconds": "",
            "tts_seconds": "",
            "total_seconds": "",
        }

        try:
            # ---------------------------------------------------------
            # 1. Speech-to-text transcription
            # ---------------------------------------------------------
            stage_start = time.perf_counter()

            transcript = stt.transcribe(str(audio_path))

            stt_elapsed = time.perf_counter() - stage_start
            row["stt_seconds"] = round(
                stt_elapsed, DECIMAL_PLACES
            )

            transcript = (
                str(transcript).strip()
                if transcript is not None
                else ""
            )

            row["transcript"] = transcript

            print(
                f"STT ({stt_elapsed:.4f}s): {transcript}"
            )

            if not transcript:
                raise RuntimeError(
                    "STT returned an empty transcript. "
                    "Sample excluded from successful metrics."
                )

            # ---------------------------------------------------------
            # 2. RAG retrieval and context preparation
            # ---------------------------------------------------------
            stage_start = time.perf_counter()

            retrieved_results = rag.retriever.search(
                query=transcript,
                top_k=rag.top_k,
            )

            context = rag._build_context(retrieved_results)

            rag_elapsed = time.perf_counter() - stage_start
            row["rag_retrieval_seconds"] = round(
                rag_elapsed, DECIMAL_PLACES
            )

            print(
                f"RAG retrieval/context: {rag_elapsed:.4f}s"
            )

            # ---------------------------------------------------------
            # 3. LLM/service response stage
            # ---------------------------------------------------------
            stage_start = time.perf_counter()

            answer = rag.llm.generate(
                question=transcript,
                context=context,
            )

            llm_elapsed = time.perf_counter() - stage_start
            row["llm_seconds"] = round(
                llm_elapsed, DECIMAL_PLACES
            )

            answer_length = (
                len(answer)
                if isinstance(answer, str)
                else "non-string"
            )

            print(
                f"LLM/service response: {llm_elapsed:.4f}s | "
                f"answer length: {answer_length}"
            )

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "LLM/service returned an empty answer."
                )

            # ---------------------------------------------------------
            # 4. Text-to-speech synthesis and audio preparation
            # ---------------------------------------------------------
            output_wav = (
                LATENCY_AUDIO_DIR
                / f"response_{test_id:03d}.wav"
            )

            stage_start = time.perf_counter()

            tts.synthesize(answer, output_wav)

            if not output_wav.exists():
                raise FileNotFoundError(
                    f"TTS output WAV was not created: {output_wav}"
                )

            _ = convert_wav_to_8khz_pcm(output_wav)

            tts_elapsed = time.perf_counter() - stage_start
            row["tts_seconds"] = round(
                tts_elapsed, DECIMAL_PLACES
            )

            print(
                f"TTS synthesis + conversion: {tts_elapsed:.4f}s"
            )

            row["status"] = "ok"

        except Exception as exc:
            row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(f"[FAILED] {row['error']}")

        finally:
            total_elapsed = time.perf_counter() - pipeline_start

            row["total_seconds"] = round(
                total_elapsed, DECIMAL_PLACES
            )

            print(
                f"Total elapsed for sample: {total_elapsed:.4f}s"
            )

            results.append(row)

            # Save progress after every sample.
            save_csv(results)

    # -----------------------------------------------------------------
    # Save final evaluation outputs
    # -----------------------------------------------------------------
    write_summary(results)
    make_average_graph(results)

    print("\nEvaluation completed.")
    print(f"Detailed CSV: {CSV_FILE}")
    print(f"Summary text: {SUMMARY_FILE}")
    print(f"Average graph: {GRAPH_FILE}")


if __name__ == "__main__":
    main()
