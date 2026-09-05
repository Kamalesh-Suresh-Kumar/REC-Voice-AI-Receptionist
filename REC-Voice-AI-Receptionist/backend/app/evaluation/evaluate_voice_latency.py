from __future__ import annotations

import csv
import statistics
import time
import wave
from pathlib import Path

from app.rag.rag_service import RAGService
from app.voice.stt_service import STTService
from app.voice.tts_service import TTSService


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = PROJECT_ROOT / "evaluation_results"
AUDIO_DIR = EVALUATION_DIR / "latency_test_audio"

CSV_FILE = EVALUATION_DIR / "voice_latency_results.csv"
GRAPH_FILE = EVALUATION_DIR / "voice_latency_graph.png"
AVERAGE_GRAPH_FILE = EVALUATION_DIR / "voice_latency_average_graph.png"

TTS_MODEL = PROJECT_ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "What should a student do if they fail a course from an earlier semester?",
    "Can I register for a subject that I could not attend the examination for previously?",
    "If I successfully complete my earlier semesters, can I take subjects that normally belong to a higher semester?",
    "What happens to my academic record if I have an attendance shortage?",
    "If I have a medical emergency, what documentation might the college require?",
    "How early should I complete my registration before the previous semester ends?",
    "Can I replace a failed elective with another elective?",
    "What are some of the main areas covered in the database course?",
    "What is the normal academic duration of an undergraduate engineering degree here?",
    "Are academic reports and examinations conducted in English?",
]


# ============================================================
# HELPERS
# ============================================================

def create_test_wav(path: Path, duration_seconds: float = 1.0) -> None:
    """
    Creates a short silent WAV file.

    NOTE:
    This file is only used so the STT/TTS pipeline can be tested
    without changing audio_gateway.py.

    For real STT latency, replace these files with actual recorded
    question WAV files.
    """

    sample_rate = 16000
    sample_width = 2
    channels = 1

    num_samples = int(sample_rate * duration_seconds)
    silence = b"\x00\x00" * num_samples

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)


def wav_to_8khz_pcm(input_wav: Path) -> bytes:
    """
    Convert generated TTS WAV to 8 kHz mono 16-bit PCM.
    """

    with wave.open(str(input_wav), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    # Standard library conversion.
    import audioop

    if channels == 2:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)

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


def save_results(results: list[dict]) -> None:

    fieldnames = [
        "test_id",
        "question",
        "stt_seconds",
        "rag_seconds",
        "llm_seconds",
        "tts_seconds",
        "total_seconds",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:

    metrics = [
        "stt_seconds",
        "rag_seconds",
        "llm_seconds",
        "tts_seconds",
        "total_seconds",
    ]

    print("\n" + "=" * 70)
    print("VOICE AI LATENCY SUMMARY")
    print("=" * 70)

    for metric in metrics:
        values = [float(row[metric]) for row in results]

        print(
            f"{metric:20s} "
            f"AVG: {statistics.mean(values):7.3f}s   "
            f"MIN: {min(values):7.3f}s   "
            f"MAX: {max(values):7.3f}s"
        )

    print("=" * 70)


def generate_graphs(results: list[dict]) -> None:

    import matplotlib.pyplot as plt

    test_ids = [int(row["test_id"]) for row in results]

    stt = [float(row["stt_seconds"]) for row in results]
    rag = [float(row["rag_seconds"]) for row in results]
    llm = [float(row["llm_seconds"]) for row in results]
    tts = [float(row["tts_seconds"]) for row in results]
    total = [float(row["total_seconds"]) for row in results]

    # ========================================================
    # GRAPH 1: COMPONENT LATENCY PER TEST
    # ========================================================

    plt.figure(figsize=(12, 7))

    plt.plot(test_ids, stt, marker="o", label="STT")
    plt.plot(test_ids, rag, marker="o", label="RAG")
    plt.plot(test_ids, llm, marker="o", label="LLM")
    plt.plot(test_ids, tts, marker="o", label="TTS")
    plt.plot(test_ids, total, marker="o", label="Total")

    plt.xlabel("Test Case")
    plt.ylabel("Time (seconds)")
    plt.title("Voice AI Pipeline Latency per Test Case")

    plt.xticks(test_ids)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)
    plt.close()

    # ========================================================
    # GRAPH 2: AVERAGE COMPONENT LATENCY
    # ========================================================

    labels = [
        "STT",
        "RAG",
        "LLM",
        "TTS",
        "Total",
    ]

    averages = [
        statistics.mean(stt),
        statistics.mean(rag),
        statistics.mean(llm),
        statistics.mean(tts),
        statistics.mean(total),
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, averages)

    plt.xlabel("Pipeline Component")
    plt.ylabel("Average Time (seconds)")
    plt.title("Average Voice AI Pipeline Latency")

    for bar, value in zip(bars, averages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(AVERAGE_GRAPH_FILE, dpi=300)
    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("REC VOICE AI RECEPTIONIST - LATENCY EVALUATION")
    print("=" * 70)

    print("\nLoading STT model...")
    stt = STTService()

    print("Loading RAG service...")
    rag = RAGService()

    print("Loading TTS service...")
    tts = TTSService(model_path=str(TTS_MODEL))

    print("\nAll services loaded successfully.")

    results = []

    for index, question in enumerate(TEST_QUESTIONS, start=1):

        print("\n" + "-" * 70)
        print(f"TEST {index}/{len(TEST_QUESTIONS)}")
        print(f"Question: {question}")
        print("-" * 70)

        pipeline_start = time.perf_counter()

        # ====================================================
        # STT
        # ====================================================

        audio_file = AUDIO_DIR / f"test_{index:02d}.wav"

        create_test_wav(audio_file)

        stt_start = time.perf_counter()

        try:
            transcript = stt.transcribe(str(audio_file))
        except Exception as e:
            print(f"[STT ERROR] {e}")
            transcript = question

        stt_end = time.perf_counter()

        stt_seconds = stt_end - stt_start

        print(f"STT       : {stt_seconds:.3f}s")
        print(f"Transcript: {transcript}")

        # ====================================================
        # RAG RETRIEVAL
        # ====================================================

        rag_start = time.perf_counter()

        retrieved_results = rag.retriever.search(
            query=question,
            top_k=rag.top_k,
        )

        context = rag._build_context(retrieved_results)

        rag_end = time.perf_counter()

        rag_seconds = rag_end - rag_start

        print(f"RAG       : {rag_seconds:.3f}s")

        # ====================================================
        # LLM
        # ====================================================

        llm_start = time.perf_counter()

        answer = rag.llm.generate(
            question=question,
            context=context,
        )

        llm_end = time.perf_counter()

        llm_seconds = llm_end - llm_start

        print(f"LLM       : {llm_seconds:.3f}s")

        # ====================================================
        # TTS
        # ====================================================

        output_wav = AUDIO_DIR / f"response_{index:02d}.wav"

        tts_start = time.perf_counter()

        tts.synthesize(
            answer,
            output_wav,
        )

        # Read output so TTS preparation is included.
        _ = wav_to_8khz_pcm(output_wav)

        tts_end = time.perf_counter()

        tts_seconds = tts_end - tts_start

        print(f"TTS       : {tts_seconds:.3f}s")

        # ====================================================
        # TOTAL
        # ====================================================

        pipeline_end = time.perf_counter()

        total_seconds = pipeline_end - pipeline_start

        print(f"TOTAL     : {total_seconds:.3f}s")

        results.append(
            {
                "test_id": index,
                "question": question,
                "stt_seconds": round(stt_seconds, 3),
                "rag_seconds": round(rag_seconds, 3),
                "llm_seconds": round(llm_seconds, 3),
                "tts_seconds": round(tts_seconds, 3),
                "total_seconds": round(total_seconds, 3),
            }
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_results(results)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print_summary(results)

    # ========================================================
    # GENERATE GRAPHS
    # ========================================================

    print("\nGenerating graphs...")

    generate_graphs(results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(f"\nCSV:")
    print(CSV_FILE)

    print(f"\nGraph 1:")
    print(GRAPH_FILE)

    print(f"\nGraph 2:")
    print(AVERAGE_GRAPH_FILE)

    print("\nEvaluation folder:")
    print(EVALUATION_DIR)


if __name__ == "__main__":
    main()