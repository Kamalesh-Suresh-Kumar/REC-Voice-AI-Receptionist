from pathlib import Path

from app.voice.stt_service import STTService
from app.rag.rag_service import RAGService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_FILE = (
    PROJECT_ROOT
    / "call_20260819_195003.wav"
)


print("=" * 60)
print("REC AI RECEPTIONIST - VOICE BRAIN TEST")
print("=" * 60)


# --------------------------------------------------
# 1. Speech to Text
# --------------------------------------------------

print()
print("[1] SPEECH TO TEXT")

stt = STTService(
    model_size="small",
    device="cuda",
    compute_type="float16",
)

question = stt.transcribe(
    AUDIO_FILE
)


# --------------------------------------------------
# 2. RAG + Local LLM
# --------------------------------------------------

print()
print("[2] RAG + LOCAL LLM")

rag = RAGService()

result = rag.ask(
    question
)


# --------------------------------------------------
# 3. Final result
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(
    result["answer"]
)

print()
print("=" * 60)
print("SOURCES")
print("=" * 60)

for source in result["sources"]:
    print(source)