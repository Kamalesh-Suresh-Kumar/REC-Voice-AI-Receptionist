from pathlib import Path

from app.voice.stt_service import STTService


# Change this if your WAV has a different name.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_FILE = (
    PROJECT_ROOT
    / "call_20260825_091706.wav"
)


print("=" * 60)
print("REC AI RECEPTIONIST - STT TEST")
print("=" * 60)

print()
print(f"Audio file: {AUDIO_FILE}")

stt = STTService(
    model_size="small",
    device="cuda",
    compute_type="float16",
)

transcript = stt.transcribe(
    AUDIO_FILE
)

print()
print("=" * 60)
print("FINAL TRANSCRIPT")
print("=" * 60)

print(transcript)