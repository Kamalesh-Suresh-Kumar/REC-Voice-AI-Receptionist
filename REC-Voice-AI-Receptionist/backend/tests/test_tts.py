from pathlib import Path

from app.voice.tts_service import TTSService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tts"
    / "en_US-lessac-medium.onnx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "temp_audio"
    / "tts_test.wav"
)


def main():

    print("=" * 60)
    print("REC AI RECEPTIONIST - TTS TEST")
    print("=" * 60)

    tts = TTSService(
        model_path=MODEL_PATH
    )

    text = (
        "Hello. This is the REC AI receptionist. "
        "How can I help you?"
    )

    output = tts.synthesize(
        text=text,
        output_path=OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("TTS TEST COMPLETE")
    print("=" * 60)
    print(f"Audio: {output}")


if __name__ == "__main__":
    main()