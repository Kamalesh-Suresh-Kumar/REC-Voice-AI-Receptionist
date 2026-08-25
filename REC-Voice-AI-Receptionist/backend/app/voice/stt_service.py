from __future__ import annotations

from pathlib import Path

from app.voice.cuda_setup import configure_cuda_dll_paths

configure_cuda_dll_paths()

from faster_whisper import WhisperModel


class STTService:
    """
    Local Speech-to-Text service using Faster-Whisper.

    No cloud API.
    No API key.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        print("=" * 60)
        print("REC AI RECEPTIONIST - STT")
        print("=" * 60)

        print(
            f"[STT] Loading model: {model_size}"
        )
        print(
            f"[STT] Device: {device}"
        )
        print(
            f"[STT] Compute type: {compute_type}"
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print("[STT] Model loaded.")

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> str:
        """
        Transcribe a WAV/audio file.

        Returns only the combined transcript.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        print()
        print(
            f"[STT] Transcribing: {audio_path}"
        )

        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            language="en",
        )

        transcript_parts: list[str] = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            print(
                f"[STT] "
                f"{segment.start:.2f}s → "
                f"{segment.end:.2f}s: "
                f"{text}"
            )

            transcript_parts.append(text)

        transcript = " ".join(
            transcript_parts
        ).strip()

        print()
        print(
            f"[STT] Transcript: {transcript}"
        )

        print(
            f"[STT] Detected language: "
            f"{info.language}"
        )

        return transcript