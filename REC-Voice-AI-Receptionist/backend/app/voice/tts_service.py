from __future__ import annotations

from pathlib import Path
import wave

from piper import PiperVoice


class TTSService:
    """
    Local Text-to-Speech service using Piper.
    """

    def __init__(
        self,
        model_path: str | Path,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TTS model not found: {self.model_path}"
            )

        print("=" * 60)
        print("REC AI RECEPTIONIST - TTS")
        print("=" * 60)

        print(
            f"[TTS] Loading model: {self.model_path}"
        )

        self.voice = PiperVoice.load(
            str(self.model_path)
        )

        print("[TTS] Model loaded.")

    def synthesize(
     self,
     text: str,
     output_path: str | Path,
     ) -> Path:

     output_path = Path(output_path)

     output_path.parent.mkdir(
          parents=True,
          exist_ok=True,
     )

     print(f"[TTS] Synthesizing: {text}")

     with wave.open(
          str(output_path),
          "wb",
     ) as wav_file:

          first_chunk = True

          for chunk in self.voice.synthesize(text):

               if first_chunk:
                    wav_file.setframerate(
                         chunk.sample_rate
                    )

                    wav_file.setsampwidth(
                         chunk.sample_width
                    )

                    wav_file.setnchannels(
                         chunk.sample_channels
                    )

                    first_chunk = False

               wav_file.writeframes(
                    chunk.audio_int16_bytes
               )

     print(
          f"[TTS] Saved: {output_path}"
     )

     return output_path