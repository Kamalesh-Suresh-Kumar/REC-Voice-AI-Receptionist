from __future__ import annotations

from datetime import datetime

import audioop
import socket
import struct
import threading
import time
import uuid
import wave
import webrtcvad 

from pathlib import Path

from app.rag.rag_service import RAGService
from app.voice.stt_service import STTService
from app.voice.tts_service import TTSService


# ============================================================
# SERVER CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 9019

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMP_AUDIO = PROJECT_ROOT / "temp_audio"

TTS_MODEL = (
    PROJECT_ROOT
    / "models"
    / "tts"
    / "en_US-lessac-medium.onnx"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

AUDIO_TYPE = 0x10

SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1

# 20 ms @ 8000 Hz / 16-bit / mono
FRAME_BYTES = 320


# ============================================================
# AUDIOSOCKET FRAME TYPES
# ============================================================

FRAME_HANGUP = 0x00
FRAME_UUID = 0x01
FRAME_DTMF = 0x03
FRAME_AUDIO = 0x10
FRAME_ERROR = 0xFF


# ============================================================
# VAD CONFIGURATION
# ============================================================

VAD_MODE = 3

VAD_FRAME_MS = 20

VAD_FRAME_BYTES = (
    SAMPLE_RATE
    * VAD_FRAME_MS
    // 1000
    * SAMPLE_WIDTH
)

# 3 x 20 ms = 60 ms
SPEECH_START_FRAMES = 3

# End speech after 0.7 seconds of silence.
SILENCE_TIMEOUT_SECONDS = 0.7

SILENCE_FRAMES = int(
    SILENCE_TIMEOUT_SECONDS
    * 1000
    / VAD_FRAME_MS
)

# Safety limit.
MAX_UTTERANCE_SECONDS = 30

MAX_UTTERANCE_BYTES = int(
    MAX_UTTERANCE_SECONDS
    * SAMPLE_RATE
    * SAMPLE_WIDTH
)


# ============================================================
# CONVERSATION CONFIGURATION
# ============================================================

# 5 x 20 ms = 100 ms
TRAILING_SILENCE_FRAMES = 5

# Prevent immediate retrigger after TTS.
TTS_COOLDOWN_SECONDS = 1.0


# ============================================================
# AUDIOSOCKET KEEPALIVE
# ============================================================

KEEPALIVE_INTERVAL_SECONDS = 1.0

SILENCE_FRAME = b"\x00" * FRAME_BYTES


# ============================================================
# RECEPTIONIST GREETING
# ============================================================

def get_time_based_greeting() -> str:
    """
    Generate a 24/7 receptionist greeting based on
    the local time of the machine running this server.

    Good night is intentionally avoided because the
    receptionist is available 24/7.
    """

    current_hour = datetime.now().hour

    if 5 <= current_hour < 12:

        return (
            "Good morning. "
            "I am the REC AI Receptionist. "
            "How can I help you?"
        )

    if 12 <= current_hour < 17:

        return (
            "Good afternoon. "
            "I am the REC AI Receptionist. "
            "How can I help you?"
        )

    if 17 <= current_hour < 21:

        return (
            "Good evening. "
            "I am the REC AI Receptionist. "
            "How can I help you?"
        )

    return (
        "Good evening. "
        "I hope you're having a pleasant night. "
        "I am the REC AI Receptionist. "
        "How can I help you?"
    )


# ============================================================
# FAREWELL CONFIGURATION
# ============================================================

FAREWELL_PHRASES = {
    "bye",
    "goodbye",
    "good bye",
    "thank you",
    "thanks",
    "that's all",
    "that is all",
    "thats all",
    "no more questions",
    "i am done",
    "im done",
    "i'm done",
    "i am finished",
    "im finished",
    "i'm finished",
    "finish the call",
    "end the call",
    "hang up",
}


def normalize_text(text: str) -> str:
    """
    Normalize transcript text for simple intent detection.
    """

    return (
        text
        .lower()
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
    )


def is_farewell_request(
    transcript: str,
) -> bool:
    """
    Detect common caller farewell requests.

    This intentionally handles both exact phrases and
    common natural-language variations.
    """

    normalized = normalize_text(
        transcript
    )

    if normalized in FAREWELL_PHRASES:

        return True

    # Natural conversational variations.
    farewell_patterns = (
        "that's all",
        "that is all",
        "no more questions",
        "i am done",
        "i'm done",
        "im done",
        "i am finished",
        "i'm finished",
        "im finished",
        "end the call",
        "hang up",
        "goodbye",
        "good bye",
        "thank you goodbye",
        "thanks goodbye",
    )

    for phrase in farewell_patterns:

        if phrase in normalized:

            return True

    return False


def get_farewell_message() -> str:
    """
    Generate a time-appropriate final receptionist message.
    """

    current_hour = datetime.now().hour

    if 5 <= current_hour < 17:

        return (
            "Thank you for calling REC. "
            "Have a great day. "
            "Goodbye."
        )

    if 17 <= current_hour < 21:

        return (
            "Thank you for calling REC. "
            "Have a pleasant evening. "
            "Goodbye."
        )

    return (
        "Thank you for calling REC. "
        "Have a pleasant night. "
        "Goodbye."
    )


# ============================================================
# AUDIOSOCKET RECEIVE
# ============================================================

def recv_exact(
    conn: socket.socket,
    size: int,
) -> bytes | None:
    """
    Receive exactly `size` bytes.
    """

    data = bytearray()

    while len(data) < size:

        chunk = conn.recv(
            size - len(data)
        )

        if not chunk:

            return None

        data.extend(chunk)

    return bytes(data)


def read_audiosocket_frame(
    conn: socket.socket,
) -> tuple[int, bytes] | None:
    """
    Read one AudioSocket frame.

    Header:

        1 byte  frame type
        2 bytes payload length
    """

    header = recv_exact(
        conn,
        3,
    )

    if header is None:

        return None

    frame_type = header[0]

    payload_length = struct.unpack(
        "!H",
        header[1:3],
    )[0]

    if payload_length == 0:

        return frame_type, b""

    payload = recv_exact(
        conn,
        payload_length,
    )

    if payload is None:

        return None

    return frame_type, payload


# ============================================================
# AUDIOSOCKET SEND
# ============================================================

def send_audiosocket_frame(
    conn: socket.socket,
    frame_type: int,
    payload: bytes,
) -> None:
    """
    Send exactly one AudioSocket frame.

    The caller must hold send_lock when multiple threads
    are sending on the same connection.
    """

    header = struct.pack(
        "!BH",
        frame_type,
        len(payload),
    )

    conn.sendall(
        header + payload
    )


# ============================================================
# WAV -> 8 KHZ PCM
# ============================================================

def wav_to_pcm_8khz(
    wav_path: Path,
) -> bytes:
    """
    Convert WAV into:

        8000 Hz
        16-bit PCM
        mono
    """

    with wave.open(
        str(wav_path),
        "rb",
    ) as wav_file:

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

        pcm = wav_file.readframes(
            wav_file.getnframes()
        )

    print(
        "[TTS] WAV:"
        f" channels={channels}"
        f" width={sample_width}"
        f" rate={sample_rate}"
    )

    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if channels == 2:

        pcm = audioop.tomono(
            pcm,
            sample_width,
            0.5,
            0.5,
        )

        channels = 1

    elif channels != 1:

        raise RuntimeError(
            f"Unsupported channels: {channels}"
        )

    # --------------------------------------------------------
    # Convert sample width to 16-bit.
    # --------------------------------------------------------

    if sample_width != 2:

        pcm = audioop.lin2lin(
            pcm,
            sample_width,
            2,
        )

        sample_width = 2

    # --------------------------------------------------------
    # Resample to 8000 Hz.
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        pcm, _ = audioop.ratecv(
            pcm,
            2,
            1,
            sample_rate,
            SAMPLE_RATE,
            None,
        )

    print(
        "[TTS] Converted:"
        " 16-bit / 8000 Hz / mono"
    )

    print(
        f"[TTS] PCM size: {len(pcm)} bytes"
    )

    return pcm


# ============================================================
# SEND TTS AUDIO
# ============================================================

def send_audio(
    conn: socket.socket,
    pcm_data: bytes,
    send_lock: threading.Lock,
) -> None:
    """
    Send PCM to Asterisk as 20 ms AudioSocket frames.

    The complete playback is protected by send_lock so the
    keepalive thread cannot insert silence in the middle
    of TTS playback.
    """

    if not pcm_data:

        print(
            "[TTS] No audio to send."
        )

        return

    print()
    print(
        f"[TTS] Sending "
        f"{len(pcm_data)} bytes..."
    )

    total_frames = (
        len(pcm_data)
        + FRAME_BYTES
        - 1
    ) // FRAME_BYTES

    frame_number = 0

    # --------------------------------------------------------
    # IMPORTANT:
    # Hold the lock for the ENTIRE TTS playback.
    # --------------------------------------------------------

    with send_lock:

        for offset in range(
            0,
            len(pcm_data),
            FRAME_BYTES,
        ):

            frame = pcm_data[
                offset:
                offset + FRAME_BYTES
            ]

            if len(frame) < FRAME_BYTES:

                frame += b"\x00" * (
                    FRAME_BYTES
                    - len(frame)
                )

            send_audiosocket_frame(
                conn,
                AUDIO_TYPE,
                frame,
            )

            frame_number += 1

            # 20 ms real-time pacing.
            time.sleep(0.020)

    print(
        f"[TTS] Playback sent. "
        f"Frames: {frame_number}/{total_frames}"
    )


# ============================================================
# SAVE TEMPORARY CALL AUDIO
# ============================================================

def save_pcm_as_wav(
    pcm_data: bytes,
    output_path: Path,
) -> None:
    """
    Save 8 kHz PCM as a temporary WAV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(
            SAMPLE_WIDTH
        )

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            pcm_data
        )


# ============================================================
# TEXT -> SPEECH -> AUDIO SOCKET
# ============================================================

def synthesize_and_send_message(
    conn: socket.socket,
    message: str,
    tts: TTSService,
    send_lock: threading.Lock,
) -> bool:
    """
    Synthesize a system message and send it through
    AudioSocket.

    Returns True when playback was sent successfully.
    """

    output_wav = (
        TEMP_AUDIO
        / f"system_message_{uuid.uuid4()}.wav"
    )

    try:

        TEMP_AUDIO.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "[SYSTEM TTS]"
        )

        print(
            f"[SYSTEM TTS] "
            f"{message}"
        )

        tts.synthesize(
            message,
            output_wav,
        )

        if not output_wav.exists():

            raise RuntimeError(
                "System TTS output WAV was not created."
            )

        pcm = wav_to_pcm_8khz(
            output_wav
        )

        if not pcm:

            return False

        print(
            "[SYSTEM TTS] "
            "Sending message..."
        )

        send_audio(
            conn,
            pcm,
            send_lock,
        )

        print(
            "[SYSTEM TTS] "
            "Message playback completed."
        )

        return True

    except (
        BrokenPipeError,
        ConnectionResetError,
        OSError,
    ) as exc:

        print(
            "[SYSTEM TTS] "
            f"Connection unavailable: {exc}"
        )

        return False

    except Exception as exc:

        print(
            "[SYSTEM TTS ERROR] "
            f"{exc}"
        )

        return False

    finally:

        try:

            if output_wav.exists():

                output_wav.unlink()

        except Exception as exc:

            print(
                "[SYSTEM TTS CLEANUP WARNING] "
                f"{exc}"
            )


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def process_call_audio(
    audio_buffer: bytes,
    stt: STTService,
    rag: RAGService,
    tts: TTSService,
) -> tuple[bytes, bool]:
    """
    Complete pipeline:

        PCM
         ↓
        WAV
         ↓
        STT
         ↓
        Farewell intent OR RAG
         ↓
        LLM
         ↓
        TTS
         ↓
        8 kHz PCM

    Returns:

        (pcm_audio, farewell_requested)
    """

    TEMP_AUDIO.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_wav = (
        TEMP_AUDIO
        / f"call_input_{uuid.uuid4()}.wav"
    )

    output_wav = (
        TEMP_AUDIO
        / f"call_output_{uuid.uuid4()}.wav"
    )

    try:

        print()
        print("=" * 60)

        print(
            "[PIPELINE] "
            "PROCESSING UTTERANCE"
        )

        print("=" * 60)

        # ====================================================
        # 1. Caller PCM -> WAV
        # ====================================================

        save_pcm_as_wav(
            audio_buffer,
            input_wav,
        )

        print(
            f"[PIPELINE] Input WAV: "
            f"{input_wav.name}"
        )

        # ====================================================
        # 2. STT
        # ====================================================

        print()
        print(
            "[PIPELINE] SPEECH TO TEXT"
        )

        transcript = stt.transcribe(
            str(input_wav)
        )

        transcript = transcript.strip()

        print(
            f"[STT] Transcript: "
            f"{transcript}"
        )

        # ====================================================
        # 3. INTENT / RAG
        # ====================================================

        farewell_requested = False

        # ----------------------------------------------------
        # Empty speech
        # ----------------------------------------------------

        if not transcript:

            answer = (
                "I'm sorry, I could not hear "
                "your question clearly. "
                "Please try again."
            )

        # ----------------------------------------------------
        # Farewell intent
        # ----------------------------------------------------

        elif is_farewell_request(
            transcript
        ):

            farewell_requested = True

            print()
            print(
                "[INTENT] "
                "Farewell request detected."
            )

            answer = get_farewell_message()

            print(
                f"[FAREWELL] "
                f"{answer}"
            )

        # ----------------------------------------------------
        # Normal question
        # ----------------------------------------------------

        else:

            print()
            print(
                "[PIPELINE] RAG + LOCAL LLM"
            )

            result = rag.ask(
                transcript
            )

            answer = result.get(
                "answer",
                (
                    "I'm sorry, I don't have "
                    "enough information to answer "
                    "that. Please contact reception."
                ),
            )

            answer = str(
                answer
            ).strip()

            print(
                f"[LLM] Answer: "
                f"{answer}"
            )

        # ====================================================
        # 4. TTS
        # ====================================================

        print()
        print(
            "[PIPELINE] TEXT TO SPEECH"
        )

        tts.synthesize(
            answer,
            output_wav,
        )

        if not output_wav.exists():

            raise RuntimeError(
                "TTS output WAV was not created."
            )

        print(
            f"[TTS] Output: "
            f"{output_wav.name}"
        )

        # ====================================================
        # 5. WAV -> 8 kHz PCM
        # ====================================================

        pcm = wav_to_pcm_8khz(
            output_wav
        )

        print(
            "[PIPELINE] "
            "Audio prepared."
        )

        return (
            pcm,
            farewell_requested,
        )

    finally:

        # ====================================================
        # 6. CLEANUP
        # ====================================================

        for path in (
            input_wav,
            output_wav,
        ):

            try:

                if path.exists():

                    path.unlink()

                    print(
                        f"[CLEANUP] "
                        f"Deleted: {path.name}"
                    )

            except Exception as exc:

                print(
                    "[CLEANUP WARNING] "
                    f"{path.name}: {exc}"
                )


# ============================================================
# BACKGROUND AI WORKER
# ============================================================

def process_and_send_response(
    conn: socket.socket,
    audio_data: bytes,
    stt: STTService,
    rag: RAGService,
    tts: TTSService,
    send_lock: threading.Lock,
    on_complete=None,
    on_farewell=None,
) -> None:
    """
    Run:

        STT
        ↓
        Intent / RAG
        ↓
        LLM
        ↓
        TTS
        ↓
        AudioSocket

    If farewell is detected, the farewell audio is played
    before requesting call termination.
    """

    farewell_requested = False

    try:

        print()
        print("=" * 60)

        print(
            "[AI] STARTING PROCESSING"
        )

        print("=" * 60)

        pcm, farewell_requested = (
            process_call_audio(
                audio_buffer=audio_data,
                stt=stt,
                rag=rag,
                tts=tts,
            )
        )

        if not pcm:

            print(
                "[AI] No response audio."
            )

            return

        print()
        print(
            "[AI] Sending response..."
        )

        send_audio(
            conn,
            pcm,
            send_lock,
        )

        print(
            "[AI] Voice response sent."
        )

        # ----------------------------------------------------
        # Farewell was successfully generated and played.
        # ----------------------------------------------------

        if farewell_requested:

            print()
            print(
                "=" * 60
            )

            print(
                "[CALL] Farewell response completed."
            )

            print(
                "[CALL] Requesting call termination..."
            )

            print(
                "=" * 60
            )

            if on_farewell is not None:

                try:

                    on_farewell()

                except Exception as exc:

                    print(
                        "[CALL] "
                        f"Farewell callback error: {exc}"
                    )

    except ConnectionResetError:

        print(
            "[AI] Asterisk closed connection."
        )

    except BrokenPipeError:

        print(
            "[AI] Broken pipe."
        )

    except OSError as exc:

        print(
            f"[AI] Socket error: {exc}"
        )

    except Exception as exc:

        print(
            f"[AI ERROR] {exc}"
        )

    finally:

        if on_complete is not None:

            try:

                on_complete()

            except Exception as exc:

                print(
                    "[STATE ERROR] "
                    f"Completion callback failed: {exc}"
                )


# ============================================================
# VAD HELPER
# ============================================================

def is_speech_frame(
    vad: webrtcvad.Vad,
    frame: bytes,
) -> bool:
    """
    Determine whether one 20 ms PCM frame contains speech.
    """

    if len(frame) != VAD_FRAME_BYTES:

        return False

    try:

        return vad.is_speech(
            frame,
            SAMPLE_RATE,
        )

    except Exception:

        return False


# ============================================================
# HANDLE ONE CALL
# ============================================================

def handle_call(
    conn: socket.socket,
    address,
    stt: STTService,
    rag: RAGService,
    tts: TTSService,
) -> None:

    print()
    print("=" * 60)

    print(
        f"[CALL] AudioSocket connection "
        f"from {address}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_audio = 0

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    vad = webrtcvad.Vad(
        VAD_MODE
    )

    vad_input_buffer = bytearray()

    speech_buffer = bytearray()

    speech_started = False
    speech_frame_count = 0
    silence_frame_count = 0

    # --------------------------------------------------------
    # Processing state
    # --------------------------------------------------------

    processing = False

    state_lock = threading.Lock()

    # --------------------------------------------------------
    # Call UUID
    # --------------------------------------------------------

    call_uuid = None

    # --------------------------------------------------------
    # AI worker
    # --------------------------------------------------------

    worker: threading.Thread | None = None

    # --------------------------------------------------------
    # TTS cooldown
    # --------------------------------------------------------

    tts_playing_until = 0.0

    # --------------------------------------------------------
    # Socket send lock
    # --------------------------------------------------------

    send_lock = threading.Lock()

    # --------------------------------------------------------
    # Keepalive
    # --------------------------------------------------------

    keepalive_stop = threading.Event()

    keepalive_thread: threading.Thread | None = None

    # --------------------------------------------------------
    # Farewell state
    # --------------------------------------------------------

    farewell_requested = False

    farewell_lock = threading.Lock()

    # ========================================================
    # STATE HELPERS
    # ========================================================

    def is_processing() -> bool:

        with state_lock:

            return processing

    def start_processing() -> None:

        nonlocal processing

        with state_lock:

            processing = True

        print()
        print(
            "[STATE] PROCESSING"
        )

    def finish_processing() -> None:

        nonlocal processing
        nonlocal tts_playing_until

        with state_lock:

            processing = False

        tts_playing_until = (
            time.monotonic()
            + TTS_COOLDOWN_SECONDS
        )

        print()
        print(
            "[PIPELINE] "
            "AI processing completed."
        )

        print(
            "[STATE] TTS cooldown: "
            f"{TTS_COOLDOWN_SECONDS:.1f}s"
        )

        print(
            "[STATE] READY FOR NEXT QUESTION"
        )

    # ========================================================
    # RESET VAD
    # ========================================================

    def reset_vad_state(
        clear_input_buffer: bool = False,
    ) -> None:

        nonlocal speech_started
        nonlocal speech_frame_count
        nonlocal silence_frame_count

        speech_buffer.clear()

        speech_started = False

        speech_frame_count = 0

        silence_frame_count = 0

        if clear_input_buffer:

            vad_input_buffer.clear()

    # ========================================================
    # PROCESSING CALLBACK
    # ========================================================

    def processing_finished() -> None:

        finish_processing()

    # ========================================================
    # FAREWELL CALLBACK
    # ========================================================

    def request_farewell_hangup() -> None:

        nonlocal farewell_requested

        with farewell_lock:

            farewell_requested = True

    # ========================================================
    # KEEPALIVE
    # ========================================================

    def audiosocket_keepalive() -> None:
        """
        Keep AudioSocket alive during periods where there
        is no caller audio.

        Keepalive is intentionally quiet in the terminal
        to avoid flooding the presentation output.
        """

        print(
            "[KEEPALIVE] "
            "AudioSocket keepalive started."
        )

        while not keepalive_stop.wait(
            KEEPALIVE_INTERVAL_SECONDS
        ):

            try:

                with send_lock:

                    send_audiosocket_frame(
                        conn,
                        AUDIO_TYPE,
                        SILENCE_FRAME,
                    )

            except (
                BrokenPipeError,
                ConnectionResetError,
                OSError,
            ):

                break

            except Exception as exc:

                print(
                    "[KEEPALIVE ERROR] "
                    f"{exc}"
                )

                break

        print(
            "[KEEPALIVE] "
            "AudioSocket keepalive stopped."
        )

    # ========================================================
    # START AI WORKER
    # ========================================================

    def start_ai_worker(
        utterance: bytes,
    ) -> None:

        nonlocal worker

        if is_processing():

            print(
                "[STATE] "
                "AI is already processing."
            )

            return

        start_processing()

        reset_vad_state(
            clear_input_buffer=True
        )

        print()
        print(
            "[PIPELINE] "
            "Starting AI processing..."
        )

        worker = threading.Thread(
            target=process_and_send_response,
            args=(
                conn,
                utterance,
                stt,
                rag,
                tts,
                send_lock,
                processing_finished,
                request_farewell_hangup,
            ),
            daemon=True,
            name=(
                f"AI-Call-"
                f"{call_uuid or 'unknown'}"
            ),
        )

        worker.start()

        print(
            "[PIPELINE] "
            "Audio reception continues."
        )

    # ========================================================
    # MAIN CALL LOOP
    # ========================================================

    try:

        print(
            "[STATE] READY FOR CALLER INPUT"
        )

        # ----------------------------------------------------
        # Start keepalive.
        # ----------------------------------------------------

        keepalive_thread = threading.Thread(
            target=audiosocket_keepalive,
            daemon=True,
            name="AudioSocket-Keepalive",
        )

        keepalive_thread.start()

        # ----------------------------------------------------
        # Initial greeting.
        #
        # We send this after the AudioSocket connection is
        # established.
        # ----------------------------------------------------

        greeting = get_time_based_greeting()

        print()
        print(
            "[GREETING] "
            f"{greeting}"
        )

        greeting_sent = (
            synthesize_and_send_message(
                conn=conn,
                message=greeting,
                tts=tts,
                send_lock=send_lock,
            )
        )

        if greeting_sent:

            print(
                "[GREETING] "
                "Initial greeting completed."
            )

        else:

            print(
                "[GREETING] "
                "Greeting could not be delivered."
            )

        print()
        print(
            "[STATE] "
            "Unmute MicroSIP → ask question → "
            "mute MicroSIP."
        )

        # ====================================================
        # RECEIVE LOOP
        # ====================================================

        while True:

            frame = read_audiosocket_frame(
                conn
            )

            # =================================================
            # SOCKET CLOSED
            # =================================================

            if frame is None:

                print(
                    "[CALL] "
                    "Asterisk closed connection."
                )

                break

            frame_type, payload = frame

            # =================================================
            # UUID
            # =================================================

            if frame_type == FRAME_UUID:

                if len(payload) == 16:

                    call_uuid = str(
                        uuid.UUID(
                            bytes=payload
                        )
                    )

                    print(
                        f"[CALL] UUID: "
                        f"{call_uuid}"
                    )

                continue

            # =================================================
            # AUDIO
            # =================================================

            if frame_type == FRAME_AUDIO:

                total_audio += len(
                    payload
                )

                vad_input_buffer.extend(
                    payload
                )

                while (
                    len(vad_input_buffer)
                    >= VAD_FRAME_BYTES
                ):

                    vad_frame = bytes(
                        vad_input_buffer[
                            :VAD_FRAME_BYTES
                        ]
                    )

                    del vad_input_buffer[
                        :VAD_FRAME_BYTES
                    ]

                    # =========================================
                    # AI PROCESSING
                    # =========================================

                    if is_processing():

                        continue

                    # =========================================
                    # FAREWELL ALREADY REQUESTED
                    # =========================================

                    with farewell_lock:

                        if farewell_requested:

                            continue

                    # =========================================
                    # TTS COOLDOWN
                    # =========================================

                    if (
                        time.monotonic()
                        < tts_playing_until
                    ):

                        reset_vad_state(
                            clear_input_buffer=True
                        )

                        continue

                    # =========================================
                    # VAD
                    # =========================================

                    speech = is_speech_frame(
                        vad,
                        vad_frame,
                    )

                    # =========================================
                    # SPEECH
                    # =========================================

                    if speech:

                        if not speech_started:

                            speech_frame_count += 1

                            speech_buffer.extend(
                                vad_frame
                            )

                            if (
                                speech_frame_count
                                >= SPEECH_START_FRAMES
                            ):

                                speech_started = True

                                silence_frame_count = 0

                                print()
                                print(
                                    "[VAD] "
                                    "Speech started."
                                )

                                print(
                                    "[VAD] "
                                    "Recording caller question..."
                                )

                        else:

                            speech_buffer.extend(
                                vad_frame
                            )

                        silence_frame_count = 0

                    # =========================================
                    # SILENCE
                    # =========================================

                    else:

                        if not speech_started:

                            speech_frame_count = 0

                            speech_buffer.clear()

                            continue

                        speech_buffer.extend(
                            vad_frame
                        )

                        silence_frame_count += 1

                        # =====================================
                        # QUESTION COMPLETE
                        # =====================================

                        if (
                            silence_frame_count
                            >= SILENCE_FRAMES
                        ):

                            trim_bytes = (
                                TRAILING_SILENCE_FRAMES
                                * VAD_FRAME_BYTES
                            )

                            if (
                                len(speech_buffer)
                                > trim_bytes
                            ):

                                utterance = bytes(
                                    speech_buffer[
                                        :-trim_bytes
                                    ]
                                )

                            else:

                                utterance = bytes(
                                    speech_buffer
                                )

                            print()
                            print("=" * 60)

                            print(
                                "[VAD] "
                                "Speech ended."
                            )

                            print(
                                f"[VAD] Utterance: "
                                f"{len(utterance)} bytes"
                            )

                            print(
                                "[VAD] "
                                "Question captured successfully."
                            )

                            print("=" * 60)

                            reset_vad_state(
                                clear_input_buffer=True
                            )

                            minimum_bytes = int(
                                SAMPLE_RATE
                                * SAMPLE_WIDTH
                                * 0.25
                            )

                            if (
                                len(utterance)
                                < minimum_bytes
                            ):

                                print(
                                    "[VAD] "
                                    "Utterance too short."
                                )

                                print(
                                    "[STATE] "
                                    "READY FOR NEXT QUESTION"
                                )

                                continue

                            start_ai_worker(
                                utterance
                            )

                            continue

                    # =========================================
                    # MAXIMUM UTTERANCE
                    # =========================================

                    if (
                        speech_started
                        and
                        len(speech_buffer)
                        >= MAX_UTTERANCE_BYTES
                    ):

                        print()
                        print("=" * 60)

                        print(
                            "[VAD] "
                            "Maximum utterance "
                            "length reached."
                        )

                        print(
                            f"[VAD] Limit: "
                            f"{MAX_UTTERANCE_SECONDS} seconds"
                        )

                        print("=" * 60)

                        utterance = bytes(
                            speech_buffer
                        )

                        reset_vad_state(
                            clear_input_buffer=True
                        )

                        minimum_bytes = int(
                            SAMPLE_RATE
                            * SAMPLE_WIDTH
                            * 0.25
                        )

                        if (
                            len(utterance)
                            >= minimum_bytes
                        ):

                            start_ai_worker(
                                utterance
                            )

                continue

            # =================================================
            # DTMF
            # =================================================

            if frame_type == FRAME_DTMF:

                if payload:

                    print(
                        f"[DTMF] "
                        f"{payload!r}"
                    )

                continue

            # =================================================
            # HANGUP
            # =================================================

            if frame_type == FRAME_HANGUP:

                print()
                print(
                    "=" * 60
                )

                print(
                    "[CALL] "
                    "Asterisk sent AudioSocket hangup."
                )

                print(
                    "[CALL] "
                    "The SIP call is already being terminated."
                )

                print(
                    "[CALL] "
                    "A farewell cannot be guaranteed after "
                    "a MicroSIP End Call action."
                )

                print(
                    "=" * 60
                )

                break

            # =================================================
            # AUDIO SOCKET ERROR
            # =================================================

            if frame_type == FRAME_ERROR:

                print(
                    "[AUDIO SOCKET ERROR] "
                    f"{payload.hex()}"
                )

                break

            # =================================================
            # UNKNOWN FRAME
            # =================================================

            print(
                "[WARN] "
                "Unknown frame type: "
                f"0x{frame_type:02x}"
            )

    except ConnectionResetError:

        print(
            "[CALL] "
            "Connection reset by Asterisk."
        )

    except BrokenPipeError:

        print(
            "[CALL] "
            "Asterisk closed the socket."
        )

    except OSError as exc:

        print(
            f"[CALL] "
            f"Socket error: {exc}"
        )

    except Exception as exc:

        print(
            f"[CALL ERROR] "
            f"{exc}"
        )

    finally:

        # ----------------------------------------------------
        # Stop keepalive.
        # ----------------------------------------------------

        keepalive_stop.set()

        if keepalive_thread is not None:

            keepalive_thread.join(
                timeout=1.0
            )

        print()
        print(
            f"[CALL] "
            f"Total caller audio: "
            f"{total_audio} bytes"
        )

        if call_uuid:

            print(
                f"[CALL] UUID: "
                f"{call_uuid}"
            )

        if worker is not None:

            if worker.is_alive():

                print(
                    "[CALL] "
                    "AI worker is still processing."
                )

            else:

                print(
                    "[CALL] "
                    "AI worker completed."
                )

        print(
            "[CALL] Finished."
        )


# ============================================================
# MAIN SERVER
# ============================================================

def main() -> None:

    print("=" * 60)

    print(
        "REC AI RECEPTIONIST - "
        "CONVERSATIONAL AUDIOSOCKET"
    )

    print("=" * 60)

    print(
        f"Listening on "
        f"{HOST}:{PORT}"
    )

    # ========================================================
    # TTS
    # ========================================================

    print()
    print(
        "[STARTUP] Loading TTS model..."
    )

    tts = TTSService(
        model_path=TTS_MODEL
    )

    print(
        "[STARTUP] TTS ready."
    )

    # ========================================================
    # STT
    # ========================================================

    print(
        "[STARTUP] Loading STT model..."
    )

    stt = STTService()

    print(
        "[STARTUP] STT ready."
    )

    # ========================================================
    # RAG
    # ========================================================

    print(
        "[STARTUP] Loading RAG service..."
    )

    rag = RAGService()

    print(
        "[STARTUP] RAG ready."
    )

    # ========================================================
    # SERVICES READY
    # ========================================================

    print()
    print(
        "[STARTUP] All services ready."
    )

    print()
    print(
        "[CONFIG] VAD mode: "
        f"{VAD_MODE}"
    )

    print(
        "[CONFIG] Speech start: "
        f"{SPEECH_START_FRAMES} frames "
        f"({SPEECH_START_FRAMES * VAD_FRAME_MS} ms)"
    )

    print(
        "[CONFIG] Silence timeout: "
        f"{SILENCE_TIMEOUT_SECONDS}s"
    )

    print(
        "[CONFIG] Maximum utterance: "
        f"{MAX_UTTERANCE_SECONDS}s"
    )

    print(
        "[CONFIG] TTS cooldown: "
        f"{TTS_COOLDOWN_SECONDS}s"
    )

    print(
        "[CONFIG] Keepalive interval: "
        f"{KEEPALIVE_INTERVAL_SECONDS}s"
    )

    # ========================================================
    # TCP SERVER
    # ========================================================

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    server.bind(
        (HOST, PORT)
    )

    server.listen(5)

    server.settimeout(
        1.0
    )

    print()
    print("=" * 60)

    print(
        f"[READY] Listening on "
        f"{HOST}:{PORT}"
    )

    print(
        "[READY] Press Ctrl+C to stop."
    )

    print("=" * 60)

    print(
        "[READY] Waiting for Asterisk connection..."
    )

    # ========================================================
    # SERVER LOOP
    # ========================================================

    try:

        while True:

            try:

                conn, address = (
                    server.accept()
                )

            except socket.timeout:

                continue

            print()
            print(
                "[SERVER] "
                "New AudioSocket connection."
            )

            # ------------------------------------------------
            # One call at a time.
            # ------------------------------------------------

            with conn:

                handle_call(
                    conn,
                    address,
                    stt,
                    rag,
                    tts,
                )

            print()
            print(
                "[SERVER] "
                "Call connection ended."
            )

            print(
                "[SERVER] "
                "Waiting for next call..."
            )

    except KeyboardInterrupt:

        print()
        print(
            "[SERVER] "
            "Shutdown requested."
        )

    finally:

        try:

            server.close()

        except Exception:

            pass

        print(
            "[SERVER] "
            "Audio gateway stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()