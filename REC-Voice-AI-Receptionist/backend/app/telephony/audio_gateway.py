import asyncio
import struct
from datetime import datetime
import uuid
import wave

HOST = "0.0.0.0"
PORT = 9019

# Asterisk AudioSocket message types
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xFF


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    address = writer.get_extra_info("peername")

    print("=" * 60)
    print(f"[CALL] AudioSocket connection from {address}")
    print(f"[CALL] Time: {datetime.now().isoformat()}")
    print("=" * 60)
    audio_data = bytearray()
    total_audio_bytes = 0
    audio_frames = 0

    try:
        while True:
            # AudioSocket frame:
            # 1 byte  = message type
            # 2 bytes = payload length (big endian)
            header = await reader.readexactly(3)

            message_type = header[0]
            payload_length = struct.unpack(">H", header[1:3])[0]

            payload = await reader.readexactly(payload_length)

            if message_type == MSG_UUID:
                 if len(payload) == 16:
                      call_uuid = str(uuid.UUID(bytes=payload))
                      print(f"[CALL] UUID: {call_uuid}")
                 else:
                      print(f"[CALL] Unexpected UUID payload: {len(payload)} bytes")

            elif message_type == MSG_AUDIO:
                audio_frames += 1
                total_audio_bytes += len(payload)
                
                audio_data.extend(payload)

                if audio_frames == 1:
                    print("[AUDIO] Receiving caller audio...")

                if audio_frames % 100 == 0:
                    seconds = total_audio_bytes / (8000 * 2)

                    print(
                        f"[AUDIO] frames={audio_frames} "
                        f"bytes={total_audio_bytes} "
                        f"~{seconds:.1f}s"
                    )

            elif message_type == MSG_DTMF:
                digit = payload.decode("ascii", errors="replace")
                print(f"[DTMF] {digit}")

            elif message_type == MSG_ERROR:
                print(f"[ERROR] AudioSocket error: {payload!r}")

            else:
                print(
                    f"[WARN] Unknown message type "
                    f"0x{message_type:02x}, "
                    f"{payload_length} bytes"
                )

    except asyncio.IncompleteReadError:
        print("[CALL] AudioSocket connection closed.")

    except ConnectionResetError:
        print("[CALL] Client reset connection.")

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")

    finally:
        print(
            f"[CALL] Total audio received: "
            f"{total_audio_bytes} bytes"
        )
        
        if audio_data:
          filename = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

          with wave.open(filename, "wb") as wav:
               wav.setnchannels(1)
               wav.setsampwidth(2)
               wav.setframerate(8000)
               wav.writeframes(audio_data)

          print(f"[CALL] Saved audio: {filename}")

        writer.close()

        try:
            await writer.wait_closed()
        except Exception:
            pass

        print("[CALL] Connection finished.")
        print()


async def main():
    print("=" * 60)
    print("REC AI RECEPTIONIST - AUDIOSOCKET GATEWAY")
    print("=" * 60)
    print(f"Listening on {HOST}:{PORT}")
    print("Waiting for Asterisk...")
    print()

    server = await asyncio.start_server(
        handle_client,
        HOST,
        PORT,
    )

    addresses = ", ".join(
        str(sock.getsockname())
        for sock in server.sockets or []
    )

    print(f"[READY] Listening: {addresses}")
    print()

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Audio gateway stopped.")