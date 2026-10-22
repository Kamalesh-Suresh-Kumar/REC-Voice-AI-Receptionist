
from pathlib import Path
import numpy as np
import soundfile as sf

from scipy.signal import butter, sosfiltfilt, resample_poly


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLEAN_DIR = PROJECT_ROOT / "evaluation_results" / "stt_audio" / "clean"
NOISE_DIR = PROJECT_ROOT / "evaluation_results" / "stt_audio" / "noise"
TELEPHONE_DIR = PROJECT_ROOT / "evaluation_results" / "stt_audio" / "telephone"

# Reproducible noise settings
NOISE_SNR_DB = 10.0
RANDOM_SEED = 42
TARGET_TELEPHONE_RATE = 8000


def normalize_peak(audio, peak=0.98):
    """Scale audio to avoid clipping while preserving its relative SNR."""
    maximum = np.max(np.abs(audio))

    if maximum > 0:
        audio = audio * (peak / maximum)

    return audio


def create_noisy_audio(audio, snr_db, seed):
    """Add reproducible white Gaussian noise at the specified SNR."""
    audio = audio.astype(np.float64)

    signal_power = np.mean(audio ** 2)

    if signal_power <= 1e-12:
        raise ValueError("Input audio has insufficient signal energy.")

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(audio.shape)

    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    mixed = audio + noise

    return normalize_peak(mixed)


def create_telephone_audio(audio, sample_rate):
    """Apply a 300–3400 Hz telephone-band filter and resample to 8 kHz."""
    audio = audio.astype(np.float64)

    if sample_rate <= 6800:
        raise ValueError(
            f"Source sample rate {sample_rate} Hz is too low "
            "for a 3400 Hz upper cutoff."
        )

    # Telephone speech bandwidth
    sos = butter(
        4,
        [300, 3400],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    filtered = sosfiltfilt(sos, audio)

    # Resample using a rational conversion ratio
    from math import gcd

    divisor = gcd(sample_rate, TARGET_TELEPHONE_RATE)

    up = TARGET_TELEPHONE_RATE // divisor
    down = sample_rate // divisor

    telephone_audio = resample_poly(filtered, up, down)

    return normalize_peak(telephone_audio)


def main():
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Clean audio directory not found: {CLEAN_DIR}"
        )

    clean_files = sorted(CLEAN_DIR.glob("q*.wav"))

    if not clean_files:
        raise FileNotFoundError(
            f"No q*.wav files found in {CLEAN_DIR}"
        )

    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    TELEPHONE_DIR.mkdir(parents=True, exist_ok=True)

    noise_success = 0
    telephone_success = 0
    failures = []

    print(f"Clean audio files found: {len(clean_files)}")
    print(f"Noise SNR: {NOISE_SNR_DB} dB")
    print(f"Telephone sample rate: {TARGET_TELEPHONE_RATE} Hz")
    print()

    for index, source_path in enumerate(clean_files, start=1):
        try:
            audio, sample_rate = sf.read(
                source_path,
                dtype="float32",
                always_2d=False,
            )

            # These generated samples are intended to be mono.
            if audio.ndim != 1:
                raise ValueError(
                    f"Expected mono audio, got shape {audio.shape}"
                )

            if len(audio) < 100:
                raise ValueError("Audio file is too short.")

            # Use the question number in the filename as a stable seed.
            digits = "".join(
                character
                for character in source_path.stem
                if character.isdigit()
            )
            question_id = int(digits) if digits else index

            # Condition 1: controlled synthetic background noise
            noisy = create_noisy_audio(
                audio,
                snr_db=NOISE_SNR_DB,
                seed=RANDOM_SEED + question_id,
            )

            noise_path = NOISE_DIR / source_path.name
            sf.write(
                noise_path,
                noisy,
                sample_rate,
                subtype="PCM_16",
            )

            # Condition 2: simulated telephone-band speech
            telephone = create_telephone_audio(
                audio,
                sample_rate,
            )

            telephone_path = TELEPHONE_DIR / source_path.name
            sf.write(
                telephone_path,
                telephone,
                TARGET_TELEPHONE_RATE,
                subtype="PCM_16",
            )

            # Verify both generated files
            noise_check, noise_rate = sf.read(noise_path)
            phone_check, phone_rate = sf.read(telephone_path)

            if len(noise_check) == 0 or noise_rate != sample_rate:
                raise ValueError("Noise audio verification failed.")

            if (
                len(phone_check) == 0
                or phone_rate != TARGET_TELEPHONE_RATE
            ):
                raise ValueError("Telephone audio verification failed.")

            noise_success += 1
            telephone_success += 1

            if index % 10 == 0 or index == len(clean_files):
                print(
                    f"Processed {index}/{len(clean_files)} files"
                )

        except Exception as exc:
            failures.append((source_path.name, str(exc)))
            print(f"FAILED: {source_path.name}: {exc}")

    print("\n========== SUMMARY ==========")
    print(f"Clean files found:       {len(clean_files)}")
    print(f"Noise files successful:  {noise_success}")
    print(f"Telephone successful:    {telephone_success}")
    print(f"Failures:                {len(failures)}")
    print(f"Noise directory:         {NOISE_DIR}")
    print(f"Telephone directory:     {TELEPHONE_DIR}")

    if failures:
        print("\nFailed files:")
        for filename, reason in failures:
            print(f"  {filename}: {reason}")

    if noise_success != len(clean_files) or (
        telephone_success != len(clean_files)
    ):
        raise SystemExit(
            "Audio generation was incomplete. Review the failures above."
        )

    print("\nAudio-condition generation completed successfully.")


if __name__ == "__main__":
    main()
