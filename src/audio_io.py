"""Audio I/O utilities"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

TARGET_SR = 16_000


def load_mono_16k(path: str | Path) -> np.ndarray:
    """Load any WAV file and return a mono float32 signal at 16 kHz."""
    y, sr = librosa.load(str(path), sr=None, mono=False) # Load with native sampling rate
    if y.ndim > 1:
        y = y.mean(axis=0) # Convert to mono
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq") # Resample to 16 kHz
    return y.astype(np.float32) # Convert to float32


def loop_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) >= target_len:
        return x[:target_len] # If the audio is already longer than the target length, return the first target_len samples
    reps = int(np.ceil(target_len / len(x)))
    return np.tile(x, reps)[:target_len] # Repeat the audio reps times and return the first target_len samples


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Compute alpha to later mix the clean and noise at the given SNR."""
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    alpha = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    return clean + alpha * noise


def peak_normalize(x: np.ndarray) -> np.ndarray:
    """max|x| = 1."""
    return x / (np.max(np.abs(x)) + 1e-12)


def save_wav(path: str | Path, signal: np.ndarray, sr: int = TARGET_SR):
    sf.write(str(path), signal, sr, subtype="PCM_16")


def build_dataset(audio_dir: str | Path, out_dir: str | Path,
                  snrs: list[float] = (-5, 0, 5, 10, 15)):
    """Load raw audio, build the composite clean signal, and save mixes at each SNR."""
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load the audio files
    doctor = load_mono_16k(audio_dir / "Doctor.wav")
    patient = load_mono_16k(audio_dir / "Patient.wav")
    noise_raw = load_mono_16k(audio_dir / "Background_Noise.wav")

    min_len = min(len(doctor), len(patient))
    doctor, patient = doctor[:min_len], patient[:min_len]
    # Combine the doctor and patient voices and normalize the amplitude
    clean = peak_normalize(0.5 * (doctor + patient))
    noise = loop_to_length(noise_raw, min_len)

    save_wav(out_dir / "clean.wav", clean)
    save_wav(out_dir / "noise_looped.wav", noise)

    paths = {"clean": out_dir / "clean.wav"}
    for snr in snrs: # Mix the clean and noise at the given SNR
        z = mix_at_snr(clean, noise, snr)
        z = peak_normalize(z)
        fname = f"mix_snr{snr:+.0f}dB.wav"
        save_wav(out_dir / fname, z)
        paths[f"snr{snr:+.0f}dB"] = out_dir / fname

    return paths, clean, noise 
