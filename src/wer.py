"""WER evaluation: Whisper transcription + jiwer Word Error Rate."""

from pathlib import Path
from jiwer import wer as compute_wer


def transcribe_whisper(audio_path: str | Path, model_size: str = "base") -> str:
    """Transcribe an audio file using faster-whisper (local, CPU-friendly)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="en")
    return " ".join(seg.text.strip() for seg in segments)


def evaluate_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    if not reference.strip() or not hypothesis.strip():
        return 1.0
    return float(compute_wer(reference, hypothesis))
