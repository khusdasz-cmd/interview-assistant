"""Whisper speech-to-text transcription."""

import os
import numpy as np

_whisper = None
WHISPER_MODEL = "base"


def init(model_name: str = "base"):
    """Initialize Whisper model (lazy-loaded on first call)."""
    global _whisper, WHISPER_MODEL
    WHISPER_MODEL = model_name
    if _whisper is None:
        from faster_whisper import WhisperModel
        cache = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             ".whisper_cache")
        _whisper = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8",
            download_root=cache,
        )
    return _whisper


def get_whisper():
    if _whisper is None:
        return init()
    return _whisper


def transcribe(pcm_bytes: bytes) -> str:
    """Transcribe PCM int16 audio bytes to text."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    model = get_whisper()
    segments, _ = model.transcribe(
        audio, language="zh", beam_size=3, initial_prompt=""
    )
    text = " ".join(s.text for s in segments).strip()
    try:
        from zhconv import convert
        text = convert(text, "zh-cn")
    except Exception:
        pass
    return text
