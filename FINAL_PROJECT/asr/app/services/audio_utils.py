"""Audio helpers: decode to 16 kHz mono float32."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = wav.astype(np.float32, copy=False)
    if sr == 16000:
        return wav
    if wav.size == 0:
        return wav
    duration = wav.shape[0] / float(sr)
    n16 = int(round(duration * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def load_audio_16k(path: str | Path) -> np.ndarray:
    """Load any common audio/container via soundfile, ffmpeg fallback."""
    path = Path(path)
    try:
        wav, sr = sf.read(str(path), always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        return resample_to_16k(np.asarray(wav, dtype=np.float32), int(sr))
    except Exception:
        return _ffmpeg_to_16k(path)


def load_audio_bytes_16k(data: bytes, suffix: str = ".wav") -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return load_audio_16k(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _ffmpeg_to_16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 변환 실패: {proc.stderr.decode(errors='ignore')[:400]}"
        )
    wav, sr = sf.read(io.BytesIO(proc.stdout), always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return resample_to_16k(np.asarray(wav, dtype=np.float32), int(sr))
