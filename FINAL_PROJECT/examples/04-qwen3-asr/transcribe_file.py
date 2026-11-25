"""Qwen3 ASR로 wav 파일 하나를 전사합니다.

Windows + NVIDIA GPU (예: RTX 3060)용입니다.
vLLM 없이 PyTorch CUDA(transformers 백엔드)로 모델을 띄웁니다.

실행:
  python transcribe_file.py sample.wav
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def peak_normalize(wav: np.ndarray, target: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < 1e-6:
        return wav
    if peak >= target:
        return wav
    return (wav * (target / peak)).astype(np.float32)


def loudest_slice(wav: np.ndarray, sr: int, seconds: float, search_seconds: float = 600.0) -> tuple[np.ndarray, float]:
    """앞쪽 search_seconds 안에서 평균 에너지가 가장 큰 구간을 고릅니다."""
    want = int(seconds * sr)
    if wav.shape[0] <= want:
        return wav, 0.0
    limit = min(wav.shape[0], int(search_seconds * sr))
    hop = int(5 * sr)
    best_start = 0
    best_rms = -1.0
    for start in range(0, max(1, limit - want + 1), hop):
        chunk = wav[start : start + want]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    return wav[best_start : best_start + want], best_start / sr


def load_16k_mono(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if int(sr) == 16000:
        return wav
    duration = wav.shape[0] / float(sr)
    n16 = int(round(duration * 16000))
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python transcribe_file.py <audio.wav>")
        print("예: python transcribe_file.py sample.wav")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"파일 없음: {audio_path}")
        sys.exit(1)

    import torch

    if not torch.cuda.is_available():
        print("CUDA GPU를 찾지 못했습니다.")
        print("PyTorch CUDA 빌드가 설치됐는지, NVIDIA 드라이버가 동작하는지 확인하세요.")
        print("확인: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"")
        sys.exit(1)

    model_path = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
    language = os.getenv("ASR_LANGUAGE", "Korean")
    dtype_name = os.getenv("ASR_DTYPE", "bfloat16").lower()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    device = "cuda:0"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"[1/3] 오디오 로드: {audio_path}")
    wav = load_16k_mono(audio_path)
    duration = wav.shape[0] / 16000
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0
    print(f"      samples={wav.shape[0]}, duration={duration:.2f}s, peak={peak:.4f}, rms={rms:.4f}")

    # 0이면 전체를 전사합니다. 기본 20초는 RTX 3060에서 첫 테스트용입니다.
    max_seconds = float(os.getenv("ASR_MAX_SECONDS", "20"))
    if max_seconds > 0 and duration > max_seconds:
        wav, offset = loudest_slice(wav, 16000, max_seconds)
        print(
            f"      파일이 {duration:.0f}초라 {offset:.0f}초 지점부터 {max_seconds:.0f}초만 전사합니다. "
            "전체를 쓰려면 ASR_MAX_SECONDS=0 을 설정하세요."
        )

    before = float(np.max(np.abs(wav))) if wav.size else 0.0
    wav = peak_normalize(wav)
    after = float(np.max(np.abs(wav))) if wav.size else 0.0
    if after > before + 1e-6:
        print(f"      음량이 작아 peak {before:.3f} -> {after:.3f} 로 키웠습니다.")

    print(f"[2/3] Qwen3 ASR 로드 (PyTorch CUDA): {model_path}")
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=4,
        max_new_tokens=256,
    )

    print("[3/3] 전사 중...")
    results = asr.transcribe(audio=(wav, 16000), language=language)
    text = (getattr(results[0], "text", None) or "").strip()
    print("\n=== 결과 ===")
    print(text or "(빈 결과)")


if __name__ == "__main__":
    main()
