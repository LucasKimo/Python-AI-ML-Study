"""pyannote로 wav 파일의 화자 구간을 출력합니다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


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
        print("사용법: python diarize_file.py <audio.wav>")
        sys.exit(1)

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN 환경변수를 설정하세요. (Hugging Face Access Token)")
        sys.exit(1)

    model_id = os.getenv(
        "DIARIZE_MODEL_PATH", "pyannote/speaker-diarization-community-1"
    )
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"파일 없음: {audio_path}")
        sys.exit(1)

    print(f"[1/3] 오디오 로드: {audio_path}")
    wav = load_16k_mono(audio_path)
    waveform = torch.from_numpy(wav).unsqueeze(0).float()

    print(f"[2/3] pyannote 로드: {model_id}")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_id, token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"      device={device}")

    print("[3/3] 화자분리 실행...")
    annotation = pipeline({"waveform": waveform, "sample_rate": 16000})

    print("\n=== 화자 구간 ===")
    count = 0
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        count += 1
        print(f"{speaker:12s}  {turn.start:7.2f} ~ {turn.end:7.2f}")
    print(f"\n총 {count}개 구간")


if __name__ == "__main__":
    main()
