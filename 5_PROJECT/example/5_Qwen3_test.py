import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# 음량 정규화
def peak_normalize(wav: np.ndarray, target: float = 0.9) -> np.ndarray:
    # 최대 진폭 구하기
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    
    if peak < 1e-6:  # 0.000001 >> 거의 무음이 오디오는 불필요하게 증폭하지 않고 그대로 반환
        return wav
    if peak >= target:  # 음량이 충분히 크면 그대로 반환
        return wav

    # 목표 진폭에 맞게 음량을 증폭하고 float32 타입으로 변환    
    return (wav * (target / peak)).astype(np.float32)  


# WAV 전체를 인식하지 않고 가장 소리가 큰 구간 추출
def loudest_slice(wav: np.ndarray, sr: int, seconds: float, search_seconds: float = 600.0) -> tuple[np.ndarray, float]:
    """앞쪽 search_seconds 안에서 평균 에너지가 가장 큰 구간을 고른다."""

    # 팔요한 샘플 개수 계산 
    # 예) 샘플링 주파수(sr)가 16,000Hz라면 1초에 16,000개 데이터 존재 >> 20초를 추출하려면 320,000개의 샘플 필요
    want = int(seconds * sr)

    if wav.shape[0] <= want:
        return wav, 0.0

    # 탐색 범위 및 이동 간격 설정
    limit = min(wav.shape[0], int(search_seconds * sr))  # 최대 처음 600초까지만 탐색
    hop = int(5 * sr)   # 5초 간격으로 이동

    best_start = 0
    best_rms = -1.0

    # Sliding Window를 사용하여 탐색
    for start in range(0, max(1, limit - want + 1), hop):
        chunk = wav[start : start + want]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > best_rms:
            best_rms = rms
            best_start = start

    # 최종 결과 반환: 가장 소리가 큰 오디오 구간과 해당 구간의 시작 시간을 반환
    return wav[best_start : best_start + want], best_start / sr


# 오디오 전처리 함수
def load_16k_mono(path: Path) -> np.ndarray:

    # 오디오 파일 읽기: wav, sr 반환
    wav, sr = sf.read(str(path), always_2d=False) 

    # Stereo를 Mono로 변환
        # Stereo: 좌우 오디오 채널 사용
        # Mono: 하나의 채널만 사용
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)

    # float32로 데이터 타입 변환 >> float64에 비해 메모리 사용량을 절반으로 줄일 수 있음
    wav = wav.astype(np.float32)

    # Sampling Rate 변환 (Resampling)
    # Qwen3-ASR 모델의 입력 형식에 맞추기 위해 sampling rate(샘플링 주파수) 변경 >> 16kHz
    if int(sr) == 16000:
        return wav
    
    duration = wav.shape[0] / float(sr)
    n16 = int(round(duration * 16000))

    x_old = np.linspace(
        0.0, 
        duration, 
        num=wav.shape[0], 
        endpoint=False
    )

    x_new = np.linspace(
        0.0, 
        duration, 
        num=n16, 
        endpoint=False
    )

    return np.interp(x_new, x_old, wav).astype(np.float32)


# 실행 장치 자동 선택
def resolve_runtime(torch):
    """CUDA를 우선하고, 없으면 MPS, 그것도 없으면 CPU를 고릅니다."""
    requested = os.getenv("ASR_DTYPE", "").strip().lower()

    def pick_dtype(default: str):
        name = requested or default
        if name == "float32":
            return torch.float32
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        print(f"알 수 없는 ASR_DTYPE={name}. {default}을 사용합니다.")
        name = default
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]

    if torch.cuda.is_available():
        return "cuda:0", pick_dtype("bfloat16"), f"CUDA {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps", pick_dtype("float16"), "Apple MPS"
    return "cpu", pick_dtype("float32"), "CPU"


# 메인 실행 함수
def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python 5_Quen3_test.py <audio.wav>")
        print("예: python transcribe_file.py sample.wav")
        sys.exit(1)

    # Input Validation
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"파일 없음: {audio_path}")
        sys.exit(1)

    # 모델 및 실행 환경 설정
    import torch

    model_path = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")  # 모델 지정
    language = os.getenv("ASR_LANGUAGE", "Korean")  # 언어 지정
    device, dtype, device_label = resolve_runtime(torch)

    print(f"장치: {device_label} ({device}, {dtype})")
    if not device.startswith("cuda"):
        print("CUDA가 없어 이 장치로 실행합니다. 메모리가 부족하면 ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B 를 쓰세요.")
    print(f"[1/3] 오디오 로드: {audio_path}")
    wav = load_16k_mono(audio_path)
    duration = wav.shape[0] / 16000
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0
    print(f"      samples={wav.shape[0]}, duration={duration:.2f}s, peak={peak:.4f}, rms={rms:.4f}")

    # 오디오 길이 제한
    max_seconds = float(os.getenv("ASR_MAX_SECONDS", "20"))
    if max_seconds > 0 and duration > max_seconds:
        wav, offset = loudest_slice(wav, 16000, max_seconds)
        print(
            f"      파일이 {duration:.0f}초라 {offset:.0f}초 지점부터 {max_seconds:.0f}초만 전사합니다. "
            "전체를 쓰려면 ASR_MAX_SECONDS=0 을 설정하세요."
        )

    # 음량 정규화
    before = float(np.max(np.abs(wav))) if wav.size else 0.0
    wav = peak_normalize(wav)
    after = float(np.max(np.abs(wav))) if wav.size else 0.0

    if after > before + 1e-6:
        print(f"      음량이 작아 peak {before:.3f} -> {after:.3f} 로 키웠습니다.")

    # Qwen3-ASR 모델 로드
    print(f"[2/3] Qwen3 ASR 로드 ({device}): {model_path}")
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(  # from_pretrained(): 이미 학습된 모델의 가중치를 불러오는 메서드
        model_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=4,
        max_new_tokens=256,  # 단어의 개수가 아닌 토큰의 개수. 너무 작게 설정하면 음성의 인식 결과가 중간에 잘릴 수 있음
    )

    # 실제 음성 인식
    print("[3/3] 전사 중...")
    results = asr.transcribe(
        audio=(wav, 16000), 
        language=language
    )
    
    text = (getattr(results[0], "text", None) or "").strip()
    print("\n=== 결과 ===")
    print(text or "(빈 결과)")


if __name__ == "__main__":
    main()