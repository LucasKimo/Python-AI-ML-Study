"""
Qwen3-ASR 실시간 음성 인식 예제

- 마이크에서 일정 길이(기본 4초)씩 소리를 녹음하고,
- 조용한 구간은 건너뛰고, 말소리가 있는 구간만 Qwen3-ASR 모델로 글자로 바꿔(전사) 출력합니다.
- Ctrl+C로 종료하면 지금까지 전사된 문장을 모아서 보여 줍니다.

실행:
    python 6_Qwen3_live.py                 # 기본 마이크로 실행
    python 6_Qwen3_live.py --list-devices  # 사용 가능한 오디오 장치 목록 보기

환경 변수로 설정 변경 (PowerShell 예: $env:ASR_CHUNK_SECONDS="3"):
    ASR_MODEL_PATH     사용할 모델 (기본 Qwen/Qwen3-ASR-1.7B)
    ASR_LANGUAGE       인식할 언어 (기본 Korean)
    ASR_DTYPE          bfloat16 또는 float16 (기본 bfloat16)
    ASR_CHUNK_SECONDS  한 번에 녹음/전사할 길이(초) (기본 4)
    ASR_SILENCE_RMS    이 값보다 소리가 작으면 조용하다고 판단 (기본 0.01)
    ASR_MIC_DEVICE     마이크 장치 번호 (비우면 시스템 기본 마이크)
"""

import os
import sys

import numpy as np


def peak_normalize(wav: np.ndarray, target: float = 0.9) -> np.ndarray:
    """작게 녹음된 소리를 키워서, 가장 큰 진폭이 target(0.9)이 되도록 맞춥니다.

    마이크 소리가 너무 작으면 인식률이 떨어지므로 음량을 보정합니다.
    예) 최대 진폭이 0.3이면 전체에 0.9/0.3 = 3배를 곱함
    """
    # 가장 큰 진폭(절댓값 기준). 빈 배열이면 0으로 처리
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    # 사실상 무음이거나(0으로 나누기 방지), 이미 충분히 크면 그대로 반환
    if peak < 1e-6 or peak >= target:
        return wav
    return (wav * (target / peak)).astype(np.float32)


def list_devices() -> None:
    """컴퓨터에 연결된 오디오 입출력 장치 목록을 출력합니다.

    여기서 확인한 장치 번호를 ASR_MIC_DEVICE 환경 변수에 넣으면 그 마이크를 사용합니다.
    """
    import sounddevice as sd

    print(sd.query_devices())


def main() -> None:
    # --list-devices 옵션이 있으면 장치 목록만 보여 주고 종료 (모델을 로드하지 않음)
    if "--list-devices" in sys.argv:
        list_devices()
        return

    # 무거운 라이브러리는 실제로 필요할 때 import (--list-devices 실행을 빠르게 하기 위함)
    import sounddevice as sd  # 마이크 녹음 라이브러리
    import torch
    from qwen_asr import Qwen3ASRModel  # Qwen3 음성 인식(ASR) 모델

    # 이 예제는 GPU 전용. GPU가 없으면 안내 후 종료
    if not torch.cuda.is_available():
        print("CUDA GPU를 찾지 못했습니다. NVIDIA 드라이버와 CUDA용 PyTorch를 확인하세요.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 설정값 읽기 (환경 변수가 없으면 기본값 사용)
    # ------------------------------------------------------------------
    model_path = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")  # Hugging Face 모델 이름 또는 로컬 경로
    language = os.getenv("ASR_LANGUAGE", "Korean")
    dtype_name = os.getenv("ASR_DTYPE", "bfloat16").lower()

    # 16비트 실수형을 사용해 GPU 메모리를 절약. bfloat16을 지원하지 않는 GPU는 float16 사용
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    chunk_seconds = float(os.getenv("ASR_CHUNK_SECONDS", "4"))  # 몇 초 단위로 끊어서 전사할지
    silence_rms = float(os.getenv("ASR_SILENCE_RMS", "0.01"))  # 무음 판단 기준(음량)
    sample_rate = 16000  # 초당 샘플 수(16kHz). 음성 인식 모델이 주로 사용하는 값
    device_index = os.getenv("ASR_MIC_DEVICE", "").strip()
    mic_device = int(device_index) if device_index else None  # None이면 시스템 기본 마이크

    # ------------------------------------------------------------------
    # 모델 로드 (처음 실행 시 Hugging Face에서 내려받으므로 시간이 걸림)
    # ------------------------------------------------------------------
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"모델 로드: {model_path}")
    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda:0",          # 첫 번째 GPU에 모델을 올림
        max_inference_batch_size=1,   # 한 번에 오디오 1개씩 처리
        max_new_tokens=128,           # 한 구간에서 생성할 최대 토큰 수(전사 결과 길이 제한)
    )

    print()
    print(f"마이크를 엽니다. {chunk_seconds:.0f}초마다 전사합니다. 종료는 Ctrl+C")
    print("조용하면 건너뛰고, 말하면 그 구간의 글자가 바로 출력됩니다.")
    print()

    lines: list[str] = []  # 전사된 문장을 모아 두었다가 종료 시 한꺼번에 출력
    try:
        # 마이크 입력 스트림 열기. with 블록을 벗어나면 마이크가 자동으로 닫힘
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,          # 모노(1채널) 녹음
            dtype="float32",     # 샘플 값을 -1.0 ~ 1.0 범위의 실수로 받음
            device=mic_device,
            blocksize=int(sample_rate * chunk_seconds),  # 한 블록 = chunk_seconds초 분량
        ) as stream:
            while True:
                # chunk_seconds초 분량의 샘플이 모일 때까지 기다렸다가 읽음
                # audio 모양: (샘플 수, 채널 수), overflowed: 버퍼가 넘쳐 소리가 일부 유실됐는지 여부
                audio, overflowed = stream.read(int(sample_rate * chunk_seconds))

                # (N, 1) -> (N,) 1차원 배열로 펼침 (모델 입력 형식)
                wav = np.asarray(audio, dtype=np.float32).reshape(-1)

                # RMS(제곱 평균의 제곱근) = 이 구간의 평균적인 소리 크기
                rms = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0

                # 전사가 녹음보다 오래 걸리면 처리하지 못한 소리가 쌓여 버퍼가 넘칠 수 있음
                if overflowed:
                    print("(마이크 버퍼가 밀렸습니다. 말을 조금 천천히 해보세요.)")

                # 너무 조용하면 모델을 돌리지 않고 건너뜀 (GPU 낭비와 엉뚱한 인식 결과 방지)
                if rms < silence_rms:
                    print(f"... 조용함 (rms={rms:.4f})")
                    continue
                wav = peak_normalize(wav)  # 작은 소리를 키워서 인식률 향상

                # (오디오 배열, 샘플레이트) 튜플로 전달 -> 결과 리스트 반환 (입력 1개이므로 결과도 1개)
                results = asr.transcribe(audio=(wav, sample_rate), language=language)
                
                # 결과 객체에 text 속성이 없거나 None이어도 에러가 나지 않도록 안전하게 꺼냄
                text = (getattr(results[0], "text", None) or "").strip()
                if not text:
                    print("... 인식된 문장 없음")
                    continue
                lines.append(text)
                print(f">> {text}")
    except KeyboardInterrupt:
        # Ctrl+C를 누르면 무한 루프를 빠져나와 정상 종료
        print("\n종료합니다.")
    except Exception as exc:
        # 주로 마이크 장치 문제(잘못된 장치 번호, 권한 등)일 때 발생
        # 참고: 녹음/전사 도중의 다른 에러도 여기로 들어옴
        print(f"마이크를 열지 못했습니다: {exc}")
        print("장치 목록: python 6_Qwen3_live.py --list-devices")
        print("다른 마이크를 쓰려면: $env:ASR_MIC_DEVICE=\"1\"")
        sys.exit(1)

    # 종료 시 지금까지 전사된 문장을 한 줄씩 출력
    print("\n=== 지금까지의 전사 ===")
    print("\n".join(lines) if lines else "(없음)")


if __name__ == "__main__":
    main()
