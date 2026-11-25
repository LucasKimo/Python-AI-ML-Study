"""Qwen3 ASR과 pyannote를 처음 쓸 때 한 번만 올리는 로더.

이론 — 왜 서버 시작 때 바로 안 읽나
----------------------------------
두 모델은 용량이 크고, Qwen3 경로는 GPU 위의 vLLM이 필요합니다.
import 시점에 올리면 ASR 서버가 뜨기도 전에 실패하거나 메모리를 잡습니다.
그래서 전역 변수는 None으로 두고, get_asr() / get_diarizer()가
처음 호출될 때 디스크에서 읽습니다. 이후 호출은 같은 객체를 돌려줍니다.
이걸 지연 로드(lazy load)라고 합니다.
기동 직후 미리 올리고 싶으면 ASR_WARMUP=true 로 warmup()을 부릅니다.

스레드 락을 거는 이유: 요청 두 개가 동시에 '아직 None'을 보고
모델을 두 번 올리는 일을 막습니다. 락 안에서 None 검사를 한 번 더 합니다.

Qwen3ASRModel.LLM(...)
  vLLM은 큰 언어 모델을 GPU에 상주시킨 뒤 여러 조각의 전사를 배치로 처리합니다.
  gpu_memory_utilization 은 GPU 메모리 중 이 모델이 쓸 비율,
  max_model_len 은 한 번에 다루는 최대 토큰 길이입니다.

pyannote Pipeline.from_pretrained(...)
  화자분리 파이프라인(활동 감지 + 임베딩 + 군집화)을 한 객체로 받습니다.
  모델 카드가 사용 동의가 필요한 게이트 모델이라 Hugging Face 토큰(HF_TOKEN)이 있어야 합니다.
  huggingface.co 의 해당 모델 페이지에서 라이선스에 동의한 계정이어야 다운로드됩니다.
  GPU가 있으면 cuda, 없으면 cpu로 올립니다. cpu는 동작은 하되 긴 음성에서 느립니다.
"""

from __future__ import annotations

import threading
from typing import Any

from shared import config

_lock = threading.Lock()
_asr: Any = None
_diarizer: Any = None
_load_errors: dict[str, str] = {}


def model_status() -> dict[str, Any]:
    return {
        "asr_loaded": _asr is not None,
        "diarizer_loaded": _diarizer is not None,
        "asr_model_path": config.ASR_MODEL_PATH,
        "diarize_model_path": config.DIARIZE_MODEL_PATH,
        "hf_token_set": bool(config.HF_TOKEN),
        "errors": dict(_load_errors),
    }


def get_asr():
    """Load Qwen3ASRModel (vLLM backend)."""
    global _asr
    if _asr is not None:
        return _asr
    with _lock:
        if _asr is not None:
            return _asr
        try:
            from qwen_asr import Qwen3ASRModel

            print(f"[ASR] Loading Qwen3 ASR: {config.ASR_MODEL_PATH}")
            _asr = Qwen3ASRModel.LLM(
                model=config.ASR_MODEL_PATH,
                gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                max_model_len=config.MAX_MODEL_LEN,
            )
            _load_errors.pop("asr", None)
            print("[ASR] Qwen3 ASR ready")
        except Exception as exc:  # noqa: BLE001
            _load_errors["asr"] = str(exc)
            raise RuntimeError(
                f"Qwen3 ASR 로드 실패: {exc}. "
                "GPU + vLLM + qwen-asr, ASR_MODEL_PATH를 확인하세요."
            ) from exc
    return _asr


def get_diarizer():
    """Load pyannote speaker-diarization pipeline."""
    global _diarizer
    if _diarizer is not None:
        return _diarizer
    with _lock:
        if _diarizer is not None:
            return _diarizer
        try:
            import torch
            from pyannote.audio import Pipeline

            if not config.HF_TOKEN:
                raise RuntimeError("HF_TOKEN 환경변수가 필요합니다 (pyannote 모델 접근).")

            print(f"[ASR] Loading pyannote: {config.DIARIZE_MODEL_PATH}")
            _diarizer = Pipeline.from_pretrained(
                config.DIARIZE_MODEL_PATH,
                token=config.HF_TOKEN,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _diarizer.to(device)
            _load_errors.pop("diarizer", None)
            print(f"[ASR] pyannote ready on {device}")
        except Exception as exc:  # noqa: BLE001
            _load_errors["diarizer"] = str(exc)
            raise RuntimeError(
                f"pyannote 로드 실패: {exc}. "
                "HF_TOKEN과 DIARIZE_MODEL_PATH, pyannote 라이선스 동의를 확인하세요."
            ) from exc
    return _diarizer


def warmup() -> dict[str, Any]:
    """Eager-load models at startup (optional)."""
    get_asr()
    get_diarizer()
    return model_status()
