"""실시간 마이크 스트림을 '한 번 말한 덩어리'로 자릅니다.

Gradio 스트리밍은 말이 끝난 문장을 알아서 주지 않습니다.
짧은 조각이 계속 들어옵니다. 이 버퍼가 에너지로 말인지 침묵인지 가릅니다.

이론 — 에너지 기반 끝점 검출
----------------------------
각 조각의 RMS(제곱 평균 제곱근, 소리의 세기)를 봅니다.
SPEECH_RMS 이상이면 말소리로 보고 버퍼에 붙입니다.
말소리가 시작된 뒤 SILENCE_SEC(기본 0.7초) 동안 조용하면
"한 발화가 끝났다"고 보고 모은 파형을 돌려줍니다.
너무 길어지면 MAX_SPEECH_SEC(12초)에서 강제로 잘라 Whisper에 넘깁니다.
MIN_SPEECH_SEC보다 짧으면 숨이나 탁 소리로 보고 버립니다.

이것은 신경망 VAD가 아닙니다. 크기로만 나누므로
에어컨·키보드가 크면 말소리로 오인할 수 있습니다.
화자 이름과 문장은 여기서 만들지 않습니다. backend voice_engine이 합니다.

take_new_samples: Gradio가 가끔 지금까지의 녹음 전체를 다시 보냅니다.
앞부분이 이전 조각과 같으면 새로 붙은 꼬리만 남깁니다.
"""

from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 16000
SPEECH_RMS = 0.008
MIN_SPEECH_SEC = 0.7
SILENCE_SEC = 0.7
MAX_SPEECH_SEC = 12.0


def to_mono_float(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size and float(np.max(np.abs(arr))) > 1.5:
        arr = arr / 32768.0
    return np.clip(arr, -1.0, 1.0)


def take_new_samples(previous: np.ndarray | None, chunk: np.ndarray) -> np.ndarray:
    """Keep only the new tail when Gradio resends the whole recording."""
    if previous is None or previous.size == 0 or chunk.size <= previous.size:
        return chunk
    width = previous.size
    if float(np.max(np.abs(chunk[:width] - previous))) < 1e-3:
        return chunk[width:]
    return chunk


class UtteranceBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.speech = np.zeros(0, dtype=np.float32)
        self.speech_sec = 0.0
        self.silence_sec = 0.0
        self.heard_speech = False
        self.previous_chunk: np.ndarray | None = None
        self.allow = False

    def reset_audio(self) -> None:
        with self._lock:
            self._clear_locked()

    def push(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
        if samples.size == 0:
            return None
        with self._lock:
            rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
            duration = samples.size / float(sample_rate)
            if rms >= SPEECH_RMS:
                if not self.heard_speech:
                    self.speech = samples.copy()
                else:
                    self.speech = np.concatenate([self.speech, samples])
                self.heard_speech = True
                self.speech_sec += duration
                self.silence_sec = 0.0
            elif self.heard_speech:
                self.speech = np.concatenate([self.speech, samples])
                self.silence_sec += duration

            ready = self.heard_speech and (
                self.silence_sec >= SILENCE_SEC or self.speech_sec >= MAX_SPEECH_SEC
            )
            if not ready:
                return None
            utterance = self.speech.copy()
            speech_sec = self.speech_sec
            self._clear_locked()
            if speech_sec < MIN_SPEECH_SEC:
                return None
            return utterance

    def flush(self) -> np.ndarray | None:
        with self._lock:
            if not self.heard_speech or self.speech_sec < MIN_SPEECH_SEC:
                self._clear_locked()
                return None
            utterance = self.speech.copy()
            self._clear_locked()
            return utterance

    def _clear_locked(self) -> None:
        self.speech = np.zeros(0, dtype=np.float32)
        self.speech_sec = 0.0
        self.silence_sec = 0.0
        self.heard_speech = False
