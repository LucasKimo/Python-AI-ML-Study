"""파일 하나를 '누가 / 언제 / 무슨 말'로 바꿉니다. Qwen3 ASR + pyannote.

이론 — 화자분리와 음성인식은 다른 일이다
----------------------------------------
사람 귀는 한 번에 하지만, 이 파이프라인은 두 모델로 나눕니다.

1. 화자분리 (diarization, pyannote)
   입력은 파형뿐입니다. 글자는 모릅니다.
   "이 시각부터 이 시각까지는 목소리 A, 그다음은 목소리 B"처럼
   시간을 구간으로 자르고 같은 목소리끼리 묶습니다.
   이름은 SPEAKER_00 같은 임시 번호입니다. 회의 참가자 실명과 자동으로 연결되지 않습니다.
   (실시간 회의의 실명 맞추기는 voice_engine.py의 목소리 등록이 담당합니다. 모델도 다릅니다.)

   pyannote 안에서는 대략 다음이 일어납니다.
   - 음성 활동 감지: 말소리인 구간과 침묵을 가른다.
   - 화자 임베딩: 짧은 조각마다 목소리 지문(숫자 벡터)을 만든다.
   - 군집화: 지문이 비슷한 조각을 한 사람으로 묶는다.
   그래서 "몇 명이 말하는지"를 미리 알려 주지 않아도 구간이 나옵니다.

2. 음성인식 (ASR, Qwen3)
   잘린 구간마다 "이 소리의 문장이 무엇이냐"만 묻습니다.
   파일 전체를 한 번에 넘기면 화자가 바뀌는 지점이 글에 남지 않습니다.
   구간을 나눠 넣는 이유가 그것입니다.

3. 합치기와 버리기 (_merge_adjacent)
   같은 사람이 잠깐 쉬었다가 이어서 말하면 pyannote가 구간을 두 개로 낼 수 있습니다.
   간격이 MERGE_GAP_SEC(기본 0.5초)보다 짧고 화자 번호가 같으면 한 구간으로 붙입니다.
   MIN_SEGMENT_SEC(기본 0.8초)보다 짧은 조각은 숨·추임새로 보고 전사하지 않습니다.

4. 16kHz 모노
   두 모델 모두 1채널, 초당 16000샘플을 기준으로 학습된 입력에 맞춥니다.
   스테레오는 좌우를 평균 내 한 줄로 만들고, 다른 샘플링 레이트는 보간으로 16kHz에 맞춥니다.
   그 변환은 audio_utils.load_audio_16k 에 있습니다.

구간이 하나도 안 남으면 파일 전체를 SPEAKER_00 한 명의 말로 전사하는 보조 경로가 있습니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from asr.app.core import models as model_hub
from asr.app.services.audio_utils import load_audio_16k
from shared import config

KST = timezone(timedelta(hours=9))


def _format_ts(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def diarize_segments(wav16k: np.ndarray) -> list[dict[str, Any]]:
    """pyannote로 '누가 언제 말했는지'만 구합니다. text 칸은 아직 없습니다.

    waveform은 (채널, 샘플) 모양이라 모노 1차원 배열 앞에 채널 축을 하나 붙입니다.
    itertracks는 (시간 구간, 트랙, 화자 라벨)을 돌려줍니다.
    """
    import torch

    pipeline = model_hub.get_diarizer()
    waveform = torch.from_numpy(wav16k).unsqueeze(0).float()
    annotation = pipeline({"waveform": waveform, "sample_rate": 16000})

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
    segments.sort(key=lambda s: s["start"])
    return segments


def _merge_adjacent(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap < config.MERGE_GAP_SEC:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return [
        s
        for s in merged
        if (s["end"] - s["start"]) >= config.MIN_SEGMENT_SEC
    ]


def asr_chunked(wav16k: np.ndarray, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """화자 구간마다 파형을 잘라 Qwen3에 한꺼번에 넘기고, 글자를 구간에 붙입니다.

    샘플 위치 = 초 * 16000. 너무 짧은 조각은 모델에 넣지 않습니다.
    transcribe()는 조각 리스트와 같은 순서로 결과를 돌려줍니다.
    """
    merged = _merge_adjacent(segments)
    if not merged:
        # fallback: whole-file ASR as single speaker
        text = asr_full(wav16k)
        if not text:
            return []
        return [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": float(len(wav16k) / 16000.0),
                "text": text,
            }
        ]

    audio_chunks = []
    valid = []
    for seg in merged:
        start = int(seg["start"] * 16000)
        end = int(seg["end"] * 16000)
        chunk = wav16k[start:end]
        if chunk.size < int(16000 * config.MIN_SEGMENT_SEC):
            continue
        audio_chunks.append((chunk, 16000))
        valid.append(seg)

    if not audio_chunks:
        return []

    asr = model_hub.get_asr()
    results = asr.transcribe(audio=audio_chunks, language=config.ASR_LANGUAGE)

    out: list[dict[str, Any]] = []
    for seg, res in zip(valid, results):
        text = (getattr(res, "text", None) or str(res) or "").strip()
        if text:
            out.append({**seg, "text": text})
    return out


def asr_full(wav16k: np.ndarray) -> str:
    asr = model_hub.get_asr()
    results = asr.transcribe(audio=[(wav16k, 16000)], language=config.ASR_LANGUAGE)
    return (getattr(results[0], "text", None) or "").strip()


def to_utterances(segments_with_text: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(KST)
    utterances = []
    for i, seg in enumerate(segments_with_text, start=1):
        utterances.append(
            {
                "seq": i,
                "speaker": seg.get("speaker", f"SPEAKER_{i:02d}"),
                "text": seg.get("text", ""),
                "timestamp": _format_ts(float(seg.get("start", 0.0))),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "absolute_time": (
                    now + timedelta(seconds=float(seg.get("start", 0.0)))
                ).strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    return utterances


def transcribe_file(path: str) -> dict[str, Any]:
    wav = load_audio_16k(path)
    return _run_pipeline(wav)


def _run_pipeline(wav16k: np.ndarray) -> dict[str, Any]:
    if wav16k.size == 0:
        raise ValueError("오디오가 비어 있습니다.")

    segments = diarize_segments(wav16k)
    with_text = asr_chunked(wav16k, segments)
    utterances = to_utterances(with_text)
    return {
        "utterances": utterances,
        "mode": "qwen3+pyannote",
        "segment_count": len(segments),
        "message": (
            f"pyannote 세그먼트 {len(segments)}개 → "
            f"Qwen3 전사 {len(utterances)}개 발화"
        ),
    }
