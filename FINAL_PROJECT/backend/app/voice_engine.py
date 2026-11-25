"""실시간 회의용 화자 이름 맞추기 + 문장 인식.

업로드 파일 파이프라인(asr/app/services/pipeline.py)과 하는 일이 다릅니다.
  - 파일 경로: pyannote가 SPEAKER_00처럼 '서로 다른 사람'만 나눕니다. 실명을 모릅니다.
  - 이 파일: 회의 전에 사람마다 목소리를 등록해 두고, 말할 때마다 그 이름에 맞춥니다.
    글자는 Qwen3가 아니라 로컬 Whisper가 만듭니다. 말이 끊길 때마다 화면에 바로 올리기 위해서입니다.

이론 — 목소리 임베딩과 코사인 유사도
------------------------------------
임베딩 모델(WeSpeaker 계열, pyannote의 PretrainedSpeakerEmbedding)은
파형을 고정 길이 숫자 벡터로 바꿉니다. 이 벡터가 그 사람의 목소리 지문입니다.
문장 내용이 달라도, 같은 사람이면 벡터의 방향이 비슷하게 나오도록 학습되어 있습니다.

등록(enroll) 때 그 벡터를 SQLite voice_profiles에 저장합니다.
실시간 구간이 들어오면 같은 모델로 벡터를 만들고, 등록분과 코사인 유사도를 잽니다.

  코사인 유사도 = (A · B) / (|A| |B|)
  같은 방향이면 1에 가깝고, 직각이면 0, 반대 방향이면 -1입니다.
  embed_wave()가 이미 길이를 1로 맞춰 두므로 비교는 방향만 봅니다.

가장 높은 점수가 SPEAKER_MATCH_THRESHOLD(기본 0.5)보다 낮으면
누구의 목소리로 보기 어려워 "미확인"으로 남깁니다.
임계값을 낮추면 다른 사람을 그 이름으로 붙일 위험이 커지고,
높이면 본인인데도 미확인이 늘어납니다.

Whisper는 소음·무음에서 같은 구절을 반복하거나
"시청해 주셔서 감사합니다" 같은 환각 문장을 만들 수 있습니다.
_usable_text()가 그 문장을 비워, 회의록에 들어가지 않게 합니다.
"""

from __future__ import annotations

import re
import threading
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from shared import config, db
from shared.prompts import ENROLL_PROMPT

_lock = threading.Lock()
_embedder: Any = None
_whisper: Any = None

_HALLUCINATIONS = (
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "한글자막",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
)


def _safe_name(speaker: str) -> str:
    cleaned = re.sub(r"[^\w가-힣.-]+", "_", speaker, flags=re.UNICODE).strip("._")
    return cleaned or "speaker"


def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            import torch
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )

            print(f"[voice] loading speaker embedding: {config.SPEAKER_EMBEDDING_MODEL}")
            _embedder = PretrainedSpeakerEmbedding(
                config.SPEAKER_EMBEDDING_MODEL,
                device=torch.device("cpu"),
                token=config.HF_TOKEN or None,
            )
            print("[voice] speaker embedding ready")
    return _embedder


def get_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    with _lock:
        if _whisper is None:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            model_id = config.WHISPER_MODEL
            print(f"[voice] loading whisper: {model_id} on {device}")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            _whisper = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=device,
            )
            print("[voice] whisper ready")
    return _whisper


def _as_float(wav: np.ndarray) -> np.ndarray:
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("녹음이 비어 있습니다.")
    if float(np.max(np.abs(audio))) > 1.5:
        audio = audio / 32768.0
    return audio


def _peak_norm(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    return audio / peak * 0.9


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav)) + 1e-12))


def embed_wave(wav: np.ndarray) -> list[float]:
    """파형을 길이 1인 목소리 벡터로 바꿉니다. 너무 짧거나 거의 무음이면 등록을 거절합니다."""
    import torch

    audio = _as_float(wav)
    if audio.size < 16000 * 0.6:
        raise ValueError("녹음이 너무 짧습니다. 안내 문장을 끝까지 읽어 주세요.")
    if _rms(audio) < 0.008:
        raise ValueError("목소리가 거의 들리지 않습니다. 마이크를 가까이 하고 다시 읽어 주세요.")
    audio = _peak_norm(audio)
    tensor = torch.from_numpy(np.ascontiguousarray(audio)).view(1, 1, -1)
    vector = np.asarray(get_embedder()(tensor)[0], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise RuntimeError("목소리 특징을 만들지 못했습니다. 다시 녹음해 주세요.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise RuntimeError("목소리 특징이 비어 있습니다. 다시 녹음해 주세요.")
    return (vector / norm).astype(np.float32).tolist()


def _cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def match_speaker(meeting_id: int, wav: np.ndarray) -> tuple[str, float]:
    """등록된 벡터 중 코사인 유사도가 가장 높은 이름을 고릅니다. 임계값 미만이면 미확인입니다."""
    current = embed_wave(wav)
    best_name = "미확인"
    best_score = -1.0
    for profile in db.list_voice_embeddings(meeting_id):
        score = _cosine(current, profile["embedding"])
        if score > best_score:
            best_name = str(profile["speaker"])
            best_score = score
    if best_score < config.SPEAKER_MATCH_THRESHOLD:
        return "미확인", best_score
    return best_name, best_score


def _is_repetitive(text: str) -> bool:
    """Whisper loops the same phrase on noise. Drop those lines."""
    parts = [part for part in re.split(r"[\s.!?。,…~]+", text) if part]
    if parts:
        top = max(parts.count(part) for part in set(parts))
        if top >= 4 and top / len(parts) >= 0.6:
            return True
    compact = re.sub(r"[\s.!?。,…~]+", "", text)
    if len(compact) < 12:
        return False
    raw = compact.encode("utf-8")
    if len(raw) / max(len(zlib.compress(raw, 9)), 1) >= 2.4:
        return True
    for size in range(1, min(12, len(compact) // 4) + 1):
        counts: dict[str, int] = {}
        for start in range(0, len(compact) - size + 1, size):
            unit = compact[start : start + size]
            counts[unit] = counts.get(unit, 0) + 1
        best = max(counts.values(), default=0)
        if best >= 4 and best * size >= int(len(compact) * 0.65):
            return True
    return False


def _usable_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .!?。")
    lowered = cleaned.lower()
    if len(cleaned) < 2:
        return ""
    for snippet in _HALLUCINATIONS:
        if snippet in lowered or snippet in cleaned:
            return ""
    if _is_repetitive(cleaned):
        return ""
    return cleaned


def _has_speech(wav: np.ndarray) -> bool:
    """Ignore a chunk that is mostly room noise."""
    audio = _as_float(wav)
    if audio.size < int(16000 * 0.5) or _rms(audio) < 0.01:
        return False
    frame = 480
    count = audio.size // frame
    if count < 8:
        return False
    frames = audio[: count * frame].reshape(count, frame)
    levels = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    return float(np.mean(levels >= 0.015)) >= 0.12


def transcribe(wav: np.ndarray) -> str:
    audio = _peak_norm(_as_float(wav))
    pipe = get_whisper()
    result = pipe(
        {"array": audio, "sampling_rate": 16000},
        generate_kwargs={
            "language": "korean",
            "task": "transcribe",
            "condition_on_prev_tokens": False,
        },
    )
    return _usable_text(str(result.get("text") or ""))


def registration_view(meeting_id: int) -> dict[str, Any] | None:
    view = db.voice_registration(meeting_id)
    if view is None:
        return None
    view["prompt"] = ENROLL_PROMPT
    return view


def enroll(meeting_id: int, speaker: str, wav: np.ndarray) -> dict[str, Any]:
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise ValueError("회의를 찾을 수 없습니다.")
    name = speaker.strip()
    participants = list(meeting.get("participants") or [])
    if not name:
        raise ValueError("참가자를 선택하세요.")
    if name not in participants:
        raise ValueError("이 회의 참가자 중에서 선택하세요.")
    embedding = embed_wave(wav)
    folder = Path(config.VOICE_DIR) / str(meeting_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_safe_name(name)}.wav"
    _write_wav(path, _peak_norm(_as_float(wav)))
    db.upsert_voice_profile(
        meeting_id,
        name,
        ENROLL_PROMPT,
        str(path),
        embedding,
    )
    view = registration_view(meeting_id)
    assert view is not None
    if view["ready"] and meeting.get("status") in {"draft", "enrolling"}:
        db.update_meeting(meeting_id, status="voices_ready")
    elif meeting.get("status") == "draft":
        db.update_meeting(meeting_id, status="enrolling")
    print(f"[voice] enrolled meeting={meeting_id} speaker={name}")
    return view


def transcribe_live(meeting_id: int, wav: np.ndarray) -> dict[str, Any]:
    view = registration_view(meeting_id)
    if view is None:
        raise ValueError("회의를 찾을 수 없습니다.")
    if not view["ready"]:
        raise ValueError("모든 참가자 목소리를 등록한 뒤 실시간 회의를 시작하세요.")
    if not _has_speech(wav):
        return {
            "skipped": True,
            "reason": "말소리가 거의 없어서 건너뛰었습니다.",
            "transcript": db.get_meeting(meeting_id).get("transcript") or [],  # type: ignore[union-attr]
        }
    speaker, score = match_speaker(meeting_id, wav)
    text = transcribe(wav)
    if not text:
        return {
            "skipped": True,
            "reason": "이번 구간에서는 문장을 인식하지 못했습니다.",
            "speaker": speaker,
            "score": round(score, 3),
            "transcript": db.get_meeting(meeting_id).get("transcript") or [],  # type: ignore[union-attr]
        }
    updated = db.append_utterance(meeting_id, speaker, text)
    print(
        f"[voice] live meeting={meeting_id} speaker={speaker} "
        f"score={score:.3f} text={text[:40]}"
    )
    return {
        "skipped": False,
        "speaker": speaker,
        "text": text,
        "score": round(score, 3),
        "transcript": (updated or {}).get("transcript") or [],
    }


def _write_wav(path: Path, wav: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), wav, 16000, subtype="PCM_16")
