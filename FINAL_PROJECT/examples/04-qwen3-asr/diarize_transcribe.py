"""화자를 구분해서 전사하는 예제.

pyannote가 '누가 언제 말했는지' 구간을 나누고,
각 구간을 Qwen3(PyTorch CUDA)가 글자로 바꿉니다.
미리 녹음해 둔 목소리와 비슷하면 SPEAKER_00 대신 이름으로 표시합니다.

준비:
  pip install pyannote.audio
  $env:HF_TOKEN="hf_..."

목소리 등록 (한 사람씩, 5초):
  python diarize_transcribe.py --enroll 김
  python diarize_transcribe.py --enroll 이

실행:
  python diarize_transcribe.py --mic
  python diarize_transcribe.py sample.wav

종료: Ctrl+C
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
HERE = Path(__file__).resolve().parent
SPEAKER_DIR = HERE / "speakers"
MIN_SEGMENT_SEC = 0.8


def load_dotenv() -> None:
    env_path = HERE.parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def hf_token() -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN 이 없습니다. Hugging Face 토큰을 설정하세요.")
        print('  $env:HF_TOKEN="hf_..."')
        print("pyannote 모델 페이지에서 라이선스 동의도 필요합니다.")
        sys.exit(1)
    return token


def load_16k_mono(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if int(sr) == SAMPLE_RATE:
        return wav
    duration = wav.shape[0] / float(sr)
    n16 = int(round(duration * SAMPLE_RATE))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def peak_normalize(wav: np.ndarray, target: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < 1e-6 or peak >= target:
        return wav
    return (wav * (target / peak)).astype(np.float32)


def record_seconds(seconds: float) -> np.ndarray:
    import sounddevice as sd

    print(f"{seconds:.0f}초 동안 말하세요...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def load_diarizer(token: str):
    import torch
    from pyannote.audio import Pipeline

    model_id = os.getenv(
        "DIARIZE_MODEL_PATH", "pyannote/speaker-diarization-community-1"
    )
    print(f"화자분리 모델 로드: {model_id}")
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def load_embedder(token: str):
    import torch
    from pyannote.audio import Inference, Model

    model_id = os.getenv(
        "SPEAKER_EMBED_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
    )
    print(f"목소리 임베딩 모델 로드: {model_id}")
    model = Model.from_pretrained(model_id, token=token)
    inference = Inference(model, window="whole")
    if torch.cuda.is_available():
        inference.to(torch.device("cuda"))
    return inference


def embedding_of(inference, wav: np.ndarray) -> np.ndarray:
    import torch

    waveform = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    emb = inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return arr
    return arr / norm


def enroll(name: str, token: str) -> None:
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    wav = peak_normalize(record_seconds(5))
    wav_path = SPEAKER_DIR / f"{name}.wav"
    sf.write(str(wav_path), wav, SAMPLE_RATE)
    inference = load_embedder(token)
    emb = embedding_of(inference, wav)
    np.save(SPEAKER_DIR / f"{name}.npy", emb)
    print(f"등록 완료: {name}  ({wav_path.name})")


def load_enrolled(token: str) -> dict[str, np.ndarray]:
    if not SPEAKER_DIR.exists():
        return {}
    people: dict[str, np.ndarray] = {}
    inference = None
    for wav_path in sorted(SPEAKER_DIR.glob("*.wav")):
        npy_path = wav_path.with_suffix(".npy")
        if npy_path.exists():
            people[wav_path.stem] = np.load(npy_path)
            continue
        if inference is None:
            inference = load_embedder(token)
        emb = embedding_of(inference, load_16k_mono(wav_path))
        np.save(npy_path, emb)
        people[wav_path.stem] = emb
    return people


def match_name(emb: np.ndarray, people: dict[str, np.ndarray], threshold: float) -> str | None:
    best_name = None
    best_score = threshold
    for name, saved in people.items():
        score = float(np.dot(emb, saved))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def diarize(pipeline, wav: np.ndarray) -> list[dict]:
    import torch

    waveform = torch.from_numpy(wav).unsqueeze(0).float()
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    annotation = getattr(output, "speaker_diarization", output)
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
        )
    segments.sort(key=lambda item: item["start"])
    return _merge_short(segments)


def _merge_short(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap < 0.5:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return [seg for seg in merged if seg["end"] - seg["start"] >= MIN_SEGMENT_SEC]


def label_segments(wav: np.ndarray, segments: list[dict], people: dict[str, np.ndarray], inference) -> None:
    threshold = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.45"))
    cache: dict[str, str] = {}
    for seg in segments:
        raw = seg["speaker"]
        if raw in cache:
            seg["speaker"] = cache[raw]
            continue
        if not people or inference is None:
            cache[raw] = raw
            continue
        start = int(seg["start"] * SAMPLE_RATE)
        end = int(seg["end"] * SAMPLE_RATE)
        piece = wav[start:end]
        if piece.size < int(0.5 * SAMPLE_RATE):
            cache[raw] = raw
            continue
        name = match_name(embedding_of(inference, piece), people, threshold)
        cache[raw] = name or raw
        seg["speaker"] = cache[raw]


def transcribe_segments(asr, wav: np.ndarray, segments: list[dict], language: str) -> list[str]:
    lines = []
    for seg in segments:
        start = int(seg["start"] * SAMPLE_RATE)
        end = int(seg["end"] * SAMPLE_RATE)
        piece = peak_normalize(wav[start:end])
        if piece.size < int(MIN_SEGMENT_SEC * SAMPLE_RATE):
            continue
        results = asr.transcribe(audio=(piece, SAMPLE_RATE), language=language)
        text = (getattr(results[0], "text", None) or "").strip()
        if not text:
            continue
        line = f"{seg['speaker']}: {text}"
        lines.append(line)
        print(f">> {line}")
    return lines


def slice_for_demo(wav: np.ndarray) -> np.ndarray:
    max_seconds = float(os.getenv("ASR_MAX_SECONDS", "20"))
    want = int(max_seconds * SAMPLE_RATE)
    if max_seconds <= 0 or wav.shape[0] <= want:
        return wav
    hop = SAMPLE_RATE * 5
    best_start = 0
    best_rms = -1.0
    limit = min(wav.shape[0], SAMPLE_RATE * 600)
    for start in range(0, max(1, limit - want + 1), hop):
        chunk = wav[start : start + want]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    print(f"파일이 길어 {best_start / SAMPLE_RATE:.0f}초 지점부터 {max_seconds:.0f}초만 사용합니다.")
    return wav[best_start : best_start + want]


def load_asr():
    import torch
    from qwen_asr import Qwen3ASRModel

    if not torch.cuda.is_available():
        print("CUDA GPU를 찾지 못했습니다.")
        sys.exit(1)
    model_path = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
    dtype_name = os.getenv("ASR_DTYPE", "bfloat16").lower()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    print(f"Qwen3 ASR 로드: {model_path}")
    return Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=128,
    )


def run_once(asr, pipeline, inference, people: dict[str, np.ndarray], wav: np.ndarray, language: str) -> None:
    rms = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0
    if rms < float(os.getenv("ASR_SILENCE_RMS", "0.01")):
        print("... 조용해서 건너뜁니다.")
        return
    segments = diarize(pipeline, wav)
    if not segments:
        print("... 화자 구간이 없습니다.")
        return
    label_segments(wav, segments, people, inference)
    transcribe_segments(asr, wav, segments, language)


def main() -> None:
    token = hf_token()
    if len(sys.argv) >= 3 and sys.argv[1] == "--enroll":
        enroll(sys.argv[2], token)
        return

    people = load_enrolled(token)
    inference = load_embedder(token) if people else None
    if people:
        print("등록된 목소리:", ", ".join(people))
    else:
        print("등록된 목소리가 없습니다. 결과는 SPEAKER_00 처럼 표시됩니다.")
        print('이름까지 붙이려면 먼저: python diarize_transcribe.py --enroll 김')

    asr = load_asr()
    pipeline = load_diarizer(token)
    language = os.getenv("ASR_LANGUAGE", "Korean")

    if "--mic" in sys.argv:
        chunk = float(os.getenv("ASR_CHUNK_SECONDS", "12"))
        print(f"\n마이크를 엽니다. {chunk:.0f}초마다 화자를 나눠 전사합니다. 종료는 Ctrl+C\n")
        try:
            while True:
                wav = record_seconds(chunk)
                run_once(asr, pipeline, inference, people, wav, language)
        except KeyboardInterrupt:
            print("\n종료합니다.")
        return

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python diarize_transcribe.py --enroll 김")
        print("  python diarize_transcribe.py --mic")
        print("  python diarize_transcribe.py sample.wav")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"파일 없음: {audio_path}")
        sys.exit(1)
    wav = slice_for_demo(load_16k_mono(audio_path))
    run_once(asr, pipeline, inference, people, wav, language)


if __name__ == "__main__":
    main()
