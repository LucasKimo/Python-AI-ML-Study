# 화자분리
# 한 녹음 안에서 누가 언제 말했는지 구간을 나누는 기술
"""
0.0 ~ 2.1초 SPEAKER_00
2.1 ~ 4.3초 SPEAKER_01
4.4 ~ 7.1초 SPEAKER_00
...
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
SPEAKER_DIR = HERE / "speakers"
PEOPLE_PATH = SPEAKER_DIR / "participants.json"
SAMPLE_RATE = 16000
ENROLL_SECONDS = 5.0
READ_ALOUD = (
    "안녕하세요. 저는 이번 회의 참가자입니다. "
    "제 목소리를 등록하기 위해 이 문장을 또박또박 읽겠습니다."
)


def load_dotenv() -> None:
    env_path = ROOT / ".env"
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
        print("HF_TOKEN 이 없습니다. Hugging Face 읽기 토큰을 설정하세요.")
        print('  $env:HF_TOKEN="hf_..."')
        print("모델 페이지에서 라이선스 동의도 필요합니다.")
        sys.exit(1)
    return token


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name, flags=re.UNICODE).strip(" ._")
    return cleaned or "speaker"


def load_people() -> list[str]:
    if not PEOPLE_PATH.exists():
        return []
    data = json.loads(PEOPLE_PATH.read_text(encoding="utf-8"))
    return [str(name) for name in data.get("participants", [])]


def save_people(names: list[str]) -> None:
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    PEOPLE_PATH.write_text(
        json.dumps({"participants": names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def embedding_path(name: str) -> Path:
    return SPEAKER_DIR / f"{safe_name(name)}.npy"


def add_participant(name: str) -> None:
    """회의 명단에 이름만 올립니다. 목소리는 아직 없습니다."""
    name = name.strip()
    if not name:
        print("이름을 입력하세요.")
        return
    people = load_people()
    if name in people:
        print(f"이미 참가자입니다: {name}")
        return
    people.append(name)
    save_people(people)
    print(f"참가자 등록: {name}")
    print("다음은 그 사람의 목소리를 녹음합니다.")
    print(f"  python identify_speaker.py enroll {name}")


def show_people() -> None:
    people = load_people()
    if not people:
        print("등록된 참가자가 없습니다.")
        print("  python identify_speaker.py add 김매니저")
        return
    print("참가자")
    for name in people:
        state = "목소리 등록됨" if embedding_path(name).exists() else "목소리 아직"
        print(f"  - {name}: {state}")


def mic_device():
    raw = os.getenv("MIC_DEVICE", "").strip()
    return int(raw) if raw else None


def record_seconds(seconds: float) -> np.ndarray:
    import sounddevice as sd

    print(f"{seconds:.0f}초 동안 읽어 주세요.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=mic_device(),
    )
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def record_one_utterance() -> np.ndarray | None:
    """말소리가 시작됐다가 0.7초 조용해지면 그 덩어리를 돌려줍니다."""
    import sounddevice as sd

    # 말소리로 판단할 최소 음량. 마이크 소리가 작으면 $env:SPEECH_RMS="0.005" 처럼 낮춥니다.
    speech_rms = float(os.getenv("SPEECH_RMS", "0.01"))
    block = int(0.1 * SAMPLE_RATE)  # 0.1초 단위로 소리 크기를 검사
    speech: list[np.ndarray] = []
    heard = False      # 말소리가 한 번이라도 들렸는지
    silence = 0.0      # 말이 시작된 뒤 이어진 조용한 시간(초)
    speech_sec = 0.0   # 말소리가 들린 시간(초)
    print("듣는 중... 한 문장을 말해 보세요. 멈추면 누구인지 출력합니다.")
    # 0.1초마다 sd.rec()로 마이크를 새로 열고 닫으면, 열 때마다 앞부분이 무음(0)으로
    # 녹음되어 말소리를 놓치기 쉽습니다. 그래서 스트림을 한 번만 열고 계속 읽습니다.
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=mic_device(),
        blocksize=block,
    ) as stream:
        while True:
            audio, _ = stream.read(block)
            chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
            # 현재 음량을 같은 줄에 계속 덮어써서 보여 줌 (마이크가 소리를 받는지 확인용)
            bar = "#" * min(int(rms / speech_rms * 10), 40)
            print(f"\r음량 {rms:.4f} {bar:<40}", end="", flush=True)
            if rms >= speech_rms:
                speech.append(chunk)
                heard = True
                silence = 0.0
                speech_sec += 0.1
            elif heard:
                speech.append(chunk)  # 말 사이의 짧은 쉼도 함께 저장
                silence += 0.1
            # 말이 시작된 뒤 0.7초 조용하거나, 8초 넘게 말하면 한 문장으로 끝냄
            if heard and (silence >= 0.7 or speech_sec >= 8.0):
                break
    print()  # 음량 표시 줄 끝내기
    if speech_sec < 0.6:
        print("너무 짧아서 건너뜁니다.")
        return None
    return np.concatenate(speech)


def get_embedder():
    import torch
    from pyannote.audio.pipelines.speaker_verification import (
        PretrainedSpeakerEmbedding,
    )

    # pyannote/ 버전은 PyTorch로 동작합니다.
    # (hbredin/ 버전은 ONNX 모델이라 onnxruntime 설치가 추가로 필요)
    model_id = os.getenv(
        "SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"목소리 임베딩 모델 로드: {model_id} ({device})")
    return PretrainedSpeakerEmbedding(model_id, device=device, token=hf_token())


def embed_wave(embedder, wav: np.ndarray) -> np.ndarray:
    """16kHz 파형을 길이 1인 목소리 벡터로 바꿉니다."""
    import torch

    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size < int(SAMPLE_RATE * 0.6):
        raise ValueError("녹음이 너무 짧습니다. 문장을 끝까지 읽어 주세요.")
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms < 0.008:
        raise ValueError("목소리가 거의 들리지 않습니다. 마이크를 가까이 하고 다시 말해 주세요.")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    tensor = torch.from_numpy(np.ascontiguousarray(audio)).view(1, 1, -1)
    vector = np.asarray(embedder(tensor)[0], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0 or not np.isfinite(vector).all():
        raise RuntimeError("목소리 특징을 만들지 못했습니다. 다시 녹음해 주세요.")
    return vector / norm


def enroll(name: str) -> None:
    """명단에 있는 사람의 목소리를 벡터로 저장합니다."""
    name = name.strip()
    people = load_people()
    if name not in people:
        print(f"참가자 명단에 없습니다: {name or '(빈 이름)'}")
        print("먼저 이름을 등록하세요.")
        print(f"  python 7_pyannote.py add {name or '김매니저'}")
        return
    print("읽을 문장:")
    print(f"  {READ_ALOUD}")
    print()
    try:
        wav = record_seconds(ENROLL_SECONDS)
        embedder = get_embedder()
        vector = embed_wave(embedder, wav)
    except Exception as exc:  # noqa: BLE001
        print(f"등록 실패: {exc}")
        return
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path(name), vector)
    print(f"목소리 등록 완료: {name}")


def load_embeddings() -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for name in load_people():
        path = embedding_path(name)
        if path.exists():
            found[name] = np.load(path)
    return found


def match_speaker(vector: np.ndarray, people: dict[str, np.ndarray]) -> tuple[str, float]:
    """등록 벡터와 내적(코사인 유사도)이 가장 큰 이름을 고릅니다."""
    threshold = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.5"))
    best_name = "미확인"
    best_score = -1.0
    for name, saved in people.items():
        score = float(np.dot(vector, np.asarray(saved, dtype=np.float32).reshape(-1)))
        if score > best_score:
            best_name = name
            best_score = score
    if best_score < threshold:
        return "미확인", best_score
    return best_name, best_score


def listen() -> None:
    people = load_embeddings()
    if not people:
        print("등록된 목소리가 없습니다. 참가자를 추가한 뒤 목소리를 녹음하세요.")
        show_people()
        return
    missing = [name for name in load_people() if name not in people]
    print("비교할 목소리:", ", ".join(people))
    if missing:
        print("아직 목소리가 없는 참가자:", ", ".join(missing))
    try:
        embedder = get_embedder()
    except Exception as exc:  # noqa: BLE001
        print(f"모델 로드 실패: {exc}")
        return
    print("종료는 Ctrl+C")
    print()
    try:
        while True:
            wav = record_one_utterance()
            if wav is None:
                continue
            try:
                vector = embed_wave(embedder, wav)
            except ValueError as exc:
                print(exc)
                continue
            name, score = match_speaker(vector, people)
            print(f">> {name}  (유사도 {score:.2f})")
            print()
    except KeyboardInterrupt:
        print("\n종료합니다.")


def menu() -> None:
    print("회의 참가자 목소리를 등록하고, 누가 말하는지 확인합니다.")
    print("1) 참가자 추가   2) 목소리 등록   3) 말하기   4) 목록   q) 종료")
    while True:
        choice = input("\n선택> ").strip().lower()
        if choice in {"q", "quit", "0"}:
            return
        if choice == "1":
            add_participant(input("참가자 이름> "))
        elif choice == "2":
            show_people()
            enroll(input("목소리를 등록할 이름> "))
        elif choice == "3":
            listen()
        elif choice == "4":
            show_people()
        else:
            print("1, 2, 3, 4, q 중에서 고르세요.")


def main() -> None:
    if "--list-devices" in sys.argv:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if len(sys.argv) == 1:
        menu()
        return
    command = sys.argv[1]
    if command == "add" and len(sys.argv) >= 3:
        add_participant(" ".join(sys.argv[2:]))
        return
    if command == "enroll" and len(sys.argv) >= 3:
        enroll(" ".join(sys.argv[2:]))
        return
    if command == "list":
        show_people()
        return
    if command == "listen":
        listen()
        return
    print("사용법:")
    print("  python identify_speaker.py")
    print("  python identify_speaker.py add 김매니저")
    print("  python identify_speaker.py enroll 김매니저")
    print("  python identify_speaker.py listen")
    print("  python identify_speaker.py list")
    sys.exit(1)


if __name__ == "__main__":
    main()