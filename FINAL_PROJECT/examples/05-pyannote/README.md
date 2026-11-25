# 05. pyannote — 누가 말했는지 구간 나누기 (화자분리)

## 이 예제로 배우는 것

- Speaker Diarization(화자분리)이 뭔지
- pyannote Pipeline으로 `SPEAKER_00`, `SPEAKER_01` 구간을 얻는 방법
- Hugging Face 토큰·라이선스 설정
- 본편에서 Qwen3와 어떻게 합쳐지는지

## 화자분리란?

한 녹음 안에서 **누가 언제 말했는지** 구간을 나누는 기술입니다.

```
0.0~2.1초  SPEAKER_00
2.1~5.0초  SPEAKER_01
5.0~7.3초  SPEAKER_00
```

본편 흐름:

```
pyannote로 구간 찾기 → 각 구간을 Qwen3가 글자로 변환 → 회의록
```

---

## 설치해야 하는 모듈

```powershell
pip install torch torchaudio pyannote.audio soundfile numpy
```

| 모듈 | 용도 |
|------|------|
| `pyannote.audio` | 화자분리 Pipeline |
| `torch` / `torchaudio` | 모델 실행 |
| `soundfile` / `numpy` | 오디오 입출력 |

확인:

```powershell
python -c "from pyannote.audio import Pipeline; print('pyannote OK')"
```

---

## Hugging Face 설정 (필수에 가까움)

1. https://huggingface.co 계정 생성
2. Access Token 발급 (Read 권한)
3. 모델 페이지에서 **라이선스 동의**  
   - 예: `pyannote/speaker-diarization-community-1`  
   - 또는 수업/환경에 맞는 pyannote diarization 모델
4. 환경 변수 설정

```powershell
$env:HF_TOKEN="hf_xxxxxxxx"
$env:DIARIZE_MODEL_PATH="pyannote/speaker-diarization-community-1"
```

Linux:

```bash
export HF_TOKEN=hf_xxxxxxxx
export DIARIZE_MODEL_PATH=pyannote/speaker-diarization-community-1
```

토큰이 없거나 라이선스 미동의면 모델 다운로드가 거부됩니다.

---

## 파일 설명

| 파일 | 내용 | 모델 |
|------|------|------|
| `demo_offline_print.py` | 개념 설명만 | 불필요 |
| `diarize_file.py` | wav → 이름 없는 화자 구간 | 필요 |
| `identify_speaker.py` | 참가자·목소리 등록 후, 말하면 누구인지 출력 | 필요 |
| `requirements-diarize.txt` | 패키지 목록 | - |

---

## 실행 방법

### A. 개념만

```powershell
cd examples\05-pyannote
python demo_offline_print.py
```

### B. 실제 화자분리

```powershell
$env:HF_TOKEN="hf_..."
python diarize_file.py path\to\meeting.wav
```

출력 예:

```
SPEAKER_00  0.52 ~ 3.10
SPEAKER_01  3.10 ~ 6.44
...
```

---

## 코드 핵심 아이디어

```python
from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN,
)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

annotation = pipeline({"waveform": waveform, "sample_rate": 16000})
for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(speaker, turn.start, turn.end)
```

본편 연결: `asr/app/services/pipeline.py` 의 `diarize_segments()`.

---

## 참가자 등록 후 누구인지 맞추기

`diarize_file.py` 는 `SPEAKER_00` 처럼 구간만 나눕니다. 실명은 모릅니다.  
회의에서 쓰는 흐름은 다릅니다. 사람을 먼저 명단에 올리고, 그 사람 목소리를 벡터로 저장한 뒤, 새로 말한 목소리와 코사인 유사도로 비교합니다. 본편은 `backend/app/voice_engine.py` 가 이 일을 합니다.

임베딩 모델은 파형을 숫자 벡터로 바꿉니다. 같은 사람이면 문장이 달라도 벡터 방향이 비슷합니다. 가장 높은 유사도가 0.5 이상일 때만 그 이름으로 봅니다. 낮으면 `미확인`입니다.

추가 설치:

```powershell
pip install sounddevice
```

Hugging Face에서 `hbredin/wespeaker-voxceleb-resnet34-LM` 사용 조건에도 동의합니다. `HF_TOKEN` 은 위와 같습니다.

한 명씩 이름과 목소리를 등록한 뒤 말해 봅니다.

```powershell
cd examples\05-pyannote
python identify_speaker.py add 김매니저
python identify_speaker.py enroll 김매니저
python identify_speaker.py add 이개발
python identify_speaker.py enroll 이개발
python identify_speaker.py listen
```

`enroll` 은 안내 문장을 5초 동안 읽게 합니다. `listen` 은 말이 끊기면 `김매니저  (유사도 0.72)` 처럼 출력합니다. 종료는 `Ctrl+C` 입니다.

인자 없이 `python identify_speaker.py` 를 실행하면 같은 일을 메뉴로 할 수 있습니다. 등록 목록은 `python identify_speaker.py list` 입니다.

마이크가 여러 개면 `python identify_speaker.py --list-devices` 로 번호를 본 뒤 `$env:MIC_DEVICE="1"` 을 지정합니다. 목소리 파일은 `speakers/` 에 저장되며 깃에는 올리지 않습니다.

---

## 연습 과제

1. 구간 길이가 0.8초 미만인 것은 버려 보세요. (본편 `MIN_SEGMENT_SEC`)
2. 같은 화자가 0.5초 이내로 이어지면 구간을 합쳐 보세요.
3. 각 구간 wav를 잘라 저장해 보세요. (`soundfile.write`)

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| 401 / gated repo | HF 토큰, 모델 페이지 라이선스 동의 |
| `HF_TOKEN` 없음 | 환경변수 설정 후 터미널 재시작 |
| 느림 | GPU 사용 여부 확인 (`cuda`) |
| 화자가 너무 많이 쪼개짐 | 인접 구간 merge 로직 추가 |
