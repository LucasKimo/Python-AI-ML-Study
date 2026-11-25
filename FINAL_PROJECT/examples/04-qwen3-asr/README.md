# 04. Qwen3 ASR — PyTorch CUDA로 음성을 글자로 바꾸기

Windows + NVIDIA GPU(RTX 3060 등)에서 쓰는 예제입니다.  
vLLM은 사용하지 않습니다. PyTorch가 GPU에 모델을 올리고 전사합니다.

## 이 예제로 배우는 것

- 음성 파일 하나를 16kHz로 읽기
- `Qwen3ASRModel.from_pretrained`로 CUDA 장치에 모델 올리기
- `transcribe` 결과 문장 확인

GPU가 없거나 드라이버만 확인하고 싶을 때는 `demo_offline_print.py`만 실행하면 됩니다. 모델은 받지 않습니다.

## 필요한 것

| 항목 | 내용 |
|------|------|
| OS | Windows 10/11 |
| GPU | NVIDIA (RTX 3060 이상 권장, 12GB면 1.7B 모델이 들어갑니다) |
| 드라이버 | 최신 GeForce Game Ready 또는 Studio 드라이버 |
| Python | 3.11 이상 |
| 디스크 | 모델 다운로드용 여유 공간 약 5GB |

플래시 어텐션 패키지는 설치하지 않습니다. Windows에서 빌드가 자주 실패하고, 이 예제는 기본 어텐션으로 동작합니다.

## 설치

프로젝트 가상환경을 켠 뒤, **CUDA용 PyTorch를 먼저** 설치합니다.  
`pip install torch`만 하면 CPU 버전이 깔릴 수 있습니다.

```powershell
cd C:\Users\Administrator\ryuzy-project
.\.venv\Scripts\Activate.ps1

nvidia-smi
```

`nvidia-smi` 표 오른쪽 위에 `CUDA Version: 12.x`가 보이면 아래를 사용합니다.

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r examples\04-qwen3-asr\requirements-asr.txt
```

드라이버가 CUDA 12.8 이상을 가리키면 `cu124` 대신 `cu128` 인덱스를 써도 됩니다.  
[pytorch.org](https://pytorch.org/get-started/locally/)에서 Windows + CUDA 명령을 확인한 뒤 그 명령을 써도 됩니다.

설치 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from qwen_asr import Qwen3ASRModel; print('qwen-asr OK')"
```

`torch.cuda.is_available()`이 `True`이고 장치 이름에 `3060`이 보여야 합니다.  
`False`이면 CUDA 휠이 아니라 CPU용 PyTorch가 설치된 것입니다. 가상환경에서 `pip uninstall torch torchaudio -y` 후 CUDA 명령을 다시 실행하세요.

`qwen-asr`만 설치합니다. `vllm`은 설치하지 마세요. Windows에서 경로가 너무 길어 설치가 끊깁니다.

## 환경 변수 (선택)

기본값으로 바로 실행할 수 있습니다.

```powershell
$env:ASR_MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
$env:ASR_LANGUAGE="Korean"
$env:ASR_DTYPE="bfloat16"
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ASR_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B` | Hugging Face 모델 ID |
| `ASR_LANGUAGE` | `Korean` | 전사 언어. 자동 감지는 코드에서 `None` |
| `ASR_DTYPE` | `bfloat16` | RTX 3060은 bfloat16을 지원합니다. 문제가 있으면 `float16` |

메모리가 6GB인 3060 노트북이면 더 작은 모델을 씁니다.

```powershell
$env:ASR_MODEL_PATH="Qwen/Qwen3-ASR-0.6B"
```

첫 실행 때 Hugging Face에서 가중치를 받습니다. 인터넷이 필요합니다.

## 실행

짧은 wav 파일을 준비합니다. 16kHz가 아니어도 스크립트가 16kHz mono로 바꿉니다.

```powershell
cd C:\Users\Administrator\ryuzy-project\examples\04-qwen3-asr
python demo_offline_print.py
python transcribe_file.py sample.wav
```

성공하면 마지막에 `=== 결과 ===` 아래에 인식된 문장이 나옵니다.  
첫 로딩은 1~2분 걸릴 수 있습니다.

## 실시간 마이크 전사

파일 대신 말하면 몇 초마다 글자가 나옵니다. 회의 중 음성을 계속 받아 글로 바꾸는 흐름과 같습니다.

Qwen3의 `streaming_transcribe`는 vLLM 전용입니다. 이 PC에서는 그 API를 쓰지 않고, 마이크 입력을 4초 조각으로 잘라 `transcribe`를 반복 호출합니다.

```powershell
pip install sounddevice
python live_mic.py
```

말하면 약 4초 뒤 `>>` 뒤에 문장이 출력됩니다. 조용한 구간은 건너뜁니다. 끝내려면 `Ctrl+C`를 누릅니다.

마이크가 여러 개면 번호를 지정합니다.

```powershell
python live_mic.py --list-devices
$env:ASR_MIC_DEVICE="1"
python live_mic.py
```

조각을 더 짧게 하면 반응이 빨라지고, 문장은 더 잘릴 수 있습니다.

```powershell
$env:ASR_CHUNK_SECONDS="3"
```

## 화자를 구분해서 전사

누구 목소리인지 나눈 뒤, 그 구간만 Qwen3로 전사합니다. 본편 ASR이 pyannote로 화자를 나누고 Qwen3로 글을 만드는 흐름과 같습니다.

```powershell
pip install pyannote.audio
$env:HF_TOKEN="hf_본인토큰"
```

Hugging Face에서 `pyannote/speaker-diarization-community-1` 와 `pyannote/wespeaker-voxceleb-resnet34-LM` 사용 조건에 동의해야 합니다.

이름을 붙이려면 사람마다 5초씩 목소리를 등록합니다. 두 사람이 번갈아 등록하세요.

```powershell
python diarize_transcribe.py --enroll 김
python diarize_transcribe.py --enroll 이
python diarize_transcribe.py --mic
```

등록 전에는 `SPEAKER_00: ...` 처럼 나옵니다. 등록 후에는 비슷한 목소리를 `김: ...` 으로 표시합니다. 파일로 확인하려면 `python diarize_transcribe.py sample.wav` 입니다. 기본으로 소리가 큰 20초만 사용합니다.

## 코드에서 하는 일

```python
asr = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
results = asr.transcribe(audio=(wav16k, 16000), language="Korean")
print(results[0].text)
```

`Qwen3ASRModel.LLM(...)`은 vLLM 경로라 이 예제에서는 쓰지 않습니다.

## 문제 해결

| 증상 | 조치 |
|------|------|
| `CUDA GPU를 찾지 못했습니다` | `nvidia-smi` 확인 후 CUDA용 torch 재설치 |
| `No module named qwen_asr` | `pip install qwen-asr` |
| vLLM 긴 경로 오류 | vLLM 설치를 중단합니다. 이 예제에는 필요 없습니다 |
| CUDA out of memory | `ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B` |
| 모델 다운로드 실패 | 네트워크, 디스크 공간, Hugging Face 접속 확인 |
| dtype 관련 오류 | `$env:ASR_DTYPE="float16"` 후 다시 실행 |
| `No module named sounddevice` | `pip install sounddevice` |
| 마이크를 열지 못함 | `python live_mic.py --list-devices` 후 `ASR_MIC_DEVICE` 지정 |
| 실시간인데 `... 조용함`만 나옴 | 마이크를 가까이 하거나 `ASR_SILENCE_RMS=0.005` |
