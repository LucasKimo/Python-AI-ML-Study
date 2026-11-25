# 2. 설치 · 실행

## 2.1 사전 요구

- Python 3.11+
- (ASR 실사용) Linux + CUDA GPU, ffmpeg, Hugging Face 토큰
- (LLM 분석) OpenAI API Key — 없으면 LangGraph가 규칙 기반 노드로 동작

## 2.2 기본 설치

```powershell
cd C:\Users\Administrator\ryuzy-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## 2.3 ASR 실사용 추가 설치 (Linux+CUDA 권장)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyannote.audio qwen-asr vllm soundfile scipy
```

`.env` 설정:

```env
HF_TOKEN=hf_xxx          # pyannote 모델 접근 + 라이선스 동의 필요
ASR_MODEL_PATH=Qwen/Qwen3-ASR-1.7B
ALIGNER_MODEL_PATH=Qwen/Qwen3-ForcedAligner-0.6B
DIARIZE_MODEL_PATH=pyannote/speaker-diarization-community-1
OPENAI_API_KEY=sk-xxx    # LangGraph LLM 노드
ASR_WARMUP=true          # 기동 시 모델 로드(선택)
```

> Windows에서는 vLLM/Qwen3 ASR 지원이 제한적일 수 있습니다.  
> GPU가 없으면 Gradio에서 **수동 회의록** 또는 **텍스트 힌트**로 LangGraph/MCP 실습이 가능합니다.

## 2.4 실행

```powershell
# Terminal 1
python -m backend.app.main

# Terminal 2
python -m asr.app.main

# Terminal 3
python frontend/app.py
```

| 서비스 | URL |
|--------|-----|
| Backend | http://127.0.0.1:8000/docs |
| ASR | http://127.0.0.1:8001/health |
| Gradio | http://127.0.0.1:7860 |

한 번에 실행하려면 프로젝트 루트에서 아래를 실행합니다. Backend, ASR, Gradio를 같이 켜고, `Ctrl+C` 한 번으로 같이 멈춥니다.

```powershell
python scripts/run_all.py
```

샘플 시드: `python scripts/seed_sample.py`

## 2.5 ASR 예열

```powershell
curl -X POST http://127.0.0.1:8001/warmup
```

## 2.6 MCP

`mcp_config.example.json`을 Cursor MCP 설정에 복사하고 `cwd`를 프로젝트 경로로 수정합니다. Backend가 떠 있어야 `analyze_meeting`, `ask_meeting`이 동작합니다.

## 2.6.1 챗봇 확인

1. Gradio **4. 회의 챗봇**에서 예: `API 연동은 언제까지야?`
2. 또는 API:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"question\":\"API 연동 일정\",\"session_id\":\"class\",\"use_llm\":false}"
```

OpenAI 키가 없으면 회의록에서 키워드가 들어 있는 줄을 답으로 돌려줍니다.

## 2.7 문제 해결

| 증상 | 조치 |
|------|------|
| `Qwen3 ASR 로드 실패` | CUDA/vLLM/qwen-asr, `ASR_MODEL_PATH` 확인 |
| `pyannote 로드 실패` | `HF_TOKEN`, 모델 페이지 라이선스 동의 |
| LLM 요약 실패 | 키 확인 또는 Gradio에서 LLM 체크 해제(규칙 폴백) |
| `ModuleNotFoundError: shared` | 프로젝트 루트에서 실행 |
