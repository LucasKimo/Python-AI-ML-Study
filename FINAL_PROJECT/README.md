# 류지 프로젝트

여러 사람의 목소리를 모아 화자를 나누고, 회의록과 할 일을 만드는 회의 어시스턴트입니다.

| 구성요소 | 기술 |
|----------|------|
| Frontend | **Gradio** |
| Backend | **FastAPI + LangGraph** (분석 + 회의 챗봇) |
| ASR | **Qwen3 ASR + pyannote** diarization |
| Tooling | **MCP** server |
| Storage | SQLite |

## 빠른 시작

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# ASR 실사용 시 (Linux+CUDA 권장)
# pip install qwen-asr vllm torch torchaudio pyannote.audio

copy .env.example .env
# HF_TOKEN, OPENAI_API_KEY, 모델 경로를 .env에 설정

# 한 번에 실행 (Backend :8000, ASR :8001, Gradio :7860)
python scripts/run_all.py
# 종료는 Ctrl+C. 세 프로세스를 함께 멈춥니다.

# 터미널을 나눠 실행할 때
python -m backend.app.main     # :8000
python -m asr.app.main         # :8001
python frontend/app.py         # :7860
```

문서: [`docs/`](docs/)

기술별 미니 예제(설치·설명·실행 포함): [`examples/`](examples/)

## 파이프라인

```
오디오 → pyannote 화자분리 → Qwen3 세그먼트 ASR → SQLite
회의록 → LangGraph 분석 → 요약/결정/액션 → MCP Tools
질문 → LangGraph 챗봇(retrieve → answer) → 회의록 근거 답변
```
