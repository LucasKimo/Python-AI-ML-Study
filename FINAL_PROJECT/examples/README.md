# 예제 모음 — 류지 프로젝트 기술 미니 실습

본 폴더는 본 프로젝트에서 쓰는 기술을 **하나씩 아주 작게** 연습하기 위한 공간입니다.  
본편(`frontend/`, `backend/`, `asr/`, `mcp_server/`)을 보기 전에 여기서 감을 잡는 것을 권장합니다.

| 폴더 | 기술 | 난이도 | GPU 필요 |
|------|------|--------|----------|
| [01-gradio](01-gradio/) | Gradio UI | ★☆☆ | 아니오 |
| [02-fastapi](02-fastapi/) | FastAPI REST API | ★☆☆ | 아니오 |
| [03-langgraph](03-langgraph/) | LangGraph 파이프라인 | ★★☆ | 아니오 |
| [04-qwen3-asr](04-qwen3-asr/) | Qwen3 음성 인식 (PyTorch CUDA) | ★★★ | **NVIDIA GPU** |
| [05-pyannote](05-pyannote/) | pyannote 화자분리·목소리 등록 | ★★★ | 권장 |
| [06-mcp](06-mcp/) | 슬랙 Incoming Webhook으로 문장 보내기 | ★☆☆ | 아니오 |
| [07-sqlite](07-sqlite/) | SQLite 저장 | ★☆☆ | 아니오 |
| [08-chatbot](08-chatbot/) | 회의 챗봇 (검색 답변) | ★★☆ | 아니오 |

## 권장 학습 순서

```
01 Gradio → 02 FastAPI → 07 SQLite → 03 LangGraph → 08 Chatbot → 06 Slack
         ↘ (환경 되면) 05 pyannote → 04 Qwen3 ASR
```

## 공통 준비

프로젝트 루트의 가상환경을 그대로 써도 됩니다.

```powershell
cd C:\Users\Administrator\ryuzy-project
.\.venv\Scripts\Activate.ps1
```

각 폴더의 `README.md`에 **받아야 하는 모듈**과 **실행 명령**이 자세히 적혀 있습니다.
