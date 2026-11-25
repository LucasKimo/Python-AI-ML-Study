# 4. 아키텍처

## 4.1 구성도

```
┌────────────┐  HTTP  ┌─────────────────────┐
│  Gradio    │───────►│ Backend :8000       │
│  :7860     │        │ FastAPI + LangGraph │
└─────┬──────┘        └──────────┬──────────┘
      │                          │
      │ /transcribe              │ SQLite
      ▼                          ▼
┌─────────────────────┐   ┌────────────┐
│ ASR :8001           │   │ ryuzy.db   │
│ Qwen3 + pyannote    │   └────────────┘
└─────────────────────┘
            ▲
┌───────────┴───────────┐
│ MCP Server (stdio)    │
│ → Backend REST Tools  │
└───────────────────────┘
```

## 4.2 ASR 파이프라인

```
audio file
  → 16kHz mono
  → pyannote Pipeline (화자 구간)
  → 인접 동일 화자 merge / 짧은 구간 필터
  → Qwen3ASRModel.transcribe(chunks)
  → utterances[{seq, speaker, text, start, end}]
```

보조 경로: `text_hint` Form 필드로 모델 없이 발화 파싱.

## 4.3 LangGraph

```
START → summarize → decisions → actions → quality → END
```

| Node | 역할 |
|------|------|
| summarize | 전체 요약 |
| decisions | 결정사항 추출 |
| actions | 액션 아이템 |
| quality | accuracy_score |

`use_llm=true` 이고 `OPENAI_API_KEY`가 있으면 각 노드가 LLM JSON을 사용하고, 실패/미설정 시 규칙 기반으로 폴백합니다.

### 챗봇 그래프

```
START → retrieve → answer → END
```

| Node | 역할 |
|------|------|
| retrieve | 선택 회의 또는 전체 회의에서 질문과 겹치는 회의록·요약·결정·액션을 모음 |
| answer | OpenAI가 있으면 자료 안에서만 답변, 없으면 키워드가 포함된 줄을 답변 |

대화는 SQLite `chat_messages`에 세션별로 저장합니다. 원본 Knowledge 챗봇의 RAG/Chroma 대신 회의 텍스트 검색을 사용합니다.

## 4.4 API

### Backend
- `POST /api/v1/meetings/analyze` → LangGraph 분석 후 DB 저장
- `POST /api/v1/chat` → 질문, 선택 회의, 세션 ID
- `GET /api/v1/chat/{session_id}` → 대화 기록

### ASR
- `POST /transcribe` — multipart audio 또는 text_hint
- `POST /warmup` — 모델 예열
- `GET /health` — 로드 상태 details
