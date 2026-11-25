# 3. 디렉터리 구조

```
FINAL_PROJECT/
├── README.md
├── requirements.txt
├── .env.example
├── mcp_config.example.json
│
├── docs/
│   ├── 01-planning.md
│   ├── 02-install-run.md
│   ├── 03-directory.md
│   ├── 04-architecture.md
│   └── 05-origin.md
│
├── frontend/
│   └── app.py                 # Gradio UI
│
├── backend/
│   └── app/
│       ├── main.py            # FastAPI REST
│       └── graph/             # LangGraph
│           ├── state.py       # MeetingState
│           ├── nodes.py       # summarize/decisions/actions/quality
│           ├── workflow.py    # 분석 StateGraph
│           └── chat.py        # 챗봇 retrieve → answer
│
├── asr/
│   └── app/
│       ├── main.py            # FastAPI /transcribe, /warmup
│       ├── core/
│       │   └── models.py      # Qwen3 ASR / Aligner / pyannote loaders
│       └── services/
│           ├── audio_utils.py # 16k mono decode
│           └── pipeline.py    # diarize → chunk ASR → utterances
│
├── mcp_server/
│   └── server.py              # FastMCP tools
│
├── shared/
│   ├── config.py
│   ├── schemas.py
│   └── db.py                  # SQLite
│
├── data/
│   ├── samples/
│   └── uploads/
│
└── scripts/
    ├── run_all.py
    └── seed_sample.py
```

## 원본 매핑

| 수업판 | 원본 |
|--------|------|
| `frontend/app.py` | 이전 React 화면 |
| `backend/app/graph/*` | 이전 LangGraph 백엔드 |
| `asr/app/core/models.py` | 이전 ASR 서버 `core/models.py` |
| `asr/app/services/pipeline.py` | offline diarization + chunked ASR |
| `mcp_server/` | (신규) |
