"""Backend FastAPI. 화면과 MCP가 회의 데이터를 주고받는 HTTP API입니다.

이론 — 이 프로세스가 허브인 이유
--------------------------------
Gradio와 MCP는 계산을 나눠 갖지 않고, 둘 다 이 서버의 URL을 호출합니다.
그래서 회의 저장, LangGraph 분석, 챗봇, 슬랙 전송의 입구가 여기에 모입니다.

FastAPI는 함수 인자의 타입(Pydantic 모델)을 보고 JSON 본문을 검사하고,
틀리면 422를 돌려줍니다. /docs 에 그 스키마가 자동으로 열립니다.
CORS는 Gradio(다른 포트)의 브라우저 요청을 받기 위한 허용 헤더입니다.

경로와 담당:
  POST /api/v1/meetings/analyze      LangGraph 요약·결정·액션
  POST /api/v1/chat                  검색 후 OpenAI 답변
  POST /api/v1/meetings/{id}/slack   슬랙 Incoming Webhook
  POST /api/v1/meetings/{id}/voices  목소리 임베딩 등록
  POST /api/v1/meetings/{id}/live    실시간 발화 (Whisper + 화자 매칭)
  POST /api/v1/meetings/{id}/finish  회의 종료 후 분석 그래프 실행

파일 전사(pyannote + Qwen3)는 ASR 프로세스(:8001)가 맡습니다.
Gradio가 그 결과를 받아 이 서버의 회의록 저장 API로 SQLite에 넣습니다.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr.app.services.audio_utils import load_audio_bytes_16k  # noqa: E402
from backend.app import voice_engine  # noqa: E402
from backend.app.graph.chat import run_chat  # noqa: E402
from backend.app.graph.workflow import run_meeting_analysis  # noqa: E402
from backend.app.slack_export import post_meeting as post_meeting_slack  # noqa: E402
from shared import config, db  # noqa: E402
from shared.schemas import (  # noqa: E402
    AnalyzeRequest,
    ChatRequest,
    HealthResponse,
    MeetingCreate,
    MeetingUpdate,
    TranscriptPayload,
)

KST = timezone(timedelta(hours=9))

app = FastAPI(
    title="류지 프로젝트 Backend",
    description="FastAPI + LangGraph meeting assistant API",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="backend",
        time=datetime.now(KST),
        details={
            "engine": "langgraph",
            "chat": "retrieve-answer",
            "openai_configured": bool(config.OPENAI_API_KEY),
            "model": config.OPENAI_MODEL,
        },
    )


@app.post("/api/v1/meetings")
def create_meeting(payload: MeetingCreate):
    return db.create_meeting(
        title=payload.title,
        participants=payload.participants,
        agenda=payload.agenda,
    )


@app.get("/api/v1/meetings")
def list_meetings():
    return db.list_meetings()


@app.get("/api/v1/meetings/search/{query}")
def search(query: str):
    return db.search_meetings(query)


def _openai_detail(prefix: str, exc: Exception) -> str:
    text = str(exc).lower()
    if any(
        token in text
        for token in ("401", "invalidated", "invalid_api_key", "expired", "incorrect api key")
    ):
        return (
            "OpenAI API 키가 거부되었습니다. "
            ".env의 OPENAI_API_KEY를 새로 발급한 키로 바꾼 뒤 서버를 다시 시작하세요."
        )
    return f"{prefix}: {exc}"


@app.post("/api/v1/meetings/analyze")
def analyze(payload: AnalyzeRequest):
    """저장된 회의록으로 LangGraph를 끝까지 돌리고, 요약·결정·액션을 DB에 덮어씁니다."""
    meeting = db.get_meeting(payload.meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    try:
        result = run_meeting_analysis(
            meeting_id=payload.meeting_id,
            title=meeting.get("title", ""),
            transcript=meeting.get("transcript") or [],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_openai_detail("OpenAI 분석 실패", exc)) from exc
    updated = db.update_meeting(
        payload.meeting_id,
        summary=result["summary"],
        decisions=result["decisions"],
        action_items=result["action_items"],
        status="analyzed",
    )
    updated["analysis_meta"] = {
        "engine": result["engine"],
        "accuracy_score": result["accuracy_score"],
        "errors": result["errors"],
    }
    return updated


@app.get("/api/v1/meetings/{meeting_id}")
def get_meeting(meeting_id: int):
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.patch("/api/v1/meetings/{meeting_id}")
def patch_meeting(meeting_id: int, payload: MeetingUpdate):
    if not db.get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return db.update_meeting(
        meeting_id,
        title=payload.title,
        participants=payload.participants,
        agenda=payload.agenda,
        status=payload.status,
    )


@app.delete("/api/v1/meetings/{meeting_id}")
def remove_meeting(meeting_id: int):
    if not db.delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"deleted": True, "id": meeting_id}


@app.post("/api/v1/meetings/{meeting_id}/slack")
def send_meeting_slack(meeting_id: int):
    """슬랙 Incoming Webhook으로 회의 한 건을 보냅니다.

    주소는 환경변수 SLACK_WEBHOOK_URL입니다. 만드는 법은
    mcp_server/server.py 맨 위 주석과 Gradio 6번 탭에 적혀 있습니다.
    """
    if not db.get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        return post_meeting_slack(meeting_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/v1/chat")
def chat(payload: ChatRequest):
    if payload.meeting_id is not None and not db.get_meeting(payload.meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        return run_chat(
            question=payload.question.strip(),
            meeting_id=payload.meeting_id,
            session_id=payload.session_id or "default",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_openai_detail("OpenAI 챗봇 실패", exc)) from exc


@app.get("/api/v1/chat/{session_id}")
def chat_history(session_id: str):
    return db.list_chat_messages(session_id)


@app.get("/api/v1/meetings/{meeting_id}/voices")
def meeting_voices(meeting_id: int):
    view = voice_engine.registration_view(meeting_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return view


@app.post("/api/v1/meetings/{meeting_id}/voices")
async def enroll_voice(
    meeting_id: int,
    speaker: str = Form(...),
    file: UploadFile = File(...),
):
    data = await file.read()
    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    try:
        wav = load_audio_bytes_16k(data, suffix=suffix)
        return voice_engine.enroll(meeting_id, speaker, wav)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"목소리 등록 실패: {exc}") from exc


@app.post("/api/v1/meetings/{meeting_id}/live")
async def live_utterance(meeting_id: int, file: UploadFile = File(...)):
    data = await file.read()
    suffix = Path(file.filename or "utterance.wav").suffix or ".wav"
    try:
        wav = load_audio_bytes_16k(data, suffix=suffix)
        return voice_engine.transcribe_live(meeting_id, wav)
    except ValueError as exc:
        status = 409 if "등록" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"실시간 인식 실패: {exc}") from exc


@app.post("/api/v1/meetings/{meeting_id}/finish")
def finish_meeting(meeting_id: int):
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    view = voice_engine.registration_view(meeting_id)
    if not view or not view["ready"]:
        raise HTTPException(status_code=409, detail="모든 참가자 목소리를 먼저 등록하세요.")
    if not (meeting.get("transcript") or []):
        raise HTTPException(
            status_code=400,
            detail="인식된 발화가 없습니다. 회의를 진행한 뒤 종료하세요.",
        )
    return analyze(AnalyzeRequest(meeting_id=meeting_id))


@app.put("/api/v1/meetings/{meeting_id}/transcript")
def save_transcript(meeting_id: int, payload: TranscriptPayload):
    if meeting_id != payload.meeting_id:
        raise HTTPException(status_code=400, detail="meeting_id mismatch")
    if not db.get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    utterances = [u.model_dump() for u in payload.utterances]
    return db.update_meeting(
        meeting_id,
        transcript=utterances,
        status="transcribed",
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
