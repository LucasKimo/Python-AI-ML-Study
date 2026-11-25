"""Pydantic schemas shared across Backend / ASR / MCP / Gradio."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MeetingCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    participants: list[str] = Field(default_factory=list)
    agenda: str = ""


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    participants: Optional[list[str]] = None
    agenda: Optional[str] = None
    status: Optional[str] = None


class Utterance(BaseModel):
    seq: int
    speaker: str
    text: str
    timestamp: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


class TranscriptPayload(BaseModel):
    meeting_id: int
    utterances: list[Utterance]
    source: str = "manual"  # manual | asr


class AnalyzeRequest(BaseModel):
    meeting_id: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    meeting_id: Optional[int] = None
    session_id: str = "default"


class HealthResponse(BaseModel):
    status: str
    service: str
    time: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class TranscribeResponse(BaseModel):
    utterances: list[Utterance]
    mode: str
    message: str = ""
    segment_count: int = 0
