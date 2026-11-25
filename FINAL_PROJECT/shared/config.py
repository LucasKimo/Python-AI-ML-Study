"""Shared configuration for 류지 프로젝트."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# ── Service hosts ──────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = _int("BACKEND_PORT", 8000)
ASR_HOST = os.getenv("ASR_HOST", "127.0.0.1")
ASR_PORT = _int("ASR_PORT", 8001)
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = _int("GRADIO_PORT", 7860)

BACKEND_URL = os.getenv("BACKEND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
ASR_URL = os.getenv("ASR_URL", f"http://{ASR_HOST}:{ASR_PORT}")

# ── Storage ────────────────────────────────────────────────────
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "ryuzy.db")))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = ROOT_DIR / DATABASE_PATH

UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Qwen3 ASR + pyannote ───────────────────────────────────────
# ASR_MODEL_PATH: 글자로 바꾸는 모델. DIARIZE_MODEL_PATH: 누가 언제 말했는지.
# HF_TOKEN: pyannote 모델 카드 동의 계정의 Hugging Face 토큰.
# MIN_SEGMENT_SEC보다 짧은 화자 구간은 버리고, MERGE_GAP_SEC 안의 같은 화자는 한 구간으로 붙입니다.
ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
DIARIZE_MODEL_PATH = os.getenv(
    "DIARIZE_MODEL_PATH", "pyannote/speaker-diarization-community-1"
)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GPU_MEMORY_UTILIZATION = _float("GPU_MEMORY_UTILIZATION", 0.55)
MAX_MODEL_LEN = _int("MAX_MODEL_LEN", 4096)
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "Korean")
MIN_SEGMENT_SEC = _float("MIN_SEGMENT_SEC", 0.8)
MERGE_GAP_SEC = _float("MERGE_GAP_SEC", 0.5)

# ── LangGraph / LLM / Slack ────────────────────────────────────
# OPENAI_*: 요약·결정·액션·챗봇 답변. 그래프가 매 노드에서 이 키로 Chat Completions를 부릅니다.
# SLACK_WEBHOOK_URL: Incoming Webhook 주소 (https://hooks.slack.com/services/...).
#   발급: api.slack.com/apps → Create New App → Incoming Webhooks On
#   → Add New Webhook to Workspace → 채널 선택 → URL을 .env에만 붙여 넣고 Backend 재시작.
#   채팅·깃에 붙이지 않습니다. MCP 도구가 아니라 Gradio/Backend 슬랙 API가 이 값을 읽습니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

# ── 실시간 회의: 목소리 지문 + Whisper ─────────────────────────
# 파일 업로드용 pyannote/Qwen3와 별도입니다.
# SPEAKER_EMBEDDING_MODEL이 목소리를 벡터로 만들고, 코사인 유사도가
# SPEAKER_MATCH_THRESHOLD 이상인 등록 이름을 붙입니다. 글자는 WHISPER_MODEL이 만듭니다.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
SPEAKER_EMBEDDING_MODEL = os.getenv(
    "SPEAKER_EMBEDDING_MODEL", "hbredin/wespeaker-voxceleb-resnet34-LM"
)
SPEAKER_MATCH_THRESHOLD = _float("SPEAKER_MATCH_THRESHOLD", 0.5)
VOICE_DIR = ROOT_DIR / "data" / "voices"
VOICE_DIR.mkdir(parents=True, exist_ok=True)
