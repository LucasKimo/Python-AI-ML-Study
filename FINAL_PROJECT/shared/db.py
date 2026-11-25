"""회의·목소리 지문·챗봇 대화를 파일 하나(SQLite)에 저장합니다.

이론 — 왜 SQLite인가
---------------------
서버 프로세스마다 따로 띄운 MySQL이 없어도, data/ryuzy.db 파일 하나로
회의 CRUD가 됩니다. 수업·로컬 실행에서 설치 대상을 줄이기 위한 선택입니다.

표는 세 개입니다.
  meetings        회의 메타데이터, 발화(transcript), 요약, 결정, 액션
  voice_profiles  참가자 이름과 목소리 임베딩(실수 리스트)
  chat_messages   챗봇 세션의 질문·답·근거 회의

transcript, decisions, action_items, embedding은 표 칸을 늘리지 않고
JSON 문자열로 넣습니다. 읽을 때 json.loads, 쓸 때 json.dumps입니다.
한 회의의 발화 개수가 매번 달라서, 행마다 컬럼을 고정하기 어렵기 때문입니다.

connect()는 성공하면 commit, 예외면 rollback 뒤 연결을 닫습니다.
row_factory = sqlite3.Row 라서 컬럼 이름으로 값을 읽습니다.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from shared.config import DATABASE_PATH, VOICE_DIR

KST = timezone(timedelta(hours=9))


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                participants TEXT NOT NULL DEFAULT '[]',
                agenda TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                transcript TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '',
                decisions TEXT NOT NULL DEFAULT '[]',
                action_items TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                prompt TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(meeting_id, speaker)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                meeting_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
            """
        )


def _row_to_meeting(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "participants": json.loads(row["participants"] or "[]"),
        "agenda": row["agenda"] or "",
        "status": row["status"],
        "transcript": json.loads(row["transcript"] or "[]"),
        "summary": row["summary"] or "",
        "decisions": json.loads(row["decisions"] or "[]"),
        "action_items": json.loads(row["action_items"] or "[]"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_meeting(
    title: str,
    participants: list[str] | None = None,
    agenda: str = "",
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    ts = now_kst()
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO meetings (title, participants, agenda, status, created_at, updated_at)
            VALUES (?, ?, ?, 'draft', ?, ?)
            """,
            (title, json.dumps(participants or [], ensure_ascii=False), agenda, ts, ts),
        )
        meeting_id = int(cur.lastrowid)
    return get_meeting(meeting_id, db_path=db_path)  # type: ignore[return-value]


def list_meetings(db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY id DESC"
        ).fetchall()
    return [_row_to_meeting(r) for r in rows]


def get_meeting(meeting_id: int, db_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
    return _row_to_meeting(row) if row else None


def update_meeting(meeting_id: int, **fields: Any) -> Optional[dict[str, Any]]:
    allowed = {
        "title",
        "participants",
        "agenda",
        "status",
        "transcript",
        "summary",
        "decisions",
        "action_items",
    }
    payload: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in allowed or value is None:
            continue
        if key in {"participants", "transcript", "decisions", "action_items"}:
            payload[key] = json.dumps(value, ensure_ascii=False)
        else:
            payload[key] = value
    if not payload:
        return get_meeting(meeting_id)

    payload["updated_at"] = now_kst()
    assignments = ", ".join(f"{k} = ?" for k in payload)
    values = list(payload.values()) + [meeting_id]
    with connect() as conn:
        conn.execute(f"UPDATE meetings SET {assignments} WHERE id = ?", values)
    return get_meeting(meeting_id)


def delete_meeting(meeting_id: int) -> bool:
    """회의와 그 회의에 딸린 목소리 등록·챗봇 대화·녹음 파일을 함께 지웁니다.

    세 표의 DELETE는 한 트랜잭션이라, 중간에 실패하면 모두 되돌아갑니다.
    녹음 폴더는 DB 삭제가 끝난 뒤에 지웁니다.

    AUTOINCREMENT는 지운 번호를 다시 쓰지 않으려고 sqlite_sequence 표에
    지금까지의 최대 번호를 기억합니다. 회의가 하나도 남지 않으면 그 기록을 지워
    다음 회의가 1번부터 시작하게 합니다. 이때 '전체 회의' 범위의 챗봇 대화도
    지웁니다. 남겨 두면 근거로 적힌 옛 회의 번호가 새 회의를 가리키게 되기 때문입니다.
    """
    with connect() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        if cur.rowcount == 0:
            return False
        conn.execute("DELETE FROM voice_profiles WHERE meeting_id = ?", (meeting_id,))
        conn.execute("DELETE FROM chat_messages WHERE meeting_id = ?", (meeting_id,))
        if conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0:
            conn.execute("DELETE FROM voice_profiles")
            conn.execute("DELETE FROM chat_messages")
            conn.execute(
                "DELETE FROM sqlite_sequence "
                "WHERE name IN ('meetings', 'voice_profiles', 'chat_messages')"
            )
    shutil.rmtree(Path(VOICE_DIR) / str(meeting_id), ignore_errors=True)
    return True


def search_meetings(query: str) -> list[dict[str, Any]]:
    q = f"%{query.strip()}%"
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM meetings
            WHERE title LIKE ? OR agenda LIKE ? OR summary LIKE ? OR transcript LIKE ?
            ORDER BY id DESC
            """,
            (q, q, q, q),
        ).fetchall()
    return [_row_to_meeting(r) for r in rows]


def upsert_voice_profile(
    meeting_id: int,
    speaker: str,
    prompt: str,
    audio_path: str,
    embedding: list[float],
) -> None:
    ts = now_kst()
    payload = json.dumps(embedding)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO voice_profiles
                (meeting_id, speaker, prompt, audio_path, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id, speaker) DO UPDATE SET
                prompt = excluded.prompt,
                audio_path = excluded.audio_path,
                embedding = excluded.embedding,
                created_at = excluded.created_at
            """,
            (meeting_id, speaker, prompt, audio_path, payload, ts),
        )


def list_voice_profiles(meeting_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT speaker, prompt, audio_path, created_at
            FROM voice_profiles
            WHERE meeting_id = ?
            ORDER BY speaker
            """,
            (meeting_id,),
        ).fetchall()
    return [
        {
            "speaker": row["speaker"],
            "prompt": row["prompt"],
            "audio_path": row["audio_path"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def list_voice_embeddings(meeting_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT speaker, embedding
            FROM voice_profiles
            WHERE meeting_id = ?
            """,
            (meeting_id,),
        ).fetchall()
    return [
        {"speaker": row["speaker"], "embedding": json.loads(row["embedding"])}
        for row in rows
    ]


def voice_registration(meeting_id: int) -> dict[str, Any] | None:
    meeting = get_meeting(meeting_id)
    if not meeting:
        return None
    profiles = list_voice_profiles(meeting_id)
    registered = [item["speaker"] for item in profiles]
    participants = list(meeting.get("participants") or [])
    ready = bool(participants) and all(name in registered for name in participants)
    return {
        "meeting_id": meeting_id,
        "participants": participants,
        "registered": registered,
        "ready": ready,
    }


def append_utterance(meeting_id: int, speaker: str, text: str) -> dict[str, Any] | None:
    meeting = get_meeting(meeting_id)
    if not meeting:
        return None
    cleaned = text.strip()
    transcript = list(meeting.get("transcript") or [])
    if transcript:
        last = transcript[-1]
        if last.get("speaker") == speaker and last.get("text") == cleaned:
            return meeting
    transcript.append(
        {
            "seq": len(transcript) + 1,
            "speaker": speaker,
            "text": cleaned,
            "timestamp": now_kst(),
        }
    )
    return update_meeting(meeting_id, transcript=transcript, status="live")


def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    meeting_id: Optional[int] = None,
    sources: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages (session_id, meeting_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                meeting_id,
                role,
                content,
                json.dumps(sources or [], ensure_ascii=False),
                ts,
            ),
        )
        row_id = int(cur.lastrowid)
    return {
        "id": row_id,
        "session_id": session_id,
        "meeting_id": meeting_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": ts,
    }


def list_chat_messages(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    items = []
    for row in reversed(rows):
        items.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "meeting_id": row["meeting_id"],
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources"] or "[]"),
                "created_at": row["created_at"],
            }
        )
    return items
