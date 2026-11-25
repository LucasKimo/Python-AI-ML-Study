"""Gradio 화면. 브라우저에서 회의를 만들고, 백엔드·ASR·슬랙을 버튼으로 부릅니다.

이론 — Gradio가 이 프로젝트에서 하는 일
--------------------------------------
Gradio는 파이썬 함수를 웹 폼으로 감싸는 UI 라이브러리입니다.
React 화면을 따로 두지 않고, gr.Blocks 안에 탭·버튼·마이크를 선언합니다.
버튼의 .click(함수, inputs, outputs)이 "이 칸의 값을 함수에 넣고, 결과를 저 칸에 그린다"는
이벤트 연결입니다.

이 파일은 회의 로직을 직접 계산하지 않습니다.
httpx로 Backend(:8000)에 JSON을 보내고, 받은 JSON을 마크다운으로 보여 줍니다.
  - 목소리 등록 / 실시간 회의 → voice API (임베딩 + Whisper)
  - 오디오 파일 전사 → Gradio가 ASR(:8001, pyannote + Qwen3)을 직접 호출한 뒤
    발화를 Backend 회의록 저장 API로 넣습니다
  - 분석 버튼 → LangGraph
  - 챗봇 → retrieve → answer 그래프
  - 슬랙 버튼 → Incoming Webhook (설정 방법은 6번 탭과 slack_export.py)
  - 6번 탭은 실행 버튼이 없는 설명입니다. MCP는 Cursor가 stdio로 붙습니다.

실시간 마이크는 streaming=True 이고, stream 이벤트가 live_audio.UtteranceBuffer에
조각을 쌓았다가 침묵이 오면 한 발화로 Backend에 넘깁니다.
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr.app.services.audio_utils import resample_to_16k  # noqa: E402
from frontend.live_audio import (  # noqa: E402
    SAMPLE_RATE,
    UtteranceBuffer,
    take_new_samples,
    to_mono_float,
)
from shared import config  # noqa: E402
from shared.prompts import ENROLL_PROMPT  # noqa: E402

_BUFFERS: dict[int, UtteranceBuffer] = {}
_BUFFER_GUARD = threading.Lock()

SAMPLE_TRANSCRIPT = """김매니저: 오늘 스프린트 목표를 먼저 공유하겠습니다.
이개발: 백엔드 API 연동은 수요일까지 완료하기로 했습니다.
박기획: UI 초안은 금요일까지 제출하는 것으로 결정합니다.
김매니저: 액션: 이개발이 테스트 케이스를 작성하기로 합니다.
박기획: 다음 주 월요일에 중간 점검 회의를 잡겠습니다."""


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def api_get(path: str) -> Any:
    with _client() as client:
        r = client.get(f"{config.BACKEND_URL}{path}")
        r.raise_for_status()
        return r.json()


def api_post(path: str, payload: dict[str, Any], timeout: float = 120.0) -> Any:
    with _client(timeout) as client:
        r = client.post(f"{config.BACKEND_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def api_put(path: str, payload: dict[str, Any]) -> Any:
    with _client() as client:
        r = client.put(f"{config.BACKEND_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def api_delete(path: str) -> Any:
    with _client() as client:
        r = client.delete(f"{config.BACKEND_URL}{path}")
        r.raise_for_status()
        return r.json()


def format_meeting(m: dict[str, Any]) -> str:
    lines = [
        f"### #{m['id']} {m['title']}",
        f"- 상태: `{m.get('status')}`",
        f"- 참가자: {', '.join(m.get('participants') or []) or '(없음)'}",
        f"- 안건: {m.get('agenda') or '(없음)'}",
        f"- 생성: {m.get('created_at')} / 수정: {m.get('updated_at')}",
    ]
    meta = m.get("analysis_meta") or {}
    if meta:
        lines.append(
            f"- 분석엔진: `{meta.get('engine')}` / "
            f"accuracy: `{meta.get('accuracy_score')}`"
        )
    transcript = m.get("transcript") or []
    if transcript:
        lines.append("\n**회의록**")
        for u in transcript:
            ts = u.get("timestamp") or ""
            prefix = f"[{ts}] " if ts else ""
            lines.append(
                f"- [{u.get('seq')}] {prefix}{u.get('speaker')}: {u.get('text')}"
            )
    if m.get("summary"):
        lines.append(f"\n**요약 (LangGraph)**\n{m['summary']}")
    if m.get("decisions"):
        lines.append("\n**결정사항**")
        for d in m["decisions"]:
            lines.append(f"- {d}")
    if m.get("action_items"):
        lines.append("\n**액션 아이템**")
        for a in m["action_items"]:
            lines.append(
                f"- [{a.get('status', 'open')}] "
                f"{a.get('assignee', '?')}: {a.get('task', '')}"
            )
    return "\n".join(lines)


def refresh_meeting_choices() -> gr.Dropdown:
    try:
        meetings = api_get("/api/v1/meetings")
        choices = [f"{m['id']} | {m['title']}" for m in meetings]
    except Exception:
        choices = []
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def load_home():
    dropdown = refresh_meeting_choices()
    label = getattr(dropdown, "value", None)
    speaker, status, live = sync_voice_panel(label if isinstance(label, str) else None)
    return dropdown, speaker, status, live


def ask_delete_meeting(label: str | None):
    """삭제 버튼: 바로 지우지 않고 확인 줄을 보여 줍니다."""
    if parse_meeting_id(label) is None:
        return gr.Row(visible=False), "삭제할 회의를 먼저 선택하세요."
    return (
        gr.Row(visible=True),
        f"**{label}** 회의를 지울까요? 회의록, 목소리 등록, 녹음 파일, "
        "이 회의의 챗봇 대화가 모두 사라지고 되돌릴 수 없습니다.",
    )


def cancel_delete_meeting():
    return gr.Row(visible=False), "삭제를 취소했습니다."


def confirm_delete_meeting(label: str | None):
    hidden = gr.Row(visible=False)
    mid = parse_meeting_id(label)
    if mid is None:
        return (hidden, "삭제할 회의를 먼저 선택하세요.",
                gr.skip(), gr.skip(), gr.skip(), gr.skip())
    try:
        api_delete(f"/api/v1/meetings/{mid}")
    except Exception as exc:  # noqa: BLE001
        return (hidden, f"삭제하지 못했습니다: {exc}",
                gr.skip(), gr.skip(), gr.skip(), gr.skip())
    with _BUFFER_GUARD:
        _BUFFERS.pop(mid, None)
    dropdown, speaker, status, live = load_home()
    return hidden, f"**{label}** 회의를 삭제했습니다.", dropdown, speaker, status, live


def parse_meeting_id(label: str | None) -> int | None:
    if not label:
        return None
    return int(str(label).split("|", 1)[0].strip())


def _buffer(meeting_id: int) -> UtteranceBuffer:
    with _BUFFER_GUARD:
        if meeting_id not in _BUFFERS:
            _BUFFERS[meeting_id] = UtteranceBuffer()
        return _BUFFERS[meeting_id]


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail", response.text)
    except Exception:  # noqa: BLE001
        detail = response.text
    return str(detail)[:500]


def format_voice_status(payload: dict[str, Any]) -> str:
    registered = set(payload.get("registered") or [])
    participants = list(payload.get("participants") or [])
    lines = ["### 등록 체크"]
    if not participants:
        lines.append("참가자가 없습니다. 1단계에서 이름을 쉼표로 넣어 회의를 다시 만드세요.")
        return "\n".join(lines)
    for name in participants:
        state = "등록됨" if name in registered else "아직"
        lines.append(f"- **{name}**: {state}")
    missing = [name for name in participants if name not in registered]
    if missing:
        lines.append(
            f"\n**다음 할 일:** `{missing[0]}`을 고른 뒤 "
            "Record → 문장 읽기 → Stop → **이 목소리 저장**."
        )
    else:
        lines.append(
            "\n**다음 할 일:** 3단계로 내려가 회의 마이크의 Record를 누르세요."
        )
    return "\n".join(lines)


def format_live(transcript: list[dict[str, Any]] | None) -> str:
    if not transcript:
        return "_아직 인식된 발화가 없습니다._"
    lines = []
    for item in transcript:
        stamp = item.get("timestamp") or ""
        when = f"`{stamp}` " if stamp else ""
        lines.append(f"{when}**{item.get('speaker', '?')}**: {item.get('text', '')}")
    return "\n\n".join(lines)


def sync_voice_panel(label: str | None):
    mid = parse_meeting_id(label)
    if mid is None:
        return (
            gr.Dropdown(choices=[], value=None),
            "1단계에서 회의를 만들면, 누구부터 녹음하면 되는지 여기에 나옵니다.",
            "_실시간 회의 내용_",
        )
    try:
        meeting = api_get(f"/api/v1/meetings/{mid}")
        voices = api_get(f"/api/v1/meetings/{mid}/voices")
    except Exception as exc:  # noqa: BLE001
        return gr.Dropdown(choices=[], value=None), f"회의 정보를 불러오지 못했습니다: {exc}", ""
    people = list(voices.get("participants") or [])
    _buffer(mid).allow = bool(voices.get("ready"))
    return (
        gr.Dropdown(choices=people, value=people[0] if people else None),
        format_voice_status(voices),
        format_live(meeting.get("transcript") or []),
    )


def create_meeting(title: str, participants: str, agenda: str):
    if not title.strip():
        return "제목을 입력하세요.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    people = [p.strip() for p in participants.split(",") if p.strip()]
    if not people:
        return "참가자를 쉼표로 구분해 입력하세요.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    meeting = api_post(
        "/api/v1/meetings",
        {"title": title.strip(), "participants": people, "agenda": agenda},
    )
    label = f"{meeting['id']} | {meeting['title']}"
    speaker, status, live = sync_voice_panel(label)
    dropdown = refresh_meeting_choices()
    return format_meeting(meeting), dropdown, speaker, status, live


def save_voice(label: str, speaker: str | None, audio_path: str | None):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 먼저 만드세요.", gr.skip(), gr.skip()
    if not speaker:
        return "등록할 참가자를 선택하세요.", gr.skip(), gr.skip()
    if not audio_path:
        return (
            "아직 녹음이 없습니다. Record를 누르고 문장을 읽은 뒤 Stop을 누르고, 그다음 이 목소리 저장을 누르세요.",
            gr.skip(),
            gr.skip(),
        )
    try:
        with open(audio_path, "rb") as handle:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{config.BACKEND_URL}/api/v1/meetings/{mid}/voices",
                    data={"speaker": speaker},
                    files={"file": ("voice.wav", handle, "audio/wav")},
                )
        if response.status_code >= 400:
            return f"목소리 저장 실패: {_error_detail(response)}", gr.skip(), gr.skip()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return f"목소리 저장 실패: {exc}", gr.skip(), gr.skip()
    _buffer(mid).allow = bool(payload.get("ready"))
    try:
        preview = preview_recording(audio_path)
    except Exception as exc:  # noqa: BLE001
        return (
            format_voice_status(payload) + f"\n\n재생 파일을 만들지 못했습니다: {exc}",
            None,
            gr.skip(),
        )
    if preview is None:
        return (
            format_voice_status(payload) + "\n\n녹음 파일이 비어 있어 재생할 수 없습니다.",
            None,
            gr.skip(),
        )
    return format_voice_status(payload), None, preview


def preview_recording(audio):
    """Turn a microphone recording into audio the browser player can play."""
    if audio is None or audio == "":
        return None
    from asr.app.services.audio_utils import load_audio_16k, resample_to_16k

    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, data = audio
        wav = resample_to_16k(to_mono_float(data), int(sample_rate))
    else:
        wav = load_audio_16k(str(audio))
    if wav.size == 0:
        return None
    pcm = np.clip(wav, -1.0, 1.0)
    return (16000, (pcm * 32767.0).astype(np.int16))


def show_preview(audio):
    """Enable the listen player only after a recording file exists."""
    if not audio:
        return gr.skip(), gr.skip()
    try:
        playable = preview_recording(audio)
    except Exception as exc:  # noqa: BLE001
        return gr.skip(), f"재생 준비 실패: {exc}"
    if playable is None:
        return gr.skip(), "녹음이 비어 있습니다. Record로 다시 녹음해 주세요."
    return playable, "녹음되었습니다. **들어보기**의 재생 버튼을 누르세요."


def reset_enrollment_recorder():
    return None, None, "Stop을 누르면 여기에 재생 버튼이 켜집니다."


def _wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _post_live(meeting_id: int, samples: np.ndarray) -> dict[str, Any]:
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{config.BACKEND_URL}/api/v1/meetings/{meeting_id}/live",
            files={"file": ("utterance.wav", _wav_bytes(samples), "audio/wav")},
        )
    if response.status_code >= 400:
        raise RuntimeError(_error_detail(response))
    return response.json()


def _chunk_samples(audio: tuple[int, np.ndarray] | None, meeting_id: int) -> np.ndarray | None:
    if audio is None:
        return None
    sample_rate, data = audio
    mono = to_mono_float(data)
    buf = _buffer(meeting_id)
    fresh = take_new_samples(buf.previous_chunk, mono)
    buf.previous_chunk = mono
    if fresh.size == 0:
        return None
    return resample_to_16k(fresh, int(sample_rate))


def _apply_live_payload(payload: dict[str, Any]) -> tuple[str, str]:
    transcript = format_live(payload.get("transcript") or [])
    if payload.get("skipped"):
        return str(payload.get("reason") or "이번 구간은 건너뛰었습니다."), transcript
    speaker = payload.get("speaker") or "미확인"
    score = payload.get("score")
    score_text = f" (유사도 {score:.2f})" if isinstance(score, (int, float)) else ""
    return f"인식: {speaker}{score_text}", transcript


def on_record_start(label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 먼저 선택하세요."
    try:
        voices = api_get(f"/api/v1/meetings/{mid}/voices")
    except Exception as exc:  # noqa: BLE001
        return f"등록 상태를 확인하지 못했습니다: {exc}"
    buf = _buffer(mid)
    buf.allow = bool(voices.get("ready"))
    buf.reset_audio()
    if not buf.allow:
        return "아직 등록되지 않은 참가자가 있습니다. 모두 위 문장을 읽은 뒤 녹음하세요."
    return (
        "녹음 중입니다. 말이 잠시 끊기면 화자와 문장이 표시됩니다. "
        "처음 한 문장은 음성 모델을 불러오느라 더 걸릴 수 있습니다."
    )


def on_live_audio(audio, label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 먼저 선택하세요.", gr.skip()
    buf = _buffer(mid)
    if audio is None:
        return "회의 마이크에서 녹음을 시작하세요.", gr.skip()
    samples = _chunk_samples(audio, mid)
    if samples is None:
        return gr.skip(), gr.skip()
    utterance = buf.push(samples, SAMPLE_RATE)
    if utterance is None:
        if buf.heard_speech:
            return "듣는 중…", gr.skip()
        return gr.skip(), gr.skip()
    if not buf.allow:
        try:
            voices = api_get(f"/api/v1/meetings/{mid}/voices")
            buf.allow = bool(voices.get("ready"))
        except Exception as exc:  # noqa: BLE001
            return f"등록 상태를 확인하지 못했습니다: {exc}", gr.skip()
    if not buf.allow:
        return "모든 참가자 목소리를 등록한 뒤 회의를 시작하세요.", gr.skip()
    try:
        payload = _post_live(mid, utterance)
    except Exception as exc:  # noqa: BLE001
        return f"인식 오류: {exc}", gr.skip()
    return _apply_live_payload(payload)


def on_record_stop(label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 먼저 선택하세요.", gr.skip()
    utterance = _buffer(mid).flush()
    if utterance is None or not _buffer(mid).allow:
        return "녹음을 멈췄습니다.", gr.skip()
    try:
        payload = _post_live(mid, utterance)
    except Exception as exc:  # noqa: BLE001
        return f"인식 오류: {exc}", gr.skip()
    status, transcript = _apply_live_payload(payload)
    return f"녹음을 멈췄습니다. {status}", transcript


def finish_meeting(label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 먼저 선택하세요.", gr.skip()
    on_record_stop(label)
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(f"{config.BACKEND_URL}/api/v1/meetings/{mid}/finish")
        if response.status_code >= 400:
            return f"회의록 생성 실패: {_error_detail(response)}", gr.skip()
        meeting = response.json()
    except Exception as exc:  # noqa: BLE001
        return f"회의록 생성 실패: {exc}", gr.skip()
    return format_meeting(meeting), format_live(meeting.get("transcript") or [])


def load_meeting(label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 선택하세요."
    return format_meeting(api_get(f"/api/v1/meetings/{mid}"))


def save_manual_transcript(label: str, raw: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 선택하세요."
    utterances = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
        else:
            speaker, text = f"SPEAKER_{(i - 1) % 3:02d}", line
        utterances.append(
            {
                "seq": i,
                "speaker": speaker.strip(),
                "text": text.strip(),
                "timestamp": None,
            }
        )
    meeting = api_put(
        f"/api/v1/meetings/{mid}/transcript",
        {"meeting_id": mid, "utterances": utterances, "source": "manual"},
    )
    return format_meeting(meeting)


def run_asr(label: str, audio_path: str | None, text_hint: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 선택하세요."

    files = None
    data = {"text_hint": text_hint or ""}
    if audio_path:
        files = {"file": open(audio_path, "rb")}

    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(
                f"{config.ASR_URL}/transcribe", data=data, files=files
            )
            if resp.status_code >= 400:
                return f"ASR 오류 ({resp.status_code}): {resp.text}"
            asr_result = resp.json()
    finally:
        if files:
            files["file"].close()

    utterances = asr_result.get("utterances", [])
    meeting = api_put(
        f"/api/v1/meetings/{mid}/transcript",
        {"meeting_id": mid, "utterances": utterances, "source": "asr"},
    )
    note = asr_result.get("message", "")
    mode = asr_result.get("mode", "")
    segs = asr_result.get("segment_count", 0)
    return f"(ASR `{mode}`, segments={segs}) {note}\n\n" + format_meeting(meeting)


def analyze(label: str):
    mid = parse_meeting_id(label)
    if mid is None:
        return "회의를 선택하세요."
    try:
        meeting = api_post(
            "/api/v1/meetings/analyze",
            {"meeting_id": mid},
            timeout=180.0,
        )
    except Exception as exc:  # noqa: BLE001
        return f"OpenAI 분석 실패: {exc}"
    return format_meeting(meeting)


def _wants_slack(question: str) -> bool:
    lowered = question.lower()
    mentions_slack = "슬랙" in question or "slack" in lowered
    asks_to_send = any(word in question for word in ("보내", "전송", "공유", "올려"))
    return mentions_slack and asks_to_send


def send_meeting_slack(label: str) -> str:
    mid = parse_meeting_id(label)
    if mid is None:
        return "위에서 회의를 선택한 뒤 다시 보내세요."
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{config.BACKEND_URL}/api/v1/meetings/{mid}/slack")
        if response.status_code >= 400:
            return f"슬랙 전송 실패: {_error_detail(response)}"
        payload = response.json()
        title = payload.get("title") or "회의"
        return f"「{title}」 결과를 슬랙으로 보냈습니다."
    except Exception as exc:  # noqa: BLE001
        return f"슬랙 전송 실패: {exc}"


def send_slack_in_chat(label: str, history: list | None):
    history = list(history or [])
    history.append({"role": "user", "content": "이 회의 결과를 슬랙으로 보내줘"})
    history.append({"role": "assistant", "content": send_meeting_slack(label)})
    return history


def chat_reply(message: str, history: list | None, label: str, scope_selected: bool):
    history = list(history or [])
    question = (message or "").strip()
    if not question:
        return history, ""
    if _wants_slack(question):
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": send_meeting_slack(label)})
        return history, ""
    payload: dict[str, Any] = {
        "question": question,
        "session_id": "gradio",
    }
    if scope_selected:
        mid = parse_meeting_id(label)
        if mid is None:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "특정 회의만 보려면 위에서 회의를 선택하세요."})
            return history, ""
        payload["meeting_id"] = mid
    try:
        result = api_post("/api/v1/chat", payload, timeout=180.0)
        answer = result.get("answer") or "(빈 답변)"
        sources = result.get("sources") or []
        if sources:
            src = ", ".join(f"#{s.get('meeting_id')} {s.get('title')}" for s in sources[:3])
            answer = f"{answer}\n\n근거: {src}"
    except Exception as exc:  # noqa: BLE001
        answer = f"챗봇 호출 실패: {exc}"
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def list_all():
    meetings = api_get("/api/v1/meetings")
    if not meetings:
        return "등록된 회의가 없습니다."
    return "\n\n---\n\n".join(format_meeting(m) for m in meetings)


def mcp_demo_info():
    """6번 탭에 보여줄 MCP·슬랙 설정 설명. 서버를 띄우는 함수가 아닙니다."""
    return f"""
## MCP 연동

MCP는 Cursor 같은 채팅 앱이 이 프로젝트의 회의 기능을 **도구**로 부르게 하는 약속입니다.
모델이 SQL을 짜는 대신 `list_meetings`, `analyze_meeting` 같은 함수를 고르고,
이 프로세스(`mcp_server/server.py`)가 Backend API를 대신 호출합니다.

서버는 포트를 열지 않습니다. Cursor가 파이썬을 자식 프로세스로 켜고,
표준입출력(stdio)으로 JSON 메시지를 주고받습니다. Backend(`{config.BACKEND_URL}`)가 먼저 떠 있어야 합니다.

### Tools
`list_meetings`, `get_meeting`, `get_transcript`, `search_meetings`,
`analyze_meeting`, `create_meeting`, `ask_meeting`

### Cursor `mcp.json` 예시
```json
{{
  "mcpServers": {{
    "ryuzy": {{
      "command": "python",
      "args": ["{str(ROOT / 'mcp_server' / 'server.py').replace(chr(92), '/')}"],
      "cwd": "{str(ROOT).replace(chr(92), '/')}",
      "env": {{"BACKEND_URL": "{config.BACKEND_URL}"}}
    }}
  }}
}}
```
`command`는 가능하면 가상환경의 `python.exe` 전체 경로로 바꿉니다.
설정 후 Cursor에서 MCP를 새로고침하고, Backend를 켠 뒤 "회의 목록 보여 줘"라고 말하면 됩니다.

## 슬랙 설정

슬랙은 MCP 도구가 아닙니다. 4번 탭이 회의 결과를 **채널 메시지**로 올립니다.
수단은 Incoming Webhook입니다. 그 주소로 `{{"text": "..."}}` 를 POST하면 고른 채널에 글이 생깁니다.

1. https://api.slack.com/apps 에서 **Create New App → From scratch** (앱 이름 예: 류지 회의).
2. **Incoming Webhooks** → **Activate Incoming Webhooks** 를 On.
3. **Add New Webhook to Workspace** → 결과를 받을 채널을 고르고 허용.
4. 생긴 `https://hooks.slack.com/services/...` 주소를 복사합니다. 비밀번호처럼 다룹니다. 채팅이나 깃에 붙이지 마세요.
5. 프로젝트 루트 `.env` 에 `SLACK_WEBHOOK_URL=복사한_주소` 를 넣습니다.
6. Backend를 다시 시작합니다. 주소는 서버가 켜질 때 읽습니다.
7. 4번 탭에서 회의를 고르고 **선택 회의를 슬랙으로 보내기**, 또는 "이 회의 슬랙으로 보내줘".

채널에 요약·결정·액션이 보이면 성공입니다.
`SLACK_WEBHOOK_URL이 없습니다` 가 나오면 5~6단계를 다시 확인하세요.

Backend: `{config.BACKEND_URL}` · ASR: `{config.ASR_URL}`
"""


def check_health():
    lines = []
    for name, url in [("Backend", config.BACKEND_URL), ("ASR", config.ASR_URL)]:
        try:
            with _client() as client:
                r = client.get(f"{url}/health")
                payload = r.json()
                details = payload.get("details") or {}
                lines.append(f"- **{name}**: OK")
                if details:
                    lines.append(f"  - details: `{details}`")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- **{name}**: FAIL ({exc})")
    lines.append(
        "\n> 오디오 ASR은 **Qwen3 + pyannote** (GPU/HF_TOKEN 필요). "
        "텍스트 힌트·수동 회의록은 모델 없이 가능합니다."
    )
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="류지 프로젝트") as demo:
        gr.Markdown(
            """
# Meeting Recap
목소리로 회의하고, 화자를 나눈 뒤 회의록을 만듭니다.
**Gradio UI** · **Qwen3 + pyannote ASR** · **LangGraph 분석** · **회의 챗봇** · **MCP**
"""
        )

        with gr.Row():
            health_btn = gr.Button("서비스 상태 확인", variant="secondary")
            health_out = gr.Markdown()
        health_btn.click(check_health, outputs=health_out)

        meeting_dd = gr.Dropdown(label="회의 선택", choices=[], interactive=True)
        with gr.Row():
            refresh_btn = gr.Button("회의 목록 새로고침")
            delete_btn = gr.Button("선택 회의 삭제", variant="stop")
        refresh_btn.click(refresh_meeting_choices, outputs=meeting_dd)
        with gr.Row(visible=False) as delete_confirm_row:
            delete_yes_btn = gr.Button("정말 삭제", variant="stop")
            delete_no_btn = gr.Button("취소")
        delete_out = gr.Markdown()

        with gr.Tab("1. 회의 생성"):
            with gr.Column(elem_classes=["guide-step"]):
                gr.Markdown(
                    """
### 1단계. 회의를 만듭니다
제목, 참가자, 안건을 적고 **회의 생성**을 누릅니다.
"""
                )
                title = gr.Textbox(label="회의 제목", placeholder="예: 스프린트 계획 회의")
                participants = gr.Textbox(
                    label="참가자 (쉼표 구분)", placeholder="예: 김매니저, 이개발, 박기획"
                )
                agenda = gr.Textbox(label="안건", placeholder="예: 스프린트 목표 공유 및 일정 확정")
                create_btn = gr.Button("회의 생성", variant="primary")
                create_out = gr.Markdown()

            with gr.Column(elem_classes=["guide-step"]):
                gr.Markdown(
                    """
### 2단계. 목소리를 한 명씩 등록합니다
참가자마다 아래 순서를 반복합니다.

1. **등록할 참가자**에서 이름을 고릅니다. 이름을 바꾸면 녹음 칸이 비워지고 **Record**가 다시 나타납니다.
2. **읽을 문장**을 확인합니다.
3. **Record**를 누르고 문장을 끝까지 읽습니다. 다 읽으면 **Stop**을 누릅니다.
4. **들어보기**에 파형이 생기면 그 재생 버튼으로 확인합니다.
5. **이 목소리 저장**을 누릅니다. 다음 사람을 고르면 **Record**가 다시 나타납니다.
"""
                )
                speaker_dd = gr.Dropdown(label="1. 등록할 참가자", choices=[], interactive=True)
                gr.Textbox(
                    label="2. 읽을 문장",
                    value=ENROLL_PROMPT,
                    lines=3,
                    interactive=False,
                )
                enroll_audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    format="wav",
                    label="3. Record로 녹음하고, 다 읽으면 Stop",
                )
                enroll_preview = gr.Audio(
                    label="4. 들어보기",
                    interactive=False,
                    autoplay=False,
                    format="wav",
                )
                preview_note = gr.Markdown("Stop을 누르면 여기에 재생 버튼이 켜집니다.")
                with gr.Row():
                    enroll_btn = gr.Button("5. 이 목소리 저장", variant="primary", size="lg")
                    rerecord_btn = gr.Button("녹음 다시 하기", variant="secondary")
                voice_status = gr.Markdown(
                    "1단계에서 회의를 만들면, 누구부터 녹음하면 되는지 여기에 나옵니다."
                )

            with gr.Column(elem_classes=["guide-step"]):
                gr.Markdown(
                    """
### 3단계. 실시간 회의를 하고 회의록을 만듭니다
위 체크가 모두 `등록됨`이 된 뒤에 진행합니다.

1. **회의 마이크**의 **Record**를 누릅니다. 이 버튼이 회의 시작입니다.
2. 이야기합니다. 말이 잠깐 끊기면 화자와 문장이 아래에 쌓입니다.
3. 회의가 끝나면 **Stop**을 누릅니다.
4. **회의 종료 후 회의록 생성**을 누릅니다.

한 사람씩 말하면 화자 구분이 더 정확합니다.
"""
                )
                live_mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="1. 회의 마이크 — Record로 시작, Stop으로 종료",
                    buttons=[],
                )
                live_status = gr.Markdown("대기. 모두 등록된 뒤 Record를 누르세요.")
                live_out = gr.Markdown("_실시간 회의 내용_")
                finish_btn = gr.Button("4. 회의 종료 후 회의록 생성", variant="primary", size="lg")
                finish_out = gr.Markdown()

            create_btn.click(
                create_meeting,
                inputs=[title, participants, agenda],
                outputs=[create_out, meeting_dd, speaker_dd, voice_status, live_out],
            )
            enroll_btn.click(
                save_voice,
                inputs=[meeting_dd, speaker_dd, enroll_audio],
                outputs=[voice_status, enroll_audio, enroll_preview],
            )
            enroll_audio.stop_recording(
                show_preview,
                inputs=enroll_audio,
                outputs=[enroll_preview, preview_note],
            )
            enroll_audio.change(
                show_preview,
                inputs=enroll_audio,
                outputs=[enroll_preview, preview_note],
            )
            speaker_dd.change(
                reset_enrollment_recorder,
                outputs=[enroll_audio, enroll_preview, preview_note],
            )
            rerecord_btn.click(
                reset_enrollment_recorder,
                outputs=[enroll_audio, enroll_preview, preview_note],
            )
            meeting_dd.change(
                sync_voice_panel,
                inputs=meeting_dd,
                outputs=[speaker_dd, voice_status, live_out],
            )
            live_mic.start_recording(on_record_start, inputs=meeting_dd, outputs=live_status)
            live_mic.stream(
                on_live_audio,
                inputs=[live_mic, meeting_dd],
                outputs=[live_status, live_out],
                stream_every=0.5,
                concurrency_limit=1,
            )
            live_mic.stop_recording(
                on_record_stop,
                inputs=meeting_dd,
                outputs=[live_status, live_out],
            )
            finish_btn.click(
                finish_meeting,
                inputs=meeting_dd,
                outputs=[finish_out, live_out],
            )

        with gr.Tab("2. 회의록 / Qwen3+pyannote ASR"):
            gr.Markdown(
                "수동 텍스트 저장, 또는 오디오 업로드 → **pyannote 화자분리 + Qwen3 ASR**."
            )
            raw = gr.Textbox(
                label="수동 회의록 (화자: 내용)", lines=10, value=SAMPLE_TRANSCRIPT
            )
            save_btn = gr.Button("회의록 저장")
            audio = gr.Audio(
                sources=["upload", "microphone"], type="filepath", label="오디오"
            )
            hint = gr.Textbox(
                label="텍스트 힌트(모델 없이 파싱, 선택)",
                lines=3,
                placeholder="모델/GPU 없을 때 임시로 사용",
            )
            asr_btn = gr.Button("ASR 전사 후 저장", variant="primary")
            transcript_out = gr.Markdown()
            save_btn.click(
                save_manual_transcript, inputs=[meeting_dd, raw], outputs=transcript_out
            )
            asr_btn.click(
                run_asr, inputs=[meeting_dd, audio, hint], outputs=transcript_out
            )

        with gr.Tab("3. LangGraph 분석"):
            gr.Markdown(
                "`summarize → decisions → actions → quality` 파이프라인을 "
                f"**OpenAI `{config.OPENAI_MODEL}`** 로 실행합니다."
            )
            analyze_btn = gr.Button("LangGraph 분석 실행", variant="primary")
            analyze_out = gr.Markdown()
            analyze_btn.click(analyze, inputs=[meeting_dd], outputs=analyze_out)

        with gr.Tab("4. 회의 챗봇"):
            gr.Markdown(
                "저장된 회의록·요약·결정·액션을 근거로 "
                f"**OpenAI `{config.OPENAI_MODEL}`** 이 답합니다. "
                "위에서 회의를 고른 뒤 **슬랙으로 보내기**를 누르거나, "
                "「이 회의 슬랙으로 보내줘」라고 입력하면 그 회의 결과를 슬랙에 올립니다."
            )
            scope_selected = gr.Checkbox(
                label="위에서 선택한 회의만 검색",
                value=False,
            )
            chatbot = gr.Chatbot(label="대화", height=360)
            slack_btn = gr.Button("선택 회의를 슬랙으로 보내기", variant="secondary")
            chat_box = gr.Textbox(label="질문", placeholder="예: API 연동은 언제까지야? / 이 회의 슬랙으로 보내줘")
            chat_btn = gr.Button("보내기", variant="primary")
            slack_btn.click(
                send_slack_in_chat,
                inputs=[meeting_dd, chatbot],
                outputs=chatbot,
            )
            chat_btn.click(
                chat_reply,
                inputs=[chat_box, chatbot, meeting_dd, scope_selected],
                outputs=[chatbot, chat_box],
            )
            chat_box.submit(
                chat_reply,
                inputs=[chat_box, chatbot, meeting_dd, scope_selected],
                outputs=[chatbot, chat_box],
            )

        with gr.Tab("5. 전체 목록"):
            list_btn = gr.Button("전체 회의 보기")
            list_out = gr.Markdown()
            list_btn.click(list_all, outputs=list_out)
            load_btn = gr.Button("선택 회의 상세")
            load_out = gr.Markdown()
            load_btn.click(load_meeting, inputs=meeting_dd, outputs=load_out)

        with gr.Tab("6. MCP"):
            gr.Markdown(mcp_demo_info())

        demo.load(
            load_home,
            outputs=[meeting_dd, speaker_dd, voice_status, live_out],
        )

        delete_btn.click(
            ask_delete_meeting,
            inputs=meeting_dd,
            outputs=[delete_confirm_row, delete_out],
        )
        delete_no_btn.click(cancel_delete_meeting, outputs=[delete_confirm_row, delete_out])
        delete_yes_btn.click(
            confirm_delete_meeting,
            inputs=meeting_dd,
            outputs=[delete_confirm_row, delete_out, meeting_dd, speaker_dd, voice_status, live_out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
        css="""
        .guide-step {
            border: 1px solid var(--border-color-primary);
            border-radius: 12px;
            padding: 4px 16px 12px 16px;
            margin: 8px 0 16px 0;
        }
        """,
    )


if __name__ == "__main__":
    main()
