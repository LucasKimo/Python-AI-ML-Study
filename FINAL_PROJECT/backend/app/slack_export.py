"""저장된 회의를 슬랙 채널로 보냅니다. 전송 수단은 Incoming Webhook입니다.

이론
----
Incoming Webhook은 "이 주소로 글자를 POST하면 그 채널에 메시지가 생긴다"는
한 방향 출구입니다. 슬랙의 전체 API(봇 토큰, 채널 목록, 스레드)보다 좁습니다.
회의 결과를 올리기만 하면 되므로 이 범위로 충분합니다.

요청 본문은 JSON 한 조각입니다: {"text": "올릴 글"}.
text 안의 *단어* 는 슬랙 mrkdwn에서 굵은 글씨가 됩니다.
글자 수 상한(아래 3500자)은 웹훅 메시지 한 통이 너무 길어 잘리는 것을 막기 위한 잘라내기입니다.

설정 (비밀번호처럼 다룰 것)
--------------------------
1. https://api.slack.com/apps → Create New App → From scratch.
2. Incoming Webhooks → Activate Incoming Webhooks = On.
3. Add New Webhook to Workspace → 받을 채널 선택 → 허용.
4. 발급된 https://hooks.slack.com/services/... 주소를 복사합니다.
   채팅이나 깃에 붙이지 않습니다. .env 한 곳에만 둡니다.
5. 프로젝트 루트 .env:
       SLACK_WEBHOOK_URL=복사한_주소
6. Backend를 다시 시작합니다. 환경변수는 프로세스 기동 때 읽습니다.
7. Gradio 4번 탭의 "선택 회의를 슬랙으로 보내기",
   또는 "이 회의 슬랙으로 보내줘"가 POST /api/v1/meetings/{id}/slack 을 호출하고
   그 요청이 이 파일의 post_meeting()으로 들어옵니다.

MCP 서버(mcp_server/server.py)에는 슬랙 도구가 없습니다.
Cursor용 조회/분석과, 사람 채널로 올리는 일이 나뉘어 있습니다.
같은 설정 설명은 MCP 서버 파일 맨 위 주석과 Gradio 6번 탭에도 있습니다.
"""

from __future__ import annotations

from typing import Any

import httpx

from shared import config, db


def format_meeting_message(meeting: dict[str, Any]) -> str:
    title = str(meeting.get("title") or "회의")
    lines = [f"*{title}*"]
    agenda = str(meeting.get("agenda") or "").strip()
    if agenda:
        lines.append(f"안건: {agenda}")

    summary = str(meeting.get("summary") or "").strip()
    decisions = [str(item) for item in (meeting.get("decisions") or []) if str(item).strip()]
    actions = [item for item in (meeting.get("action_items") or []) if isinstance(item, dict)]
    transcript = list(meeting.get("transcript") or [])

    if summary:
        lines.extend(["", "*요약*", summary])
    if decisions:
        lines.extend(["", "*결정*", *(f"• {item}" for item in decisions)])
    if actions:
        lines.append("")
        lines.append("*액션*")
        for item in actions:
            who = str(item.get("assignee") or "미정")
            task = str(item.get("task") or "").strip()
            if task:
                lines.append(f"• {who}: {task}")
    if not summary and not decisions and not actions:
        if not transcript:
            raise ValueError("보낼 회의 내용이 없습니다. 회의를 진행하거나 회의록을 만든 뒤 보내세요.")
        lines.extend(["", "*회의록*"])
        for utterance in transcript[:40]:
            speaker = str(utterance.get("speaker") or "?")
            text = str(utterance.get("text") or "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        if len(transcript) > 40:
            lines.append(f"… 외 {len(transcript) - 40}개")

    return "\n".join(lines)[:3500]


def post_meeting(meeting_id: int) -> dict[str, Any]:
    """회의 한 건을 웹훅으로 보냅니다. 주소가 https로 시작하지 않으면 설정이 비어 있는 것입니다."""
    webhook = config.SLACK_WEBHOOK_URL
    if not webhook.startswith("https://"):
        raise RuntimeError(
            "SLACK_WEBHOOK_URL이 없습니다. .env에 슬랙 Incoming Webhook 주소를 넣은 뒤 서버를 다시 시작하세요."
        )
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise ValueError("회의를 찾을 수 없습니다.")
    message = format_meeting_message(meeting)
    response = httpx.post(webhook, json={"text": message}, timeout=20.0)
    if response.status_code >= 400:
        raise RuntimeError(f"슬랙이 전송을 거부했습니다 ({response.status_code}).")
    return {"ok": True, "meeting_id": meeting_id, "title": meeting.get("title") or ""}
