"""류지 프로젝트 MCP Server.

Cursor / Claude Desktop 등 MCP 클라이언트가 회의 데이터를 도구로 조회·분석할 수 있게 합니다.

────────────────────────────────────────────────────────────────
MCP가 뭔지 (이론)
────────────────────────────────────────────────────────────────
MCP(Model Context Protocol)는 "채팅 안의 LLM"과 "내 컴퓨터의 프로그램"이
서로를 부르는 공통 약속입니다.

채팅창의 모델은 기본적으로 글을 읽고 글을 쓸 뿐입니다.
회의 목록을 DB에서 꺼내거나, 분석을 다시 돌리려면 바깥 함수가 필요합니다.
그 함수를 이 프로토콜에서는 Tool(도구)이라고 부릅니다.

동작은 세 단계입니다.

1. 발견
   클라이언트(Cursor)가 서버를 켜고 "어떤 도구가 있니?"라고 묻습니다.
   FastMCP는 @mcp.tool()이 붙은 함수의 이름, 인자 타입, docstring을
   JSON Schema로 만들어 모델에게 보여 줍니다.
   그래서 docstring은 사람용 주석이면서, 동시에 모델이 도구를 고르는 설명서입니다.

2. 호출
   사용자가 "지난 회의 요약해 줘"라고 하면, 모델이 직접 SQL을 짜는 대신
   analyze_meeting(meeting_id=3) 같은 도구 호출을 선택합니다.
   인자는 JSON으로 넘어오고, 반환 문자열을 모델이 다시 읽어 답으로 바꿉니다.

3. 운반 (stdio)
   이 서버는 HTTP 포트를 열지 않습니다.
   Cursor가 `python mcp_server/server.py`를 자식 프로세스로 띄우고,
   표준입력(stdin)과 표준출력(stdout)으로 JSON-RPC 메시지를 주고받습니다.
   이 방식을 transport="stdio"라고 합니다.
   사람이 터미널에서 서버를 직접 실행해 대화하는 구조가 아닙니다.
   stdout에 디버그 print를 찍으면 프로토콜 메시지가 깨질 수 있어,
   로그가 필요하면 stderr로 보냅니다. FastMCP가 그 구분을 처리합니다.

이 파일의 도구는 회의 로직을 다시 구현하지 않습니다.
이미 떠 있는 Backend(FastAPI + LangGraph, 기본 http://127.0.0.1:8000)를
httpx로 호출하는 얇은 포장입니다. Backend가 꺼져 있으면 도구가 실패합니다.

Resource는 Tool과 다릅니다.
Tool은 모델이 "실행"하는 함수이고,
Resource는 "읽어 보는" 자료입니다. 아래 ryuzy://health 가 그 예입니다.

────────────────────────────────────────────────────────────────
Cursor에 이 서버를 붙이는 방법
────────────────────────────────────────────────────────────────
1. 프로젝트 루트에 가상환경이 있고, `pip install "mcp>=1.2.0,<2"` 가 끝난 상태여야 합니다.
   mcp 2.x는 FastMCP 이름이 달라질 수 있어 1.x로 고정합니다.
2. Cursor 설정 → MCP (또는 프로젝트의 mcp.json)에 mcp_config.example.json 내용을 넣습니다.
3. command는 가상환경의 python.exe 전체 경로를 권장합니다.
   그냥 "python"이면 Cursor가 다른 파이썬을 집어 모듈을 못 찾을 수 있습니다.
4. args는 mcp_server/server.py, cwd는 이 프로젝트 루트입니다.
5. env에 BACKEND_URL=http://127.0.0.1:8000 을 넣습니다.
6. Backend를 먼저 실행합니다: python -m backend.app.main
7. Cursor에서 MCP를 새로고침한 뒤, 채팅에서 "회의 목록 보여 줘"처럼 말합니다.

────────────────────────────────────────────────────────────────
슬랙 설정 (MCP와 나란히 쓰는 출구)
────────────────────────────────────────────────────────────────
MCP는 Cursor가 회의를 "읽고 분석"하는 통로입니다.
슬랙은 그 결과를 사람 채널에 "메시지로 올리는" 통로입니다.
슬랙 전송 도구는 이 MCP 서버에 없습니다.
Gradio 4번 탭 → Backend POST /api/v1/meetings/{id}/slack
→ backend/app/slack_export.py 가 Incoming Webhook으로 보냅니다.

Incoming Webhook은 슬랙이 발급하는 비밀 주소입니다.
그 주소로 HTTP POST {"text": "메시지"} 를 보내면, 고른 채널에 글이 올라갑니다.
봇 토큰이나 채널 ID를 코드에 넣지 않아도 됩니다.

설정 순서:

1. 브라우저에서 https://api.slack.com/apps 를 열고 워크스페이스에 로그인합니다.
2. Create New App → From scratch 를 고릅니다.
   앱 이름은 예: "류지 회의", 워크스페이스는 메시지를 받을 곳입니다.
3. 왼쪽 메뉴에서 Incoming Webhooks 를 열고, Activate Incoming Webhooks 를 On 으로 둡니다.
4. Add New Webhook to Workspace 를 누르고, 회의 결과를 받을 채널을 고른 뒤 허용합니다.
5. Webhook URL이 https://hooks.slack.com/services/T.../B.../... 형태로 생깁니다.
   이 주소는 비밀번호와 같습니다. 채팅, 깃, 주석에 붙여 넣지 않습니다.
6. 프로젝트 루트 .env 에 한 줄 추가합니다.
       SLACK_WEBHOOK_URL=여기에_복사한_주소
   .env.example 의 빈 칸이 그 자리입니다. 실제 값은 .env 에만 둡니다.
7. Backend는 시작할 때 .env 를 읽습니다. 값을 바꾼 뒤에는
   python -m backend.app.main 을 다시 실행합니다.
8. Gradio(python frontend/app.py)의 4번 탭에서 회의를 고르고
   "선택 회의를 슬랙으로 보내기"를 누르거나,
   "이 회의 슬랙으로 보내줘"라고 입력합니다.
9. 채널에 *요약* *결정* *액션* 이 보이면 성공입니다.
   "SLACK_WEBHOOK_URL이 없습니다"가 나오면 6~7단계를 다시 확인합니다.
   슬랙이 403/404를 주면 웹훅이 취소됐거나 주소가 잘린 경우이니 4단계부터 다시 만듭니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import config  # noqa: E402

mcp = FastMCP(
    "ryuzy",
    instructions=(
        "류지 프로젝트 회의 어시스턴트 MCP 서버입니다. "
        "회의 목록 조회, 회의록 검색, 요약/액션아이템 분석, 회의 챗봇 질문을 도구로 제공합니다. "
        "슬랙 전송은 이 서버의 도구가 아닙니다. "
        "회의 결과를 채널에 올리려면 .env의 SLACK_WEBHOOK_URL과 Gradio의 슬랙 버튼을 사용합니다."
    ),
)


def _backend_get(path: str) -> Any:
    """Backend REST를 GET으로 호출합니다. MCP 도구는 이 함수로 DB에 직접 닿지 않습니다."""
    url = f"{config.BACKEND_URL.rstrip('/')}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def _backend_post(path: str, payload: dict[str, Any]) -> Any:
    url = f"{config.BACKEND_URL.rstrip('/')}{path}"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def list_meetings() -> str:
    """등록된 회의 목록을 조회합니다."""
    data = _backend_get("/api/v1/meetings")
    slim = [
        {
            "id": m["id"],
            "title": m["title"],
            "status": m["status"],
            "participants": m.get("participants", []),
            "created_at": m.get("created_at"),
        }
        for m in data
    ]
    return json.dumps(slim, ensure_ascii=False, indent=2)


@mcp.tool()
def get_meeting(meeting_id: int) -> str:
    """회의 ID로 상세 정보(회의록·요약·액션 포함)를 조회합니다."""
    data = _backend_get(f"/api/v1/meetings/{meeting_id}")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def get_transcript(meeting_id: int) -> str:
    """특정 회의의 발화(transcript)만 조회합니다."""
    data = _backend_get(f"/api/v1/meetings/{meeting_id}")
    return json.dumps(data.get("transcript", []), ensure_ascii=False, indent=2)


@mcp.tool()
def search_meetings(query: str) -> str:
    """제목/안건/요약/회의록에서 키워드로 회의를 검색합니다."""
    data = _backend_get(f"/api/v1/meetings/search/{query}")
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_meeting(meeting_id: int) -> str:
    """회의록을 OpenAI로 분석해 요약·결정사항·액션아이템을 생성하고 저장합니다."""
    data = _backend_post(
        "/api/v1/meetings/analyze",
        {"meeting_id": meeting_id},
    )
    result = {
        "id": data["id"],
        "title": data["title"],
        "summary": data.get("summary", ""),
        "decisions": data.get("decisions", []),
        "action_items": data.get("action_items", []),
        "status": data.get("status"),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def create_meeting(title: str, participants: str = "", agenda: str = "") -> str:
    """새 회의를 생성합니다. participants는 쉼표로 구분하세요."""
    people = [p.strip() for p in participants.split(",") if p.strip()]
    data = _backend_post(
        "/api/v1/meetings",
        {"title": title, "participants": people, "agenda": agenda},
    )
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def ask_meeting(question: str, meeting_id: int = 0) -> str:
    """저장된 회의록을 근거로 OpenAI가 질문에 답합니다. meeting_id가 0이면 전체 회의를 검색합니다."""
    payload: dict[str, Any] = {
        "question": question,
        "session_id": "mcp",
    }
    if meeting_id:
        payload["meeting_id"] = meeting_id
    data = _backend_post("/api/v1/chat", payload)
    slim = {
        "answer": data.get("answer", ""),
        "sources": data.get("sources", []),
        "engine": data.get("engine"),
    }
    return json.dumps(slim, ensure_ascii=False, indent=2)


@mcp.resource("ryuzy://health")
def backend_health() -> str:
    """Backend 헬스 상태를 리소스로 노출합니다."""
    try:
        data = _backend_get("/health")
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "detail": str(exc)}, ensure_ascii=False)


def main() -> None:
    # stdio: Cursor가 이 프로세스를 띄우고 stdin/stdout으로 JSON-RPC를 주고받습니다.
    # 슬랙 웹훅 주소는 여기서 읽지 않습니다. Backend의 SLACK_WEBHOOK_URL이 담당합니다.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
