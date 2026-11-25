"""회의 챗봇 LangGraph: retrieve → answer.

이론 — 검색 다음 생성
---------------------
모델에게 회의 원문 전부를 매번 넣으면 글자 한도와 비용이 커집니다.
그래서 두 칸으로 나눕니다.

1. retrieve
   질문에서 두 글자 이상인 단어를 뽑고, 저장된 회의 글(제목·안건·요약·결정·할 일·발화)에
   그 단어가 들어 있는 회의만 고릅니다.
   벡터 DB나 임베딩 검색은 쓰지 않습니다. 부분 문자열 겹침입니다.
   아무것도 안 겹치면 최근 쪽 회의 몇 개를 그래도 넣어, 빈 자료로 답하지 않게 합니다.
   meeting_id가 있으면 그 회의만 봅니다.

2. answer
   고른 글을 "회의 자료"로 붙이고, OpenAI에게
   "이 자료 안에 있는 내용으로만 한국어로 답하고, 없으면 모른다고 말하라"고 시킵니다.
   이렇게 근거 글을 먼저 모으고 그 안에서만 답하게 하는 흐름을
   RAG(Retrieval-Augmented Generation, 검색으로 보강한 생성)의 작은 형태라고 부릅니다.
   여기서 검색 엔진은 SQLite에 들어 있는 회의 텍스트입니다.

그래프 모양은 분석 그래프와 같고, 상태 타입만 ChatState로 다릅니다.
  START → retrieve → answer → END
대화 원문은 그래프 밖 run_chat()이 chat_messages 테이블에 세션별로 남깁니다.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from shared import config, db


class ChatState(TypedDict, total=False):
    question: str
    meeting_id: int | None
    context: str
    sources: list[dict[str, Any]]
    answer: str
    engine: str


def _tokens(question: str) -> list[str]:
    parts = re.split(r"[\s,?!.]+", question.strip())
    stop = {"은", "는", "이", "가", "을", "를", "에", "의", "뭐", "무엇", "알려줘", "해줘"}
    return [p for p in parts if len(p) >= 2 and p not in stop]


def _meeting_text(meeting: dict[str, Any]) -> str:
    lines = [
        f"회의 #{meeting['id']} {meeting.get('title', '')}",
        f"안건: {meeting.get('agenda') or ''}",
        f"요약: {meeting.get('summary') or ''}",
    ]
    for decision in meeting.get("decisions") or []:
        lines.append(f"결정: {decision}")
    for item in meeting.get("action_items") or []:
        lines.append(f"액션: {item.get('assignee', '')} - {item.get('task', '')}")
    for utterance in meeting.get("transcript") or []:
        lines.append(f"{utterance.get('speaker', '?')}: {utterance.get('text', '')}")
    return "\n".join(lines)


def retrieve_node(state: ChatState) -> dict[str, Any]:
    """질문과 겹치는 회의 글을 context로 모읍니다. 답 문장은 여기서 만들지 않습니다."""
    meeting_id = state.get("meeting_id")
    if meeting_id:
        meeting = db.get_meeting(int(meeting_id))
        meetings = [meeting] if meeting else []
    else:
        meetings = db.list_meetings()

    tokens = _tokens(state.get("question") or "")
    sources: list[dict[str, Any]] = []
    chunks: list[str] = []

    for meeting in meetings:
        if not meeting:
            continue
        text = _meeting_text(meeting)
        matched = not tokens or any(token in text for token in tokens)
        if not matched:
            continue
        sources.append({"meeting_id": meeting["id"], "title": meeting["title"]})
        chunks.append(text[:2500])

    if not chunks and meetings:
        for meeting in meetings[:3]:
            sources.append({"meeting_id": meeting["id"], "title": meeting["title"]})
            chunks.append(_meeting_text(meeting)[:1500])

    context = "\n\n".join(chunks)[:8000]
    return {"context": context, "sources": sources[:8]}


def _llm_answer(question: str, context: str) -> str:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env에 키를 설정한 뒤 서버를 다시 시작하세요.")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    material = context.strip() or "(저장된 회의 자료 없음)"
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 회의 어시스턴트입니다. 주어진 회의 자료 안에서만 한국어로 답하세요. "
                    "자료에 없으면 모른다고 말하세요."
                ),
            },
            {
                "role": "user",
                "content": f"질문: {question}\n\n회의 자료:\n{material}",
            },
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("OpenAI 응답이 비어 있습니다.")
    return answer


def answer_node(state: ChatState) -> dict[str, Any]:
    """retrieve가 넣어 둔 context 안에서만 답합니다. 키가 없으면 여기서 예외가 납니다."""
    question = state.get("question") or ""
    context = state.get("context") or ""
    return {
        "answer": _llm_answer(question, context),
        "engine": "langgraph-chat-llm",
    }


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


_CHAT_APP = None


def get_chat_graph():
    global _CHAT_APP
    if _CHAT_APP is None:
        _CHAT_APP = build_chat_graph()
    return _CHAT_APP


def run_chat(
    *,
    question: str,
    meeting_id: int | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """Answer from stored meetings. The reply always comes from OpenAI."""
    db.init_db()
    db.add_chat_message(session_id, "user", question, meeting_id=meeting_id)
    final = get_chat_graph().invoke(
        {
            "question": question,
            "meeting_id": meeting_id,
            "context": "",
            "sources": [],
            "answer": "",
            "engine": "",
        }
    )
    answer = final.get("answer") or ""
    sources = final.get("sources") or []
    db.add_chat_message(
        session_id,
        "assistant",
        answer,
        meeting_id=meeting_id,
        sources=sources,
    )
    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "engine": final.get("engine") or "langgraph-chat-llm",
        "history": db.list_chat_messages(session_id),
    }
