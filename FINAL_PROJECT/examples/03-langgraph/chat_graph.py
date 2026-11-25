"""LangGraph 미니 챗봇.

회의 메모에서 관련 줄을 찾은 뒤 답합니다.
본편 backend/app/graph/chat.py 의 retrieve -> answer 와 같은 모양입니다.
LLM 키 없이 터미널에서 바로 질문할 수 있습니다.

실행:
  python chat_graph.py          # 직접 질문 (종료: quit)
  python chat_graph.py --demo   # 샘플 질문 2개만 실행
"""

from __future__ import annotations

import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

NOTES = [
    "김매니저: 오늘 스프린트 목표를 공유합니다.",
    "이개발: 백엔드 API 연동은 수요일까지 완료하기로 했습니다.",
    "박기획: UI 초안은 금요일까지 제출하는 것으로 결정합니다.",
    "김매니저: 액션: 이개발이 테스트 케이스를 작성하기로 합니다.",
]


class ChatState(TypedDict):
    question: str
    hits: list[str]
    answer: str


def _tokens(question: str) -> list[str]:
    raw = [part for part in question.replace("?", " ").split() if len(part) >= 2]
    tokens = []
    for part in raw:
        for ending in ("은", "는", "이", "가", "을", "를", "에", "요"):
            if part.endswith(ending) and len(part) > len(ending) + 1:
                part = part[: -len(ending)]
                break
        tokens.append(part)
    return tokens


def retrieve(state: ChatState) -> dict:
    """질문 단어가 들어 있는 회의 메모를 고릅니다."""
    tokens = _tokens(state["question"])
    hits = [line for line in NOTES if any(token in line for token in tokens)]
    print(f"[node:retrieve] 찾은 줄 {len(hits)}개")
    return {"hits": hits}


def answer(state: ChatState) -> dict:
    """찾은 줄로 답변 문장을 만듭니다."""
    if not state["hits"]:
        text = "관련 발화를 찾지 못했습니다. 메모 전체:\n" + "\n".join(
            f"- {line}" for line in NOTES
        )
    else:
        text = "관련 발화:\n" + "\n".join(f"- {line}" for line in state["hits"])
    print(f"[node:answer] {text.splitlines()[0]}")
    return {"answer": text}


def build_app():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def ask(app, question: str) -> str:
    result = app.invoke({"question": question, "hits": [], "answer": ""})
    print(result["answer"])
    print()
    return result["answer"]


def main() -> None:
    app = build_app()
    print("LangGraph 챗봇 (retrieve -> answer)")
    print("메모: API는 수요일, UI는 금요일, 테스트 케이스는 이개발")
    print()

    if "--demo" in sys.argv:
        ask(app, "API 연동은 언제까지야?")
        ask(app, "UI는 언제 제출해?")
        return

    print("질문을 입력하세요. 종료는 quit")
    while True:
        try:
            question = input("나> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        ask(app, question)


if __name__ == "__main__":
    main()
